"""Build & incrementally update the Master HBR Dataset JSONL DB + persistent pHash index.

Supports two distinct modes:
  - --mode danbooru : Uses Danbooru metadata (base_chars) for Hungarian 1:1 box-to-character matching
                      + saves embeddings for v3 classifier upgrade.
  - --mode official : For untagged official/custom images (no metadata). Uses pure HBR-55 + 5-bucket
                      safety valve (ACCEPT / EXT_COLLAB / REVIEW_COSPLAY_OR_EXT / REVIEW / CROWD_MASK).

Both modes share a persistent global pHash index (`data/db/phash_index.jsonl`) so future
additions are automatically deduplicated against all previously indexed images.
Resume-safe: already processed UIDs are skipped automatically.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.hbr_pipeline.classifier import (
    CamieV2Onnx,
    classify_instance_probs,
    embed_crops,
    load_camie_backbone,
    load_ext_whitelist,
    route_safety_valve,
)
from src.hbr_pipeline.detector import (
    IMAGE_EXTS,
    PersonHeadDetector,
    crop_box,
    get_ort_providers,
)
from src.hbr_pipeline.phash import PersistentPHashIndex, compute_phash_hex


def load_done_uids(out_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not out_jsonl.exists():
        return done
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update Master HBR JSONL DB + persistent pHash index.")
    parser.add_argument("--mode", choices=["danbooru", "official"], required=True)
    parser.add_argument(
        "--danbooru-manifest",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_manifest.jsonl"),
    )
    parser.add_argument(
        "--meta",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_meta.jsonl"),
    )
    parser.add_argument(
        "--danbooru-dir",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru"),
        help="Fallback directory for Danbooru images if manifest local_path is moved",
    )
    parser.add_argument("--input-dir", default="", help="Required for --mode official")
    parser.add_argument("--db-dir", default=str(PROJECT_ROOT / "data" / "db"))
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "db" / "master_hbr.db"),
        help="Path to SQLite master_hbr.db (informational / parent directory reference)",
    )
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz"),
    )
    parser.add_argument(
        "--v3-embs-out",
        default=str(PROJECT_ROOT / "data" / "embeddings" / "danbooru_multi_crops_for_v3.npz"),
    )
    parser.add_argument("--phash-thr", type=int, default=4)
    parser.add_argument("--accept-thr", type=float, default=0.45)
    parser.add_argument("--ext-alarm-thr", type=float, default=0.75)
    parser.add_argument("--min-area-pct", type=float, default=1.5)
    parser.add_argument("--save-v3-embs", action="store_true", help="Save high-conf crop embeddings for v3 upgrade")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    phash_idx = PersistentPHashIndex(db_dir / "phash_index.jsonl", max_hamming=args.phash_thr)

    out_jsonl = db_dir / f"{args.mode}_master.jsonl"
    done_uids = load_done_uids(out_jsonl)
    print(f"Mode={args.mode} | Existing pHash index={len(phash_idx.uids):,} | Already processed={len(done_uids):,}")

    w = np.load(args.weights, allow_pickle=True)
    classes = [str(c) for c in w["classes"]]
    c2i = {c: i for i, c in enumerate(classes)}

    providers = get_ort_providers()
    detector = PersonHeadDetector(providers=providers, conf_thr=0.25)
    orig_camie = CamieV2Onnx(providers) if args.mode == "official" else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_camie_backbone(device)

    ext_whitelist = load_ext_whitelist(Path(args.meta) if args.mode == "official" else None, c2i)

    # Build work queue
    items: list[dict[str, Any]] = []
    danbooru_dir = Path(args.danbooru_dir)
    if args.mode == "danbooru":
        with Path(args.danbooru_manifest).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                uid = f"danbooru:{r['id']}"
                if uid in done_uids:
                    continue
                items.append({**r, "uid": uid, "source": "danbooru"})
    else:
        in_dir = Path(args.input_dir)
        if not in_dir.exists():
            raise SystemExit(f"Input dir not found: {in_dir}")
        for p in sorted(in_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rel = p.relative_to(in_dir).as_posix()
                uid = f"official:{in_dir.name}/{rel}"
                if uid in done_uids:
                    continue
                items.append({"uid": uid, "source": "official", "local_path": str(p), "rel_path": rel})

    print(f"Remaining items to process: {len(items):,}")
    stats: collections.Counter[str] = collections.Counter()
    v3_p_embs, v3_h_embs, v3_labels, v3_uids = [], [], [], []
    t0 = time.time()

    out_f = out_jsonl.open("a", encoding="utf-8")
    try:
        for idx_item, it in enumerate(tqdm(items, desc=f"MasterDB [{args.mode}]")):
            img_path = Path(it["local_path"])
            if not img_path.exists() and args.mode == "danbooru":
                fallback = danbooru_dir / img_path.name
                if fallback.exists():
                    img_path = fallback
            if not img_path.exists():
                stats["missing_file"] += 1
                continue
            try:
                with Image.open(img_path) as raw_im:
                    pil_rgb = raw_im.convert("RGB")
                    img_rgb = np.array(pil_rgb)
            except Exception:
                stats["corrupt_img"] += 1
                continue

            phash_hex = it.get("phash") or compute_phash_hex(pil_rgb)
            dup_of = phash_idx.find_duplicate(phash_hex)
            if dup_of is not None:
                stats["phash_dup"] += 1
                rec = {
                    "uid": it["uid"],
                    "source": it["source"],
                    "has_metadata": args.mode == "danbooru",
                    "local_path": str(img_path),
                    "phash": phash_hex,
                    "is_duplicate": True,
                    "duplicate_of": dup_of,
                    "status": "DUPLICATE",
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            phash_idx.add(it["uid"], it["source"], str(img_path), phash_hex)
            img_area = float(pil_rgb.width * pil_rgb.height)

            instances = detector.detect_paired(img_rgb)

            main_insts: list[dict[str, Any]] = []
            crowd_boxes: list[dict[str, Any]] = []
            for inst in instances:
                b = inst["person_bbox"] or inst["head_bbox"]
                area_pct = 100.0 * ((b[2] - b[0]) * (b[3] - b[1])) / max(1.0, img_area)
                inst["area_pct"] = round(area_pct, 2)
                if area_pct < args.min_area_pct:
                    crowd_boxes.append({"bbox": b, "area_pct": round(area_pct, 2), "tier": "CROWD_MASK"})
                else:
                    main_insts.append(inst)

            p_crops, p_idx = [], []
            h_crops, h_idx = [], []
            for k, inst in enumerate(main_insts):
                if inst["person_bbox"] is not None:
                    p_crops.append(crop_box(pil_rgb, inst["person_bbox"], 0.06))
                    p_idx.append(k)
                if inst["head_bbox"] is not None:
                    h_crops.append(crop_box(pil_rgb, inst["head_bbox"], 0.15))
                    h_idx.append(k)
            p_embs = embed_crops(backbone, p_crops, device)
            h_embs = embed_crops(backbone, h_crops, device)
            p_map = {k: p_embs[j] for j, k in enumerate(p_idx)}
            h_map = {k: h_embs[j] for j, k in enumerate(h_idx)}

            prob_rows = []
            for k, inst in enumerate(main_insts):
                probs, mode = classify_instance_probs(k, p_map, h_map, w)
                inst["mode"] = mode
                best_i = int(np.argmax(probs))
                inst["free_pred"] = classes[best_i]
                inst["free_conf"] = round(float(probs[best_i]), 4)
                prob_rows.append(probs)

            if args.mode == "danbooru":
                base_chars = it.get("base_chars", [])
                known_cands = [c for c in base_chars if c in c2i]
                unknown_cands = [c for c in base_chars if c not in c2i]

                if main_insts and known_cands:
                    cand_ids = [c2i[c] for c in known_cands]
                    P = np.stack(prob_rows, axis=0)[:, cand_ids]
                    row_ind, col_ind = linear_sum_assignment(-P)
                    matched_rows = set(row_ind.tolist())
                    for ri, ci in zip(row_ind, col_ind):
                        inst = main_insts[ri]
                        ch_name = known_cands[ci]
                        mprob = float(P[ri, ci])
                        inst["character"] = ch_name
                        inst["match_conf"] = round(mprob, 4)
                        inst["tier"] = "ACCEPT" if (mprob >= 0.15 or len(known_cands) == 1) else "REVIEW"
                        if args.save_v3_embs and len(base_chars) >= 2 and not it.get("is_cosplay") and mprob >= 0.35:
                            if ri in p_map and ri in h_map:
                                v3_p_embs.append(p_map[ri])
                                v3_h_embs.append(h_map[ri])
                                v3_labels.append(ch_name)
                                v3_uids.append(it["uid"])
                    for k, inst in enumerate(main_insts):
                        if k not in matched_rows:
                            inst["character"] = unknown_cands[0] if len(unknown_cands) == 1 else "UNKNOWN"
                            inst["match_conf"] = 0.0
                            inst["tier"] = "REVIEW"
                else:
                    for inst in main_insts:
                        inst["character"] = unknown_cands[0] if len(unknown_cands) == 1 else inst["free_pred"]
                        inst["match_conf"] = 0.0
                        inst["tier"] = "REVIEW"

                img_status = "ACCEPT"
                if not main_insts:
                    img_status = "NO_BOX"
                elif it.get("is_cosplay"):
                    img_status = "COSPLAY"
                elif len(main_insts) != len(base_chars) or any(x["tier"] != "ACCEPT" for x in main_insts):
                    img_status = "REVIEW"

                stats[img_status] += 1
                rec = {
                    "uid": it["uid"],
                    "source": "danbooru",
                    "has_metadata": True,
                    "danbooru_id": it["id"],
                    "local_path": str(img_path),
                    "width": pil_rgb.width,
                    "height": pil_rgb.height,
                    "phash": phash_hex,
                    "is_duplicate": False,
                    "status": img_status,
                    "is_comic": it.get("is_comic", False),
                    "is_cosplay": it.get("is_cosplay", False),
                    "copyrights": it.get("copyrights", []),
                    "artists": it.get("artists", []),
                    "rating": it.get("rating"),
                    "score": it.get("score", 0),
                    "meta_base_chars": base_chars,
                    "characters": main_insts,
                    "crowd_mask_boxes": crowd_boxes,
                }
            else:
                assert orig_camie is not None
                for inst in main_insts:
                    b = inst["person_bbox"] or inst["head_bbox"]
                    routed = route_safety_valve(
                        pil_rgb=pil_rgb,
                        bbox=b,
                        pred=inst["free_pred"],
                        conf=inst["free_conf"],
                        area_pct=inst["area_pct"],
                        orig_camie=orig_camie,
                        c2i=c2i,
                        ext_whitelist=ext_whitelist,
                        accept_thr=args.accept_thr,
                        ext_alarm_thr=args.ext_alarm_thr,
                        min_area_pct=args.min_area_pct,
                    )
                    inst["character"] = routed["character"]
                    inst["tier"] = routed["tier"]
                    if "ext_conf" in routed:
                        inst["ext_conf"] = routed["ext_conf"]
                    if "ext_alarm" in routed:
                        inst["ext_alarm"] = routed["ext_alarm"]

                img_status = (
                    "NO_BOX"
                    if not main_insts
                    else (
                        "ACCEPT"
                        if all(x["tier"] in ("ACCEPT", "EXT_COLLAB") for x in main_insts)
                        else "REVIEW"
                    )
                )
                stats[img_status] += 1
                rec = {
                    "uid": it["uid"],
                    "source": "official",
                    "has_metadata": False,
                    "local_path": str(img_path),
                    "width": pil_rgb.width,
                    "height": pil_rgb.height,
                    "phash": phash_hex,
                    "is_duplicate": False,
                    "status": img_status,
                    "characters": main_insts,
                    "crowd_mask_boxes": crowd_boxes,
                }

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (idx_item + 1) % 100 == 0:
                out_f.flush()
    finally:
        out_f.close()

    if args.save_v3_embs and v3_p_embs:
        v3_path = Path(args.v3_embs_out)
        v3_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            v3_path,
            X_p=np.stack(v3_p_embs, axis=0).astype(np.float32),
            X_h=np.stack(v3_h_embs, axis=0).astype(np.float32),
            labels=np.array(v3_labels),
            uids=np.array(v3_uids),
        )
        print(f"Saved {len(v3_labels):,} high-conf multi-char crop embeddings for v3 -> {v3_path}")

    elapsed = time.time() - t0
    print(f"\nCompleted [{args.mode}] in {elapsed/60:.1f} min | Stats: {dict(stats)}")
    print(f"Master DB   : {out_jsonl}")
    print(f"pHash index : {db_dir / 'phash_index.jsonl'} (total {len(phash_idx.uids):,} unique images)")


if __name__ == "__main__":
    main()
