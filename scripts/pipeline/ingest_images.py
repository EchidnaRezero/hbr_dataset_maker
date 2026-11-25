"""Directly ingest untagged official/custom images into `master_hbr.db` (SQLite).

Runs the full detection + pure HBR-55 (v3/v2) classifier + 5-bucket safety valve
(ACCEPT / EXT_COLLAB / REVIEW_COSPLAY_OR_EXT / REVIEW / CROWD_MASK) + SQLite pHash
deduplication, writing directly to `data/db/master_hbr.db` with no intermediate JSONL.

Resume-safe: UIDs already present in `images` are skipped unless `--overwrite` is set.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hbr_pipeline.db import find_phash_duplicate, init_db, print_stats, upsert_image_record

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest official/custom images directly into master_hbr.db with v3/v2 classifier + 5-bucket safety valve."
    )
    parser.add_argument("--input-dir", required=True, help="Directory of images to ingest (scanned recursively)")
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz"),
        help="Path to v3 or v2 pure HBR-55 .npz weights",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "db" / "master_hbr.db"),
        help="Path to SQLite master_hbr.db",
    )
    parser.add_argument(
        "--meta",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_meta.jsonl"),
        help="Optional Danbooru metadata JSONL for expanding external collab whitelist",
    )
    parser.add_argument("--source", choices=["official", "custom"], default="official")
    parser.add_argument(
        "--uid-prefix",
        default="",
        help="Optional UID prefix (default: '<source>:<input_dir_name>')",
    )
    parser.add_argument("--phash-thr", type=int, default=4, help="Max 64-bit pHash Hamming distance for dedup")
    parser.add_argument("--accept-thr", type=float, default=0.45)
    parser.add_argument("--ext-alarm-thr", type=float, default=0.75)
    parser.add_argument("--min-area-pct", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true", help="Re-process UIDs already in DB")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    db_path = Path(args.db)
    conn = init_db(db_path)

    existing_uids: set[str] = set()
    if not args.overwrite:
        existing_uids = {row[0] for row in conn.execute("SELECT uid FROM images")}

    uid_prefix = args.uid_prefix.strip() or f"{args.source}:{in_dir.name}"
    items: list[dict[str, Any]] = []
    for p in sorted(in_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(in_dir).as_posix()
            uid = f"{uid_prefix}/{rel}"
            if uid in existing_uids:
                continue
            items.append({"uid": uid, "source": args.source, "local_path": str(p)})

    print(
        f"DB={db_path} | Weights={weights_path.name} | Existing UIDs={len(existing_uids):,} | To process={len(items):,}"
    )
    if not items:
        print("Nothing new to ingest.")
        conn.close()
        return

    # Lazy-import heavy ML dependencies only when there is work to do
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    from src.hbr_pipeline.classifier import (
        CamieV2Onnx,
        classify_instance_probs,
        embed_crops,
        load_camie_backbone,
        load_ext_whitelist,
        route_safety_valve,
    )
    from src.hbr_pipeline.detector import PersonHeadDetector, crop_box, get_ort_providers
    from src.hbr_pipeline.phash import compute_phash_hex

    w = np.load(str(weights_path), allow_pickle=True)
    classes = [str(c) for c in w["classes"]]
    c2i = {c: i for i, c in enumerate(classes)}

    providers = get_ort_providers()
    detector = PersonHeadDetector(providers=providers, conf_thr=0.25)
    orig_camie = CamieV2Onnx(providers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_camie_backbone(device)

    ext_whitelist = load_ext_whitelist(Path(args.meta), c2i)

    stats: collections.Counter[str] = collections.Counter()
    t0 = time.time()

    try:
        for it in tqdm(items, desc=f"Ingest [{args.source}] -> SQLite"):
            img_path = Path(it["local_path"])
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

            phash_hex = compute_phash_hex(pil_rgb)
            dup_of = find_phash_duplicate(
                conn, phash_hex, max_hamming=args.phash_thr, exclude_uid=it["uid"]
            )
            if dup_of is not None:
                stats["DUPLICATE"] += 1
                rec = {
                    "uid": it["uid"],
                    "source": it["source"],
                    "has_metadata": False,
                    "local_path": str(img_path),
                    "width": pil_rgb.width,
                    "height": pil_rgb.height,
                    "phash": phash_hex,
                    "is_duplicate": True,
                    "duplicate_of": dup_of,
                    "status": "DUPLICATE",
                    "characters": [],
                    "crowd_mask_boxes": [],
                }
                upsert_image_record(conn, rec)
                continue

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

            for k, inst in enumerate(main_insts):
                probs, mode = classify_instance_probs(k, p_map, h_map, w)
                inst["mode"] = mode

                best_i = int(np.argmax(probs))
                pred = classes[best_i]
                conf = round(float(probs[best_i]), 4)
                inst["free_pred"] = pred
                inst["free_conf"] = conf

                b = inst["person_bbox"] or inst["head_bbox"]
                routed = route_safety_valve(
                    pil_rgb=pil_rgb,
                    bbox=b,
                    pred=pred,
                    conf=conf,
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
                "source": it["source"],
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
            upsert_image_record(conn, rec)
    finally:
        conn.close()

    elapsed = time.time() - t0
    print(f"\nIngested {sum(stats.values()):,} image(s) in {elapsed/60:.1f} min | Stats: {dict(stats)}")
    print_stats(db_path)


if __name__ == "__main__":
    main()
