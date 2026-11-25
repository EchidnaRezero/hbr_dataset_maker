"""Real-world hard testset evaluation (178 unseen images: 31D/31X/31F, multi_general, collab, cosplay).
Uses GPU-accelerated deepghs ONNX detectors (person + head) + frozen Camie-v2 (86M) backbone
with saved HBR linear head weights (.npz).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.hbr_pipeline.classifier import (
    classify_instance_probs,
    embed_crops,
    load_camie_backbone,
)
from src.hbr_pipeline.detector import (
    PersonHeadDetector,
    collapse_costume_subtags,
    crop_box,
    draw_result,
    get_ort_providers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HBR classifier on hard testset.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "data" / "test_sets" / "hard_testset_manifest.jsonl"),
    )
    parser.add_argument(
        "--meta",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_meta.jsonl"),
    )
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs" / "eval_v3" / "hard_testset"),
    )
    parser.add_argument("--vis-per-cat", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load saved HBR linear head weights
    w = np.load(args.weights, allow_pickle=True)
    classes = [str(c) for c in w["classes"]]
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"Loaded classifier weights ({len(classes)} classes): {args.weights}")

    # 2. Load GPU detectors + Camie-v2 backbone
    providers = get_ort_providers()
    detector = PersonHeadDetector(providers=providers, conf_thr=0.35)
    print("Detector providers:", detector.person_det.sess.get_providers())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_camie_backbone(device)

    meta_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(args.meta).open(encoding="utf-8")
        if line.strip()
    }
    manifest = [
        json.loads(line)
        for line in Path(args.manifest).open(encoding="utf-8")
        if line.strip()
    ]

    hard_testset_dir = PROJECT_ROOT / "data" / "test_sets" / "hard_testset"

    cat_stats = collections.defaultdict(
        lambda: {
            "n": 0,
            "boxes_total": 0,
            "gt_known_total": 0,
            "free_hits": 0,
            "free_preds_total": 0,
            "free_conf50_hits": 0,
            "free_conf50_preds": 0,
            "solo_n": 0,
            "solo_top1_ok": 0,
            "multi_n": 0,
            "multi_recall_hits": 0,
            "multi_gt_total": 0,
            "confs_correct": [],
            "confs_wrong": [],
        }
    )
    vis_counts: collections.Counter[str] = collections.Counter()
    all_out = []

    for item in tqdm(manifest, desc="Eval hard_testset"):
        pid = item["id"]
        cat = item["category"]
        img_path = Path(item["local_path"])
        if not img_path.exists():
            alt = hard_testset_dir / cat / img_path.name
            if alt.exists():
                img_path = alt
        raw_chars = meta_by_id[pid]["characters"]
        gt_all = collapse_costume_subtags(raw_chars)
        gt_known = [c for c in gt_all if c in c2i]
        gt_unknown = [c for c in gt_all if c not in c2i]

        with Image.open(img_path) as raw_im:
            pil_rgb = raw_im.convert("RGB")
            img_rgb = np.array(pil_rgb)

        instances = detector.detect_paired(img_rgb)

        p_crops, p_idx = [], []
        h_crops, h_idx = [], []
        for idx, inst in enumerate(instances):
            if inst["person_bbox"] is not None:
                p_crops.append(crop_box(pil_rgb, inst["person_bbox"], pad_ratio=0.06))
                p_idx.append(idx)
            if inst["head_bbox"] is not None:
                h_crops.append(crop_box(pil_rgb, inst["head_bbox"], pad_ratio=0.15))
                h_idx.append(idx)

        p_embs = embed_crops(backbone, p_crops, device)
        h_embs = embed_crops(backbone, h_crops, device)
        p_map = {idx: p_embs[k] for k, idx in enumerate(p_idx)}
        h_map = {idx: h_embs[k] for k, idx in enumerate(h_idx)}

        prob_matrix = []
        for idx, inst in enumerate(instances):
            probs, mode = classify_instance_probs(idx, p_map, h_map, w)
            inst["mode"] = mode
            best_i = int(np.argmax(probs))
            inst["pred"] = classes[best_i]
            inst["conf"] = float(probs[best_i])
            prob_matrix.append(probs)

        # Mode B: Hungarian matching restricted to gt_known candidates
        if instances and gt_known:
            cand_ids = [c2i[c] for c in gt_known]
            P = np.stack(prob_matrix, axis=0)[:, cand_ids]
            row_ind, col_ind = linear_sum_assignment(-P)
            for r_i, c_i in zip(row_ind, col_ind):
                instances[r_i]["matched_pred"] = gt_known[c_i]
                instances[r_i]["matched_prob"] = float(P[r_i, c_i])

        # Accumulate metrics
        st = cat_stats[cat]
        st["n"] += 1
        st["boxes_total"] += len(instances)
        gt_known_set = set(gt_known)
        pred_set = {inst["pred"] for inst in instances}
        pred_conf50_set = {inst["pred"] for inst in instances if inst["conf"] >= 0.50}

        st["gt_known_total"] += len(gt_known_set)
        st["free_hits"] += len(gt_known_set & pred_set)
        st["free_preds_total"] += len(pred_set)
        st["free_conf50_hits"] += len(gt_known_set & pred_conf50_set)
        st["free_conf50_preds"] += len(pred_conf50_set)

        for inst in instances:
            if inst["pred"] in gt_known_set:
                st["confs_correct"].append(inst["conf"])
            else:
                st["confs_wrong"].append(inst["conf"])

        if len(gt_all) == 1 and len(gt_known) == 1:
            st["solo_n"] += 1
            if instances and instances[0]["pred"] == gt_known[0]:
                st["solo_top1_ok"] += 1
        elif len(gt_all) >= 2 and gt_known_set:
            st["multi_n"] += 1
            st["multi_gt_total"] += len(gt_known_set)
            st["multi_recall_hits"] += len(gt_known_set & pred_set)

        all_out.append(
            {
                "id": pid,
                "category": cat,
                "gt_all": gt_all,
                "gt_known": gt_known,
                "gt_unknown": gt_unknown,
                "instances": instances,
            }
        )

        if vis_counts[cat] < args.vis_per_cat:
            title = f"[{cat}] GT:{','.join(gt_all[:4])} | boxes={len(instances)}"
            draw_result(img_rgb, instances, title, vis_dir / cat / f"{vis_counts[cat]:02d}_{pid}.jpg")
            vis_counts[cat] += 1

    # Summarize
    summary = {}
    print(f"\n================ HARD TESTSET (178 UNSEEN IMAGES) — CAMIE-V2 + HBR-{len(classes)} HEAD ================")
    for cat in ["squad_31D", "squad_31X", "squad_31F", "multi_general", "collab", "cosplay"]:
        st = cat_stats[cat]
        recall = 100.0 * st["free_hits"] / max(1, st["gt_known_total"])
        prec = 100.0 * st["free_hits"] / max(1, st["free_preds_total"])
        prec50 = 100.0 * st["free_conf50_hits"] / max(1, st["free_conf50_preds"])
        rec50 = 100.0 * st["free_conf50_hits"] / max(1, st["gt_known_total"])
        solo_acc = 100.0 * st["solo_top1_ok"] / max(1, st["solo_n"]) if st["solo_n"] else None
        multi_rec = 100.0 * st["multi_recall_hits"] / max(1, st["multi_gt_total"]) if st["multi_n"] else None
        c_ok = float(np.mean(st["confs_correct"])) if st["confs_correct"] else 0.0
        c_ng = float(np.mean(st["confs_wrong"])) if st["confs_wrong"] else 0.0

        summary[cat] = {
            "n": st["n"],
            "avg_boxes": round(st["boxes_total"] / max(1, st["n"]), 2),
            "free_recall_pct": round(recall, 1),
            "free_precision_pct": round(prec, 1),
            "conf50_precision_pct": round(prec50, 1),
            "conf50_recall_pct": round(rec50, 1),
            "solo_top1_pct": round(solo_acc, 1) if solo_acc is not None else None,
            "multi_recall_pct": round(multi_rec, 1) if multi_rec is not None else None,
            "avg_conf_correct": round(c_ok, 3),
            "avg_conf_wrong": round(c_ng, 3),
        }
        solo_str = f"solo={solo_acc:5.1f}%({st['solo_top1_ok']}/{st['solo_n']})" if solo_acc is not None else "solo=  -  "
        multi_str = f"multi_rec={multi_rec:5.1f}%" if multi_rec is not None else "multi_rec=  -  "
        print(
            f"  [{cat:<13s}] N={st['n']:2d} | {solo_str:<19s} | {multi_str:<16s} | "
            f"All P/R={prec:5.1f}%/{recall:5.1f}% | conf>=0.5 P/R={prec50:5.1f}%/{rec50:5.1f}% | "
            f"conf(ok/ng)={c_ok:.2f}/{c_ng:.2f}"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved summary: {out_dir / 'summary.json'}")
    print(f"Saved visualizations ({sum(vis_counts.values())} images): {vis_dir}")


if __name__ == "__main__":
    main()
