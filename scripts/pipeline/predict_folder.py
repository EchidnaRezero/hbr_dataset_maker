"""Run saved HBR-55 classifier + 5-tier safety-valve routing on any folder of images
and save big-font visualizations + predictions.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.hbr_pipeline.detector import (
    IMAGE_EXTS,
    PersonHeadDetector,
    crop_box,
    draw_result,
    get_ort_providers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HBR-55 classifier + 5-tier safety-valve routing on a folder of images."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "predict"))
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz"),
    )
    parser.add_argument(
        "--meta",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_meta.jsonl"),
    )
    parser.add_argument("--accept-thr", type=float, default=0.45)
    parser.add_argument("--ext-alarm-thr", type=float, default=0.75)
    parser.add_argument("--min-area-pct", type=float, default=1.5)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    w = np.load(args.weights, allow_pickle=True)
    classes = [str(c) for c in w["classes"]]
    c2i = {c: i for i, c in enumerate(classes)}

    providers = get_ort_providers()
    detector = PersonHeadDetector(providers=providers, conf_thr=0.25)
    orig_camie = CamieV2Onnx(providers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_camie_backbone(device)

    ext_whitelist = load_ext_whitelist(Path(args.meta), c2i)

    imgs = sorted([p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    print(f"Found {len(imgs)} images in {in_dir}")

    tier_counts: collections.Counter[str] = collections.Counter()
    results = []
    t0 = time.time()

    for img_path in tqdm(imgs, desc=f"Predict [{in_dir.name}]"):
        try:
            with Image.open(img_path) as raw_im:
                pil_rgb = raw_im.convert("RGB")
                img_rgb = np.array(pil_rgb)
        except Exception as exc:
            print(f"Skip {img_path.name}: {exc}")
            continue
        img_area = float(pil_rgb.width * pil_rgb.height)

        instances = detector.detect_paired(img_rgb)

        p_crops, p_idx = [], []
        h_crops, h_idx = [], []
        for idx, inst in enumerate(instances):
            if inst["person_bbox"] is not None:
                p_crops.append(crop_box(pil_rgb, inst["person_bbox"], 0.06))
                p_idx.append(idx)
            if inst["head_bbox"] is not None:
                h_crops.append(crop_box(pil_rgb, inst["head_bbox"], 0.15))
                h_idx.append(idx)
        p_embs = embed_crops(backbone, p_crops, device)
        h_embs = embed_crops(backbone, h_crops, device)
        p_map = {idx: p_embs[k] for k, idx in enumerate(p_idx)}
        h_map = {idx: h_embs[k] for k, idx in enumerate(h_idx)}

        for idx, inst in enumerate(instances):
            probs, mode = classify_instance_probs(idx, p_map, h_map, w)
            inst["mode"] = mode

            best_i = int(np.argmax(probs))
            pred = classes[best_i]
            conf = float(probs[best_i])
            b = inst["person_bbox"] or inst["head_bbox"]
            area_pct = 100.0 * ((b[2] - b[0]) * (b[3] - b[1])) / max(1.0, img_area)

            routed = route_safety_valve(
                pil_rgb=pil_rgb,
                bbox=b,
                pred=pred,
                conf=conf,
                area_pct=area_pct,
                orig_camie=orig_camie,
                c2i=c2i,
                ext_whitelist=ext_whitelist,
                accept_thr=args.accept_thr,
                ext_alarm_thr=args.ext_alarm_thr,
                min_area_pct=args.min_area_pct,
            )

            inst["pred"] = routed["pred"]
            inst["conf"] = routed["conf"]
            inst["tier"] = routed["tier"]
            inst["area_pct"] = round(area_pct, 2)
            inst["display_label"] = routed["display_label"]
            tier_counts[routed["tier"]] += 1

        rel = img_path.relative_to(in_dir)
        out_img = vis_dir / rel.with_suffix(".jpg")
        draw_result(img_rgb, instances, f"{rel.as_posix()} | boxes={len(instances)}", out_img)
        results.append({"file": rel.as_posix(), "num_boxes": len(instances), "instances": instances})

    elapsed = time.time() - t0
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone {len(results)} images in {elapsed:.1f}s | Tiers: {dict(tier_counts)}")
    for r in results:
        s = ", ".join(x["display_label"] for x in r["instances"]) or "(no box)"
        print(f"  {r['file']:<40s} -> {s}")
    print(f"\nVisualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
