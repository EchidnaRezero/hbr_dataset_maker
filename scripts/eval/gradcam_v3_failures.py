"""Failure & low-confidence REVIEW Grad-CAM diagnostic for v3_pure_hbr55_plus_multi_camie.npz.

Pure PyTorch (zero extra installs). Hooks Camie-v2 ViT-Base `blocks[-1].norm1`
([1, 1025, 768] -> drop CLS -> 32x32 spatial patch map) and saves clean 3-panel
horizontal comparisons to `outputs/gradcam/v3_failures/`:
  - If Pred != GT: [Original Crop | v3 Pred CAM (wrong top-1) | v3 GT CAM (true char)]
  - If Pred == GT (low-conf REVIEW) or GT is None (unseen 19A):
                   [Original Crop | v3 Top-1 CAM | v3 Top-2 Competitor CAM]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import numpy as np
import torch
from PIL import Image

from compare_gradcam_v1_v3 import (
    build_3panel,
    compute_vit_gradcam,
    overlay_cam,
    short_name,
)
from src.hbr_pipeline.classifier import load_camie_backbone_fp32
from src.hbr_pipeline.detector import crop_box, pad_square_rgb, preprocess_pil


def get_topk(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: list[str],
    device: torch.device,
    k: int = 3,
) -> list[tuple[str, float]]:
    with torch.inference_mode():
        feat = backbone(x.to(device=device, dtype=torch.float32))
        feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        coef_t = torch.from_numpy(coef).to(device=device, dtype=torch.float32)
        inter_t = torch.from_numpy(intercept).to(device=device, dtype=torch.float32)
        probs = torch.softmax(feat_norm @ coef_t.T + inter_t[None, :], dim=-1)[0]
        vals, idxs = torch.topk(probs, k=k)
        return [(classes[int(i)], float(v)) for v, i in zip(vals, idxs)]


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "gradcam" / "v3_failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    v3 = np.load(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz", allow_pickle=True)
    classes = list(v3["classes"])

    preds_neo: dict[str, dict[str, Any]] = {}
    for line in (PROJECT_ROOT / "outputs" / "eval_v3" / "neo" / "predictions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        preds_neo[r["file"]] = r

    preds_neo2: dict[str, dict[str, Any]] = {}
    for line in (PROJECT_ROOT / "outputs" / "eval_v3" / "neo2" / "predictions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        preds_neo2[r["file"]] = r

    cases = [
        {
            "out_name": "01_IMinaseHR_R3_head.jpg",
            "split": "neo",
            "file": "IMinaseHR_R3.png",
            "inst_idx": 0,
            "gt": "minase_ichigo",
            "desc": "IMinaseHR_R3 (GT: minase_ichigo | v3 ph_pred: carole_reaper 0.19)",
        },
        {
            "out_name": "02_BIYamawakiEvil_R3a_head.jpg",
            "split": "neo",
            "file": "BIYamawakiEvil_R3a.png",
            "inst_idx": 0,
            "gt": "yamawaki_bon_ivar",
            "desc": "BIYamawakiEvil_R3a (GT: yamawaki_bon_ivar | low-conf REVIEW 0.27)",
        },
        {
            "out_name": "03_Still_MC03_08_10_00_002_AsakuraKaren_bath_head.jpg",
            "split": "neo",
            "file": "Still_MC03_08_10_00_002.png",
            "inst_idx": 1,
            "gt": "asakura_karen_(heaven_burns_red)",
            "desc": "31A Bath - Asakura Karen (GT: asakura_karen | low-conf REVIEW 0.19, pred correct)",
        },
        {
            "out_name": "04_HNaruseDefault_R3_head.jpg",
            "split": "neo",
            "file": "HNaruseDefault_R3.png",
            "inst_idx": 0,
            "gt": None,
            "desc": "HNaruseDefault_R3 (Unseen 19A char | not in HBR-55 classes)",
        },
        {
            "out_name": "05_official_alternate_costume_YuinaCosplay_head.jpg",
            "split": "neo2",
            "file": "official_alternate_costume.png",
            "inst_idx": 0,
            "gt": "shirakawa_yuina",
            "desc": "ZZZ Evelyn Cosplay Yuina (GT: shirakawa_yuina | low-conf REVIEW 0.16)",
        },
        {
            "out_name": "06_Still_MC04_01_02_00_003_MonakaHospital_head.jpg",
            "split": "neo",
            "file": "Still_MC04_01_02_00_003.png",
            "inst_idx": 0,
            "gt": "tsukishiro_monaka",
            "desc": "Hospital Hair-Down Monaka (GT: tsukishiro_monaka | low-conf REVIEW 0.32)",
        },
        {
            "out_name": "07_Still_MC04_04_09_02_001_SugawaraChie_sideprofile_head.jpg",
            "split": "neo",
            "file": "Still_MC04_04_09_02_001.png",
            "inst_idx": 0,
            "gt": "sugawara_chie",
            "desc": "Side-Profile Sugawara Chie (GT: sugawara_chie | low-conf REVIEW 0.37)",
        },
        {
            "out_name": "08_Still_MC05B_05_09_00_003_IzumiYuki_noglasses_water_head.jpg",
            "split": "neo",
            "file": "Still_MC05B_05_09_00_003.png",
            "inst_idx": 2,
            "gt": "izumi_yuki",
            "desc": "Water Scene - Izumi Yuki w/o glasses (GT: izumi_yuki | v3 ph_pred: tachibana_kanade 0.32)",
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Camie-v2 ViT-Base backbone on {device} (fp32)...")
    backbone = load_camie_backbone_fp32(device)

    summary = []
    for c in cases:
        if c["split"] == "neo":
            img_path = PROJECT_ROOT / "data" / "test_sets" / "neo" / c["file"]
            inst = preds_neo[c["file"]]["instances"][c["inst_idx"]]
        else:
            img_path = PROJECT_ROOT / "data" / "test_sets" / "neo2" / c["file"]
            inst = preds_neo2[c["file"]]["instances"][c["inst_idx"]]

        bbox = inst["head_bbox"]
        with Image.open(img_path) as raw_im:
            pil_rgb = raw_im.convert("RGB")
        crop = crop_box(pil_rgb, bbox, pad_ratio=0.15)
        orig_512 = pad_square_rgb(crop).resize((512, 512), Image.BICUBIC)
        orig_512_np = np.asarray(orig_512, dtype=np.uint8)
        x = preprocess_pil(crop, 512).unsqueeze(0)

        topk = get_topk(backbone, x, v3["h_coef"], v3["h_intercept"], classes, device, k=3)
        top1_cls, top1_p = topk[0]
        top2_cls, top2_p = topk[1]
        gt = c["gt"]

        if gt is None:
            p1_cls, p2_cls = top1_cls, top2_cls
            p1_tag, p2_tag = "v3 Top-1 Pred CAM", "v3 Top-2 Pred CAM"
        elif top1_cls == gt:
            p1_cls, p2_cls = top1_cls, top2_cls
            p1_tag, p2_tag = "v3 Top-1 (GT) CAM", "v3 Top-2 Competitor CAM"
        else:
            p1_cls, p2_cls = top1_cls, gt
            p1_tag, p2_tag = "v3 Wrong Pred CAM", "v3 GT CAM"

        cam1 = compute_vit_gradcam(backbone, x, v3["h_coef"], v3["h_intercept"], classes, p1_cls, device)
        cam2 = compute_vit_gradcam(backbone, x, v3["h_coef"], v3["h_intercept"], classes, p2_cls, device)

        img1 = overlay_cam(orig_512_np, cam1["cam_32"])
        img2 = overlay_cam(orig_512_np, cam2["cam_32"])

        gt_disp = short_name(gt) if gt else "UNSEEN (19A)"
        orig_sub = f"Original Head Crop\nGT: {gt_disp} | ph: {short_name(inst['pred'])} ({inst['conf']:.2f})"
        sub1 = f"{p1_tag}\n{short_name(p1_cls)} (head={cam1['target_prob']:.2f})"
        sub2 = f"{p2_tag}\n{short_name(p2_cls)} (head={cam2['target_prob']:.2f})"

        panel = build_3panel(orig_512, img1, img2, c["desc"], orig_sub, sub1, sub2)
        out_path = out_dir / c["out_name"]
        panel.save(out_path, quality=95)

        rec = {
            "out_name": c["out_name"],
            "file": c["file"],
            "gt": gt,
            "ph_pred": inst["pred"],
            "ph_conf": round(inst["conf"], 4),
            "head_top3": [(cls, round(p, 4)) for cls, p in topk],
            "panel1_class": p1_cls,
            "panel1_prob": round(cam1["target_prob"], 4),
            "panel2_class": p2_cls,
            "panel2_prob": round(cam2["target_prob"], 4),
        }
        summary.append(rec)
        print(
            f"Saved {c['out_name']} | ph={short_name(inst['pred'])}({inst['conf']:.2f}) | "
            f"head top2={short_name(top1_cls)}({top1_p:.2f}), {short_name(top2_cls)}({top2_p:.2f}) | "
            f"P1={short_name(p1_cls)}({cam1['target_prob']:.2f}) vs P2={short_name(p2_cls)}({cam2['target_prob']:.2f})"
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nAll failure Grad-CAMs saved to: {out_dir}")


if __name__ == "__main__":
    main()
