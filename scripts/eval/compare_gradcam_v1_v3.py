"""Pure-PyTorch ViT Grad-CAM visual comparison between v1 (HBR-54, 1,053 solo) and
v3 (HBR-55, pure solo + multi-char) on frozen Camie-v2 (86M) ViT-Base backbone.

Hooks `model.blocks[-1].norm1` (shape [1, 1025, 768]), drops CLS token -> 32x32 patch map,
and saves clean 3-panel horizontal comparison images [Original Crop | v1 Grad-CAM | v3 Grad-CAM]
to `outputs/gradcam/v1_vs_v3/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.hbr_pipeline.classifier import load_camie_backbone_fp32
from src.hbr_pipeline.detector import crop_box, pad_square_rgb, preprocess_pil


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def compute_vit_gradcam(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: list[str],
    target_class: str | None,
    device: torch.device,
) -> dict[str, Any]:
    """Run forward + backward through ViT-Base + linear head and compute 32x32 Grad-CAM."""
    captured: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, output: torch.Tensor):
        captured["act"] = output

        def grad_hook(grad: torch.Tensor):
            captured["grad"] = grad

        output.register_hook(grad_hook)

    handle = backbone.blocks[-1].norm1.register_forward_hook(fwd_hook)
    try:
        backbone.zero_grad(set_to_none=True)
        x_in = x.to(device=device, dtype=torch.float32).requires_grad_(True)
        coef_t = torch.from_numpy(coef).to(device=device, dtype=torch.float32)
        inter_t = torch.from_numpy(intercept).to(device=device, dtype=torch.float32)

        feat = backbone(x_in)  # [1, 768]
        feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        logits = feat_norm @ coef_t.T + inter_t[None, :]  # [1, C]
        probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs).item())
        pred_name = classes[pred_idx]
        pred_prob = float(probs[pred_idx].item())

        has_target = target_class is not None and target_class in classes
        if has_target:
            cam_idx = classes.index(target_class)
            target_prob = float(probs[cam_idx].item())
        else:
            cam_idx = pred_idx
            target_prob = None

        score = logits[0, cam_idx]
        score.backward()

        act = captured["act"][0, 1:, :]  # [1024, 768] (drop CLS)
        grad = captured["grad"][0, 1:, :]  # [1024, 768] (drop CLS)

        weights = grad.mean(dim=0)  # [768]
        cam = torch.relu((act * weights[None, :]).sum(dim=-1))  # [1024]
        cam_2d = cam.reshape(32, 32).detach().cpu().numpy()

        cmin, cmax = float(cam_2d.min()), float(cam_2d.max())
        if cmax - cmin > 1e-8:
            cam_norm = (cam_2d - cmin) / (cmax - cmin)
        else:
            cam_norm = np.zeros_like(cam_2d)

        return {
            "cam_32": cam_norm,
            "pred": pred_name,
            "pred_prob": pred_prob,
            "has_target": has_target,
            "target_prob": target_prob,
            "cam_class": classes[cam_idx],
        }
    finally:
        handle.remove()


def overlay_cam(base_512_rgb: np.ndarray, cam_32: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_512 = cv2.resize(cam_32.astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
    cam_512 = np.clip(cam_512, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((cam_512 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    blended = np.clip((1.0 - alpha) * base_512_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32), 0, 255)
    return Image.fromarray(blended.astype(np.uint8))


def short_name(c: str) -> str:
    return c.removesuffix("_(heaven_burns_red)")


def build_3panel(
    orig_512: Image.Image,
    v1_img: Image.Image,
    v3_img: Image.Image,
    title: str,
    orig_sub: str,
    v1_sub: str,
    v3_sub: str,
) -> Image.Image:
    pw, ph = 512, 512
    gap = 12
    top_h = 44
    sub_h = 52
    total_w = pw * 3 + gap * 4
    total_h = top_h + sub_h + ph + gap * 2

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    f_title = get_font(22)
    f_sub = get_font(16)

    draw.text((gap, 10), title, fill=(245, 245, 245), font=f_title)

    panels = [
        (orig_512, orig_sub, (220, 220, 220)),
        (v1_img, v1_sub, (255, 196, 120)),
        (v3_img, v3_sub, (130, 235, 170)),
    ]
    for i, (im, sub, color) in enumerate(panels):
        x0 = gap + i * (pw + gap)
        y_sub = top_h
        y_img = top_h + sub_h
        draw.multiline_text((x0 + 4, y_sub + 4), sub, fill=color, font=f_sub, spacing=4)
        canvas.paste(im, (x0, y_img))

    return canvas


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "gradcam" / "v1_vs_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = np.load(PROJECT_ROOT / "models" / "archive" / "v1_sample1053_camie_hbr54.npz", allow_pickle=True)
    v3 = np.load(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz", allow_pickle=True)
    v1_classes = list(v1["classes"])
    v3_classes = list(v3["classes"])

    preds_by_file: dict[str, dict[str, Any]] = {}
    for line in (PROJECT_ROOT / "outputs" / "eval_v3" / "neo" / "predictions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        preds_by_file[r["file"]] = r

    cases = [
        {
            "out_name": "01a_YShirakawaSocial_head.jpg",
            "file": "YShirakawaSocial_R3.png",
            "inst_idx": 1,
            "crop_kind": "head",
            "target": "shirakawa_yuina",
            "desc": "Yuina Social (Head crop) - Face/eyes vs hair ornament",
        },
        {
            "out_name": "01b_YShirakawaSocial_person.jpg",
            "file": "YShirakawaSocial_R3.png",
            "inst_idx": 1,
            "crop_kind": "person",
            "target": "shirakawa_yuina",
            "desc": "Yuina Social (Person crop) - Upper body & hair ornament",
        },
        {
            "out_name": "02_Still_MC02_09_04_01_001_AsakuraKaren_head.jpg",
            "file": "Still_MC02_09_04_01_001.png",
            "inst_idx": 0,
            "crop_kind": "head",
            "target": "asakura_karen_(heaven_burns_red)",
            "desc": "Civilian black-hair Asakura Karen (Head crop)",
        },
        {
            "out_name": "03_YShirakawaArmor_head.jpg",
            "file": "YShirakawaArmor_R3.png",
            "inst_idx": 0,
            "crop_kind": "head",
            "target": "shirakawa_yuina",
            "desc": "Yuina Armor with headgear (Head crop)",
        },
        {
            "out_name": "04_Still_MC03_08_10_00_002_IzumiYuki_noglasses_head.jpg",
            "file": "Still_MC03_08_10_00_002.png",
            "inst_idx": 1,
            "crop_kind": "head",
            "target": "izumi_yuki",
            "desc": "31A Bath - Izumi Yuki without glasses (Head crop)",
        },
        {
            "out_name": "05_Still_MC04B_06_Aikawa_AkagiRumi_head.jpg",
            "file": "Still_MC04B_06_Aikawa_05_01_001.png",
            "inst_idx": 1,
            "crop_kind": "head",
            "target": "akagi_rumi",
            "desc": "Akagi Rumi (Head crop) - v3 only (v1 lacks Rumi)",
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Camie-v2 ViT-Base backbone on {device} (fp32)...")
    backbone = load_camie_backbone_fp32(device)

    summary = []
    for c in cases:
        img_path = PROJECT_ROOT / "data" / "test_sets" / "neo" / c["file"]
        inst = preds_by_file[c["file"]]["instances"][c["inst_idx"]]
        if c["crop_kind"] == "head":
            bbox = inst["head_bbox"]
            pad_ratio = 0.15
            coef_key, inter_key = "h_coef", "h_intercept"
        else:
            bbox = inst["person_bbox"]
            pad_ratio = 0.06
            coef_key, inter_key = "p_coef", "p_intercept"

        with Image.open(img_path) as raw_im:
            pil_rgb = raw_im.convert("RGB")
        crop = crop_box(pil_rgb, bbox, pad_ratio=pad_ratio)
        orig_512 = pad_square_rgb(crop).resize((512, 512), Image.BICUBIC)
        orig_512_np = np.asarray(orig_512, dtype=np.uint8)
        x = preprocess_pil(crop, 512).unsqueeze(0)

        r1 = compute_vit_gradcam(backbone, x, v1[coef_key], v1[inter_key], v1_classes, c["target"], device)
        r3 = compute_vit_gradcam(backbone, x, v3[coef_key], v3[inter_key], v3_classes, c["target"], device)

        v1_overlay = overlay_cam(orig_512_np, r1["cam_32"])
        v3_overlay = overlay_cam(orig_512_np, r3["cam_32"])

        t_short = short_name(c["target"])
        orig_sub = f"Original ({c['crop_kind']} crop)\nTarget: {t_short}"
        if r1["has_target"]:
            v1_sub = (
                f"v1 (HBR-54) | CAM={t_short} ({r1['target_prob']:.2f})\n"
                f"Top-1: {short_name(r1['pred'])} ({r1['pred_prob']:.2f})"
            )
        else:
            v1_sub = (
                f"v1 (HBR-54, no {t_short}) | CAM=Top-1\n"
                f"Top-1: {short_name(r1['pred'])} ({r1['pred_prob']:.2f})"
            )
        v3_sub = (
            f"v3 (HBR-55) | CAM={t_short} ({r3['target_prob']:.2f})\n"
            f"Top-1: {short_name(r3['pred'])} ({r3['pred_prob']:.2f})"
        )

        panel = build_3panel(orig_512, v1_overlay, v3_overlay, c["desc"], orig_sub, v1_sub, v3_sub)
        out_path = out_dir / c["out_name"]
        panel.save(out_path, quality=95)

        rec = {
            "out_name": c["out_name"],
            "file": c["file"],
            "crop_kind": c["crop_kind"],
            "target": c["target"],
            "v1_pred": r1["pred"],
            "v1_pred_prob": round(r1["pred_prob"], 4),
            "v1_target_prob": round(r1["target_prob"], 4) if r1["target_prob"] is not None else None,
            "v3_pred": r3["pred"],
            "v3_pred_prob": round(r3["pred_prob"], 4),
            "v3_target_prob": round(r3["target_prob"], 4) if r3["target_prob"] is not None else None,
        }
        summary.append(rec)
        print(
            f"Saved {c['out_name']} | "
            f"v1: {short_name(r1['pred'])}({r1['pred_prob']:.2f}, tgt={r1['target_prob'] if r1['target_prob'] is None else round(r1['target_prob'], 2)}) | "
            f"v3: {short_name(r3['pred'])}({r3['pred_prob']:.2f}, tgt={round(r3['target_prob'], 2)})"
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nAll Grad-CAM comparisons saved to: {out_dir}")


if __name__ == "__main__":
    main()
