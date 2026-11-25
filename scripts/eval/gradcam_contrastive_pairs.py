"""Pairwise Contrastive ViT Grad-CAM for top look-alike character pairs.

For each pair (A, B), computes:
- Standard Grad-CAM on logit_A (where the model looks to recognize A)
- Contrastive Grad-CAM on (logit_A - logit_B) (which specific visual cues separate A from B)

Saves 2x3 panels to `outputs/gradcam/contrastive/`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.eval.compare_gradcam_v1_v3 import get_font, overlay_cam
from src.hbr_pipeline.classifier import load_camie_backbone_fp32
from src.hbr_pipeline.detector import crop_box, pad_square_rgb, preprocess_pil


def compute_contrastive_gradcam(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    coef: np.ndarray,
    intercept: np.ndarray,
    idx_pos: int,
    idx_neg: int | None,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Compute 32x32 Grad-CAM for logit[idx_pos] (or logit[idx_pos] - logit[idx_neg])."""
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

        feat = backbone(x_in)
        feat_norm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        logits = feat_norm @ coef_t.T + inter_t[None, :]
        score = logits[0, idx_pos] if idx_neg is None else (logits[0, idx_pos] - logits[0, idx_neg])
        score_val = float(score.item())
        score.backward()

        act = captured["act"][0, 1:, :]
        grad = captured["grad"][0, 1:, :]
        weights = grad.mean(dim=0)
        cam = torch.relu((act * weights[None, :]).sum(dim=-1))
        cam_2d = cam.reshape(32, 32).detach().cpu().numpy()
        cmin, cmax = float(cam_2d.min()), float(cam_2d.max())
        cam_norm = (cam_2d - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else np.zeros_like(cam_2d)
        return cam_norm, score_val
    finally:
        handle.remove()


def fetch_best_crop(conn: sqlite3.Connection, char_name: str, prefer_uid: str | None = None) -> tuple[Path, list[float]]:
    if prefer_uid:
        row = conn.execute(
            """
            SELECT i.local_path, b.head_bbox_json
            FROM boxes b JOIN images i ON i.uid = b.image_uid
            WHERE i.uid = ? AND b.character = ? AND b.head_bbox_json IS NOT NULL
            LIMIT 1
            """,
            (prefer_uid, char_name),
        ).fetchone()
        if row:
            return Path(row[0]), json.loads(row[1])
    row = conn.execute(
        """
        SELECT i.local_path, b.head_bbox_json
        FROM boxes b JOIN images i ON i.uid = b.image_uid
        WHERE i.is_duplicate = 0 AND b.character = ? AND b.tier = 'ACCEPT' AND b.head_bbox_json IS NOT NULL
        ORDER BY (CASE WHEN i.source='official' THEN 0 ELSE 1 END), b.free_conf DESC
        LIMIT 1
        """,
        (char_name,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"No head crop found for {char_name}")
    return Path(row[0]), json.loads(row[1])


def render_pair_panel(
    backbone: torch.nn.Module,
    conn: sqlite3.Connection,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: list[str],
    pair_spec: tuple[tuple[str, str, str | None], tuple[str, str, str | None]],
    out_path: Path,
    device: torch.device,
) -> None:
    (ca, na, uid_a), (cb, nb, uid_b) = pair_spec
    ia, ib = classes.index(ca), classes.index(cb)

    tile = 384
    pad = 16
    row_hdr = 42
    title_h = 56
    W = tile * 3 + pad * 4
    H = title_h + 2 * (row_hdr + tile + pad) + pad

    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    draw = ImageDraw.Draw(canvas)
    f_title = get_font(22)
    f_sub = get_font(16)

    draw.text(
        (pad, 14),
        f"Pairwise Contrastive ViT Grad-CAM:  {na}  vs  {nb}",
        fill=(17, 24, 39),
        font=f_title,
    )

    rows = [
        (ca, na, uid_a, ia, ib, nb),
        (cb, nb, uid_b, ib, ia, na),
    ]
    for r_idx, (ch, name, uid, i_pos, i_neg, other_name) in enumerate(rows):
        img_path, bbox = fetch_best_crop(conn, ch, uid)
        img_pil = Image.open(img_path).convert("RGB")
        crop_pil = crop_box(img_pil, bbox, pad_ratio=0.12)
        sq_pil = pad_square_rgb(crop_pil).resize((512, 512), Image.BICUBIC)
        base_512 = np.array(sq_pil)
        x = preprocess_pil(crop_pil, 512).unsqueeze(0)

        cam_std, logit_std = compute_contrastive_gradcam(backbone, x, coef, intercept, i_pos, None, device)
        cam_con, margin = compute_contrastive_gradcam(backbone, x, coef, intercept, i_pos, i_neg, device)

        im_orig = sq_pil.resize((tile, tile), Image.BICUBIC)
        im_std = overlay_cam(base_512, cam_std, alpha=0.48).resize((tile, tile), Image.BICUBIC)
        im_con = overlay_cam(base_512, cam_con, alpha=0.48).resize((tile, tile), Image.BICUBIC)

        y_hdr = title_h + r_idx * (row_hdr + tile + pad)
        y_img = y_hdr + row_hdr
        short_pos = name.split()[-1]
        short_neg = other_name.split()[-1]
        labels = [
            f"{name} — Original Head",
            f"Why {short_pos}? (logit = {logit_std:+.2f})",
            f"Why {short_pos} > {short_neg}? (margin = {margin:+.2f})",
        ]
        for c_idx, (im, lbl) in enumerate(zip([im_orig, im_std, im_con], labels)):
            x0 = pad + c_idx * (tile + pad)
            draw.text((x0 + 4, y_hdr + 12), lbl, fill=(31, 41, 55), font=f_sub)
            canvas.paste(im, (x0, y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=93)
    print(f"Saved: {out_path}")


def render_failure_contrastive_panel(
    backbone: torch.nn.Module,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: list[str],
    out_path: Path,
    device: torch.device,
) -> None:
    preds_neo: dict[str, dict] = {}
    for line in (PROJECT_ROOT / "outputs" / "eval_v3" / "neo" / "predictions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        preds_neo[r["file"]] = r

    cases = [
        ("IMinaseHR_R3.png", 0, "minase_ichigo", "Ichigo", "carole_reaper", "Carole", "Wrong Pred"),
        ("Still_MC05B_05_09_00_003.png", 2, "izumi_yuki", "Yuki", "tachibana_kanade", "Kanade", "Wrong Pred"),
        ("Still_MC03_08_10_00_002.png", 1, "asakura_karen_(heaven_burns_red)", "Karen", "yanagi_mion", "Mion", "Low-conf #2"),
    ]

    tile = 360
    pad = 16
    row_hdr = 42
    title_h = 56
    W = tile * 3 + pad * 4
    H = title_h + len(cases) * (row_hdr + tile + pad) + pad

    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    draw = ImageDraw.Draw(canvas)
    f_title = get_font(21)
    f_sub = get_font(15)

    draw.text(
        (pad, 14),
        "Contrastive ViT Grad-CAM on v3 Hard / Misclassification Cases (Competitor vs True GT)",
        fill=(17, 24, 39),
        font=f_title,
    )

    for r_idx, (fname, inst_idx, gt_cls, gt_short, comp_cls, comp_short, tag) in enumerate(cases):
        img_path = PROJECT_ROOT / "data" / "test_sets" / "neo" / fname
        bbox = preds_neo[fname]["instances"][inst_idx]["head_bbox"]
        img_pil = Image.open(img_path).convert("RGB")
        crop_pil = crop_box(img_pil, bbox, pad_ratio=0.08)
        sq_pil = pad_square_rgb(crop_pil).resize((512, 512), Image.BICUBIC)
        base_512 = np.array(sq_pil)
        x = preprocess_pil(crop_pil, 512).unsqueeze(0)

        i_gt = classes.index(gt_cls)
        i_comp = classes.index(comp_cls)

        cam_comp, m_comp = compute_contrastive_gradcam(backbone, x, coef, intercept, i_comp, i_gt, device)
        cam_gt, m_gt = compute_contrastive_gradcam(backbone, x, coef, intercept, i_gt, i_comp, device)

        im_orig = sq_pil.resize((tile, tile), Image.BICUBIC)
        im_comp = overlay_cam(base_512, cam_comp, alpha=0.48).resize((tile, tile), Image.BICUBIC)
        im_gt = overlay_cam(base_512, cam_gt, alpha=0.48).resize((tile, tile), Image.BICUBIC)

        y_hdr = title_h + r_idx * (row_hdr + tile + pad)
        y_img = y_hdr + row_hdr
        labels = [
            f"GT: {gt_short}  ({tag}: {comp_short})",
            f"Why {comp_short} > {gt_short}? ({m_comp:+.2f})",
            f"Why {gt_short} > {comp_short}? ({m_gt:+.2f})",
        ]
        for c_idx, (im, lbl) in enumerate(zip([im_orig, im_comp, im_gt], labels)):
            x0 = pad + c_idx * (tile + pad)
            draw.text((x0 + 4, y_hdr + 12), lbl, fill=(31, 41, 55), font=f_sub)
            canvas.paste(im, (x0, y_img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=93)
    print(f"Saved: {out_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = np.load(PROJECT_ROOT / "models/v3_pure_hbr55_plus_multi_camie.npz", allow_pickle=True)
    classes = list(w["classes"])
    h_coef = w["h_coef"].astype(np.float32)
    h_inter = w["h_intercept"].astype(np.float32)

    backbone = load_camie_backbone_fp32(device)
    conn = sqlite3.connect(PROJECT_ROOT / "data/db/master_hbr.db")
    out_dir = PROJECT_ROOT / "outputs/gradcam/contrastive"

    try:
        render_pair_panel(
            backbone,
            conn,
            h_coef,
            h_inter,
            classes,
            (
                ("shirakawa_yuina", "Shirakawa Yuina", "official:neo/Still_MC04_04_09_02_001.png"),
                ("tojo_tsukasa", "Tojo Tsukasa", "official:neo/Still_MC01_12_09_04_002.png"),
            ),
            out_dir / "contrastive_yuina_vs_tsukasa.jpg",
            device,
        )
        render_pair_panel(
            backbone,
            conn,
            h_coef,
            h_inter,
            classes,
            (
                ("higuchi_seika", "Higuchi Seika", "official:neo/SHiguchiDoctor_R3.png"),
                ("izumi_yuki", "Izumi Yuki", "official:neo/Still_MC05A_14_05_00_003.png"),
            ),
            out_dir / "contrastive_seika_vs_yuki.jpg",
            device,
        )
        render_failure_contrastive_panel(
            backbone,
            h_coef,
            h_inter,
            classes,
            out_dir / "contrastive_failure_cases.jpg",
            device,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
