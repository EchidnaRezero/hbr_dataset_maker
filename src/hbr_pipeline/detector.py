"""ONNX anime person/head detection, pairing, cropping, preprocessing, and visualization helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

DEFAULT_PERSON_REPO = "deepghs/anime_person_detection"
DEFAULT_PERSON_FILE = "person_detect_v1.3_s/model.onnx"
DEFAULT_HEAD_REPO = "deepghs/anime_head_detection"
DEFAULT_HEAD_FILE = "head_detect_v2.0_s_yv11/model.onnx"


def get_ort_providers() -> list[str]:
    avail = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in avail:
        if sys.platform == "win32" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def load_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        return None


def letterbox(
    img_rgb: np.ndarray,
    new_size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    h, w = img_rgb.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_size - nw) // 2
    pad_h = (new_size - nh) // 2
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    canvas[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized
    return canvas, scale, pad_w, pad_h


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloOnnxDetector:
    """Single-class YOLO ONNX detector (e.g. deepghs anime person/head models)."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
        providers: list[str],
        conf_thr: float = 0.35,
        iou_thr: float = 0.5,
    ) -> None:
        model_path = hf_hub_download(repo_id, filename)
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    def detect(self, img_rgb: np.ndarray) -> list[dict[str, Any]]:
        h, w = img_rgb.shape[:2]
        canvas, scale, pad_w, pad_h = letterbox(img_rgb, 640)
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.in_name: x})[0][0]
        if out.shape[0] == 5:
            out = out.T
        scores = out[:, 4]
        mask = scores >= self.conf_thr
        if not np.any(mask):
            return []
        sel = out[mask]
        sc = scores[mask]
        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        x1 = np.clip((cx - bw / 2 - pad_w) / scale, 0, w)
        y1 = np.clip((cy - bh / 2 - pad_h) / scale, 0, h)
        x2 = np.clip((cx + bw / 2 - pad_w) / scale, 0, w)
        y2 = np.clip((cy + bh / 2 - pad_h) / scale, 0, h)
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms_xyxy(boxes, sc, self.iou_thr)
        return [{"bbox": [float(v) for v in boxes[idx]], "conf": float(sc[idx])} for idx in keep]


class PersonHeadDetector:
    """Paired anime person + head ONNX detector wrapper."""

    def __init__(
        self,
        providers: list[str] | None = None,
        conf_thr: float = 0.25,
        iou_thr: float = 0.5,
        person_repo: str = DEFAULT_PERSON_REPO,
        person_file: str = DEFAULT_PERSON_FILE,
        head_repo: str = DEFAULT_HEAD_REPO,
        head_file: str = DEFAULT_HEAD_FILE,
    ) -> None:
        resolved_providers = providers if providers is not None else get_ort_providers()
        self.person_det = YoloOnnxDetector(
            person_repo, person_file, resolved_providers, conf_thr=conf_thr, iou_thr=iou_thr
        )
        self.head_det = YoloOnnxDetector(
            head_repo, head_file, resolved_providers, conf_thr=conf_thr, iou_thr=iou_thr
        )

    def detect_paired(self, img_rgb: np.ndarray) -> list[dict[str, Any]]:
        persons = self.person_det.detect(img_rgb)
        heads = self.head_det.detect(img_rgb)
        return pair_persons_and_heads(persons, heads)


def pair_persons_and_heads(persons: list[dict], heads: list[dict]) -> list[dict]:
    """Associate each detected person box with the best head box inside it (plus unmatched heads)."""
    used_heads: set[int] = set()
    instances: list[dict[str, Any]] = []
    persons_sorted = sorted(persons, key=lambda d: d["bbox"][0])
    for p in persons_sorted:
        px1, py1, px2, py2 = p["bbox"]
        best_hi = None
        best_score = -1.0
        for hi, h in enumerate(heads):
            if hi in used_heads:
                continue
            hx1, hy1, hx2, hy2 = h["bbox"]
            hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            ix1, iy1 = max(px1, hx1), max(py1, hy1)
            ix2, iy2 = min(px2, hx2), min(py2, hy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            h_area = max(1.0, (hx2 - hx1) * (hy2 - hy1))
            ioa = inter / h_area
            if ioa >= 0.45 or (px1 <= hcx <= px2 and py1 <= hcy <= py2):
                score = ioa + 0.1 * h["conf"]
                if score > best_score:
                    best_score = score
                    best_hi = hi
        if best_hi is not None:
            used_heads.add(best_hi)
            instances.append({"person_bbox": p["bbox"], "head_bbox": heads[best_hi]["bbox"]})
        else:
            instances.append({"person_bbox": p["bbox"], "head_bbox": None})

    for hi, h in enumerate(heads):
        if hi not in used_heads:
            instances.append({"person_bbox": None, "head_bbox": h["bbox"]})
    instances.sort(key=lambda d: (d["person_bbox"] or d["head_bbox"])[0])
    return instances


def pad_square_rgb(im: Image.Image, fill: tuple[int, int, int] = (124, 116, 104)) -> Image.Image:
    w, h = im.size
    if w == h:
        return im
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def crop_box(im: Image.Image, bbox: list[float], pad_ratio: float = 0.08) -> Image.Image:
    w, h = im.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(w, int(round(x2 + px)))
    cy2 = min(h, int(round(y2 + py)))
    if cx2 <= cx1 + 4 or cy2 <= cy1 + 4:
        return im
    return im.crop((cx1, cy1, cx2, cy2))


def preprocess_pil(im: Image.Image, size: int = 512) -> torch.Tensor:
    import torch

    sq = pad_square_rgb(im.convert("RGB"))
    sq = sq.resize((size, size), Image.BICUBIC)
    arr = np.asarray(sq, dtype=np.float32) / 255.0
    arr = (arr.transpose(2, 0, 1) - MEAN) / STD
    return torch.from_numpy(arr)


def collapse_costume_subtags(chars: list[str]) -> list[str]:
    base_chars: list[str] = []
    for c in chars:
        base_no_hbr = c.removesuffix("_(heaven_burns_red)")
        is_sub = False
        for other in chars:
            if other == c:
                continue
            other_base = other.removesuffix("_(heaven_burns_red)")
            if base_no_hbr.startswith(other_base + "_("):
                is_sub = True
                break
        if not is_sub:
            base_chars.append(c)
    return base_chars


def load_scaled_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_result(img_rgb: np.ndarray, instances: list[dict], title: str, out_path: Path) -> None:
    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)
    short_side = min(im.width, im.height)
    font_size = max(18, int(short_side * 0.028))
    title_size = max(18, int(short_side * 0.024))
    line_w = max(3, int(short_side * 0.004))
    font = load_scaled_font(font_size)
    title_font = load_scaled_font(title_size)

    # Draw title bar at bottom-left FIRST so character labels are never covered
    tb = draw.textbbox((0, 0), title[:140], font=title_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tpad = max(6, title_size // 4)
    ty1 = im.height - (th + tpad * 2)
    draw.rectangle([0, ty1, min(im.width, tw + tpad * 2), im.height], fill=(0, 0, 0))
    draw.text((tpad, ty1 + tpad - tb[1]), title[:140], fill=(255, 255, 0), font=title_font)

    colors = [(0, 255, 0), (255, 180, 0), (0, 230, 255), (255, 110, 255), (255, 255, 0), (120, 255, 120)]
    for idx, inst in enumerate(instances):
        col = colors[idx % len(colors)]
        pb = inst["person_bbox"]
        hb = inst["head_bbox"]
        label = inst.get("display_label") or f"{inst['pred']} ({inst['conf']:.2f})"
        if inst.get("matched_pred") and inst["matched_pred"] != inst["pred"]:
            label += f" [M:{inst['matched_pred']}]"

        if pb is not None:
            x1, y1, x2, y2 = pb
            draw.rectangle([x1, y1, x2, y2], outline=col, width=line_w)
        if hb is not None:
            x1, y1, x2, y2 = hb
            draw.rectangle([x1, y1, x2, y2], outline=col, width=max(2, line_w - 1))

        # Anchor label to head box if available (much clearer on full-height CGs), else person box
        anchor_box = hb if hb is not None else pb
        if anchor_box is not None:
            ax1, ay1, ax2, ay2 = anchor_box
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = max(5, font_size // 5)
            if ay1 >= th + pad * 2 + 4:
                by1 = ay1 - (th + pad * 2) - 2
            elif ay2 + th + pad * 2 + 4 <= im.height:
                by1 = ay2 + 2
            else:
                by1 = max(0, ay1 + 4)
            bx1 = max(0, min(ax1, im.width - tw - pad * 2))
            draw.rectangle([bx1, by1, bx1 + tw + pad * 2, by1 + th + pad * 2], fill=(0, 0, 0))
            draw.text((bx1 + pad, by1 + pad - tb[1]), label, fill=col, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def draw_vis(img_rgb: np.ndarray, persons: list[dict], heads: list[dict], title: str, out_path: Path) -> None:
    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)
    for d in persons:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, max(0, y1 + 4)), f"P:{d['conf']:.2f}", fill=(0, 255, 0))
    for d in heads:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 255), width=3)
        draw.text((x1 + 4, max(0, y1 + 4)), f"H:{d['conf']:.2f}", fill=(0, 220, 255))
    draw.rectangle([0, 0, min(im.width, 900), 26], fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
