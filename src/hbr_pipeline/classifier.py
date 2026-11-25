"""Camie-v2 ONNX tagger, ViT-Base embedding extractor, linear head inference, and 5-tier safety-valve routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from src.hbr_pipeline.detector import crop_box, letterbox, preprocess_pil

EXT_COLLAB_COPYRIGHTS = {
    "angel_beats!",
    "persona",
    "persona_5",
    "persona_4",
    "persona_3",
    "summer_pockets",
    "clannad",
    "little_busters!",
    "charlotte_(anime)",
    "kanon",
    "air_(visual_novel)",
    "jikkyou_powerful_pro_yakyuu",
}

EXT_COLLAB_SEED_CHARS = {
    # Angel Beats!
    "yui_(angel_beats!)",
    "iwasawa_masami",
    "hisako_(angel_beats!)",
    "sekine_shiori",
    "tachibana_kanade",
    "nakamura_yuri",
    "irie_miyuki",
    "otonashi_yuzuru",
    "hinata_hideki",
    "shiina_(angel_beats!)",
    "yusa_(angel_beats!)",
    # Persona
    "yoshizawa_kasumi",
    "violet_(persona_5)",
    "amamiya_ren",
    "joker_(persona_5)",
    "takamaki_ann",
    "panther_(persona_5)",
    "sakamoto_ryuji",
    "morgana_(persona_5)",
    "kitagawa_yusuke",
    "niijima_makoto",
    "sakura_futaba",
    "okumura_haru",
    "akechi_goro",
    # Key: Summer Pockets, CLANNAD, Little Busters!, Charlotte, Kanon, AIR
    "naruse_shiroha",
    "sorakado_ao",
    "kushima_kamome",
    "tsumugi_wenders",
    "katou_umi",
    "nomiki",
    "shizuku_(summer_pockets)",
    "furukawa_nagisa",
    "fujibayashi_kyou",
    "fujibayashi_ryou",
    "sakagami_tomoyo",
    "ichinose_kotomi",
    "ibuki_fuko",
    "okazaki_ushio",
    "natsume_rin",
    "noumi_kudryavka",
    "kamikita_komari",
    "saigusa_haruka",
    "futaki_kanata",
    "kurugaya_yuiko",
    "nishizono_mio",
    "tomori_nao",
    "nishimori_yusa",
    "kurobane_misa",
    "otosaka_ayumi",
    "tsukimiya_ayu",
    "minase_nayuki",
    "sawatari_makoto",
    "kawasumi_mai",
    "misaka_shiori",
    "kamio_misuzu",
    "tohno_minagi",
    "kirishima_kano",
}


class CamieV2Onnx:
    """Out-of-the-box Camais03/camie-tagger-v2 ONNX inference wrapper."""

    def __init__(self, providers: list[str]) -> None:
        meta_path = hf_hub_download("Camais03/camie-tagger-v2", "camie-tagger-v2-metadata.json")
        onnx_path = hf_hub_download("Camais03/camie-tagger-v2", "camie-tagger-v2.onnx")
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.refined_idx = (
            self.out_names.index("refined_predictions") if "refined_predictions" in self.out_names else 1
        )

        tm = self.meta["dataset_info"]["tag_mapping"]
        self.tag_to_idx: dict[str, int] = tm["tag_to_idx"]
        self.idx_to_tag: dict[int, str] = {int(k): v for k, v in tm["idx_to_tag"].items()}
        self.tag_to_cat: dict[str, str] = tm["tag_to_category"]

        self.char_indices = np.array(
            [idx for idx, tag in self.idx_to_tag.items() if self.tag_to_cat.get(tag) == "character"],
            dtype=np.int64,
        )
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        canvas, _, _, _ = letterbox(img_rgb, 512, color=(124, 116, 104))
        x = canvas.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        x = (x - self.mean) / self.std
        return x

    def predict_batch(self, imgs_rgb: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self.preprocess(im) for im in imgs_rgb], axis=0)
        outs = self.sess.run(None, {self.in_name: batch})
        logits = outs[self.refined_idx]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return probs

    def top_character(self, img_rgb: np.ndarray) -> tuple[str, float]:
        probs = self.predict_batch([img_rgb])[0]
        cprobs = probs[self.char_indices]
        bi = int(np.argmax(cprobs))
        tag = self.idx_to_tag[int(self.char_indices[bi])]
        return tag, float(cprobs[bi])


def load_camie_backbone(device: torch.device, dtype: torch.dtype | None = None) -> torch.nn.Module:
    """Load frozen Camie-v2 ViT-Base (86M, 512x512) backbone from safetensors."""
    st_path = hf_hub_download("Camais03/camie-tagger-v2", "camie-tagger-v2.safetensors")
    sd = load_file(st_path, device="cpu")
    vit_sd = {k[len("backbone.vit.") :]: v for k, v in sd.items() if k.startswith("backbone.vit.")}
    model = timm.create_model("vit_base_patch16_384", pretrained=False, img_size=512, num_classes=0)
    model.load_state_dict(vit_sd, strict=True)
    target_dtype = dtype if dtype is not None else (torch.float16 if device.type == "cuda" else torch.float32)
    return model.to(device=device, dtype=target_dtype).eval()


def load_camie_backbone_fp32(device: torch.device) -> torch.nn.Module:
    """Load Camie-v2 ViT-Base backbone in fp32 (for Grad-CAM / gradient inspection)."""
    return load_camie_backbone(device, dtype=torch.float32)


@torch.inference_mode()
def embed_crops(
    backbone: torch.nn.Module,
    pil_crops: list[Image.Image],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract L2-normalized 768-D Camie-v2 ViT embeddings for a list of PIL crops."""
    if not pil_crops:
        return np.zeros((0, 768), dtype=np.float32)
    out_list: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    dtype = torch.float16 if use_amp else torch.float32
    for s in range(0, len(pil_crops), batch_size):
        xs = torch.stack([preprocess_pil(im, 512) for im in pil_crops[s : s + batch_size]], dim=0).to(
            device=device, dtype=dtype
        )
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                e = backbone(xs).float().cpu().numpy()
        else:
            e = backbone(xs).float().cpu().numpy()
        out_list.append(e)
    emb = np.concatenate(out_list, axis=0)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)


def predict_from_npz(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return softmax(X @ coef.T + intercept[None, :])


def load_ext_whitelist(meta_path: Path | None, hbr_classes: set[str] | dict[str, Any]) -> set[str]:
    """Build external collab character whitelist from seed chars + Danbooru metadata."""
    ext_whitelist = set(EXT_COLLAB_SEED_CHARS)
    if meta_path is not None and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                if set(m.get("copyrights", [])) & EXT_COLLAB_COPYRIGHTS:
                    for ch in m.get("characters", []):
                        if ch not in hbr_classes:
                            ext_whitelist.add(ch)
    return ext_whitelist


def classify_instance_probs(
    idx: int,
    p_map: dict[int, np.ndarray],
    h_map: dict[int, np.ndarray],
    weights: Any,
) -> tuple[np.ndarray, str]:
    """Compute class probability vector and mode ('person+head', 'person_only', or 'head_only')."""
    if idx in p_map and idx in h_map:
        feat = np.concatenate([p_map[idx], h_map[idx]])
        feat /= np.linalg.norm(feat) + 1e-8
        probs = predict_from_npz(feat[None], weights["ph_coef"], weights["ph_intercept"])[0]
        return probs, "person+head"
    if idx in p_map:
        probs = predict_from_npz(p_map[idx][None], weights["p_coef"], weights["p_intercept"])[0]
        return probs, "person_only"
    probs = predict_from_npz(h_map[idx][None], weights["h_coef"], weights["h_intercept"])[0]
    return probs, "head_only"


def route_safety_valve(
    pil_rgb: Image.Image,
    bbox: list[float],
    pred: str,
    conf: float,
    area_pct: float,
    orig_camie: CamieV2Onnx,
    c2i: dict[str, int],
    ext_whitelist: set[str],
    accept_thr: float = 0.45,
    ext_alarm_thr: float = 0.75,
    min_area_pct: float = 1.5,
) -> dict[str, Any]:
    """5-tier safety-valve router: CROWD_MASK / EXT_COLLAB / REVIEW_COSPLAY_OR_EXT / ACCEPT / REVIEW."""
    if area_pct < min_area_pct:
        return {
            "character": pred,
            "pred": pred,
            "conf": conf,
            "tier": "CROWD_MASK",
            "display_label": f"[CROWD:{area_pct:.1f}%]",
        }

    ext_crop = np.array(crop_box(pil_rgb, bbox, pad_ratio=0.06))
    etag, eprob = orig_camie.top_character(ext_crop)
    etag_is_hbr = (etag in c2i) or (etag.removesuffix("_(heaven_burns_red)") in c2i)

    if conf < accept_thr and eprob >= 0.70 and etag in ext_whitelist and not etag_is_hbr:
        return {
            "character": f"EXT:{etag}",
            "pred": f"EXT:{etag}",
            "conf": eprob,
            "tier": "EXT_COLLAB",
            "ext_conf": round(eprob, 4),
            "display_label": f"[EXT] {etag} ({eprob:.2f})",
        }
    if conf < 0.60 and eprob >= ext_alarm_thr and not etag_is_hbr and etag not in ext_whitelist:
        return {
            "character": pred,
            "pred": pred,
            "conf": conf,
            "tier": "REVIEW_COSPLAY_OR_EXT",
            "ext_conf": round(eprob, 4),
            "ext_alarm": {"tag": etag, "prob": round(eprob, 4)},
            "display_label": f"?EXT_ALARM:{etag}({eprob:.2f})|HBR:{pred}({conf:.2f})",
        }
    if conf >= accept_thr:
        return {
            "character": pred,
            "pred": pred,
            "conf": conf,
            "tier": "ACCEPT",
            "display_label": f"{pred} ({conf:.2f})",
        }
    return {
        "character": pred,
        "pred": pred,
        "conf": conf,
        "tier": "REVIEW",
        "display_label": f"?REVIEW:{pred} ({conf:.2f})",
    }
