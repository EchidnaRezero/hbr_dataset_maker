"""64-bit perceptual hash (pHash) computation and persistent deduplication index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct


def compute_phash_hex(pil_rgb: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Pure PIL + scipy DCT pHash (identical algorithm to imagehash.phash, 64-bit hex)."""
    img_size = hash_size * highfreq_factor
    gray = pil_rgb.convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    d = dct(dct(pixels, axis=0), axis=1)
    d_low = d[:hash_size, :hash_size]
    med = np.median(d_low[1:, 1:]) if hash_size > 1 else np.median(d_low)
    bits = (d_low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:016x}"


def phash_hamming(hash_a: str | int, hash_b: str | int) -> int:
    """Compute Hamming distance between two 64-bit hex or integer pHashes."""
    a = int(hash_a, 16) if isinstance(hash_a, str) else int(hash_a)
    b = int(hash_b, 16) if isinstance(hash_b, str) else int(hash_b)
    return (a ^ b).bit_count()


class PersistentPHashIndex:
    """Persistent 64-bit pHash index for fast deduplication across current and future datasets."""

    def __init__(self, index_path: Path, max_hamming: int = 4) -> None:
        self.index_path = index_path
        self.max_hamming = max_hamming
        self.hashes: list[int] = []
        self.uids: list[str] = []
        self.uid_set: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = rec["uid"]
                if uid in self.uid_set:
                    continue
                self.uid_set.add(uid)
                if rec.get("phash"):
                    self.hashes.append(int(rec["phash"], 16))
                    self.uids.append(uid)

    def find_duplicate(self, phash_hex: str) -> str | None:
        if not phash_hex:
            return None
        ph = int(phash_hex, 16)
        thr = self.max_hamming
        for existing_h, existing_uid in zip(self.hashes, self.uids):
            if (ph ^ existing_h).bit_count() <= thr:
                return existing_uid
        return None

    def add(self, uid: str, source: str, path: str, phash_hex: str) -> None:
        if uid in self.uid_set:
            return
        self.uid_set.add(uid)
        if phash_hex:
            self.hashes.append(int(phash_hex, 16))
            self.uids.append(uid)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"uid": uid, "source": source, "path": path, "phash": phash_hex}, ensure_ascii=False)
                + "\n"
            )
