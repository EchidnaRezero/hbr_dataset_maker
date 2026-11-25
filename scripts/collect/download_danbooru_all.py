#!/usr/bin/env python3
"""Download ALL static downloadable Danbooru Heaven Burns Red images.

Reads data/images/danbooru_meta.jsonl, filters to static downloadable images
(has_file_url == True and file_ext in {'jpg', 'jpeg', 'png', 'webp'}),
reuses already-downloaded local images if present, downloads remaining images
with 8 threads (large_file_url -> file_url fallback, polite User-Agent, retry
with backoff, resume-safe), computes pHash & wHash for every image, and writes
data/images/danbooru_manifest.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imagehash
import requests
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
DEFAULT_UA = "HBRDatasetResearch/0.1 (Python requests)"

_thread_local = threading.local()


def get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": DEFAULT_UA})
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


def collapse_costume_subtags(chars: list[str]) -> list[str]:
    base_chars: list[str] = []
    for c in chars:
        bn = c.removesuffix("_(heaven_burns_red)")
        if not any(
            other != c and bn.startswith(other.removesuffix("_(heaven_burns_red)") + "_(")
            for other in chars
        ):
            base_chars.append(c)
    return base_chars


def url_ext(url: str, fallback_ext: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
    if suffix in IMAGE_EXTS:
        return suffix
    fb = fallback_ext.lower().lstrip(".")
    return fb if fb in IMAGE_EXTS else "jpg"


def scan_existing_dirs(dirs: list[Path]) -> dict[int, Path]:
    """Scan existing image directories and map numeric post_id stem -> Path."""
    existing: dict[int, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for fp in d.rglob("*"):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower().lstrip(".")
            if ext not in IMAGE_EXTS:
                continue
            if not fp.stem.isdigit():
                continue
            try:
                if fp.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            pid = int(fp.stem)
            if pid not in existing:
                existing[pid] = fp
    return existing


def download_or_reuse(
    rec: dict[str, Any],
    images_dir: Path,
    existing_map: dict[int, Path],
    timeout: float = 30.0,
) -> tuple[dict[str, Any], str, str | None, str]:
    """Return (rec, status, local_path, downloaded_url)."""
    pid = rec["id"]

    # 1. Resume-safe: check if destination already exists under any valid extension
    for ext in IMAGE_EXTS:
        cand = images_dir / f"{pid}.{ext}"
        try:
            if cand.exists() and cand.stat().st_size > 0:
                return rec, "exists", str(cand), ""
        except OSError:
            pass

    # 2. Reuse already-downloaded image from local caches (read-only copy)
    src = existing_map.get(pid)
    if src is not None and src.exists():
        ext = src.suffix.lower().lstrip(".")
        if ext not in IMAGE_EXTS:
            ext = (rec.get("file_ext") or "jpg").lower()
        dest = images_dir / f"{pid}.{ext}"
        tmp = images_dir / f"{pid}.{ext}.part"
        try:
            shutil.copy2(src, tmp)
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                return rec, "reused", str(dest), ""
        except OSError:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # 3. Download using large_file_url with fallback to file_url
    urls: list[str] = []
    if rec.get("large_file_url"):
        urls.append(rec["large_file_url"])
    if rec.get("file_url") and rec["file_url"] not in urls:
        urls.append(rec["file_url"])

    if not urls:
        return rec, "error: no_url", None, ""

    fallback_ext = (rec.get("file_ext") or "jpg").lower()
    sess = get_session()
    last_err = ""

    for url in urls:
        ext = url_ext(url, fallback_ext)
        dest = images_dir / f"{pid}.{ext}"
        tmp = images_dir / f"{pid}.{ext}.part"

        for attempt in range(1, 6):
            try:
                with sess.get(url, timeout=timeout, stream=True) as resp:
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        wait = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else min(30.0, 2.0 * (2 ** (attempt - 1)))
                        )
                        time.sleep(wait)
                        last_err = "HTTP 429"
                        continue
                    if resp.status_code in (403, 404, 410):
                        last_err = f"HTTP {resp.status_code}"
                        break
                    if resp.status_code >= 500:
                        last_err = f"HTTP {resp.status_code}"
                        time.sleep(min(20.0, 1.5 * attempt))
                        continue
                    resp.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                if tmp.exists() and tmp.stat().st_size > 0:
                    os.replace(tmp, dest)
                    return rec, "downloaded", str(dest), url
                last_err = "empty response body"
            except Exception as exc:
                last_err = str(exc)
                time.sleep(min(20.0, 1.5 * attempt))
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    return rec, f"error: {last_err}", None, ""


def compute_hashes(path_str: str) -> tuple[str | None, str | None]:
    try:
        with Image.open(path_str) as im:
            rgb = im.convert("RGB")
            ph = str(imagehash.phash(rgb))
            wh = str(imagehash.whash(rgb))
            return ph, wh
    except Exception:
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download static Danbooru HBR images and build manifest.")
    parser.add_argument("--meta", default=str(PROJECT_ROOT / "data" / "images" / "danbooru_meta.jsonl"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "images" / "danbooru"))
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_manifest.jsonl"),
        help="Output manifest JSONL path",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()

    meta_path = Path(args.meta)
    images_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load & filter danbooru_meta.jsonl to static downloadable images
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec["id"]
            if pid in seen_ids:
                continue
            if rec.get("has_file_url") is not True:
                continue
            if (rec.get("file_ext") or "").lower() not in IMAGE_EXTS:
                continue
            seen_ids.add(pid)
            records.append(rec)

    print(f"Filtered static downloadable records: {len(records)}", flush=True)

    # 2. Scan existing local folders for fast reuse (strictly read-only)
    reuse_dirs = [
        PROJECT_ROOT / "data" / "test_sets" / "hard_testset",
    ]
    existing_map = scan_existing_dirs(reuse_dirs)
    reusable_count = sum(1 for r in records if r["id"] in existing_map)
    print(
        f"Indexed {len(existing_map)} existing local images ({reusable_count} match filtered records).",
        flush=True,
    )

    # 3. Reuse or download in parallel
    outcomes: list[tuple[dict[str, Any], str, str | None, str]] = []
    status_counts: collections.Counter[str] = collections.Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_or_reuse, rec, images_dir, existing_map) for rec in records]
        pbar = tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="Downloading/Reusing",
            mininterval=2.0,
        )
        for fut in pbar:
            res = fut.result()
            outcomes.append(res)
            st = res[1] if not res[1].startswith("error:") else "failed"
            status_counts[st] += 1

    print(f"Download/Reuse summary: {dict(status_counts)}", flush=True)

    failed_items = [(rec["id"], status) for rec, status, lp, _ in outcomes if lp is None]
    if failed_items:
        print(f"Failed count: {len(failed_items)} (showing up to 20):", flush=True)
        for pid, st in failed_items[:20]:
            print(f"  id={pid}: {st}", flush=True)

    ok_outcomes = [o for o in outcomes if o[2] is not None]

    # 4. Compute pHash and wHash in parallel
    print(f"Computing pHash and wHash for {len(ok_outcomes)} images...", flush=True)
    paths = [o[2] for o in ok_outcomes]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.hash_workers) as ex:
        hashes = list(
            tqdm(
                ex.map(compute_hashes, paths),
                total=len(paths),
                desc="Computing pHash/wHash",
                mininterval=2.0,
            )
        )

    hash_failed = sum(1 for ph, wh in hashes if ph is None or wh is None)
    if hash_failed:
        print(f"WARNING: {hash_failed} images failed pHash/wHash computation.", flush=True)

    # 5. Build and write manifest.jsonl sorted deterministically by post id
    manifest_rows: list[dict[str, Any]] = []
    for (rec, status, local_path, _dl_url), (ph, wh) in zip(ok_outcomes, hashes):
        raw_chars = rec.get("characters") or []
        base_chars = collapse_costume_subtags(raw_chars)
        gen_tags = set((rec.get("tag_string_general") or "").split())
        row = {
            "id": rec["id"],
            "local_path": local_path,
            "phash": ph,
            "whash": wh,
            "base_chars": base_chars,
            "num_base_chars": len(base_chars),
            "is_comic": bool(rec.get("is_comic", False)),
            "is_cosplay": "cosplay" in gen_tags,
            "copyrights": rec.get("copyrights") or [],
            "artists": rec.get("artists") or [],
            "rating": rec.get("rating") or "",
            "score": rec.get("score", 0),
            "width": rec.get("width", 0),
            "height": rec.get("height", 0),
            "file_ext": (rec.get("file_ext") or "").lower(),
        }
        manifest_rows.append(row)

    manifest_rows.sort(key=lambda x: x["id"])
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. Compute total disk size of images_dir + manifest
    total_bytes = 0
    for fp in images_dir.iterdir():
        if fp.is_file():
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                pass
    manifest_bytes = manifest_path.stat().st_size if manifest_path.exists() else 0

    print("\n=== FINAL REPORT ===", flush=True)
    print(f"Total target records: {len(records)}", flush=True)
    print(f"Total images saved:   {len(manifest_rows)}", flush=True)
    print(f"  reused:             {status_counts.get('reused', 0)}", flush=True)
    print(f"  exists (resumed):   {status_counts.get('exists', 0)}", flush=True)
    print(f"  newly downloaded:   {status_counts.get('downloaded', 0)}", flush=True)
    print(f"  failed:             {len(failed_items)}", flush=True)
    print(f"  hash_failed:        {hash_failed}", flush=True)
    print(
        f"Disk size (images):   {total_bytes / (1024 ** 3):.3f} GB ({total_bytes / (1024 ** 2):.1f} MB)",
        flush=True,
    )
    print(f"Manifest size:        {manifest_bytes / (1024 ** 2):.2f} MB", flush=True)
    print(f"Manifest path:        {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
