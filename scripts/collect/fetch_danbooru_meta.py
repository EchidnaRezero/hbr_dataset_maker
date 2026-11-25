"""Fetch Danbooru post metadata for a tag using official JSON API cursor pagination."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from tqdm import tqdm

API_URL = "https://danbooru.donmai.us/posts.json"
DEFAULT_UA = "HBRDatasetResearch/0.1 (Python requests)"
COMIC_TAGS = {"comic", "4koma", "speech_bubble", "manga", "2koma", "3koma"}


def fetch_page(
    session: requests.Session,
    tag: str,
    before_id: int | None,
    limit: int = 200,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "tags": tag,
        "limit": limit,
    }
    if before_id is not None:
        params["page"] = f"b{before_id}"

    for attempt in range(1, 6):
        try:
            resp = session.get(API_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = min(30, 2**attempt)
                print(f"[429] Rate limited. Sleeping {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected API response: {data}")
            return data
        except requests.RequestException as exc:
            if attempt == 5:
                raise
            wait = min(20, 2**attempt)
            print(f"[Attempt {attempt}] Request error: {exc}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return []


def normalize_post(post: dict[str, Any]) -> dict[str, Any]:
    chars = post.get("tag_string_character", "").split()
    general = post.get("tag_string_general", "").split()
    artists = post.get("tag_string_artist", "").split()
    copyrights = post.get("tag_string_copyright", "").split()

    general_set = set(general)
    is_comic = bool(general_set & COMIC_TAGS)

    return {
        "id": post["id"],
        "created_at": post.get("created_at"),
        "score": post.get("score", 0),
        "rating": post.get("rating"),
        "width": post.get("image_width"),
        "height": post.get("image_height"),
        "file_ext": post.get("file_ext"),
        "file_url": post.get("file_url"),
        "large_file_url": post.get("large_file_url"),
        "preview_file_url": post.get("preview_file_url"),
        "characters": chars,
        "num_characters": len(chars),
        "artists": artists,
        "copyrights": copyrights,
        "is_comic": is_comic,
        "tag_string_general": post.get("tag_string_general", ""),
        "tag_string_meta": post.get("tag_string_meta", ""),
        "has_file_url": bool(post.get("file_url") or post.get("large_file_url")),
    }


def print_summary(records: list[dict[str, Any]]) -> None:
    total = len(records)
    with_url = sum(1 for r in records if r["has_file_url"])
    no_char = sum(1 for r in records if r["num_characters"] == 0)
    solo = [r for r in records if r["num_characters"] == 1]
    multi = [r for r in records if r["num_characters"] >= 2]
    comics = sum(1 for r in records if r["is_comic"])
    ext_counts = collections.Counter(r["file_ext"] for r in records)

    solo_char_counts = collections.Counter(r["characters"][0] for r in solo)
    all_char_counts = collections.Counter(c for r in records for c in r["characters"])

    print("\n================ Danbooru HBR Metadata Summary ================")
    print(f"Total posts fetched : {total:,}")
    print(f"With downloadable URL: {with_url:,} (missing/gold-gated: {total - with_url:,})")
    print(f"  - Solo (1 char)   : {len(solo):,}")
    print(f"  - Multi (2+ chars): {len(multi):,}")
    print(f"  - No char tag (0) : {no_char:,}")
    print(f"  - Comic/4koma/etc : {comics:,}")
    print(f"File extensions     : {dict(ext_counts.most_common(6))}")
    print(f"Unique character tags (all) : {len(all_char_counts):,}")
    print(f"Unique character tags (solo): {len(solo_char_counts):,}")
    print(f"Characters with >=10 solo   : {sum(1 for _, c in solo_char_counts.items() if c >= 10)}")
    print(f"Characters with >=30 solo   : {sum(1 for _, c in solo_char_counts.items() if c >= 30)}")

    print("\nTop 20 characters (solo count / total count):")
    for ch, cnt in solo_char_counts.most_common(20):
        print(f"  {ch:<40s} solo={cnt:4d}  total={all_char_counts[ch]:4d}")

    print("\nCharacters with 1~9 solo images (sample 15):")
    low = [(ch, cnt) for ch, cnt in solo_char_counts.items() if cnt < 10]
    low.sort(key=lambda x: (-x[1], x[0]))
    for ch, cnt in low[:15]:
        print(f"  {ch:<40s} solo={cnt:4d}  total={all_char_counts[ch]:4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Danbooru metadata for a tag.")
    parser.add_argument("--tag", default="heaven_burns_red", help="Danbooru tag to query")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "data" / "images" / "danbooru_meta.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API requests (seconds)")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages to fetch (0 = all)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = DEFAULT_UA
    login = os.environ.get("DANBOORU_LOGIN")
    api_key = os.environ.get("DANBOORU_API_KEY")
    if login and api_key:
        session.auth = (login, api_key)
        print(f"Using Danbooru API auth as '{login}'.")

    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    before_id: int | None = None
    page_num = 0

    pbar = tqdm(desc=f"Fetching '{args.tag}'", unit=" posts")
    try:
        while True:
            page_num += 1
            posts = fetch_page(session, args.tag, before_id=before_id, limit=200)
            if not posts:
                break

            new_count = 0
            for post in posts:
                pid = post.get("id")
                if pid is None or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                records.append(normalize_post(post))
                new_count += 1

            pbar.update(new_count)
            before_id = min(p["id"] for p in posts if "id" in p)

            if len(posts) < 200:
                break
            if args.max_pages and page_num >= args.max_pages:
                break
            time.sleep(args.delay)
    finally:
        pbar.close()

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records):,} records to: {out_path}")
    print_summary(records)


if __name__ == "__main__":
    main()
