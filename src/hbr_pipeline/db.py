"""SQLite database layer & CLI for the Master HBR Dataset (`data/db/master_hbr.db`).

Uses Python standard library (`sqlite3`) only — zero external dependencies.
Supports:
  - WAL mode, foreign keys ON, and indexes for fast querying
  - Cached 64-bit pHash Hamming duplicate detection over non-duplicate images
  - Atomic upsert of image records + box replacement (preserving `captions`)
  - Bulk JSONL import (`--import-jsonl`) and summary statistics (`--stats`)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS images (
    uid TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    has_metadata INTEGER NOT NULL,
    danbooru_id INTEGER,
    local_path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    phash TEXT,
    whash TEXT,
    is_duplicate INTEGER DEFAULT 0,
    duplicate_of TEXT,
    status TEXT NOT NULL,
    is_comic INTEGER DEFAULT 0,
    is_cosplay INTEGER DEFAULT 0,
    rating TEXT,
    score INTEGER DEFAULT 0,
    copyrights_json TEXT,
    artists_json TEXT,
    meta_base_chars_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS boxes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_uid TEXT NOT NULL REFERENCES images(uid) ON DELETE CASCADE,
    box_idx INTEGER NOT NULL,
    character TEXT,
    tier TEXT NOT NULL,
    free_pred TEXT,
    free_conf REAL,
    match_conf REAL,
    ext_conf REAL,
    mode TEXT,
    area_pct REAL,
    person_bbox_json TEXT,
    head_bbox_json TEXT,
    extra_json TEXT
);

CREATE TABLE IF NOT EXISTS captions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_uid TEXT NOT NULL REFERENCES images(uid) ON DELETE CASCADE,
    model TEXT NOT NULL,
    prompt_ver TEXT NOT NULL,
    caption TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_uid, model, prompt_ver)
);

CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_danbooru_id ON images(danbooru_id);
CREATE INDEX IF NOT EXISTS idx_images_is_duplicate ON images(is_duplicate);

CREATE INDEX IF NOT EXISTS idx_boxes_image_uid ON boxes(image_uid);
CREATE INDEX IF NOT EXISTS idx_boxes_character ON boxes(character);
CREATE INDEX IF NOT EXISTS idx_boxes_tier ON boxes(tier);

CREATE INDEX IF NOT EXISTS idx_captions_image_uid ON captions(image_uid);
"""

_KNOWN_BOX_KEYS = frozenset(
    {
        "box_idx",
        "character",
        "tier",
        "free_pred",
        "free_conf",
        "match_conf",
        "ext_conf",
        "mode",
        "area_pct",
        "person_bbox",
        "head_bbox",
        "person_bbox_json",
        "head_bbox_json",
        "bbox",
        "extra_json",
    }
)


@dataclass
class _PHashCacheEntry:
    db_file: str
    synced_changes: int
    uid_to_hash: dict[str, int]


_PHASH_CACHE: dict[int, _PHashCacheEntry] = {}


def _get_db_file(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list;").fetchone()
    if row is None:
        return ""
    return str(row[2] if not isinstance(row, sqlite3.Row) else row["file"])


def invalidate_phash_cache(conn: sqlite3.Connection | None = None) -> None:
    """Clear the in-memory pHash cache for a connection (or all connections)."""
    if conn is None:
        _PHASH_CACHE.clear()
    else:
        _PHASH_CACHE.pop(id(conn), None)


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, foreign keys ON, and Row factory."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize the Master HBR SQLite database schema and return an open connection."""
    conn = connect_db(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    invalidate_phash_cache(conn)
    return conn


def _parse_phash_int(phash_hex: str | None) -> int | None:
    if not phash_hex:
        return None
    s = str(phash_hex).strip()
    if not s:
        return None
    try:
        return int(s, 16)
    except ValueError:
        return None


def _get_or_sync_phash_cache(conn: sqlite3.Connection) -> _PHashCacheEntry:
    key = id(conn)
    db_file = _get_db_file(conn)
    changes = conn.total_changes
    entry = _PHASH_CACHE.get(key)
    if entry is not None and entry.db_file == db_file and entry.synced_changes == changes:
        return entry

    uid_to_hash: dict[str, int] = {}
    cursor = conn.execute(
        "SELECT uid, phash FROM images WHERE is_duplicate = 0 AND phash IS NOT NULL AND phash != ''"
    )
    for row in cursor:
        uid = row[0]
        h_int = _parse_phash_int(row[1])
        if h_int is not None:
            uid_to_hash[uid] = h_int

    entry = _PHashCacheEntry(db_file=db_file, synced_changes=changes, uid_to_hash=uid_to_hash)
    _PHASH_CACHE[key] = entry
    return entry


def find_phash_duplicate(
    conn: sqlite3.Connection,
    phash_hex: str | None,
    max_hamming: int = 4,
    exclude_uid: str | None = None,
) -> str | None:
    """Return the UID of an existing non-duplicate image within `max_hamming` bits, or None.

    Uses a per-connection in-memory 64-bit integer cache over `images.phash WHERE is_duplicate=0`,
    automatically kept in sync with `upsert_image_record` and invalidated on external mutations.
    """
    target = _parse_phash_int(phash_hex)
    if target is None:
        return None

    entry = _get_or_sync_phash_cache(conn)
    for existing_uid, existing_h in entry.uid_to_hash.items():
        if exclude_uid is not None and existing_uid == exclude_uid:
            continue
        if (target ^ existing_h).bit_count() <= max_hamming:
            return existing_uid
    return None


def _to_json_or_none(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return json.dumps(val, ensure_ascii=False)


def _normalize_box_row(
    image_uid: str, box_idx: int, box: dict[str, Any], is_crowd: bool = False
) -> tuple[Any, ...]:
    if is_crowd:
        person_bbox = box.get("person_bbox") if box.get("person_bbox") is not None else box.get("bbox")
        head_bbox = box.get("head_bbox")
        tier = str(box.get("tier") or "CROWD_MASK")
        mode = str(box.get("mode") or "crowd")
        character = box.get("character")
    else:
        person_bbox = box.get("person_bbox")
        head_bbox = box.get("head_bbox")
        tier = str(box.get("tier") or "REVIEW")
        mode = box.get("mode")
        character = box.get("character")

    person_bbox_json = (
        box["person_bbox_json"] if "person_bbox_json" in box else _to_json_or_none(person_bbox)
    )
    head_bbox_json = (
        box["head_bbox_json"] if "head_bbox_json" in box else _to_json_or_none(head_bbox)
    )

    if "extra_json" in box and box["extra_json"] is not None:
        extra_json = _to_json_or_none(box["extra_json"])
    else:
        extras = {k: v for k, v in box.items() if k not in _KNOWN_BOX_KEYS and v is not None}
        extra_json = json.dumps(extras, ensure_ascii=False) if extras else None

    return (
        image_uid,
        box_idx,
        character,
        tier,
        box.get("free_pred"),
        box.get("free_conf"),
        box.get("match_conf"),
        box.get("ext_conf"),
        mode,
        box.get("area_pct"),
        person_bbox_json,
        head_bbox_json,
        extra_json,
    )


def upsert_image_record(conn: sqlite3.Connection, rec: dict[str, Any]) -> None:
    """Atomically upsert an `images` row and replace its `boxes` rows.

    Handles both `characters` and `crowd_mask_boxes` lists from JSONL or live ingestion.
    Uses `ON CONFLICT(uid) DO UPDATE` so existing `captions` rows and `created_at` are preserved.
    """
    uid = str(rec["uid"])
    source = str(rec["source"])
    has_metadata = int(bool(rec.get("has_metadata", False)))
    danbooru_id = rec.get("danbooru_id")
    if danbooru_id is None and source == "danbooru" and uid.startswith("danbooru:"):
        try:
            danbooru_id = int(uid.split(":", 1)[1])
        except ValueError:
            danbooru_id = None

    local_path = str(rec["local_path"])
    width = rec.get("width")
    height = rec.get("height")
    phash = rec.get("phash")
    whash = rec.get("whash")
    is_duplicate = int(bool(rec.get("is_duplicate", False)))
    duplicate_of = rec.get("duplicate_of")
    status = str(rec["status"])
    is_comic = int(bool(rec.get("is_comic", False)))
    is_cosplay = int(bool(rec.get("is_cosplay", False)))
    rating = rec.get("rating")
    score = int(rec.get("score") or 0)

    copyrights_json = (
        rec["copyrights_json"]
        if "copyrights_json" in rec
        else _to_json_or_none(rec.get("copyrights"))
    )
    artists_json = (
        rec["artists_json"]
        if "artists_json" in rec
        else _to_json_or_none(rec.get("artists"))
    )
    meta_base_chars_json = (
        rec["meta_base_chars_json"]
        if "meta_base_chars_json" in rec
        else _to_json_or_none(rec.get("meta_base_chars"))
    )

    box_rows: list[tuple[Any, ...]] = []
    next_idx = 0
    for ch_box in rec.get("characters") or []:
        box_rows.append(_normalize_box_row(uid, next_idx, ch_box, is_crowd=False))
        next_idx += 1
    for cr_box in rec.get("crowd_mask_boxes") or []:
        box_rows.append(_normalize_box_row(uid, next_idx, cr_box, is_crowd=True))
        next_idx += 1

    cache_key = id(conn)
    cache_entry = _PHASH_CACHE.get(cache_key)
    cache_was_synced = (
        cache_entry is not None
        and cache_entry.synced_changes == conn.total_changes
        and cache_entry.db_file == _get_db_file(conn)
    )

    def _execute_upsert() -> None:
        conn.execute(
            """
            INSERT INTO images (
                uid, source, has_metadata, danbooru_id, local_path,
                width, height, phash, whash,
                is_duplicate, duplicate_of, status,
                is_comic, is_cosplay, rating, score,
                copyrights_json, artists_json, meta_base_chars_json
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(uid) DO UPDATE SET
                source = excluded.source,
                has_metadata = excluded.has_metadata,
                danbooru_id = excluded.danbooru_id,
                local_path = excluded.local_path,
                width = excluded.width,
                height = excluded.height,
                phash = excluded.phash,
                whash = excluded.whash,
                is_duplicate = excluded.is_duplicate,
                duplicate_of = excluded.duplicate_of,
                status = excluded.status,
                is_comic = excluded.is_comic,
                is_cosplay = excluded.is_cosplay,
                rating = excluded.rating,
                score = excluded.score,
                copyrights_json = excluded.copyrights_json,
                artists_json = excluded.artists_json,
                meta_base_chars_json = excluded.meta_base_chars_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                uid,
                source,
                has_metadata,
                danbooru_id,
                local_path,
                width,
                height,
                phash,
                whash,
                is_duplicate,
                duplicate_of,
                status,
                is_comic,
                is_cosplay,
                rating,
                score,
                copyrights_json,
                artists_json,
                meta_base_chars_json,
            ),
        )
        conn.execute("DELETE FROM boxes WHERE image_uid = ?", (uid,))
        if box_rows:
            conn.executemany(
                """
                INSERT INTO boxes (
                    image_uid, box_idx, character, tier,
                    free_pred, free_conf, match_conf, ext_conf,
                    mode, area_pct, person_bbox_json, head_bbox_json, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                box_rows,
            )

    if conn.in_transaction:
        conn.execute("SAVEPOINT sp_upsert_image")
        try:
            _execute_upsert()
        except Exception:
            conn.execute("ROLLBACK TO SAVEPOINT sp_upsert_image")
            conn.execute("RELEASE SAVEPOINT sp_upsert_image")
            raise
        else:
            conn.execute("RELEASE SAVEPOINT sp_upsert_image")
    else:
        with conn:
            _execute_upsert()

    if cache_was_synced and cache_entry is not None:
        h_int = _parse_phash_int(phash)
        if is_duplicate == 0 and h_int is not None:
            cache_entry.uid_to_hash[uid] = h_int
        else:
            cache_entry.uid_to_hash.pop(uid, None)
        cache_entry.synced_changes = conn.total_changes


def import_jsonl(
    db_path: str | Path,
    jsonl_paths: Iterable[Path | str],
    batch_size: int = 1000,
) -> dict[str, int]:
    """Import one or more Master DB JSONL files into SQLite in batched transactions."""
    paths = [Path(p) for p in jsonl_paths]
    conn = init_db(db_path)
    imported = 0
    errors = 0
    files_done = 0
    t0 = time.time()

    try:
        for jp in paths:
            if not jp.exists():
                raise FileNotFoundError(f"JSONL file not found: {jp}")
            file_count = 0
            conn.execute("BEGIN")
            try:
                with jp.open("r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            rec = json.loads(s)
                            upsert_image_record(conn, rec)
                            imported += 1
                            file_count += 1
                        except Exception as exc:
                            errors += 1
                            print(f"[WARN] {jp.name}:{lineno} skipped: {exc}")
                            continue
                        if file_count % batch_size == 0:
                            conn.commit()
                            conn.execute("BEGIN")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            files_done += 1
            print(f"Imported {file_count:,} records from {jp}")
    finally:
        conn.close()

    elapsed = time.time() - t0
    print(f"Done: {imported:,} records imported from {files_done} file(s) in {elapsed:.2f}s (errors={errors})")
    return {"files": files_done, "imported": imported, "errors": errors}


def print_stats(db_path: str | Path, top_n: int = 20) -> None:
    """Print summary statistics for the Master HBR SQLite database."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    conn = connect_db(path)
    try:
        total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        unique_images = conn.execute("SELECT COUNT(*) FROM images WHERE is_duplicate = 0").fetchone()[0]
        dup_images = conn.execute("SELECT COUNT(*) FROM images WHERE is_duplicate = 1").fetchone()[0]
        total_boxes = conn.execute("SELECT COUNT(*) FROM boxes").fetchone()[0]
        total_captions = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]

        print(f"=== Master HBR SQLite Stats: {path} ===")
        print(
            f"Images : {total_images:,} total | {unique_images:,} unique | {dup_images:,} duplicates"
        )
        print(f"Boxes  : {total_boxes:,} | Captions: {total_captions:,}")

        print("\n[By Source]")
        for row in conn.execute(
            """
            SELECT source,
                   COUNT(*) AS total,
                   SUM(CASE WHEN is_duplicate = 0 THEN 1 ELSE 0 END) AS uniq,
                   SUM(CASE WHEN is_duplicate = 1 THEN 1 ELSE 0 END) AS dups
            FROM images
            GROUP BY source
            ORDER BY total DESC
            """
        ):
            print(f"  {row['source']:<12} : {row['total']:>7,} (unique={row['uniq']:,}, dup={row['dups']:,})")

        print("\n[By Image Status]")
        for row in conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM images GROUP BY status ORDER BY cnt DESC"
        ):
            print(f"  {row['status']:<12} : {row['cnt']:>7,}")

        print("\n[By Box Tier]")
        for row in conn.execute(
            "SELECT tier, COUNT(*) AS cnt FROM boxes GROUP BY tier ORDER BY cnt DESC"
        ):
            print(f"  {row['tier']:<22} : {row['cnt']:>7,}")

        print(f"\n[Top {top_n} Characters (excl. CROWD_MASK)]")
        for row in conn.execute(
            """
            SELECT character,
                   COUNT(*) AS total,
                   SUM(CASE WHEN tier = 'ACCEPT' THEN 1 ELSE 0 END) AS accepted
            FROM boxes
            WHERE character IS NOT NULL AND character != '' AND tier != 'CROWD_MASK'
            GROUP BY character
            ORDER BY total DESC, character ASC
            LIMIT ?
            """,
            (top_n,),
        ):
            print(f"  {row['character']:<36} : {row['total']:>6,} (ACCEPT={row['accepted']:,})")

        if dup_images > 0:
            print("\n[Duplicate Summary]")
            for row in conn.execute(
                """
                SELECT source, COUNT(*) AS cnt
                FROM images
                WHERE is_duplicate = 1
                GROUP BY source
                ORDER BY cnt DESC
                """
            ):
                print(f"  {row['source']:<12} : {row['cnt']:>7,} duplicates")
    finally:
        conn.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_db = project_root / "data" / "db" / "master_hbr.db"

    parser = argparse.ArgumentParser(description="Master HBR SQLite DB manager & CLI.")
    parser.add_argument("--db", default=str(default_db), help="Path to master_hbr.db")
    parser.add_argument("--init", action="store_true", help="Initialize database schema")
    parser.add_argument(
        "--import-jsonl",
        nargs="+",
        default=None,
        help="One or more JSONL files to import into SQLite",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Commit batch size for JSONL import")
    parser.add_argument("--stats", action="store_true", help="Print database statistics")
    parser.add_argument("--top-chars", type=int, default=20, help="Number of top characters to display in --stats")
    args = parser.parse_args()

    if not (args.init or args.import_jsonl or args.stats):
        parser.error("Specify at least one action: --init, --import-jsonl, or --stats")

    db_path = Path(args.db)
    if args.init:
        conn = init_db(db_path)
        conn.close()
        print(f"Initialized SQLite DB: {db_path}")

    if args.import_jsonl:
        import_jsonl(db_path, [Path(p) for p in args.import_jsonl], batch_size=args.batch_size)

    if args.stats:
        print_stats(db_path, top_n=args.top_chars)


if __name__ == "__main__":
    main()
