"""Export HBR master dataset to sharded Parquet and upload model & dataset repos to Hugging Face.

SAFETY:
- Reads HF token from `HF_TOKEN` env var (or cached `huggingface-cli login`) without printing or logging it.
- Creates each repo and uploads README.md with `gated: manual` FIRST, then verifies
  `gated == "manual"` via HfApi BEFORE uploading any model weights, Parquet shards, or DB files.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token

BASE_DIR = PROJECT_ROOT
DB_PATH = BASE_DIR / "data" / "db" / "master_hbr.db"
EXPORT_DIR = BASE_DIR / "outputs" / "hf_export"

MODEL_REPO_ID = "Kamome33/hbr-character-classifier"
DATASET_REPO_ID = "Kamome33/hbr-dataset-master"

MODEL_FILES = [
    (BASE_DIR / "models" / "v3_pure_hbr55_plus_multi_camie.npz", "v3_pure_hbr55_plus_multi_camie.npz"),
    (BASE_DIR / "models" / "archive" / "v2_pure_hbr55_camie.npz", "v2_pure_hbr55_camie.npz"),
    (BASE_DIR / "models" / "archive" / "v1_sample1053_camie_hbr54.npz", "v1_sample1053_camie_hbr54.npz"),
    (BASE_DIR / "data" / "embeddings" / "v3_train_report.json", "v3_train_report.json"),
]

MODEL_README = """---
gated: manual
tags:
  - image-classification
  - anime
  - heaven-burns-red
---

# HBR Character Classifier

Heaven Burns Red character classifier weights (Camie-based embeddings) and training report.

## Files
- `v3_pure_hbr55_plus_multi_camie.npz`
- `v2_pure_hbr55_camie.npz`
- `v1_sample1053_camie_hbr54.npz`
- `v3_train_report.json`
"""

DATASET_README = """---
gated: manual
configs:
  - config_name: default
    data_files:
      - split: danbooru
        path: data/danbooru-*.parquet
      - split: official_neo
        path: data/official_neo-*.parquet
tags:
  - image-classification
  - anime
  - heaven-burns-red
---

# HBR Dataset Master

Master dataset for Heaven Burns Red character classification, including sharded Parquet files with embedded images + bounding box metadata (`boxes_json`) and the SQLite master database (`master_hbr.db`).

## Splits
- `danbooru`: 10,831 images (`data/danbooru-00000-of-00005.parquet` .. `00004`)
- `official_neo`: 72 images (`data/official_neo-00000-of-00001.parquet`)
"""

IMAGE_COLUMNS = [
    "uid",
    "source",
    "has_metadata",
    "danbooru_id",
    "local_path",
    "width",
    "height",
    "phash",
    "whash",
    "is_duplicate",
    "duplicate_of",
    "status",
    "is_comic",
    "is_cosplay",
    "rating",
    "score",
    "copyrights_json",
    "artists_json",
    "meta_base_chars_json",
    "created_at",
    "updated_at",
]


def load_hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or get_token() or "").strip()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Set the HF_TOKEN environment variable or run `huggingface-cli login`."
        )
    return token


def build_arrow_schema() -> pa.Schema:
    fields = [
        pa.field("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        pa.field("uid", pa.string()),
        pa.field("source", pa.string()),
        pa.field("has_metadata", pa.int64()),
        pa.field("danbooru_id", pa.int64()),
        pa.field("local_path", pa.string()),
        pa.field("width", pa.int64()),
        pa.field("height", pa.int64()),
        pa.field("phash", pa.string()),
        pa.field("whash", pa.string()),
        pa.field("is_duplicate", pa.int64()),
        pa.field("duplicate_of", pa.string()),
        pa.field("status", pa.string()),
        pa.field("is_comic", pa.int64()),
        pa.field("is_cosplay", pa.int64()),
        pa.field("rating", pa.string()),
        pa.field("score", pa.int64()),
        pa.field("copyrights_json", pa.string()),
        pa.field("artists_json", pa.string()),
        pa.field("meta_base_chars_json", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string()),
        pa.field("boxes_json", pa.string()),
    ]
    hf_features = {
        "image": {"_type": "Image"},
        "uid": {"dtype": "string", "_type": "Value"},
        "source": {"dtype": "string", "_type": "Value"},
        "has_metadata": {"dtype": "int64", "_type": "Value"},
        "danbooru_id": {"dtype": "int64", "_type": "Value"},
        "local_path": {"dtype": "string", "_type": "Value"},
        "width": {"dtype": "int64", "_type": "Value"},
        "height": {"dtype": "int64", "_type": "Value"},
        "phash": {"dtype": "string", "_type": "Value"},
        "whash": {"dtype": "string", "_type": "Value"},
        "is_duplicate": {"dtype": "int64", "_type": "Value"},
        "duplicate_of": {"dtype": "string", "_type": "Value"},
        "status": {"dtype": "string", "_type": "Value"},
        "is_comic": {"dtype": "int64", "_type": "Value"},
        "is_cosplay": {"dtype": "int64", "_type": "Value"},
        "rating": {"dtype": "string", "_type": "Value"},
        "score": {"dtype": "int64", "_type": "Value"},
        "copyrights_json": {"dtype": "string", "_type": "Value"},
        "artists_json": {"dtype": "string", "_type": "Value"},
        "meta_base_chars_json": {"dtype": "string", "_type": "Value"},
        "created_at": {"dtype": "string", "_type": "Value"},
        "updated_at": {"dtype": "string", "_type": "Value"},
        "boxes_json": {"dtype": "string", "_type": "Value"},
    }
    hf_meta = {"info": {"features": hf_features}}
    return pa.schema(fields, metadata={b"huggingface": json.dumps(hf_meta).encode("utf-8")})


def create_and_gate_repo(api: HfApi, repo_id: str, repo_type: str, readme_content: str) -> None:
    print(f"[{repo_type}] Creating repo {repo_id} (private=False, exist_ok=True)...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False, exist_ok=True)

    print(f"[{repo_type}] Uploading gated README.md FIRST to {repo_id}...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Initialize gated repo README (gated: manual)",
    )

    print(f"[{repo_type}] Enabling manual gating via update_repo_settings for {repo_id}...")
    api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, gated="manual")

    gated_val = None
    for _attempt in range(5):
        if repo_type == "model":
            info = api.model_info(repo_id)
        else:
            info = api.dataset_info(repo_id)
        gated_val = info.gated
        if gated_val == "manual":
            break
        time.sleep(1.5)

    print(f"[{repo_type}] Verified {repo_id} gated={gated_val!r}")
    if gated_val != "manual":
        raise RuntimeError(
            f"SAFETY ABORT: {repo_id} ({repo_type}) gated status is {gated_val!r}, expected 'manual'!"
        )


def load_boxes_by_uid(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    boxes_by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in cur.execute("SELECT * FROM boxes ORDER BY image_uid, box_idx, id"):
        d = dict(row)
        uid = d["image_uid"]
        boxes_by_uid.setdefault(uid, []).append(d)
    return boxes_by_uid


def export_split_to_parquet(
    conn: sqlite3.Connection,
    boxes_by_uid: dict[str, list[dict[str, Any]]],
    source: str,
    split_name: str,
    num_shards: int,
    schema: pa.Schema,
) -> list[Path]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        f"SELECT {', '.join(IMAGE_COLUMNS)} FROM images WHERE source = ? ORDER BY uid",
        (source,),
    ).fetchall()
    total = len(rows)
    print(f"[export] Split '{split_name}' (source={source}): {total} rows -> {num_shards} shard(s)")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    shard_size = math.ceil(total / num_shards)
    out_paths: list[Path] = []

    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        shard_rows = rows[start:end]
        out_name = f"{split_name}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        out_path = EXPORT_DIR / out_name

        col_data: dict[str, list[Any]] = {"image": [], "boxes_json": []}
        for col in IMAGE_COLUMNS:
            col_data[col] = []

        for r in shard_rows:
            local_path = r["local_path"]
            img_bytes = Path(local_path).read_bytes()
            filename = os.path.basename(local_path)
            col_data["image"].append({"bytes": img_bytes, "path": filename})
            for col in IMAGE_COLUMNS:
                col_data[col].append(r[col])
            uid = r["uid"]
            col_data["boxes_json"].append(
                json.dumps(boxes_by_uid.get(uid, []), ensure_ascii=False)
            )

        table = pa.Table.from_pydict(col_data, schema=schema)
        pq.write_table(table, out_path, compression="snappy", row_group_size=100)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  wrote {out_name}: {len(shard_rows)} rows, {size_mb:.2f} MiB")
        out_paths.append(out_path)

    return out_paths


def verify_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    if repo_type == "model":
        info = api.model_info(repo_id, files_metadata=True)
    else:
        info = api.dataset_info(repo_id, files_metadata=True)
    print(f"\n=== VERIFY {repo_type.upper()}: {repo_id} ===")
    print(f"gated: {info.gated!r}, private: {info.private!r}")
    if info.gated != "manual":
        raise RuntimeError(f"Verification failed: {repo_id} gated={info.gated!r}")
    for sib in sorted(info.siblings or [], key=lambda s: s.rfilename):
        size = getattr(sib, "size", None)
        size_str = f"{size:,} bytes ({size / (1024 * 1024):.2f} MiB)" if size is not None else "unknown"
        print(f"  - {sib.rfilename}: {size_str}")


def main() -> None:
    token = load_hf_token()
    api = HfApi(token=token)

    # 1. CRITICAL SAFETY ORDER: Create and gate BOTH repos BEFORE any data/weight upload
    create_and_gate_repo(api, MODEL_REPO_ID, "model", MODEL_README)
    create_and_gate_repo(api, DATASET_REPO_ID, "dataset", DATASET_README)

    # 2. Upload model files to Repo 1
    print(f"\n[model] Uploading {len(MODEL_FILES)} files to {MODEL_REPO_ID}...")
    for local_path, path_in_repo in MODEL_FILES:
        if not local_path.exists():
            raise FileNotFoundError(f"Missing model/report file: {local_path}")
        print(f"  uploading {path_in_repo} ({local_path.stat().st_size:,} bytes)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message=f"Upload {path_in_repo}",
        )

    # 3. Build sharded Parquet files from master_hbr.db (read-only)
    print(f"\n[export] Opening {DB_PATH} (read-only)...")
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        boxes_by_uid = load_boxes_by_uid(conn)
        print(f"[export] Loaded boxes for {len(boxes_by_uid)} images.")
        schema = build_arrow_schema()
        danbooru_shards = export_split_to_parquet(
            conn, boxes_by_uid, source="danbooru", split_name="danbooru", num_shards=5, schema=schema
        )
        official_shards = export_split_to_parquet(
            conn, boxes_by_uid, source="official", split_name="official_neo", num_shards=1, schema=schema
        )
    finally:
        conn.close()

    # 4. Upload Parquet shards under data/ and master_hbr.db at root
    all_shards = danbooru_shards + official_shards
    print(f"\n[dataset] Uploading {len(all_shards)} Parquet shards to {DATASET_REPO_ID}/data/ ...")
    for shard_path in all_shards:
        path_in_repo = f"data/{shard_path.name}"
        print(f"  uploading {path_in_repo} ({shard_path.stat().st_size / (1024 * 1024):.2f} MiB)...")
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=path_in_repo,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload {path_in_repo}",
        )

    print(f"[dataset] Uploading master_hbr.db ({DB_PATH.stat().st_size / (1024 * 1024):.2f} MiB)...")
    api.upload_file(
        path_or_fileobj=str(DB_PATH),
        path_in_repo="master_hbr.db",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Upload master_hbr.db",
    )

    # 5. Final verification
    verify_repo(api, MODEL_REPO_ID, "model")
    verify_repo(api, DATASET_REPO_ID, "dataset")
    print("\nALL DONE!")


if __name__ == "__main__":
    main()
