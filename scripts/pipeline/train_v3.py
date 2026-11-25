"""Train v3 Pure HBR-55 classifier by merging v2 solo embeddings with high-conf multi-char crop embeddings.

Merges:
  - data/embeddings/v2_solo_embeddings.npz (excluding '_other')
  - data/embeddings/danbooru_multi_crops_for_v3.npz
Capping each class at --max-per-class (default 250) total samples so popular characters
do not overwhelm minor characters.

Outputs:
  - models/v3_pure_hbr55_plus_multi_camie.npz (person+head, person_only, head_only)
  - data/embeddings/v3_train_report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def eval_5fold_cv(X: np.ndarray, y: np.ndarray, rumi_idx: int) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.zeros_like(y)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        cv_preds[va] = clf.predict(X[va])
    overall = 100.0 * float(np.mean(cv_preds == y))
    rumi_mask = y == rumi_idx
    rumi_acc = 100.0 * float(np.mean(cv_preds[rumi_mask] == y[rumi_mask])) if rumi_mask.any() else 0.0
    return overall, rumi_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train v3 pure HBR-55 classifier from v2 solo + master DB multi-char crops."
    )
    parser.add_argument(
        "--v2-embs",
        default=str(PROJECT_ROOT / "data" / "embeddings" / "v2_solo_embeddings.npz"),
    )
    parser.add_argument(
        "--multi-embs",
        default=str(PROJECT_ROOT / "data" / "embeddings" / "danbooru_multi_crops_for_v3.npz"),
    )
    parser.add_argument(
        "--out-weights",
        default=str(PROJECT_ROOT / "models" / "v3_pure_hbr55_plus_multi_camie.npz"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "data" / "embeddings" / "v3_train_report.json"),
    )
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # 1. Load v2 embeddings (excluding '_other')
    v2 = np.load(args.v2_embs, allow_pickle=True)
    v2_labels_all = [str(x) for x in v2["labels"]]
    keep_v2 = np.array([lbl != "_other" for lbl in v2_labels_all], dtype=bool)
    v2_Xp = v2["X_p"][keep_v2].astype(np.float32)
    v2_Xh = v2["X_h"][keep_v2].astype(np.float32)
    v2_labels = [lbl for lbl, k in zip(v2_labels_all, keep_v2) if k]
    v2_counts = collections.Counter(v2_labels)
    classes = sorted(v2_counts.keys())
    c2i = {c: i for i, c in enumerate(classes)}

    # 2. Load multi-character crop embeddings from master DB
    multi = np.load(args.multi_embs, allow_pickle=True)
    m_labels_all = [str(x) for x in multi["labels"]]
    keep_m = np.array([lbl in c2i for lbl in m_labels_all], dtype=bool)
    m_Xp = multi["X_p"][keep_m].astype(np.float32)
    m_Xh = multi["X_h"][keep_m].astype(np.float32)
    m_labels = [lbl for lbl, k in zip(m_labels_all, keep_m) if k]
    m_counts = collections.Counter(m_labels)

    print(f"Loaded v2 solo (excl. _other): {len(v2_labels):,} samples across {len(classes)} classes")
    print(f"Loaded multi-char crops      : {len(m_labels):,} samples across {len(m_counts)} classes")

    # 3. Merge per class with cap at --max-per-class
    v2_idx_by_c: dict[str, list[int]] = collections.defaultdict(list)
    for i, lbl in enumerate(v2_labels):
        v2_idx_by_c[lbl].append(i)

    m_idx_by_c: dict[str, list[int]] = collections.defaultdict(list)
    for i, lbl in enumerate(m_labels):
        m_idx_by_c[lbl].append(i)

    Xp_parts, Xh_parts, merged_labels = [], [], []
    added_multi_total = 0
    capped_classes = 0
    per_class_stats: dict[str, dict[str, int]] = {}

    for c in classes:
        v2_idxs = np.array(v2_idx_by_c[c], dtype=np.int64)
        if len(v2_idxs) > args.max_per_class:
            v2_idxs = rng.choice(v2_idxs, size=args.max_per_class, replace=False)

        rem = max(0, args.max_per_class - len(v2_idxs))
        m_idxs = np.array(m_idx_by_c[c], dtype=np.int64)
        if len(m_idxs) > rem:
            m_idxs = rng.choice(m_idxs, size=rem, replace=False) if rem > 0 else np.array([], dtype=np.int64)
            capped_classes += 1

        if len(v2_idxs) > 0:
            Xp_parts.append(v2_Xp[v2_idxs])
            Xh_parts.append(v2_Xh[v2_idxs])
            merged_labels.extend([c] * len(v2_idxs))
        if len(m_idxs) > 0:
            Xp_parts.append(m_Xp[m_idxs])
            Xh_parts.append(m_Xh[m_idxs])
            merged_labels.extend([c] * len(m_idxs))
            added_multi_total += len(m_idxs)

        per_class_stats[c] = {
            "v2_used": int(len(v2_idxs)),
            "multi_avail": int(len(m_idx_by_c[c])),
            "multi_used": int(len(m_idxs)),
            "total": int(len(v2_idxs) + len(m_idxs)),
        }

    X_p = l2_norm(np.concatenate(Xp_parts, axis=0).astype(np.float32))
    X_h = l2_norm(np.concatenate(Xh_parts, axis=0).astype(np.float32))
    X_ph = l2_norm(np.concatenate([X_p, X_h], axis=1).astype(np.float32))
    y = np.array([c2i[lbl] for lbl in merged_labels], dtype=np.int64)

    counts = collections.Counter(merged_labels)
    min_c, min_n = min(counts.items(), key=lambda kv: kv[1])
    max_c, max_n = max(counts.items(), key=lambda kv: kv[1])
    print(
        f"Merged dataset (cap={args.max_per_class}): {len(y):,} samples "
        f"(v2={len(y) - added_multi_total:,}, multi_added={added_multi_total:,}, capped_classes={capped_classes})"
    )
    print(
        f"Class count range: min={min_n} ({min_c}), max={max_n} ({max_c}), "
        f"akagi_rumi={counts['akagi_rumi']} (v2={per_class_stats['akagi_rumi']['v2_used']}, "
        f"multi_used={per_class_stats['akagi_rumi']['multi_used']})"
    )

    # 4. 5-fold CV for person+head, person_only, head_only
    rumi_idx = c2i["akagi_rumi"]
    cv_ph_all, cv_ph_rumi = eval_5fold_cv(X_ph, y, rumi_idx)
    cv_p_all, cv_p_rumi = eval_5fold_cv(X_p, y, rumi_idx)
    cv_h_all, cv_h_rumi = eval_5fold_cv(X_h, y, rumi_idx)

    print("\n================ V3 5-FOLD CV RESULTS ================")
    print(f"person+head : Top-1 = {cv_ph_all:.2f}% | akagi_rumi = {cv_ph_rumi:.2f}%")
    print(f"person_only : Top-1 = {cv_p_all:.2f}% | akagi_rumi = {cv_p_rumi:.2f}%")
    print(f"head_only   : Top-1 = {cv_h_all:.2f}% | akagi_rumi = {cv_h_rumi:.2f}%")

    # 5. Train final heads on full merged dataset
    clf_ph = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced").fit(X_ph, y)
    clf_p = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced").fit(X_p, y)
    clf_h = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced").fit(X_h, y)

    out_weights = Path(args.out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_weights,
        classes=np.array(classes),
        ph_coef=clf_ph.coef_.astype(np.float32),
        ph_intercept=clf_ph.intercept_.astype(np.float32),
        p_coef=clf_p.coef_.astype(np.float32),
        p_intercept=clf_p.intercept_.astype(np.float32),
        h_coef=clf_h.coef_.astype(np.float32),
        h_intercept=clf_h.intercept_.astype(np.float32),
    )

    report = {
        "v2_solo_samples": int(len(y) - added_multi_total),
        "multi_available": int(len(m_labels)),
        "multi_added": int(added_multi_total),
        "merged_total_samples": int(len(y)),
        "num_classes": len(classes),
        "max_per_class": args.max_per_class,
        "capped_classes": capped_classes,
        "min_class": {"name": min_c, "count": min_n},
        "max_class": {"name": max_c, "count": max_n},
        "akagi_rumi_stats": per_class_stats["akagi_rumi"],
        "cv5_person_head_top1_pct": round(cv_ph_all, 2),
        "cv5_person_head_akagi_rumi_pct": round(cv_ph_rumi, 2),
        "cv5_person_only_top1_pct": round(cv_p_all, 2),
        "cv5_person_only_akagi_rumi_pct": round(cv_p_rumi, 2),
        "cv5_head_only_top1_pct": round(cv_h_all, 2),
        "cv5_head_only_akagi_rumi_pct": round(cv_h_rumi, 2),
        "out_weights": str(out_weights),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved v3 weights : {out_weights}")
    print(f"Saved v3 report  : {out_report}")
    print(f"Elapsed          : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
