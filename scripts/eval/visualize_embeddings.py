"""Generate portfolio-grade t-SNE visualizations of Frozen Camie-v2 (person+head 1536-D) embeddings.

Outputs:
1. tsne_12chars_solo_vs_multi.png
   - 12 representative HBR characters (31A + popular + long-tail akagi_rumi & irie_miyuki)
   - Distinguishes Solo (circle) vs Multi-char mined crops (triangle)
2. tsne_hbr_vs_other.png
   - Same 12 HBR characters overlaid with v2 `_other` class samples (gray 'x')
   - Visually demonstrates why `_other` scatters across HBR boundaries and motivates 5-Tier Safety-Valve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SELECTED_CHARS = [
    "kayamori_ruka",
    "izumi_yuki",
    "tojo_tsukasa",
    "asakura_karen_(heaven_burns_red)",
    "aikawa_megumi",
    "kunimi_tama",
    "shirakawa_yuina",
    "aoi_erika",
    "ooshima_ichiko",
    "yamawaki_bon_ivar",
    "akagi_rumi",
    "irie_miyuki",
]

DISPLAY_NAMES = {
    "kayamori_ruka": "Kayamori Ruka",
    "izumi_yuki": "Izumi Yuki",
    "tojo_tsukasa": "Tojo Tsukasa",
    "asakura_karen_(heaven_burns_red)": "Asakura Karen",
    "aikawa_megumi": "Aikawa Megumi",
    "kunimi_tama": "Kunimi Tama",
    "shirakawa_yuina": "Shirakawa Yuina",
    "aoi_erika": "Aoi Erika",
    "ooshima_ichiko": "Ooshima Ichiko",
    "yamawaki_bon_ivar": "Yamawaki Bon Ivar",
    "akagi_rumi": "Akagi Rumi (Long-tail)",
    "irie_miyuki": "Irie Miyuki (Long-tail)",
}

# 12 high-contrast distinct colors
PALETTE = [
    "#E6194B",  # red
    "#3CB44B",  # green
    "#4363D8",  # blue
    "#F58231",  # orange
    "#911EB4",  # purple
    "#42D4F4",  # cyan
    "#F032E6",  # magenta
    "#808000",  # olive
    "#008080",  # teal
    "#9A6324",  # brown
    "#D4AC0D",  # gold (akagi_rumi)
    "#1F2937",  # slate-black (irie_miyuki)
]


def l2_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def concat_ph(d: np.lib.npyio.NpzFile) -> np.ndarray:
    xp = l2_norm(d["X_p"])
    xh = l2_norm(d["X_h"])
    return l2_norm(np.concatenate([xp, xh], axis=1))


def subsample_per_class(
    X: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    target_chars: list[str],
    max_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_keep: list[int] = []
    for ch in target_chars:
        ch_idx = np.where(labels == ch)[0]
        if len(ch_idx) > max_per_class:
            # Keep balanced mix of solo and multi if both exist
            solo_idx = ch_idx[sources[ch_idx] == "solo"]
            multi_idx = ch_idx[sources[ch_idx] == "multi"]
            half = max_per_class // 2
            take_s = solo_idx if len(solo_idx) <= half else rng.choice(solo_idx, size=half, replace=False)
            rem = max_per_class - len(take_s)
            take_m = multi_idx if len(multi_idx) <= rem else rng.choice(multi_idx, size=rem, replace=False)
            chosen = np.concatenate([take_s, take_m])
            if len(chosen) < max_per_class and len(solo_idx) > len(take_s):
                extra_pool = np.setdiff1d(solo_idx, take_s)
                extra = rng.choice(extra_pool, size=min(len(extra_pool), max_per_class - len(chosen)), replace=False)
                chosen = np.concatenate([chosen, extra])
            idx_keep.extend(chosen.tolist())
        else:
            idx_keep.extend(ch_idx.tolist())
    idx_arr = np.array(idx_keep, dtype=np.int64)
    return X[idx_arr], labels[idx_arr], sources[idx_arr]


def plot_solo_vs_multi(
    X_ph: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    out_path: Path,
    seed: int = 42,
) -> None:
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(X_ph)

    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFC")

    for i, ch in enumerate(SELECTED_CHARS):
        color = PALETTE[i]
        m_solo = (labels == ch) & (sources == "solo")
        m_multi = (labels == ch) & (sources == "multi")

        if np.any(m_solo):
            ax.scatter(
                coords[m_solo, 0],
                coords[m_solo, 1],
                c=color,
                marker="o",
                s=34,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
        if np.any(m_multi):
            ax.scatter(
                coords[m_multi, 0],
                coords[m_multi, 1],
                c=color,
                marker="^",
                s=46,
                alpha=0.88,
                edgecolors="#111827",
                linewidths=0.45,
                zorder=4,
            )

        # Label at cluster median
        m_all = labels == ch
        if np.any(m_all):
            cx, cy = np.median(coords[m_all, 0]), np.median(coords[m_all, 1])
            short = DISPLAY_NAMES[ch].replace(" (Long-tail)", "*")
            ax.text(
                cx,
                cy,
                short,
                fontsize=8.5,
                fontweight="bold",
                color="#111827",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.86, lw=1.1),
                zorder=6,
            )

    n_solo = int(np.sum(sources == "solo"))
    n_multi = int(np.sum(sources == "multi"))
    ax.set_title(
        "Frozen Camie-v2 (ViT-Base) Person+Head 1536-D Embedding Space (t-SNE, Cosine)\n"
        f"12 Representative HBR Characters — Solo ({n_solo}) vs Multi-char Mined Crops ({n_multi})",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#111827",
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, color="#374151")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, color="#374151")
    ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: characters + marker shapes
    char_handles = [
        Line2D([0], [0], marker="o", color="w", label=DISPLAY_NAMES[ch], markerfacecolor=PALETTE[i], markersize=8)
        for i, ch in enumerate(SELECTED_CHARS)
    ]
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", label="Solo image (v2)", markerfacecolor="#6B7280", markersize=8),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Multi-char crop (v3 mined)",
            markerfacecolor="#6B7280",
            markeredgecolor="#111827",
            markersize=8.5,
        ),
    ]
    leg1 = ax.legend(
        handles=char_handles,
        title="Characters (* = Long-tail)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        fontsize=8.5,
        title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=shape_handles,
        title="Data Source",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        fontsize=8.5,
        title_fontsize=9,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hbr_vs_other(
    X_hbr: np.ndarray,
    labels_hbr: np.ndarray,
    X_other: np.ndarray,
    out_path: Path,
    seed: int = 42,
) -> None:
    X_all = np.concatenate([X_hbr, X_other], axis=0)
    labels_all = np.concatenate([labels_hbr, np.array(["_other"] * len(X_other))], axis=0)

    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFC")

    # Plot HBR clusters first
    for i, ch in enumerate(SELECTED_CHARS):
        m = labels_all == ch
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            c=PALETTE[i],
            marker="o",
            s=30,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )
        cx, cy = np.median(coords[m, 0]), np.median(coords[m, 1])
        short = DISPLAY_NAMES[ch].replace(" (Long-tail)", "")
        ax.text(
            cx,
            cy,
            short,
            fontsize=8.2,
            fontweight="bold",
            color="#111827",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=PALETTE[i], alpha=0.82, lw=1.0),
            zorder=5,
        )

    # Overlay `_other` samples clearly
    m_other = labels_all == "_other"
    ax.scatter(
        coords[m_other, 0],
        coords[m_other, 1],
        c="#111827",
        marker="x",
        s=42,
        alpha=0.72,
        linewidths=1.2,
        zorder=4,
        label=f"_other / Non-HBR ({int(np.sum(m_other))} samples)",
    )

    ax.set_title(
        "Why Closed-Set `_other` Fails: HBR Clusters vs `_other` Distribution (t-SNE, Cosine)\n"
        "`_other` does not form a compact cluster — it scatters across the space & encroaches on HBR boundaries",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
        color="#111827",
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, color="#374151")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, color="#374151")
    ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="#111827",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label=f"_other / Non-HBR ({int(np.sum(m_other))})",
        )
    ] + [
        Line2D([0], [0], marker="o", color="w", label=DISPLAY_NAMES[ch], markerfacecolor=PALETTE[i], markersize=8)
        for i, ch in enumerate(SELECTED_CHARS)
    ]
    ax.legend(
        handles=handles,
        title="Classes",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        fontsize=8.5,
        title_fontsize=9,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


LOOKALIKE_PAIRS = [
    ("higuchi_seika", "izumi_yuki", "Higuchi Seika", "Izumi Yuki"),
    ("kura_satomi", "tsukishiro_monaka", "Kura Satomi", "Tsukishiro Monaka"),
    ("kayamori_ruka", "shirakawa_yuina", "Kayamori Ruka", "Shirakawa Yuina"),
    ("shirakawa_yuina", "tojo_tsukasa", "Shirakawa Yuina", "Tojo Tsukasa"),
]


def plot_decision_margins(
    X_v2: np.ndarray,
    y_v2: np.ndarray,
    X_multi: np.ndarray,
    y_multi: np.ndarray,
    out_path: Path,
    max_per_class: int = 250,
    seed: int = 42,
) -> None:
    """Plot 5-Fold OOF 1536-D signed logit margins (logit_A - logit_B) for the top look-alike pairs."""
    rng = np.random.default_rng(seed)
    m_v2 = y_v2 != "_other"
    X_all = np.concatenate([X_v2[m_v2], X_multi], axis=0)
    y_all = np.concatenate([y_v2[m_v2], y_multi], axis=0)

    # Cap per class (same protocol as train_v3.py)
    keep: list[int] = []
    classes = sorted(set(y_all.tolist()))
    for c in classes:
        idx = np.where(y_all == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(idx.tolist())
    keep_arr = np.array(keep, dtype=np.int64)
    X_tr = X_all[keep_arr]
    y_tr = y_all[keep_arr]

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_int = np.array([cls_to_idx[c] for c in y_tr], dtype=np.int64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_logits = np.zeros((len(y_int), len(classes)), dtype=np.float32)
    for tr_i, va_i in skf.split(X_tr, y_int):
        clf = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X_tr[tr_i], y_int[tr_i])
        oof_logits[va_i] = clf.decision_function(X_tr[va_i]).astype(np.float32)

    # Compute centroid cosine sims on X_tr
    centroids = {c: l2_norm(X_tr[y_tr == c].mean(axis=0, keepdims=True))[0] for c in classes}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), dpi=180)
    fig.patch.set_facecolor("white")

    for ax, (ca, cb, na, nb) in zip(axes.flat, LOOKALIKE_PAIRS):
        ax.set_facecolor("#FAFAFC")
        ia, ib = cls_to_idx[ca], cls_to_idx[cb]
        ma, mb = (y_int == ia), (y_int == ib)

        # Signed margin along (w_A - w_B): >0 favors A, <0 favors B
        marg_a = oof_logits[ma, ia] - oof_logits[ma, ib]
        marg_b = oof_logits[mb, ia] - oof_logits[mb, ib]
        cos_sim = float(np.dot(centroids[ca], centroids[cb]))

        correct = int(np.sum(marg_a > 0) + np.sum(marg_b < 0))
        total = int(len(marg_a) + len(marg_b))
        acc = 100.0 * correct / max(total, 1)

        lo = float(min(marg_b.min(), -1.0)) - 0.5
        hi = float(max(marg_a.max(), 1.0)) + 0.5
        bins = np.linspace(lo, hi, 32)

        ax.hist(marg_b, bins=bins, color="#3B82F6", alpha=0.68, edgecolor="white", linewidth=0.5, label=f"{nb} (n={len(marg_b)})")
        ax.hist(marg_a, bins=bins, color="#EF4444", alpha=0.68, edgecolor="white", linewidth=0.5, label=f"{na} (n={len(marg_a)})")
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.6, label="1536-D Hyperplane (margin = 0)")

        ax.set_title(
            f"{na}  vs  {nb}\nCentroid Cosine = {cos_sim:.3f}  |  5-Fold OOF Pairwise Acc = {acc:.2f}% ({correct}/{total})",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
            pad=8,
        )
        ax.set_xlabel(f"← Favors {nb}   |   Logit Margin (logit_A - logit_B)   |   Favors {na} →", fontsize=8.5, color="#374151")
        ax.set_ylabel("OOF Sample Count", fontsize=8.5, color="#374151")
        ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="upper right")

    fig.suptitle(
        "High-Dimensional Separability of Top Look-Alike Character Pairs (5-Fold Out-of-Fold)\n"
        "Even pairs with ~0.90 centroid cosine similarity separate cleanly across the 1536-D linear hyperplane",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Camie-v2 1536-D character embeddings via t-SNE.")
    parser.add_argument("--v2-embs", type=Path, default=PROJECT_ROOT / "data/embeddings/v2_solo_embeddings.npz")
    parser.add_argument("--multi-embs", type=Path, default=PROJECT_ROOT / "data/embeddings/danbooru_multi_crops_for_v3.npz")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs/embeddings")
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-margins", action="store_true", help="Only generate the 5-Fold OOF decision margin plot.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    d_v2 = np.load(args.v2_embs, allow_pickle=True)
    d_multi = np.load(args.multi_embs, allow_pickle=True)

    X_v2 = concat_ph(d_v2)
    y_v2 = d_v2["labels"].astype(str)

    X_multi = concat_ph(d_multi)
    y_multi = d_multi["labels"].astype(str)

    if not args.only_margins:
        # Combine solo + multi for Plot 1
        X_comb = np.concatenate([X_v2, X_multi], axis=0)
        y_comb = np.concatenate([y_v2, y_multi], axis=0)
        src_comb = np.array(["solo"] * len(y_v2) + ["multi"] * len(y_multi))

        X_sub, y_sub, src_sub = subsample_per_class(
            X_comb, y_comb, src_comb, SELECTED_CHARS, max_per_class=args.max_per_class, rng=rng
        )
        out1 = args.out_dir / "tsne_12chars_solo_vs_multi.png"
        print(f"[1/3] Running t-SNE for 12 HBR chars ({len(X_sub)} samples) -> {out1}")
        plot_solo_vs_multi(X_sub, y_sub, src_sub, out1, seed=args.seed)

        # Plot 2: HBR solo (12 chars, max 80/class) + all `_other` from v2
        X_hbr_s, y_hbr_s, _ = subsample_per_class(
            X_v2, y_v2, np.array(["solo"] * len(y_v2)), SELECTED_CHARS, max_per_class=80, rng=rng
        )
        X_other = X_v2[y_v2 == "_other"]
        out2 = args.out_dir / "tsne_hbr_vs_other.png"
        print(f"[2/3] Running t-SNE for HBR ({len(X_hbr_s)}) vs _other ({len(X_other)}) -> {out2}")
        plot_hbr_vs_other(X_hbr_s, y_hbr_s, X_other, out2, seed=args.seed)

    out3 = args.out_dir / "oof_decision_margins_lookalikes.png"
    print(f"[3/3] Computing 5-Fold OOF 1536-D decision margins -> {out3}")
    plot_decision_margins(X_v2, y_v2, X_multi, y_multi, out3, max_per_class=250, seed=args.seed)
    print("Done!")


if __name__ == "__main__":
    main()
