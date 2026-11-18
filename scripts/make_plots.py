#!/usr/bin/env python
"""Generate every figure in the report from the saved result artifacts.

Nothing here invents numbers: each plot reads the JSON/CSV/NPY files written by the
three pipelines' ``test.py`` runs.

    python scripts/make_plots.py

Figures land in ``Results/plots/``:

    confusion_matrix_<variant>.png     test confusion matrix per variant
    training_curves_<variant>.png      loss + accuracy curves per variant
    tsne_<block>.png                   t-SNE of each learned representation block
    tsne_blocks_comparison.png         the three blocks side by side
    per_class_f1_comparison.png        per-emotion F1 across the three variants
    accuracy_comparison.png            headline accuracy / macro-F1 comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from models.common import config, io_utils as IO  # noqa: E402

VARIANTS = ("speech", "text", "fusion")
# Representation blocks required by the assignment, in the order they appear in
# the architecture figure.
BLOCKS = (
    ("temporal_block", "Speech temporal block (BiLSTM over mel frames)", "speech"),
    ("contextual_block", "Text contextual block (BiLSTM over word embeddings)", "text"),
    ("fusion_block", "Fusion block (concatenated speech + text)", "fusion"),
)

SHORT = {"speech": "Speech only", "text": "Text only", "fusion": "Multimodal"}
TITLES = {"speech": "Speech-only", "text": "Text-only", "fusion": "Multimodal (fusion)"}


def _emotion_colors() -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {e: cmap(i % 10) for i, e in enumerate(config.EMOTIONS)}


def plot_confusion_matrix(variant: str, out_path: Path) -> None:
    payload = IO.load_json(config.METRICS_DIR / f"metrics_{variant}.json")
    cm = np.array(payload["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n = len(config.EMOTIONS)
    ax.set_xticks(range(n), config.EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(n), config.EMOTIONS)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(f"{TITLES[variant]} - test confusion matrix\n"
                 f"accuracy {payload['accuracy']:.3f} | macro F1 {payload['macro_f1']:.3f}"
                 f" | n={payload['n_test']}")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised proportion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_training_curves(variant: str, out_path: Path) -> None:
    hist = pd.read_csv(config.METRICS_DIR / f"history_{variant}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", ms=3, label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], marker="s", ms=3, label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title(f"{TITLES[variant]}: loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist["epoch"], hist["train_acc"], marker="o", ms=3, label="train acc")
    axes[1].plot(hist["epoch"], hist["val_acc"], marker="s", ms=3, label="val acc")
    axes[1].plot(hist["epoch"], hist["val_macro_f1"], marker="^", ms=3,
                 label="val macro F1")
    axes[1].axhline(1 / config.NUM_CLASSES, ls=":", c="grey",
                    label="chance (1/7 = 0.143)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"{TITLES[variant]}: accuracy / macro F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def _tsne_coords(rep: np.ndarray, seed: int) -> np.ndarray:
    perplexity = float(min(30, max(5, (rep.shape[0] - 1) / 3)))
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init="pca", learning_rate="auto").fit_transform(rep)


def _scatter(ax, coords, labels, title, subtitle=None):
    colors = _emotion_colors()
    for i, emotion in enumerate(config.EMOTIONS):
        m = labels == i
        ax.scatter(coords[m, 0], coords[m, 1], s=17, alpha=0.8,
                   color=colors[emotion], label=emotion, edgecolors="none")
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def compute_tsne_all(seed: int) -> dict:
    """t-SNE + silhouette per block; returns the coordinates and scores."""
    labels_path = config.REPS_DIR / "test_labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"{labels_path} missing - run the pipelines' test.py first.")
    labels = np.load(labels_path)

    results = {}
    for block, title, variant in BLOCKS:
        rep_path = config.REPS_DIR / f"reps_{variant}_test.npy"
        if not rep_path.is_file():
            print(f"  ! missing {rep_path.name}; skipping block {block}")
            continue
        rep = np.load(rep_path)
        coords = _tsne_coords(rep, seed)
        sil = float(silhouette_score(rep, labels, metric="euclidean"))
        results[block] = {"variant": variant, "title": title, "coords": coords,
                          "labels": labels, "silhouette": sil, "dim": rep.shape[1]}

        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        _scatter(ax, coords, labels, f"t-SNE of {title}",
                 f"{rep.shape[1]}-d representation, {len(labels)} test utterances | "
                 f"silhouette {sil:.3f}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / f"tsne_{block}.png", dpi=160)
        plt.close(fig)
        print(f"  wrote tsne_{block}.png  (silhouette {sil:.3f})")
    return results


def plot_tsne_comparison(results: dict, out_path: Path) -> None:
    present = [b for b in BLOCKS if b[0] in results]
    fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 5.2))
    if len(present) == 1:
        axes = [axes]
    for ax, (block, title, variant) in zip(axes, present):
        r = results[block]
        _scatter(ax, r["coords"], r["labels"], TITLES[variant],
                 f"{r['dim']}-d | silhouette {r['silhouette']:.3f}")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(config.EMOTIONS),
               fontsize=9, frameon=False)
    fig.suptitle("Separability of emotion clusters across the three learned "
                 "representation blocks (identical test set)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_accuracy_comparison(out_path: Path) -> pd.DataFrame | None:
    rows = []
    for v in VARIANTS:
        p = config.METRICS_DIR / f"metrics_{v}.json"
        if p.is_file():
            rows.append(IO.load_json(p))
    if not rows:
        return None
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(df))
    w = 0.38
    axes[0].bar(x - w / 2, df["accuracy"], w, label="accuracy", color="#4C72B0")
    axes[0].bar(x + w / 2, df["macro_f1"], w, label="macro F1", color="#DD8452")
    axes[0].axhline(1 / config.NUM_CLASSES, ls=":", c="grey", label="chance (1/7)")
    axes[0].set_xticks(x, [SHORT[v] for v in df["variant"]])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Test accuracy and macro F1 by variant")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for xi, (a, f) in enumerate(zip(df["accuracy"], df["macro_f1"])):
        axes[0].text(xi - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(xi + w / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=8)

    n_emo = len(config.EMOTIONS)
    xg = np.arange(n_emo)
    wg = 0.26
    for k, v in enumerate(df["variant"]):
        vals = [IO.load_json(config.METRICS_DIR / f"metrics_{v}.json")["per_class_f1"][e]
                for e in config.EMOTIONS]
        axes[1].bar(xg + (k - 1) * wg, vals, wg, label=SHORT[v])
    axes[1].set_xticks(xg, config.EMOTIONS, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("F1")
    axes[1].set_title("Per-emotion F1 by variant")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path.name}")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Confusion matrices + curves:")
    for v in VARIANTS:
        if (config.METRICS_DIR / f"metrics_{v}.json").is_file():
            plot_confusion_matrix(v, config.PLOTS_DIR / f"confusion_matrix_{v}.png")
            plot_training_curves(v, config.PLOTS_DIR / f"training_curves_{v}.png")
        else:
            print(f"  ! no metrics for '{v}' yet; skipping")

    print("t-SNE:")
    results = compute_tsne_all(args.seed)
    if results:
        plot_tsne_comparison(results, config.PLOTS_DIR / "tsne_blocks_comparison.png")
        IO.save_json(
            {b: {"variant": r["variant"], "rep_dim": r["dim"],
                 "silhouette_euclidean": r["silhouette"]}
             for b, r in results.items()},
            config.METRICS_DIR / "silhouette_scores.json")

    print("Comparison:")
    df = plot_accuracy_comparison(config.PLOTS_DIR / "accuracy_comparison.png")
    if df is not None:
        print(df[["variant", "accuracy", "macro_f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
