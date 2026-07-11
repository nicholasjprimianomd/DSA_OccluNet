"""Visualization helpers: see what goes INTO the model (preprocessed clips) and what
comes OUT (predictions, class separability) as it trains.

All functions render to PNG/CSV on disk (headless `Agg` backend) so they work over SSH
and inside the training loop. Wire them in with `train_dsa_backbone.py --viz-dir DIR`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", torch.float32).numpy()


def save_clip_montage(clip: torch.Tensor, title: str, path: Path, max_frames: int = 16) -> None:
    """Grid of the frames actually fed to the backbone. `clip` is (T, 3, H, W) in [0, 1]."""
    frames = _to_numpy(clip)[:, 0]  # channels are replicated grayscale; take one
    n = min(len(frames), max_frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(frames[i], cmap="gray", vmin=0.0, vmax=1.0)
            axes[i].set_title(f"f{i}", fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def save_input_samples(batch: dict[str, Any], out_dir: Path, label_names: Sequence[str], tag: str,
                       max_samples: int = 8) -> None:
    """Dump montages for a batch so you can eyeball the model's actual inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]
    for i in range(min(len(labels), max_samples)):
        name = label_names[int(labels[i])]
        meta = batch["metadata"][i]
        title = f"{tag} | {name} | {meta.get('view','?')} {meta.get('accession','?')} {meta.get('run_column','')}"
        save_clip_montage(pixel_values[i], title, out_dir / f"{tag}_{i:02d}_{name}.png")


def save_label_distribution(counts: dict[str, int], path: Path, title: str = "Label distribution") -> None:
    names = list(counts.keys())
    values = [counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 0.9), 3))
    ax.bar(names, values, color="#4c78a8")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str],
                          path: Path, title: str = "Confusion matrix") -> None:
    k = len(label_names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(1.4 * k + 1.5, 1.4 * k + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(k), labels=label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k), labels=label_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_embedding_scatter(features: np.ndarray, labels: np.ndarray, label_names: Sequence[str],
                           path: Path, title: str = "Pooled features (PCA)") -> None:
    """2-D PCA of the backbone's pooled features, colored by class — shows whether the
    representation separates the classes at all. Needs >= 2 samples."""
    if features.shape[0] < 2:
        return
    x = features - features.mean(axis=0, keepdims=True)
    # PCA via SVD (no sklearn dependency needed here).
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.get_cmap("tab10")
    for idx, name in enumerate(label_names):
        mask = labels == idx
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=28, color=cmap(idx % 10), label=name, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_per_class_bars(per_class: dict, label_names: Sequence[str], path: Path,
                        title: str = "Per-class metrics") -> None:
    """Grouped bars of precision / recall / F1 per class — see which class the model fails."""
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(label_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(5, len(label_names) * 1.4), 4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, per_class[m], width, label=m)
    ax.set_xticks(x, labels=[f"{n}\n(n={int(s)})" for n, s in zip(label_names, per_class["support"])],
                  fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_metric_hbar(labels: Sequence[str], values: Sequence[float], path: Path,
                     title: str = "", baseline: float | None = None) -> None:
    """Horizontal bar chart comparing a metric across many recipes."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.42)))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c78a8")
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.set_xlabel("macro-F1")
    ax.set_xlim(0, max(0.6, max(values) * 1.15) if values else 1)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    if baseline is not None:
        ax.axvline(baseline, color="crimson", ls="--", lw=1, label=f"baseline {baseline:.3f}")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_predictions_csv(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                         label_names: Sequence[str], metadata: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["accession", "view", "run_column", "true", "pred", "correct",
                    *[f"p_{n}" for n in label_names]])
        for i in range(len(y_true)):
            m = metadata[i]
            w.writerow([
                m.get("accession", ""), m.get("view", ""), m.get("run_column", ""),
                label_names[int(y_true[i])], label_names[int(y_pred[i])],
                int(y_true[i] == y_pred[i]), *[f"{p:.4f}" for p in probs[i]],
            ])


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Unweighted mean per-class F1 — the metric to trust under class imbalance."""
    f1s = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0
