"""Classification metrics + human-readable reports for the DSA subtype models.

Built on scikit-learn so the numbers match what people expect, with plain-text
formatting so a training run prints something you can actually read in a terminal.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    labels = list(range(num_classes))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _fmt_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def format_per_class(metrics: dict[str, Any], label_names: Sequence[str]) -> str:
    pc = metrics["per_class"]
    w = [max(12, max(len(n) for n in label_names)), 10, 8, 8, 8]
    lines = [_fmt_row(["class", "precision", "recall", "f1", "support"], w)]
    lines.append("-" * (sum(w) + 2 * len(w)))
    for i, name in enumerate(label_names):
        lines.append(_fmt_row(
            [name, f"{pc['precision'][i]:.3f}", f"{pc['recall'][i]:.3f}",
             f"{pc['f1'][i]:.3f}", str(int(pc['support'][i]))], w))
    lines.append("-" * (sum(w) + 2 * len(w)))
    lines.append(_fmt_row(
        ["macro avg", f"{np.mean(pc['precision']):.3f}", f"{np.mean(pc['recall']):.3f}",
         f"{metrics['macro_f1']:.3f}", str(int(np.sum(pc['support'])))], w))
    return "\n".join(lines)


def format_confusion(metrics: dict[str, Any], label_names: Sequence[str]) -> str:
    cm = np.array(metrics["confusion"])
    short = [n[:8] for n in label_names]
    w0 = max(10, max(len(n) for n in label_names))
    header = _fmt_row(["true \\ pred"] + short, [w0] + [8] * len(short))
    lines = [header, "-" * len(header)]
    for i, name in enumerate(label_names):
        row = cm[i]
        pct = row / row.sum() if row.sum() else row
        cells = [f"{row[j]} ({pct[j]:.0%})" for j in range(len(short))]
        lines.append(_fmt_row([name] + cells, [w0] + [8] * len(short)))
    return "\n".join(lines)


def format_cv_report(
    config: dict[str, Any],
    class_counts: dict[str, int],
    per_fold: list[dict[str, Any]],
    oof_metrics: dict[str, Any],
    label_names: Sequence[str],
) -> str:
    """The full, readable summary printed at the end of cross-validation."""
    L = []
    L.append("=" * 70)
    L.append(f"Cross-validation — {config['view']} / {config['stage']}")
    L.append("=" * 70)
    L.append(f"Backbone : {config['backbone']} ({'frozen' if config['frozen'] else 'fine-tuned'})")
    L.append(f"Probe    : {config['probe']}   folds={config['folds']}   "
             f"samples={config['n_samples']}   patients={config['n_groups']}")
    L.append("Classes  : " + "  ".join(f"{n}({class_counts.get(n, 0)})" for n in label_names))
    L.append("")

    # Per-fold table
    fw = [6, 9, 7, 10, 9, 8]
    L.append("Per-fold (model trained on all-but-one fold, scored on the held-out fold):")
    L.append(_fmt_row(["fold", "n_train", "n_val", "macro_f1", "bal_acc", "acc"], fw))
    L.append("-" * (sum(fw) + 2 * len(fw)))
    for f in per_fold:
        L.append(_fmt_row(
            [f["fold"], f["n_train"], f["n_val"], f"{f['macro_f1']:.3f}",
             f"{f['balanced_accuracy']:.3f}", f"{f['accuracy']:.3f}"], fw))

    def ms(key: str) -> str:
        vals = [f[key] for f in per_fold]
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    L.append("-" * (sum(fw) + 2 * len(fw)))
    L.append(f"Aggregate over folds:  macro_f1 {ms('macro_f1')}   "
             f"bal_acc {ms('balanced_accuracy')}   acc {ms('accuracy')}")
    L.append("")
    L.append("Pooled out-of-fold performance (every sample scored once, held out):")
    L.append(f"  macro_f1={oof_metrics['macro_f1']:.3f}   "
             f"balanced_accuracy={oof_metrics['balanced_accuracy']:.3f}   "
             f"accuracy={oof_metrics['accuracy']:.3f}")
    L.append("")
    L.append(format_per_class(oof_metrics, label_names))
    L.append("")
    L.append("Confusion matrix (row=true label, col=predicted, % of the true row):")
    L.append(format_confusion(oof_metrics, label_names))
    return "\n".join(L)


def baseline_macro_f1(y_true: np.ndarray, num_classes: int) -> float:
    """Macro-F1 of the majority-class predictor — the floor to beat."""
    majority = int(np.bincount(y_true, minlength=num_classes).argmax())
    return compute_metrics(y_true, np.full_like(y_true, majority), num_classes)["macro_f1"]
