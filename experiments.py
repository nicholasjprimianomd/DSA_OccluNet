"""Sweep feature/probe recipes to improve per-class DSA subtype performance.

Everything runs on frozen V-JEPA 2 features, so a whole grid of recipes is cheap. To make
comparisons honest, every recipe is scored on the *same* patient-grouped folds
(StratifiedGroupKFold on Study_Key), and any fitted preprocessing (e.g. standardization)
is fit on the training fold only.

Levers explored:
  - representation : mean-pool (baseline) vs mean+max+std pooling over backbone tokens
  - preprocessing  : raw / standardize / L2-normalize
  - classifier     : balanced logistic regression (C sweep) / RBF-SVM / linear-SVM / MLP
  - class weighting: none vs 'balanced'

Example:
    python experiments.py --view AP --stage positive_subtype --device cuda --amp --out runs/ap_exp
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader

import dataclasses

import metrics as M
import viz
from occlusion_loader import default_base_dir, default_excel_path
from train_dsa_backbone import (
    BACKBONE,
    BackboneReadyDataset,
    DsaVideoClassifier,
    build_base_dataset,
    build_records,
    collate_task_samples,
    select_device,
    stage_label_names,
)


# ---- rich feature extraction (mean / max / std pooling over tokens) --------------------
def extract_rich_features(view, stage, records, device, amp, batch_size, num_workers, clip_length=None):
    spec = BACKBONE if clip_length is None else dataclasses.replace(BACKBONE, clip_length=clip_length)
    dataset = BackboneReadyDataset(build_base_dataset(view, records), stage, stage_label_names(stage), spec=spec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_task_samples)
    model = DsaVideoClassifier(num_labels=len(stage_label_names(stage)), freeze_backbone=True).to(device)
    model.eval()

    means, maxes, stds, labels, groups, meta = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                out = model.backbone(**{model.spec.input_key: batch["pixel_values"].to(device)})
            h = out.last_hidden_state.float()          # (B, tokens, hidden)
            means.append(h.mean(dim=1).cpu())
            maxes.append(h.amax(dim=1).cpu())
            stds.append(h.std(dim=1).cpu())
            labels.append(batch["labels"])
            for m in batch["metadata"]:
                groups.append(m["study_key"])
                meta.append({k: m.get(k, "") for k in ("accession", "view", "run_column", "study_key", "label_text")})
            print(f"  features {min(i * batch_size, len(dataset))}/{len(dataset)}", end="\r", flush=True)
    print()
    return {
        "mean": torch.cat(means).numpy(),
        "max": torch.cat(maxes).numpy(),
        "std": torch.cat(stds).numpy(),
        "labels": torch.cat(labels).numpy(),
        "groups": np.array(groups, dtype=object),
        "meta": meta,
    }


def load_or_extract(cache, view, stage, records, device, amp, batch_size, num_workers, clip_length=None):
    tag = f"_f{clip_length}" if clip_length else ""
    path = Path(cache) / f"rich_{view}_{stage}{tag}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if len(d["labels"]) == len(records):
            print(f"Using cached rich features: {path}")
            return {k: d[k] for k in ("mean", "max", "std", "labels", "groups")} | {"meta": list(d["meta"])}
    data = extract_rich_features(view, stage, records, device, amp, batch_size, num_workers, clip_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v for k, v in data.items() if k != "meta"}, meta=np.array(data["meta"], dtype=object))
    return data


def representation(data, name: str) -> np.ndarray:
    if name == "mean":
        return data["mean"]
    if name == "meanmaxstd":
        return np.concatenate([data["mean"], data["max"], data["std"]], axis=1)
    raise ValueError(name)


# ---- recipe grid -----------------------------------------------------------------------
def prep_step(name):
    return {"raw": "passthrough", "std": StandardScaler(), "l2": Normalizer()}[name]


def clf_factory(name):
    bal = "balanced"
    return {
        "logreg":      lambda: LogisticRegression(max_iter=5000, C=1.0, class_weight=bal),
        "logreg_C0.3": lambda: LogisticRegression(max_iter=5000, C=0.3, class_weight=bal),
        "logreg_C3":   lambda: LogisticRegression(max_iter=5000, C=3.0, class_weight=bal),
        "logreg_noW":  lambda: LogisticRegression(max_iter=5000, C=1.0),
        "svm_rbf":     lambda: SVC(kernel="rbf", C=3.0, gamma="scale", class_weight=bal),
        "linsvm":      lambda: LinearSVC(C=0.5, class_weight=bal, max_iter=20000),
        "mlp":         lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=1500,
                                             early_stopping=True, random_state=0),
    }[name]


RECIPES = [
    ("baseline: mean / raw / logreg-noW", "mean", "raw", "logreg_noW"),
    ("mean / std / logreg", "mean", "std", "logreg"),
    ("mean / std / logreg C0.3", "mean", "std", "logreg_C0.3"),
    ("mean / std / logreg C3", "mean", "std", "logreg_C3"),
    ("mean / l2 / logreg", "mean", "l2", "logreg"),
    ("mean / std / svm_rbf", "mean", "std", "svm_rbf"),
    ("mean / std / linsvm", "mean", "std", "linsvm"),
    ("mean / std / mlp", "mean", "std", "mlp"),
    ("meanmaxstd / std / logreg", "meanmaxstd", "std", "logreg"),
    ("meanmaxstd / std / logreg C0.3", "meanmaxstd", "std", "logreg_C0.3"),
    ("meanmaxstd / std / svm_rbf", "meanmaxstd", "std", "svm_rbf"),
    ("meanmaxstd / l2 / logreg", "meanmaxstd", "l2", "logreg"),
    ("meanmaxstd / std / mlp", "meanmaxstd", "std", "mlp"),
]


def evaluate_recipe(X, y, folds, num_classes, prep, clf):
    oof = np.full(len(y), -1, dtype=int)
    for tr, va in folds:
        pipe = make_pipeline(prep_step(prep), clf_factory(clf)())
        pipe.fit(X[tr], y[tr])
        oof[va] = pipe.predict(X[va])
    return oof, M.compute_metrics(y, oof, num_classes)


def make_folds(y, groups, n_splits, seed):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", choices=("AP", "Lateral"), required=True)
    p.add_argument("--stage", default="positive_subtype", choices=("positive_subtype", "binary_detection"))
    p.add_argument("--excel", default=str(default_excel_path()))
    p.add_argument("--base-dir", default=str(default_base_dir()))
    p.add_argument("--treat-blank-as-negative", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-length", type=int, default=None,
                   help="Override frames per clip (default 16). DSA runs max ~34 frames, so 32 uses "
                   "every real frame; higher just interpolates.")
    p.add_argument("--out", default="runs/exp")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    rec_args = types.SimpleNamespace(
        excel=args.excel, base_dir=args.base_dir, view=args.view, stage=args.stage,
        treat_blank_as_negative=args.treat_blank_as_negative)
    records = build_records(rec_args)
    label_names = stage_label_names(args.stage)
    num_classes = len(label_names)

    clip_len = args.clip_length or BACKBONE.clip_length
    print(f"Extracting rich features for {len(records)} {args.view} runs "
          f"(frozen {BACKBONE.pretrained_name}, clip_length={clip_len})…")
    data = load_or_extract(out / "cache", args.view, args.stage, records,
                           device, args.amp, args.batch_size, args.num_workers, args.clip_length)
    y, groups = data["labels"], data["groups"]
    class_counts = {label_names[c]: int((y == c).sum()) for c in range(num_classes)}

    n_splits = min(args.folds, len(set(groups)), int(np.min(np.bincount(y, minlength=num_classes))))
    folds = make_folds(y, groups, n_splits, args.seed)
    baseline = M.baseline_macro_f1(y, num_classes)

    print(f"\nSweeping {len(RECIPES)} recipes on {args.view}/{args.stage} "
          f"({len(y)} runs, {len(set(groups))} patients, {n_splits} folds)")
    print(f"Classes: " + "  ".join(f"{n}({class_counts[n]})" for n in label_names))
    print(f"Majority-class baseline macro-F1 = {baseline:.3f}\n")

    results = []
    reps = {}
    for name, rep_name, prep, clf in RECIPES:
        if rep_name not in reps:
            reps[rep_name] = representation(data, rep_name)
        oof, m = evaluate_recipe(reps[rep_name], y, folds, num_classes, prep, clf)
        results.append({"recipe": name, "rep": rep_name, "prep": prep, "clf": clf,
                        "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                        "accuracy": m["accuracy"], "per_class_f1": m["per_class"]["f1"],
                        "metrics": m, "oof": oof})

    results.sort(key=lambda r: r["macro_f1"], reverse=True)

    # readable table
    cw = [40, 9, 8, 8] + [8] * num_classes
    header = ["recipe", "macroF1", "bal_acc", "acc"] + [f"F1:{n[:6]}" for n in label_names]
    print("  ".join(str(c).ljust(w) for c, w in zip(header, cw)))
    print("-" * (sum(cw) + 2 * len(cw)))
    for r in results:
        row = [r["recipe"][:40], f"{r['macro_f1']:.3f}", f"{r['balanced_accuracy']:.3f}", f"{r['accuracy']:.3f}"]
        row += [f"{f:.3f}" for f in r["per_class_f1"]]
        print("  ".join(str(c).ljust(w) for c, w in zip(row, cw)))

    best = results[0]
    print(f"\nBest: {best['recipe']}  macro-F1 {best['macro_f1']:.3f} "
          f"(baseline {baseline:.3f}, prior linear-probe ~0.42 AP / 0.41 Lat)")
    print("\nBest recipe per-class:")
    print(M.format_per_class(best["metrics"], label_names))

    # artifacts
    with (out / "sweep_results.json").open("w") as f:
        json.dump({"view": args.view, "baseline_macro_f1": baseline, "class_counts": class_counts,
                   "results": [{k: v for k, v in r.items() if k not in ("metrics", "oof")} for r in results]},
                  f, indent=2, default=float)
    labels = [r["recipe"] for r in results]
    macro = [r["macro_f1"] for r in results]
    viz.save_metric_hbar(labels[::-1], macro[::-1], out / "recipe_macro_f1.png",
                         title=f"{args.view}/{args.stage} macro-F1 by recipe", baseline=baseline)
    viz.save_confusion_matrix(y, best["oof"], label_names, out / "best_confusion_matrix.png",
                              title=f"{args.view} best recipe (macro-F1={best['macro_f1']:.3f})")
    viz.save_per_class_bars(best["metrics"]["per_class"], label_names, out / "best_per_class.png",
                            title=f"{args.view} best: {best['recipe']}")
    print(f"\nWrote sweep table + plots to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
