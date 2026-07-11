"""Patient-grouped, stratified k-fold cross-validation for the DSA subtype models.

Why this and not train_dsa_backbone.py's single split: with < 400 samples a single
train/val split is too noisy to trust, and runs can leak a patient's multiple angiographic
runs across train and val. This harness:

  1. Extracts V-JEPA 2 features ONCE per clip (the backbone is frozen, so features are
     deterministic) and caches them — so k-fold over the classifier head is fast.
  2. Splits with StratifiedGroupKFold grouped on Study_Key, so every run of a patient
     stays on one side of each fold, and class balance is preserved per fold.
  3. Trains a fresh class-weighted linear probe per fold, collects out-of-fold predictions,
     and reports macro-F1 / balanced accuracy / per-class precision-recall-F1 / confusion
     matrix — printed readably and saved as PNG/CSV/JSON/Markdown.

Example:
    python cross_validate.py --view AP --stage positive_subtype \
        --folds 5 --device cuda --amp --out runs/ap_cv
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

import metrics as M
import viz
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
from occlusion_loader import default_base_dir, default_excel_path


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda")


def extract_features(view, stage, records, device, amp, batch_size, num_workers):
    """Run each clip through the frozen backbone once -> (features, labels, groups, meta)."""
    dataset = BackboneReadyDataset(build_base_dataset(view, records), stage, stage_label_names(stage))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_task_samples)
    model = DsaVideoClassifier(num_labels=len(stage_label_names(stage)), freeze_backbone=True).to(device)
    model.eval()

    feats, labels, groups, meta = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            with _autocast(device, amp):
                _, f = model(batch["pixel_values"].to(device), return_features=True)
            feats.append(f.float().cpu())
            labels.append(batch["labels"])
            for m in batch["metadata"]:
                groups.append(m["study_key"])
                meta.append({k: m.get(k, "") for k in ("accession", "view", "run_column", "study_key", "label_text")})
            print(f"  features {min(i * batch_size, len(dataset))}/{len(dataset)}", end="\r", flush=True)
    print()
    return (torch.cat(feats).numpy(), torch.cat(labels).numpy(), np.array(groups, dtype=object), meta)


def load_or_extract(cache_dir, view, stage, records, device, amp, batch_size, num_workers):
    cache = Path(cache_dir) / f"features_{view}_{stage}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if len(data["labels"]) == len(records):
            print(f"Using cached features: {cache}")
            return data["features"], data["labels"], data["groups"], list(data["meta"])
    X, y, g, meta = extract_features(view, stage, records, device, amp, batch_size, num_workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=X, labels=y, groups=g, meta=np.array(meta, dtype=object))
    return X, y, g, meta


def inverse_freq_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    w = np.where(counts > 0, total / (num_classes * np.maximum(counts, 1)), 0.0)
    return w.astype(np.float32)


def train_probe(Xtr, ytr, num_classes, device, epochs, lr, seed):
    """Full-batch class-weighted linear probe on frozen features (tiny + fast)."""
    torch.manual_seed(seed)
    clf = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    cw = torch.tensor(inverse_freq_weights(ytr, num_classes), device=device)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xt), yt, weight=cw)
        loss.backward()
        opt.step()
    return clf


def predict(clf, X, device):
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.tensor(X, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs.argmax(axis=1), probs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", choices=("AP", "Lateral"), required=True)
    p.add_argument("--stage", default="positive_subtype", choices=("positive_subtype", "binary_detection"))
    p.add_argument("--excel", default=str(default_excel_path()))
    p.add_argument("--base-dir", default=str(default_base_dir()))
    p.add_argument("--treat-blank-as-negative", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--head-epochs", type=int, default=300, help="Probe training epochs per fold.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/cv", help="Directory for report, plots, predictions, cache.")
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

    print(f"Extracting features for {len(records)} {args.view} runs (frozen {BACKBONE.pretrained_name})…")
    X, y, groups, meta = load_or_extract(out / "cache", args.view, args.stage, records,
                                         device, args.amp, args.batch_size, args.num_workers)
    class_counts = {label_names[c]: int((y == c).sum()) for c in range(num_classes)}
    n_groups = len(set(groups))

    n_splits = min(args.folds, n_groups, int(np.min(np.bincount(y, minlength=num_classes))))
    if n_splits < 2:
        raise SystemExit("Not enough per-class samples/patients for cross-validation.")
    if n_splits != args.folds:
        print(f"Reducing folds {args.folds} -> {n_splits} (limited by rarest class / patient count).")

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_prob = np.zeros((len(y), num_classes), dtype=float)
    per_fold = []

    for fold, (tr, va) in enumerate(skf.split(X, y, groups), 1):
        clf = train_probe(X[tr], y[tr], num_classes, device, args.head_epochs, args.lr, args.seed)
        pred, prob = predict(clf, X[va], device)
        oof_pred[va], oof_prob[va] = pred, prob
        fm = M.compute_metrics(y[va], pred, num_classes)
        per_fold.append({"fold": fold, "n_train": len(tr), "n_val": len(va),
                         "macro_f1": fm["macro_f1"], "balanced_accuracy": fm["balanced_accuracy"],
                         "accuracy": fm["accuracy"]})

    oof = M.compute_metrics(y, oof_pred, num_classes)
    baseline = M.baseline_macro_f1(y, num_classes)

    config = {"view": args.view, "stage": args.stage, "backbone": BACKBONE.pretrained_name,
              "frozen": True, "probe": "linear", "folds": n_splits,
              "n_samples": len(y), "n_groups": n_groups}
    report = M.format_cv_report(config, class_counts, per_fold, oof, label_names)
    report += (f"\n\nMajority-class baseline macro_f1 = {baseline:.3f}  "
               f"(model {'BEATS' if oof['macro_f1'] > baseline else 'does NOT beat'} it)")
    print("\n" + report)

    # Artifacts
    (out / "report.md").write_text("```\n" + report + "\n```\n")
    with (out / "metrics.json").open("w") as f:
        json.dump({"config": config, "class_counts": class_counts, "per_fold": per_fold,
                   "out_of_fold": oof, "baseline_macro_f1": baseline}, f, indent=2)
    viz.save_confusion_matrix(y, oof_pred, label_names, out / "confusion_matrix.png",
                              title=f"{args.view}/{args.stage} out-of-fold (macro-F1={oof['macro_f1']:.3f})")
    viz.save_per_class_bars(oof["per_class"], label_names, out / "per_class_metrics.png",
                            title=f"{args.view}/{args.stage} out-of-fold per-class")
    viz.save_embedding_scatter(X, y, label_names, out / "feature_pca.png",
                               title=f"{args.view} V-JEPA2 features (PCA), colored by subtype")
    viz.save_predictions_csv(y, oof_pred, oof_prob, label_names, meta, out / "oof_predictions.csv")
    print(f"\nWrote report + plots + predictions to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
