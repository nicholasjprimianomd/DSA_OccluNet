"""Partial fine-tuning of V-JEPA 2 for DSA subtype — the only lever left that can change
the *features* (experiments showed frozen features can't separate m2 from m3).

Design against overfitting on ~290 training samples:
  - unfreeze only the last N encoder blocks + final layernorm + the head (everything else frozen)
  - discriminative LRs: a tiny LR on the backbone, a normal LR on the head
  - class-weighted loss, weight decay, dropout on the head, few epochs
  - patient-grouped k-fold with CLEAN held-out folds (no early-stopping leakage), and
    per-epoch train-vs-val curves so overfitting is visible at a glance

Reported honestly: `best-val` (epoch picked per fold on the held-out fold — optimistic) AND
`final-epoch` (no peeking). If final-epoch doesn't beat the frozen probe, fine-tuning isn't
helping at this data scale.

    python finetune.py --view AP --stage positive_subtype --unfreeze-blocks 2 \
        --epochs 10 --device cuda --amp --out runs/ap_ft
"""
from __future__ import annotations

import argparse
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
from occlusion_loader import default_base_dir, default_excel_path
from train_dsa_backbone import (
    BACKBONE, BackboneReadyDataset, DsaVideoClassifier, build_base_dataset,
    build_records, collate_task_samples, select_device, stage_label_names,
)


def preprocess_all(view, stage, records, batch_size, num_workers):
    """Preprocess every clip once into a CPU fp16 tensor so fine-tuning epochs don't re-read DICOMs."""
    ds = BackboneReadyDataset(build_base_dataset(view, records), stage, stage_label_names(stage))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_task_samples)
    clips, labels, groups = [], [], []
    for i, batch in enumerate(loader, 1):
        clips.append(batch["pixel_values"].half())
        labels.append(batch["labels"])
        groups.extend(m["study_key"] for m in batch["metadata"])
        print(f"  preprocess {min(i*batch_size, len(ds))}/{len(ds)}", end="\r", flush=True)
    print()
    return torch.cat(clips), torch.cat(labels).numpy(), np.array(groups, dtype=object)


def build_model(num_classes, unfreeze_blocks, dropout, device):
    model = DsaVideoClassifier(num_labels=num_classes, freeze_backbone=True).to(device)
    hidden = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, num_classes)
    ).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    enc = model.backbone.encoder
    for p in enc.layernorm.parameters():
        p.requires_grad_(True)
    n = len(enc.layer)
    for i in range(max(0, n - unfreeze_blocks), n):
        for p in enc.layer[i].parameters():
            p.requires_grad_(True)
    return model


def param_groups(model, backbone_lr, head_lr):
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}]


def inv_freq_weights(y, k):
    c = np.bincount(y, minlength=k).astype(np.float64)
    return torch.tensor(np.where(c > 0, c.sum() / (k * np.maximum(c, 1)), 0.0), dtype=torch.float32)


def run_epoch(model, X, y, idx, device, amp, cw, opt=None, bs=4):
    train = opt is not None
    model.train(train)
    order = np.random.permutation(idx) if train else idx
    logits_all, loss_sum = [], 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            xb = X[b].to(device).float()
            yb = torch.tensor(y[b], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=cw)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
            loss_sum += loss.item() * len(b)
            logits_all.append((b, logits.detach().float().argmax(1).cpu().numpy()))
    preds = np.empty(len(y), dtype=int)
    for b, p in logits_all:
        preds[b] = p
    return preds, loss_sum / len(order)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", choices=("AP", "Lateral"), required=True)
    p.add_argument("--stage", default="positive_subtype")
    p.add_argument("--excel", default=str(default_excel_path()))
    p.add_argument("--base-dir", default=str(default_base_dir()))
    p.add_argument("--treat-blank-as-negative", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--unfreeze-blocks", type=int, default=2, help="How many final encoder blocks to unfreeze.")
    p.add_argument("--warmup-epochs", type=int, default=3, help="Head-only epochs before unfreezing the backbone.")
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--frozen-ref", type=float, default=0.46, help="Frozen-probe macro-F1 to beat (for the plot).")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ft")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.device)

    records = build_records(types.SimpleNamespace(
        excel=args.excel, base_dir=args.base_dir, view=args.view, stage=args.stage,
        treat_blank_as_negative=args.treat_blank_as_negative))
    label_names = stage_label_names(args.stage)
    k = len(label_names)

    print(f"Preprocessing {len(records)} {args.view} clips once…")
    X, y, groups = preprocess_all(args.view, args.stage, records, args.batch_size, args.num_workers)
    counts = {label_names[c]: int((y == c).sum()) for c in range(k)}
    n_splits = min(args.folds, len(set(groups)), int(np.min(np.bincount(y, minlength=k))))
    print(f"{len(y)} clips, {len(set(groups))} patients, {n_splits} folds, classes {counts}")
    print(f"Unfreezing last {args.unfreeze_blocks} of 24 encoder blocks + final norm + head "
          f"(backbone_lr={args.backbone_lr}, head_lr={args.head_lr}, wd={args.weight_decay}, dropout={args.dropout})\n")

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    oof_best = np.full(len(y), -1, dtype=int)
    oof_final = np.full(len(y), -1, dtype=int)
    per_fold, curve_tr, curve_va = [], np.zeros(args.epochs), np.zeros(args.epochs)

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y, groups), 1):
        model = build_model(k, args.unfreeze_blocks, args.dropout, device)
        opt = AdamW(param_groups(model, args.backbone_lr, args.head_lr), weight_decay=args.weight_decay)
        cw = inv_freq_weights(y[tr], k).to(device)
        best_f1, best_pred = -1.0, None
        for ep in range(args.epochs):
            # Head-only warmup: keep the backbone frozen (lr=0) for the first epochs, then unfreeze.
            opt.param_groups[0]["lr"] = 0.0 if ep < args.warmup_epochs else args.backbone_lr
            tr_pred, tr_loss = run_epoch(model, X, y, tr, device, args.amp, cw, opt, args.batch_size)
            va_pred, _ = run_epoch(model, X, y, va, device, args.amp, cw, None, args.batch_size)
            tr_acc = float((tr_pred[tr] == y[tr]).mean())
            va_f1 = M.compute_metrics(y[va], va_pred[va], k)["macro_f1"]
            curve_tr[ep] += tr_acc / n_splits
            curve_va[ep] += va_f1 / n_splits
            if va_f1 > best_f1:
                best_f1, best_pred = va_f1, va_pred[va].copy()
            print(f"  fold {fold} ep {ep+1}/{args.epochs}: train_acc={tr_acc:.3f}  val_macroF1={va_f1:.3f}", flush=True)
        oof_best[va] = best_pred
        oof_final[va] = va_pred[va]
        per_fold.append({"fold": fold, "n_train": len(tr), "n_val": len(va),
                         "best_val_macro_f1": best_f1,
                         "final_val_macro_f1": M.compute_metrics(y[va], va_pred[va], k)["macro_f1"]})

    m_best = M.compute_metrics(y, oof_best, k)
    m_final = M.compute_metrics(y, oof_final, k)
    base = M.baseline_macro_f1(y, k)

    lines = ["=" * 66, f"Partial fine-tune — {args.view}/{args.stage} "
             f"(last {args.unfreeze_blocks} blocks, {n_splits}-fold)", "=" * 66]
    for f in per_fold:
        lines.append(f"  fold {f['fold']}: best_val={f['best_val_macro_f1']:.3f}  final_val={f['final_val_macro_f1']:.3f}")
    lines.append("")
    lines.append(f"OOF macro-F1  best-epoch (optimistic) = {m_best['macro_f1']:.3f}")
    lines.append(f"OOF macro-F1  final-epoch (no peeking) = {m_final['macro_f1']:.3f}")
    lines.append(f"frozen linear probe reference          = {args.frozen_ref:.3f}   |  baseline = {base:.3f}")
    lines.append("")
    lines.append("Final-epoch per-class:")
    lines.append(M.format_per_class(m_final, label_names))
    verdict = ("HELPS" if m_final["macro_f1"] > args.frozen_ref + 0.01 else
               "OVERFITS / no gain" if curve_tr[-1] - curve_va[-1] > 0.25 else "no clear gain")
    lines.append(f"\nVerdict: fine-tuning {verdict} vs the frozen probe.")
    report = "\n".join(lines)
    print("\n" + report)

    (out / "report.md").write_text("```\n" + report + "\n```\n")
    viz.save_training_curves(list(range(1, args.epochs + 1)), curve_tr.tolist(), curve_va.tolist(),
                             out / "training_curves.png", frozen_ref=args.frozen_ref,
                             title=f"{args.view} fine-tune: train acc vs held-out val macro-F1 (mean over folds)")
    viz.save_confusion_matrix(y, oof_final, label_names, out / "confusion_matrix.png",
                              title=f"{args.view} fine-tune final-epoch (macro-F1={m_final['macro_f1']:.3f})")
    viz.save_per_class_bars(m_final["per_class"], label_names, out / "per_class.png",
                            title=f"{args.view} fine-tune final-epoch per-class")
    print(f"\nWrote report + curves + plots to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
