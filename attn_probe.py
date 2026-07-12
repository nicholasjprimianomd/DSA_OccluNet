"""Attentive-pooling probe over frozen V-JEPA 2 *token* features.

Diagnostics showed mean-pooling preserves global layout (L/R is easy) but loses the local
branch-level detail that separates M2 from M3. This probe keeps the full token sequence and
learns a lightweight attention pooling (a single scalar score per token) instead of averaging
— the literature-recommended lever — then a class-weighted linear head. Minimal params, so it
can survive ~290 training samples. Same patient-grouped folds as everything else.

    python attn_probe.py --view AP --stage positive_subtype --device cuda --amp --out runs/ap_attn
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

import metrics as M
import viz
from occlusion_loader import default_base_dir, default_excel_path
from train_dsa_backbone import (
    BACKBONE, BackboneReadyDataset, DsaVideoClassifier, build_base_dataset,
    build_records, collate_task_samples, select_device, stage_label_names,
)


def extract_tokens(view, stage, records, device, amp, batch_size, num_workers, cache):
    norm_tag = "_norm" if BACKBONE.normalize_input else "_raw"
    revision_tag = f"_{BACKBONE.revision[:8]}" if BACKBONE.revision else ""
    path = Path(cache) / f"tokens_{view}_{stage}{norm_tag}{revision_tag}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if len(d["labels"]) == len(records):
            print(f"Using cached token features: {path}")
            return d["tokens"], d["labels"], d["groups"], list(d["meta"])
    dataset = BackboneReadyDataset(build_base_dataset(view, records), stage, stage_label_names(stage))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_task_samples)
    model = DsaVideoClassifier(num_labels=len(stage_label_names(stage)), freeze_backbone=True).to(device).eval()
    toks, labels, groups, meta = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                model_inputs = {model.spec.input_key: batch["pixel_values"].to(device)}
                if model.spec.name == "vjepa2":
                    model_inputs["skip_predictor"] = True
                out = model.backbone(**model_inputs)
            toks.append(out.last_hidden_state.half().cpu())
            labels.append(batch["labels"])
            for m in batch["metadata"]:
                groups.append(m["study_key"])
                meta.append({k: m.get(k, "") for k in ("accession", "view", "run_column", "study_key", "label_text")})
            print(f"  tokens {min(i*batch_size, len(dataset))}/{len(dataset)}", end="\r", flush=True)
    print()
    tokens = torch.cat(toks).numpy()
    y = torch.cat(labels).numpy()
    g = np.array(groups, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tokens=tokens, labels=y, groups=g, meta=np.array(meta, dtype=object))
    return tokens, y, g, meta


class AttnPool(nn.Module):
    """Attention pooling: one learned score per token -> weighted sum -> linear head."""
    def __init__(self, dim, num_classes, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, tokens):                       # (B, Tk, D)
        t = self.norm(tokens)
        attn = self.score(t).squeeze(-1).softmax(dim=1)   # (B, Tk)
        ctx = (attn.unsqueeze(-1) * tokens).sum(dim=1)    # (B, D)
        return self.head(ctx)


def inv_freq_weights(y, k):
    c = np.bincount(y, minlength=k).astype(np.float64)
    return torch.tensor(np.where(c > 0, c.sum() / (k * np.maximum(c, 1)), 0.0), dtype=torch.float32)


def train_attn(Xtr, ytr, k, device, epochs, lr, wd, bs, seed):
    torch.manual_seed(seed)
    model = AttnPool(Xtr.shape[-1], k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cw = inv_freq_weights(ytr, k).to(device)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(yt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_attn(model, X, device, bs=64):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    preds = []
    for s in range(0, len(Xt), bs):
        preds.append(model(Xt[s:s + bs].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view", choices=("AP", "Lateral"), required=True)
    p.add_argument("--stage", default="positive_subtype")
    p.add_argument("--excel", default=str(default_excel_path()))
    p.add_argument("--base-dir", default=str(default_base_dir()))
    p.add_argument("--treat-blank-as-negative", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--head-batch", type=int, default=32)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/attn")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    rec_args = types.SimpleNamespace(excel=args.excel, base_dir=args.base_dir, view=args.view,
                                     stage=args.stage, treat_blank_as_negative=args.treat_blank_as_negative)
    records = build_records(rec_args)
    label_names = stage_label_names(args.stage)
    k = len(label_names)

    print(f"Extracting token features for {len(records)} {args.view} runs (frozen {BACKBONE.pretrained_name})…")
    X, y, groups, meta = extract_tokens(args.view, args.stage, records, device, args.amp,
                                        args.batch_size, args.num_workers, out / "cache")
    counts = {label_names[c]: int((y == c).sum()) for c in range(k)}
    n_splits = min(args.folds, len(set(groups)), int(np.min(np.bincount(y, minlength=k))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    oof = np.full(len(y), -1)
    per_fold = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y, groups), 1):
        model = train_attn(X[tr], y[tr], k, device, args.epochs, args.lr, args.weight_decay, args.head_batch, args.seed)
        oof[va] = predict_attn(model, X[va], device)
        fm = M.compute_metrics(y[va], oof[va], k)
        per_fold.append({"fold": fold, "n_train": len(tr), "n_val": len(va),
                         "macro_f1": fm["macro_f1"], "balanced_accuracy": fm["balanced_accuracy"],
                         "accuracy": fm["accuracy"]})
        print(f"  fold {fold}: macro_f1={fm['macro_f1']:.3f}")

    m = M.compute_metrics(y, oof, k)
    base = M.baseline_macro_f1(y, k)
    cfg = {"view": args.view, "stage": args.stage, "backbone": BACKBONE.pretrained_name,
           "backbone_revision": BACKBONE.revision, "normalize_input": BACKBONE.normalize_input, "frozen": True,
           "probe": "attention-pooling", "folds": n_splits, "n_samples": len(y), "n_groups": len(set(groups))}
    report = M.format_cv_report(cfg, counts, per_fold, m, label_names)
    report += f"\n\nMajority baseline macro-F1={base:.3f}  |  prior best linear probe ≈0.46 AP / 0.45 Lat"
    print("\n" + report)
    (out / "report.md").write_text("```\n" + report + "\n```\n")
    viz.save_confusion_matrix(y, oof, label_names, out / "confusion_matrix.png",
                              title=f"{args.view} attn-pool (macro-F1={m['macro_f1']:.3f})")
    viz.save_per_class_bars(m["per_class"], label_names, out / "per_class.png",
                            title=f"{args.view} attention-pooling per-class")
    print(f"\nWrote report + plots to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
