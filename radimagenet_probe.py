"""Closest publicly-available *medical* backbone: RadImageNet (CNNs trained only on
radiologic images, incl. angiography/X-ray). There is no public angiography *video*
foundation model, so we use RadImageNet 2D per-frame and mean-pool over frames, then run
the same patient-grouped CV + recipe sweep as experiments.py for an apples-to-apples
comparison against frozen V-JEPA 2 (best ~0.46 AP).

    python radimagenet_probe.py --view AP --arch ResNet50 --device cuda --out runs/ap_radimagenet
"""
from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

import metrics as M
import viz
from occlusion_loader import default_base_dir, default_excel_path
from train_dsa_backbone import (
    BackboneReadyDataset, build_base_dataset, build_records, collate_task_samples,
    normalize_sequence, sample_frame_indices, select_device, stage_label_names,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _FeatureNet(nn.Module):
    """Wraps a headless CNN so forward returns a flat (B, D) feature vector."""
    def __init__(self, feature: nn.Module):
        super().__init__()
        self.feature = feature

    def forward(self, x):
        return self.feature(x).flatten(1)


def load_radimagenet(arch: str, device):
    """RadImageNet .pt is a state_dict of a torchvision backbone (fc removed), keys prefixed
    'backbone.' — e.g. ResNet50 as Sequential(conv1, bn1, relu, maxpool, layer1..4, avgpool)."""
    import torchvision
    path = hf_hub_download(repo_id="Lab-Rasool/RadImageNet", filename=f"{arch}.pt")
    sd = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith("backbone.") for k in sd):
        sd = {k.split("backbone.", 1)[1]: v for k, v in sd.items() if k.startswith("backbone.")}
    net = getattr(torchvision.models, arch.lower())(weights=None)
    feature = nn.Sequential(*list(net.children())[:-1])   # drop the classifier -> (B, C, 1, 1)
    res = feature.load_state_dict(sd, strict=False)
    print(f"  loaded {len(sd)} tensors (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")
    if len(res.missing_keys) > 5:
        raise SystemExit(f"RadImageNet {arch} state_dict didn't map cleanly onto torchvision {arch}.")
    return _FeatureNet(feature).to(device).eval()


@torch.no_grad()
def extract(view, stage, records, model, device, n_frames, out_cache, arch):
    cache = Path(out_cache) / f"radimagenet_{arch}_{view}_{stage}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if len(d["labels"]) == len(records):
            print(f"Using cached RadImageNet features: {cache}")
            return d["mean"], d["labels"], d["groups"], list(d["meta"])
    # iterate the base dataset (gives raw DICOM sequences + metadata) directly
    base = build_base_dataset(view, records)
    mean_mn, mean_std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    feats, labels, groups, meta = [], [], [], []
    from train_dsa_backbone import positive_subtype_from_label
    label_names = stage_label_names(stage)
    l2i = {n: i for i, n in enumerate(label_names)}
    for i in range(len(base)):
        sample = base[i]
        seq = sample["sequence"]
        seq = seq if isinstance(seq, torch.Tensor) else torch.as_tensor(seq)
        seq = normalize_sequence(seq)                                  # (T,H,W) in [0,1]
        idx = sample_frame_indices(seq.shape[0], n_frames)
        frames = seq.index_select(0, idx).unsqueeze(1)                 # (n,1,H,W)
        frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        frames = frames.repeat(1, 3, 1, 1).to(device)                 # (n,3,224,224)
        frames = (frames - mean_mn) / mean_std
        f = model(frames)                                             # (n, D)
        feats.append(f.mean(0).float().cpu().numpy())                 # temporal mean-pool
        labels.append(l2i[positive_subtype_from_label(sample["label_text"])])
        groups.append(sample["study_key"])
        meta.append({k: sample.get(k, "") for k in ("accession", "view", "run_column", "study_key")})
        print(f"  radimagenet {i+1}/{len(base)}", end="\r", flush=True)
    print()
    X = np.stack(feats)
    y = np.array(labels)
    g = np.array(groups, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mean=X, labels=y, groups=g, meta=np.array(meta, dtype=object))
    return X, y, g, meta


RECIPES = {
    "raw / logreg-noW": (lambda: LogisticRegression(max_iter=5000)),
    "std / logreg-bal": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))),
    "std / linsvm-bal": (lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.5, class_weight="balanced", max_iter=20000))),
    "l2  / logreg-bal": (lambda: make_pipeline(Normalizer(), LogisticRegression(max_iter=5000, class_weight="balanced"))),
}


def cv_eval(X, y, groups, k, folds, seed):
    n = min(folds, len(set(groups)), int(np.min(np.bincount(y, minlength=k))))
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    results = {}
    for name, factory in RECIPES.items():
        oof = np.full(len(y), -1)
        for tr, va in skf.split(X, y, groups):
            clf = factory()
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict(X[va])
        results[name] = (M.compute_metrics(y, oof, k), oof)
    return results, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=("AP", "Lateral"), required=True)
    ap.add_argument("--stage", default="positive_subtype")
    ap.add_argument("--arch", default="ResNet50", choices=("ResNet50", "DenseNet121", "InceptionV3"))
    ap.add_argument("--excel", default=str(default_excel_path()))
    ap.add_argument("--base-dir", default=str(default_base_dir()))
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/radimagenet")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    records = build_records(types.SimpleNamespace(
        excel=args.excel, base_dir=args.base_dir, view=args.view, stage=args.stage,
        treat_blank_as_negative=False))
    label_names = stage_label_names(args.stage)
    k = len(label_names)

    print(f"Loading RadImageNet {args.arch}…")
    model = load_radimagenet(args.arch, device)
    print(f"Extracting per-frame RadImageNet features for {len(records)} {args.view} runs…")
    X, y, groups, meta = extract(args.view, args.stage, records, model, device, args.n_frames, out / "cache", args.arch)
    print(f"feature dim = {X.shape[1]}")
    counts = {label_names[c]: int((y == c).sum()) for c in range(k)}
    base = M.baseline_macro_f1(y, k)

    results, n_splits = cv_eval(X, y, groups, k, args.folds, args.seed)
    print(f"\nRadImageNet {args.arch}, {args.view}/{args.stage}  "
          f"({len(y)} runs, {len(set(groups))} patients, {n_splits} folds)  classes {counts}")
    print(f"baseline macro-F1={base:.3f}  |  frozen V-JEPA 2 best ≈0.46 AP / 0.45 Lat\n")
    w = [20, 9, 8, 8] + [10] * k
    print("  ".join(str(c).ljust(x) for c, x in zip(["recipe", "macroF1", "bal_acc", "acc"] + [f"F1:{n[:6]}" for n in label_names], w)))
    print("-" * (sum(w) + 2 * len(w)))
    best_name, best = None, -1
    for name, (m, oof) in sorted(results.items(), key=lambda kv: kv[1][0]["macro_f1"], reverse=True):
        row = [name, f"{m['macro_f1']:.3f}", f"{m['balanced_accuracy']:.3f}", f"{m['accuracy']:.3f}"] + [f"{f:.3f}" for f in m["per_class"]["f1"]]
        print("  ".join(str(c).ljust(x) for c, x in zip(row, w)))
        if m["macro_f1"] > best:
            best_name, best, best_oof, best_m = name, m["macro_f1"], oof, m
    print(f"\nBest: {best_name}  macro-F1 {best:.3f}")
    print(M.format_per_class(best_m, label_names))
    viz.save_confusion_matrix(y, best_oof, label_names, out / "confusion_matrix.png",
                              title=f"RadImageNet {args.arch} {args.view} (macro-F1={best:.3f})")
    (out / "report.md").write_text(f"RadImageNet {args.arch} {args.view}: best macro-F1 {best:.3f} "
                                   f"({best_name}); baseline {base:.3f}; V-JEPA2 ~0.46.\n")
    print(f"\nWrote to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
