"""Closest publicly-available *medical* backbone: RadImageNet (CNNs trained only on
radiologic images, incl. angiography/X-ray). There is no public angiography *video*
foundation model, so we use RadImageNet 2D per-frame and mean-pool over frames, then run
the same patient-grouped CV + recipe sweep as experiments.py for an apples-to-apples
comparison against frozen V-JEPA 2 (best ~0.46 AP).

    python radimagenet_probe.py --view AP --arch ResNet50 --device cuda --out runs/ap_radimagenet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

import metrics as M
import viz
from experiments import (
    LINEAR_SVM_DUAL,
    LINEAR_SVM_MAX_ITER,
    LINEAR_SVM_TOL,
    grouped_fold_count,
    make_folds,
)
from occlusion_loader import default_base_dir, default_excel_path
from train_dsa_backbone import (
    BackboneReadyDataset, build_base_dataset, build_records, collate_task_samples,
    normalize_sequence, sample_frame_indices, select_device, stage_label_names,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RADIMAGENET_REVISION = "14460ee4c1276f6925611a63aa9a54a05d39eae0"
IDENTITY_METADATA_KEYS = ("accession", "view", "run_column", "study_key")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_cache_metadata(metadata, expected, require_label_text: bool) -> tuple[bool, bool]:
    if len(metadata) != len(expected):
        return False, False
    identity_matches = all(
        all(str(actual.get(key, "")) == str(wanted[key]) for key in IDENTITY_METADATA_KEYS)
        for actual, wanted in zip(metadata, expected)
    )
    label_key_presence = ["label_text" in actual for actual in metadata]
    if any(label_key_presence) and not all(label_key_presence):
        return False, False
    labels_match = bool(metadata) and all(label_key_presence) and all(
        str(actual["label_text"]) == str(wanted["label_text"])
        for actual, wanted in zip(metadata, expected)
    )
    metadata_matches = identity_matches and (
        labels_match if require_label_text else (labels_match or not any(label_key_presence))
    )
    return metadata_matches, labels_match


class _FeatureNet(nn.Module):
    """Wraps a headless CNN so forward returns a flat (B, D) feature vector."""
    def __init__(self, feature: nn.Module):
        super().__init__()
        self.feature = feature

    def forward(self, x):
        return self.feature(x).flatten(1)


def load_radimagenet(arch: str, device, revision: str):
    """RadImageNet .pt is a state_dict of a torchvision backbone (fc removed), keys prefixed
    'backbone.' — e.g. ResNet50 as Sequential(conv1, bn1, relu, maxpool, layer1..4, avgpool)."""
    import torchvision
    path = Path(
        hf_hub_download(
            repo_id="Lab-Rasool/RadImageNet",
            filename=f"{arch}.pt",
            revision=revision,
        )
    )
    checkpoint_hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_hasher.update(chunk)
    parts = path.parts
    resolved_revision = (
        parts[parts.index("snapshots") + 1] if "snapshots" in parts else revision
    )
    sd = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith("backbone.") for k in sd):
        sd = {k.split("backbone.", 1)[1]: v for k, v in sd.items() if k.startswith("backbone.")}
    net = getattr(torchvision.models, arch.lower())(weights=None)
    feature = nn.Sequential(*list(net.children())[:-1])   # drop the classifier -> (B, C, 1, 1)
    res = feature.load_state_dict(sd, strict=False)
    print(f"  loaded {len(sd)} tensors (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")
    if len(res.missing_keys) > 5:
        raise SystemExit(f"RadImageNet {arch} state_dict didn't map cleanly onto torchvision {arch}.")
    provenance = {
        "repository": "Lab-Rasool/RadImageNet",
        "filename": f"{arch}.pt",
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "checkpoint_sha256": checkpoint_hasher.hexdigest(),
    }
    return _FeatureNet(feature).to(device).eval(), provenance


@torch.no_grad()
def extract(
    view,
    stage,
    records,
    model,
    device,
    n_frames,
    out_cache,
    arch,
    model_provenance,
    allow_legacy_cache=False,
):
    cache_dir = Path(out_cache)
    legacy_cache = cache_dir / f"radimagenet_{arch}_{view}_{stage}.npz"
    from train_dsa_backbone import positive_subtype_from_label

    if stage != "positive_subtype":
        raise ValueError("RadImageNet extraction currently supports positive_subtype only.")
    label_names = stage_label_names(stage)
    label_to_index = {name: index for index, name in enumerate(label_names)}
    expected_labels = np.asarray(
        [label_to_index[positive_subtype_from_label(record.label_text)] for record in records]
    )
    expected_groups = np.asarray([record.study_key for record in records], dtype=object)
    expected_meta = [
        {
            "accession": record.accession,
            "view": record.view,
            "run_column": record.run_column,
            "study_key": record.study_key,
            "label_text": record.label_text,
        }
        for record in records
    ]
    record_hasher = hashlib.sha256()
    for record in records:
        path = Path(record.dicom_path)
        stat = path.stat()
        identity = (
            record.study_key,
            record.accession,
            record.run_column,
            str(path.resolve()),
            record.label_text,
            stat.st_size,
            stat.st_mtime_ns,
        )
        record_hasher.update((json.dumps(identity, separators=(",", ":")) + "\n").encode())
    record_sha256 = record_hasher.hexdigest()
    if allow_legacy_cache:
        if n_frames != 16:
            raise ValueError("The historical legacy RadImageNet cache contains 16-frame features.")
        if not legacy_cache.exists():
            raise FileNotFoundError(
                "--allow-legacy-cache audits an existing historical artifact, but it was "
                f"not found at {legacy_cache}. Restore that exact file or omit the flag and "
                "use a separate --out directory for a new signed extraction."
            )
        signature = None
        cache = legacy_cache
    else:
        if model is None or model_provenance is None or device is None:
            raise ValueError("A model, model provenance, and device are required for extraction.")
        signature = {
            "architecture": arch,
            "model": model_provenance,
            "view": view,
            "stage": stage,
            "n_frames": n_frames,
            "record_sha256": record_sha256,
            "image_size": 224,
            "image_mean": IMAGENET_MEAN.flatten().tolist(),
            "image_std": IMAGENET_STD.flatten().tolist(),
            "preprocessing": "dicom_window_to_0_1_bilinear_rgb_imagenet_normalization_v1",
            "metadata_schema": "sample_identity_plus_label_text_v2",
        }
        checkpoint_prefix = model_provenance["checkpoint_sha256"][:12]
        record_prefix = record_sha256[:12]
        cache = cache_dir / (
            f"radimagenet_{arch}_{view}_{stage}_f{n_frames}_ckpt{checkpoint_prefix}_"
            f"data{record_prefix}_prep1_meta2.npz"
        )
    expected_feature_dimension = {"ResNet50": 2048}[arch]
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        required = ("mean", "labels", "groups", "meta")
        has_required_arrays = all(name in d for name in required)
        saved_signature = (
            json.loads(str(d["signature_json"].item())) if "signature_json" in d else None
        )
        legacy_signature_ok = (
            saved_signature is None and n_frames == 16 and allow_legacy_cache
        )
        signature_matches = legacy_signature_ok if allow_legacy_cache else saved_signature == signature
        metadata = list(d["meta"]) if has_required_arrays else []
        metadata_matches, metadata_labels_match = validate_cache_metadata(
            metadata,
            expected_meta,
            require_label_text=not allow_legacy_cache,
        )
        if (
            has_required_arrays
            and np.asarray(d["mean"]).ndim == 2
            and len(d["mean"]) == len(records)
            and d["mean"].shape[1] == expected_feature_dimension
            and np.isfinite(d["mean"]).all()
            and np.array_equal(d["labels"], expected_labels)
            and np.array_equal(d["groups"], expected_groups)
            and metadata_matches
            and signature_matches
        ):
            if saved_signature is None:
                print(
                    "Using identity-validated legacy RadImageNet cache (feature-extraction "
                    "provenance is unavailable): " + str(cache)
                )
            else:
                print(f"Using cached RadImageNet features: {cache}")
            cache_info = {
                "path": str(cache),
                "sha256": file_sha256(cache),
                "signature": saved_signature,
                "legacy_unsigned": saved_signature is None,
                "validation": (
                    "finite features, dimensions, labels, groups, ordered sample identity"
                    + (", and label text" if metadata_labels_match else "")
                ),
            }
            return d["mean"], d["labels"], d["groups"], list(d["meta"]), cache_info
        cache_kind = "historical unsigned" if cache == legacy_cache else "signed"
        raise RuntimeError(
            f"The {cache_kind} RadImageNet cache failed validation; refusing to overwrite "
            f"it: {cache}. Use a separate --out directory for a new extraction."
        )
    # iterate the base dataset (gives raw DICOM sequences + metadata) directly
    base = build_base_dataset(view, records)
    mean_mn, mean_std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    feats, labels, groups, meta = [], [], [], []
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
        labels.append(label_to_index[positive_subtype_from_label(sample["label_text"])])
        groups.append(sample["study_key"])
        meta.append(
            {
                key: sample.get(key, "")
                for key in ("accession", "view", "run_column", "study_key", "label_text")
            }
        )
        print(f"  radimagenet {i+1}/{len(base)}", end="\r", flush=True)
    print()
    X = np.stack(feats)
    y = np.array(labels)
    g = np.array(groups, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        mean=X,
        labels=y,
        groups=g,
        meta=np.array(meta, dtype=object),
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
    )
    cache_info = {
        "path": str(cache),
        "sha256": file_sha256(cache),
        "signature": signature,
        "legacy_unsigned": False,
        "validation": "newly extracted from the recorded signed protocol",
    }
    return X, y, g, meta, cache_info


RECIPES = {
    "raw / logreg-noW": (lambda: LogisticRegression(max_iter=5000)),
    "std / logreg-bal": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))),
    "std / linsvm-bal": (
        lambda: make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.5,
                class_weight="balanced",
                dual=LINEAR_SVM_DUAL,
                max_iter=LINEAR_SVM_MAX_ITER,
                random_state=0,
                tol=LINEAR_SVM_TOL,
            ),
        )
    ),
    "l2  / logreg-bal": (lambda: make_pipeline(Normalizer(), LogisticRegression(max_iter=5000, class_weight="balanced"))),
}


def cv_eval(X, y, groups, k, folds, seed):
    n = grouped_fold_count(y, groups, folds, k)
    grouped_folds = make_folds(y, groups, n, seed)
    results = {}
    for name, factory in RECIPES.items():
        oof = np.full(len(y), -1)
        for fold_index, (tr, va) in enumerate(grouped_folds):
            clf = factory()
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    clf.fit(X[tr], y[tr])
                except ConvergenceWarning as error:
                    raise RuntimeError(
                        f"{name} did not converge in fold {fold_index} for seed {seed}."
                    ) from error
            oof[va] = clf.predict(X[va])
        results[name] = (M.compute_metrics(y, oof, k), oof)
    return results, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=("AP", "Lateral"), required=True)
    ap.add_argument("--stage", default="positive_subtype")
    ap.add_argument(
        "--arch",
        default="ResNet50",
        choices=("ResNet50",),
        help="Validated RadImageNet architecture (the legacy generic DenseNet/Inception "
        "wrapping was not architecture-correct).",
    )
    ap.add_argument(
        "--revision",
        default=RADIMAGENET_REVISION,
        help="Immutable Lab-Rasool/RadImageNet Hub revision.",
    )
    ap.add_argument("--excel", default=str(default_excel_path()))
    ap.add_argument("--base-dir", default=str(default_base_dir()))
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="Explicitly reuse the historical unsigned 16-frame cache after validating its "
        "current labels, groups, and sample identities. Its original model/data provenance "
        "cannot be reconstructed; no current checkpoint is downloaded or loaded.",
    )
    ap.add_argument("--out", default="runs/radimagenet")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    records = build_records(types.SimpleNamespace(
        excel=args.excel, base_dir=args.base_dir, view=args.view, stage=args.stage,
        treat_blank_as_negative=False))
    label_names = stage_label_names(args.stage)
    k = len(label_names)

    if args.allow_legacy_cache:
        device = None
        model = None
        model_provenance = None
        print(
            f"Auditing historical RadImageNet cache for {len(records)} {args.view} runs "
            "without loading a checkpoint…"
        )
    else:
        device = select_device(args.device)
        print(f"Loading RadImageNet {args.arch}…")
        model, model_provenance = load_radimagenet(args.arch, device, args.revision)
        print(f"Extracting per-frame RadImageNet features for {len(records)} {args.view} runs…")
    X, y, groups, meta, cache_info = extract(
        args.view,
        args.stage,
        records,
        model,
        device,
        args.n_frames,
        out / "cache",
        args.arch,
        model_provenance,
        args.allow_legacy_cache,
    )
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
    provenance_note = (
        " Historical unsigned cache reuse was explicitly allowed; original extraction "
        "provenance is unavailable."
        if cache_info["legacy_unsigned"]
        else ""
    )
    checkpoint_note = (
        f"Current checkpoint: `{model_provenance['repository']}` revision "
        f"`{model_provenance['resolved_revision']}`, SHA-256 "
        f"`{model_provenance['checkpoint_sha256']}`."
        if model_provenance is not None
        else "No current checkpoint was loaded for this legacy-cache audit."
    )
    (out / "report.md").write_text(
        f"RadImageNet {args.arch} {args.view}: best macro-F1 {best:.3f} "
        f"({best_name}); baseline {base:.3f}; V-JEPA2 ~0.46.{provenance_note}\n\n"
        f"{checkpoint_note}\n"
    )
    structured_results = {
        "protocol": {
            "view": args.view,
            "stage": args.stage,
            "architecture": args.arch,
            "seed": args.seed,
            "requested_folds": args.folds,
            "actual_folds": n_splits,
            "group": "Study_Key",
            "n_frames": args.n_frames,
            "linear_svm_max_iterations": LINEAR_SVM_MAX_ITER,
            "linear_svm_dual": LINEAR_SVM_DUAL,
            "linear_svm_tolerance": LINEAR_SVM_TOL,
            "convergence_policy": "fail on any scikit-learn ConvergenceWarning",
            "script_sha256": file_sha256(Path(__file__)),
            "current_model_provenance": model_provenance,
            "feature_cache": cache_info,
            "provenance_note": provenance_note.strip() or None,
        },
        "data": {
            "n_samples": len(y),
            "n_groups": len(set(groups)),
            "class_names": label_names,
            "class_counts": counts,
            "baseline_macro_f1": base,
        },
        "recipes": {
            name: {"metrics": metrics, "oof_predictions": oof.tolist()}
            for name, (metrics, oof) in results.items()
        },
        "best_recipe": best_name,
    }
    with (out / "results.json").open("w") as handle:
        json.dump(structured_results, handle, indent=2)
    print(f"\nWrote to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
