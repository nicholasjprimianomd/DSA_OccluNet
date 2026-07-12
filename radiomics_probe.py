"""Extract hand-crafted radiomics texture features from DSA runs.

This is the "old-school" counterpart to ``image_backbone_probe.py``.  Instead of
a frozen neural backbone it computes PyRadiomics first-order and texture features
(GLCM / GLRLM / GLSZM / GLDM / NGTDM, and 2-D shape when a real ROI exists) per
sampled frame, then pools those features over time.  The frame sampling, DICOM
loading, percentile windowing, label mapping, and cache format are deliberately
identical to the neural probes so the resulting features drop straight into
``compare_feature_caches.py`` and ``ensemble_experiments.py`` on the *same*
patient-grouped folds.

The radiomics features are a genuinely different representation from the deep
features (hand-crafted texture statistics, not learned embeddings), which is
exactly what makes them a candidate second member of an ensemble.

PyRadiomics + SimpleITK are imported lazily, so ``--dry-run`` and the unit tests
work without them installed (mirroring how the neural probes avoid importing
``transformers`` until a real forward pass is needed).

Example:
    python radiomics_probe.py --view AP --stage positive_subtype \
        --n-frames 16 --roi otsu --out runs/ap_radiomics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

import numpy as np

import metrics as M
from experiments import RECIPES, evaluate_recipe, make_folds, representation
from occlusion_loader import default_base_dir, default_excel_path
from train_dsa_backbone import (
    binary_target_from_label,
    build_base_dataset,
    build_records,
    normalize_sequence,
    positive_subtype_from_label,
    sample_frame_indices,
    select_device,
    stage_label_names,
)

# Feature classes enabled on every frame.  ``shape2D`` is only meaningful when the
# ROI is not the whole frame, so it is enabled dynamically per-frame below.
TEXTURE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
ROI_CHOICES = ("otsu", "full")


def build_extractor(bin_count: int):
    """Create a PyRadiomics 2-D extractor (imported lazily)."""
    from radiomics import featureextractor

    settings = {
        "binCount": int(bin_count),
        "force2D": True,
        "force2Ddimension": 0,
        "label": 1,
        "correctMask": True,
        "geometryTolerance": 1e-4,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in TEXTURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)
    extractor.enableFeatureClassByName("shape2D")
    return extractor


def frame_to_uint(frame: np.ndarray, levels: int) -> np.ndarray:
    """Map a normalized [0, 1] frame to a bounded integer grid for discretization."""
    scaled = np.clip(frame, 0.0, 1.0) * (levels - 1)
    return np.rint(scaled).astype(np.int32)


def otsu_threshold(values: np.ndarray, levels: int) -> int:
    """Plain Otsu threshold on an integer-valued image histogram."""
    histogram = np.bincount(values.ravel(), minlength=levels).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0
    intensities = np.arange(levels, dtype=np.float64)
    weight_background = np.cumsum(histogram)
    weight_foreground = total - weight_background
    cumulative_mean = np.cumsum(histogram * intensities)
    global_mean = cumulative_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_background = cumulative_mean / weight_background
        mean_foreground = (global_mean - cumulative_mean) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    between = np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
    return int(between.argmax())


def frame_roi_mask(quantized: np.ndarray, levels: int, roi: str) -> np.ndarray:
    """Return a boolean ROI mask for one quantized frame.

    ``otsu`` selects the minority intensity population (the structured
    vessel/contrast pixels) and falls back to the full frame when that
    population is degenerate; ``full`` always uses the whole frame.
    """
    full = np.ones(quantized.shape, dtype=bool)
    if roi == "full":
        return full
    threshold = otsu_threshold(quantized, levels)
    above = quantized > threshold
    fraction_above = float(above.mean())
    # Pick whichever side of the threshold is the minority "structure" class.
    mask = above if fraction_above <= 0.5 else ~above
    fraction = float(mask.mean())
    if fraction < 0.02 or fraction > 0.98:
        return full
    return mask


def extract_frame_features(extractor, frame_uint: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Run PyRadiomics on a single 2-D frame + mask, returning only numeric features."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(frame_uint[np.newaxis, :, :].astype(np.float32))
    mask_image = sitk.GetImageFromArray(mask[np.newaxis, :, :].astype(np.uint8))
    result = extractor.execute(image, mask_image)
    features: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            features[key] = float(np.asarray(value, dtype=np.float64).ravel()[0])
        except (TypeError, ValueError):
            continue
    return features


def run_radiomics(sequence, n_frames: int, roi: str, bin_count: int, levels: int):
    """Return (T, F) per-frame radiomics matrix and sorted feature names for one run."""
    import torch

    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence)
    normalized = normalize_sequence(sequence).cpu().numpy()
    indices = sample_frame_indices(normalized.shape[0], n_frames).cpu().numpy()
    extractor = build_extractor(bin_count)

    per_frame: list[dict[str, float]] = []
    for index in indices:
        frame = normalized[int(index)]
        quantized = frame_to_uint(frame, levels)
        mask = frame_roi_mask(quantized, levels, roi)
        # shape2D needs a genuine sub-frame ROI; disable it when the ROI is the whole frame.
        use_shape = roi != "full" and 0.02 < float(mask.mean()) < 0.98
        if use_shape:
            extractor.enableFeatureClassByName("shape2D")
        else:
            extractor.disableAllFeatures()
            for feature_class in TEXTURE_CLASSES:
                extractor.enableFeatureClassByName(feature_class)
        per_frame.append(extract_frame_features(extractor, quantized, mask))

    feature_names = sorted({name for frame in per_frame for name in frame})
    matrix = np.array(
        [[frame.get(name, np.nan) for name in feature_names] for frame in per_frame],
        dtype=np.float64,
    )
    return matrix, feature_names


def temporal_pool(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Pool a (T, F) per-frame matrix into mean / max / std vectors, NaN-robust."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(matrix, axis=0)
        maximum = np.nanmax(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = np.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    return {"mean": mean, "max": maximum, "std": std}


def cache_signature(args: argparse.Namespace, records) -> dict[str, object]:
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
    return {
        "extractor": "pyradiomics",
        "feature_classes": list(TEXTURE_CLASSES) + ["shape2D"],
        "roi": args.roi,
        "bin_count": args.bin_count,
        "levels": args.levels,
        "image_size": "native",
        "n_frames": args.n_frames,
        "view": args.view,
        "stage": args.stage,
        "record_sha256": record_hasher.hexdigest(),
        "preprocessing": "dicom_rescale_monochrome_fix_percentile_1_99_native_resolution_v1",
    }


def extract_features(args: argparse.Namespace, records):
    signature = cache_signature(args, records)
    cache = Path(args.out) / "cache" / (
        f"radiomics_{args.roi}_bc{args.bin_count}_{args.n_frames}_"
        f"{args.view}_{args.stage}.npz"
    )
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        saved = json.loads(str(data["signature_json"].item())) if "signature_json" in data else None
        if saved == signature:
            print(f"Using cached radiomics features: {cache}")
            return {k: data[k] for k in ("mean", "max", "std", "labels", "groups")} | {
                "meta": list(data["meta"]),
                "feature_names": list(data["feature_names"]),
            }
        print(f"Ignoring stale/incompatible radiomics cache: {cache}")

    base = build_base_dataset(args.view, records)
    label_names = stage_label_names(args.stage)
    label_to_index = {name: index for index, name in enumerate(label_names)}

    reference_names: list[str] | None = None
    means, maxes, stds, labels, groups, meta = [], [], [], [], [], []
    for index in range(len(base)):
        sample = base[index]
        matrix, feature_names = run_radiomics(
            sample["sequence"], args.n_frames, args.roi, args.bin_count, args.levels
        )
        if reference_names is None:
            reference_names = feature_names
        if feature_names != reference_names:
            # Re-index to the reference feature ordering so every row aligns.
            lookup = {name: position for position, name in enumerate(feature_names)}
            reordered = np.full((matrix.shape[0], len(reference_names)), np.nan)
            for position, name in enumerate(reference_names):
                if name in lookup:
                    reordered[:, position] = matrix[:, lookup[name]]
            matrix = reordered
        pooled = temporal_pool(matrix)
        means.append(pooled["mean"])
        maxes.append(pooled["max"])
        stds.append(pooled["std"])

        if args.stage == "positive_subtype":
            label_name = positive_subtype_from_label(sample["label_text"])
        else:
            valid, label_name = binary_target_from_label(
                sample["label_text"], args.treat_blank_as_negative
            )
            if not valid:
                raise ValueError("Binary-detection sample is missing a usable label.")
        labels.append(label_to_index[label_name])
        groups.append(sample["study_key"])
        meta.append(
            {k: sample.get(k, "") for k in ("accession", "view", "run_column", "study_key")}
            | {"label_text": sample["label_text"]}
        )
        print(f"  radiomics {index + 1}/{len(base)}", end="\r", flush=True)
    print()

    data = {
        "mean": np.asarray(means, dtype=np.float64),
        "max": np.asarray(maxes, dtype=np.float64),
        "std": np.asarray(stds, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "groups": np.asarray(groups, dtype=object),
        "meta": meta,
        "feature_names": reference_names or [],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        **{k: v for k, v in data.items() if k not in ("meta", "feature_names")},
        meta=np.asarray(meta, dtype=object),
        feature_names=np.asarray(data["feature_names"], dtype=object),
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
    )
    print(f"Wrote radiomics cache: {cache}  ({data['mean'].shape[1]} features/frame-pool)")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=("AP", "Lateral"), required=True)
    parser.add_argument("--stage", default="positive_subtype", choices=("positive_subtype", "binary_detection"))
    parser.add_argument("--n-frames", type=int, default=16, help="Frames sampled per run (matches the neural probes).")
    parser.add_argument("--roi", choices=ROI_CHOICES, default="otsu", help="Per-frame region of interest.")
    parser.add_argument("--bin-count", type=int, default=32, help="Gray-level bins for texture discretization.")
    parser.add_argument("--levels", type=int, default=256, help="Integer intensity resolution before binning.")
    parser.add_argument("--excel", default=str(default_excel_path()))
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--treat-blank-as-negative", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--out", default="runs/radiomics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    select_device(args.device)
    records = build_records(
        types.SimpleNamespace(
            excel=args.excel,
            base_dir=args.base_dir,
            view=args.view,
            stage=args.stage,
            treat_blank_as_negative=args.treat_blank_as_negative,
        )
    )
    label_names = stage_label_names(args.stage)
    num_classes = len(label_names)

    print(
        f"Extracting radiomics ({args.roi} ROI, {args.n_frames} frames) for "
        f"{len(records)} {args.view} runs…"
    )
    data = extract_features(args, records)
    y, groups = data["labels"], data["groups"]
    class_counts = {label_names[c]: int((y == c).sum()) for c in range(num_classes)}
    n_splits = min(args.folds, len(set(groups)), int(np.min(np.bincount(y, minlength=num_classes))))
    folds = make_folds(y, groups, n_splits, args.seed)
    baseline = M.baseline_macro_f1(y, num_classes)

    results = []
    reps: dict[str, np.ndarray] = {}
    for name, rep_name, prep, clf in RECIPES:
        if rep_name not in reps:
            reps[rep_name] = representation(data, rep_name)
        _, m = evaluate_recipe(reps[rep_name], y, folds, num_classes, prep, clf)
        results.append({"recipe": name, "macro_f1": m["macro_f1"], "per_class_f1": m["per_class"]["f1"]})
    results.sort(key=lambda result: result["macro_f1"], reverse=True)

    print(f"\nRadiomics-only probe (majority baseline macro-F1 {baseline:.3f}):")
    widths = [40, 9] + [8] * num_classes
    header = ["recipe", "macroF1"] + [f"F1:{name[:6]}" for name in label_names]
    print("  ".join(str(value).ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for result in results:
        row = [result["recipe"][:40], f"{result['macro_f1']:.3f}", *[f"{v:.3f}" for v in result["per_class_f1"]]]
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)))

    with (out / "radiomics_probe_results.json").open("w") as handle:
        json.dump(
            {
                "config": {
                    "view": args.view,
                    "stage": args.stage,
                    "roi": args.roi,
                    "bin_count": args.bin_count,
                    "n_frames": args.n_frames,
                    "n_samples": len(y),
                    "n_groups": len(set(groups)),
                    "n_features_per_pool": int(data["mean"].shape[1]),
                },
                "baseline_macro_f1": baseline,
                "class_counts": class_counts,
                "results": results,
            },
            handle,
            indent=2,
            default=float,
        )
    print(f"\nWrote radiomics probe summary to {out}/radiomics_probe_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
