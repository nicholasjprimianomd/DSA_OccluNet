"""Extract a reproducible mask-free radiomics sensitivity baseline from DSA runs.

This dataset has no lesion/territory segmentations and almost no usable pixel-spacing
metadata, so this is deliberately *not* presented as a conventional masked PyRadiomics
analysis.  Instead, it computes fixed first-order and GLCM texture descriptors from three
temporal projection images over a central, whole-field region.  The resulting caches use
the same row metadata as the frozen-backbone caches and can therefore be evaluated with
``anatomy_task_experiments.py`` on identical patient-grouped folds.

Two feature families are written:

* ``global``: descriptors over the full central field (72 features);
* ``spatial``: reflection-invariant upper/lower quadrant summaries (288 features).

Example:
    python radiomics_experiments.py --view AP --out-dir runs/radiomics/cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pydicom
from joblib import Parallel, delayed

from occlusion_loader import (
    build_manifests,
    default_base_dir,
    default_excel_path,
    load_dicom_sequence,
)
from train_dsa_backbone import positive_subtype_from_label


EXTRACTOR_VERSION = "whole_field_projection_radiomics_v2"
MAP_NAMES = ("temporal_mean", "temporal_std", "temporal_p90_p10")
FIRST_ORDER_NAMES = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
    "iqr",
    "mean_absolute_deviation",
    "skewness",
    "excess_kurtosis",
    "energy",
    "entropy",
    "uniformity",
)
GLCM_NAMES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "angular_second_moment",
    "energy",
    "correlation",
    "entropy",
    "maximum_probability",
)
DESCRIPTOR_NAMES = FIRST_ORDER_NAMES + tuple(f"glcm_{name}" for name in GLCM_NAMES)
SPATIAL_SUMMARY_NAMES = (
    "upper_pair_mean",
    "upper_pair_absolute_difference",
    "lower_pair_mean",
    "lower_pair_absolute_difference",
)
GLCM_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=("AP", "Lateral"), default="AP")
    parser.add_argument("--excel", type=Path, default=default_excel_path())
    parser.add_argument("--base-dir", type=Path, default=default_base_dir())
    parser.add_argument("--out-dir", type=Path, default=Path("runs/radiomics/cache"))
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--temporal-samples", type=int, default=32)
    parser.add_argument("--border-fraction", type=float, default=1 / 16)
    parser.add_argument("--gray-levels", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_settings(
    target_size: int,
    temporal_samples: int,
    border_fraction: float,
    gray_levels: int,
) -> None:
    if target_size < 8:
        raise ValueError("target_size must be at least 8.")
    if target_size % 2:
        raise ValueError("target_size must be even for reflection-invariant quadrant summaries.")
    if temporal_samples < 2:
        raise ValueError("temporal_samples must be at least 2.")
    if not 0 <= border_fraction < 0.25:
        raise ValueError("border_fraction must be in [0, 0.25).")
    if gray_levels < 2:
        raise ValueError("gray_levels must be at least 2.")


def centered_block_mean(
    sequence: np.ndarray,
    target_size: int = 128,
    border_fraction: float = 1 / 16,
) -> np.ndarray:
    """Crop a fixed border and block-average every frame to a square target."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected (frames, height, width), got {sequence.shape}.")
    _, height, width = sequence.shape
    border_y = int(round(height * border_fraction))
    border_x = int(round(width * border_fraction))
    usable_height = height - 2 * border_y
    usable_width = width - 2 * border_x
    block_y = usable_height // target_size
    block_x = usable_width // target_size
    if block_y < 1 or block_x < 1:
        raise ValueError(
            f"Image shape {(height, width)} is too small for target_size={target_size} "
            f"after border_fraction={border_fraction}."
        )

    crop_height = block_y * target_size
    crop_width = block_x * target_size
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    cropped = sequence[
        :,
        start_y : start_y + crop_height,
        start_x : start_x + crop_width,
    ]
    reduced = cropped.reshape(
        len(cropped), target_size, block_y, target_size, block_x
    ).mean(axis=(2, 4), dtype=np.float64)
    return np.asarray(reduced, dtype=np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Robustly place one run on [0, 1] without using cohort-level information."""
    low, high = np.percentile(sequence, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Sequence contains non-finite values.")
    scale = float(high - low)
    if scale <= np.finfo(np.float32).eps:
        return np.zeros_like(sequence, dtype=np.float32)
    return np.clip((sequence - low) / scale, 0.0, 1.0).astype(np.float32)


def dicom_frame_times_ms(path: str | Path, n_frames: int) -> np.ndarray:
    """Read cumulative frame times, including the common T+1 leading-zero export."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive.")
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        vector = getattr(dataset, "FrameTimeVector", None)
        if vector is not None:
            intervals = np.asarray([float(value) for value in vector], dtype=np.float64)
            if (
                len(intervals) == n_frames + 1
                and abs(float(intervals[0])) <= 1e-6
            ):
                intervals = intervals[:n_frames]
            if len(intervals) == n_frames:
                times = np.cumsum(intervals)
                return times - times[0]
            if len(intervals) == n_frames - 1:
                return np.concatenate(([0.0], np.cumsum(intervals)))
        frame_time = float(getattr(dataset, "FrameTime", 0.0))
        if frame_time > 0:
            return np.arange(n_frames, dtype=np.float64) * frame_time
    except (OSError, TypeError, ValueError):
        pass
    return np.arange(n_frames, dtype=np.float64)


def resample_normalized_time(
    sequence: np.ndarray,
    frame_times_ms: np.ndarray | None,
    temporal_samples: int,
) -> np.ndarray:
    """Linearly resample a cine on a fixed normalized-time grid."""
    if sequence.ndim != 3 or len(sequence) < 1:
        raise ValueError(f"Expected a non-empty (frames, height, width) sequence, got {sequence.shape}.")
    if temporal_samples < 2:
        raise ValueError("temporal_samples must be at least 2.")
    if len(sequence) == 1:
        return np.repeat(sequence, temporal_samples, axis=0)

    times = np.asarray(frame_times_ms, dtype=np.float64) if frame_times_ms is not None else np.asarray([])
    if (
        len(times) != len(sequence)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) < 0)
        or times[-1] - times[0] <= np.finfo(float).eps
    ):
        times = np.arange(len(sequence), dtype=np.float64)
    times = (times - times[0]) / (times[-1] - times[0])
    targets = np.linspace(0.0, 1.0, temporal_samples)
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 1, len(times) - 1)
    left = right - 1
    denominator = times[right] - times[left]
    weights = np.divide(
        targets - times[left],
        denominator,
        out=np.zeros_like(targets),
        where=denominator > np.finfo(float).eps,
    ).astype(np.float32)
    resampled = (
        sequence[left] * (1.0 - weights[:, None, None])
        + sequence[right] * weights[:, None, None]
    )
    return np.asarray(resampled, dtype=np.float32)


def projection_maps(sequence: np.ndarray) -> dict[str, np.ndarray]:
    percentiles = np.percentile(sequence, (10.0, 90.0), axis=0)
    return {
        "temporal_mean": sequence.mean(axis=0, dtype=np.float64).astype(np.float32),
        "temporal_std": sequence.std(axis=0, dtype=np.float64).astype(np.float32),
        "temporal_p90_p10": np.asarray(percentiles[1] - percentiles[0], dtype=np.float32),
    }


def first_order_features(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("First-order input must be non-empty and finite.")
    mean = float(values.mean())
    std = float(values.std())
    p10, p25, median, p75, p90 = np.percentile(values, (10, 25, 50, 75, 90))
    centered = values - mean
    if std <= np.finfo(float).eps:
        skewness = 0.0
        kurtosis = 0.0
    else:
        standardized = centered / std
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4) - 3.0)
    histogram = np.histogram(values, bins=32, range=(0.0, 1.0))[0].astype(np.float64)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    uniformity = float(np.sum(probabilities**2))
    return np.asarray(
        [
            mean,
            std,
            float(values.min()),
            float(values.max()),
            float(p10),
            float(p25),
            float(median),
            float(p75),
            float(p90),
            float(p75 - p25),
            float(np.mean(np.abs(centered))),
            skewness,
            kurtosis,
            float(np.sum(values**2)),
            entropy,
            uniformity,
        ],
        dtype=np.float64,
    )


def discretize(image: np.ndarray, gray_levels: int) -> np.ndarray:
    """Fixed-bin-count discretization after robust within-region clipping."""
    low, high = np.percentile(image, (1.0, 99.0))
    scale = float(high - low)
    if scale <= np.finfo(float).eps:
        return np.zeros(image.shape, dtype=np.int16)
    normalized = np.clip((image - low) / scale, 0.0, 1.0)
    return np.minimum((normalized * gray_levels).astype(np.int16), gray_levels - 1)


def glcm_features(image: np.ndarray, gray_levels: int = 16) -> np.ndarray:
    quantized = discretize(np.asarray(image, dtype=np.float64), gray_levels)
    matrix = np.zeros((gray_levels, gray_levels), dtype=np.float64)
    height, width = quantized.shape
    for delta_y, delta_x in GLCM_OFFSETS:
        y1_start = max(0, -delta_y)
        y1_stop = min(height, height - delta_y)
        x1_start = max(0, -delta_x)
        x1_stop = min(width, width - delta_x)
        first = quantized[y1_start:y1_stop, x1_start:x1_stop]
        second = quantized[
            y1_start + delta_y : y1_stop + delta_y,
            x1_start + delta_x : x1_stop + delta_x,
        ]
        pair_ids = first.ravel().astype(np.int64) * gray_levels + second.ravel()
        counts = np.bincount(pair_ids, minlength=gray_levels * gray_levels)
        directional = counts.reshape(gray_levels, gray_levels)
        matrix += directional + directional.T

    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("GLCM has no valid neighboring pixel pairs.")
    matrix /= total
    indices = np.arange(gray_levels, dtype=np.float64)
    row_index, column_index = np.meshgrid(indices, indices, indexing="ij")
    difference = row_index - column_index
    contrast = float(np.sum(matrix * difference**2))
    dissimilarity = float(np.sum(matrix * np.abs(difference)))
    homogeneity = float(np.sum(matrix / (1.0 + difference**2)))
    angular_second_moment = float(np.sum(matrix**2))
    energy = float(np.sqrt(angular_second_moment))
    row_probability = matrix.sum(axis=1)
    column_probability = matrix.sum(axis=0)
    row_mean = float(np.sum(indices * row_probability))
    column_mean = float(np.sum(indices * column_probability))
    row_std = float(np.sqrt(np.sum((indices - row_mean) ** 2 * row_probability)))
    column_std = float(np.sqrt(np.sum((indices - column_mean) ** 2 * column_probability)))
    denominator = row_std * column_std
    correlation = (
        float(np.sum(matrix * (row_index - row_mean) * (column_index - column_mean)) / denominator)
        if denominator > np.finfo(float).eps
        else 1.0
    )
    nonzero = matrix[matrix > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    maximum_probability = float(matrix.max())
    return np.asarray(
        [
            contrast,
            dissimilarity,
            homogeneity,
            angular_second_moment,
            energy,
            correlation,
            entropy,
            maximum_probability,
        ],
        dtype=np.float64,
    )


def describe_region(image: np.ndarray, gray_levels: int) -> np.ndarray:
    return np.concatenate((first_order_features(image), glcm_features(image, gray_levels)))


def extract_feature_families(
    sequence: np.ndarray,
    target_size: int = 128,
    border_fraction: float = 1 / 16,
    gray_levels: int = 16,
    frame_times_ms: np.ndarray | None = None,
    temporal_samples: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    validate_settings(target_size, temporal_samples, border_fraction, gray_levels)
    reduced = centered_block_mean(sequence, target_size, border_fraction)
    normalized = normalize_sequence(reduced)
    normalized = resample_normalized_time(normalized, frame_times_ms, temporal_samples)
    maps = projection_maps(normalized)
    global_features: list[np.ndarray] = []
    spatial_features: list[np.ndarray] = []
    midpoint = target_size // 2
    if midpoint < 2 or target_size - midpoint < 2:
        raise ValueError("target_size is too small for quadrant texture descriptors.")
    for map_name in MAP_NAMES:
        image = maps[map_name]
        global_features.append(describe_region(image, gray_levels))
        quadrants = (
            describe_region(image[:midpoint, :midpoint], gray_levels),
            describe_region(image[:midpoint, midpoint:], gray_levels),
            describe_region(image[midpoint:, :midpoint], gray_levels),
            describe_region(image[midpoint:, midpoint:], gray_levels),
        )
        upper_left, upper_right, lower_left, lower_right = quadrants
        spatial_features.extend(
            (
                (upper_left + upper_right) / 2,
                np.abs(upper_left - upper_right),
                (lower_left + lower_right) / 2,
                np.abs(lower_left - lower_right),
            )
        )
    global_array = np.concatenate(global_features).astype(np.float32)
    spatial_array = np.concatenate(spatial_features).astype(np.float32)
    if not np.isfinite(global_array).all() or not np.isfinite(spatial_array).all():
        raise ValueError("Radiomics extraction produced non-finite features.")
    return global_array, spatial_array


def feature_names(family: str) -> list[str]:
    if family == "global":
        return [
            f"{map_name}__global__{descriptor}"
            for map_name in MAP_NAMES
            for descriptor in DESCRIPTOR_NAMES
        ]
    if family == "spatial":
        return [
            f"{map_name}__{summary_name}__{descriptor}"
            for map_name in MAP_NAMES
            for summary_name in SPATIAL_SUMMARY_NAMES
            for descriptor in DESCRIPTOR_NAMES
        ]
    raise ValueError(f"Unknown family: {family}")


def record_signature(records) -> str:
    digest = hashlib.sha256()
    for record in records:
        path = Path(record.dicom_path)
        stat = path.stat()
        identity = (
            record.study_key,
            record.accession,
            record.view,
            record.run_column,
            record.label_text,
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
        )
        digest.update((json.dumps(identity, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def methodology(
    view: str,
    records,
    target_size: int,
    temporal_samples: int,
    border_fraction: float,
    gray_levels: int,
) -> dict[str, object]:
    return {
        "extractor": EXTRACTOR_VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pydicom": pydicom.__version__,
            "joblib": joblib.__version__,
        },
        "view": view,
        "record_sha256": record_signature(records),
        "n_records": len(records),
        "roi_policy": "fixed central whole-field crop; no lesion or territory mask is available",
        "claim_scope": "mask-free radiomics sensitivity baseline; not conventional masked PyRadiomics",
        "border_fraction": border_fraction,
        "downsample": f"centered block mean to {target_size}x{target_size}",
        "normalization": "within-run 1st/99th percentile clipping on the downsampled sequence",
        "temporal_resampling": (
            f"linear interpolation to {temporal_samples} normalized-time samples using "
            "FrameTimeVector/FrameTime with frame-index fallback"
        ),
        "frame_time_vector_policy": "accept T, T-1, and leading-zero T+1 interval vectors",
        "projection_maps": list(MAP_NAMES),
        "first_order_features": list(FIRST_ORDER_NAMES),
        "first_order_histogram_bins": 32,
        "glcm_features": list(GLCM_NAMES),
        "glcm_gray_levels": gray_levels,
        "glcm_discretization": "within-region 1st/99th percentile fixed bin count",
        "glcm_offsets_image_units": [list(offset) for offset in GLCM_OFFSETS],
        "glcm_direction_aggregation": "symmetric pooled counts across four directions",
        "spatial_policy": "quadrant pairs summarized by mean and absolute left-right difference to enforce reflection invariance",
        "physical_spacing_note": "pixel spacing is absent for most source DICOMs; texture distances are normalized image units",
    }


def cache_is_current(
    path: Path,
    signature: dict[str, object],
    expected_names: list[str],
    expected_rows: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as archive:
            required = ("mean", "labels", "groups", "meta", "feature_names", "signature_json")
            if any(name not in archive for name in required):
                return False
            saved = json.loads(str(archive["signature_json"].item()))
            matrix = archive["mean"]
            return (
                saved == signature
                and matrix.shape == (expected_rows, len(expected_names))
                and np.isfinite(matrix).all()
                and len(archive["labels"]) == expected_rows
                and len(archive["groups"]) == expected_rows
                and len(archive["meta"]) == expected_rows
                and list(archive["feature_names"]) == expected_names
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def save_cache(
    path: Path,
    matrix: np.ndarray,
    names: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
    metadata: np.ndarray,
    signature: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=matrix,
        labels=labels,
        groups=groups,
        meta=metadata,
        feature_names=np.asarray(names, dtype=object),
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
    )


def extract_record(
    record,
    target_size: int,
    temporal_samples: int,
    border_fraction: float,
    gray_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequence = load_dicom_sequence(record.dicom_path)
    frame_times = dicom_frame_times_ms(record.dicom_path, len(sequence))
    return extract_feature_families(
        sequence,
        target_size,
        border_fraction,
        gray_levels,
        frame_times,
        temporal_samples,
    )


def main() -> int:
    args = parse_args()
    validate_settings(
        args.target_size,
        args.temporal_samples,
        args.border_fraction,
        args.gray_levels,
    )
    ap_records, lateral_records, _ = build_manifests(args.excel, args.base_dir)
    records = ap_records if args.view == "AP" else lateral_records
    if not records:
        raise ValueError(f"No resolved {args.view} records were found.")

    base_signature = methodology(
        args.view,
        records,
        args.target_size,
        args.temporal_samples,
        args.border_fraction,
        args.gray_levels,
    )
    paths = {
        family: args.out_dir / f"wholefield_radiomics_{args.view.lower()}_{family}.npz"
        for family in ("global", "spatial")
    }
    signatures = {
        family: base_signature
        | {
            "feature_family": family,
            "feature_dimension": len(feature_names(family)),
        }
        for family in paths
    }
    if not args.force and all(
        cache_is_current(paths[family], signatures[family], feature_names(family), len(records))
        for family in paths
    ):
        print("Using current radiomics caches:")
        for path in paths.values():
            print(f"  {path}")
        return 0

    print(
        f"Extracting mask-free whole-field radiomics from {len(records)} {args.view} runs "
        f"with {args.jobs} worker(s)..."
    )
    extracted = Parallel(n_jobs=args.jobs, prefer="threads", verbose=5)(
        delayed(extract_record)(
            record,
            args.target_size,
            args.temporal_samples,
            args.border_fraction,
            args.gray_levels,
        )
        for record in records
    )
    global_matrix = np.stack([row[0] for row in extracted])
    spatial_matrix = np.stack([row[1] for row in extracted])
    label_names = ("m2", "m3", "other_positive")
    label_to_index = {name: index for index, name in enumerate(label_names)}
    labels = np.asarray(
        [label_to_index[positive_subtype_from_label(record.label_text)] for record in records],
        dtype=np.int64,
    )
    groups = np.asarray([record.study_key for record in records], dtype=object)
    metadata = np.asarray(
        [
            {
                "accession": record.accession,
                "view": record.view,
                "run_column": record.run_column,
                "study_key": record.study_key,
                "label_text": record.label_text,
            }
            for record in records
        ],
        dtype=object,
    )
    matrices = {"global": global_matrix, "spatial": spatial_matrix}
    for family, path in paths.items():
        save_cache(
            path,
            matrices[family],
            feature_names(family),
            labels,
            groups,
            metadata,
            signatures[family],
        )
        print(f"Wrote {matrices[family].shape} {family} features to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
