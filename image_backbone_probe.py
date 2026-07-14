"""Probe frozen image backbones frame-by-frame on DSA runs.

This is the image-model counterpart to ``experiments.py``.  It samples the
same frames from each DICOM run, applies the checkpoint's native channel
normalization at an explicit spatial resolution, extracts one global feature
per frame, and pools those features over time.  The resulting representations
are evaluated with the identical patient-grouped folds and classifier recipes
used for V-JEPA 2.

Examples:
    python image_backbone_probe.py --view AP --model facebook/dinov2-large \
        --image-size 518 --device cuda --amp --out runs/ap_dinov2l518
    python image_backbone_probe.py --view AP --model microsoft/rad-dino \
        --image-size 518 --device cuda --amp --out runs/ap_raddino518
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import metrics as M
import viz
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


INPUT_VARIANTS = (
    "uniform",
    "hflip",
    "border90",
    "multicrop",
    "top_contrast",
    "temporal_rgb",
    "phase_rgb",
)


INPUT_VARIANT_PARAMETERS = {
    "hflip": {"version": 1, "axis": "width", "clip_coherent": True},
    "border90": {"version": 1, "exact_name": "center_crop_90", "center_fraction": 0.90},
    "multicrop": {
        "version": 1,
        "exact_name": "mean_pool_full_center90_center80",
        "center_fractions": [1.0, 0.90, 0.80],
    },
    "top_contrast": {
        "version": 2,
        "exact_name": "top_temporal_change",
        "score": "mean_abs_adjacent_frame_change",
        "retain_frame_zero": True,
        "short_run_fallback": "uniform_with_repetition",
    },
    "temporal_rgb": {
        "version": 2,
        "exact_name": "full_run_temporal_stats_rgb_absmax_std_argmax",
        "channels": ["absolute_peak_from_first3_median", "temporal_std", "weighted_peak_time"],
        "single_or_flat_fallback": "original_frame_repeated_rgb",
    },
    "phase_rgb": {
        "version": 1,
        "channels": ["mid_onset_to_peak", "arterial_peak", "peak_plus_2000ms"],
        "enhancement": "relu(first3_median_minus_frame)",
        "score": "mean_top_5_percent_enhancement",
    },
}


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip("-")


def preprocessing_metadata(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoConfig, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    return {
        "requested_revision": args.revision,
        "resolved_revision": getattr(config, "_commit_hash", None) or args.revision or "unknown",
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
    }


def cache_signature(args: argparse.Namespace, records, preprocessing, device: torch.device) -> dict[str, object]:
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
    signature = {
        "model": args.model,
        **preprocessing,
        "image_size": args.image_size,
        "n_frames": args.n_frames,
        "frame_batch_size": args.frame_batch_size,
        "view": args.view,
        "stage": args.stage,
        "record_sha256": record_hasher.hexdigest(),
        "amp": args.amp,
        "model_dtype": "bfloat16" if args.amp and device.type == "cuda" else "float32",
        "preprocessing": "dicom_rescale_monochrome_fix_percentile_1_99_square_resize_rgb_v2",
    }
    # Preserve compatibility with the already-computed uniform-frame cache.
    if args.input_variant != "uniform":
        signature["input_variant"] = args.input_variant
        signature["input_variant_parameters"] = INPUT_VARIANT_PARAMETERS[args.input_variant]
    return signature


def load_model_and_normalization(args: argparse.Namespace, preprocessing, device: torch.device):
    from transformers import AutoModel

    dtype = torch.bfloat16 if args.amp and device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device).eval()
    mean = torch.tensor(preprocessing["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(preprocessing["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
    return model, mean.to(device), std.to(device)


def global_frame_features(outputs: object) -> torch.Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise ValueError("Image backbone returned neither pooler_output nor last_hidden_state.")
    return hidden[:, 0]


def normalize_map(image: torch.Tensor) -> torch.Tensor:
    """Robustly map one derived 2-D image to [0, 1]."""
    flattened = image.flatten()
    lower = torch.quantile(flattened, 0.01)
    upper = torch.quantile(flattened, 0.99)
    if torch.isclose(lower, upper):
        lower, upper = image.amin(), image.amax()
    if torch.isclose(lower, upper):
        return torch.zeros_like(image)
    return ((image - lower) / (upper - lower)).clamp(0.0, 1.0)


def select_top_contrast_frames(sequence: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Select the most temporally informative frames, returned in acquisition order."""
    if sequence.shape[0] <= n_frames:
        return sample_frame_indices(sequence.shape[0], n_frames)
    count = n_frames
    changes = torch.zeros(sequence.shape[0], dtype=torch.float32)
    changes[1:] = (sequence[1:] - sequence[:-1]).abs().mean(dim=(1, 2))
    # Always retain one early reference frame, then take the strongest changes.
    if count == 1:
        return changes.argmax().view(1)
    selected = torch.topk(changes[1:], k=count - 1).indices + 1
    return torch.cat((torch.zeros(1, dtype=torch.long), selected)).sort().values


def resize_grayscale_frames(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(
        frames.unsqueeze(1),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def centered_crop(sequence: torch.Tensor, fraction: float) -> torch.Tensor:
    height, width = sequence.shape[-2:]
    crop_h = max(1, round(height * fraction))
    crop_w = max(1, round(width * fraction))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return sequence[:, top : top + crop_h, left : left + crop_w]


def dicom_frame_times_ms(path: str | Path | None, n_frames: int) -> torch.Tensor:
    """Return cumulative physical frame times, with a conservative uniform fallback."""
    if path:
        try:
            import pydicom

            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            vector = getattr(dataset, "FrameTimeVector", None)
            if vector is not None:
                intervals = torch.as_tensor([float(value) for value in vector], dtype=torch.float32)
                # Some XA exports store T+1 entries: a leading zero followed by one
                # increment per frame.  T frames need only the leading zero plus T-1
                # increments, so discard the trailing extra interval.
                if (
                    len(intervals) == n_frames + 1
                    and abs(float(intervals[0])) <= 1e-6
                ):
                    intervals = intervals[:n_frames]
                if len(intervals) == n_frames:
                    times = intervals.cumsum(dim=0)
                    return times - times[0]
                if len(intervals) == n_frames - 1:
                    return torch.cat((torch.zeros(1), intervals.cumsum(dim=0)))
            frame_time = float(getattr(dataset, "FrameTime", 0.0))
            if frame_time > 0:
                return torch.arange(n_frames, dtype=torch.float32) * frame_time
        except (OSError, TypeError, ValueError):
            pass
    return torch.arange(n_frames, dtype=torch.float32)


def arterial_phase_indices(sequence: torch.Tensor, dicom_path: str | Path | None) -> tuple[int, int, int]:
    """Select early, peak, and two-seconds-late phases without using class labels."""
    n_frames = sequence.shape[0]
    if n_frames < 4:
        indices = sample_frame_indices(n_frames, 3).tolist()
        return int(indices[0]), int(indices[1]), int(indices[2])
    baseline_count = min(3, n_frames)
    baseline = sequence[:baseline_count].median(dim=0).values
    enhancement = (baseline.unsqueeze(0) - sequence).clamp_min(0.0)
    pixels = enhancement.flatten(1)
    top_count = max(1, round(pixels.shape[1] * 0.05))
    scores = pixels.topk(top_count, dim=1).values.mean(dim=1)
    if n_frames >= 3:
        padded = F.pad(scores.view(1, 1, -1), (1, 1), mode="replicate").view(-1)
        scores = padded.unfold(0, 3, 1).median(dim=1).values
    peak_score = scores.max()
    if peak_score <= 1e-6:
        indices = sample_frame_indices(n_frames, 3).tolist()
        return int(indices[0]), int(indices[1]), int(indices[2])

    post_baseline = torch.arange(n_frames) >= baseline_count
    onset_candidates = torch.where(post_baseline & (scores >= 0.10 * peak_score))[0]
    onset = int(onset_candidates[0]) if len(onset_candidates) else baseline_count
    peak_candidates = torch.where((torch.arange(n_frames) >= onset) & (scores >= 0.95 * peak_score))[0]
    peak = int(peak_candidates[0]) if len(peak_candidates) else int(scores.argmax())
    peak = max(onset, peak)

    times = dicom_frame_times_ms(dicom_path, n_frames)
    early_target = 0.5 * (times[onset] + times[peak])
    early = int((times - early_target).abs().argmin())
    late_candidates = torch.where(times >= times[peak] + 2000.0)[0]
    late = int(late_candidates[0]) if len(late_candidates) else n_frames - 1
    return early, peak, late


def prepare_model_images(
    sequence: torch.Tensor,
    args: argparse.Namespace,
    dicom_path: str | Path | None = None,
) -> torch.Tensor:
    """Build label-preserving image inputs for a frozen 2-D backbone."""
    sequence = normalize_sequence(sequence)

    if args.input_variant == "temporal_rgb":
        temporal_variation = (sequence - sequence[:1]).abs().amax()
        if sequence.shape[0] == 1 or temporal_variation <= 1e-6:
            frame = sequence[0].unsqueeze(0).unsqueeze(0)
            return F.interpolate(
                frame,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            ).repeat(1, 3, 1, 1)
        reference = sequence[: min(3, sequence.shape[0])].median(dim=0).values
        enhancement = (sequence - reference).abs()
        peak_enhancement, peak_index = enhancement.max(dim=0)
        temporal_std = sequence.std(dim=0, correction=0)
        peak_time = peak_index.float() / max(sequence.shape[0] - 1, 1)
        temporal_image = torch.stack(
            (
                normalize_map(peak_enhancement),
                normalize_map(temporal_std),
                peak_time * normalize_map(peak_enhancement),
            ),
            dim=0,
        ).unsqueeze(0)
        return F.interpolate(
            temporal_image,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )

    if args.input_variant == "phase_rgb":
        early, peak, late = arterial_phase_indices(sequence, dicom_path)
        phase_image = torch.stack((sequence[early], sequence[peak], sequence[late]), dim=0).unsqueeze(0)
        return F.interpolate(
            phase_image,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )

    if args.input_variant == "top_contrast":
        frame_indices = select_top_contrast_frames(sequence, args.n_frames)
    else:
        frame_indices = sample_frame_indices(sequence.shape[0], args.n_frames)
    frames = sequence.index_select(0, frame_indices)

    if args.input_variant == "border90":
        frames = centered_crop(frames, 0.90)

    if args.input_variant == "multicrop":
        # Pooling full, 90%, and 80% centered fields tests whether scanner borders
        # and magnification variance obscure the vascular signal.
        resized = [
            resize_grayscale_frames(centered_crop(frames, fraction), args.image_size)
            for fraction in (1.0, 0.90, 0.80)
        ]
        images = torch.cat(resized, dim=0).repeat(1, 3, 1, 1)
    else:
        images = resize_grayscale_frames(frames, args.image_size).repeat(1, 3, 1, 1)

    if args.input_variant == "hflip":
        images = images.flip(-1)
    return images


@torch.inference_mode()
def extract_features(args: argparse.Namespace, records, device: torch.device):
    preprocessing = preprocessing_metadata(args)
    signature = cache_signature(args, records, preprocessing, device)
    variant_tag = "" if args.input_variant == "uniform" else f"_{args.input_variant}"
    cache = Path(args.out) / "cache" / (
        f"image_{model_slug(args.model)}_{args.image_size}_{args.n_frames}_"
        f"{args.view}_{args.stage}{variant_tag}.npz"
    )
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        saved_signature = json.loads(str(data["signature_json"].item())) if "signature_json" in data else None
        if saved_signature == signature:
            print(f"Using cached image-backbone features: {cache}")
            return (
                {k: data[k] for k in ("mean", "max", "std", "labels", "groups")}
                | {"meta": list(data["meta"])},
                preprocessing["resolved_revision"],
            )
        print(f"Ignoring stale/incompatible image-backbone cache: {cache}")

    model, channel_mean, channel_std = load_model_and_normalization(args, preprocessing, device)
    base = build_base_dataset(args.view, records)
    label_names = stage_label_names(args.stage)
    label_to_index = {name: index for index, name in enumerate(label_names)}

    means, maxes, stds, labels, groups, meta = [], [], [], [], [], []
    for index in range(len(base)):
        sample = base[index]
        sequence = sample["sequence"]
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence)
        frames = prepare_model_images(sequence, args, sample.get("dicom_path"))

        frame_features = []
        for start in range(0, len(frames), args.frame_batch_size):
            chunk = frames[start : start + args.frame_batch_size].to(device)
            chunk = (chunk - channel_mean) / channel_std
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                outputs = model(pixel_values=chunk)
                frame_features.append(global_frame_features(outputs).float().cpu())
        frame_features = torch.cat(frame_features, dim=0)
        means.append(frame_features.mean(dim=0))
        maxes.append(frame_features.amax(dim=0))
        stds.append(frame_features.std(dim=0, correction=0 if len(frame_features) == 1 else 1))

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
        print(f"  features {index + 1}/{len(base)}", end="\r", flush=True)
    print()

    data = {
        "mean": torch.stack(means).numpy(),
        "max": torch.stack(maxes).numpy(),
        "std": torch.stack(stds).numpy(),
        "labels": np.asarray(labels, dtype=np.int64),
        "groups": np.asarray(groups, dtype=object),
        "meta": meta,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        **{k: v for k, v in data.items() if k != "meta"},
        meta=np.asarray(meta, dtype=object),
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
        model_revision=np.asarray(preprocessing["resolved_revision"]),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data, preprocessing["resolved_revision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=("AP", "Lateral"), required=True)
    parser.add_argument("--stage", default="positive_subtype", choices=("positive_subtype", "binary_detection"))
    parser.add_argument("--model", default="facebook/dinov2-large")
    parser.add_argument("--revision", default=None, help="Optional immutable Hugging Face checkpoint revision.")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument(
        "--input-variant",
        choices=INPUT_VARIANTS,
        default="uniform",
        help="Label-preserving input construction to compare against uniform full-frame sampling.",
    )
    parser.add_argument("--excel", default=str(default_excel_path()))
    parser.add_argument("--base-dir", default=str(default_base_dir()))
    parser.add_argument("--treat-blank-as-negative", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", default="runs/image_backbone")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
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
        f"Extracting {args.model} frame features for {len(records)} {args.view} runs "
        f"({args.n_frames} frames, {args.image_size}px, variant={args.input_variant})…"
    )
    data, model_revision = extract_features(args, records, device)
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
        oof, metrics = evaluate_recipe(reps[rep_name], y, folds, num_classes, prep, clf)
        results.append(
            {
                "recipe": name,
                "rep": rep_name,
                "prep": prep,
                "clf": clf,
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "per_class_f1": metrics["per_class"]["f1"],
                "metrics": metrics,
                "oof": oof,
            }
        )
    results.sort(key=lambda result: result["macro_f1"], reverse=True)

    widths = [40, 9, 8, 8] + [8] * num_classes
    header = ["recipe", "macroF1", "bal_acc", "acc"] + [f"F1:{name[:6]}" for name in label_names]
    print("\n" + "  ".join(str(value).ljust(width) for value, width in zip(header, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for result in results:
        row = [
            result["recipe"][:40],
            f"{result['macro_f1']:.3f}",
            f"{result['balanced_accuracy']:.3f}",
            f"{result['accuracy']:.3f}",
            *[f"{value:.3f}" for value in result["per_class_f1"]],
        ]
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)))

    best = results[0]
    print(f"\nBest: {best['recipe']}  macro-F1 {best['macro_f1']:.3f}")
    print(M.format_per_class(best["metrics"], label_names))

    config = {
        "model": args.model,
        "requested_revision": args.revision,
        "model_revision": model_revision,
        "image_size": args.image_size,
        "n_frames": args.n_frames,
        "frame_batch_size": args.frame_batch_size,
        "input_variant": args.input_variant,
        "view": args.view,
        "stage": args.stage,
        "folds": n_splits,
        "seed": args.seed,
        "amp": args.amp,
        "model_dtype": "bfloat16" if args.amp and device.type == "cuda" else "float32",
        "pooling": "per-frame global token, temporal mean or mean+max+std",
        "n_samples": len(y),
        "n_groups": len(set(groups)),
    }
    with (out / "sweep_results.json").open("w") as handle:
        json.dump(
            {
                "config": config,
                "baseline_macro_f1": baseline,
                "class_counts": class_counts,
                "results": [
                    {k: v for k, v in result.items() if k not in ("metrics", "oof")}
                    for result in results
                ],
            },
            handle,
            indent=2,
            default=float,
        )
    viz.save_metric_hbar(
        [result["recipe"] for result in results][::-1],
        [result["macro_f1"] for result in results][::-1],
        out / "recipe_macro_f1.png",
        title=f"{args.model} {args.image_size}px macro-F1",
        baseline=baseline,
    )
    viz.save_confusion_matrix(
        y,
        best["oof"],
        label_names,
        out / "best_confusion_matrix.png",
        title=f"{args.model} {args.image_size}px (macro-F1={best['macro_f1']:.3f})",
    )
    viz.save_per_class_bars(
        best["metrics"]["per_class"],
        label_names,
        out / "best_per_class.png",
        title=f"{args.model} {args.image_size}px best recipe",
    )
    print(f"\nWrote sweep table + plots to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
