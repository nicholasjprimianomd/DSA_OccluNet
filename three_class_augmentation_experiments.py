"""Evaluate direct m2-vs-m3-vs-other augmentation and test-time averaging.

This script deliberately preserves the project's existing run-level three-class
target.  ``other_positive`` means "not labelled M2 or M3"; it is never treated
as non-MCA.  Every fit uses patient-grouped folds, and all augmented views of a
run remain in the training side of the same fold.

Example:
    .venv/bin/python three_class_augmentation_experiments.py \
      --vjepa-original runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
      --vjepa-flip runs/ap_exp_vitg384_norm_flip/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm_hflip.npz \
      --dino-original runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
      --dino-flip runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_hflip.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize

import metrics as M
from compare_feature_caches import canonical_metadata, file_sha256, parse_feature, parse_seeds
from experiments import make_folds


LABEL_NAMES = ("m2", "m3", "other_positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vjepa-original", type=Path, required=True)
    parser.add_argument("--vjepa-flip", type=Path, required=True)
    parser.add_argument("--dino-original", type=Path, required=True)
    parser.add_argument("--dino-flip", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_feature,
        default=[],
        metavar="NAME=PATH",
        help="Additional DINO input representation to score alone and in direct V-JEPA+DINO fusion.",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:20"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("runs/three_class_augmentations/results.json"))
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, object]:
    cache = np.load(path, allow_pickle=True)
    required = ("mean", "labels", "groups", "meta")
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
    loaded = {key: cache[key] for key in required}
    n_rows = len(loaded["labels"])
    if any(len(loaded[key]) != n_rows for key in required):
        raise ValueError(f"{path} has inconsistent row counts.")
    if not np.isfinite(loaded["mean"]).all():
        raise ValueError(f"{path} contains non-finite mean features.")
    loaded["metadata_identity"] = canonical_metadata(loaded["meta"])
    loaded["signature"] = (
        json.loads(str(cache["signature_json"].item())) if "signature_json" in cache else None
    )
    loaded["sha256"] = file_sha256(path)
    loaded["path"] = str(path)
    return loaded


def aligned_to_reference(cache: dict[str, object], reference_identity) -> dict[str, object]:
    """Align by canonical run metadata, never by an unverified row position."""
    identities = cache["metadata_identity"]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate canonical metadata identities in {cache['path']}.")
    index_by_identity = {identity: index for index, identity in enumerate(identities)}
    missing = [identity for identity in reference_identity if identity not in index_by_identity]
    extras = set(identities).difference(reference_identity)
    if missing or extras:
        raise ValueError(
            f"Metadata mismatch for {cache['path']}: {len(missing)} missing, {len(extras)} extra."
        )
    order = np.asarray([index_by_identity[identity] for identity in reference_identity], dtype=int)
    result = dict(cache)
    for key in ("mean", "labels", "groups", "meta"):
        result[key] = cache[key][order]
    result["metadata_identity"] = tuple(cache["metadata_identity"][index] for index in order)
    return result


def normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[float]]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(LABEL_NAMES))).astype(float)
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator != 0).tolist()


def macro_auprc(y: np.ndarray, probabilities: np.ndarray) -> float:
    binary = label_binarize(y, classes=np.arange(len(LABEL_NAMES)))
    return float(average_precision_score(binary, probabilities, average="macro"))


def other_subgroup(label_text: str, target: int) -> str:
    if target != 2:
        return ""
    compact = re.sub(r"\s+", "", str(label_text).upper())
    if "M1" in compact or "M4" in compact:
        return "mca_other"
    if re.search(r"A[1-4]", compact):
        return "aca"
    if re.search(r"P[1-4]", compact):
        return "pca"
    return "unclassified"


def build_experiments(features: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []
    for source_name in ("vjepa", "dino", "fusion"):
        source = features[source_name]
        for train_augmented in (False, True):
            for test_rule in ("original", "embedding_avg", "probability_avg"):
                experiments.append(
                    {
                        "name": (
                            f"{source_name}/train_"
                            f"{'original+flip' if train_augmented else 'original'}"
                            f"/test_{test_rule}"
                        ),
                        "original": source["original"],
                        "flip": source["flip"],
                        "train_augmented": train_augmented,
                        "test_rule": test_rule,
                    }
                )
        experiments.append(
            {
                "name": f"{source_name}/train_embedding_avg/test_embedding_avg",
                "original": 0.5 * (source["original"] + source["flip"]),
                "flip": None,
                "train_augmented": False,
                "test_rule": "original",
            }
        )
    late_sources = (features["vjepa"], features["dino"])
    for train_augmented in (False, True):
        for test_rule in ("original", "embedding_avg", "probability_avg"):
            experiments.append(
                {
                    "name": (
                        "late_fusion/train_"
                        f"{'original+flip' if train_augmented else 'original'}"
                        f"/test_{test_rule}"
                    ),
                    "late_sources": late_sources,
                    "train_augmented": train_augmented,
                    "test_rule": test_rule,
                }
            )
    experiments.append(
        {
            "name": "late_fusion/train_embedding_avg/test_embedding_avg",
            "late_sources": tuple(
                {
                    "original": 0.5 * (source["original"] + source["flip"]),
                    "flip": None,
                }
                for source in late_sources
            ),
            "train_augmented": False,
            "test_rule": "original",
        }
    )
    for name, source in features.items():
        if name in {"vjepa", "dino", "fusion"}:
            continue
        experiments.append(
            {
                "name": f"{name}/direct",
                "original": source["original"],
                "flip": None,
                "train_augmented": False,
                "test_rule": "original",
            }
        )
    if "dino_temporal_change" in features:
        temporal_source = features["dino_temporal_change"]
        secondary_combinations = {
            "late_vjepa_temporal_change/direct": (
                {"original": features["vjepa"]["original"], "flip": None},
                temporal_source,
            ),
            "late_dino_temporal_change/direct": (
                {"original": features["dino"]["original"], "flip": None},
                temporal_source,
            ),
            "late_baselinefusion_temporal_change/direct": (
                {"original": features["fusion"]["original"], "flip": None},
                temporal_source,
            ),
        }
        for name, sources in secondary_combinations.items():
            experiments.append(
                {
                    "name": name,
                    "late_sources": sources,
                    "train_augmented": False,
                    "test_rule": "original",
                }
            )
        routed_sources = (
            {"original": features["fusion"]["original"], "flip": None},
            temporal_source,
        )
        experiments.extend(
            (
                {
                    "name": "fixed_class_route_temporal_m2m3_baselinefusion_other/direct",
                    "class_route_sources": routed_sources,
                    "fixed_route": (1, 1, 0),
                },
                {
                    "name": "nested_class_route_baselinefusion_vs_temporal/direct",
                    "class_route_sources": routed_sources,
                    "nested_route": True,
                },
            )
        )
    return experiments


def choose_nested_class_route(sources, y, groups, outer_train, seed, outer_fold_index):
    """Choose one direct three-class source per class using grouped inner OOF predictions."""
    inner_y = y[outer_train]
    inner_groups = groups[outer_train]
    n_splits = min(3, len(set(inner_groups)), int(np.min(np.bincount(inner_y, minlength=3))))
    local_folds = make_folds(inner_y, inner_groups, n_splits, seed + 1009 * (outer_fold_index + 1))
    source_f1 = []
    for source in sources:
        predictions = np.full(len(outer_train), -1, dtype=int)
        for local_train, local_valid in local_folds:
            global_train = outer_train[local_train]
            global_valid = outer_train[local_valid]
            predicted, _, _, _ = fit_predict_fold(
                source["original"],
                None,
                y,
                global_train,
                global_valid,
                False,
                "original",
            )
            predictions[local_valid] = predicted
        source_f1.append(
            f1_score(inner_y, predictions, labels=np.arange(3), average=None, zero_division=0)
        )
    # Source zero is the stable baseline and wins exact ties.
    return tuple(int(value) for value in np.asarray(source_f1).argmax(axis=0))


def fit_predict_fold(
    original: np.ndarray,
    flip: np.ndarray | None,
    y: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    train_augmented: bool,
    test_rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if train_augmented:
        if flip is None:
            raise ValueError("Training augmentation requires a flipped feature matrix.")
        x_train = np.concatenate((original[train], flip[train]), axis=0)
        y_train = np.concatenate((y[train], y[train]), axis=0)
        sample_weight = np.full(len(x_train), 0.5, dtype=float)
    else:
        x_train = original[train]
        y_train = y[train]
        sample_weight = None

    scaler = StandardScaler()
    scaler.fit(x_train, sample_weight=sample_weight)
    x_train = scaler.transform(x_train)
    classifier = LogisticRegression(max_iter=5000, C=3.0, class_weight="balanced")
    classifier.fit(x_train, y_train, sample_weight=sample_weight)

    prob_original = classifier.predict_proba(scaler.transform(original[valid]))
    prob_flip = None
    if flip is not None:
        prob_flip = classifier.predict_proba(scaler.transform(flip[valid]))

    if test_rule == "original":
        probabilities = prob_original
    elif test_rule == "embedding_avg":
        if flip is None:
            raise ValueError("Embedding averaging requires flipped features.")
        averaged = 0.5 * (
            scaler.transform(original[valid]) + scaler.transform(flip[valid])
        )
        probabilities = classifier.predict_proba(averaged)
    elif test_rule == "probability_avg":
        if prob_flip is None:
            raise ValueError("Probability averaging requires flipped features.")
        probabilities = 0.5 * (prob_original + prob_flip)
    else:
        raise ValueError(test_rule)
    return probabilities.argmax(axis=1), probabilities, prob_original, prob_flip


def evaluate_experiment(
    experiment: dict[str, object],
    y: np.ndarray,
    groups: np.ndarray,
    other_subgroups: np.ndarray,
    folds,
    seed: int,
) -> tuple[str, dict[str, object]]:
    predictions = np.full(len(y), -1, dtype=int)
    probabilities = np.zeros((len(y), len(LABEL_NAMES)), dtype=float)
    original_probabilities = np.zeros_like(probabilities)
    if "late_sources" in experiment:
        has_flip = all(source["flip"] is not None for source in experiment["late_sources"])
    elif "class_route_sources" in experiment:
        has_flip = False
    else:
        has_flip = experiment["flip"] is not None
    flip_probabilities = np.zeros_like(probabilities) if has_flip else None
    selected_routes = []
    for outer_fold_index, (train, valid) in enumerate(folds):
        if "class_route_sources" in experiment:
            sources = experiment["class_route_sources"]
            if experiment.get("nested_route"):
                route = choose_nested_class_route(
                    sources, y, groups, train, seed, outer_fold_index
                )
            else:
                route = tuple(experiment["fixed_route"])
            selected_routes.append(route)
            source_results = [
                fit_predict_fold(
                    source["original"], None, y, train, valid, False, "original"
                )
                for source in sources
            ]
            prob = np.column_stack(
                [source_results[route[class_index]][1][:, class_index] for class_index in range(3)]
            )
            prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
            prob_original = prob
            prob_flip = None
            predicted = prob.argmax(axis=1)
        elif "late_sources" in experiment:
            source_results = [
                fit_predict_fold(
                    source["original"],
                    source["flip"],
                    y,
                    train,
                    valid,
                    bool(experiment["train_augmented"]),
                    str(experiment["test_rule"]),
                )
                for source in experiment["late_sources"]
            ]
            prob = np.mean([result[1] for result in source_results], axis=0)
            prob_original = np.mean([result[2] for result in source_results], axis=0)
            prob_flip = (
                np.mean([result[3] for result in source_results], axis=0) if has_flip else None
            )
            predicted = prob.argmax(axis=1)
        else:
            predicted, prob, prob_original, prob_flip = fit_predict_fold(
                experiment["original"],
                experiment["flip"],
                y,
                train,
                valid,
                bool(experiment["train_augmented"]),
                str(experiment["test_rule"]),
            )
        predictions[valid] = predicted
        probabilities[valid] = prob
        original_probabilities[valid] = prob_original
        if flip_probabilities is not None and prob_flip is not None:
            flip_probabilities[valid] = prob_flip

    metrics = M.compute_metrics(y, predictions, len(LABEL_NAMES))
    row: dict[str, object] = {
        "seed": seed,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_auprc": macro_auprc(y, probabilities),
        "per_class_precision": metrics["per_class"]["precision"],
        "per_class_recall": metrics["per_class"]["recall"],
        "per_class_f1": metrics["per_class"]["f1"],
        "normalized_confusion": normalized_confusion(y, predictions),
        "other_subgroup_recall": {
            subgroup: float(np.mean(predictions[other_subgroups == subgroup] == 2))
            for subgroup in ("mca_other", "aca", "pca", "unclassified")
            if np.any(other_subgroups == subgroup)
        },
    }
    if flip_probabilities is not None:
        midpoint = np.clip(0.5 * (original_probabilities + flip_probabilities), 1e-12, 1.0)
        clipped_original = np.clip(original_probabilities, 1e-12, 1.0)
        clipped_flip = np.clip(flip_probabilities, 1e-12, 1.0)
        js = 0.5 * (
            (clipped_original * np.log(clipped_original / midpoint)).sum(axis=1)
            + (clipped_flip * np.log(clipped_flip / midpoint)).sum(axis=1)
        )
        row["flip_prediction_disagreement"] = float(
            np.mean(original_probabilities.argmax(axis=1) != flip_probabilities.argmax(axis=1))
        )
        row["flip_js_divergence"] = float(js.mean())
    if selected_routes:
        row["selected_routes"] = [list(route) for route in selected_routes]
    return str(experiment["name"]), row


def evaluate_seed(seed, experiments, y, groups, other_subgroups, n_splits):
    folds = make_folds(y, groups, n_splits, seed)
    return [
        evaluate_experiment(experiment, y, groups, other_subgroups, folds, seed)
        for experiment in experiments
    ]


def array_summary(rows: list[dict[str, object]], key: str) -> tuple[object, object]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return mean.tolist() if mean.ndim else float(mean), std.tolist() if std.ndim else float(std)


def summarize(per_experiment, reference_name: str) -> list[dict[str, object]]:
    reference = np.asarray([row["macro_f1"] for row in per_experiment[reference_name]], dtype=float)
    summaries = []
    for name, rows in per_experiment.items():
        scores = np.asarray([row["macro_f1"] for row in rows], dtype=float)
        delta = scores - reference
        macro_mean, macro_std = array_summary(rows, "macro_f1")
        bal_mean, bal_std = array_summary(rows, "balanced_accuracy")
        auprc_mean, auprc_std = array_summary(rows, "macro_auprc")
        precision_mean, precision_std = array_summary(rows, "per_class_precision")
        recall_mean, recall_std = array_summary(rows, "per_class_recall")
        f1_mean, f1_std = array_summary(rows, "per_class_f1")
        confusion_mean, confusion_std = array_summary(rows, "normalized_confusion")
        summary = {
            "name": name,
            "seed0_macro_f1": float(rows[0]["macro_f1"]) if rows and rows[0]["seed"] == 0 else None,
            "macro_f1_mean": macro_mean,
            "macro_f1_std": macro_std,
            "balanced_accuracy_mean": bal_mean,
            "balanced_accuracy_std": bal_std,
            "macro_auprc_mean": auprc_mean,
            "macro_auprc_std": auprc_std,
            "delta_vs_reference_mean": float(delta.mean()),
            "delta_vs_reference_min": float(delta.min()),
            "delta_vs_reference_max": float(delta.max()),
            "wins_vs_reference": int((delta > 0).sum()),
            "ties_vs_reference": int((delta == 0).sum()),
            "per_class_precision_mean": precision_mean,
            "per_class_precision_std": precision_std,
            "per_class_recall_mean": recall_mean,
            "per_class_recall_std": recall_std,
            "per_class_f1_mean": f1_mean,
            "per_class_f1_std": f1_std,
            "normalized_confusion_mean": confusion_mean,
            "normalized_confusion_std": confusion_std,
            "other_subgroup_recall_mean": {
                subgroup: float(np.mean([row["other_subgroup_recall"][subgroup] for row in rows]))
                for subgroup in rows[0]["other_subgroup_recall"]
            },
            "other_subgroup_recall_std": {
                subgroup: float(np.std(
                    [row["other_subgroup_recall"][subgroup] for row in rows],
                    ddof=1 if len(rows) > 1 else 0,
                ))
                for subgroup in rows[0]["other_subgroup_recall"]
            },
        }
        if "flip_prediction_disagreement" in rows[0]:
            summary["flip_prediction_disagreement_mean"], summary["flip_prediction_disagreement_std"] = (
                array_summary(rows, "flip_prediction_disagreement")
            )
            summary["flip_js_divergence_mean"], summary["flip_js_divergence_std"] = array_summary(
                rows, "flip_js_divergence"
            )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: row["macro_f1_mean"], reverse=True)


def main() -> int:
    args = parse_args()
    named_paths = {
        "vjepa_original": args.vjepa_original,
        "vjepa_flip": args.vjepa_flip,
        "dino_original": args.dino_original,
        "dino_flip": args.dino_flip,
        **{f"variant_{name}": path for name, path in args.variant},
    }
    if len(named_paths) != 4 + len(args.variant):
        raise ValueError("Every --variant name must be unique and must not shadow a base source.")
    loaded = {name: load_cache(path) for name, path in named_paths.items()}
    reference_identity = loaded["vjepa_original"]["metadata_identity"]
    loaded = {
        name: aligned_to_reference(cache, reference_identity) for name, cache in loaded.items()
    }
    y = np.asarray(loaded["vjepa_original"]["labels"], dtype=int)
    groups = np.asarray(loaded["vjepa_original"]["groups"], dtype=object)
    other_subgroups = np.asarray(
        [
            other_subgroup(identity[4], int(target))
            for identity, target in zip(reference_identity, y)
        ],
        dtype=object,
    )
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError(f"Expected direct three-class labels 0/1/2, found {np.unique(y).tolist()}.")
    for name, cache in loaded.items():
        if not np.array_equal(cache["labels"], y):
            raise ValueError(f"Labels in {name} do not match after metadata alignment.")
        if not np.array_equal(cache["groups"], groups):
            raise ValueError(f"Patient groups in {name} do not match after metadata alignment.")

    vjepa_original = np.asarray(loaded["vjepa_original"]["mean"], dtype=np.float32)
    vjepa_flip = np.asarray(loaded["vjepa_flip"]["mean"], dtype=np.float32)
    dino_original = np.asarray(loaded["dino_original"]["mean"], dtype=np.float32)
    dino_flip = np.asarray(loaded["dino_flip"]["mean"], dtype=np.float32)
    features = {
        "vjepa": {"original": vjepa_original, "flip": vjepa_flip},
        "dino": {"original": dino_original, "flip": dino_flip},
        "fusion": {
            "original": np.concatenate((vjepa_original, dino_original), axis=1),
            "flip": np.concatenate((vjepa_flip, dino_flip), axis=1),
        },
    }
    for name, _ in args.variant:
        variant = np.asarray(loaded[f"variant_{name}"]["mean"], dtype=np.float32)
        features[f"dino_{name}"] = {"original": variant, "flip": None}
        features[f"fusion_{name}"] = {
            "original": np.concatenate((vjepa_original, variant), axis=1),
            "flip": None,
        }

    experiments = build_experiments(features)
    n_splits = min(
        args.folds,
        len(set(groups)),
        int(np.min(np.bincount(y, minlength=len(LABEL_NAMES)))),
    )
    evaluated = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(evaluate_seed)(seed, experiments, y, groups, other_subgroups, n_splits)
        for seed in args.seeds
    )
    per_experiment = {str(experiment["name"]): [] for experiment in experiments}
    for seed_rows in evaluated:
        for name, row in seed_rows:
            per_experiment[name].append(row)
    for rows in per_experiment.values():
        rows.sort(key=lambda row: row["seed"])

    reference_name = "vjepa/train_original/test_original"
    summaries = summarize(per_experiment, reference_name)
    widths = (43, 12, 12, 12, 9)
    print("  ".join(value.ljust(width) for value, width in zip(
        ("experiment", "macroF1", "class F1", "delta ref", "wins"), widths
    )))
    print("-" * (sum(widths) + 2 * len(widths)))
    for row in summaries:
        values = (
            row["name"][: widths[0]],
            f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}",
            "/".join(f"{value:.2f}" for value in row["per_class_f1_mean"]),
            f"{row['delta_vs_reference_mean']:+.3f}",
            f"{row['wins_vs_reference']}/{len(args.seeds)}",
        )
        print("  ".join(str(value).ljust(width) for value, width in zip(values, widths)))

    protocol_hash = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    output = {
        "protocol": {
            "target": list(LABEL_NAMES),
            "other_definition": "positive run not mapped to M2 or M3; may include MCA, ACA, or PCA",
            "endpoint": "run-level",
            "seeds": args.seeds,
            "folds": n_splits,
            "group": "Study_Key",
            "classifier": "StandardScaler + balanced multinomial LogisticRegression(C=3)",
            "train_augmentation_weight_per_view": 0.5,
            "reference": reference_name,
            "n_samples": len(y),
            "n_groups": len(set(groups)),
            "class_counts": {
                name: int((y == index).sum()) for index, name in enumerate(LABEL_NAMES)
            },
            "other_subgroup_counts": {
                subgroup: int(np.sum(other_subgroups == subgroup))
                for subgroup in ("mca_other", "aca", "pca", "unclassified")
                if np.any(other_subgroups == subgroup)
            },
            "script_sha256": protocol_hash,
            "feature_sources": {
                name: {
                    "path": cache["path"],
                    "sha256": cache["sha256"],
                    "signature": cache["signature"],
                }
                for name, cache in loaded.items()
            },
        },
        "summaries": summaries,
        "per_seed": per_experiment,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(output, handle, indent=2, default=float)
    print(f"\nWrote direct three-class comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
