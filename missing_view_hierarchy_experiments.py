"""Test paired and genuinely unpaired studies with a missing-view feature hierarchy.

Each strict run identity contributes exactly one evaluation row. Concordant AP/lateral
pairs use both views; AP-only and lateral-only identities use their available view. The
inputs are frozen, per-run AP and lateral feature caches and may come from any backbone or
video representation that follows the project cache schema. The historical ``--ap-temporal``
and ``--lat-temporal`` option names are retained for command compatibility; the hierarchy
does not assume DINO features or temporal-mean pooling. Models, preprocessing, gate/blend
selection, and every split remain patient-grouped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed

import metrics as M
from anatomy_task_experiments import load_cache
from experiments import LINEAR_SVM_DUAL, LINEAR_SVM_MAX_ITER, LINEAR_SVM_TOL, make_folds
from multiview_anatomy_experiments import LABELS, strict_label
from multiview_ensemble_experiments import (
    THRESHOLD_GRID,
    fit_standardized,
    hierarchy_prediction,
    hierarchy_scores,
    parse_seeds,
    predict_logreg,
    predict_svm,
    selection_metrics,
    source_record,
    split_count,
    unique_pair_map,
)


PAIR_WEIGHT_GRID = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
PAIRED = "paired"
AP_ONLY = "ap_only"
LATERAL_ONLY = "lateral_only"

METHOD_DEFINITIONS = {
    "paired_concat_hierarchy": (
        "Paired-only concatenated AP+lateral PCA gate with paired-only AP M2/M3 expert."
    ),
    "paired_viewwise_hierarchy": (
        "Paired-only AP and lateral PCA gates blended late; paired-only AP M2/M3 expert."
    ),
    "augmented_hierarchy_paired_tuned": (
        "View experts trained on every available strict case; pair weight and threshold tuned on paired inner OOF."
    ),
    "augmented_hierarchy_union_tuned": (
        "View experts trained on every available strict case; pair weight and threshold tuned on union inner OOF."
    ),
    "paired_concat_logreg": "Paired-only direct AP+lateral three-class logistic regression.",
    "missing_view_logreg_paired_tuned": (
        "AP/lateral three-class experts trained on all available cases; pair blend tuned on paired inner OOF."
    ),
    "missing_view_logreg_union_tuned": (
        "AP/lateral three-class experts trained on all available cases; pair blend tuned on union inner OOF."
    ),
}

PAIRED_ONLY_METHODS = {
    "paired_concat_hierarchy",
    "paired_viewwise_hierarchy",
    "paired_concat_logreg",
}


@dataclass(frozen=True)
class UnionCohort:
    """Aligned frozen run features with explicit AP/lateral availability."""

    ap: np.ndarray
    lateral: np.ndarray
    ap_available: np.ndarray
    lateral_available: np.ndarray
    availability: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    identities: tuple[str, ...]
    audit: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ap-temporal",
        type=Path,
        required=True,
        help=(
            "AP frozen-feature cache. The option name is retained for compatibility; "
            "features may come from any backbone or video representation."
        ),
    )
    parser.add_argument(
        "--lat-temporal",
        type=Path,
        required=True,
        help=(
            "Lateral frozen-feature cache. The option name is retained for compatibility; "
            "features may come from any backbone or video representation."
        ),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:20"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/missing_view_hierarchy/results_nested_20seeds.json"),
    )
    return parser.parse_args()


def build_union_cohort(ap_cache: dict[str, object], lat_cache: dict[str, object]) -> UnionCohort:
    """Align arbitrary AP/lateral run-feature caches into the missing-view cohort.

    Each cache supplies its frozen representation through the standard ``mean`` array. The
    current union matrix stores the two views at one common feature width and therefore
    rejects mismatched widths; no assumption is made about how a backbone produced those
    vectors.
    """

    ap_map = unique_pair_map(ap_cache, "AP")
    lat_map = unique_pair_map(lat_cache, "lateral")
    rows: list[tuple[tuple[str, str, int], int | None, int | None, int, str]] = []
    exclusions = {
        "discordant_strict_pairs": [],
        "paired_nonstrict": 0,
        "ap_only_nonstrict": 0,
        "lateral_only_nonstrict": 0,
    }
    for key in sorted(set(ap_map).union(lat_map)):
        ap_index = ap_map.get(key)
        lat_index = lat_map.get(key)
        ap_label = None if ap_index is None else strict_label(ap_cache["meta"][ap_index]["label_text"])
        lat_label = None if lat_index is None else strict_label(lat_cache["meta"][lat_index]["label_text"])
        if ap_index is not None and lat_index is not None:
            if ap_label is not None and ap_label == lat_label:
                rows.append((key, ap_index, lat_index, ap_label, PAIRED))
            elif ap_label is not None and lat_label is not None:
                exclusions["discordant_strict_pairs"].append(
                    {
                        "study_key": key[0],
                        "accession": key[1],
                        "run_index": key[2],
                        "ap_label": LABELS[ap_label],
                        "lateral_label": LABELS[lat_label],
                    }
                )
            else:
                exclusions["paired_nonstrict"] += 1
        elif ap_index is not None:
            if ap_label is not None:
                rows.append((key, ap_index, None, ap_label, AP_ONLY))
            else:
                exclusions["ap_only_nonstrict"] += 1
        elif lat_label is not None:
            rows.append((key, None, lat_index, lat_label, LATERAL_ONLY))
        else:
            exclusions["lateral_only_nonstrict"] += 1

    if not rows:
        raise ValueError("No strict paired or unpaired cases were found.")
    ap_dimension = int(np.asarray(ap_cache["mean"]).shape[1])
    lat_dimension = int(np.asarray(lat_cache["mean"]).shape[1])
    if ap_dimension != lat_dimension:
        raise ValueError(f"AP/lateral feature dimensions differ: {ap_dimension} vs {lat_dimension}.")
    ap = np.full((len(rows), ap_dimension), np.nan, dtype=np.float32)
    lateral = np.full_like(ap, np.nan)
    ap_available = np.zeros(len(rows), dtype=bool)
    lat_available = np.zeros(len(rows), dtype=bool)
    groups = np.empty(len(rows), dtype=object)
    labels = np.empty(len(rows), dtype=int)
    availability = np.empty(len(rows), dtype=object)
    identities = []
    for row_index, (key, ap_index, lat_index, label, kind) in enumerate(rows):
        labels[row_index] = label
        availability[row_index] = kind
        groups[row_index] = key[0]
        identities.append(f"{key[0]}:{key[1]}:{key[2]}")
        if ap_index is not None:
            if str(ap_cache["groups"][ap_index]) != key[0]:
                raise ValueError(f"AP cache group differs from Study_Key for {key}.")
            ap[row_index] = ap_cache["mean"][ap_index]
            ap_available[row_index] = True
        if lat_index is not None:
            if str(lat_cache["groups"][lat_index]) != key[0]:
                raise ValueError(f"Lateral cache group differs from Study_Key for {key}.")
            lateral[row_index] = lat_cache["mean"][lat_index]
            lat_available[row_index] = True

    if not np.isfinite(ap[ap_available]).all() or not np.isfinite(lateral[lat_available]).all():
        raise ValueError("Available feature rows contain non-finite values.")
    if np.any(ap_available != np.isin(availability, (PAIRED, AP_ONLY))):
        raise RuntimeError("AP availability audit failed.")
    if np.any(lat_available != np.isin(availability, (PAIRED, LATERAL_ONLY))):
        raise RuntimeError("Lateral availability audit failed.")

    counts_by_availability = {}
    for kind in (PAIRED, AP_ONLY, LATERAL_ONLY):
        keep = availability == kind
        counts_by_availability[kind] = {
            "n_cases": int(np.sum(keep)),
            "n_groups": len(set(groups[keep])),
            "class_counts": {LABELS[index]: int(np.sum(labels[keep] == index)) for index in range(3)},
        }
    audit = {
        "n_cases": len(rows),
        "n_groups": len(set(groups)),
        "class_counts": {LABELS[index]: int(np.sum(labels == index)) for index in range(3)},
        "groups_per_class": {
            LABELS[index]: len(set(groups[labels == index])) for index in range(3)
        },
        "by_availability": counts_by_availability,
        "feature_dimension_per_view": ap_dimension,
        "exclusions": exclusions,
        "case_definition": (
            "concordant strict pair, or strict single-view identity with no counterpart; "
            "discordant and non-strict matched identities excluded"
        ),
    }
    return UnionCohort(
        ap=ap,
        lateral=lateral,
        ap_available=ap_available,
        lateral_available=lat_available,
        availability=availability,
        y=labels,
        groups=groups,
        identities=tuple(identities),
        audit=audit,
    )


def pair_features(cohort: UnionCohort, indices: np.ndarray) -> np.ndarray:
    """Concatenate the two available frozen view vectors for genuinely paired cases."""

    return np.concatenate((cohort.ap[indices], cohort.lateral[indices]), axis=1)


def empty_primitives(length: int) -> dict[str, np.ndarray]:
    return {
        "pair_gate_concat": np.full(length, np.nan),
        "pair_gate_ap": np.full(length, np.nan),
        "pair_gate_lateral": np.full(length, np.nan),
        "pair_segment_ap_pred": np.full(length, -1, dtype=int),
        "pair_segment_ap_decision": np.full(length, np.nan),
        "pair_probability_concat": np.full((length, 3), np.nan),
        "all_gate_ap": np.full(length, np.nan),
        "all_gate_lateral": np.full(length, np.nan),
        "all_segment_ap_pred": np.full(length, -1, dtype=int),
        "all_segment_ap_decision": np.full(length, np.nan),
        "all_segment_lateral_pred": np.full(length, -1, dtype=int),
        "all_segment_lateral_decision": np.full(length, np.nan),
        "all_probability_ap": np.full((length, 3), np.nan),
        "all_probability_lateral": np.full((length, 3), np.nan),
    }


def fit_fold_primitives(
    cohort: UnionCohort,
    train: np.ndarray,
    valid: np.ndarray,
    context: str,
) -> tuple[dict[str, np.ndarray], int]:
    output = empty_primitives(len(valid))
    maximum_iterations = 0
    train_pair = train[cohort.availability[train] == PAIRED]
    valid_pair_local = np.flatnonzero(cohort.availability[valid] == PAIRED)
    valid_pair = valid[valid_pair_local]
    pair_gate_labels = (cohort.y[train_pair] == 2).astype(int)

    fitted = fit_standardized(
        pair_features(cohort, train_pair), pair_gate_labels, "logreg", f"{context}/pair_gate_concat"
    )
    maximum_iterations = max(maximum_iterations, fitted[2])
    output["pair_gate_concat"][valid_pair_local] = predict_logreg(
        fitted, pair_features(cohort, valid_pair)
    )[:, 1]
    for name, features in (("pair_gate_ap", cohort.ap), ("pair_gate_lateral", cohort.lateral)):
        fitted = fit_standardized(
            features[train_pair], pair_gate_labels, "logreg", f"{context}/{name}"
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        output[name][valid_pair_local] = predict_logreg(fitted, features[valid_pair])[:, 1]

    pair_mca_train = train_pair[cohort.y[train_pair] != 2]
    fitted = fit_standardized(
        cohort.ap[pair_mca_train],
        cohort.y[pair_mca_train],
        "svm",
        f"{context}/pair_segment_ap",
    )
    maximum_iterations = max(maximum_iterations, fitted[2])
    prediction, decision = predict_svm(fitted, cohort.ap[valid_pair])
    output["pair_segment_ap_pred"][valid_pair_local] = prediction
    output["pair_segment_ap_decision"][valid_pair_local] = decision[:, 1]

    fitted = fit_standardized(
        pair_features(cohort, train_pair),
        cohort.y[train_pair],
        "logreg",
        f"{context}/pair_probability_concat",
    )
    maximum_iterations = max(maximum_iterations, fitted[2])
    output["pair_probability_concat"][valid_pair_local] = predict_logreg(
        fitted, pair_features(cohort, valid_pair)
    )

    for view, features, available in (
        ("ap", cohort.ap, cohort.ap_available),
        ("lateral", cohort.lateral, cohort.lateral_available),
    ):
        view_train = train[available[train]]
        view_valid_local = np.flatnonzero(available[valid])
        view_valid = valid[view_valid_local]
        fitted = fit_standardized(
            features[view_train],
            (cohort.y[view_train] == 2).astype(int),
            "logreg",
            f"{context}/all_gate_{view}",
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        output[f"all_gate_{view}"][view_valid_local] = predict_logreg(
            fitted, features[view_valid]
        )[:, 1]

        mca_train = view_train[cohort.y[view_train] != 2]
        fitted = fit_standardized(
            features[mca_train],
            cohort.y[mca_train],
            "svm",
            f"{context}/all_segment_{view}",
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        prediction, decision = predict_svm(fitted, features[view_valid])
        output[f"all_segment_{view}_pred"][view_valid_local] = prediction
        output[f"all_segment_{view}_decision"][view_valid_local] = decision[:, 1]

        fitted = fit_standardized(
            features[view_train],
            cohort.y[view_train],
            "logreg",
            f"{context}/all_probability_{view}",
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        output[f"all_probability_{view}"][view_valid_local] = predict_logreg(
            fitted, features[view_valid]
        )
    return output, maximum_iterations


def validate_primitives(
    primitives: dict[str, np.ndarray],
    availability: np.ndarray,
) -> None:
    pair = availability == PAIRED
    ap = np.isin(availability, (PAIRED, AP_ONLY))
    lateral = np.isin(availability, (PAIRED, LATERAL_ONLY))
    expected = {
        "pair_gate_concat": pair,
        "pair_gate_ap": pair,
        "pair_gate_lateral": pair,
        "pair_segment_ap_pred": pair,
        "pair_segment_ap_decision": pair,
        "pair_probability_concat": pair,
        "all_gate_ap": ap,
        "all_segment_ap_pred": ap,
        "all_segment_ap_decision": ap,
        "all_probability_ap": ap,
        "all_gate_lateral": lateral,
        "all_segment_lateral_pred": lateral,
        "all_segment_lateral_decision": lateral,
        "all_probability_lateral": lateral,
    }
    for name, mask in expected.items():
        values = primitives[name][mask]
        if np.issubdtype(values.dtype, np.integer):
            valid = np.all(values >= 0)
        else:
            valid = np.isfinite(values).all()
        if not valid:
            raise RuntimeError(f"Incomplete primitive {name} on its available-view rows.")


def crossfit_primitives(
    cohort: UnionCohort,
    outer_train: np.ndarray,
    inner_splits,
    context: str,
) -> tuple[dict[str, np.ndarray], int]:
    output = empty_primitives(len(outer_train))
    maximum_iterations = 0
    for inner_index, (inner_train, inner_valid) in enumerate(inner_splits):
        fold, iterations = fit_fold_primitives(
            cohort,
            outer_train[inner_train],
            outer_train[inner_valid],
            f"{context}/inner={inner_index}",
        )
        maximum_iterations = max(maximum_iterations, iterations)
        for name in output:
            output[name][inner_valid] = fold[name]
    validate_primitives(output, cohort.availability[outer_train])
    return output, maximum_iterations


def paired_concat_prediction(
    primitives: dict[str, np.ndarray], threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    prediction = hierarchy_prediction(
        primitives["pair_gate_concat"], primitives["pair_segment_ap_pred"], threshold
    )
    scores = hierarchy_scores(
        primitives["pair_gate_concat"], primitives["pair_segment_ap_decision"]
    )
    return prediction, scores


def viewwise_hierarchy_outputs(
    primitives: dict[str, np.ndarray],
    availability: np.ndarray,
    lateral_weight: float,
    threshold: float,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(availability)
    gate = np.full(length, np.nan)
    segment_prediction = np.full(length, -1, dtype=int)
    segment_decision = np.full(length, np.nan)
    pair = availability == PAIRED
    ap_only = availability == AP_ONLY
    lateral_only = availability == LATERAL_ONLY
    gate[pair] = (
        (1.0 - lateral_weight) * primitives[f"{prefix}_gate_ap"][pair]
        + lateral_weight * primitives[f"{prefix}_gate_lateral"][pair]
    )
    segment_prediction[pair] = primitives[f"{prefix}_segment_ap_pred"][pair]
    segment_decision[pair] = primitives[f"{prefix}_segment_ap_decision"][pair]
    if prefix == "all":
        gate[ap_only] = primitives["all_gate_ap"][ap_only]
        segment_prediction[ap_only] = primitives["all_segment_ap_pred"][ap_only]
        segment_decision[ap_only] = primitives["all_segment_ap_decision"][ap_only]
        gate[lateral_only] = primitives["all_gate_lateral"][lateral_only]
        segment_prediction[lateral_only] = primitives["all_segment_lateral_pred"][lateral_only]
        segment_decision[lateral_only] = primitives["all_segment_lateral_decision"][lateral_only]
    expected = pair if prefix == "pair" else np.ones(length, dtype=bool)
    if not np.isfinite(gate[expected]).all() or np.any(segment_prediction[expected] < 0):
        raise RuntimeError(f"Incomplete {prefix} viewwise hierarchy outputs.")
    return hierarchy_prediction(gate, segment_prediction, threshold), hierarchy_scores(
        gate, segment_decision
    )


def direct_missing_view_probability(
    primitives: dict[str, np.ndarray],
    availability: np.ndarray,
    lateral_weight: float,
) -> np.ndarray:
    probability = np.full((len(availability), 3), np.nan)
    pair = availability == PAIRED
    ap_only = availability == AP_ONLY
    lateral_only = availability == LATERAL_ONLY
    probability[pair] = (
        (1.0 - lateral_weight) * primitives["all_probability_ap"][pair]
        + lateral_weight * primitives["all_probability_lateral"][pair]
    )
    probability[ap_only] = primitives["all_probability_ap"][ap_only]
    probability[lateral_only] = primitives["all_probability_lateral"][lateral_only]
    if not np.isfinite(probability).all():
        raise RuntimeError("Incomplete direct missing-view probabilities.")
    return probability


def select_concat_threshold(
    primitives: dict[str, np.ndarray], labels: np.ndarray, pair_mask: np.ndarray
) -> dict[str, float]:
    candidates = []
    for threshold in THRESHOLD_GRID:
        prediction, _ = paired_concat_prediction(primitives, float(threshold))
        macro, m3 = selection_metrics(labels[pair_mask], prediction[pair_mask])
        candidates.append(
            {"threshold": float(threshold), "inner_macro_f1": macro, "inner_m3_f1": m3}
        )
    return min(
        candidates,
        key=lambda row: (
            -row["inner_macro_f1"],
            -row["inner_m3_f1"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )


def select_viewwise_hierarchy(
    primitives: dict[str, np.ndarray],
    labels: np.ndarray,
    availability: np.ndarray,
    selection_mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    candidates = []
    for lateral_weight in PAIR_WEIGHT_GRID:
        for threshold in THRESHOLD_GRID:
            prediction, _ = viewwise_hierarchy_outputs(
                primitives, availability, float(lateral_weight), float(threshold), prefix
            )
            macro, m3 = selection_metrics(labels[selection_mask], prediction[selection_mask])
            candidates.append(
                {
                    "lateral_weight": float(lateral_weight),
                    "threshold": float(threshold),
                    "inner_macro_f1": macro,
                    "inner_m3_f1": m3,
                }
            )
    return min(
        candidates,
        key=lambda row: (
            -row["inner_macro_f1"],
            -row["inner_m3_f1"],
            abs(row["lateral_weight"] - 0.5),
            abs(row["threshold"] - 0.5),
            row["lateral_weight"],
            row["threshold"],
        ),
    )


def select_direct_weight(
    primitives: dict[str, np.ndarray],
    labels: np.ndarray,
    availability: np.ndarray,
    selection_mask: np.ndarray,
) -> dict[str, float]:
    candidates = []
    for lateral_weight in PAIR_WEIGHT_GRID:
        probability = direct_missing_view_probability(
            primitives, availability, float(lateral_weight)
        )
        macro, m3 = selection_metrics(
            labels[selection_mask], np.argmax(probability[selection_mask], axis=1)
        )
        candidates.append(
            {
                "lateral_weight": float(lateral_weight),
                "inner_macro_f1": macro,
                "inner_m3_f1": m3,
            }
        )
    return min(
        candidates,
        key=lambda row: (
            -row["inner_macro_f1"],
            -row["inner_m3_f1"],
            abs(row["lateral_weight"] - 0.5),
            row["lateral_weight"],
        ),
    )


def metrics_row(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    metric = M.compute_metrics(labels, predictions, 3)
    return {
        "macro_f1": metric["macro_f1"],
        "balanced_accuracy": metric["balanced_accuracy"],
        "accuracy": metric["accuracy"],
        "per_class_precision": metric["per_class"]["precision"],
        "per_class_recall": metric["per_class"]["recall"],
        "per_class_f1": metric["per_class"]["f1"],
        "confusion": metric["confusion"],
        "n_cases": len(labels),
    }


def evaluation_masks(availability: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "union": np.ones(len(availability), dtype=bool),
        "paired": availability == PAIRED,
        "ap_only": availability == AP_ONLY,
        "lateral_only": availability == LATERAL_ONLY,
    }


def evaluate_seed(
    cohort: UnionCohort,
    seed: int,
    requested_folds: int,
    requested_inner_folds: int,
) -> tuple[int, dict[str, dict[str, object]]]:
    fold_count = split_count(cohort.y, cohort.groups, requested_folds)
    outer_splits = make_folds(cohort.y, cohort.groups, fold_count, seed)
    predictions = {name: np.full(len(cohort.y), -1, dtype=int) for name in METHOD_DEFINITIONS}
    scores = {name: np.full((len(cohort.y), 3), np.nan) for name in METHOD_DEFINITIONS}
    selected = {name: [] for name in METHOD_DEFINITIONS}
    maximum_iterations = {name: 0 for name in METHOD_DEFINITIONS}

    for outer_index, (train, valid) in enumerate(outer_splits):
        inner_count = split_count(cohort.y[train], cohort.groups[train], requested_inner_folds)
        inner_splits = make_folds(
            cohort.y[train], cohort.groups[train], inner_count, seed + 1009 * (outer_index + 1)
        )
        context = f"seed={seed}/outer={outer_index}"
        inner, inner_iterations = crossfit_primitives(cohort, train, inner_splits, context)
        inner_y = cohort.y[train]
        inner_availability = cohort.availability[train]
        inner_pair = inner_availability == PAIRED
        concat_parameters = select_concat_threshold(inner, inner_y, inner_pair)
        paired_viewwise_parameters = select_viewwise_hierarchy(
            inner, inner_y, inner_availability, inner_pair, "pair"
        )
        augmented_paired_parameters = select_viewwise_hierarchy(
            inner, inner_y, inner_availability, inner_pair, "all"
        )
        augmented_union_parameters = select_viewwise_hierarchy(
            inner,
            inner_y,
            inner_availability,
            np.ones(len(inner_y), dtype=bool),
            "all",
        )
        direct_paired_parameters = select_direct_weight(
            inner, inner_y, inner_availability, inner_pair
        )
        direct_union_parameters = select_direct_weight(
            inner,
            inner_y,
            inner_availability,
            np.ones(len(inner_y), dtype=bool),
        )

        outer, outer_iterations = fit_fold_primitives(cohort, train, valid, context)
        validate_primitives(outer, cohort.availability[valid])
        primitive_iterations = max(inner_iterations, outer_iterations)
        valid_availability = cohort.availability[valid]
        valid_pair_local = valid_availability == PAIRED
        valid_pair_absolute = valid[valid_pair_local]

        prediction, method_scores = paired_concat_prediction(
            outer, concat_parameters["threshold"]
        )
        predictions["paired_concat_hierarchy"][valid_pair_absolute] = prediction[valid_pair_local]
        scores["paired_concat_hierarchy"][valid_pair_absolute] = method_scores[valid_pair_local]
        selected["paired_concat_hierarchy"].append({"outer_fold": outer_index, **concat_parameters})

        prediction, method_scores = viewwise_hierarchy_outputs(
            outer,
            valid_availability,
            paired_viewwise_parameters["lateral_weight"],
            paired_viewwise_parameters["threshold"],
            "pair",
        )
        predictions["paired_viewwise_hierarchy"][valid_pair_absolute] = prediction[valid_pair_local]
        scores["paired_viewwise_hierarchy"][valid_pair_absolute] = method_scores[valid_pair_local]
        selected["paired_viewwise_hierarchy"].append(
            {"outer_fold": outer_index, **paired_viewwise_parameters}
        )

        for method_name, parameters in (
            ("augmented_hierarchy_paired_tuned", augmented_paired_parameters),
            ("augmented_hierarchy_union_tuned", augmented_union_parameters),
        ):
            prediction, method_scores = viewwise_hierarchy_outputs(
                outer,
                valid_availability,
                parameters["lateral_weight"],
                parameters["threshold"],
                "all",
            )
            predictions[method_name][valid] = prediction
            scores[method_name][valid] = method_scores
            selected[method_name].append({"outer_fold": outer_index, **parameters})

        paired_probability = outer["pair_probability_concat"]
        predictions["paired_concat_logreg"][valid_pair_absolute] = np.argmax(
            paired_probability[valid_pair_local], axis=1
        )
        scores["paired_concat_logreg"][valid_pair_absolute] = paired_probability[valid_pair_local]
        selected["paired_concat_logreg"].append({"outer_fold": outer_index, "fixed": True})

        for method_name, parameters in (
            ("missing_view_logreg_paired_tuned", direct_paired_parameters),
            ("missing_view_logreg_union_tuned", direct_union_parameters),
        ):
            probability = direct_missing_view_probability(
                outer, valid_availability, parameters["lateral_weight"]
            )
            predictions[method_name][valid] = np.argmax(probability, axis=1)
            scores[method_name][valid] = probability
            selected[method_name].append({"outer_fold": outer_index, **parameters})

        for method_name in METHOD_DEFINITIONS:
            maximum_iterations[method_name] = max(
                maximum_iterations[method_name], primitive_iterations
            )

    masks = evaluation_masks(cohort.availability)
    result = {}
    for method_name in METHOD_DEFINITIONS:
        scopes = ("paired",) if method_name in PAIRED_ONLY_METHODS else tuple(masks)
        scope_metrics = {}
        for scope in scopes:
            mask = masks[scope]
            if np.any(predictions[method_name][mask] < 0) or not np.isfinite(
                scores[method_name][mask]
            ).all():
                raise RuntimeError(f"Incomplete {method_name}/{scope} predictions for seed {seed}.")
            scope_metrics[scope] = metrics_row(
                cohort.y[mask], predictions[method_name][mask], scores[method_name][mask]
            )
        result[method_name] = {
            "seed": seed,
            "folds": fold_count,
            "inner_folds_requested": requested_inner_folds,
            "scopes": scope_metrics,
            "selected_parameters": selected[method_name],
            "max_fit_iterations": maximum_iterations[method_name],
            "convergence_verified": True,
            "oof_predictions": [int(value) if value >= 0 else None for value in predictions[method_name]],
            "oof_scores": [row.tolist() if np.isfinite(row).all() else None for row in scores[method_name]],
        }
    return seed, result


def aggregate_scope(rows: list[dict[str, object]], scope: str) -> dict[str, object]:
    output = {}
    for key in (
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
    ):
        values = np.asarray([row["scopes"][scope][key] for row in rows], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        output[f"{key}_mean"] = mean.tolist() if mean.ndim else float(mean)
        output[f"{key}_std"] = std.tolist() if std.ndim else float(std)
    output["n_cases"] = int(rows[0]["scopes"][scope]["n_cases"])
    output["max_fit_iterations"] = max(int(row["max_fit_iterations"]) for row in rows)
    output["convergence_verified"] = all(bool(row["convergence_verified"]) for row in rows)
    return output


def paired_delta(
    method_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
    scope: str,
) -> dict[str, object]:
    deltas = np.asarray(
        [
            float(row["scopes"][scope]["macro_f1"])
            - float(reference["scopes"][scope]["macro_f1"])
            for row, reference in zip(method_rows, reference_rows)
        ]
    )
    return {
        "scope": scope,
        "macro_f1_delta_mean": float(deltas.mean()),
        "macro_f1_delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "wins": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "losses": int(np.sum(deltas < 0)),
        "per_seed_delta": deltas.tolist(),
    }


def main() -> int:
    arguments = parse_args()
    if arguments.folds < 2 or arguments.inner_folds < 2:
        raise ValueError("Outer and inner fold counts must be at least two.")
    ap_cache = load_cache(arguments.ap_temporal)
    lat_cache = load_cache(arguments.lat_temporal)
    cohort = build_union_cohort(ap_cache, lat_cache)
    counts = cohort.audit["by_availability"]
    print(
        f"Strict union: {len(cohort.y)} cases / {len(set(cohort.groups))} groups / "
        f"paired={counts[PAIRED]['n_cases']} AP-only={counts[AP_ONLY]['n_cases']} "
        f"lateral-only={counts[LATERAL_ONLY]['n_cases']}",
        flush=True,
    )
    evaluated = Parallel(n_jobs=arguments.jobs, verbose=5)(
        delayed(evaluate_seed)(
            cohort, seed, arguments.folds, arguments.inner_folds
        )
        for seed in arguments.seeds
    )
    evaluated.sort(key=lambda item: item[0])
    per_seed = {
        method: [seed_result[method] for _, seed_result in evaluated]
        for method in METHOD_DEFINITIONS
    }
    summaries = []
    for method, definition in METHOD_DEFINITIONS.items():
        scopes = per_seed[method][0]["scopes"]
        for scope in scopes:
            summary = aggregate_scope(per_seed[method], scope)
            summary.update({"method": method, "scope": scope, "definition": definition})
            summaries.append(summary)
    summaries.sort(key=lambda row: (row["scope"], -row["macro_f1_mean"]))

    comparisons = {
        "augmented_vs_paired_viewwise_on_pairs": paired_delta(
            per_seed["augmented_hierarchy_paired_tuned"],
            per_seed["paired_viewwise_hierarchy"],
            "paired",
        ),
        "augmented_vs_concat_hierarchy_on_pairs": paired_delta(
            per_seed["augmented_hierarchy_paired_tuned"],
            per_seed["paired_concat_hierarchy"],
            "paired",
        ),
        "union_tuned_hierarchy_vs_union_tuned_logreg_on_union": paired_delta(
            per_seed["augmented_hierarchy_union_tuned"],
            per_seed["missing_view_logreg_union_tuned"],
            "union",
        ),
    }
    output = {
        "protocol": {
            "endpoint": "strict M2/M3/PCA across concordant paired and genuinely unpaired run identities",
            "labels": list(LABELS),
            "seeds": arguments.seeds,
            "requested_outer_folds": arguments.folds,
            "requested_inner_folds": arguments.inner_folds,
            "outer_group": "Study_Key over the complete union before availability subsets are formed",
            "inner_tuning": "grouped cross-fitted predictions inside each outer training fold",
            "paired_inference": "blend AP/lateral PCA gates; use AP M2/M3 expert",
            "ap_only_inference": "AP PCA gate and AP M2/M3 expert",
            "lateral_only_inference": "lateral PCA gate and lateral M2/M3 expert",
            "pair_weight_grid": PAIR_WEIGHT_GRID.tolist(),
            "threshold_grid": THRESHOLD_GRID.tolist(),
            "selection_metric": "inner macro-F1; ties use M3 F1 then neutral weight/threshold",
            "linear_svm": {
                "C": 0.5,
                "class_weight": "balanced",
                "dual": LINEAR_SVM_DUAL,
                "max_iter": LINEAR_SVM_MAX_ITER,
                "tol": LINEAR_SVM_TOL,
            },
            "logistic": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "selection_note": (
                "Repeated grouped CV is exploratory and split seeds are correlated; promotion requires a locked patient test."
            ),
        },
        "cohort_audit": cohort.audit,
        "method_definitions": METHOD_DEFINITIONS,
        "feature_sources": {
            "ap_temporal": source_record(arguments.ap_temporal, ap_cache),
            "lateral_temporal": source_record(arguments.lat_temporal, lat_cache),
        },
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "per_seed": per_seed,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w") as handle:
        json.dump(output, handle, indent=2, default=float)
    print(f"Wrote {arguments.out}")
    for row in summaries:
        if row["scope"] in ("union", "paired"):
            class_f1 = "/".join(f"{value:.3f}" for value in row["per_class_f1_mean"])
            print(
                f"{row['method']:42s} {row['scope']:7s} "
                f"{row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f}  {class_f1}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
