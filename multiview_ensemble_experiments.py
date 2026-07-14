"""Evaluate nested AP+lateral ensembles for strict M2/M3/PCA classification.

The scored cohort contains only run-index-matched AP/lateral pairs whose strict labels
agree.  Every outer and inner split is grouped by Study_Key.  Gate thresholds and blend
weights are chosen only from cross-fitted predictions within each outer training fold.

The main hypothesis is deliberately narrow: lateral DINO and mask-free spatial radiomics
may help a PCA-vs-MCA gate, while a dedicated temporal-DINO expert preserves the harder
M2-vs-M3 decision.  Direct early-fusion and late-probability controls are included on the
identical cohort and folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import metrics as M
from anatomy_task_experiments import align_cache, anatomy_codes, load_cache
from experiments import (
    LINEAR_SVM_DUAL,
    LINEAR_SVM_MAX_ITER,
    LINEAR_SVM_TOL,
    make_folds,
)
from multiview_anatomy_experiments import LABELS, run_index, strict_label


METHOD_DEFINITIONS = {
    "ap_temporal_svm": "Matched AP temporal-DINO, standardized balanced LinearSVC.",
    "lateral_temporal_svm": "Matched lateral temporal-DINO, standardized balanced LinearSVC.",
    "biview_temporal_svm": "Early AP+lateral temporal-DINO concatenation, LinearSVC.",
    "biview_all_svm": "Early AP+lateral temporal-DINO plus both spatial-radiomics vectors, LinearSVC.",
    "biview_temporal_logreg": "Early AP+lateral temporal-DINO concatenation, balanced logistic C=1.",
    "biview_all_logreg": "Early AP+lateral temporal-DINO plus spatial radiomics, balanced logistic C=1.",
    "hierarchy_dino_gate_ap_segment": "Nested PCA-vs-MCA paired-DINO gate; AP-DINO M2/M3 expert.",
    "hierarchy_blended_gate_ap_segment": "Nested blend of paired-DINO and paired-radiomics PCA gates; AP-DINO M2/M3 expert.",
    "hierarchy_blended_gate_biview_segment": "Nested blended PCA gate; AP+lateral DINO M2/M3 expert.",
    "late_dino_radiomics_logreg": "Nested one-weight blend of paired-DINO and paired-radiomics three-class logistic probabilities.",
}

THRESHOLD_GRID = np.asarray([0.30, 0.40, 0.50, 0.60, 0.70])
GATE_RADIOMICS_WEIGHT_GRID = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
LATE_RADIOMICS_WEIGHT_GRID = np.asarray([0.0, 0.10, 0.20, 0.30, 0.50])


@dataclass(frozen=True)
class MatchedCohort:
    features: dict[str, np.ndarray]
    y: np.ndarray
    groups: np.ndarray
    identities: tuple[str, ...]
    audit: dict[str, object]


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", 1))
        seeds = list(range(start, stop))
    else:
        seeds = [int(part) for part in value.split(",") if part.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique list or START:STOP.")
    return sorted(seeds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ap-temporal", type=Path, required=True)
    parser.add_argument("--lat-temporal", type=Path, required=True)
    parser.add_argument("--ap-radiomics", type=Path, required=True)
    parser.add_argument("--lat-radiomics", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:20"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/multiview_ensembles/results_nested_20seeds.json"),
    )
    return parser.parse_args()


def align_within_view(reference: dict[str, object], other: dict[str, object], name: str) -> dict[str, object]:
    aligned = align_cache(other, reference["metadata_identity"])
    if aligned["metadata_identity"] != reference["metadata_identity"]:
        raise ValueError(f"Failed to align {name} metadata.")
    if not np.array_equal(aligned["groups"], reference["groups"]):
        raise ValueError(f"Patient groups differ after aligning {name}.")
    return aligned


def unique_pair_map(cache: dict[str, object], name: str) -> dict[tuple[str, str, int], int]:
    mapping = {}
    for index, meta in enumerate(cache["meta"]):
        key = (str(meta["study_key"]), str(meta["accession"]), run_index(meta))
        if key in mapping:
            raise ValueError(f"Duplicate pair key {key} in {name} cache.")
        mapping[key] = index
    return mapping


def build_matched_cohort(
    ap_dino: dict[str, object],
    lat_dino: dict[str, object],
    ap_radiomics: dict[str, object],
    lat_radiomics: dict[str, object],
) -> MatchedCohort:
    ap_radiomics = align_within_view(ap_dino, ap_radiomics, "AP radiomics")
    lat_radiomics = align_within_view(lat_dino, lat_radiomics, "lateral radiomics")
    ap_map = unique_pair_map(ap_dino, "AP")
    lat_map = unique_pair_map(lat_dino, "lateral")
    common = sorted(set(ap_map).intersection(lat_map))
    retained = []
    discordant: list[dict[str, object]] = []
    for key in common:
        ap_index, lat_index = ap_map[key], lat_map[key]
        ap_label = strict_label(ap_dino["meta"][ap_index]["label_text"])
        lat_label = strict_label(lat_dino["meta"][lat_index]["label_text"])
        if ap_label is not None and lat_label is not None:
            if ap_label == lat_label:
                if str(ap_dino["groups"][ap_index]) != str(lat_dino["groups"][lat_index]):
                    raise ValueError(f"Matched pair {key} has inconsistent patient groups.")
                retained.append((key, ap_index, lat_index, ap_label))
            else:
                discordant.append(
                    {
                        "study_key": key[0],
                        "accession": key[1],
                        "run_index": key[2],
                        "ap_label": LABELS[ap_label],
                        "lateral_label": LABELS[lat_label],
                    }
                )
    if not retained:
        raise ValueError("No concordant strict AP/lateral pairs were found.")

    ap_indices = np.asarray([row[1] for row in retained], dtype=int)
    lat_indices = np.asarray([row[2] for row in retained], dtype=int)
    ap_temporal = np.asarray(ap_dino["mean"][ap_indices], dtype=np.float32)
    lat_temporal = np.asarray(lat_dino["mean"][lat_indices], dtype=np.float32)
    ap_spatial = np.asarray(ap_radiomics["mean"][ap_indices], dtype=np.float32)
    lat_spatial = np.asarray(lat_radiomics["mean"][lat_indices], dtype=np.float32)
    features = {
        "ap_temporal": ap_temporal,
        "lateral_temporal": lat_temporal,
        "biview_temporal": np.concatenate((ap_temporal, lat_temporal), axis=1),
        "biview_radiomics": np.concatenate((ap_spatial, lat_spatial), axis=1),
        "biview_all": np.concatenate((ap_temporal, lat_temporal, ap_spatial, lat_spatial), axis=1),
    }
    if any(not np.isfinite(matrix).all() for matrix in features.values()):
        raise ValueError("Matched feature matrices contain non-finite values.")
    labels = np.asarray([row[3] for row in retained], dtype=int)
    groups = np.asarray([ap_dino["groups"][row[1]] for row in retained], dtype=object)
    identities = tuple(f"{row[0][0]}:{row[0][1]}:{row[0][2]}:paired" for row in retained)
    class_counts = {LABELS[index]: int(np.sum(labels == index)) for index in range(3)}
    audit = {
        "matched_pairs": len(common),
        "concordant_strict_pairs": len(retained),
        "discordant_strict_pairs": len(discordant),
        "discordant": discordant,
        "n_groups": len(set(groups)),
        "class_counts": class_counts,
        "groups_per_class": {
            LABELS[index]: len(set(groups[labels == index])) for index in range(3)
        },
        "feature_dimensions": {name: int(matrix.shape[1]) for name, matrix in features.items()},
    }
    return MatchedCohort(features, labels, groups, identities, audit)


def new_estimator(kind: str):
    if kind == "svm":
        return LinearSVC(
            C=0.5,
            class_weight="balanced",
            dual=LINEAR_SVM_DUAL,
            max_iter=LINEAR_SVM_MAX_ITER,
            random_state=0,
            tol=LINEAR_SVM_TOL,
        )
    if kind == "logreg":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    raise ValueError(f"Unknown estimator kind: {kind}")


def fit_standardized(
    features: np.ndarray,
    labels: np.ndarray,
    kind: str,
    context: str,
) -> tuple[StandardScaler, object, int]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    estimator = new_estimator(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(transformed, labels)
        except ConvergenceWarning as error:
            raise RuntimeError(f"{context}: {kind} failed to converge.") from error
    iterations = np.asarray(getattr(estimator, "n_iter_", 0), dtype=int)
    return scaler, estimator, int(iterations.max()) if iterations.size else 0


def predict_svm(
    fitted: tuple[StandardScaler, object, int],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler, estimator, _ = fitted
    transformed = scaler.transform(features)
    prediction = estimator.predict(transformed)
    decision = np.asarray(estimator.decision_function(transformed), dtype=float)
    if decision.ndim == 1:
        decision = np.column_stack((-decision, decision))
    return np.asarray(prediction, dtype=int), decision


def predict_logreg(
    fitted: tuple[StandardScaler, object, int],
    features: np.ndarray,
) -> np.ndarray:
    scaler, estimator, _ = fitted
    probability = np.asarray(estimator.predict_proba(scaler.transform(features)), dtype=float)
    expected = np.arange(probability.shape[1])
    if not np.array_equal(estimator.classes_, expected):
        raise ValueError(f"Unexpected logistic classes: {estimator.classes_}")
    return probability


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(M.compute_metrics(labels, predictions, 3)["macro_f1"])


def selection_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float]:
    metric = M.compute_metrics(labels, predictions, 3)
    return float(metric["macro_f1"]), float(metric["per_class"]["f1"][1])


def select_hierarchy_parameters(
    dino_gate_probability: np.ndarray,
    radiomics_gate_probability: np.ndarray,
    segment_prediction: np.ndarray,
    labels: np.ndarray,
    radiomics_weights: np.ndarray = GATE_RADIOMICS_WEIGHT_GRID,
) -> dict[str, float]:
    candidates = []
    for radiomics_weight in radiomics_weights:
        gate = (
            (1.0 - float(radiomics_weight)) * dino_gate_probability
            + float(radiomics_weight) * radiomics_gate_probability
        )
        for threshold in THRESHOLD_GRID:
            prediction = hierarchy_prediction(gate, segment_prediction, float(threshold))
            score, m3_f1 = selection_metrics(labels, prediction)
            candidates.append(
                {
                    "radiomics_weight": float(radiomics_weight),
                    "threshold": float(threshold),
                    "inner_macro_f1": score,
                    "inner_m3_f1": m3_f1,
                }
            )
    return min(
        candidates,
        key=lambda row: (
            -row["inner_macro_f1"],
            -row["inner_m3_f1"],
            row["radiomics_weight"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )


def select_late_weight(
    dino_probability: np.ndarray,
    radiomics_probability: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    candidates = []
    for radiomics_weight in LATE_RADIOMICS_WEIGHT_GRID:
        probability = (
            (1.0 - float(radiomics_weight)) * dino_probability
            + float(radiomics_weight) * radiomics_probability
        )
        score, m3_f1 = selection_metrics(labels, np.argmax(probability, axis=1))
        candidates.append(
            {
                "radiomics_weight": float(radiomics_weight),
                "inner_macro_f1": score,
                "inner_m3_f1": m3_f1,
            }
        )
    return min(
        candidates,
        key=lambda row: (
            -row["inner_macro_f1"],
            -row["inner_m3_f1"],
            row["radiomics_weight"],
        ),
    )


def hierarchy_prediction(gate_probability: np.ndarray, segment_prediction: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(gate_probability >= threshold, 2, segment_prediction).astype(int)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def hierarchy_scores(gate_probability: np.ndarray, segment_decision: np.ndarray) -> np.ndarray:
    m3_conditional = sigmoid(segment_decision)
    mca_probability = 1.0 - gate_probability
    return np.column_stack(
        (
            mca_probability * (1.0 - m3_conditional),
            mca_probability * m3_conditional,
            gate_probability,
        )
    )


def split_count(labels: np.ndarray, groups: np.ndarray, requested: int) -> int:
    count = min(
        requested,
        *(len(set(groups[labels == class_index])) for class_index in range(3)),
    )
    if count < 2:
        raise ValueError("Every class needs at least two patient groups for grouped CV.")
    return count


def crossfit_primitives(
    cohort: MatchedCohort,
    outer_train: np.ndarray,
    inner_splits,
    context: str,
) -> tuple[dict[str, np.ndarray], int]:
    y = cohort.y[outer_train]
    length = len(outer_train)
    output = {
        "gate_dino": np.full(length, np.nan),
        "gate_radiomics": np.full(length, np.nan),
        "segment_ap_pred": np.full(length, -1, dtype=int),
        "segment_ap_decision": np.full(length, np.nan),
        "segment_biview_pred": np.full(length, -1, dtype=int),
        "segment_biview_decision": np.full(length, np.nan),
        "dino_probability": np.full((length, 3), np.nan),
        "radiomics_probability": np.full((length, 3), np.nan),
    }
    maximum_iterations = 0
    for inner_index, (inner_train, inner_valid) in enumerate(inner_splits):
        absolute_train = outer_train[inner_train]
        absolute_valid = outer_train[inner_valid]
        gate_labels = (cohort.y[absolute_train] == 2).astype(int)
        for output_name, feature_name in (
            ("gate_dino", "biview_temporal"),
            ("gate_radiomics", "biview_radiomics"),
        ):
            fitted = fit_standardized(
                cohort.features[feature_name][absolute_train],
                gate_labels,
                "logreg",
                f"{context}/inner={inner_index}/{output_name}",
            )
            maximum_iterations = max(maximum_iterations, fitted[2])
            output[output_name][inner_valid] = predict_logreg(
                fitted, cohort.features[feature_name][absolute_valid]
            )[:, 1]

        mca_train = absolute_train[cohort.y[absolute_train] != 2]
        for prefix, feature_name in (("segment_ap", "ap_temporal"), ("segment_biview", "biview_temporal")):
            fitted = fit_standardized(
                cohort.features[feature_name][mca_train],
                cohort.y[mca_train],
                "svm",
                f"{context}/inner={inner_index}/{prefix}",
            )
            maximum_iterations = max(maximum_iterations, fitted[2])
            prediction, decision = predict_svm(fitted, cohort.features[feature_name][absolute_valid])
            output[f"{prefix}_pred"][inner_valid] = prediction
            output[f"{prefix}_decision"][inner_valid] = decision[:, 1]

        for output_name, feature_name in (
            ("dino_probability", "biview_temporal"),
            ("radiomics_probability", "biview_radiomics"),
        ):
            fitted = fit_standardized(
                cohort.features[feature_name][absolute_train],
                cohort.y[absolute_train],
                "logreg",
                f"{context}/inner={inner_index}/{output_name}",
            )
            maximum_iterations = max(maximum_iterations, fitted[2])
            output[output_name][inner_valid] = predict_logreg(
                fitted, cohort.features[feature_name][absolute_valid]
            )
    for name, values in output.items():
        if (np.issubdtype(values.dtype, np.integer) and np.any(values < 0)) or (
            np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all()
        ):
            raise RuntimeError(f"Incomplete inner OOF primitive: {name}")
    return output, maximum_iterations


def fit_outer_primitives(
    cohort: MatchedCohort,
    train: np.ndarray,
    valid: np.ndarray,
    context: str,
) -> tuple[dict[str, np.ndarray], int]:
    output: dict[str, np.ndarray] = {}
    maximum_iterations = 0
    gate_labels = (cohort.y[train] == 2).astype(int)
    for output_name, feature_name in (
        ("gate_dino", "biview_temporal"),
        ("gate_radiomics", "biview_radiomics"),
    ):
        fitted = fit_standardized(
            cohort.features[feature_name][train], gate_labels, "logreg", f"{context}/{output_name}"
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        output[output_name] = predict_logreg(fitted, cohort.features[feature_name][valid])[:, 1]

    mca_train = train[cohort.y[train] != 2]
    for prefix, feature_name in (("segment_ap", "ap_temporal"), ("segment_biview", "biview_temporal")):
        fitted = fit_standardized(
            cohort.features[feature_name][mca_train],
            cohort.y[mca_train],
            "svm",
            f"{context}/{prefix}",
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        prediction, decision = predict_svm(fitted, cohort.features[feature_name][valid])
        output[f"{prefix}_pred"] = prediction
        output[f"{prefix}_decision"] = decision[:, 1]

    for output_name, feature_name in (
        ("dino_probability", "biview_temporal"),
        ("radiomics_probability", "biview_radiomics"),
    ):
        fitted = fit_standardized(
            cohort.features[feature_name][train],
            cohort.y[train],
            "logreg",
            f"{context}/{output_name}",
        )
        maximum_iterations = max(maximum_iterations, fitted[2])
        output[output_name] = predict_logreg(fitted, cohort.features[feature_name][valid])
    return output, maximum_iterations


def metrics_row(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    metric = M.compute_metrics(labels, predictions, 3)
    return {
        "macro_f1": metric["macro_f1"],
        "balanced_accuracy": metric["balanced_accuracy"],
        "accuracy": metric["accuracy"],
        "macro_auprc": float(
            np.mean(
                [average_precision_score((labels == index).astype(int), scores[:, index]) for index in range(3)]
            )
        ),
        "per_class_precision": metric["per_class"]["precision"],
        "per_class_recall": metric["per_class"]["recall"],
        "per_class_f1": metric["per_class"]["f1"],
        "confusion": metric["confusion"],
    }


def evaluate_seed(
    cohort: MatchedCohort,
    seed: int,
    requested_folds: int,
    requested_inner_folds: int,
) -> tuple[int, dict[str, dict[str, object]]]:
    fold_count = split_count(cohort.y, cohort.groups, requested_folds)
    outer_splits = make_folds(cohort.y, cohort.groups, fold_count, seed)
    predictions = {name: np.full(len(cohort.y), -1, dtype=int) for name in METHOD_DEFINITIONS}
    scores = {name: np.full((len(cohort.y), 3), np.nan) for name in METHOD_DEFINITIONS}
    selected_parameters = {name: [] for name in METHOD_DEFINITIONS}
    maximum_iterations = {name: 0 for name in METHOD_DEFINITIONS}

    for outer_index, (train, valid) in enumerate(outer_splits):
        inner_count = split_count(cohort.y[train], cohort.groups[train], requested_inner_folds)
        inner_splits = make_folds(
            cohort.y[train],
            cohort.groups[train],
            inner_count,
            seed + 1009 * (outer_index + 1),
        )
        context = f"seed={seed}/outer={outer_index}"
        inner, inner_max = crossfit_primitives(cohort, train, inner_splits, context)
        y_inner = cohort.y[train]
        dino_gate_ap_parameters = select_hierarchy_parameters(
            inner["gate_dino"],
            inner["gate_radiomics"],
            inner["segment_ap_pred"],
            y_inner,
            radiomics_weights=np.asarray([0.0]),
        )
        blended_gate_ap_parameters = select_hierarchy_parameters(
            inner["gate_dino"],
            inner["gate_radiomics"],
            inner["segment_ap_pred"],
            y_inner,
        )
        blended_gate_biview_parameters = select_hierarchy_parameters(
            inner["gate_dino"],
            inner["gate_radiomics"],
            inner["segment_biview_pred"],
            y_inner,
        )
        late_parameters = select_late_weight(
            inner["dino_probability"],
            inner["radiomics_probability"],
            y_inner,
        )

        outer, outer_max = fit_outer_primitives(cohort, train, valid, context)
        primitive_max = max(inner_max, outer_max)

        # Fixed outer-fold controls.
        for method_name, feature_name in (
            ("ap_temporal_svm", "ap_temporal"),
            ("lateral_temporal_svm", "lateral_temporal"),
            ("biview_temporal_svm", "biview_temporal"),
            ("biview_all_svm", "biview_all"),
        ):
            fitted = fit_standardized(
                cohort.features[feature_name][train],
                cohort.y[train],
                "svm",
                f"{context}/{method_name}",
            )
            prediction, decision = predict_svm(fitted, cohort.features[feature_name][valid])
            predictions[method_name][valid] = prediction
            scores[method_name][valid] = decision
            maximum_iterations[method_name] = max(maximum_iterations[method_name], fitted[2])
            selected_parameters[method_name].append({"outer_fold": outer_index, "fixed": True})

        predictions["biview_temporal_logreg"][valid] = np.argmax(
            outer["dino_probability"], axis=1
        )
        scores["biview_temporal_logreg"][valid] = outer["dino_probability"]
        selected_parameters["biview_temporal_logreg"].append({"outer_fold": outer_index, "fixed": True})
        maximum_iterations["biview_temporal_logreg"] = max(
            maximum_iterations["biview_temporal_logreg"], primitive_max
        )

        fitted_all_logreg = fit_standardized(
            cohort.features["biview_all"][train],
            cohort.y[train],
            "logreg",
            f"{context}/biview_all_logreg",
        )
        all_probability = predict_logreg(fitted_all_logreg, cohort.features["biview_all"][valid])
        predictions["biview_all_logreg"][valid] = np.argmax(all_probability, axis=1)
        scores["biview_all_logreg"][valid] = all_probability
        selected_parameters["biview_all_logreg"].append({"outer_fold": outer_index, "fixed": True})
        maximum_iterations["biview_all_logreg"] = max(
            maximum_iterations["biview_all_logreg"], fitted_all_logreg[2]
        )

        hierarchy_specs = (
            (
                "hierarchy_dino_gate_ap_segment",
                dino_gate_ap_parameters,
                "segment_ap",
            ),
            (
                "hierarchy_blended_gate_ap_segment",
                blended_gate_ap_parameters,
                "segment_ap",
            ),
            (
                "hierarchy_blended_gate_biview_segment",
                blended_gate_biview_parameters,
                "segment_biview",
            ),
        )
        for method_name, parameters, segment_prefix in hierarchy_specs:
            radiomics_weight = parameters["radiomics_weight"]
            threshold = parameters["threshold"]
            gate = (
                (1.0 - radiomics_weight) * outer["gate_dino"]
                + radiomics_weight * outer["gate_radiomics"]
            )
            segment_pred = outer[f"{segment_prefix}_pred"]
            segment_decision = outer[f"{segment_prefix}_decision"]
            predictions[method_name][valid] = hierarchy_prediction(
                gate, segment_pred, threshold
            )
            scores[method_name][valid] = hierarchy_scores(gate, segment_decision)
            selected_parameters[method_name].append(
                {"outer_fold": outer_index, **parameters}
            )
            maximum_iterations[method_name] = max(maximum_iterations[method_name], primitive_max)

        radiomics_weight = late_parameters["radiomics_weight"]
        late_probability = (
            (1.0 - radiomics_weight) * outer["dino_probability"]
            + radiomics_weight * outer["radiomics_probability"]
        )
        predictions["late_dino_radiomics_logreg"][valid] = np.argmax(
            late_probability, axis=1
        )
        scores["late_dino_radiomics_logreg"][valid] = late_probability
        selected_parameters["late_dino_radiomics_logreg"].append(
            {"outer_fold": outer_index, **late_parameters}
        )
        maximum_iterations["late_dino_radiomics_logreg"] = max(
            maximum_iterations["late_dino_radiomics_logreg"], primitive_max
        )

    result: dict[str, dict[str, object]] = {}
    for method_name in METHOD_DEFINITIONS:
        if np.any(predictions[method_name] < 0) or not np.isfinite(scores[method_name]).all():
            raise RuntimeError(f"Incomplete outer predictions for {method_name}, seed={seed}.")
        row = metrics_row(cohort.y, predictions[method_name], scores[method_name])
        row.update(
            {
                "seed": seed,
                "folds": fold_count,
                "inner_folds_requested": requested_inner_folds,
                "selected_parameters": selected_parameters[method_name],
                "max_fit_iterations": maximum_iterations[method_name],
                "convergence_verified": True,
                "oof_predictions": predictions[method_name].tolist(),
                "oof_scores": scores[method_name].tolist(),
            }
        )
        result[method_name] = row
    return seed, result


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in (
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "macro_auprc",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
    ):
        values = np.asarray([row[key] for row in rows], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        output[f"{key}_mean"] = mean.tolist() if mean.ndim else float(mean)
        output[f"{key}_std"] = std.tolist() if std.ndim else float(std)
    output["max_fit_iterations"] = max(int(row["max_fit_iterations"]) for row in rows)
    output["convergence_verified"] = all(bool(row["convergence_verified"]) for row in rows)
    return output


def paired_comparison(
    method_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> dict[str, object]:
    if [row["seed"] for row in method_rows] != [row["seed"] for row in reference_rows]:
        raise ValueError("Paired comparison seeds are not aligned.")
    deltas = np.asarray(
        [float(row["macro_f1"]) - float(ref["macro_f1"]) for row, ref in zip(method_rows, reference_rows)],
        dtype=float,
    )
    return {
        "macro_f1_delta_mean": float(deltas.mean()),
        "macro_f1_delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "wins": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "losses": int(np.sum(deltas < 0)),
        "per_seed_delta": deltas.tolist(),
    }


def source_record(path: Path, cache: dict[str, object]) -> dict[str, object]:
    return {"path": str(path), "sha256": cache["sha256"], "signature": cache["signature"]}


def main() -> int:
    arguments = parse_args()
    if arguments.folds < 2 or arguments.inner_folds < 2:
        raise ValueError("Outer and inner fold counts must be at least 2.")
    ap_dino = load_cache(arguments.ap_temporal)
    lat_dino = load_cache(arguments.lat_temporal)
    ap_radiomics = load_cache(arguments.ap_radiomics)
    lat_radiomics = load_cache(arguments.lat_radiomics)
    cohort = build_matched_cohort(ap_dino, lat_dino, ap_radiomics, lat_radiomics)
    print(
        f"Matched strict cohort: {len(cohort.y)} pairs / {len(set(cohort.groups))} groups / "
        + " ".join(f"{label}={cohort.audit['class_counts'][label]}" for label in LABELS),
        flush=True,
    )
    evaluated = Parallel(n_jobs=arguments.jobs, verbose=5)(
        delayed(evaluate_seed)(
            cohort,
            seed,
            arguments.folds,
            arguments.inner_folds,
        )
        for seed in arguments.seeds
    )
    evaluated.sort(key=lambda item: item[0])
    per_seed = {
        method_name: [seed_result[method_name] for _, seed_result in evaluated]
        for method_name in METHOD_DEFINITIONS
    }
    summaries = []
    for method_name, definition in METHOD_DEFINITIONS.items():
        summary = aggregate(per_seed[method_name])
        summary.update({"method": method_name, "definition": definition})
        summaries.append(summary)
    summaries.sort(key=lambda row: row["macro_f1_mean"], reverse=True)

    comparisons = {
        method_name: {
            "vs_ap_temporal_svm": paired_comparison(rows, per_seed["ap_temporal_svm"]),
            "vs_biview_temporal_logreg": paired_comparison(
                rows, per_seed["biview_temporal_logreg"]
            ),
        }
        for method_name, rows in per_seed.items()
    }
    output = {
        "protocol": {
            "endpoint": "strict concordant matched AP/lateral M2/M3/PCA",
            "labels": list(LABELS),
            "seeds": arguments.seeds,
            "requested_outer_folds": arguments.folds,
            "requested_inner_folds": arguments.inner_folds,
            "group": "Study_Key",
            "outer_validation": "patient-grouped; every run/view from a patient remains in one fold",
            "inner_tuning": "patient-grouped cross-fitted predictions within each outer training fold only",
            "threshold_grid": THRESHOLD_GRID.tolist(),
            "gate_radiomics_weight_grid": GATE_RADIOMICS_WEIGHT_GRID.tolist(),
            "late_radiomics_weight_grid": LATE_RADIOMICS_WEIGHT_GRID.tolist(),
            "selection_metric": "inner cross-fitted macro-F1; ties use M3 F1, then less radiomics",
            "linear_svm": {
                "C": 0.5,
                "class_weight": "balanced",
                "dual": LINEAR_SVM_DUAL,
                "max_iter": LINEAR_SVM_MAX_ITER,
                "tol": LINEAR_SVM_TOL,
            },
            "logistic": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
            "convergence_policy": "fail on any ConvergenceWarning",
            "selection_note": "Comparing methods on repeated outer CV is exploratory; final promotion still requires a locked patient-level test.",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
        },
        "cohort_audit": cohort.audit,
        "method_definitions": METHOD_DEFINITIONS,
        "feature_sources": {
            "ap_temporal": source_record(arguments.ap_temporal, ap_dino),
            "lateral_temporal": source_record(arguments.lat_temporal, lat_dino),
            "ap_radiomics": source_record(arguments.ap_radiomics, ap_radiomics),
            "lateral_radiomics": source_record(arguments.lat_radiomics, lat_radiomics),
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
        class_f1 = "/".join(f"{value:.3f}" for value in row["per_class_f1_mean"])
        print(
            f"{row['method']:42s} {row['macro_f1_mean']:.3f} +/- "
            f"{row['macro_f1_std']:.3f}  {class_f1}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
