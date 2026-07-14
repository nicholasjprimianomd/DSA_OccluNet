"""Compare two saved missing-view hierarchy experiments without exposing identifiers.

The comparator is intentionally stricter than a table-building script.  It verifies that
both result files used the same repeated grouped-CV protocol, seed sequence, and fold
counts. If the four source caches are supplied, it also reconstructs the strict union
internally and requires identical ordered identities, labels, patient groups, and view
availability.
Only aggregate cohort facts are returned or printed; cache identifiers never leave the
validation functions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from multiview_anatomy_experiments import LABELS, strict_label


PRIMARY_METHOD = "augmented_hierarchy_union_tuned"
DIRECT_METHOD = "missing_view_logreg_union_tuned"
SCOPE_ORDER = ("union", "paired", "ap_only", "lateral_only")
AVAILABILITY_ORDER = ("paired", "ap_only", "lateral_only")
PROTOCOL_KEYS = (
    "endpoint",
    "labels",
    "seeds",
    "requested_outer_folds",
    "requested_inner_folds",
    "outer_group",
    "inner_tuning",
    "paired_inference",
    "ap_only_inference",
    "lateral_only_inference",
    "pair_weight_grid",
    "threshold_grid",
    "selection_metric",
    "linear_svm",
    "logistic",
)


class ComparisonError(ValueError):
    """Raised when two experiments are not safe to compare."""


@dataclass(frozen=True)
class CohortOrder:
    """Internal-only ordered cohort representation.

    ``identities`` and ``groups`` may contain sensitive source values.  This dataclass must
    never be serialized, formatted, or returned from the public comparison report.
    """

    identities: tuple[tuple[str, str, int], ...]
    labels: np.ndarray
    groups: tuple[str, ...]
    availability: tuple[str, ...]
    safe_summary: dict[str, Any]


@dataclass(frozen=True)
class ValidatedResult:
    name: str
    data: dict[str, Any]
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    aggregates: dict[tuple[str, str], dict[str, Any]]


def _sample_std(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(values.shape[1:] if values.ndim > 1 else (), dtype=float)
    return np.std(values, axis=0, ddof=1)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError("A result file could not be read as JSON.") from error
    if not isinstance(value, dict):
        raise ComparisonError("Each result file must contain a JSON object.")
    return value


def _aggregate_rows(rows: Sequence[dict[str, Any]], scope: str) -> dict[str, Any]:
    try:
        metrics = [row["scopes"][scope] for row in rows]
        macro = np.asarray([metric["macro_f1"] for metric in metrics], dtype=float)
        balanced = np.asarray([metric["balanced_accuracy"] for metric in metrics], dtype=float)
        accuracy = np.asarray([metric["accuracy"] for metric in metrics], dtype=float)
        class_f1 = np.asarray([metric["per_class_f1"] for metric in metrics], dtype=float)
        case_counts = {int(metric["n_cases"]) for metric in metrics}
    except (KeyError, TypeError, ValueError) as error:
        raise ComparisonError("A per-seed metric row is incomplete or malformed.") from error
    if len(case_counts) != 1:
        raise ComparisonError("A scope changes case count across split seeds.")
    if class_f1.shape != (len(rows), len(LABELS)):
        raise ComparisonError("Per-class F1 does not match the declared label count.")
    for values in (macro, balanced, accuracy, class_f1):
        if not np.isfinite(values).all():
            raise ComparisonError("A saved metric contains a non-finite value.")
    return {
        "n_cases": case_counts.pop(),
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_std": float(_sample_std(macro)),
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_std": float(_sample_std(balanced)),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(_sample_std(accuracy)),
        "per_class_f1_mean": class_f1.mean(axis=0).tolist(),
        "per_class_f1_std": _sample_std(class_f1).tolist(),
    }


def _assert_summary_matches(
    result: dict[str, Any], method: str, scope: str, aggregate: dict[str, Any]
) -> None:
    summaries = result.get("summaries")
    if not isinstance(summaries, list):
        raise ComparisonError("The result is missing its aggregate summary table.")
    matches = [
        row
        for row in summaries
        if isinstance(row, dict) and row.get("method") == method and row.get("scope") == scope
    ]
    if len(matches) != 1:
        raise ComparisonError("The result does not have exactly one summary for a method/scope.")
    saved = matches[0]
    for key in (
        "n_cases",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "accuracy_mean",
        "accuracy_std",
        "per_class_f1_mean",
        "per_class_f1_std",
    ):
        if key not in saved:
            raise ComparisonError("A saved aggregate summary is incomplete.")
        if key == "n_cases":
            equal = int(saved[key]) == int(aggregate[key])
        else:
            equal = np.allclose(saved[key], aggregate[key], rtol=1e-10, atol=1e-12)
        if not equal:
            raise ComparisonError("A saved summary disagrees with its per-seed metrics.")


def _safe_result_audit(result: dict[str, Any]) -> dict[str, Any]:
    audit = result.get("cohort_audit")
    if not isinstance(audit, dict):
        raise ComparisonError("The result is missing its cohort audit.")
    try:
        by_availability = {
            kind: {
                "n_cases": int(audit["by_availability"][kind]["n_cases"]),
                "n_groups": int(audit["by_availability"][kind]["n_groups"]),
                "class_counts": {
                    label: int(audit["by_availability"][kind]["class_counts"][label])
                    for label in LABELS
                },
            }
            for kind in AVAILABILITY_ORDER
        }
        exclusions = audit["exclusions"]
        discordant = exclusions["discordant_strict_pairs"]
        safe_exclusions = {
            "discordant_strict_pairs": (
                len(discordant) if isinstance(discordant, list) else int(discordant)
            ),
            "paired_nonstrict": int(exclusions["paired_nonstrict"]),
            "ap_only_nonstrict": int(exclusions["ap_only_nonstrict"]),
            "lateral_only_nonstrict": int(exclusions["lateral_only_nonstrict"]),
        }
        return {
            "n_cases": int(audit["n_cases"]),
            "n_groups": int(audit["n_groups"]),
            "class_counts": {label: int(audit["class_counts"][label]) for label in LABELS},
            "groups_per_class": {
                label: int(audit["groups_per_class"][label]) for label in LABELS
            },
            "by_availability": by_availability,
            "exclusions": safe_exclusions,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ComparisonError("The cohort audit is incomplete or malformed.") from error


def validate_result(
    name: str,
    result: dict[str, Any],
    *,
    expected_seed_count: int = 20,
    primary_method: str = PRIMARY_METHOD,
    direct_method: str = DIRECT_METHOD,
) -> ValidatedResult:
    protocol = result.get("protocol")
    per_seed = result.get("per_seed")
    if not isinstance(protocol, dict) or not isinstance(per_seed, dict):
        raise ComparisonError("The result is missing protocol or per-seed data.")
    try:
        seeds = tuple(int(seed) for seed in protocol["seeds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ComparisonError("The protocol seed list is invalid.") from error
    if len(seeds) != expected_seed_count or len(set(seeds)) != len(seeds):
        raise ComparisonError(
            f"Expected {expected_seed_count} distinct split seeds in each result."
        )
    methods = tuple(sorted(per_seed))
    if primary_method not in per_seed or direct_method not in per_seed:
        raise ComparisonError("The primary hierarchy or direct control is missing.")
    union_count = int(_safe_result_audit(result)["n_cases"])
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for method in methods:
        rows = per_seed[method]
        if not isinstance(rows, list) or len(rows) != len(seeds):
            raise ComparisonError("A method does not contain one row per split seed.")
        row_seeds = tuple(int(row.get("seed", -1)) for row in rows if isinstance(row, dict))
        if row_seeds != seeds:
            raise ComparisonError("A method's per-seed order differs from the protocol.")
        scope_set: set[str] | None = None
        for row in rows:
            try:
                folds = int(row["folds"])
                selected = row["selected_parameters"]
                predictions = row["oof_predictions"]
                scores = row["oof_scores"]
                scopes = set(row["scopes"])
            except (KeyError, TypeError, ValueError) as error:
                raise ComparisonError("A per-seed model row is incomplete.") from error
            if folds < 2:
                raise ComparisonError("A saved outer-fold count is invalid.")
            if not isinstance(selected, list) or len(selected) != folds:
                raise ComparisonError("Selected parameters do not cover every outer fold.")
            try:
                selected_folds = {int(item["outer_fold"]) for item in selected}
            except (KeyError, TypeError, ValueError) as error:
                raise ComparisonError("Selected-parameter fold metadata are invalid.") from error
            if selected_folds != set(range(folds)):
                raise ComparisonError("Selected parameters have incomplete outer-fold indices.")
            if len(predictions) != union_count or len(scores) != union_count:
                raise ComparisonError("OOF output length differs from the audited union size.")
            if scope_set is None:
                scope_set = scopes
            elif scope_set != scopes:
                raise ComparisonError("A method changes evaluation scopes across seeds.")
        assert scope_set is not None
        for scope in scope_set:
            aggregate = _aggregate_rows(rows, scope)
            _assert_summary_matches(result, method, scope, aggregate)
            aggregates[(method, scope)] = aggregate
    for required_method in (primary_method, direct_method):
        missing_scopes = [
            scope for scope in SCOPE_ORDER if (required_method, scope) not in aggregates
        ]
        if missing_scopes:
            raise ComparisonError(
                "The primary hierarchy or direct control is missing a required evaluation scope."
            )
    return ValidatedResult(name, result, seeds, methods, aggregates)


def _protocol_projection(result: dict[str, Any]) -> dict[str, Any]:
    protocol = result["protocol"]
    missing = [key for key in PROTOCOL_KEYS if key not in protocol]
    if missing:
        raise ComparisonError("A protocol is missing required comparison fields.")
    return {key: protocol[key] for key in PROTOCOL_KEYS}


def validate_matched_structure(reference: ValidatedResult, candidate: ValidatedResult) -> None:
    if reference.seeds != candidate.seeds:
        raise ComparisonError("The two results use different split seed sequences.")
    if reference.methods != candidate.methods:
        raise ComparisonError("The two results do not contain the same method set.")
    if _protocol_projection(reference.data) != _protocol_projection(candidate.data):
        raise ComparisonError("The two results do not use the same evaluation protocol.")
    if reference.data.get("method_definitions") != candidate.data.get("method_definitions"):
        raise ComparisonError("The two results do not use the same method definitions.")
    if _safe_result_audit(reference.data) != _safe_result_audit(candidate.data):
        raise ComparisonError("The two result files describe different aggregate cohorts.")
    for method in reference.methods:
        reference_rows = reference.data["per_seed"][method]
        candidate_rows = candidate.data["per_seed"][method]
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if int(reference_row["folds"]) != int(candidate_row["folds"]):
                raise ComparisonError("The two results have different outer-fold structures.")
            if int(reference_row["inner_folds_requested"]) != int(
                candidate_row["inner_folds_requested"]
            ):
                raise ComparisonError("The two results have different inner-fold structures.")


def _cache_arrays(path: Path) -> dict[str, Any]:
    try:
        cache = np.load(path, allow_pickle=True)
        required = ("mean", "groups", "meta")
        if any(key not in cache for key in required):
            raise ComparisonError("A feature cache is missing required arrays.")
        return {key: cache[key] for key in required}
    except (OSError, ValueError) as error:
        raise ComparisonError("A feature cache could not be read.") from error


def _run_index(run_column: Any) -> int:
    text = str(run_column)
    try:
        prefix, suffix = text.rsplit("_", 1)
        value = int(suffix)
    except (ValueError, TypeError) as error:
        raise ComparisonError("A cache contains an invalid run-column value.") from error
    if not prefix or value < 1:
        raise ComparisonError("A cache contains an invalid run-column value.")
    return value


def _cache_map(cache: dict[str, Any], expected_view: str) -> dict[tuple[str, str, int], int]:
    features = np.asarray(cache["mean"])
    groups = np.asarray(cache["groups"], dtype=object)
    metadata = np.asarray(cache["meta"], dtype=object)
    if features.ndim != 2 or len(features) != len(groups) or len(features) != len(metadata):
        raise ComparisonError("A cache has inconsistent row counts or feature rank.")
    if not np.isfinite(features).all():
        raise ComparisonError("A cache contains non-finite features.")
    mapping: dict[tuple[str, str, int], int] = {}
    for index, raw in enumerate(metadata):
        try:
            item = dict(raw)
            study_key = str(item["study_key"])
            accession = str(item["accession"])
            view = str(item["view"])
            run_index = _run_index(item["run_column"])
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError("A cache metadata row is incomplete.") from error
        if view.casefold() != expected_view.casefold():
            raise ComparisonError("A cache metadata row has the wrong view.")
        if str(groups[index]) != study_key:
            raise ComparisonError("A cache group does not match its metadata group.")
        key = (study_key, accession, run_index)
        if key in mapping:
            raise ComparisonError("A cache contains duplicate case identities.")
        mapping[key] = index
    return mapping


def reconstruct_cohort(ap_path: Path, lateral_path: Path) -> CohortOrder:
    ap_cache = _cache_arrays(ap_path)
    lateral_cache = _cache_arrays(lateral_path)
    if np.asarray(ap_cache["mean"]).shape[1] != np.asarray(lateral_cache["mean"]).shape[1]:
        raise ComparisonError("AP and lateral feature dimensions differ within a backbone.")
    ap_map = _cache_map(ap_cache, "AP")
    lateral_map = _cache_map(lateral_cache, "Lateral")
    rows: list[tuple[tuple[str, str, int], int, str]] = []
    exclusion_counts = {
        "discordant_strict_pairs": 0,
        "paired_nonstrict": 0,
        "ap_only_nonstrict": 0,
        "lateral_only_nonstrict": 0,
    }
    for key in sorted(set(ap_map).union(lateral_map)):
        ap_index = ap_map.get(key)
        lateral_index = lateral_map.get(key)
        ap_label = (
            None
            if ap_index is None
            else strict_label(dict(ap_cache["meta"][ap_index]).get("label_text", ""))
        )
        lateral_label = (
            None
            if lateral_index is None
            else strict_label(
                dict(lateral_cache["meta"][lateral_index]).get("label_text", "")
            )
        )
        if ap_index is not None and lateral_index is not None:
            if ap_label is not None and ap_label == lateral_label:
                rows.append((key, ap_label, "paired"))
            elif ap_label is not None and lateral_label is not None:
                exclusion_counts["discordant_strict_pairs"] += 1
            else:
                exclusion_counts["paired_nonstrict"] += 1
        elif ap_index is not None:
            if ap_label is None:
                exclusion_counts["ap_only_nonstrict"] += 1
            else:
                rows.append((key, ap_label, "ap_only"))
        elif lateral_label is None:
            exclusion_counts["lateral_only_nonstrict"] += 1
        else:
            rows.append((key, lateral_label, "lateral_only"))
    if not rows:
        raise ComparisonError("The supplied caches produce an empty strict cohort.")
    identities = tuple(row[0] for row in rows)
    labels = np.asarray([row[1] for row in rows], dtype=int)
    groups = tuple(row[0][0] for row in rows)
    availability = tuple(row[2] for row in rows)
    by_availability: dict[str, Any] = {}
    availability_array = np.asarray(availability, dtype=object)
    groups_array = np.asarray(groups, dtype=object)
    for kind in AVAILABILITY_ORDER:
        keep = availability_array == kind
        by_availability[kind] = {
            "n_cases": int(keep.sum()),
            "n_groups": len(set(groups_array[keep])),
            "class_counts": {
                label: int(np.sum(labels[keep] == index)) for index, label in enumerate(LABELS)
            },
        }
    safe_summary = {
        "n_cases": len(rows),
        "n_groups": len(set(groups)),
        "class_counts": {
            label: int(np.sum(labels == index)) for index, label in enumerate(LABELS)
        },
        "groups_per_class": {
            label: len(set(groups_array[labels == index])) for index, label in enumerate(LABELS)
        },
        "by_availability": by_availability,
        "exclusions": exclusion_counts,
    }
    return CohortOrder(identities, labels, groups, availability, safe_summary)


def validate_cache_cohorts(
    reference_ap: Path,
    reference_lateral: Path,
    candidate_ap: Path,
    candidate_lateral: Path,
    reference_result: ValidatedResult,
    candidate_result: ValidatedResult,
) -> dict[str, Any]:
    reference = reconstruct_cohort(reference_ap, reference_lateral)
    candidate = reconstruct_cohort(candidate_ap, candidate_lateral)
    if reference.identities != candidate.identities:
        raise ComparisonError("Cache cohorts have different ordered case identities.")
    if not np.array_equal(reference.labels, candidate.labels):
        raise ComparisonError("Cache cohorts have different ordered labels.")
    if reference.groups != candidate.groups:
        raise ComparisonError("Cache cohorts have different ordered patient groups.")
    if reference.availability != candidate.availability:
        raise ComparisonError("Cache cohorts have different ordered view availability.")
    if reference.safe_summary != _safe_result_audit(reference_result.data):
        raise ComparisonError("Reference caches do not reproduce the reference result cohort.")
    if candidate.safe_summary != _safe_result_audit(candidate_result.data):
        raise ComparisonError("Candidate caches do not reproduce the candidate result cohort.")
    return {
        "performed": True,
        "exact_identity_order_match": True,
        "exact_label_order_match": True,
        "exact_group_order_match": True,
        "exact_availability_order_match": True,
        **reference.safe_summary,
    }


def paired_seed_delta(
    candidate: ValidatedResult,
    candidate_method: str,
    reference: ValidatedResult,
    reference_method: str,
    scope: str,
) -> dict[str, Any]:
    candidate_rows = candidate.data["per_seed"][candidate_method]
    reference_rows = reference.data["per_seed"][reference_method]
    candidate_macro = np.asarray(
        [row["scopes"][scope]["macro_f1"] for row in candidate_rows], dtype=float
    )
    reference_macro = np.asarray(
        [row["scopes"][scope]["macro_f1"] for row in reference_rows], dtype=float
    )
    deltas = candidate_macro - reference_macro
    candidate_f1 = np.asarray(
        [row["scopes"][scope]["per_class_f1"] for row in candidate_rows], dtype=float
    )
    reference_f1 = np.asarray(
        [row["scopes"][scope]["per_class_f1"] for row in reference_rows], dtype=float
    )
    class_deltas = candidate_f1 - reference_f1
    tolerance = 1e-12
    return {
        "candidate": candidate.name,
        "candidate_method": candidate_method,
        "reference": reference.name,
        "reference_method": reference_method,
        "scope": scope,
        "macro_f1_delta_mean": float(deltas.mean()),
        "macro_f1_delta_std": float(_sample_std(deltas)),
        "macro_f1_delta_median": float(np.median(deltas)),
        "macro_f1_delta_min": float(deltas.min()),
        "macro_f1_delta_max": float(deltas.max()),
        "wins": int(np.sum(deltas > tolerance)),
        "ties": int(np.sum(np.abs(deltas) <= tolerance)),
        "losses": int(np.sum(deltas < -tolerance)),
        "per_seed_delta": deltas.tolist(),
        "per_class_f1_delta_mean": class_deltas.mean(axis=0).tolist(),
        "per_class_f1_delta_std": _sample_std(class_deltas).tolist(),
    }


def _method_scopes(result: ValidatedResult, method: str) -> dict[str, Any]:
    return {
        scope: result.aggregates[(method, scope)]
        for scope in SCOPE_ORDER
        if (method, scope) in result.aggregates
    }


def build_comparison(
    reference_name: str,
    reference_data: dict[str, Any],
    candidate_name: str,
    candidate_data: dict[str, Any],
    *,
    expected_seed_count: int = 20,
    primary_method: str = PRIMARY_METHOD,
    direct_method: str = DIRECT_METHOD,
    cache_paths: tuple[Path, Path, Path, Path] | None = None,
) -> dict[str, Any]:
    reference = validate_result(
        reference_name,
        reference_data,
        expected_seed_count=expected_seed_count,
        primary_method=primary_method,
        direct_method=direct_method,
    )
    candidate = validate_result(
        candidate_name,
        candidate_data,
        expected_seed_count=expected_seed_count,
        primary_method=primary_method,
        direct_method=direct_method,
    )
    validate_matched_structure(reference, candidate)
    cohort_validation: dict[str, Any] = {
        "performed": False,
        **_safe_result_audit(reference.data),
    }
    if cache_paths is not None:
        cohort_validation = validate_cache_cohorts(
            *cache_paths, reference_result=reference, candidate_result=candidate
        )
    diagnostic_methods = (
        "paired_concat_hierarchy",
        "paired_viewwise_hierarchy",
        "paired_concat_logreg",
    )
    report = {
        "comparison_direction": "candidate_minus_reference",
        "validation": {
            "matched_protocol": True,
            "matched_seed_sequence": True,
            "matched_method_set": True,
            "matched_outer_and_inner_fold_counts": True,
            "seed_count": len(reference.seeds),
            "cohort": cohort_validation,
        },
        "primary_method": primary_method,
        "direct_control_method": direct_method,
        "models": {
            reference.name: {
                "primary": _method_scopes(reference, primary_method),
                "direct_control": _method_scopes(reference, direct_method),
                "diagnostic_controls": {
                    method: _method_scopes(reference, method)
                    for method in diagnostic_methods
                    if method in reference.methods
                },
            },
            candidate.name: {
                "primary": _method_scopes(candidate, primary_method),
                "direct_control": _method_scopes(candidate, direct_method),
                "diagnostic_controls": {
                    method: _method_scopes(candidate, method)
                    for method in diagnostic_methods
                    if method in candidate.methods
                },
            },
        },
        "paired_seed_comparisons": {
            "candidate_primary_vs_reference_primary": paired_seed_delta(
                candidate, primary_method, reference, primary_method, "union"
            ),
            "candidate_direct_vs_reference_direct": paired_seed_delta(
                candidate, direct_method, reference, direct_method, "union"
            ),
            "reference_primary_vs_its_direct_control": paired_seed_delta(
                reference, primary_method, reference, direct_method, "union"
            ),
            "candidate_primary_vs_its_direct_control": paired_seed_delta(
                candidate, primary_method, candidate, direct_method, "union"
            ),
        },
        "interpretation_note": (
            "Split seeds reuse the same patients and are correlated. The paired-seed deltas "
            "measure split sensitivity, not independent-sample uncertainty or external validation."
        ),
    }
    return report


def _metric_text(row: dict[str, Any]) -> str:
    return f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}"


def _class_text(row: dict[str, Any]) -> str:
    return " / ".join(f"{value:.3f}" for value in row["per_class_f1_mean"])


def render_markdown(report: dict[str, Any]) -> str:
    names = list(report["models"])
    reference_name, candidate_name = names[0], names[1]
    lines = [
        "# Missing-view backbone comparison",
        "",
        (
            f"Validated {report['validation']['seed_count']} matched split seeds with the same "
            "deterministic split protocol, outer/inner fold counts, and aggregate cohort composition."
        ),
        "",
        "## Primary union result",
        "",
        "| Model | Macro-F1 | M2 / M3 / PCA F1 |",
        "|---|---:|---:|",
    ]
    for name in (reference_name, candidate_name):
        row = report["models"][name]["primary"]["union"]
        lines.append(f"| {name} | {_metric_text(row)} | {_class_text(row)} |")
    delta = report["paired_seed_comparisons"]["candidate_primary_vs_reference_primary"]
    lines.extend(
        [
            "",
            (
                f"Candidate-reference paired-seed delta: {delta['macro_f1_delta_mean']:+.4f} "
                f"± {delta['macro_f1_delta_std']:.4f}; "
                f"wins/ties/losses {delta['wins']}/{delta['ties']}/{delta['losses']}."
            ),
            "",
            "Per-seed macro-F1 deltas: "
            + ", ".join(f"{value:+.4f}" for value in delta["per_seed_delta"]),
            "",
            "## Primary result by availability",
            "",
            "| Model | Scope | Cases | Macro-F1 | M2 / M3 / PCA F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name in (reference_name, candidate_name):
        for scope in SCOPE_ORDER:
            row = report["models"][name]["primary"].get(scope)
            if row is not None:
                lines.append(
                    f"| {name} | {scope} | {row['n_cases']} | "
                    f"{_metric_text(row)} | {_class_text(row)} |"
                )
    lines.extend(
        [
            "",
            "## Direct missing-view controls",
            "",
            "| Model | Union macro-F1 | M2 / M3 / PCA F1 | Primary-control delta |",
            "|---|---:|---:|---:|",
        ]
    )
    own_keys = {
        reference_name: "reference_primary_vs_its_direct_control",
        candidate_name: "candidate_primary_vs_its_direct_control",
    }
    for name in (reference_name, candidate_name):
        row = report["models"][name]["direct_control"]["union"]
        own = report["paired_seed_comparisons"][own_keys[name]]
        lines.append(
            f"| {name} | {_metric_text(row)} | {_class_text(row)} | "
            f"{own['macro_f1_delta_mean']:+.4f} ± {own['macro_f1_delta_std']:.4f} |"
        )
    cache_validation = report["validation"]["cohort"]
    if cache_validation["performed"]:
        lines.extend(
            [
                "",
                (
                    "Cache validation passed: ordered case identity, label, patient-group, and "
                    "view-availability arrays match exactly. No identifiers are included in this report."
                ),
            ]
        )
    lines.extend(["", report["interpretation_note"]])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--primary-method", default=PRIMARY_METHOD)
    parser.add_argument("--direct-method", default=DIRECT_METHOD)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--reference-ap-cache", type=Path)
    parser.add_argument("--reference-lateral-cache", type=Path)
    parser.add_argument("--candidate-ap-cache", type=Path)
    parser.add_argument("--candidate-lateral-cache", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    cache_values = (
        arguments.reference_ap_cache,
        arguments.reference_lateral_cache,
        arguments.candidate_ap_cache,
        arguments.candidate_lateral_cache,
    )
    supplied = [value is not None for value in cache_values]
    if any(supplied) and not all(supplied):
        raise ComparisonError("Supply all four feature caches or none of them.")
    report = build_comparison(
        arguments.reference_name,
        _load_json(arguments.reference_results),
        arguments.candidate_name,
        _load_json(arguments.candidate_results),
        expected_seed_count=arguments.expected_seeds,
        primary_method=arguments.primary_method,
        direct_method=arguments.direct_method,
        cache_paths=cache_values if all(supplied) else None,
    )
    markdown = render_markdown(report)
    print(markdown, end="")
    if arguments.json_out is not None:
        arguments.json_out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.json_out.open("w") as handle:
            json.dump(report, handle, indent=2)
    if arguments.markdown_out is not None:
        arguments.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.markdown_out.write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
