from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import compare_missing_view_results as C


LABELS = ("m2", "m3", "pca")
SCOPES = {"union": 5, "paired": 3, "ap_only": 1, "lateral_only": 1}


def _metric(macro: float, n_cases: int) -> dict[str, object]:
    return {
        "macro_f1": macro,
        "balanced_accuracy": macro - 0.01,
        "accuracy": macro + 0.02,
        "per_class_precision": [macro, macro - 0.1, macro + 0.1],
        "per_class_recall": [macro + 0.01, macro - 0.09, macro + 0.08],
        "per_class_f1": [macro + 0.02, macro - 0.12, macro + 0.10],
        "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "n_cases": n_cases,
    }


def _audit() -> dict[str, object]:
    return {
        "n_cases": 5,
        "n_groups": 5,
        "class_counts": {"m2": 2, "m3": 2, "pca": 1},
        "groups_per_class": {"m2": 2, "m3": 2, "pca": 1},
        "by_availability": {
            "paired": {
                "n_cases": 3,
                "n_groups": 3,
                "class_counts": {"m2": 1, "m3": 1, "pca": 1},
            },
            "ap_only": {
                "n_cases": 1,
                "n_groups": 1,
                "class_counts": {"m2": 1, "m3": 0, "pca": 0},
            },
            "lateral_only": {
                "n_cases": 1,
                "n_groups": 1,
                "class_counts": {"m2": 0, "m3": 1, "pca": 0},
            },
        },
        "feature_dimension_per_view": 4,
        "exclusions": {
            "discordant_strict_pairs": [],
            "paired_nonstrict": 0,
            "ap_only_nonstrict": 0,
            "lateral_only_nonstrict": 0,
        },
    }


def _summary(method: str, scope: str, rows: list[dict[str, object]]) -> dict[str, object]:
    aggregate = C._aggregate_rows(rows, scope)
    return {"method": method, "scope": scope, "definition": method, **aggregate}


def _result(primary_values: list[float], direct_values: list[float]) -> dict[str, object]:
    methods: dict[str, list[dict[str, object]]] = {}
    for method, values in (
        (C.PRIMARY_METHOD, primary_values),
        (C.DIRECT_METHOD, direct_values),
    ):
        rows = []
        for seed, value in enumerate(values):
            rows.append(
                {
                    "seed": seed,
                    "folds": 2,
                    "inner_folds_requested": 2,
                    "scopes": {
                        scope: _metric(value - index * 0.01, count)
                        for index, (scope, count) in enumerate(SCOPES.items())
                    },
                    "selected_parameters": [
                        {"outer_fold": 0, "threshold": 0.5},
                        {"outer_fold": 1, "threshold": 0.5},
                    ],
                    "max_fit_iterations": 10,
                    "convergence_verified": True,
                    "oof_predictions": [0, 1, 2, 0, 1],
                    "oof_scores": [[0.8, 0.1, 0.1]] * 5,
                }
            )
        methods[method] = rows
    summaries = [
        _summary(method, scope, rows)
        for method, rows in methods.items()
        for scope in SCOPES
    ]
    protocol = {
        "endpoint": "strict test endpoint",
        "labels": list(LABELS),
        "seeds": [0, 1],
        "requested_outer_folds": 2,
        "requested_inner_folds": 2,
        "outer_group": "patient",
        "inner_tuning": "grouped",
        "paired_inference": "paired rule",
        "ap_only_inference": "ap rule",
        "lateral_only_inference": "lateral rule",
        "pair_weight_grid": [0.0, 0.5, 1.0],
        "threshold_grid": [0.4, 0.5, 0.6],
        "selection_metric": "macro-f1",
        "linear_svm": {"C": 0.5},
        "logistic": {"C": 1.0},
    }
    return {
        "protocol": protocol,
        "cohort_audit": _audit(),
        "method_definitions": {
            C.PRIMARY_METHOD: "primary",
            C.DIRECT_METHOD: "direct",
        },
        "summaries": summaries,
        "per_seed": methods,
    }


def _write_cache(path: Path, view: str, keys: list[tuple[str, str, int, str]]) -> None:
    metadata = []
    groups = []
    for study_key, accession, run_index, label in keys:
        metadata.append(
            {
                "study_key": study_key,
                "accession": accession,
                "run_column": f"{view}_{run_index}",
                "view": view,
                "label_text": label,
            }
        )
        groups.append(study_key)
    np.savez(
        path,
        mean=np.arange(len(keys) * 4, dtype=np.float32).reshape(len(keys), 4),
        groups=np.asarray(groups, dtype=object),
        meta=np.asarray(metadata, dtype=object),
    )


def _cache_set(tmp_path: Path, prefix: str) -> tuple[Path, Path]:
    paired = [
        ("patient-secret-a", "accession-secret-a", 1, "L M2"),
        ("patient-secret-b", "accession-secret-b", 1, "R M3"),
        ("patient-secret-c", "accession-secret-c", 1, "P2"),
    ]
    ap = tmp_path / f"{prefix}_ap.npz"
    lateral = tmp_path / f"{prefix}_lateral.npz"
    _write_cache(
        ap,
        "AP",
        paired + [("patient-secret-d", "accession-secret-d", 1, "L M2")],
    )
    _write_cache(
        lateral,
        "Lateral",
        paired + [("patient-secret-e", "accession-secret-e", 1, "R M3")],
    )
    return ap, lateral


class ComparisonTests(unittest.TestCase):
    def test_comparison_reports_paired_seed_deltas_and_no_identifiers(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            reference = _result([0.50, 0.65], [0.45, 0.55])
            candidate = _result([0.60, 0.70], [0.50, 0.58])
            reference_ap, reference_lateral = _cache_set(tmp_path, "reference")
            candidate_ap, candidate_lateral = _cache_set(tmp_path, "candidate")

            report = C.build_comparison(
                "dino",
                reference,
                "vjepa",
                candidate,
                expected_seed_count=2,
                cache_paths=(
                    reference_ap,
                    reference_lateral,
                    candidate_ap,
                    candidate_lateral,
                ),
            )

            delta = report["paired_seed_comparisons"][
                "candidate_primary_vs_reference_primary"
            ]
            np.testing.assert_allclose(delta["per_seed_delta"], [0.10, 0.05])
            self.assertAlmostEqual(delta["macro_f1_delta_mean"], 0.075)
            self.assertEqual((delta["wins"], delta["ties"], delta["losses"]), (2, 0, 0))
            self.assertTrue(
                report["validation"]["cohort"]["exact_identity_order_match"]
            )
            self.assertEqual(
                report["models"]["dino"]["primary"]["union"]["n_cases"], 5
            )

            serialized = json.dumps(report) + C.render_markdown(report)
            self.assertNotIn("patient-secret", serialized)
            self.assertNotIn("accession-secret", serialized)

    def test_cache_identity_mismatch_error_does_not_emit_identifier(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = _result([0.50, 0.60], [0.45, 0.55])
            reference_ap, reference_lateral = _cache_set(tmp_path, "reference")
            candidate_ap, candidate_lateral = _cache_set(tmp_path, "candidate")
            with np.load(candidate_ap, allow_pickle=True) as cache:
                features = cache["mean"].copy()
                groups = cache["groups"].copy()
                metadata = list(cache["meta"].copy())
            metadata[0] = {
                **dict(metadata[0]),
                "accession": "highly-secret-new-accession",
            }
            np.savez(
                candidate_ap,
                mean=features,
                groups=groups,
                meta=np.asarray(metadata, dtype=object),
            )

            with self.assertRaises(C.ComparisonError) as raised:
                C.build_comparison(
                    "dino",
                    result,
                    "vjepa",
                    copy.deepcopy(result),
                    expected_seed_count=2,
                    cache_paths=(
                        reference_ap,
                        reference_lateral,
                        candidate_ap,
                        candidate_lateral,
                    ),
                )
            self.assertNotIn("highly-secret", str(raised.exception))
            self.assertNotIn("patient-secret", str(raised.exception))

    def test_seed_structure_mismatch_is_rejected(self) -> None:
        reference = _result([0.50, 0.60], [0.45, 0.55])
        candidate = copy.deepcopy(reference)
        candidate["protocol"]["seeds"] = [0, 2]

        with self.assertRaisesRegex(C.ComparisonError, "per-seed order"):
            C.build_comparison(
                "dino", reference, "vjepa", candidate, expected_seed_count=2
            )

    def test_saved_summary_disagreement_is_rejected(self) -> None:
        reference = _result([0.50, 0.60], [0.45, 0.55])
        candidate = copy.deepcopy(reference)
        candidate["summaries"][0]["macro_f1_mean"] += 0.01

        with self.assertRaisesRegex(C.ComparisonError, "disagrees"):
            C.build_comparison(
                "dino", reference, "vjepa", candidate, expected_seed_count=2
            )


if __name__ == "__main__":
    unittest.main()
