from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

import numpy as np

from compare_feature_caches import canonical_metadata
import multiview_ensemble_experiments as E


def _meta(study: str, accession: str, view: str, label: str, run: int = 1) -> dict[str, str]:
    return {
        "study_key": study,
        "accession": accession,
        "view": view,
        "run_column": f"{view}_{run}",
        "label_text": label,
    }


def _cache(meta: list[dict[str, str]], features: list[list[float]], path: str) -> dict[str, object]:
    metadata = np.asarray(meta, dtype=object)
    return {
        "mean": np.asarray(features, dtype=np.float32),
        "labels": np.zeros(len(meta), dtype=int),
        "groups": np.asarray([row["study_key"] for row in meta], dtype=object),
        "meta": metadata,
        "metadata_identity": canonical_metadata(metadata),
        "path": path,
    }


class MultiviewEnsembleExperimentTests(unittest.TestCase):
    def test_seed_parser_sorts_and_rejects_duplicates(self) -> None:
        self.assertEqual(E.parse_seeds("5,0,2"), [0, 2, 5])
        self.assertEqual(E.parse_seeds("2:5"), [2, 3, 4])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unique"):
            E.parse_seeds("0,0")

    def test_pair_key_includes_accession_and_rejects_true_duplicates(self) -> None:
        distinct_accessions = _cache(
            [
                _meta("study", "accession-a", "AP", "M2"),
                _meta("study", "accession-b", "AP", "M3"),
            ],
            [[1.0], [2.0]],
            "distinct.npz",
        )
        self.assertEqual(len(E.unique_pair_map(distinct_accessions, "AP")), 2)

        duplicate = _cache(
            [
                _meta("study", "accession-a", "AP", "M2"),
                _meta("study", "accession-a", "AP", "M2"),
            ],
            [[1.0], [2.0]],
            "duplicate.npz",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate pair key"):
            E.unique_pair_map(duplicate, "AP")

    def test_matched_cohort_aligns_each_view_and_excludes_discordant_labels(self) -> None:
        # The two concordant samples deliberately share study/run but have distinct
        # accessions. Every cache also uses a different row order.
        ap_dino = _cache(
            [
                _meta("study", "accession-b", "AP", "M3"),
                _meta("study", "accession-c", "AP", "M2"),
                _meta("study", "accession-a", "AP", "M2"),
            ],
            [[20.0, 21.0], [30.0, 31.0], [10.0, 11.0]],
            "ap-dino.npz",
        )
        lat_dino = _cache(
            [
                _meta("study", "accession-a", "Lateral", "M2"),
                _meta("study", "accession-b", "Lateral", "M3"),
                _meta("study", "accession-c", "Lateral", "M3"),
            ],
            [[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]],
            "lat-dino.npz",
        )
        ap_radiomics = _cache(
            [
                _meta("study", "accession-a", "AP", "M2"),
                _meta("study", "accession-c", "AP", "M2"),
                _meta("study", "accession-b", "AP", "M3"),
            ],
            [[1_000.0], [3_000.0], [2_000.0]],
            "ap-radiomics.npz",
        )
        lat_radiomics = _cache(
            [
                _meta("study", "accession-b", "Lateral", "M3"),
                _meta("study", "accession-c", "Lateral", "M3"),
                _meta("study", "accession-a", "Lateral", "M2"),
            ],
            [[20_000.0], [30_000.0], [10_000.0]],
            "lat-radiomics.npz",
        )

        cohort = E.build_matched_cohort(ap_dino, lat_dino, ap_radiomics, lat_radiomics)

        self.assertEqual(cohort.audit["matched_pairs"], 3)
        self.assertEqual(cohort.audit["concordant_strict_pairs"], 2)
        self.assertEqual(cohort.audit["discordant_strict_pairs"], 1)
        np.testing.assert_array_equal(cohort.y, [0, 1])
        np.testing.assert_allclose(cohort.features["ap_temporal"], [[10, 11], [20, 21]])
        np.testing.assert_allclose(
            cohort.features["lateral_temporal"], [[100, 101], [200, 201]]
        )
        np.testing.assert_allclose(
            cohort.features["biview_radiomics"], [[1_000, 10_000], [2_000, 20_000]]
        )

    def test_hierarchy_tuning_uses_oof_score_and_deterministic_ties(self) -> None:
        labels = np.asarray([0, 1, 2])
        segment = np.asarray([0, 1, 0])
        # Thresholds 0.3 and 0.4 are equally perfect; 0.4 is nearest 0.5.
        gate = np.asarray([0.1, 0.1, 0.45])

        selected = E.select_hierarchy_parameters(gate, gate, segment, labels)

        self.assertEqual(selected["radiomics_weight"], 0.0)
        self.assertEqual(selected["threshold"], 0.4)
        self.assertEqual(selected["inner_macro_f1"], 1.0)

    def test_late_blend_selects_smallest_radiomics_weight_that_repairs_errors(self) -> None:
        labels = np.asarray([0, 1, 2])
        dino = np.asarray(
            [
                [0.4, 0.5, 0.1],
                [0.1, 0.4, 0.5],
                [0.5, 0.1, 0.4],
            ]
        )
        radiomics = np.eye(3)

        selected = E.select_late_weight(dino, radiomics, labels)

        self.assertEqual(selected["radiomics_weight"], 0.1)
        self.assertEqual(selected["inner_macro_f1"], 1.0)

    def test_hierarchy_scores_are_finite_and_sum_to_one(self) -> None:
        scores = E.hierarchy_scores(
            np.asarray([0.0, 0.25, 1.0]),
            np.asarray([-1_000.0, 0.0, 1_000.0]),
        )

        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1.0)
        np.testing.assert_allclose(scores[-1], [0.0, 0.0, 1.0])

    def test_crossfit_primitives_never_predict_on_a_fitted_sample(self) -> None:
        sample_ids = np.arange(8, dtype=float)[:, None]
        cohort = E.MatchedCohort(
            features={
                name: sample_ids.copy()
                for name in (
                    "ap_temporal",
                    "lateral_temporal",
                    "biview_temporal",
                    "biview_radiomics",
                    "biview_all",
                )
            },
            y=np.asarray([2, 0, 1, 2, 2, 0, 1, 2]),
            groups=np.asarray([f"group-{index}" for index in range(8)], dtype=object),
            identities=tuple(str(index) for index in range(8)),
            audit={},
        )
        outer_train = np.asarray([1, 2, 4, 5, 6, 7])
        inner_splits = [
            (np.asarray([0, 1, 2]), np.asarray([3, 4, 5])),
            (np.asarray([3, 4, 5]), np.asarray([0, 1, 2])),
        ]

        def fake_fit(features, labels, kind, context):
            return (set(features[:, 0]), context, 0)

        def check_disjoint(fitted, features):
            fitted_ids, context, _ = fitted
            predicted_ids = set(features[:, 0])
            self.assertFalse(
                fitted_ids.intersection(predicted_ids),
                f"fit/predict leakage in {context}",
            )
            return context

        def fake_logreg(fitted, features):
            context = check_disjoint(fitted, features)
            width = 2 if "gate_" in context else 3
            return np.full((len(features), width), 1.0 / width)

        def fake_svm(fitted, features):
            check_disjoint(fitted, features)
            return np.zeros(len(features), dtype=int), np.zeros((len(features), 2))

        with (
            patch.object(E, "fit_standardized", side_effect=fake_fit),
            patch.object(E, "predict_logreg", side_effect=fake_logreg),
            patch.object(E, "predict_svm", side_effect=fake_svm),
        ):
            output, maximum_iterations = E.crossfit_primitives(
                cohort, outer_train, inner_splits, "test"
            )

        self.assertEqual(maximum_iterations, 0)
        for values in output.values():
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
