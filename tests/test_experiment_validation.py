from __future__ import annotations

import argparse
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning

import experiments
from compare_feature_caches import load_cache, parse_seeds, score_for_seed
from anatomy_task_experiments import new_probe, parse_name_list, select_named
from image_backbone_probe import dicom_frame_times_ms
from radimagenet_probe import extract as extract_radimagenet
from radimagenet_probe import validate_cache_metadata


class _UnconvergedClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features, labels):
        warnings.warn("did not converge", ConvergenceWarning)
        self.classes_ = np.unique(labels)
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=int)


class ExperimentValidationTests(unittest.TestCase):
    def test_frame_time_vector_accepts_leading_zero_t_plus_one_export(self):
        dataset = SimpleNamespace(FrameTimeVector=[0.0, 10.0, 20.0, 30.0])
        with patch("pydicom.dcmread", return_value=dataset):
            times = dicom_frame_times_ms("unused.dcm", n_frames=3)

        np.testing.assert_allclose(times.numpy(), [0.0, 10.0, 30.0])

    def test_grouped_fold_count_uses_distinct_groups_per_class(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        groups = np.asarray(["same", "same", "same", "b", "c", "d"], dtype=object)

        with self.assertRaisesRegex(ValueError, "two distinct groups in every class"):
            experiments.grouped_fold_count(labels, groups, requested=5, num_classes=2)

    def test_make_folds_keeps_groups_disjoint_and_covers_every_sample(self):
        labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)

        folds = experiments.make_folds(labels, groups, n_splits=2, seed=0)

        held_out = np.concatenate([valid for _, valid in folds])
        np.testing.assert_array_equal(np.sort(held_out), np.arange(len(labels)))
        for train, valid in folds:
            self.assertFalse(set(groups[train]).intersection(groups[valid]))

    def test_linear_svm_convergence_warning_fails_the_run(self):
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0, 0, 1, 1])
        folds = [(np.asarray([0, 2]), np.asarray([1, 3]))]

        with patch("experiments.clf_factory", return_value=lambda: _UnconvergedClassifier()):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                experiments.evaluate_recipe(
                    features,
                    labels,
                    folds,
                    num_classes=2,
                    prep="std",
                    clf="linsvm",
                )

    def test_anatomy_linear_svm_uses_hardened_iteration_limit(self):
        estimator = new_probe("std_linsvm").steps[-1][1]
        self.assertEqual(estimator.max_iter, experiments.LINEAR_SVM_MAX_ITER)
        self.assertEqual(estimator.dual, experiments.LINEAR_SVM_DUAL)
        self.assertEqual(estimator.random_state, 0)
        self.assertEqual(estimator.tol, experiments.LINEAR_SVM_TOL)

    def test_mean_comparison_accepts_validated_legacy_mean_only_cache(self):
        metadata = np.asarray(
            [
                {
                    "study_key": "study-a",
                    "accession": "accession-a",
                    "view": "AP",
                    "run_column": "AP_1",
                },
                {
                    "study_key": "study-b",
                    "accession": "accession-b",
                    "view": "AP",
                    "run_column": "AP_1",
                },
            ],
            dtype=object,
        )
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "legacy.npz"
            np.savez(
                cache_path,
                mean=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                labels=np.asarray([0, 1]),
                groups=np.asarray(["study-a", "study-b"], dtype=object),
                meta=metadata,
            )

            loaded = load_cache(cache_path, "mean")

            self.assertFalse(loaded["has_label_text_metadata"])
            self.assertEqual(loaded["mean"].shape, (2, 2))
            with self.assertRaisesRegex(ValueError, "max, std"):
                load_cache(cache_path, "meanmaxstd")

    def test_partial_label_metadata_is_rejected(self):
        metadata = np.asarray(
            [
                {
                    "study_key": "study-a",
                    "accession": "accession-a",
                    "view": "AP",
                    "run_column": "AP_1",
                    "label_text": "M2",
                },
                {
                    "study_key": "study-b",
                    "accession": "accession-b",
                    "view": "AP",
                    "run_column": "AP_1",
                },
            ],
            dtype=object,
        )
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "partial.npz"
            np.savez(
                cache_path,
                mean=np.asarray([[1.0], [2.0]]),
                labels=np.asarray([0, 1]),
                groups=np.asarray(["study-a", "study-b"], dtype=object),
                meta=metadata,
            )

            with self.assertRaisesRegex(ValueError, "partial label_text"):
                load_cache(cache_path, "mean")

    def test_present_but_empty_label_text_is_modern_metadata(self):
        metadata = np.asarray(
            [
                {
                    "study_key": "study-a",
                    "accession": "accession-a",
                    "view": "AP",
                    "run_column": "AP_1",
                    "label_text": "",
                },
                {
                    "study_key": "study-b",
                    "accession": "accession-b",
                    "view": "AP",
                    "run_column": "AP_1",
                    "label_text": "M2",
                },
            ],
            dtype=object,
        )
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "empty-label.npz"
            np.savez(
                cache_path,
                mean=np.asarray([[1.0], [2.0]]),
                labels=np.asarray([0, 1]),
                groups=np.asarray(["study-a", "study-b"], dtype=object),
                meta=metadata,
            )

            self.assertTrue(load_cache(cache_path, "mean")["has_label_text_metadata"])

    def test_seed_zero_lookup_is_independent_of_cli_order(self):
        rows = [
            {"seed": 2, "macro_f1": 0.2},
            {"seed": 0, "macro_f1": 0.7},
            {"seed": 5, "macro_f1": 0.1},
        ]

        self.assertEqual(score_for_seed(rows, 0), 0.7)

    def test_seed_parser_sorts_and_rejects_duplicates(self):
        self.assertEqual(parse_seeds("5,0,2"), [0, 2, 5])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unique"):
            parse_seeds("0,0")

    def test_named_subset_parser_and_selector_preserve_requested_order(self):
        requested = parse_name_list("fusion,dino")
        self.assertEqual(select_named(["vjepa", "dino", "fusion"], requested, "source"), ["fusion", "dino"])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "unique"):
            parse_name_list("dino,dino")
        with self.assertRaisesRegex(ValueError, "Unknown source: missing"):
            select_named(["dino"], ("missing",), "source")

    def test_radimagenet_signed_metadata_requires_label_text(self):
        expected = [
            {
                "accession": "a",
                "view": "AP",
                "run_column": "AP_1",
                "study_key": "study-a",
                "label_text": "M2",
            }
        ]
        legacy = [{key: expected[0][key] for key in ("accession", "view", "run_column", "study_key")}]
        modern = [dict(expected[0])]

        self.assertEqual(validate_cache_metadata(legacy, expected, require_label_text=False), (True, False))
        self.assertEqual(validate_cache_metadata(legacy, expected, require_label_text=True), (False, False))
        self.assertEqual(validate_cache_metadata(modern, expected, require_label_text=True), (True, True))

    def test_radimagenet_legacy_audit_needs_no_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dicom_path = root / "sample.dcm"
            dicom_path.write_bytes(b"identity-only")
            cache_dir = root / "cache"
            cache_dir.mkdir()
            metadata = {
                "accession": "accession-a",
                "view": "AP",
                "run_column": "AP_1",
                "study_key": "study-a",
            }
            np.savez(
                cache_dir / "radimagenet_ResNet50_AP_positive_subtype.npz",
                mean=np.zeros((1, 2048), dtype=np.float32),
                labels=np.asarray([0]),
                groups=np.asarray(["study-a"], dtype=object),
                meta=np.asarray([metadata], dtype=object),
            )
            record = SimpleNamespace(
                accession="accession-a",
                view="AP",
                run_column="AP_1",
                study_key="study-a",
                label_text="M2",
                dicom_path=dicom_path,
            )

            _, _, _, _, cache_info = extract_radimagenet(
                "AP",
                "positive_subtype",
                [record],
                model=None,
                device=None,
                n_frames=16,
                out_cache=cache_dir,
                arch="ResNet50",
                model_provenance=None,
                allow_legacy_cache=True,
            )

            self.assertTrue(cache_info["legacy_unsigned"])


if __name__ == "__main__":
    unittest.main()
