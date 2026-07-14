from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import radiomics_experiments as R


class RadiomicsExperimentTests(unittest.TestCase):
    def test_dicom_timing_accepts_leading_zero_t_plus_one_export(self) -> None:
        dataset = SimpleNamespace(FrameTimeVector=[0.0, 10.0, 20.0, 30.0])
        with patch("radiomics_experiments.pydicom.dcmread", return_value=dataset):
            times = R.dicom_frame_times_ms("unused.dcm", n_frames=3)

        np.testing.assert_allclose(times, [0.0, 10.0, 30.0])

    def test_irregular_frame_times_are_resampled_on_normalized_time(self) -> None:
        values = np.asarray([0.0, 0.25, 1.0], dtype=np.float32)
        sequence = np.broadcast_to(values[:, None, None], (3, 2, 2)).copy()
        resampled = R.resample_normalized_time(
            sequence,
            frame_times_ms=np.asarray([0.0, 1.0, 4.0]),
            temporal_samples=5,
        )

        np.testing.assert_allclose(resampled[:, 0, 0], np.linspace(0, 1, 5), atol=1e-6)

    def test_odd_target_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be even"):
            R.extract_feature_families(
                np.ones((2, 18, 18), dtype=np.float32),
                target_size=9,
                border_fraction=0,
            )

    def test_feature_names_match_extracted_dimensions(self) -> None:
        rng = np.random.default_rng(7)
        sequence = rng.normal(size=(6, 16, 16)).astype(np.float32)
        sequence[2:5, 3:11, 5:13] += 2.0

        global_features, spatial_features = R.extract_feature_families(
            sequence,
            target_size=8,
            border_fraction=0,
            gray_levels=8,
        )

        self.assertEqual(global_features.shape, (len(R.feature_names("global")),))
        self.assertEqual(spatial_features.shape, (len(R.feature_names("spatial")),))
        self.assertEqual(len(set(R.feature_names("global"))), len(R.feature_names("global")))
        self.assertEqual(len(set(R.feature_names("spatial"))), len(R.feature_names("spatial")))
        self.assertTrue(np.isfinite(global_features).all())
        self.assertTrue(np.isfinite(spatial_features).all())

    def test_spatial_features_are_left_right_reflection_invariant(self) -> None:
        rng = np.random.default_rng(19)
        sequence = rng.uniform(size=(5, 16, 16)).astype(np.float32)
        sequence[:, :8, :5] *= 0.2
        original = R.extract_feature_families(sequence, 8, 0, 8)
        reflected = R.extract_feature_families(sequence[..., ::-1], 8, 0, 8)

        np.testing.assert_allclose(original[0], reflected[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(original[1], reflected[1], rtol=1e-5, atol=1e-6)

    def test_constant_glcm_has_expected_limiting_values(self) -> None:
        features = R.glcm_features(np.zeros((8, 8), dtype=np.float32), gray_levels=8)
        by_name = dict(zip(R.GLCM_NAMES, features, strict=True))

        self.assertEqual(by_name["contrast"], 0)
        self.assertEqual(by_name["dissimilarity"], 0)
        self.assertEqual(by_name["homogeneity"], 1)
        self.assertEqual(by_name["angular_second_moment"], 1)
        self.assertEqual(by_name["energy"], 1)
        self.assertEqual(by_name["correlation"], 1)
        self.assertEqual(by_name["entropy"], 0)
        self.assertEqual(by_name["maximum_probability"], 1)

    def test_first_order_energy_is_sum_of_squared_intensity(self) -> None:
        features = R.first_order_features(np.ones((8, 8), dtype=np.float32))
        by_name = dict(zip(R.FIRST_ORDER_NAMES, features, strict=True))

        self.assertEqual(by_name["energy"], 64)

    def test_centered_block_mean_uses_only_fixed_size_blocks(self) -> None:
        sequence = np.arange(2 * 16 * 16, dtype=np.float32).reshape(2, 16, 16)
        reduced = R.centered_block_mean(sequence, target_size=8, border_fraction=0)
        expected = sequence.reshape(2, 8, 2, 8, 2).mean(axis=(2, 4))

        self.assertEqual(reduced.shape, (2, 8, 8))
        np.testing.assert_allclose(reduced, expected)
