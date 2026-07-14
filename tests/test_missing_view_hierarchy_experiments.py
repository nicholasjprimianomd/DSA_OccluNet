from __future__ import annotations

import unittest

import numpy as np

from compare_feature_caches import canonical_metadata
import missing_view_hierarchy_experiments as E


def _meta(study: str, accession: str, view: str, label: str) -> dict[str, str]:
    return {
        "study_key": study,
        "accession": accession,
        "view": view,
        "run_column": f"{view}_1",
        "label_text": label,
    }


def _cache(meta: list[dict[str, str]], offset: float = 0.0) -> dict[str, object]:
    metadata = np.asarray(meta, dtype=object)
    return {
        "mean": np.asarray([[offset + index, offset + index + 0.5] for index in range(len(meta))]),
        "groups": np.asarray([row["study_key"] for row in meta], dtype=object),
        "meta": metadata,
        "metadata_identity": canonical_metadata(metadata),
    }


class MissingViewHierarchyTests(unittest.TestCase):
    def test_union_keeps_one_case_per_identity_and_excludes_ambiguous_pairs(self) -> None:
        ap = _cache(
            [
                _meta("paired", "a", "AP", "M2"),
                _meta("ap-only", "b", "AP", "M3"),
                _meta("discordant", "c", "AP", "M2"),
                _meta("nonstrict", "d", "AP", "M4"),
            ]
        )
        lateral = _cache(
            [
                _meta("paired", "a", "Lateral", "M2"),
                _meta("lat-only", "e", "Lateral", "P2"),
                _meta("discordant", "c", "Lateral", "M3"),
                _meta("nonstrict", "d", "Lateral", "M4"),
            ],
            offset=10.0,
        )

        cohort = E.build_union_cohort(ap, lateral)

        self.assertEqual(cohort.audit["n_cases"], 3)
        self.assertEqual(list(cohort.availability), [E.AP_ONLY, E.LATERAL_ONLY, E.PAIRED])
        np.testing.assert_array_equal(cohort.y, [1, 2, 0])
        self.assertEqual(len(set(cohort.identities)), 3)
        self.assertEqual(len(cohort.audit["exclusions"]["discordant_strict_pairs"]), 1)
        self.assertEqual(cohort.audit["exclusions"]["paired_nonstrict"], 1)
        self.assertTrue(np.isnan(cohort.ap[cohort.availability == E.LATERAL_ONLY]).all())
        self.assertTrue(np.isnan(cohort.lateral[cohort.availability == E.AP_ONLY]).all())

    def test_viewwise_hierarchy_uses_available_fallback_and_ap_for_paired_segment(self) -> None:
        availability = np.asarray([E.PAIRED, E.AP_ONLY, E.LATERAL_ONLY], dtype=object)
        primitives = E.empty_primitives(3)
        primitives["all_gate_ap"][[0, 1]] = [0.2, 0.8]
        primitives["all_gate_lateral"][[0, 2]] = [0.6, 0.1]
        primitives["all_segment_ap_pred"][[0, 1]] = [1, 0]
        primitives["all_segment_ap_decision"][[0, 1]] = [2.0, -2.0]
        primitives["all_segment_lateral_pred"][[0, 2]] = [0, 1]
        primitives["all_segment_lateral_decision"][[0, 2]] = [-2.0, 2.0]

        prediction, scores = E.viewwise_hierarchy_outputs(
            primitives, availability, lateral_weight=0.5, threshold=0.5, prefix="all"
        )

        np.testing.assert_array_equal(prediction, [1, 2, 1])
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores.sum(axis=1), 1.0)

    def test_direct_missing_view_blends_only_true_pairs(self) -> None:
        availability = np.asarray([E.PAIRED, E.AP_ONLY, E.LATERAL_ONLY], dtype=object)
        primitives = E.empty_primitives(3)
        primitives["all_probability_ap"][[0, 1]] = [[1, 0, 0], [0, 1, 0]]
        primitives["all_probability_lateral"][[0, 2]] = [[0, 0, 1], [1, 0, 0]]

        probability = E.direct_missing_view_probability(
            primitives, availability, lateral_weight=0.25
        )

        np.testing.assert_allclose(probability[0], [0.75, 0.0, 0.25])
        np.testing.assert_allclose(probability[1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(probability[2], [1.0, 0.0, 0.0])

    def test_pair_prefix_outputs_only_need_to_be_complete_on_pairs(self) -> None:
        availability = np.asarray([E.PAIRED, E.AP_ONLY], dtype=object)
        primitives = E.empty_primitives(2)
        primitives["pair_gate_ap"][0] = 0.2
        primitives["pair_gate_lateral"][0] = 0.8
        primitives["pair_segment_ap_pred"][0] = 1
        primitives["pair_segment_ap_decision"][0] = 1.0

        prediction, scores = E.viewwise_hierarchy_outputs(
            primitives, availability, lateral_weight=0.5, threshold=0.5, prefix="pair"
        )

        self.assertEqual(prediction[0], 2)
        self.assertEqual(prediction[1], -1)
        self.assertTrue(np.isfinite(scores[0]).all())
        self.assertTrue(np.isnan(scores[1]).all())


if __name__ == "__main__":
    unittest.main()
