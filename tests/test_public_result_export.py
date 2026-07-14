from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_public_experiment_result import export_public_result


class PublicResultExportTests(unittest.TestCase):
    def test_removes_identifiers_oof_arrays_and_absolute_repo_prefix(self) -> None:
        raw = {
            "protocol": {"group": "Study_Key"},
            "cohort_audit": {
                "discordant": [
                    {
                        "study_key": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                        "accession": "123456",
                        "ap_label": "m2",
                    }
                ]
            },
            "feature_sources": {
                "ap": {"path": "/home/user/DSA_OccluNet/runs/cache.npz", "sha256": "abc"}
            },
            "per_seed": {
                "model": [
                    {
                        "seed": 0,
                        "macro_f1": 0.5,
                        "oof_predictions": [0, 1],
                        "oof_scores": [[0.8, 0.2], [0.1, 0.9]],
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            source.write_text(json.dumps(raw))
            public = export_public_result(source)

        self.assertEqual(public["cohort_audit"]["discordant_count"], 1)
        self.assertEqual(public["feature_sources"]["ap"]["path"], "runs/cache.npz")
        row = public["per_seed"]["model"][0]
        self.assertNotIn("oof_predictions", row)
        self.assertNotIn("oof_scores", row)
        self.assertEqual(row["macro_f1"], 0.5)
        encoded = json.dumps(public)
        self.assertNotIn("aaaaaaaa", encoded)
        self.assertNotIn("123456", encoded)


if __name__ == "__main__":
    unittest.main()
