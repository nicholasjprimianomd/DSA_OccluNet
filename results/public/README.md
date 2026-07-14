# Public experiment manifests

This directory preserves the aggregate and seed-level evidence for the frozen-feature,
radiomics, multiview, and missing-view experiments. The source `runs/` directory remains
gitignored because raw caches and some audit records contain accession-level metadata.

Public manifests retain protocol settings, aggregate metrics, per-seed metrics, selected
inner-fold parameters, comparisons, cache hashes, software versions, and aggregate cohort
counts. They remove accession/Study_Key examples, sample identities, per-case OOF arrays,
and local absolute path prefixes. Generate them with:

```bash
.venv/bin/python scripts/export_public_experiment_result.py RAW_RESULT.json PUBLIC_RESULT.json
```

The two principal private source artifacts for the multiview work are preserved locally and
identified publicly by SHA-256:

| Experiment | Public manifest | Private source SHA-256 |
|---|---|---|
| Nested AP+lateral ensemble | `multiview_ensembles_20seeds.json` | `778cbad6a2b44546cffc7250686efec80021d1303cec023d61176da4ac44adc3` |
| Paired+unpaired missing-view hierarchy | `missing_view_hierarchy_20seeds.json` | `f6b8b9fdee7b681439af60c1d90e6a58170cf75bc21969298adbf74831f37f67` |
| V-JEPA 2 video missing-view hierarchy | `video_vjepa2_missing_view_20seeds.json` | `f733b726a4e3ab5df1b8037066ca7e11f525a802763377e25f6825ab2117426e` |

The remaining manifests cover the repeated anatomy SVM controls, radiomics probes and
ablations, seven-backbone comparisons, the signed RadImageNet validation, and the
identifier-free matched video-versus-frame comparison in
`video_vs_frame_missing_view_comparison_20seeds.{json,md}`. Feature NPZ caches, model
checkpoints, spreadsheets, and visualization ID maps must not be committed.
