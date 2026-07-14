# Missing-view backbone comparison

Validated 20 matched split seeds with the same deterministic split protocol, outer/inner fold counts, and aggregate cohort composition.

## Primary union result

| Model | Macro-F1 | M2 / M3 / PCA F1 |
|---|---:|---:|
| DINOv2-L/252-frame-ensemble | 0.651 ± 0.027 | 0.790 / 0.347 / 0.815 |
| V-JEPA2-ViT-g/384-video | 0.633 ± 0.019 | 0.755 / 0.256 / 0.887 |

Candidate-reference paired-seed delta: -0.0178 ± 0.0345; wins/ties/losses 5/0/15.

Per-seed macro-F1 deltas: +0.0285, -0.0203, -0.0577, -0.0285, -0.0412, -0.0583, -0.0388, -0.0134, -0.0489, +0.0159, -0.0110, -0.0262, +0.0254, -0.0105, -0.0072, -0.0090, +0.0067, +0.0604, -0.0867, -0.0357

## Primary result by availability

| Model | Scope | Cases | Macro-F1 | M2 / M3 / PCA F1 |
|---|---|---:|---:|---:|
| DINOv2-L/252-frame-ensemble | union | 344 | 0.651 ± 0.027 | 0.790 / 0.347 / 0.815 |
| DINOv2-L/252-frame-ensemble | paired | 267 | 0.647 ± 0.030 | 0.795 / 0.344 / 0.802 |
| DINOv2-L/252-frame-ensemble | ap_only | 57 | 0.625 ± 0.037 | 0.785 / 0.262 / 0.828 |
| DINOv2-L/252-frame-ensemble | lateral_only | 20 | 0.733 ± 0.060 | 0.709 / 0.489 / 1.000 |
| V-JEPA2-ViT-g/384-video | union | 344 | 0.633 ± 0.019 | 0.755 / 0.256 / 0.887 |
| V-JEPA2-ViT-g/384-video | paired | 267 | 0.634 ± 0.024 | 0.752 / 0.250 / 0.900 |
| V-JEPA2-ViT-g/384-video | ap_only | 57 | 0.645 ± 0.044 | 0.808 / 0.201 / 0.927 |
| V-JEPA2-ViT-g/384-video | lateral_only | 20 | 0.334 ± 0.063 | 0.629 / 0.374 / 0.000 |

## Direct missing-view controls

| Model | Union macro-F1 | M2 / M3 / PCA F1 | Primary-control delta |
|---|---:|---:|---:|
| DINOv2-L/252-frame-ensemble | 0.612 ± 0.023 | 0.792 / 0.264 / 0.782 | +0.0383 ± 0.0362 |
| V-JEPA2-ViT-g/384-video | 0.601 ± 0.021 | 0.761 / 0.196 / 0.844 | +0.0322 ± 0.0274 |

Cache validation passed: ordered case identity, label, patient-group, and view-availability arrays match exactly. No identifiers are included in this report.

Split seeds reuse the same patients and are correlated. The paired-seed deltas measure split sensitivity, not independent-sample uncertainty or external validation.
