# Experiment 16: classical SVM + radiomics ensembled with the best neural probe

**Status:** code complete and validated on synthetic grouped data; **not yet run on the
real DSA cohort** (this was authored in an environment without the PHI data, a GPU, or the
cached neural features — all of which are gitignored and live on the training machine). The
run commands below are ready to execute where the data lives, and the results table has
placeholders to fill in.

Tooling: [radiomics_probe.py](../radiomics_probe.py) (hand-crafted texture features) and
[ensemble_experiments.py](../ensemble_experiments.py) (leak-free heterogeneous ensemble).

## Motivation

Every prior experiment probes a *single* representation — a frozen neural backbone — with a
linear head. Two findings from [experiment_findings.md](experiment_findings.md) say a
different *view* of the data, not a better head, is what m3 needs:

- The attentive-pooling probe (finding 3) could not separate m2 from m3 even with a learned
  pooling over all 2048 tokens. The m2/m3 distinction is only weakly present *in the deep
  features*, so a better head on those same features cannot fix it.
- m3 F1 has been stuck at ~0.30–0.33 across every backbone, resolution, and augmentation.

Radiomics features (first-order intensity statistics plus GLCM/GLRLM/GLSZM/GLDM/NGTDM
texture, and 2-D shape when a real ROI exists) are a genuinely different, hand-crafted view
of the same runs. If they carry even partial *independent* signal for m3, a heterogeneous
ensemble of the classical route (SVM on radiomics) and the deep route can recover accuracy
that neither member has alone. This is a well-supported pattern in medical imaging:

- A CT brain-metastases study built a stacking ensemble over radiomics with SVM as the
  strongest base learner ([Sci Rep 2024](https://www.nature.com/articles/s41598-024-80210-x)).
- Integrating radiomics with deep-learning features via early fusion (concatenation) and
  late fusion (probability stacking) improved brain-tumor MRI classification over either
  alone ([J Xray Sci Technol 2025](https://journals.sagepub.com/doi/10.1177/08953996241299996)).
- Most relevant here: a **DSA** AVM-diagnosis model combining temporal features with spatial
  radiomics reached AUC **0.942** versus **0.916 / 0.918** for temporal-only and
  radiomics-only ([Front Neurol 2021](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2021.655523/full)).

## What the code does

**`radiomics_probe.py`** reuses the exact DICOM loading, percentile windowing, frame
sampling, label mapping, and `.npz` cache format of the neural probes, so its output aligns
run-for-run with the deep features on the same patient-grouped folds. For each of the 16
sampled frames it computes PyRadiomics features inside a per-frame ROI (`otsu`, the minority
intensity population = contrast/vessel structure, with a full-frame fallback when that
population is degenerate; or `full`), then temporally pools them into `mean`/`max`/`std`
vectors. The cache carries a content signature and per-run metadata identity, exactly like
`image_backbone_probe.py`.

**`ensemble_experiments.py`** aligns the radiomics cache to one or more neural caches by
canonical run metadata (never row position), then evaluates six methods on the *same* 20
repeated `StratifiedGroupKFold` seeds:

| Method | What it is |
|---|---|
| `nn_only` | reference: balanced logistic C3 on the deep features (optionally V-JEPA+DINO early fusion) |
| `radiomics_only` | the classical route: RBF-SVM (calibrated) on radiomics |
| `late_fusion_equal` | mean of the two members' predicted probabilities |
| `late_fusion_weighted` | probability blend, weight α chosen by **inner grouped CV** on the training patients only |
| `stacking_logreg` | balanced logistic meta-learner over the two members' **inner grouped OOF** probabilities |
| `early_fusion` | one learner over the standardized concatenation of both raw feature sets |

Every learned combiner (α, the stacker) is fit with an inner grouped cross-validation on the
outer *training* patients, so no validation patient influences the combiner. This mirrors the
nested discipline already used for class routing in
[three_class_augmentation_experiments.py](../three_class_augmentation_experiments.py); the
numbers are leak-free, not a best-of selection. The neural-only probe is the reference, so
the table directly answers "does adding radiomics help, and by how much?".

The script is task-agnostic: point it at the legacy 3-class caches, or at the strict
M2/M3/PCA / clean-M2-vs-M3 caches from experiment 14, and it reads the class layout from the
cache.

## Machinery validation (synthetic, not a performance claim)

`python ensemble_experiments.py --synthetic-demo` fabricates grouped, imbalanced 3-class data
in which the deep view separates m2/other well but is deliberately weak on m3, while the
radiomics view carries partial *independent* m3 signal. It exists to prove the fold/leakage
plumbing runs and to show the mechanism. On 10 seeds (351 fake runs, 235 fake patients):

| Method | macro-F1 | class F1 (m2/m3/other) | Δ vs nn_only |
|---|---:|---:|---:|
| early_fusion | 0.682 ± 0.014 | 0.89 / 0.52 / 0.63 | **+0.087** |
| stacking_logreg | 0.675 ± 0.016 | 0.88 / 0.54 / 0.61 | +0.081 |
| late_fusion_equal | 0.612 ± 0.011 | 0.89 / 0.38 / 0.57 | +0.018 |
| late_fusion_weighted | 0.610 ± 0.012 | 0.88 / 0.38 / 0.57 | +0.015 |
| nn_only (reference) | 0.595 ± 0.009 | 0.86 / 0.35 / 0.57 | +0.000 |
| radiomics_only | 0.399 ± 0.011 | 0.80 / 0.39 / 0.00 | −0.196 |

The pattern to read here is *mechanistic*: a weak-but-independent second view lifts the
minority class (synthetic m3 0.35 → 0.52) even though it is far worse overall on its own.
**Whether real DSA radiomics carry that independent m3 signal is precisely the open question
the run below answers — do not quote these synthetic numbers as results.**

## Reproduce (on the machine that has the data + neural caches)

Prerequisite: the deep feature caches from experiments 9–13 must already exist under `runs/`
(the same ones referenced in [experiment_findings.md](experiment_findings.md) "Reproduce").

```bash
export DSA_BASE_DIR=~/M2_M3_data

# 1) Extract radiomics (CPU; ~16 frames × ~366 AP runs). Otsu ROI is the default.
.venv/bin/python radiomics_probe.py --view AP --stage positive_subtype \
    --n-frames 16 --roi otsu --out runs/ap_radiomics

# 2) Ensemble the best deep fusion (V-JEPA ViT-g/384 + DINOv2-L/252) with radiomics.
.venv/bin/python ensemble_experiments.py \
    --nn vjepa=runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
    --nn dino=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
    --radiomics runs/ap_radiomics/cache/radiomics_otsu_bc32_16_AP_positive_subtype.npz \
    --nn-classifier logreg_c3 --radiomics-classifier svm_rbf \
    --seeds 0:20 --out runs/ensemble/ap_ensemble_20seeds.json

# Optional: single-backbone deep member instead of the fusion.
.venv/bin/python ensemble_experiments.py \
    --nn dino=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
    --radiomics runs/ap_radiomics/cache/radiomics_otsu_bc32_16_AP_positive_subtype.npz \
    --seeds 0:20 --out runs/ensemble/ap_dino_radiomics_20seeds.json

# Optional: does radiomics also help the cleaner strict tasks from experiment 14?
# Re-extract radiomics for those cohorts and pass the strict-task neural caches the same way.
```

The radiomics-only sanity check (does the classical route beat the majority baseline at all?)
also prints from `radiomics_probe.py` directly, via the shared recipe sweep.

### Results — TO FILL IN from the real run

| Method | macro-F1 (20 seeds) | class F1 (m2/m3/other) | Δ vs nn_only | wins |
|---|---:|---:|---:|---:|
| nn_only (reference) | _tbd_ | _tbd_ | +0.000 | — |
| radiomics_only | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| late_fusion_equal | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| late_fusion_weighted | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| stacking_logreg | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| early_fusion | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Caveats and honest expectations

1. **No vessel segmentation.** The strongest DSA radiomics results use a segmented
   artery/vein mask; the AVM study's morphology features assume one. This probe uses a
   whole-frame Otsu ROI, which captures global texture but not vessel morphology. If
   whole-frame radiomics disappoint, the next step is a real vessel mask (there are public
   DSA cerebral-artery segmentation models), not abandoning the idea. This is the single
   biggest reason the real result could underperform the literature.
2. **The ensemble only helps if radiomics are *independently* right on m3.** If the deep
   features and radiomics fail on the *same* m3 runs, fusion cannot help — it averages two
   correlated errors. The `radiomics_only` column and the per-class confusion are the
   diagnostic: look at whether radiomics recover any m3 the deep model misses.
3. **PyRadiomics packaging.** The published `pyradiomics` wheel does not build against
   NumPy ≥ 2 / Python 3.13. Install it in an isolated environment with NumPy < 2 (e.g. a
   dedicated `.venv-radiomics` with `numpy<2 SimpleITK pyradiomics`) and point
   `radiomics_probe.py` at that interpreter; the resulting `.npz` cache is then consumed by
   the main environment. `radiomics_probe.py` imports PyRadiomics lazily so the rest of the
   toolchain never depends on it.
4. **Same locked-test-set caveat as everything else.** These are repeated patient-grouped CV
   scores; a headline gain here still needs a locked patient-level test cohort or external
   validation before it is deployable, exactly as noted throughout
   [experiment_findings.md](experiment_findings.md).

Recorded validation environment: Python 3.11, NumPy 2.4.6, scikit-learn 1.9.0, joblib 1.5.3,
PyTorch 2.13.0+cpu. The ensemble uses only NumPy/scikit-learn/joblib; PyTorch is pulled in
only transitively by the shared `experiments.py` import.
