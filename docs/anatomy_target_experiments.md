# Anatomically coherent target experiments

Updated: 2026-07-12

## Why these experiments exist

The historical `other_positive` class is not a coherent anatomical target. Its 59 AP runs
contain 5 M1, 10 M4, 15 ACA, and 29 PCA occlusions. A model can therefore score well on
`other_positive` by recognizing PCA while failing almost every M1/M4 case.

These experiments replace the catch-all label with explicit tasks. They reuse the frozen
representations that performed best in the earlier study: normalized V-JEPA 2 ViT-g/384,
DINOv2-L/252 with uniform frames, and DINOv2-L/252 with temporal-change frame selection.
The previous partial backbone fine-tune overfit, so no new large end-to-end fine-tune was
attempted on this 184-patient dataset.

## Label audit and target definitions

All counts below are resolved AP runs. Strict tasks exclude composite labels rather than
silently assigning them to one component. The excluded composites are three M2+A3 runs and
two M3+M4 runs; the latter remain valid for the territory-only task because both segments
are MCA.

| Task | Classes and counts | Included runs | Purpose |
|---|---|---:|---|
| `territory_strict` | MCA 319, ACA 15, PCA 29 | 363 | Proper territory model; excludes cross-territory M2+A3 |
| `clean_m2_m3` | M2 237, M3 65 | 302 | Clean primary benchmark; excludes every composite and non-M2/M3 label |
| `m2_m3_pca_strict` | M2 237, M3 65, PCA 29 | 331 | Coherent three-class alternative; excludes ACA, M1/M4, and composites |
| `supported_4class` | M2 237, M3 65, ACA 15, PCA 29 | 346 | Flat explicit alternative without M1/M4 |
| `mca_coarse_strict` | M2 237, M3 65, other-MCA 15 | 317 | Tests whether M1/M4 can form a usable rejection class |
| `mca_4class_strict` | M1 5, M2 237, M3 65, M4 10 | 317 | Sparse segment stress test |
| `anatomy_6class_strict` | M1 5, M2 237, M3 65, M4 10, ACA 15, PCA 29 | 361 | Full explicit anatomy stress test |

## Evaluation protocol

- Patient-grouped `StratifiedGroupKFold` on `Study_Key`; every run from a patient stays in
  one side of a fold.
- Five folds repeated for seeds 0–19. Repeated seeds measure split sensitivity, not 20
  independent test sets.
- Mean-pooled frozen features only. Feature scaling and the classifier are fit inside each
  training fold.
- Compared normalized V-JEPA, uniform DINO, temporal-change DINO, and four direct feature
  concatenations.
- Compared the convergent documented probe recipes: standardized balanced logistic
  regression at C=1 and C=3, plus L2-normalized balanced logistic regression.
- A broader LinearSVC grid was stopped because high-dimensional fusion fits repeatedly
  failed to converge. This matches the earlier instability of large rich representations.
- After cross-validation, one full-data probe per task was fit using the best mean-CV
  feature/recipe combination. Those saved probes are inference artifacts, not new held-out
  evidence.

## Results

The table reports the best repeated-CV configuration for each task. Class F1 follows the
class order shown in the second column.

| Task | Best frozen representation / probe | Macro-F1 | Mean class F1 |
|---|---|---:|---|
| **Clean M2 vs M3** | temporal-change DINO / std logreg C3 | **0.581 ± 0.025** | M2 **0.829**, M3 **0.332** |
| **M2 vs M3 vs PCA** | uniform DINO / std logreg C1 | **0.619 ± 0.026** | M2 **0.807**, M3 **0.332**, PCA **0.718** |
| Territory | V-JEPA + uniform DINO / std logreg C1 | 0.594 ± 0.019 | MCA 0.953, **ACA 0.050**, PCA 0.780 |
| Supported M2/M3/ACA/PCA | temporal-change DINO / std logreg C3 | 0.463 ± 0.016 | M2 0.779, M3 0.300, **ACA 0.073**, PCA 0.699 |
| M2/M3/other-MCA | temporal-change DINO / std logreg C3 | 0.384 ± 0.027 | M2 0.787, M3 0.293, **other-MCA 0.071** |
| MCA M1/M2/M3/M4 | temporal-change DINO / std logreg C3 | 0.282 ± 0.031 | **M1 0.027**, M2 0.787, M3 0.284, **M4 0.030** |
| Full six-class anatomy | V-JEPA / std logreg C1 | 0.301 ± 0.027 | M1 0.105, M2 0.726, M3 0.195, **M4 0.000**, **ACA 0.021**, PCA 0.758 |

Full per-seed metrics, precision/recall, confusion matrices, cache hashes, and trained-model
hashes are stored in `runs/anatomy_tasks/results.json`.

## What the experiments show

### 1. Clean M2-vs-M3 remains the defensible binary target

Removing unrelated territories and ambiguous composites raises the direct M2-vs-M3
macro-F1 to 0.581. Temporal-change DINO is best and fusion is worse, consistent with the
earlier finding that DINO carries the strongest m3 signal. M3 remains difficult: mean F1 is
0.332 and recall is 0.315, so this is a research baseline rather than a clinical classifier.

### 2. M2/M3/PCA is the best coherent three-class formulation tested

The strict M2/M3/PCA experiment uses 331 runs from 171 patient groups and reaches
**0.619 ± 0.026 macro-F1** with uniform-frame DINO and standardized balanced logistic
regression C=1. Class F1 is 0.807/0.332/0.718. Adding PCA leaves m3 F1 unchanged from the
clean binary task while adding a third class the model can learn reasonably well.

Mean recall is 0.818 for M2, 0.328 for M3, and 0.669 for PCA. Most remaining m3 errors are
still predictions of M2 (65.8% of true m3 predictions across repeated OOF runs). PCA is
misclassified as M2 22.2% and as M3 10.9% of the time. The higher macro-F1 than the binary
task is not a like-for-like improvement—the metric now includes an easier PCA class—but it
does show that PCA is a coherent, learnable replacement for the old catch-all label.

This model does **not** cover all positive angiograms. ACA, M1, M4, and composite labels are
outside its supported scope and must be excluded or handled separately.

### 3. A proper territory hierarchy is anatomically correct but not yet operational

The territory model's 0.594 macro-F1 looks respectable until class performance is examined:
MCA F1 is 0.953 and PCA is 0.780, but ACA is only 0.050. With just 15 ACA runs, the first
stage of a territory→MCA-segment cascade cannot reliably recognize ACA. The hierarchy fixes
the ontology, not the data shortage.

### 4. M1/M4 cannot currently be learned or used as a reliable abstention class

Combining all 15 M1/M4 runs into `other_mca` produces F1 0.071. Separating them produces
M1 F1 0.027 and M4 F1 0.030 in the MCA-only experiment. No tested feature or documented
probe recipe changes this conclusion. A trained M2/M3 model must therefore treat M1/M4 as
unsupported scope; it cannot be trusted to identify and abstain on them automatically.

### 5. Replacing `other_positive` with four or six explicit labels is honest but not useful yet

The four-class M2/M3/ACA/PCA model is dominated by M2 and PCA and almost never learns ACA.
The six-class model similarly fails M1, M4, and ACA. These targets should be retained as
data-readiness tests, not promoted as production models.

### 6. The negative result is robust to the tested head choices

Changing logistic regularization or using L2-normalized embeddings does not rescue ACA,
M1, M4, or other-MCA. L2 normalization often increases minority prediction frequency but
reduces the stronger classes enough to lower macro-F1. The limiting factor is label support,
not a missing linear-probe recipe.

## Recommended model policy

1. Use `clean_m2_m3` when the scientific question is specifically M2 versus M3.
2. Use `m2_m3_pca_strict` when a coherent three-class endpoint is required; explicitly state
   that ACA, M1, M4, and composites are outside scope.
3. Remove `other_positive` from new headline results.
4. Use the territory and explicit-class experiments only as diagnostic tasks until ACA,
   M1, and M4 are materially expanded.
5. Exclude composites from the strict single-label headline. The training-only half-weight
   component-presence sensitivity below is exploratory; if composites matter clinically,
   collect enough of them and move to a multilabel target rather than priority mapping.
6. Treat M1/M4 as unsupported for the saved M2/M3 probe. The current data do not support a
   reliable automatic rejection mechanism.
7. Establish a locked patient-level test cohort before treating any selected feature/recipe
   as final performance.

## AP + lateral and composite-label follow-up

The follow-up tested whether the lateral runs improve the strict M2/M3/PCA endpoint and
whether the excluded composites can contribute without corrupting its meaning. The same
frozen DINOv2-L/252 extractor was applied to 16 frames per run. A run embedding is the mean
of its 16 frame embeddings (1,024 values). Paired fusion concatenates the matched AP and
lateral embeddings into 2,048 values; scaling and the balanced logistic classifier are fit
inside each training fold.

### What “temporal pooling” means here

This is not a recurrent network, attention over a video, or a learned temporal convolution.
For each individual DICOM run, the frozen DINOv2-L image encoder produces one 1,024-value
embedding per selected frame. The run representation is the arithmetic mean of those 16
embeddings. The classifier sees one row per run after this operation. No frames from another
run or another patient are averaged together.

There were two frame-selection recipes:

- `uniform`: 16 indices evenly spaced from the first through last frame of that DICOM run;
- `top_contrast` / temporal-change: retain frame 0 plus the 15 frames with the largest mean
  absolute pixel change from the preceding frame, then sort them back into acquisition order.

The second recipe is therefore a hand-designed way of emphasizing injection/filling changes;
it is not temporal supervision. Runs shorter than 16 frames use the uniform sampler with
repeated indices. The per-frame encoder is run independently, then only the final 16 vectors
are averaged. The cache also stores per-dimension max and standard deviation, but the anatomy
follow-up uses the mean representation only.

### What “pooled AP+lateral” means

Two different operations were tested and should not be conflated:

1. **Pooled-as-separate-rows:** AP and lateral runs are placed in one training table as
   separate examples, with a one-value view indicator. An AP run is not averaged with its
   lateral run. This was the 0.588 result and was worse than matched fusion.
2. **Matched fusion:** one AP embedding is concatenated with one lateral embedding. The
   matching key is `(Study_Key, run index)`: `AP_1` with `Lateral_1`, `AP_2` with
   `Lateral_2`, and so on. The two views have the same accession in all 304 matched pairs,
   but they are separate DICOM entries (for example, the spreadsheet may list different
   AP and lateral series values). The code does not claim timestamp-level simultaneity; it
   uses the dataset’s indexed AP/lateral correspondence.

Different indexed runs are not combined. If a patient has `AP_1/Lateral_1` and
`AP_2/Lateral_2`, those are two separate paired samples and could represent different
catheter positions or acquisitions. Across the 304 available indexed matches, 167 patients
contributed at least one pair; 96 contributed more than one. The strict fusion subset contains
267 concordant pairs from 151 patients. Seven pairs disagree in their strict AP versus
lateral labels and were excluded rather than forcing them into one target. Different patients
were never fused together. For cross-validation, all runs and both views from a patient were
assigned to the same fold, so a patient could contribute multiple training rows but could not
appear in both training and validation.

There are 304 AP/lateral run-index matches. Of these, 267 have the same strict label in both
views (M2 187, M3 58, PCA 22), while seven are labeled M2 on AP and M3 on lateral. The seven
discordant pairs are excluded from paired fusion because they do not have one defensible
exclusive target. Every split remains grouped by patient.

| Experiment | Strict evaluation set | Macro-F1 | Class F1 (M2 / M3 / PCA) |
|---|---:|---:|---:|
| AP only, all eligible AP runs | 331 runs / 171 patients | 0.619 ± 0.026 | 0.807 / 0.332 / 0.718 |
| Lateral only, uniform frames | 294 / 156 | 0.601 ± 0.020 | 0.725 / 0.274 / 0.804 |
| Lateral only, temporal-change frames | 294 / 156 | 0.602 ± 0.020 | 0.726 / 0.268 / 0.811 |
| Matched AP-only control | 267 / 151 | 0.572 ± 0.031 | 0.785 / 0.308 / 0.624 |
| Matched lateral-only control | 267 / 151 | 0.591 ± 0.018 | 0.760 / 0.221 / 0.793 |
| **Matched AP+lateral fusion** | **267 / 151** | **0.646 ± 0.021** | **0.800 / 0.317 / 0.823** |
| Matched temporal-change fusion | 267 / 151 | 0.651 ± 0.020 | 0.800 / 0.304 / 0.850 |
| **Matched fusion + half-weight composites** | **267 / 151** | **0.653 ± 0.020** | **0.802 / 0.327 / 0.831** |
| AP+lateral pooled as separate samples | 625 / 172 | 0.588 ± 0.015 | 0.760 / 0.314 / 0.690 |

The matched controls are essential: the paired cohort is harder than the full AP cohort,
so 0.646 should not be compared directly with 0.619. On the exact same 267 pairs and folds,
fusion improves macro-F1 by 0.074 over AP alone and 0.055 over lateral alone. It mainly adds
PCA information; M3 remains weak. In contrast, treating AP and lateral runs as exchangeable
pooled samples reduces performance. The views should be fused for a matched study, not
simply added as more independent-looking rows.

### Composite-label audit and sensitivity

Only two patients contain usable composites:

- accession 1010050348: three matched AP/lateral runs labeled M2+A3;
- accession 9257164: two matched AP/lateral runs labeled M3+M4 (plus one pure M4 run, which
  remains unsupported).

These are five matched pairs, not five new patients. M2+A3 cannot be called pure M2, and
M3+M4 cannot be called pure M3 under an exclusive segment endpoint. The strict result
therefore continues to exclude them. The sensitivity analysis adds them only to each
training fold as `target component present` examples—M2+A3 contributes M2 and M3+M4
contributes M3—with sample weight 0.5. They are never validation examples. Full weight gives
essentially the same result (0.653 ± 0.023), and half weight gives 0.653 ± 0.020 versus
0.646 ± 0.021 without them.

That +0.007 is directionally encouraging across the repeated splits, including M3 F1
0.317→0.327, but it comes from only two additional patient groups. It is an exploratory
sensitivity, not evidence that priority-mapping composites is generally safe. The preferred
headline remains strict paired fusion without composites; the half-weight artifact is useful
for research comparison. With more composite patients, the correct long-term solution is a
multilabel presence model or a hierarchical model, evaluated against multilabel truth.

## Trained artifacts

The canonical artifact paths are listed in `runs/anatomy_tasks/results.json`. The recommended
artifacts are:

- `runs/anatomy_tasks/models/clean_m2_m3__dino_temporal__std_logreg_c3.joblib`
- `runs/anatomy_tasks/models/m2_m3_pca_strict__dino__std_logreg_c1.joblib` is the preferred
  coherent three-class probe.
- `runs/multiview_anatomy/models/paired_strict__paired_temporal__std_logreg_c1.joblib` is the
  best strict matched AP+lateral probe by macro-F1 when both views are available. The
  uniform-frame strict control is also saved because it has slightly better M3 F1 and AUPRC.
- `runs/multiview_anatomy/models/paired__paired_uniform_weak05__std_logreg_c1.joblib` is the
  exploratory matched AP+lateral model with half-weight composite component supervision.

Additional territory, supported-four-class, MCA stress-test, and six-class probes were saved
for reproducibility. `territory_then_m2_m3_cascade.json` documents the intended hierarchy and
explicitly marks M1/M4 as unsupported. The territory and cascade artifacts are exploratory
because ACA performance is inadequate.

The complete multiview audit, per-seed metrics, view-specific metrics, composite source
labels, and trained artifact paths are in `runs/multiview_anatomy/results.json`.

## Reproduce

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python anatomy_task_experiments.py \
  --feature vjepa=runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
  --feature dino=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
  --feature dino_temporal=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_top_contrast.npz \
  --fusion vjepa_dino=vjepa+dino \
  --fusion vjepa_temporal=vjepa+dino_temporal \
  --fusion dino_dual=dino+dino_temporal \
  --fusion all_three=vjepa+dino+dino_temporal \
  --recipes std_logreg_c3,std_logreg_c1,l2_logreg \
  --seeds 0:20 --folds 5 --jobs 8 --train-final \
  --out runs/anatomy_tasks/results.json
```

The AP+lateral follow-up is reproduced with:

```bash
.venv/bin/python multiview_anatomy_experiments.py \
  --ap-uniform runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
  --ap-temporal runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_top_contrast.npz \
  --lat-uniform runs/lat_dinov2l252/cache/image_facebook-dinov2-large_252_16_Lateral_positive_subtype.npz \
  --lat-temporal runs/lat_dinov2l252/cache/image_facebook-dinov2-large_252_16_Lateral_positive_subtype_top_contrast.npz \
  --seeds 0:20 --folds 5 --out runs/multiview_anatomy/results.json
```
