# Anatomically coherent target experiments

Updated: 2026-07-13

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
- Five folds were requested. The count is capped by distinct patient groups in the rarest
  class: five folds for the first five tasks and three for the two M1-containing tasks.
  Seeds 0–19 measure split sensitivity, not 20 independent test sets.
- Mean-pooled frozen features only. Feature scaling and the classifier are fit inside each
  training fold.
- Compared normalized V-JEPA, uniform DINO, temporal-change DINO, and four direct feature
  concatenations.
- Compared the convergent documented probe recipes: standardized balanced logistic
  regression at C=1 and C=3, plus L2-normalized balanced logistic regression.
- The initial fusion-wide LinearSVC run exposed unconverged high-dimensional fits. The
  corrected probe uses the primal solver (`C=0.5`, `dual=False`, `tol=1e-4`,
  `max_iter=20,000`) and aborts on any `ConvergenceWarning`. The completed base-feature
  follow-up covers all seven tasks; a targeted fusion follow-up covers the two primary tasks.
- After cross-validation, one full-data probe per task was fit using the best mean-CV
  feature/recipe combination. Those saved probes are inference artifacts, not new held-out
  evidence.

## Results

### Original logistic sweep

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

### Hardened LinearSVC follow-up

| Task | Best base-feature LinearSVC | Macro-F1 | Mean class F1 | Δ vs prior task leader |
|---|---|---:|---|---:|
| Territory | V-JEPA | 0.580 ± 0.017 | MCA .944 / ACA .000 / PCA .797 | −.014 |
| **Clean M2/M3** | temporal DINO | **0.584 ± 0.025** | M2 .816 / M3 .352 | +.004 |
| **M2/M3/PCA** | temporal DINO | **0.645 ± 0.025** | M2 .808 / M3 .357 / PCA .769 | **+.026** |
| M2/M3/ACA/PCA | temporal DINO | 0.463 ± 0.019 | .776 / .322 / .008 / .747 | +.001 |
| M2/M3/other-MCA | temporal DINO | 0.411 ± 0.027 | .782 / .330 / .121 | +.028 |
| MCA M1/M2/M3/M4 | temporal DINO | 0.283 ± 0.031 | .000 / .777 / .308 / .049 | +.001 |
| Six-class anatomy | temporal DINO | 0.299 ± 0.015 | .000 / .737 / .283 / .019 / .049 / .704 | −.002 |

All 420 base-feature seed/configuration evaluations completed, covering 1,860 fold fits;
none emitted a convergence warning, and the largest observed iteration count was 2,753.
The delta column compares each SVM leader with the previous task-level logistic leader.
The territory comparison gives logistic regression a broader search space because its prior
leader was a feature fusion. Full SVM results are in
`runs/anatomy_tasks/results_svm_base_20seeds.json`.

The targeted high-dimensional check added V-JEPA+DINO, V-JEPA+temporal-DINO,
uniform+temporal DINO, and all-three concatenation for the two primary tasks. All 160
seed/configuration evaluations (800 fold fits) completed without a convergence warning; the
largest fit used 3,614 iterations. Uniform+temporal DINO was the best fusion, but it reached
only 0.577 ± 0.024 on clean M2/M3 and 0.630 ± 0.020 on M2/M3/PCA, below temporal DINO alone
at 0.584 and 0.645. The paired deltas are −0.0070 (7/20 wins) and −0.0149 (4/20 wins),
respectively. Fusion therefore does not explain or improve the SVM gain. Full results are
in `runs/anatomy_tasks/results_svm_primary_fusions_20seeds.json`.

## What the experiments show

### 1. Clean M2-vs-M3 remains the defensible binary target

Removing unrelated territories and ambiguous composites raises the direct M2-vs-M3
macro-F1 to 0.581. Temporal-change DINO is best and fusion is worse, consistent with the
earlier finding that DINO carries the strongest m3 signal. M3 remains difficult: mean F1 is
0.332 and recall is 0.315, so this is a research baseline rather than a clinical classifier.

With the hardened SVM on the identical temporal-DINO feature, macro-F1 is 0.584. The paired
SVM-minus-logistic delta is +0.0036 ± 0.0186 across split seeds (12 wins, 8 losses), so the
headline performance is effectively tied. SVM trades M2 F1 down by 0.013 while raising M3
F1 by 0.020.

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

Temporal-DINO + LinearSVC is the strongest new candidate on this task at
**0.645 ± 0.025 macro-F1**. Against the previous overall leader, its paired delta is
+0.0258 ± 0.0280 (16 wins, 4 losses); class-F1 changes are +0.001 for M2, +0.025 for M3,
and +0.052 for PCA. Against logistic regression on the identical temporal-DINO feature, the
delta is +0.0294 ± 0.0278 (19 wins, 1 loss). This model was selected from repeated CV and
has not replaced the saved logistic artifact; it remains a candidate pending a locked test.
Its 0.645 AP-only score must not be compared directly with the 0.646 matched AP+lateral
result below, which uses a different 267-pair cohort and different folds.

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

### 5. Replacing `other_positive` with four or six explicit labels is defensible but not useful yet

The four-class M2/M3/ACA/PCA model is dominated by M2 and PCA and almost never learns ACA.
The six-class model similarly fails M1, M4, and ACA. These targets should be retained as
data-readiness tests, not promoted as production models.

### 6. The SVM gain is representation- and task-specific

LinearSVC is not generally better than logistic regression. Its useful gain is concentrated
in DINO representations, especially temporal DINO on the PCA-inclusive task. Changing the
head still does not rescue the sparsest labels: ACA F1 falls to 0.008 in the four-class SVM,
M1 remains zero, and other-MCA reaches only 0.121. The limiting factor for ACA, M1, and M4
remains label support rather than a missing linear-probe recipe.

## Recommended model policy

1. Use `clean_m2_m3` when the scientific question is specifically M2 versus M3.
2. Use `m2_m3_pca_strict` when a coherent three-class endpoint is required; explicitly state
   that ACA, M1, M4, and composites are outside scope. Carry temporal-DINO + LinearSVC as
   the leading AP-only candidate, but retain the current logistic artifact until a locked
   patient-level comparison is available.
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

- composite patient A: three matched AP/lateral runs labeled M2+A3;
- composite patient B: two matched AP/lateral runs labeled M3+M4 (plus one pure M4 run, which
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

## Mask-free radiomics sensitivity follow-up

There was no recoverable radiomics implementation in the July 12 Claude sessions, Git
history, reflogs, or scratch files. The dataset also contains no ROI masks or DICOM SEG/NIfTI
segmentations, and pixel spacing is absent from most runs. A conventional lesion- or
territory-masked PyRadiomics analysis is therefore not possible from the available inputs.
The new `radiomics_experiments.py` implements a deliberately narrower **mask-free,
whole-field radiomics-style sensitivity baseline**; it must not be described as standard
PyRadiomics or an IBSI-compliant ROI analysis.

For each of the 366 resolved AP runs, the extractor applies the existing DICOM rescale and
photometric handling, block-averages the cine to 128×128, and normalizes that run at its 1st
and 99th percentiles. DICOM `FrameTimeVector`/`FrameTime` metadata is used to linearly
resample each cine to 32 normalized-time samples; the parser handles the common leading-zero
`T+1` vector export, with frame index as a fallback. It then derives temporal mean, temporal
standard-deviation, and temporal P90−P10 projection maps. Each map receives 16 first-order
descriptors and eight symmetric four-direction GLCM descriptors. The global family has 72
features. The 288-feature spatial family summarizes upper and lower quadrant pairs by their
mean and absolute left/right difference, making it invariant to horizontal reflection. The
central-field version removes the outer 6.25% on each edge before block averaging; a
full-field version was retained as a border/collimation sensitivity. Four single-frame M2
runs necessarily have zero temporal variation. Feature definitions, implementation hash,
dependency versions, and source-file signatures are stored and validated in each cache.

Evaluation uses the same Study_Key-grouped folds, seeds 0–19, fold-local standardization,
and convergence-failing balanced LinearSVC as the frozen-feature SVM comparison. A fixed
balanced logistic C=1 control checks whether a negative result is specific to the SVM head.

| Task / feature | Probe | Macro-F1 | Mean class F1 |
|---|---|---:|---|
| Clean M2/M3, central global+spatial radiomics | LinearSVC | **0.502 ± 0.022** | M2 .755 / M3 .248 |
| Clean M2/M3, central spatial radiomics | Logistic C=1 | 0.496 ± 0.016 | M2 .745 / M3 .246 |
| M2/M3/PCA, central spatial radiomics | LinearSVC | 0.463 ± 0.030 | M2 .721 / M3 .237 / PCA .431 |
| M2/M3/PCA, central spatial radiomics | Logistic C=1 | **0.484 ± 0.024** | M2 .712 / M3 .241 / PCA .499 |
| Clean M2/M3, temporal DINO control | LinearSVC | 0.584 ± 0.025 | M2 .816 / M3 .352 |
| M2/M3/PCA, temporal DINO control | LinearSVC | 0.645 ± 0.025 | M2 .808 / M3 .357 / PCA .769 |

The handcrafted baseline is above a majority-only classifier but materially below temporal
DINO on both primary tasks. The central crop is consistently better than the full field:
central-minus-full is +0.0521 ± 0.0333 for the global binary SVM (20/20 split-seed wins) and
+0.0411 ± 0.0351 for the spatial three-class SVM (18/20 wins). This is useful evidence that
whole-field radiomics is sensitive to borders, skull, and collimation rather than evidence
for a true anatomical ROI. Logistic regression improves the spatial three-class result over
LinearSVC by +0.0207 ± 0.0237 (18/20 wins), mostly through PCA, while it is slightly worse
on the binary task. The earlier SVM advantage does not generalize uniformly to handcrafted
features.

As an exploratory complement test, concatenating only the central spatial radiomics vector
to temporal DINO gives 0.570 ± 0.020 on clean M2/M3 versus 0.584 for DINO alone (paired delta
−0.0147 ± 0.0310; 4/20 wins). On M2/M3/PCA it gives 0.655 ± 0.019 versus 0.645 (paired delta
+0.0107 ± 0.0273; 15/20 wins). The class-F1 changes are −.005 M2, −.002 M3, and +.039 PCA:
the small aggregate gain is entirely a PCA effect and does not solve M3. The spatial-only
fusion was slightly better and smaller than concatenating both radiomics families. Because
these fusion variants were compared on the same reused CV groups, 0.655 is exploratory and
does not replace the temporal-DINO candidate without a locked patient-level test.

Across the remaining tasks, the best radiomics results are 0.462 for territory, 0.356 for
M2/M3/ACA/PCA, 0.339 for M2/M3/other-MCA, 0.248 for M1/M2/M3/M4, and 0.225 for six-class
anatomy. ACA, M1, and M4 remain near zero. Handcrafted texture therefore provides a useful
negative/control representation and a possible PCA complement, not a new primary model.
No full-data radiomics artifact was trained.

## Matched AP + lateral radiomics and ensemble follow-up

The lateral follow-up applies the same timing-aware central-field extractor to all 328
resolved lateral runs, producing a 288-feature spatial cache with the same feature names,
normalization, crop, and implementation hash as the AP cache. Evaluation is restricted to
the 267 run-index- and accession-matched AP/lateral pairs with concordant strict truth:
187 M2, 58 M3, and 22 PCA runs from 151 patient groups. The seven AP-M2/lateral-M3 label
disagreements remain excluded.

All methods use the same 20 repeated five-fold `Study_Key`-grouped outer splits. Any PCA
gate threshold or radiomics blend weight is selected from three-fold grouped cross-fitted
predictions inside the relevant outer training fold. No outer validation prediction is used
for tuning. The hierarchy first decides PCA versus MCA, then applies a dedicated M2/M3
expert to cases routed to MCA. The radiomics-inclusive variants use both AP and lateral
central spatial descriptors.

| Matched-cohort method | Macro-F1 | Mean class F1 (M2 / M3 / PCA) |
|---|---:|---:|
| **Paired-DINO PCA gate + AP-DINO M2/M3 expert** | **0.654 ± 0.025** | .788 / .329 / .844 |
| Direct paired AP+lateral temporal-DINO logistic | 0.651 ± 0.020 | **.800** / .304 / **.850** |
| DINO+radiomics blended PCA gate + AP-DINO expert | 0.647 ± 0.021 | .785 / **.330** / .827 |
| Late paired-DINO/radiomics probability blend | 0.638 ± 0.025 | .798 / .275 / .842 |
| Early paired-DINO+radiomics logistic fusion | 0.626 ± 0.018 | .784 / .245 / **.850** |
| Matched AP temporal-DINO LinearSVC | 0.615 ± 0.023 | .790 / .333 / .722 |
| Matched lateral temporal-DINO LinearSVC | 0.572 ± 0.022 | .746 / .213 / .757 |

The direct paired-DINO row exactly reproduces the earlier paired temporal result seed by
seed. The DINO-only hierarchy is the numerical winner, but its paired macro-F1 change over
that direct model is only +0.0023 ± 0.0299 with 12/20 split-seed wins. It trades mean class
F1 by −.012 M2, +.024 M3, and −.006 PCA. This is useful as a candidate when M3 sensitivity
matters, but it is not a stable enough gain to replace the simpler direct paired model
without a locked patient-level test.

Radiomics does not improve the matched-view result. Against the otherwise identical
DINO-only hierarchy, adding radiomics to the gate changes macro-F1 by
−0.0063 ± 0.0137 (4/20 wins). Relative to direct paired DINO, the late blend changes it by
−0.0132 ± 0.0221 (5/20), and early concatenation by −0.0251 ± 0.0179 (2/20). Inner tuning
selected nonzero radiomics weight in 85 of 100 blended-gate outer folds, but that apparent
inner-fold benefit did not generalize to the held-out outer patients. With only 11 PCA
patient groups, this is evidence of selection instability rather than a radiomics gain.
The selected DINO-only gate threshold also ranges from 0.3 to 0.7 across the outer folds;
there is no stable deployment threshold in these data.

The practical recommendation is therefore unchanged: when both views are available, keep
the direct paired temporal-DINO logistic model as the default because it is simpler and more
stable. Carry the DINO-only hierarchy as a locked-test challenger if recovering M3 is the
priority. Do not promote an AP+lateral radiomics ensemble from these data. The complete OOF
scores, nested selections, cache hashes, cohort audit, and convergence audit are stored in
`runs/multiview_ensembles/results_nested_20seeds.json`. No new full-data artifact was fit.

## Paired + unpaired missing-view hierarchy

The deployment-oriented follow-up tests whether the hierarchy can use every strict case
without inventing AP/lateral pairs. The union is formed before any split and contains 344
unique study/accession/run identities from 171 patient groups: 267 concordant pairs, 57
AP-only cases, and 20 lateral-only cases. Class counts are 241 M2, 73 M3, and 30 PCA. The
seven discordant strict pairs, 30 paired non-strict identities, and nine non-strict
single-view identities remain excluded.

Every outer and inner split is grouped over the complete patient union. AP experts are
trained on all training-fold cases with AP; lateral experts are trained on all cases with a
lateral view. A paired case blends the two PCA-vs-MCA gates and then uses the AP M2/M3
expert. AP-only and lateral-only cases use their respective gate and M2/M3 expert. Thus each
run identity is scored once, and another run or view from the same patient cannot cross a
fold boundary. The pair weight and gate threshold are selected only from grouped inner OOF
predictions.

| Method / evaluation scope | Cases | Macro-F1 | Mean class F1 (M2 / M3 / PCA) |
|---|---:|---:|---:|
| **All-available hierarchy / full union** | **344** | **0.651 ± 0.027** | .790 / **.347** / .815 |
| Paired-only concatenated hierarchy / paired | 267 | 0.649 ± 0.026 | .788 / .338 / **.821** |
| All-available hierarchy / paired | 267 | 0.648 ± 0.030 | **.796** / **.344** / .803 |
| Paired-only viewwise hierarchy / paired | 267 | 0.647 ± 0.034 | .788 / .340 / .814 |
| Direct missing-view logistic / full union | 344 | 0.612 ± 0.023 | **.792** / .264 / .782 |
| All-available hierarchy / AP-only | 57 | 0.625 ± 0.037 | .785 / .262 / .828 |
| All-available hierarchy / lateral-only | 20 | 0.733 ± 0.060 | .709 / .489 / 1.000 |

The central result is coverage, not a paired-accuracy gain. On the paired subset, adding the
unpaired cases to the otherwise matched viewwise hierarchy changes macro-F1 by only
+0.0003 ± 0.0277 (12/20 split-seed wins). Against the stronger paired concatenated
hierarchy it is −0.0017 ± 0.0281 (9/20 wins). Extra unpaired training raises paired M2 and
M3 F1 by .007 and .006 versus the concatenated control, but lowers PCA by .018. With only
15 additional unpaired M3 and eight additional unpaired PCA cases, there is no evidence
that the extra data improve paired performance.

For a single model that must accept missing views, however, the hierarchy is clearly better
than independent three-class logistic experts: +0.0383 ± 0.0362 union macro-F1 with 17/20
wins, driven by +.084 M3 F1 and +.033 PCA F1. Union-level M3 recall is .349 and PCA recall
is .823. The lateral-only number is descriptive rather than reliable because that subset
contains only 20 cases and one PCA case.

The union-tuned paired gate assigns lateral weight 0.25/0.50/0.75/1.00 in
1/29/50/20 of 100 outer folds, supporting lateral primarily as a PCA-gating view. Thresholds
still range from 0.3 to 0.7, so a final operating point requires locked validation. Use the
all-available hierarchy when missing-view coverage is required; keep the simpler paired
model for paired-only deployment. Complete OOF predictions and selections are stored in
`runs/missing_view_hierarchy/results_nested_20seeds.json`. No full-data artifact was fit.

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

The SVM follow-up intentionally did not train replacement full-data artifacts. Its purpose
was comparative repeated CV; selecting and fitting a new final probe before locked testing
would turn the same reused groups into both the selection and headline evidence.

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

The hardened base-feature SVM follow-up is reproduced with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python anatomy_task_experiments.py \
  --feature vjepa=runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
  --feature dino=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
  --feature dino_temporal=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_top_contrast.npz \
  --recipes std_linsvm --seeds 0:20 --folds 5 --jobs 8 \
  --out runs/anatomy_tasks/results_svm_base_20seeds.json
```

For the targeted fusion check, add the four `--fusion` arguments from the logistic command,
then add `--tasks clean_m2_m3,m2_m3_pca_strict` and
`--feature-sources vjepa_dino,vjepa_temporal,dino_dual,all_three`; write to
`runs/anatomy_tasks/results_svm_primary_fusions_20seeds.json`.

The handcrafted radiomics-style caches and fixed-probe controls are reproduced with:

```bash
.venv/bin/python radiomics_experiments.py --view AP --jobs 2 \
  --border-fraction 0 --out-dir runs/radiomics/cache_full
.venv/bin/python radiomics_experiments.py --view AP --jobs 2 \
  --out-dir runs/radiomics/cache_center875
.venv/bin/python radiomics_experiments.py --view Lateral --jobs 2 \
  --out-dir runs/radiomics/cache_lateral_center875

.venv/bin/python anatomy_task_experiments.py \
  --feature radiomics_global_center875=runs/radiomics/cache_center875/wholefield_radiomics_ap_global.npz \
  --feature radiomics_spatial_center875=runs/radiomics/cache_center875/wholefield_radiomics_ap_spatial.npz \
  --fusion radiomics_all_center875=radiomics_global_center875+radiomics_spatial_center875 \
  --recipes std_linsvm --seeds 0:20 --folds 5 --jobs 8 \
  --out runs/radiomics/results_all_tasks_linsvm_20seeds.json
```

Change the final recipe to `std_logreg_c1` and output name to reproduce the logistic control.
The full-vs-central and temporal-DINO fusion commands, including every source cache hash, are
recorded in `results_primary_linsvm_20seeds.json` and
`results_dino_radiomics_ablation_linsvm_20seeds.json`.

The leakage-safe matched-view hierarchy and radiomics ensemble comparison is reproduced
with:

```bash
.venv/bin/python multiview_ensemble_experiments.py \
  --ap-temporal runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_top_contrast.npz \
  --lat-temporal runs/lat_dinov2l252/cache/image_facebook-dinov2-large_252_16_Lateral_positive_subtype_top_contrast.npz \
  --ap-radiomics runs/radiomics/cache_center875/wholefield_radiomics_ap_spatial.npz \
  --lat-radiomics runs/radiomics/cache_lateral_center875/wholefield_radiomics_lateral_spatial.npz \
  --seeds 0:20 --folds 5 --inner-folds 3 --jobs 8 \
  --out runs/multiview_ensembles/results_nested_20seeds.json
```

The paired+unpaired missing-view experiment is reproduced with:

```bash
.venv/bin/python missing_view_hierarchy_experiments.py \
  --ap-temporal runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_top_contrast.npz \
  --lat-temporal runs/lat_dinov2l252/cache/image_facebook-dinov2-large_252_16_Lateral_positive_subtype_top_contrast.npz \
  --seeds 0:20 --folds 5 --inner-folds 3 --jobs 8 \
  --out runs/missing_view_hierarchy/results_nested_20seeds.json
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
