# Experiments: improving per-class DSA subtype performance

The original positive-subtype backbone results are **patient-grouped 5-fold
cross-validation** (`StratifiedGroupKFold` on `Study_Key`) and out-of-fold. The anatomy
tasks use the same grouping, with the fold count capped by rare-class patient support; the
two M1-containing tasks therefore use three folds. The AP comparison set has 366 runs from
184 groups (m2=240, m3=67, other_positive=59). The initial experiments used one split seed;
the new backbone/resolution comparison repeats the same fixed recipe over 20 paired split
seeds. Tooling: [experiments.py](../experiments.py) (V-JEPA 2 recipe sweep),
[image_backbone_probe.py](../image_backbone_probe.py) (per-frame image backbones),
[compare_feature_caches.py](../compare_feature_caches.py) (repeated paired comparison),
[three_class_augmentation_experiments.py](../three_class_augmentation_experiments.py)
(direct three-class augmentation/fusion),
[anatomy_task_experiments.py](../anatomy_task_experiments.py)
(anatomically coherent target redesign),
[attn_probe.py](../attn_probe.py) (attentive pooling), and
[cross_validate.py](../cross_validate.py) (baseline probe).

## Headline (updated 2026-07-13)

The old best was unnormalized V-JEPA 2 ViT-g/384 at 0.476. Applying the checkpoint's
documented channel normalization raises its best seed-0 sweep result to **0.486**. A
genuinely different model, DINOv2-L/252, is effectively tied and is stronger on m3.

| Frozen AP backbone | Resolution | Seed-0 best-of-sweep | Fixed LinearSVC, 20 seeds | Fixed logreg C3, 20 seeds |
|---|---:|---:|---:|---:|
| V-JEPA 2 ViT-L (normalized) | 256 | 0.479 | 0.438 ± 0.019 | 0.442 ± 0.023 |
| V-JEPA 2 ViT-g (normalized) | 256 | 0.480 | 0.446 ± 0.025 | 0.450 ± 0.024 |
| **V-JEPA 2 ViT-g (normalized)** | **384** | **0.486** | 0.460 ± 0.021 | **0.474 ± 0.019** |
| **DINOv2-L, per-frame** | **252** | 0.477 | **0.477 ± 0.025** | 0.473 ± 0.025 |
| DINOv2-L, per-frame | 518 | 0.443 | 0.438 ± 0.025 | 0.456 ± 0.028 |
| RAD-DINO, per-frame | 518 | 0.478 | 0.457 ± 0.019 | 0.462 ± 0.020 |
| RadImageNet ResNet50, per-frame | 224 | 0.407 | 0.375 ± 0.018 | 0.377 ± 0.019 |

There is no single dominant configuration. V-JEPA 2 ViT-g/384 has the best seed-0
and repeated-logistic score, while DINOv2-L/252 has the best repeated LinearSVC score
and the best average m3 F1 (0.303). The two are tied under repeated logistic regression
(0.474 vs 0.473), so a 0.001 headline difference is not meaningful.

The subsequent direct three-class augmentation study reaches **0.491 ± 0.016** with
ordinary V-JEPA+DINO early fusion and **0.498 ± 0.030** with training-fold-selected
class routing. A post-hoc fixed route scores 0.518 ± 0.024, but it was designed after
examining these CV results and is not a valid replacement for a locked test result.

## What we tried, and what it told us

**1. Recipe sweep** (`experiments.py`) — representation (mean vs mean+max+std) ×
preprocessing (raw/standardize/L2) × classifier (logreg C-sweep / RBF-SVM / linear-SVM / MLP).
- Standardization helped every classifier (+0.03–0.04). Best: AP `mean/std/linsvm` 0.46,
  Lateral `meanmaxstd/std/logreg` 0.45.
- `mean/L2-norm/balanced-logreg` gave the **most even per-class F1** and the highest
  *balanced* accuracy — it trades m2 down to lift m3/other. A valid operating point if you
  care about the minority classes more than headline accuracy.
- Richer pooling (mean+max+std) and RBF-SVM did **not** reliably help; RBF sometimes
  collapsed m3 to zero.

**2. Multi-view fusion** — concatenate a study's mean AP + mean Lateral features (88% of
studies have a consistent label across both views). **No help**: fused 0.43 sat *between*
AP-only 0.40 and Lateral-only 0.45. Naive feature averaging dilutes rather than
disambiguates.

**3. Attentive-pooling probe** (`attn_probe.py`) — keep all 2048 tokens, learn attention
pooling instead of mean. **No help** (AP 0.41). This is the important negative: if a learned
pooling over the full token sequence can't separate m3, the m2/m3 distinction is **not
present in the frozen features** — so a better *head* won't fix it.

**4. Partial fine-tuning** ([finetune.py](../finetune.py)) — unfreeze the last 2 of 24 encoder
blocks + final norm + head, discriminative LRs (backbone 1e-5 / head 1e-3), head-only warmup,
gradient clipping, class-weighted loss, 5-fold. **No reliable gain.** The no-peek (final-epoch)
OOF macro-F1 was **0.44 — below the frozen 0.46**; the optimistic best-epoch peek (0.48) is
within fold-to-fold noise (folds 0.40–0.53). The training curve is textbook overfitting:
train accuracy climbs 0.43→0.72 while held-out val macro-F1 stays flat (~0.38) and never clears
the frozen line. m3 stayed ~0.23. ~290 samples is simply too little to move a 300M-param ViT-L.

**5. More frames** — re-extracted at 32 frames/clip (your DSA runs are ~20 frames median,
max 34, so 32 uses every real frame; 64 would only interpolate). AP best macro-F1 **0.44 —
slightly below the 16-frame 0.46**, m3 unchanged (0.20). 16 frames already captured the
temporal signal; upsampling ~20-frame runs to 32 just adds redundancy. `experiments.py
--clip-length N` supports this if revisited with more data.

**6. Higher spatial resolution** ([experiments.py](../experiments.py) `--backbone
facebook/vjepa2-vitg-fpc64-384 --image-size 384`) — your DICOMs are 1024², so 256px throws
away detail. The original run lifted the best seed-0 sweep from 0.459 to 0.476, but it
changed ViT-L→ViT-g, 256→384, and the winning classifier recipe at the same time. The
baseline-winning fixed LinearSVC recipe actually fell from 0.459 to 0.439. This historical
result motivated the controlled runs below; by itself it did **not** establish a resolution
gain.

**7. Medical backbone: RadImageNet** ([radimagenet_probe.py](../radimagenet_probe.py)) — the
first medical comparison used a RadImageNet ResNet50 per frame with temporal mean pooling.
Its historical **0.407** was a seed-0 best-of-sweep result. Under the same fixed logistic-C3
recipe and identical 20 paired grouped splits, RadImageNet scores **0.377 ± 0.019**
(class F1 0.662/0.203/0.266), while RAD-DINO scores **0.462 ± 0.020**
(0.705/0.269/0.413). The paired improvement is +0.0850 ± 0.0319, median +0.0896,
range +0.0306 to +0.1357; RAD-DINO wins all 20 split seeds. The fixed LinearSVC comparison
agrees: 0.375 ± 0.018 versus 0.457 ± 0.019, again 20/20 wins. This rules out that specific
static CNN, not medical pretraining as a category. No usable public angiography-video
checkpoint was available for these runs.

The reused historical RadImageNet cache predates extraction signatures, so its original
checkpoint and extraction provenance cannot be reconstructed. It nevertheless passed exact
validation for all 366 rows: finite 2,048-value features, labels, groups, and ordered
`(Study_Key, accession, view, run_column)` identity. Legacy reuse now requires an explicit
flag. New extractions pin revision `14460ee4c1276f6925611a63aa9a54a05d39eae0` and record
the checkpoint SHA-256.

The structured legacy audit, including out-of-fold predictions and the cache hash, is in
`runs/radimagenet_validation/results.json`.

**8. Label-scheme diagnostics** — same features, different targets:

| Target | AP macro-F1 | Note |
|---|---|---|
| **Side L vs R** | **0.85** | sanity check — features are excellent for global layout |
| Territory MCA/ACA/PCA | 0.57 | MCA F1 **0.94**; ACA/PCA data-starved |
| Binary m2 vs m3 | 0.53 | m3 F1 0.28 even in isolation |
| MCA segment M1–M4 | 0.23 | fine segmentation collapses |
| Current 3-class | 0.46 | |

On Lateral, side drops to 0.57 — exactly as expected, because left/right vessels
superimpose in the lateral projection. This is a strong sign the features (and the whole
pipeline) are working correctly.

**9. Correct V-JEPA preprocessing + deconfound scale and resolution** — the original
pipeline percentile-windowed each DICOM to [0,1] but omitted the ImageNet channel
normalization declared by every V-JEPA 2 checkpoint. `experiments.py --normalize-input`
now applies it, and normalized caches are separate from the historical ones. It improves
the best seed-0 ViT-L/256 sweep from 0.459 to **0.479**.

The official ViT-g/256 checkpoint supplies the missing control. With the same frozen
ViT-g family, 384 beats 256 by **+0.014** mean macro-F1 across 20 paired seeds under
LinearSVC (13/20 seeds) and **+0.024** under logistic C3 (15/20). The benefit again shifts
toward other_positive: under repeated LinearSVC, class F1 changes from
0.690/0.271/0.377 at 256 to 0.697/0.242/0.441 at 384. The larger model alone is not the
answer: ViT-g/256 is only +0.008 over ViT-L/256 under either fixed recipe.

**10. Different model + within-model resolution control: DINOv2-L**
([image_backbone_probe.py](../image_backbone_probe.py)) — a frozen 300M-parameter DINOv2-L
is applied to the same 16 sampled frames, using native normalization, then temporally
pooled. The same checkpoint was run at 252px (18×18 patches) and 518px (37×37 patches),
so this is the cleanest resolution test in the study.

- At 252px, DINOv2-L scores **0.477 ± 0.025** over 20 seeds with fixed LinearSVC and
  **0.473 ± 0.025** with logistic C3. Its repeated LinearSVC class F1 is
  **0.723/0.303/0.403**, the strongest m3 result in the backbone comparison.
- At 518px, it falls to **0.438 ± 0.025** (LinearSVC) and **0.456 ± 0.028** (logistic C3).
  Against 252px it loses 0.039 and 0.017 respectively. Higher resolution is therefore
  **not** a general solution; for this small dataset it can make global image features
  noisier or more variable without adding learnable signal.

**11. Stronger medical control: RAD-DINO/518** — Microsoft's radiograph-pretrained DINO
reaches **0.478** on the seed-0 sweep, nearly tying the headline leader. Across 20 seeds it
scores 0.457 ± 0.019 (LinearSVC) and 0.462 ± 0.020 (logistic C3), with repeated LinearSVC
class F1 0.701/0.264/0.406. It is a major improvement over RadImageNet ResNet50 (0.407),
but it does not consistently beat either DINOv2-L/252 or V-JEPA 2 ViT-g/384. The accurate
conclusion is that **the older medical CNN was a dead end; medical pretraining itself was
not**.

**12. Repeated paired folds** ([compare_feature_caches.py](../compare_feature_caches.py)) —
all seven feature sets were evaluated on the same 20 `StratifiedGroupKFold`
seeds with one fixed representation (mean), fold-local standardization, and the same
classifier. This removes the most obvious best-of-13 recipe selection bias.

- LinearSVC favors DINOv2-L/252 over V-JEPA 2 ViT-g/384 by +0.017 on average (15/20 seeds).
- Logistic C3 makes them indistinguishable: 0.473 vs 0.474 (DINO wins 10/20 seeds).
- All six alternatives beat RadImageNet on every logistic-C3 split; five did so on every
  LinearSVC split and V-JEPA ViT-g/256 did so on 19/20.
- Repeated folds measure split sensitivity, not independent-dataset generalization; the
  final model still needs a locked patient-level test cohort or external validation.
- The current subset has one accession per MRN, so `Study_Key` is effectively patient-level;
  future repeat admissions must be grouped by a stable patient identifier rather than accession.

Full paired outputs are in
`runs/radiology_svm_comparison/repeated_linsvm_7backbones_20seeds.json` and
`runs/radiology_svm_comparison/repeated_logregc3_7backbones_20seeds.json`.

**13. Direct m2 vs m3 vs other augmentation and fusion**
([three_class_augmentation_experiments.py](../three_class_augmentation_experiments.py)) —
the exploratory MCA-vs-other gate was discarded because `other_positive` is not non-MCA.
It contains 15 MCA runs (M1=5, M4=10), 15 ACA runs, and 29 PCA runs. Every result below
is one direct run-level three-class prediction over m2/m3/other_positive. There is no
binary gate, no study-level label aggregation, and no reinterpretation of `other`.

The fixed protocol is mean-pooled features, fold-local standardization, balanced logistic
regression C=3, and the same patient-grouped 5-fold splits for seeds 0–19. Flip copies stay
inside the training patient's fold and each duplicated view receives weight 0.5.

| Direct three-class configuration | Macro-F1, 20 seeds | m2 / m3 / other F1 | Paired interpretation |
|---|---:|---:|---|
| V-JEPA ViT-g/384 | 0.474 ± 0.019 | .708 / .209 / .505 | video baseline |
| DINOv2-L/252, uniform frames | 0.473 ± 0.024 | .724 / .301 / .393 | image baseline |
| **V-JEPA + DINO early feature fusion** | **0.491 ± 0.016** | .729 / .234 / .510 | stable pre-augmentation leader |
| V-JEPA, original+flip train, embedding-average test | 0.484 ± 0.020 | .709 / .237 / .506 | +.010 vs V-JEPA; wins 15/20 |
| Best flip late fusion | 0.491 ± 0.018 | .737 / .244 / .493 | effectively tied with early fusion |
| **DINO top-temporal-change frames** | **0.486 ± 0.023** | .733 / **.306** / .421 | +.014 vs uniform DINO; wins 13/20 |
| Early-fusion + temporal-change equal late ensemble | 0.494 ± 0.023 | .740 / .249 / .491 | only +.003 vs early fusion; loses 12/20 seeds |
| **Nested class routing** (choice made on outer-training data only) | **0.498 ± 0.030** | .739 / .258 / .496 | +.007 vs early fusion; wins 14/20, high variance |
| Post-hoc fixed route: temporal for m2/m3, fusion for other | 0.518 ± 0.024 | .748 / .312 / .493 | exploratory ceiling only; route was chosen after CV inspection |

The fixed route is still a direct three-class argmax: it takes the m2 and m3 probability
columns from a direct temporal-DINO model and the other column from a direct fusion model.
It is not the rejected MCA gate. However, because that column assignment was selected after
looking at these results, **0.518 is hypothesis-generating, not a deployable estimate**. In
the leak-free nested version, source selection is repeated inside each outer training fold and
the score contracts to 0.498.

Input variants that did not work under the same fixed direct probe:

| DINO input | Macro-F1, 20 seeds | Verdict |
|---|---:|---|
| Full + centered 90% + centered 80% mean-pooled crops | 0.456 ± 0.021 | worse; tight crops can lose distal cortex |
| Centered 90% crop | 0.442 ± 0.022 | worse |
| Early/peak/peak+2s phase RGB | 0.449 ± 0.015 | worse |
| Full-run temporal-statistics RGB | 0.413 ± 0.018 | much worse |

The temporal-change selector keeps frame 0 and the 15 largest adjacent-frame changes when
a run has more than 16 frames; shorter runs use the same repeated uniform sampling as the
baseline. A representative montage
([temporal_selection_audit.png](../runs/three_class_augmentations/temporal_selection_audit.png))
shows that it mainly removes blank mask frames and emphasizes arterial-to-late filling. It
does not visibly collapse to motion in those examples, but it remains late-phase-heavy and
should be reviewed on a larger sample before adoption.

Most importantly, `other_positive` F1 hides a clinically important failure. Mean recall
within true-other subgroups is:

| Configuration | MCA-other (M1/M4), n=15 | ACA, n=15 | PCA, n=29 |
|---|---:|---:|---:|
| Early feature fusion | **0.060** | 0.263 | **0.809** |
| Nested class routing | 0.050 | **0.277** | 0.766 |
| Post-hoc fixed route | 0.033 | 0.247 | 0.753 |

So the apparent `other` performance is overwhelmingly ACA/PCA performance; M1/M4 recall is
near zero. This validates the ontology concern and argues for explicit clinically coherent
labels rather than treating `other_positive` as a surrogate non-MCA class.

**14. Replace `other_positive` with anatomically coherent targets**
([full report](anatomy_target_experiments.md)) — seven strict-label tasks were trained and
evaluated across 20 patient-grouped split seeds using the best documented frozen features
and convergent probe recipes.

| Redesigned task | Counts | Selected probe | Macro-F1 | Critical class result |
|---|---|---|---:|---|
| **Clean M2 vs M3** | 237 / 65 | temporal DINO / SVM | **0.584 ± 0.025** | M3 F1 **0.352** |
| **M2/M3/PCA** | 237 / 65 / 29 | temporal DINO / SVM | **0.645 ± 0.025** | M2 0.808, M3 0.357, PCA 0.769 |
| Territory MCA/ACA/PCA | 319 / 15 / 29 | V-JEPA+DINO / logreg | 0.594 ± 0.019 | ACA F1 **0.050** |
| M2/M3/ACA/PCA | 237 / 65 / 15 / 29 | temporal DINO / logreg | 0.463 ± 0.016 | ACA F1 **0.073** |
| M2/M3/other-MCA | 237 / 65 / 15 | temporal DINO / SVM | 0.411 ± 0.027 | M1/M4-group F1 **0.121** |
| MCA M1/M2/M3/M4 | 5 / 237 / 65 / 10 | temporal DINO / logreg | 0.282 ± 0.031 | M1 0.027, M4 0.030 |
| M1/M2/M3/M4/ACA/PCA | 5 / 237 / 65 / 10 / 15 / 29 | V-JEPA / logreg | 0.301 ± 0.027 | M4 0.000, ACA 0.021 |

This resolves the target-design question. **Clean M2-vs-M3 is the defensible binary task,
and strict M2/M3/PCA is the best coherent three-class task tested.** Adding PCA leaves m3
F1 in the 0.33–0.36 range and yields PCA F1 as high as 0.769. The higher three-class
macro-F1 is not a like-for-like improvement over binary because PCA is easier, but it shows
that PCA is a learnable replacement for the catch-all. A territory-first hierarchy remains
non-operational with 15 ACA cases; M1/M4 also cannot form a reliable explicit or abstention
class.

The hardened primal LinearSVC completed every base-feature fold without a convergence
warning. On clean M2/M3 it is effectively tied with logistic regression (paired delta
+0.0036 ± 0.0186, 12/20 wins). On strict M2/M3/PCA, temporal DINO + SVM improves over the
previous task leader by +0.0258 ± 0.0280 (16/20 wins), with class-F1 changes of +0.001 M2,
+0.025 M3, and +0.052 PCA. This is a DINO/task-specific gain, not a universal SVM result;
ACA remains near zero and M1/M4 remain unsupported.

A targeted SVM fusion run also completed all 800 high-dimensional fold fits without a
warning. The best uniform+temporal-DINO fusion scores 0.577 on clean M2/M3 and 0.630 on
M2/M3/PCA, below temporal DINO alone at 0.584 and 0.645. Feature concatenation is not the
source of the gain (paired deltas −0.0070 and −0.0149; 7/20 and 4/20 wins).

Full-data probe artifacts were saved after CV for reproducibility. They are not additional
held-out evidence. The recommended research artifacts are
`runs/anatomy_tasks/models/clean_m2_m3__dino_temporal__std_logreg_c3.joblib` for binary and
`runs/anatomy_tasks/models/m2_m3_pca_strict__dino__std_logreg_c1.joblib` for three-class.
Territory and cascade artifacts remain diagnostic only.

**15. Lateral helps as a matched second view; the two composite patients are only an
exploratory training sensitivity** ([full report](anatomy_target_experiments.md)) — the
strict M2/M3/PCA follow-up extracted the same frozen DINOv2-L/252 features from lateral
runs and evaluated 20 repeated patient-grouped five-fold splits.

There are 267 matched AP/lateral pairs with concordant strict truth (187 M2, 58 M3,
22 PCA). Seven additional pairs disagree—AP says M2 while lateral says M3—and were excluded
from fusion rather than assigned an arbitrary truth. On the exact same 267-pair cohort,
AP-only reaches 0.572 ± 0.031, lateral-only 0.591 ± 0.018, and concatenated AP+lateral
features reach **0.646 ± 0.021 macro-F1**. Class F1 is 0.800/0.317/0.823. Thus lateral is
useful complementary information, especially for PCA. Pooling the two views as if they were
interchangeable samples is worse (0.588 ± 0.015); they should be fused by matched run.

The original-label audit found only two composite patients: three M2+A3 matched pairs and
two M3+M4 matched pairs. These cannot be called pure M2 or pure M3. Adding them only to
training as half-weight `target component present` cases, while keeping validation strictly
pure, gives **0.653 ± 0.020** and lifts M3 F1 from 0.317 to 0.327. Full weight is essentially
identical at 0.653 ± 0.023. The +0.007 is directionally useful but comes from only two new
patient groups, so it is not a new headline result or evidence that composite priority
mapping is generally safe. Strict paired fusion remains the defensible primary result;
multilabel supervision is the right destination once more composite patients exist.

**16. Mask-free handcrafted radiomics is a useful control, not a replacement for DINO**
([full report](anatomy_target_experiments.md)) — no radiomics script was recoverable from
Claude's July 12 sessions or Git recovery points, and the source data contain no ROI masks.
The new extractor therefore makes a limited whole-field, non-IBSI sensitivity baseline from
temporal mean, standard-deviation, and P90−P10 maps with fixed first-order and GLCM features.
It corrects the common leading-zero `T+1` DICOM timing export and resamples every cine to 32
normalized-time samples before projection. It explicitly does not claim to be conventional
masked PyRadiomics.

On the same 20 patient-grouped split seeds, the best handcrafted result is 0.502 ± 0.022
macro-F1 for clean M2/M3 and 0.484 ± 0.024 for strict M2/M3/PCA, versus temporal-DINO SVM
at 0.584 and 0.645. Central-field descriptors outperform full-field descriptors, confirming
sensitivity to skull/border/collimation content. LinearSVC is not universally better:
logistic C=1 improves three-class spatial radiomics by +0.0207 ± 0.0237 (18/20 split-seed
wins), mainly through PCA.

Central spatial radiomics concatenated to temporal DINO is neutral/worse for clean M2/M3
(0.570; paired delta −0.0147, 4/20 wins) and reaches 0.655 ± 0.019 for M2/M3/PCA (paired
delta +0.0107 ± 0.0273, 15/20 wins). The added class-F1 is almost entirely PCA (+.039), not
M3. Since the fusion variants reuse the same CV groups for comparison, this is an
exploratory complement signal rather than a new selected model. ACA, M1, and M4 remain
unlearnable with these descriptors.

**17. Adding lateral radiomics and nested ensembles does not beat paired temporal DINO in
a stable way** ([full report](anatomy_target_experiments.md)) — the same timing-aware
central spatial extractor was run on 328 lateral runs, then AP and lateral radiomics were
aligned to the strict 267-pair/151-patient cohort. Every threshold and blend weight was
selected with grouped inner cross-fitting inside each of the 20×5 outer training folds.

The numerical winner is a DINO-only hierarchy: a paired AP+lateral PCA-vs-MCA gate followed
by an AP-DINO M2/M3 expert. It reaches 0.654 ± 0.025 macro-F1 with class F1
0.788/0.329/0.844. Direct paired temporal-DINO logistic regression is 0.651 ± 0.020 with
0.800/0.304/0.850. The hierarchy's paired change is only +0.0023 ± 0.0299 and it wins 12/20
split seeds, so this is a class tradeoff—not a reliable general improvement. It raises M3
F1 by .024 while lowering M2 by .012 and PCA by .006.

Radiomics-inclusive alternatives are all lower: 0.647 for the nested DINO+radiomics PCA
gate, 0.638 for a nested late probability blend, and 0.626 for early concatenation. Adding
radiomics to the otherwise identical DINO hierarchy changes macro-F1 by −.006 (4/20 wins);
the late and early variants trail direct paired DINO by −.013 and −.025. Although inner
tuning chose a nonzero radiomics gate weight in 85/100 outer folds, this did not generalize
to the held-out outer patients. The rare PCA stratum has only 11 patient groups, making the
extra selection degree of freedom prone to overfit. Gate thresholds also range from 0.3 to
0.7, so no single deployment threshold is supported.

**18. Unpaired studies extend coverage but do not improve paired accuracy**
([full report](anatomy_target_experiments.md)) — a missing-view hierarchy was evaluated on
344 strict run identities from 171 patient groups: 267 paired, 57 AP-only, and 20
lateral-only. Union folds are created before the availability subsets; each view expert is
trained only on available training-fold views, and pair weights/thresholds are selected by
grouped inner OOF prediction.

The all-available hierarchy reaches 0.651 ± 0.027 macro-F1 over the complete union, with
M2/M3/PCA F1 0.790/0.347/0.815. It beats the direct missing-view logistic ensemble by
+0.0383 ± 0.0362 with 17/20 split-seed wins, mostly through M3 (+.084 F1) and PCA
(+.033). This is a useful deployable pattern: paired cases blend AP/lateral PCA gates and
use the AP M2/M3 expert, while single-view cases fall back to their view-specific hierarchy.

Unpaired training does not improve the held-out paired subset. Against the same viewwise
architecture trained only on pairs, its change is +0.0003 ± 0.0277 (12/20 wins); against
the stronger paired concatenated hierarchy it is −0.0017 ± 0.0281 (9/20). The value of the
77 extra cases is therefore coverage, not a higher paired headline. The lateral-only score
of 0.733 is highly uncertain because that subset has only 20 cases and one PCA case.

**19. A video-native encoder preserves the hierarchy gain but does not replace DINO**
([full report](anatomy_target_experiments.md),
[matched public comparison](../results/public/video_vs_frame_missing_view_comparison_20seeds.md))
— the same 20-seed nested procedure was rerun with frozen V-JEPA 2 ViT-g/384. V-JEPA
receives 16 ordered frames jointly, creates three-dimensional tubelets, and performs
spatiotemporal attention before final token pooling. This differs materially from the
reference DINO model, which independently encodes selected frames and then averages their
embeddings.

The cache-level audit confirmed the exact same 344 ordered identities, 171 patient groups,
labels, and AP/lateral availability. V-JEPA scores 0.633 ± 0.019 on the full union with
M2/M3/PCA F1 0.755/0.256/0.887, versus DINO's 0.651 ± 0.027 and
0.790/0.347/0.815. The matched V-JEPA-minus-DINO macro-F1 difference is
−0.0178 ± 0.0345; V-JEPA wins 5/20 seeds. It improves PCA by .072 F1 but loses .035 on
M2 and .091 on M3. AP-only macro-F1 improves from 0.625 to 0.645, whereas the 20-case
lateral-only subset drops sharply because V-JEPA misses its sole PCA case.

This does not invalidate the missing-view architecture. Within V-JEPA, the hierarchy beats
the direct three-class missing-view control by +0.0322 ± 0.0274 with 18/20 wins. Keep the
hierarchy, but keep DINO as the primary frozen backbone. The comparison also changes frame
selection and resolution, so it establishes the practical best-model choice on this cohort;
it does not isolate temporal attention as the only causal difference.

## Conclusions

1. **Ship correct checkpoint normalization and fold-local feature standardization.** The
   missing V-JEPA channel normalization was a real implementation gap; it lifts the best
   ViT-L/256 seed-0 sweep from 0.459 to 0.479. The historical caches remain separate.
2. **Keep two candidate configurations until there is a locked test set.** V-JEPA 2
   ViT-g/384 + mean/std/logistic C3 has the best seed-0 score (0.486) and the best repeated
   logistic score (0.474 ± 0.019). DINOv2-L/252 + mean/std/LinearSVC has the best repeated
   LinearSVC score (0.477 ± 0.025) and the strongest m3 F1 (0.303). Calling either the sole
   winner would overstate this dataset.
3. **Higher resolution is model- and class-dependent.** It gives V-JEPA ViT-g a modest
   repeated gain, mainly through other_positive, but lowers m3. The same DINOv2-L weights
   are markedly worse at 518 than 252. Do not adopt resolution based on pixel count alone.
4. **m3 remains the ceiling, but the backbone is not completely irrelevant.** DINOv2-L/252
   raises repeated m3 F1 to 0.303 versus 0.242 for V-JEPA ViT-g/384 under LinearSVC. That is
   useful, but still weak enough that more labeled m3 studies and a locked test cohort are
   the highest-value next steps. Revisit fine-tuning only after the dataset is materially
   larger; the current partial fine-tune overfits.
5. **The medical-backbone conclusion needs nuance.** RAD-DINO is competitive and much better
   than RadImageNet ResNet50: the fair logistic comparison is 0.462 versus 0.377, with
   RAD-DINO winning all 20 paired splits. It is not the overall winner. The negative result
   applies to the older static CNN, not to all medical pretraining.
6. **Consider the task, not just the model.** Territory (MCA/ACA/PCA) is far more learnable
   (MCA F1 0.94). If m2-vs-m3 isn't clinically essential, reporting MCA/ACA/PCA — or m2 vs
   m3 vs other with transparent per-class numbers — is more trustworthy than forcing a
   distinction the imaging may not support from a single view.
7. **Augmentation is not a clean breakthrough.** Flip improves V-JEPA alone but does not
   beat the existing early fusion. Temporal-change sampling is the only useful input change:
   it lifts DINO m3 F1 from 0.301 to 0.306 and macro-F1 by 0.014. Nested class routing may add
   another 0.007 over fusion, but its 0.030 split-to-split SD is too large to call decisive.
   Do not report the post-hoc 0.518 route as final performance without a locked patient cohort.
8. **Retire `other_positive` for new models.** Use clean M2-vs-M3 for the binary scientific
   question; the hardened SVM candidate scores 0.584 ± 0.025 but is effectively tied with
   logistic regression. For a coherent third class, temporal-DINO + SVM reaches
   0.645 ± 0.025 on strict M2/M3/PCA and is the clearest head-level improvement. It remains
   a CV-selected candidate, not a replacement for the saved logistic model until tested on
   a locked patient cohort. ACA, M1, M4, and composites are outside its supported scope.
9. **Use lateral as a matched second view, not pooled augmentation.** On the identical
   concordant paired cohort, the earlier same-head comparison gives 0.651 for direct
   temporal AP+lateral fusion versus 0.576 for matched AP-only logistic regression, mainly
   through PCA. The new SVM controls show that raw concatenation is not universally useful:
   AP-only is 0.615 while AP+lateral is 0.569. The best hierarchy uses both views only for
   the PCA gate and AP for M2/M3, reaching 0.654, but its +0.002 change over direct fusion
   is not stable enough to promote before locked testing. Do not compare the AP-only
   full-cohort SVM's 0.645 directly with matched-view scores; the cohorts differ.
10. **Keep handcrafted radiomics as a transparent sensitivity/control representation.**
    Without masks or patient-space calibration it is not standard ROI radiomics, and its
    best standalone result is materially below DINO. Adding matched lateral radiomics does
    not rescue it: nested gate, late-blend, and early-fusion variants all trail direct paired
    DINO. Do not promote the earlier AP-only 0.655 fusion before locked testing.
11. **Use the missing-view hierarchy only when deployment must cover incomplete studies.**
    It scores 0.651 across all 344 strict identities and materially outperforms direct
    missing-view logistic experts, especially on M3. The unpaired cases do not improve the
    paired subset, so they should not be presented as a paired-performance augmentation.
    Keep paired-only models for paired-only deployment and validate the missing-view gate
    threshold on a locked patient cohort.
12. **Do not replace the DINO hierarchy with the tested video-native V-JEPA model.** Joint
    video encoding is architecturally cleaner than pre-encoder frame averaging, and V-JEPA
    raises PCA F1, but its lower M2/M3 performance produces a 0.018 mean macro-F1 deficit and
    15/20 matched-seed losses. Preserve V-JEPA as a PCA-oriented challenger or future hybrid
    component; any hybrid must be selected in nested training folds and validated on a locked
    cohort before promotion.

## Reproduce

Checkpoint revisions used: V-JEPA ViT-L/256 `b3c1679b7c34d3255ef3547f27c7b226aefab26f`,
ViT-g/256 `875c192b7b704b87d1e1d99345769632dd5f739a`, ViT-g/384
`12ca91694b230e0d4b5b0078af6f4ae1d51e933d`, DINOv2-L
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, and RAD-DINO
`110cbc18d5133582e320b43d53bf5c44e410c936`. The pinned RadImageNet control for new
extractions is `14460ee4c1276f6925611a63aa9a54a05d39eae0`; it does not identify the
checkpoint that produced the unsigned historical cache.

Recorded environment: Python 3.14.6, NumPy 2.5.1, pandas 3.0.3, pydicom 3.0.2,
matplotlib 3.11.0, scikit-learn 1.9.0, joblib 1.5.3, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, Transformers 5.13.1, and huggingface-hub 1.23.0.

Exact reproduction of the reported RadImageNet rows requires the untracked historical
artifact at `runs/ap_radimagenet/cache/radimagenet_ResNet50_AP_positive_subtype.npz`,
SHA-256 `18d43d191c53d73727b68828963a0f2c2c9241cc18447b222c7f167b09e3533c`.
The legacy-audit command below fails if that exact-path prerequisite is absent; a new signed
extraction is a separate experiment.

```bash
export DSA_BASE_DIR=~/M2_M3_data

# Corrected V-JEPA controls.
.venv/bin/python experiments.py --view AP --stage positive_subtype --device cuda --amp \
  --backbone facebook/vjepa2-vitl-fpc64-256 --revision b3c1679b7c34d3255ef3547f27c7b226aefab26f \
  --image-size 256 --normalize-input --batch-size 4 \
  --out runs/ap_exp_vitl256_norm
.venv/bin/python experiments.py --view AP --stage positive_subtype --device cuda --amp \
  --backbone facebook/vjepa2-vitg-fpc64-256 --revision 875c192b7b704b87d1e1d99345769632dd5f739a \
  --image-size 256 --normalize-input --batch-size 4 \
  --out runs/ap_exp_vitg256_norm
.venv/bin/python experiments.py --view AP --stage positive_subtype --device cuda --amp \
  --backbone facebook/vjepa2-vitg-fpc64-384 --revision 12ca91694b230e0d4b5b0078af6f4ae1d51e933d \
  --image-size 384 --normalize-input --batch-size 2 \
  --out runs/ap_exp_vitg384_norm

# Same DINOv2-L weights at low and high resolution, plus the medical control.
.venv/bin/python image_backbone_probe.py --view AP --model facebook/dinov2-large \
  --revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
  --image-size 252 --frame-batch-size 16 --device cuda --amp --out runs/ap_dinov2l252
.venv/bin/python image_backbone_probe.py --view AP --model facebook/dinov2-large \
  --revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
  --image-size 518 --frame-batch-size 4 --device cuda --amp --out runs/ap_dinov2l518
.venv/bin/python image_backbone_probe.py --view AP --model microsoft/rad-dino \
  --revision 110cbc18d5133582e320b43d53bf5c44e410c936 \
  --image-size 518 --frame-batch-size 8 --device cuda --amp --out runs/ap_raddino518
# Reuse and audit the exact historical unsigned cache used for the reported table.
.venv/bin/python radimagenet_probe.py --view AP --arch ResNet50 \
  --allow-legacy-cache --out runs/ap_radimagenet

# For a newly signed extraction, omit --allow-legacy-cache and use a separate output such
# as runs/ap_radimagenet_pinned. That is a new experiment and its scores may differ.

# Repeated fixed-recipe comparison. The first cache is the paired-delta reference.
# For the logistic table, change --classifier to logreg_C3 and the output filename.
.venv/bin/python compare_feature_caches.py \
  --feature radimagenet_r50=runs/ap_radimagenet/cache/radimagenet_ResNet50_AP_positive_subtype.npz \
  --feature rad_dino_518=runs/ap_raddino518/cache/image_microsoft-rad-dino_518_16_AP_positive_subtype.npz \
  --feature vjepa_l256_norm=runs/ap_exp_vitl256_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitl-fpc64-256_256_norm.npz \
  --feature vjepa_g256_norm=runs/ap_exp_vitg256_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-256_256_norm.npz \
  --feature vjepa_g384_norm=runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
  --feature dinov2_l252=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
  --feature dinov2_l518=runs/ap_dinov2l518/cache/image_facebook-dinov2-large_518_16_AP_positive_subtype.npz \
  --seeds 0:20 --representation mean --preprocessing std --classifier linsvm --jobs 8 \
  --allow-legacy-metadata \
  --out runs/radiology_svm_comparison/repeated_linsvm_7backbones_20seeds.json

# Direct three-class augmentation features.
for variant in hflip border90 multicrop top_contrast temporal_rgb phase_rgb; do
  .venv/bin/python image_backbone_probe.py --view AP --stage positive_subtype \
    --model facebook/dinov2-large --revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
    --image-size 252 --n-frames 16 --frame-batch-size 16 --input-variant "$variant" \
    --device cuda --amp --out runs/ap_dinov2l252_variants
done
.venv/bin/python experiments.py --view AP --stage positive_subtype --device cuda --amp \
  --backbone facebook/vjepa2-vitg-fpc64-384 \
  --revision 12ca91694b230e0d4b5b0078af6f4ae1d51e933d \
  --image-size 384 --normalize-input --horizontal-flip --batch-size 2 \
  --out runs/ap_exp_vitg384_norm_flip

# Direct m2/m3/other comparison; the JSON records every cache hash and per-seed result.
.venv/bin/python three_class_augmentation_experiments.py \
  --vjepa-original runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
  --vjepa-flip runs/ap_exp_vitg384_norm_flip/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm_hflip.npz \
  --dino-original runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
  --dino-flip runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_hflip.npz \
  --variant center90=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_border90.npz \
  --variant multicrop=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_multicrop.npz \
  --variant temporal_change=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_top_contrast.npz \
  --variant temporal_stats_rgb=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_temporal_rgb.npz \
  --variant phase_rgb=runs/ap_dinov2l252_variants/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype_phase_rgb.npz \
  --seeds 0:20 --jobs 8 \
  --out runs/three_class_augmentations/direct_three_class_20seeds.json
```
