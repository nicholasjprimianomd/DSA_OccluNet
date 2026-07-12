# Experiments: improving per-class DSA subtype performance

All results are **patient-grouped 5-fold cross-validation** (StratifiedGroupKFold on
`Study_Key`) and out-of-fold. The AP comparison set has 366 runs from 184 groups
(m2=240, m3=67, other_positive=59). The initial experiments used one split seed; the
new backbone/resolution comparison repeats the same fixed recipe over 20 paired split
seeds. Tooling: [experiments.py](../experiments.py) (V-JEPA 2 recipe sweep),
[image_backbone_probe.py](../image_backbone_probe.py) (per-frame image backbones),
[compare_feature_caches.py](../compare_feature_caches.py) (repeated paired comparison),
[three_class_augmentation_experiments.py](../three_class_augmentation_experiments.py)
(direct three-class augmentation/fusion),
[anatomy_task_experiments.py](../anatomy_task_experiments.py)
(anatomically coherent target redesign),
[attn_probe.py](../attn_probe.py) (attentive pooling), and
[cross_validate.py](../cross_validate.py) (baseline probe).

## Headline (updated 2026-07-11)

The old best was unnormalized V-JEPA 2 ViT-g/384 at 0.476. Applying the checkpoint's
documented channel normalization raises its best seed-0 sweep result to **0.486**. A
genuinely different model, DINOv2-L/252, is effectively tied and is stronger on m3.

| Frozen AP backbone | Resolution | Seed-0 best-of-sweep | Fixed LinearSVC, 20 seeds | Fixed logreg C3, 20 seeds |
|---|---:|---:|---:|---:|
| V-JEPA 2 ViT-L (normalized) | 256 | 0.479 | 0.438 ± 0.018 | 0.442 ± 0.023 |
| V-JEPA 2 ViT-g (normalized) | 256 | 0.480 | 0.446 ± 0.025 | 0.450 ± 0.024 |
| **V-JEPA 2 ViT-g (normalized)** | **384** | **0.486** | 0.460 ± 0.021 | **0.474 ± 0.019** |
| **DINOv2-L, per-frame** | **252** | 0.477 | **0.477 ± 0.025** | 0.473 ± 0.025 |
| DINOv2-L, per-frame | 518 | 0.443 | 0.438 ± 0.025 | 0.456 ± 0.028 |
| RAD-DINO, per-frame | 518 | 0.478 | 0.457 ± 0.019 | 0.462 ± 0.020 |

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
AP best macro-F1 was **0.407**, below the original V-JEPA 2 result. This rules out that
specific static CNN, not medical pretraining as a category; RAD-DINO below is substantially
better. No usable public angiography-video checkpoint was available for these runs.

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
all six corrected feature sets were evaluated on the same 20 `StratifiedGroupKFold`
seeds with one fixed representation (mean), fold-local standardization, and the same
classifier. This removes the most obvious best-of-13 recipe selection bias.

- LinearSVC favors DINOv2-L/252 over V-JEPA 2 ViT-g/384 by +0.017 on average (15/20 seeds).
- Logistic C3 makes them indistinguishable: 0.473 vs 0.474 (DINO wins 10/20 seeds).
- Repeated folds measure split sensitivity, not independent-dataset generalization; the
  final model still needs a locked patient-level test cohort or external validation.
- The current subset has one accession per MRN, so `Study_Key` is effectively patient-level;
  future repeat admissions must be grouped by a stable patient identifier rather than accession.

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
the honest nested version, source selection is repeated inside each outer training fold and
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

| Redesigned task | Counts | Best macro-F1 | Critical class result |
|---|---|---:|---|
| **Clean M2 vs M3** | 237 / 65 | **0.581 ± 0.025** | M3 F1 **0.332** |
| **M2/M3/PCA** | 237 / 65 / 29 | **0.619 ± 0.026** | M2 0.807, M3 0.332, PCA 0.718 |
| Territory MCA/ACA/PCA | 319 / 15 / 29 | 0.594 ± 0.019 | ACA F1 **0.050** |
| M2/M3/ACA/PCA | 237 / 65 / 15 / 29 | 0.463 ± 0.016 | ACA F1 **0.073** |
| M2/M3/other-MCA | 237 / 65 / 15 | 0.384 ± 0.027 | M1/M4-group F1 **0.071** |
| MCA M1/M2/M3/M4 | 5 / 237 / 65 / 10 | 0.282 ± 0.031 | M1 0.027, M4 0.030 |
| M1/M2/M3/M4/ACA/PCA | 5 / 237 / 65 / 10 / 15 / 29 | 0.301 ± 0.027 | M4 0.000, ACA 0.021 |

This resolves the target-design question. **Clean M2-vs-M3 is the defensible binary task,
and strict M2/M3/PCA is the best coherent three-class task tested.** Adding PCA leaves m3
F1 unchanged at 0.332 and yields PCA F1 0.718. The higher three-class macro-F1 is not a
like-for-like improvement over binary because PCA is easier, but it shows that PCA is a
learnable replacement for the catch-all. A territory-first hierarchy remains non-operational
with 15 ACA cases; M1/M4 also cannot form a reliable explicit or abstention class. Changing
regularization and L2 normalization does not rescue these sparse classes.

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
   than RadImageNet ResNet50, but it is not the overall winner. The negative result applies
   to the older static CNN, not to all medical pretraining.
6. **Consider the task, not just the model.** Territory (MCA/ACA/PCA) is far more learnable
   (MCA F1 0.94). If m2-vs-m3 isn't clinically essential, reporting MCA/ACA/PCA — or m2 vs
   m3 vs other with honest per-class numbers — is more trustworthy than forcing a
   distinction the imaging may not support from a single view.
7. **Augmentation is not a clean breakthrough.** Flip improves V-JEPA alone but does not
   beat the existing early fusion. Temporal-change sampling is the only useful input change:
   it lifts DINO m3 F1 from 0.301 to 0.306 and macro-F1 by 0.014. Nested class routing may add
   another 0.007 over fusion, but its 0.030 split-to-split SD is too large to call decisive.
   Do not report the post-hoc 0.518 route as final performance without a locked patient cohort.
8. **Retire `other_positive` for new models.** Use clean M2-vs-M3 (0.581 ± 0.025) for the
   binary scientific question or strict M2/M3/PCA (0.619 ± 0.026) when a coherent third
   class is required. The latter preserves m3 F1 and learns PCA at F1 0.718. ACA, M1, M4,
   and composites are outside its supported scope and must not be silently folded into it.
9. **Use lateral as a matched second view, not pooled augmentation.** On the identical
   concordant paired cohort, AP+lateral fusion reaches 0.646 versus 0.572 AP-only and 0.591
   lateral-only. A half-weight composite sensitivity reaches 0.653, but those composites
   come from only two patients and should remain exploratory.

## Reproduce

Checkpoint revisions used: V-JEPA ViT-L/256 `b3c1679b7c34d3255ef3547f27c7b226aefab26f`,
ViT-g/256 `875c192b7b704b87d1e1d99345769632dd5f739a`, ViT-g/384
`12ca91694b230e0d4b5b0078af6f4ae1d51e933d`, DINOv2-L
`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, and RAD-DINO
`110cbc18d5133582e320b43d53bf5c44e410c936`.

Recorded environment: Python 3.14.6, NumPy 2.5.1, pandas 3.0.3, pydicom 3.0.2,
matplotlib 3.11.0, scikit-learn 1.9.0, joblib 1.5.3, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, Transformers 5.13.1, and huggingface-hub 1.23.0.

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

# Repeated fixed-recipe comparison. For the second table also change --classifier to
# logreg_C3 and --out to repeated_logregc3_norm_20seeds.json.
.venv/bin/python compare_feature_caches.py \
  --feature vjepa_l256_norm=runs/ap_exp_vitl256_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitl-fpc64-256_256_norm.npz \
  --feature vjepa_g256_norm=runs/ap_exp_vitg256_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-256_256_norm.npz \
  --feature vjepa_g384_norm=runs/ap_exp_vitg384_norm/cache/rich_AP_positive_subtype_f16_vjepa2-vitg-fpc64-384_384_norm.npz \
  --feature dinov2_l252=runs/ap_dinov2l252/cache/image_facebook-dinov2-large_252_16_AP_positive_subtype.npz \
  --feature dinov2_l518=runs/ap_dinov2l518/cache/image_facebook-dinov2-large_518_16_AP_positive_subtype.npz \
  --feature rad_dino_518=runs/ap_raddino518/cache/image_microsoft-rad-dino_518_16_AP_positive_subtype.npz \
  --seeds 0:20 --representation mean --preprocessing std --classifier linsvm --jobs 8 \
  --out runs/backbone_comparison/repeated_linsvm_norm_20seeds.json

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
