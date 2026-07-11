# Experiments: improving per-class DSA subtype performance

All results are **patient-grouped 5-fold cross-validation** (StratifiedGroupKFold on
`Study_Key`), out-of-fold, on the frozen V-JEPA 2 features. Tooling:
[experiments.py](../experiments.py) (recipe sweep), [attn_probe.py](../attn_probe.py)
(attentive-pooling probe), [cross_validate.py](../cross_validate.py) (the honest baseline).

## Headline

| | AP macro-F1 | Lateral macro-F1 |
|---|---|---|
| Majority-class baseline | 0.26 | 0.25 |
| Linear probe, raw mean-pooled features (previous default) | 0.42 | 0.41 |
| **+ standardize features + balanced linear model (best recipe)** | **0.46** | **0.45** |
| Attentive-pooling probe over tokens | 0.41 | — |
| Partial fine-tune (last 2 blocks), no-peek / best-epoch | 0.44 / 0.48 | — |
| **Higher resolution: V-JEPA 2 ViT-g @ 384px** | **0.48** | — |
| Medical backbone: RadImageNet ResNet50 (per-frame) | 0.41 | — |

**The one reliable win is feature standardization + a class-balanced linear model**
(LinearSVC / logistic regression). It's now the default in `cross_validate.py`
(`--standardize`). Everything fancier did not beat it.

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
away detail. The official 384px ViT-g checkpoint lifts AP best macro-F1 to **0.48** (from
0.46). The gain is real but lands on the easy classes: **other_positive F1 0.44→0.55** and
m2 0.70→0.76, while **m3 got *worse* (0.24→0.12)**. Higher resolution sharpens the distinct
territories; it doesn't rescue the m2/m3 subtlety. Caveat: this also swaps ViT-L→ViT-g, so
resolution and model size are confounded. Worth adopting if the extra compute is acceptable.

**7. Medical backbone: RadImageNet** ([radimagenet_probe.py](../radimagenet_probe.py)) — there
is **no public angiography video backbone**; the closest medical option is RadImageNet (CNNs
trained only on radiology, incl. angiography/X-ray), applied per-frame + temporal mean-pool.
AP best macro-F1 **0.41 — worse than V-JEPA 2's 0.46.** A 2D CNN trained on static CT/MR/US
slices doesn't beat a large natural-video model here; V-JEPA 2's temporal modeling and scale
win despite the domain mismatch.

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

## Conclusions

1. **Ship standardization** — it's a free, universal +0.04. Done (default in `cross_validate.py`).
2. **m3 is the ceiling, and it's intrinsic to the frozen features.** M2 and M3 are adjacent
   MCA branches; better heads (attentive pooling), richer pooling, and view fusion all fail
   to recover the distinction. The evidence (global L/R easy, local M2/M3 hard) says the
   detail isn't in the mean/token features as-is.
3. **Backbone fine-tuning was tested and does not help at this scale** (experiment 4:
   overfits, no reliable gain over the frozen probe). So the lever is **not the model** — it's
   the **inputs and the data**:
   - **More labeled studies, especially m3** — 67/75 m3 runs is the binding constraint. This is
     the single highest-value action, and after testing every model-side lever it's the only
     thing left that can move m3.
   - **Higher spatial resolution helps the easy classes** (ViT-g @ 384px → 0.48, experiment 6) —
     adopt it if compute allows, but it doesn't fix m3.
   - A **medical backbone does not help** (RadImageNet, experiment 7, 0.41 < 0.46) and no public
     angiography video model exists. More *frames* doesn't help either (experiment 5).
   - Revisit fine-tuning only once the dataset is materially larger.
4. **Consider the task, not just the model.** Territory (MCA/ACA/PCA) is far more learnable
   (MCA F1 0.94). If m2-vs-m3 isn't clinically essential, reporting MCA/ACA/PCA — or m2 vs
   m3 vs other with honest per-class numbers — is more trustworthy than forcing a
   distinction the imaging may not support from a single view.

## Reproduce

```bash
export DSA_BASE_DIR=~/M2_M3_data
python experiments.py  --view AP --stage positive_subtype --device cuda --amp --out runs/ap_exp
python attn_probe.py   --view AP --stage positive_subtype --device cuda --amp --out runs/ap_attn
python cross_validate.py --view AP --stage positive_subtype --device cuda --amp --out runs/ap_cv   # standardized by default
```
