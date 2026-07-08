# First Experiment: DSA Occlusion Subtype Classification with V-JEPA 2

**Status:** ready to run on the abnormal-only (positives) data we have today.
**Backbone:** V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), frozen.
**Last verified:** the full data pipeline and training loop were smoke-tested end-to-end
on this machine (`python scripts/smoke_test.py`) with a stubbed backbone.

---

## 1. Where we are

- We currently have **only abnormal (positive) studies** — every labeled run has an
  occlusion location. Normal (non-occluded) studies are coming later.
- That means **binary detection (occlusion vs. none) is not trainable yet**: there are no
  negatives. The smoke test confirms it: with no normals, the binary task collapses to a
  single class and reports a meaningless 100% accuracy.
- So the **first experiment is the positive-only subtype task**: given a DSA run that
  *does* contain an occlusion, classify *which kind*. This exercises the entire
  pipeline (loading → V-JEPA 2 → head → loss → metrics) on real data and gives us a
  genuine signal about whether V-JEPA 2 features transfer to DSA — all before normals
  arrive.

### Label distribution (approx., both views combined, from `AP_Lateral_Labels_Split.xlsx`)

The `*_Location` values collapse into the 3-class scheme (`normalize_location_label`
upper-cases and strips whitespace, so `LM2`, `" R M2"`, `"R M3 "` all map correctly):

| Class | Rough count | Source labels |
|-------|-------------|---------------|
| `m2` | ~455 | `L M2`, `R M2`, `L M2 and A3`, … (any label containing M2) |
| `m3` | ~150 | `L M3`, `R M3`, `L M3 and M4`, … (any label containing M3) |
| `other_positive` | ~120 | A2, A3, M1, M4, P1, P2, P4 and combos |

**This is heavily imbalanced (~3:1:1) and small (< 400 studies, ~725 labeled runs split
across two per-view models).** Both facts drive every design choice below.

---

## 2. The experiment

Train **two independent models**, one per view (`AP` and `Lateral`), because the anatomy
and vessel appearance differ between projections. For each:

| Choice | Value | Why |
|--------|-------|-----|
| Task | `positive_subtype` → {`m2`, `m3`, `other_positive`} | Only task trainable without normals |
| Backbone | V-JEPA 2 ViT-L, **frozen** (`--freeze-backbone`, the default) | ~300M params vs. <400 studies → full fine-tune overfits |
| Head | Attentive probe (recommended) or mean-pool + linear (current baseline) | V-JEPA 2's own downstream recipe is an attentive probe; see §5 |
| Clip | 16 frames @ 256×256 | Matches V-JEPA 2's downstream protocol; light enough for tiny data |
| Loss | Class-weighted cross-entropy (implemented) | Stops `m2` from swamping `m3`/`other` |
| Split | **Patient-grouped, stratified k-fold CV** (group on `Study_Key`) | A single split on <400 studies is too high-variance to trust |
| Primary metric | **Macro-F1** (+ balanced accuracy, per-class recall, confusion matrix) | Plain accuracy is misleading under 3:1:1 imbalance |
| Augmentation | Temporal jitter, small rotation/translation/scale, intensity jitter | Cheap regularization; the single biggest lever on small data |

### Baselines to beat (so we know V-JEPA 2 is worth it)

1. **Majority-class predictor** — the floor. Macro-F1 of a model that always says `m2`.
2. **2D CNN on a representative frame** — an ImageNet- or RadImageNet-pretrained ResNet/
   EfficientNet on the middle (or peak-contrast) frame. If a cheap 2D CNN matches frozen
   V-JEPA 2, the video model isn't earning its cost yet.

**Success = frozen V-JEPA 2 beats the majority baseline on macro-F1 with stable CV, and is
competitive with or better than the 2D-CNN baseline.** That's the go/no-go for investing
further in the video-foundation-model route.

---

## 3. How to run it

Environment (once):

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pydicom openpyxl
# GPU torch for the RTX 5090 (Blackwell / sm_120) — see §6 for the version caveat
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers            # needed only for the real backbone, not for --dry-run
export DSA_BASE_DIR=/path/to/M2_M3_data   # once the data drive is mounted
```

Verify the pipeline without downloading anything:

```bash
python scripts/smoke_test.py                                   # stubbed backbone, CPU
python train_dsa_backbone.py --view AP --stage positive_subtype --dry-run
```

Train (per view), once `transformers` + GPU torch are installed and data is mounted:

```bash
python train_dsa_backbone.py --view AP      --stage positive_subtype --epochs 30 --save-path runs/ap_subtype.pt
python train_dsa_backbone.py --view Lateral --stage positive_subtype --epochs 30 --save-path runs/lat_subtype.pt
```

`--freeze-backbone` is the default; add `--unfreeze-backbone` only to test full
fine-tuning as an ablation.

---

## 4. Roadmap (staged, so each stage is trainable when its data exists)

1. **Now — subtype warm-up (this experiment).** Positives only. Proves the pipeline and
   measures V-JEPA 2 transfer to DSA. Frozen backbone + attentive probe + CV.
2. **When normals arrive — binary detection.** Occlusion vs. none, per view. Re-enable the
   `binary_detection` stage (with real negatives, not `--treat-blank-as-negative`). This is
   the clinically primary task (triage/LVO detection).
3. **Later — finer localization.** If detection + subtype work, move toward side-specific
   or per-territory labels. Requires the dirty labels to be cleaned first (see §7).

---

## 5. Code changes still needed for a *rigorous* run

The current `train_dsa_backbone.py` runs and trains, but for the experiment above to be
publication-grade, these are the gaps (roughly in priority order):

1. **Cross-validation harness.** Today `split_records` uses the spreadsheet's `Split`
   column (single split). Add patient-grouped, stratified k-fold (group on `Study_Key`) and
   report mean ± std across folds.
2. **Real metrics.** Today only loss + accuracy are logged. Add macro-F1, balanced
   accuracy, per-class recall, and a confusion matrix (e.g. via `sklearn.metrics`).
3. **Attentive-pooling head.** Replace `hidden_state.mean(dim=1)` + linear
   ([train_dsa_backbone.py](../train_dsa_backbone.py)) with a small attentive probe
   (a learnable query + cross-attention), matching V-JEPA 2's downstream recipe. Keep it
   small so it doesn't overfit <400 samples.
4. **Augmentation** in `preprocess_clip` / the dataset transform (temporal + spatial +
   intensity).
5. **The 2D-CNN baseline** as a separate short script, for the go/no-go comparison.
6. **Run logging** (config + per-fold metrics to disk) for reproducibility.

Items 1–2 are the minimum to trust the numbers; 3–4 are the likely accuracy wins.

---

## 6. Environment caveats (this machine specifically)

- **RTX 5090 (32 GB, Blackwell/sm_120) + Python 3.14** is bleeding-edge. Stable CUDA torch
  wheels may lag; use the **cu128** build (or nightly) so the GPU is actually supported.
  CPU torch (`whl/cpu`) installs cleanly and is fine for `--dry-run` and `scripts/smoke_test.py`.
- `--dry-run` and the smoke test **do not** need `transformers` or a GPU — they never
  instantiate the backbone — so you can validate everything except the V-JEPA forward pass
  offline.
- Data location is resolved via `DSA_BASE_DIR` / `DSA_EXCEL_PATH` env vars (Linux default
  `~/M2_M3_data`); set `DSA_BASE_DIR` once the data drive is mounted.

---

## 7. Risks & limitations to keep in mind

- **Tiny data.** ~725 labeled runs across two models and 3 imbalanced classes. Some CV
  folds will have **zero** samples of a rare class — expect unstable per-class metrics and
  report ranges, not point estimates.
- **Domain gap.** V-JEPA 2 is pretrained on natural motion video; DSA is grayscale contrast
  dynamics. Frozen features may transfer poorly — which is exactly what this experiment
  measures. Don't assume transfer; the 2D-CNN baseline is the reality check.
- **Label hygiene.** A handful of dirty entries exist (`LM2`, `" R M2"`, `"R M3 "`, `LM3`).
  The 3-class mapping is robust to these, but any finer-grained future task needs them
  normalized in the spreadsheet first.
- **`other_positive` is a grab-bag** (A2/A3/M1/M4/P1/P2/P4). Good macro-F1 on it is hard and
  arguably less clinically meaningful than `m2`/`m3`; weigh that when reading results.
