"""Interactive visual inspection of the DSA occlusion model's inputs and outputs.

Renders, for real DSA runs, everything the model sees and produces, as a single
self-contained HTML page with an in-browser **cine player** per run:

  1. INPUT      the exact frames fed to the frozen backbone (DICOM windowing, temporal
                frame selection, resize, grayscale->RGB, ImageNet channel normalization).
                Scrub or play through every frame like a DSA cine loop.
  2. LABEL      the ground-truth class for the selected task and the raw spreadsheet text.
  3. AUGMENT    the deterministic input construction of the "latest experiment" feature
                source (uniform vs. temporal-change frame selection), including which
                frames were selected.
  4. OUTPUT     the probe's predicted class and full class-probability vector.
  5. SALIENCY   what the backbone attends to, per frame, toggled per class. Three methods:
                  * rollout   gradient-weighted attention rollout (Chefer et al. 2021),
                              class-specific and sharper than Grad-CAM on ViTs  [default]
                  * attention raw last-layer CLS->patch attention (class-agnostic; the
                              classic clean DINOv2 map)
                  * gradcam   Grad-CAM on the penultimate block (class-specific)

The "latest experiment" is the anatomy-task study (``runs/anatomy_tasks/results.json``):
frozen **DINOv2-large @ 252 px, 16 frames/run**, temporal mean pooling, then a
fold-selected standardized logistic-regression probe. Predictions are **held-out**
patient-grouped cross-validation (each run scored by a fold that never saw it), so the
summary macro-F1 matches the write-up. Pass ``--full-data`` for the optimistic saved
all-data probe instead (debugging only).

De-identification: the page shows only a random UUID per run — never the real accession.
A private ``id_map.csv`` (written next to the output, gitignored) maps UUID -> accession
for your own traceability. The HTML is therefore safe to host.

Example:
    .venv/bin/python visualize_model_io.py --task m2_m3_pca_strict --method rollout \
        --n-samples 9 --device cuda --out runs/viz_model_io
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import types
import uuid
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import viz
from anatomy_task_experiments import build_tasks, new_probe
from experiments import make_folds
from image_backbone_probe import prepare_model_images, select_top_contrast_frames
from occlusion_loader import build_manifests, default_base_dir, default_excel_path, load_dicom_sequence
from train_dsa_backbone import normalize_sequence, sample_frame_indices, select_device


ANATOMY_RESULTS = Path("runs/anatomy_tasks/results.json")
DINOV2_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
DISPLAY_PX = 256
JET = matplotlib.colormaps["jet"]


# --------------------------------------------------------------------------------------
# Experiment discovery
# --------------------------------------------------------------------------------------
def load_experiment(results_path: Path) -> dict:
    results = json.loads(results_path.read_text())
    feature_sources = results["protocol"]["feature_sources"]
    catalog = {}
    for task_name, art in results["final_model_artifacts"].items():
        source_name = art.get("feature_source")
        source = feature_sources.get(source_name) if source_name else None
        if source is None:
            continue
        signature = source.get("signature") or {}
        model_name = signature.get("model", "")
        if "dinov2" not in model_name.lower():
            # This tool reconstructs the DINOv2 per-frame input path; V-JEPA (video)
            # sources use different preprocessing, so skip them.
            continue
        catalog[task_name] = {
            "probe_path": art["path"],
            "feature_source": source_name,
            "cache_path": source["path"],
            "model": model_name,
            "image_size": signature.get("image_size"),
            "n_frames": signature.get("n_frames"),
            "input_variant": signature.get("input_variant", "uniform"),
            "image_mean": signature.get("image_mean", [0.485, 0.456, 0.406]),
            "image_std": signature.get("image_std", [0.229, 0.224, 0.225]),
            "labels": art["labels"],
            "recipe": art["recipe"],
            "n_samples": art["n_samples"],
        }
    return catalog


# --------------------------------------------------------------------------------------
# Probe -> effective linear head; held-out predictions
# --------------------------------------------------------------------------------------
def probe_linear_head(pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Collapse StandardScaler + LogisticRegression into one affine map on raw features."""
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["logisticregression"]
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    coef = clf.coef_.astype(np.float64)
    intercept = clf.intercept_.astype(np.float64)
    if coef.shape[0] == 1:
        coef = np.vstack([-coef[0], coef[0]])
        intercept = np.array([-intercept[0], intercept[0]])
    w_eff = coef / scale
    b_eff = intercept - (coef * (mean / scale)).sum(axis=1)
    return w_eff, b_eff


def probe_scores(pipeline, features: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)
    scores = pipeline.decision_function(features)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def crossval_predictions(X, y, groups, recipe, n_classes, n_splits, seed):
    """Patient-grouped out-of-fold predictions; also return each run's holdout fold probe."""
    folds = make_folds(y, groups, n_splits, seed)
    probs = np.zeros((len(y), n_classes))
    preds = np.full(len(y), -1, dtype=int)
    fold_of = np.full(len(y), -1, dtype=int)
    pipelines = []
    for fold_index, (train, val) in enumerate(folds):
        probe = new_probe(recipe)
        probe.fit(X[train], y[train])
        probs[val] = probe_scores(probe, X[val], n_classes)
        preds[val] = probs[val].argmax(axis=1)
        fold_of[val] = fold_index
        pipelines.append(probe)
    return probs, preds, fold_of, pipelines


# --------------------------------------------------------------------------------------
# Backbone saliency: Grad-CAM, gradient-weighted attention rollout, raw CLS attention
# --------------------------------------------------------------------------------------
class BackboneSaliency:
    def __init__(self, model_name: str, revision: str, device: torch.device):
        from transformers import AutoModel

        self.device = device
        # eager attention so output_attentions returns weights (SDPA does not).
        self.model = (
            AutoModel.from_pretrained(model_name, revision=revision, attn_implementation="eager")
            .to(device).float().eval()
        )
        self.target_layer = self.model.encoder.layer[-2]
        self._activation = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self._activation = output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _to_grid(mask_patches: torch.Tensor, out_hw) -> torch.Tensor:
        side = int(round(mask_patches.shape[-1] ** 0.5))
        grid = mask_patches.reshape(1, 1, side, side)
        up = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
        lo, hi = torch.quantile(up, 0.02), torch.quantile(up, 0.98)
        return ((up - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

    def _normalize(self, frames_rgb, channel_mean, channel_std):
        return (frames_rgb.to(self.device) - channel_mean) / channel_std

    def maps(self, frames_rgb, channel_mean, channel_std, w_eff, method, class_indices):
        """Return {view_name: (T, H, W) in [0,1]} for the requested method.

        rollout/gradcam produce one map per class; attention is class-agnostic.
        """
        if method == "attention":
            return {"attention": self._attention(frames_rgb, channel_mean, channel_std)}
        if method == "gradcam":
            return {self._cls_name(c): self._gradcam(frames_rgb, channel_mean, channel_std, w_eff, c)
                    for c in class_indices}
        return {self._cls_name(c): self._rollout(frames_rgb, channel_mean, channel_std, w_eff, c)
                for c in class_indices}

    @staticmethod
    def _cls_name(c):
        return f"class{c}"

    @torch.no_grad()
    def _attention(self, frames_rgb, channel_mean, channel_std):
        out_hw = frames_rgb.shape[-2:]
        pixel_values = self._normalize(frames_rgb, channel_mean, channel_std)
        attentions = self.model(pixel_values=pixel_values, output_attentions=True).attentions
        cls_to_patch = attentions[-1][:, :, 0, 1:].mean(dim=1)   # (T, n_patches) mean over heads
        return torch.stack([self._to_grid(cls_to_patch[t], out_hw) for t in range(cls_to_patch.shape[0])]).cpu().numpy()

    def _gradcam(self, frames_rgb, channel_mean, channel_std, w_eff, target_class):
        out_hw = frames_rgb.shape[-2:]
        pixel_values = self._normalize(frames_rgb, channel_mean, channel_std).requires_grad_(True)
        outputs = self.model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        score = pooled.mean(dim=0) @ w_eff[target_class]
        grads = torch.autograd.grad(score, self._activation)[0]
        acts = self._activation.detach()
        weights = grads[:, 1:, :].mean(dim=1, keepdim=True)
        cam = F.relu((weights * acts[:, 1:, :]).sum(dim=-1))     # (T, n_patches)
        return torch.stack([self._to_grid(cam[t], out_hw) for t in range(cam.shape[0])]).detach().cpu().numpy()

    def _rollout(self, frames_rgb, channel_mean, channel_std, w_eff, target_class):
        """Gradient-weighted attention rollout (Chefer et al., 2021), per frame."""
        out_hw = frames_rgb.shape[-2:]
        maps = []
        for t in range(frames_rgb.shape[0]):
            px = self._normalize(frames_rgb[t : t + 1], channel_mean, channel_std).requires_grad_(True)
            outputs = self.model(pixel_values=px, output_attentions=True)
            attentions = outputs.attentions
            score = outputs.pooler_output[0] @ w_eff[target_class]
            grads = torch.autograd.grad(score, attentions, retain_graph=False)
            n_tokens = attentions[0].shape[-1]
            result = torch.eye(n_tokens, device=self.device)
            eye = torch.eye(n_tokens, device=self.device)
            for attn, grad in zip(attentions, grads):
                weighted = (attn[0] * grad[0]).clamp(min=0).mean(dim=0)   # (tok, tok)
                weighted = weighted + eye
                weighted = weighted / weighted.sum(dim=-1, keepdim=True)
                result = weighted @ result
            maps.append(self._to_grid(result[0, 1:], out_hw))
        return torch.stack(maps).detach().cpu().numpy()


# --------------------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------------------
def _jpeg_uri(rgb_uint8: np.ndarray, quality: int = 85) -> str:
    image = Image.fromarray(rgb_uint8)
    if image.width != DISPLAY_PX:
        image = image.resize((DISPLAY_PX, DISPLAY_PX), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def frame_uri(frame_rgb: np.ndarray, rgb_mode: bool) -> str:
    array = np.transpose(frame_rgb, (1, 2, 0)) if rgb_mode else np.stack([frame_rgb[0]] * 3, axis=-1)
    return _jpeg_uri((np.clip(array, 0, 1) * 255).astype(np.uint8))


def overlay_uri(frame_rgb: np.ndarray, cam: np.ndarray, rgb_mode: bool, alpha: float = 0.45) -> str:
    base = np.transpose(frame_rgb, (1, 2, 0)) if rgb_mode else np.stack([frame_rgb[0]] * 3, axis=-1)
    base = np.clip(base, 0, 1)
    heat = JET(cam)[..., :3]
    blended = (1 - alpha) * base + alpha * heat
    return _jpeg_uri((blended * 255).astype(np.uint8))


def fig_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_summary_figure(conf, label_names, counts, title, cm_subtitle) -> str:
    fig, (ax_cm, ax_ct) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    k = len(label_names)
    im = ax_cm.imshow(conf, cmap="Blues")
    ax_cm.set_xticks(range(k), labels=label_names, fontsize=9)
    ax_cm.set_yticks(range(k), labels=label_names, fontsize=9)
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_title(cm_subtitle, fontsize=10)
    thresh = conf.max() / 2 if conf.max() else 0
    for i in range(k):
        for j in range(k):
            ax_cm.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=10,
                       color="white" if conf[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_ct.bar(range(k), counts, color="#4c78a8")
    ax_ct.set_xticks(range(k), labels=label_names, fontsize=9)
    ax_ct.set_ylabel("runs")
    ax_ct.set_title("Class distribution", fontsize=10)
    for i, v in enumerate(counts):
        ax_ct.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig_to_data_uri(fig)


# --------------------------------------------------------------------------------------
# Sample selection + input reconstruction
# --------------------------------------------------------------------------------------
def build_dicom_index(excel, base_dir):
    ap_records, lat_records, _ = build_manifests(excel_path=excel, base_dir=base_dir)
    return {(r.accession, r.run_column): r.dicom_path for r in [*ap_records, *lat_records]}


def choose_samples(labels, preds, n_samples, n_classes, seed):
    """Stratify across true classes; balanced mix of correct and incorrect predictions."""
    rng = np.random.default_rng(seed)
    per_class = max(1, n_samples // n_classes)
    chosen: list[int] = []
    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        wrong = list(idx[preds[idx] != labels[idx]])
        right = list(idx[preds[idx] == labels[idx]])
        rng.shuffle(wrong)
        rng.shuffle(right)
        want_right = (per_class + 1) // 2
        picked = right[:want_right] + wrong[: per_class - min(len(right), want_right)]
        pool = right[want_right:] + wrong[per_class - min(len(right), want_right):]
        picked += pool[: per_class - len(picked)]
        chosen.extend(int(i) for i in picked)
    remaining = [i for i in range(len(labels)) if i not in set(chosen)]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, n_samples - len(chosen))])
    return chosen[:n_samples]


def reconstruct_input(dicom_path, cfg):
    sequence = torch.from_numpy(load_dicom_sequence(dicom_path))
    args = types.SimpleNamespace(input_variant=cfg["input_variant"],
                                 image_size=cfg["image_size"], n_frames=cfg["n_frames"])
    frames_rgb = prepare_model_images(sequence, args, dicom_path)
    total_frames = int(sequence.shape[0])
    if cfg["input_variant"] == "top_contrast":
        selected = select_top_contrast_frames(normalize_sequence(sequence), cfg["n_frames"]).tolist()
    else:
        selected = sample_frame_indices(total_frames, cfg["n_frames"]).tolist()
    return {"frames_rgb": frames_rgb, "total_frames": total_frames, "selected_frames": selected}


# --------------------------------------------------------------------------------------
# Interactive HTML
# --------------------------------------------------------------------------------------
def build_html(cfg, task_name, method, summary_uri, samples, overall_acc, macro,
               eval_kind, full_data) -> str:
    variant_note = {
        "uniform": "Uniform temporal sampling — frames evenly spaced across the run.",
        "top_contrast": "Temporal-change selection — frame 0 plus frames with the largest "
                        "adjacent-frame change (where contrast moves).",
    }.get(cfg["input_variant"], cfg["input_variant"])
    method_note = {
        "rollout": "Gradient-weighted attention rollout (Chefer et al. 2021) — class-specific, "
                   "aggregates attention×gradient across all 24 transformer blocks.",
        "attention": "Raw last-layer CLS→patch attention (class-agnostic; the classic DINOv2 map).",
        "gradcam": "Grad-CAM on the penultimate transformer block (class-specific).",
    }[method]
    esc = html.escape
    samples_json = json.dumps(samples)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DSA model I/O — {esc(task_name)}</title>
<style>
  :root {{ --bg:#0f1216; --panel:#161b22; --line:#2a323d; --ink:#e6e9ee; --mut:#9aa4b2;
          --ok:#37b26b; --bad:#e5606a; --accent:#5b9bd5; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:var(--bg); color:var(--ink); }}
  header {{ padding:20px 26px; background:var(--panel); border-bottom:1px solid var(--line); }}
  h1 {{ margin:0 0 6px; font-size:19px; }}
  .meta {{ color:var(--mut); font-size:13px; line-height:1.7; }}
  .meta code {{ color:#d6b36a; }}
  .banner {{ background:#12261a; border:1px solid #1f5137; color:#bfe8cf; padding:9px 13px;
             border-radius:6px; font-size:13px; margin-top:12px; }}
  .banner.debug {{ background:#2e2410; border-color:#5b4a1e; color:#f0d9a0; }}
  main {{ padding:18px 26px 70px; }}
  .summary img {{ max-width:100%; background:#fff; border-radius:8px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:20px; margin-top:22px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  .chead {{ display:flex; justify-content:space-between; align-items:center; padding:10px 13px;
            border-bottom:1px solid var(--line); font-size:13px; }}
  .badge {{ font-weight:700; padding:2px 9px; border-radius:20px; font-size:12px; }}
  .badge.ok {{ background:rgba(55,178,107,.18); color:var(--ok); }}
  .badge.bad {{ background:rgba(229,96,106,.18); color:var(--bad); }}
  .stage {{ position:relative; background:#000; aspect-ratio:1/1; }}
  .stage img {{ width:100%; height:100%; display:block; object-fit:contain; }}
  .fnum {{ position:absolute; top:8px; right:10px; font-size:12px; color:#cfe; background:rgba(0,0,0,.5);
           padding:2px 7px; border-radius:12px; }}
  .ctrls {{ display:flex; align-items:center; gap:10px; padding:9px 13px; }}
  .ctrls button.play {{ width:34px; height:34px; border-radius:50%; border:1px solid var(--line);
            background:#20293280; color:var(--ink); cursor:pointer; font-size:14px; }}
  .ctrls input[type=range] {{ flex:1; accent-color:var(--accent); }}
  .views {{ display:flex; flex-wrap:wrap; gap:6px; padding:0 13px 12px; }}
  .views button {{ border:1px solid var(--line); background:#1b2129; color:var(--mut);
            padding:4px 10px; border-radius:6px; cursor:pointer; font-size:12px; }}
  .views button.active {{ background:var(--accent); color:#06121e; border-color:var(--accent); font-weight:600; }}
  .pred {{ padding:11px 13px 14px; border-top:1px solid var(--line); font-size:12.5px; color:var(--mut); }}
  .bars {{ display:flex; flex-direction:column; gap:6px; margin:6px 0 10px; }}
  .bar {{ display:grid; grid-template-columns:44px 1fr 62px; align-items:center; gap:8px; }}
  .bar .lab {{ font-weight:600; color:var(--ink); }}
  .track {{ background:#1b2129; border-radius:5px; height:16px; overflow:hidden; }}
  .fill {{ height:100%; border-radius:5px; }}
  .bar .val {{ text-align:right; font-variant-numeric:tabular-nums; }}
  code {{ background:#1b2129; padding:1px 5px; border-radius:4px; color:#d6b36a; }}
</style></head>
<body>
<header>
  <h1>What the DSA occlusion model sees and predicts — task: {esc(task_name)}</h1>
  <div class="meta">
    <b>Latest experiment:</b> anatomy-task probes (<code>runs/anatomy_tasks</code>)<br/>
    <b>Backbone:</b> frozen <code>{esc(cfg['model'])}</code> @ {cfg['image_size']} px,
      {cfg['n_frames']} frames/run, temporal mean pooling<br/>
    <b>Probe:</b> <code>{esc(cfg['recipe'])}</code> · classes: {esc(', '.join(cfg['labels']))}<br/>
    <b>Input construction:</b> {esc(variant_note)}<br/>
    <b>Saliency (“{esc(method)}”):</b> {esc(method_note)}
      Toggle <i>Input</i> / per-class overlays and scrub or ▶ play the frames.
  </div>
  <div class="banner {'debug' if full_data else ''}">
    {"<b>Debug: full-data-fit probe (optimistic — trained on these runs).</b>" if full_data
      else "<b>Predictions are held-out results.</b>"}
    Every run is scored by {esc(eval_kind)} over all {cfg['n_samples']} task runs:
    <b>macro-F1 {macro:.3f}</b>, accuracy {overall_acc:.1%}. Macro-F1 is the number to trust
    under class imbalance. Shown IDs are random UUIDs — no patient accessions appear here.
  </div>
</header>
<main>
  <div class="summary"><img src="{summary_uri}" alt="summary"/></div>
  <div class="cards" id="cards"></div>
</main>
<script>
const SAMPLES = {samples_json};
const cards = document.getElementById('cards');
SAMPLES.forEach((s, i) => {{
  const views = Object.keys(s.frames);
  const viewBtns = views.map(v => {{
    const label = v === 'input' ? 'Input' : v === 'attention' ? 'Attention' : 'CAM: ' + v;
    return `<button data-i="${{i}}" data-view="${{v}}">${{label}}</button>`;
  }}).join('');
  const bars = s.probs.map(p => {{
    const pct = Math.round(p.p * 100);
    const col = p.name === s.pred ? 'var(--ok)' : '#6b7684';
    const tag = p.name === s.true ? ' ← true' : '';
    return `<div class="bar"><span class="lab">${{p.name}}</span>
      <div class="track"><div class="fill" style="width:${{pct}}%;background:${{col}}"></div></div>
      <span class="val">${{p.p.toFixed(2)}}${{tag}}</span></div>`;
  }}).join('');
  const el = document.createElement('figure');
  el.className = 'card';
  el.style.margin = '0';
  el.innerHTML = `
    <div class="chead">
      <span><code>${{s.id}}</code> · ${{s.run}}</span>
      <span class="badge ${{s.correct ? 'ok' : 'bad'}}">${{s.correct ? '✓ correct' : '✗ wrong'}}</span>
    </div>
    <div class="stage"><img id="img${{i}}"/><span class="fnum" id="fn${{i}}"></span></div>
    <div class="ctrls">
      <button class="play" id="pl${{i}}">▶</button>
      <input type="range" id="sl${{i}}" min="0" max="${{s.n_frames - 1}}" value="${{s.headline}}"/>
    </div>
    <div class="views" id="vw${{i}}">${{viewBtns}}</div>
    <div class="pred">
      true <b>${{s.true}}</b> &nbsp;·&nbsp; predicted <b>${{s.pred}}</b><br/>
      <div class="bars">${{bars}}</div>
      raw label <code>${{s.label_raw}}</code> → ${{s.true}} &nbsp;|&nbsp; patient ${{s.patient}}<br/>
      input: ${{s.selected_note}}
    </div>`;
  cards.appendChild(el);

  const img = el.querySelector('#img' + i), slider = el.querySelector('#sl' + i);
  const fnum = el.querySelector('#fn' + i), play = el.querySelector('#pl' + i);
  let frame = s.headline, view = s.default_view, timer = null;
  function draw() {{
    img.src = s.frames[view][frame];
    fnum.textContent = 'frame ' + (frame + 1) + '/' + s.n_frames + '  ·  ' +
      (view === 'input' ? 'input' : view === 'attention' ? 'attention' : view);
    slider.value = frame;
  }}
  function setActive() {{
    el.querySelectorAll('#vw' + i + ' button').forEach(b =>
      b.classList.toggle('active', b.dataset.view === view));
  }}
  slider.addEventListener('input', () => {{ frame = +slider.value; draw(); }});
  el.querySelectorAll('#vw' + i + ' button').forEach(b =>
    b.addEventListener('click', () => {{ view = b.dataset.view; setActive(); draw(); }}));
  play.addEventListener('click', () => {{
    if (timer) {{ clearInterval(timer); timer = null; play.textContent = '▶'; }}
    else {{ play.textContent = '⏸'; timer = setInterval(() => {{
      frame = (frame + 1) % s.n_frames; draw(); }}, 130); }}
  }});
  setActive(); draw();
}});
</script>
</body></html>"""


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="m2_m3_pca_strict", help="Anatomy task (see --list).")
    p.add_argument("--list", action="store_true", help="List available tasks and exit.")
    p.add_argument("--method", default="rollout", choices=("rollout", "attention", "gradcam"),
                   help="Saliency method (default: rollout — class-specific attention rollout).")
    p.add_argument("--n-samples", type=int, default=9)
    p.add_argument("--results", default=str(ANATOMY_RESULTS))
    p.add_argument("--excel", default=str(default_excel_path()))
    p.add_argument("--base-dir", default=str(default_base_dir()))
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--full-data", action="store_true",
                   help="Use the saved full-data probe (optimistic). Debugging only.")
    p.add_argument("--no-saliency", action="store_true", help="Skip saliency (input frames only).")
    p.add_argument("--out", default="runs/viz_model_io")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_experiment(Path(args.results))

    if args.list or args.task not in catalog:
        print("Available tasks in the latest experiment:\n")
        for name, cfg in catalog.items():
            print(f"  {name:24s}  {cfg['model']:22s}  variant={cfg['input_variant']:12s}  "
                  f"classes={','.join(cfg['labels'])}")
        if args.list:
            return 0
        print(f"\nTask '{args.task}' not found. Pick one above.")
        return 1

    cfg = catalog[args.task]
    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    cache = np.load(cfg["cache_path"], allow_pickle=True)
    meta = list(cache["meta"])
    features = cache["mean"]
    task = {t.name: t for t in build_tasks([m["label_text"] for m in meta])}[args.task]
    label_names = cfg["labels"]
    n_classes = len(label_names)

    X = features[task.indices]
    y_true = task.labels
    groups = cache["groups"][task.indices]
    recipe = cfg["recipe"]

    if args.full_data:
        pipeline = joblib.load(cfg["probe_path"])["pipeline"]
        probs_all = probe_scores(pipeline, X, n_classes)
        preds_all = probs_all.argmax(axis=1)
        fold_of = np.zeros(len(y_true), dtype=int)
        fold_pipelines = [pipeline]
        eval_kind = "full-data-fit (optimistic — trained on these same runs)"
    else:
        n_splits = min(args.folds, len(set(groups)), int(np.min(np.bincount(y_true))))
        probs_all, preds_all, fold_of, fold_pipelines = crossval_predictions(
            X, y_true, groups, recipe, n_classes, n_splits, args.seed)
        eval_kind = f"held-out, patient-grouped {n_splits}-fold CV (seed {args.seed})"

    overall_acc = float((preds_all == y_true).mean())
    macro = viz.macro_f1(y_true, preds_all, n_classes)
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, preds_all):
        conf[t, p] += 1
    counts = np.bincount(y_true, minlength=n_classes)
    print(f"Task {args.task}: {len(y_true)} runs, classes {label_names}")
    print(f"  {eval_kind}: accuracy {overall_acc:.3f}, macro-F1 {macro:.3f}")
    print(f"  Saliency method: {args.method}")

    summary_uri = render_summary_figure(
        conf, label_names, counts,
        f"{args.task} — frozen {cfg['model'].split('/')[-1]} @ {cfg['image_size']}px + {recipe}"
        f"   ·   macro-F1 {macro:.3f}, accuracy {overall_acc:.1%}",
        cm_subtitle=f"Confusion — {eval_kind}")

    engine = None
    fold_heads = []
    channel_mean = channel_std = None
    if not args.no_saliency:
        engine = BackboneSaliency(cfg["model"], DINOV2_REVISION, device)
        for probe in fold_pipelines:
            w_eff, _ = probe_linear_head(probe)
            fold_heads.append(torch.tensor(w_eff, dtype=torch.float32, device=device))
        channel_mean = torch.tensor(cfg["image_mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        channel_std = torch.tensor(cfg["image_std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    dicom_index = build_dicom_index(args.excel, args.base_dir)
    chosen = choose_samples(y_true, preds_all, args.n_samples, n_classes, args.seed)

    id_rows = []
    samples = []
    for rank, local_i in enumerate(chosen, 1):
        m = meta[task.indices[local_i]]
        key = (m["accession"], m["run_column"])
        dicom_path = dicom_index.get(key)
        if not dicom_path:
            print(f"  [skip] no DICOM for run")
            continue
        pred, true = int(preds_all[local_i]), int(y_true[local_i])
        scan_uuid = uuid.uuid4()
        short_id = scan_uuid.hex[:8]
        patient = str(m["study_key"])[:8]
        id_rows.append({"uuid": str(scan_uuid), "accession": m["accession"],
                        "run_column": m["run_column"], "study_key": m["study_key"]})
        print(f"  [{rank}/{len(chosen)}] {short_id} true={label_names[true]} pred={label_names[pred]}")

        sample = reconstruct_input(dicom_path, cfg)
        frames_rgb = sample["frames_rgb"]
        frames_np = frames_rgb.numpy()
        n_frames = frames_np.shape[0]
        rgb_mode = not np.allclose(frames_np[:, 0], frames_np[:, 1], atol=1e-4)
        headline = int(frames_np[:, 0].reshape(n_frames, -1).std(axis=1).argmax())

        view_frames = {"input": [frame_uri(frames_np[t], rgb_mode) for t in range(n_frames)]}
        default_view = "input"
        if engine is not None:
            maps = engine.maps(frames_rgb, channel_mean, channel_std,
                               fold_heads[int(fold_of[local_i])], args.method, list(range(n_classes)))
            for key_name, cam in maps.items():
                view = "attention" if key_name == "attention" else label_names[int(key_name[5:])]
                view_frames[view] = [overlay_uri(frames_np[t], cam[t], rgb_mode) for t in range(n_frames)]
            default_view = "attention" if args.method == "attention" else label_names[pred]

        sel = sample["selected_frames"]
        sel_note = (f"{n_frames} frames [{','.join(map(str, sel))}] of {sample['total_frames']}"
                    if cfg["input_variant"] == "top_contrast" else
                    f"{n_frames} frames uniformly from {sample['total_frames']}")
        samples.append({
            "id": short_id, "patient": patient, "run": m["run_column"],
            "label_raw": str(m["label_text"]), "true": label_names[true], "pred": label_names[pred],
            "correct": pred == true, "n_frames": n_frames, "headline": headline,
            "default_view": default_view, "selected_note": sel_note,
            "probs": [{"name": label_names[c], "p": float(probs_all[local_i][c])} for c in range(n_classes)],
            "frames": view_frames,
        })

    with (out / "id_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uuid", "accession", "run_column", "study_key"])
        writer.writeheader()
        writer.writerows(id_rows)

    html_doc = build_html(cfg, args.task, args.method, summary_uri, samples,
                          overall_acc, macro, eval_kind, args.full_data)
    html_path = out / f"model_io_{args.task}.html"
    html_path.write_text(html_doc)
    print(f"\nWrote {len(samples)} interactive cards ({html_path.stat().st_size/1e6:.1f} MB).")
    print(f"Private UUID→accession map: {out / 'id_map.csv'} (keep local; do not publish)")
    print(f"Open: file://{html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
