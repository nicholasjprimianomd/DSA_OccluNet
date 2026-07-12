"""End-to-end visual inspection of the DSA occlusion model's inputs and outputs.

This renders, for real DSA runs, *everything the model sees and produces*:

  1. INPUT      the exact frames fed to the frozen backbone (after DICOM windowing,
                temporal frame selection, resize, grayscale->RGB, and — shown side by
                side — ImageNet channel normalization).
  2. LABEL      the ground-truth class for the selected task, plus the raw spreadsheet
                location text it came from.
  3. AUGMENT    the deterministic input construction actually used by the "latest
                experiment" feature source (uniform sampling vs. temporal-change frame
                selection vs. horizontal flip), including which frames were selected.
  4. OUTPUT     the saved probe's predicted class and full class-probability vector.
  5. GRAD-CAM   a class-discriminative heat map (Grad-CAM on the backbone's penultimate
                transformer block, targeting the probe's decision) showing which part of
                each frame drove the prediction.

The "latest experiment" is the anatomy-task study
(``runs/anatomy_tasks/results.json``): frozen **DINOv2-large @ 252 px, 16 frames per
run**, temporal mean pooling, then a fold-selected standardized logistic-regression
probe. This tool reads that results file to discover the tasks, their feature sources,
and their saved full-data probe artifacts, then reproduces the identical preprocessing.

Because the saved probes are fit on all data (an inference artifact, *not* held-out
evidence — see the experiment write-up), predictions on these same runs are optimistic;
the report labels this plainly. The point here is to *see the pipeline*, not to measure
accuracy.

Output is a single self-contained HTML file (all images embedded) plus the per-sample
PNGs, written under ``--out``. It contains patient imaging, so it stays on disk and is
never published.

Example:
    .venv/bin/python visualize_model_io.py --task m2_m3_pca_strict --n-samples 12 \
        --device cuda --out runs/viz_model_io
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import types
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from anatomy_task_experiments import build_tasks
from image_backbone_probe import prepare_model_images, select_top_contrast_frames
from occlusion_loader import build_manifests, default_base_dir, default_excel_path, load_dicom_sequence
from train_dsa_backbone import normalize_sequence, sample_frame_indices, select_device


ANATOMY_RESULTS = Path("runs/anatomy_tasks/results.json")
DINOV2_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


# --------------------------------------------------------------------------------------
# Experiment discovery: read the latest results file to learn tasks + probe + feature src
# --------------------------------------------------------------------------------------
def load_experiment(results_path: Path) -> dict:
    results = json.loads(results_path.read_text())
    feature_sources = results["protocol"]["feature_sources"]
    artifacts = results["final_model_artifacts"]
    catalog = {}
    for task_name, art in artifacts.items():
        source_name = art.get("feature_source")
        if source_name is None:
            continue  # composite/cascade artifacts have no single feature source
        source = feature_sources.get(source_name)
        if source is None:
            continue  # fusion sources (e.g. vjepa_dino) have no single cache; skip
        signature = source.get("signature") or {}
        model_name = signature.get("model", "")
        if "dinov2" not in model_name.lower():
            # This tool reconstructs the DINOv2 per-frame input path. V-JEPA (video)
            # feature sources use a different preprocessing, so skip them rather than
            # display inputs that don't match what that probe was trained on.
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
            "supports_gradcam": "dinov2" in model_name.lower(),
        }
    return catalog


# --------------------------------------------------------------------------------------
# Probe -> effective linear head over raw backbone features (for Grad-CAM targeting)
# --------------------------------------------------------------------------------------
def probe_linear_head(pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Collapse StandardScaler + LogisticRegression into one affine map on raw features.

    Returns (W_eff, b_eff) such that class scores == raw_feature @ W_eff.T + b_eff, so a
    class score is differentiable straight through the frozen backbone during Grad-CAM.
    """
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["logisticregression"]
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    coef = clf.coef_.astype(np.float64)          # (K, D) multiclass, or (1, D) binary
    intercept = clf.intercept_.astype(np.float64)
    if coef.shape[0] == 1:  # binary logreg exposes only the positive class
        coef = np.vstack([-coef[0], coef[0]])
        intercept = np.array([-intercept[0], intercept[0]])
    w_eff = coef / scale                          # (K, D)
    b_eff = intercept - (coef * (mean / scale)).sum(axis=1)
    return w_eff, b_eff


# --------------------------------------------------------------------------------------
# Backbone with Grad-CAM
# --------------------------------------------------------------------------------------
class BackboneCAM:
    def __init__(self, model_name: str, revision: str, device: torch.device):
        from transformers import AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).float().eval()
        # Penultimate transformer block: its patch tokens still mix into the CLS token
        # (which feeds the pooler the probe reads), so gradients flow to a spatial grid.
        self.target_layer = self.model.encoder.layer[-2]
        self._activation: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self._activation = output[0] if isinstance(output, tuple) else output

    def pooled_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """One global feature per frame, matching image_backbone_probe.global_frame_features."""
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        return pooled if pooled is not None else outputs.last_hidden_state[:, 0]

    def gradcam(self, frames_rgb: torch.Tensor, channel_mean: torch.Tensor,
                channel_std: torch.Tensor, w_eff: torch.Tensor, b_eff: torch.Tensor,
                target_class: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-frame Grad-CAM for `target_class` of the temporally mean-pooled probe.

        Returns (cams, frame_influence): cams is (T, H, W) in [0, 1]; frame_influence is
        (T,) each frame's signed contribution to the target class score.
        """
        pixel_values = ((frames_rgb.to(self.device) - channel_mean) / channel_std).requires_grad_(True)
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled_mean = pooled.mean(dim=0)                               # temporal mean pooling
        score = pooled_mean @ w_eff[target_class] + b_eff[target_class]
        activation = self._activation
        grads = torch.autograd.grad(score, activation)[0]             # (T, tokens, hidden)

        patch_grads = grads[:, 1:, :]                                 # drop CLS token
        patch_acts = activation[:, 1:, :].detach()
        n_patches = patch_acts.shape[1]
        side = int(round(n_patches ** 0.5))
        weights = patch_grads.mean(dim=1, keepdim=True)               # (T, 1, hidden)
        cam = F.relu((weights * patch_acts).sum(dim=-1))              # (T, n_patches)
        cam = cam.reshape(cam.shape[0], 1, side, side)
        cam = F.interpolate(cam, size=frames_rgb.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        # Percentile (not min/max) normalization so a single high-norm DINOv2 background
        # "artifact" token can't saturate the whole map.
        lo = torch.quantile(flat, 0.02, dim=1).view(-1, 1, 1)
        hi = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1)
        cam = ((cam - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

        influence = (pooled.detach() @ w_eff[target_class]) / pooled.shape[0]
        return cam.detach().cpu().numpy(), influence.detach().cpu().numpy()


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------
def fig_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def as_display_gray(frame_rgb: np.ndarray) -> np.ndarray:
    """(3,H,W) in [0,1] -> (H,W); channels are replicated grayscale for uniform frames."""
    return frame_rgb[0]


def is_rgb_derived(frames_rgb: np.ndarray) -> bool:
    return not np.allclose(frames_rgb[:, 0], frames_rgb[:, 1], atol=1e-4)


def render_sample_figure(
    sample: dict, cams: np.ndarray | None,
    label_names: list[str], probs: np.ndarray, pred: int, true: int,
) -> str:
    frames = sample["frames_rgb"].numpy()          # (T, 3, H, W) in [0,1]
    n_frames = frames.shape[0]
    rgb_mode = is_rgb_derived(frames)
    correct = pred == true
    # Headline frame = the most vascularized one (largest spatial contrast), so the main
    # Grad-CAM lands on a frame where the anatomy is actually visible.
    content = frames[:, 0].reshape(n_frames, -1).std(axis=1)
    headline = int(np.argmax(content))

    fig = plt.figure(figsize=(13, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.35, 1.15, 1.0])

    title = (
        f"{sample['accession']} · {sample['run_column']}   |   "
        f"true = {label_names[true]}   ·   pred = {label_names[pred]}   "
        f"{'✓ correct' if correct else '✗ wrong'}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold",
                 color="#2a7d3f" if correct else "#b3282d")

    # Row 1: the input frames actually fed to the backbone.
    n_show = min(n_frames, 16)
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, i % 6]) if n_show <= 6 else _sub(fig, gs, 0, i, n_show)
        frame = np.transpose(frames[i], (1, 2, 0)) if rgb_mode else as_display_gray(frames[i])
        ax.imshow(frame, cmap=None if rgb_mode else "gray", vmin=None if rgb_mode else 0,
                  vmax=None if rgb_mode else 1)
        marker = "  ★" if i == headline else ""
        ax.set_title(f"f{i}{marker}", fontsize=7)
        ax.axis("off")

    # Row 2 left: normalization before/after; center: probabilities; skip to gradcam row.
    ax_raw = fig.add_subplot(gs[1, 0])
    mid = n_frames // 2
    ax_raw.imshow(as_display_gray(frames[mid]) if not rgb_mode else np.transpose(frames[mid], (1, 2, 0)),
                  cmap="gray", vmin=0, vmax=1)
    ax_raw.set_title(f"model input\n(windowed, f{mid})", fontsize=8)
    ax_raw.axis("off")

    ax_norm = fig.add_subplot(gs[1, 1])
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    normed = (frames[mid] - mean) / std
    ax_norm.imshow(normed[0], cmap="gray")
    ax_norm.set_title("after channel\nnormalization", fontsize=8)
    ax_norm.axis("off")

    ax_prob = fig.add_subplot(gs[1, 2:4])
    colors = ["#2a7d3f" if k == pred else "#8a97a8" for k in range(len(label_names))]
    bars = ax_prob.barh(range(len(label_names)), probs, color=colors)
    ax_prob.set_yticks(range(len(label_names)), labels=label_names, fontsize=9)
    ax_prob.invert_yaxis()
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("probability", fontsize=8)
    ax_prob.set_title("model output", fontsize=9)
    for k, bar in enumerate(bars):
        ax_prob.text(min(probs[k] + 0.02, 0.9), bar.get_y() + bar.get_height() / 2,
                     f"{probs[k]:.2f}", va="center", fontsize=8)
    # Mark the true class.
    ax_prob.axhline(true, color="#b3282d", lw=0, alpha=0)
    ax_prob.text(1.0, true, "  ← true", color="#b3282d", va="center", fontsize=8)

    # Row 2 right + Row 3: Grad-CAM.
    if cams is not None:
        ax_cam = fig.add_subplot(gs[1, 4:6])
        base = as_display_gray(frames[headline]) if not rgb_mode else np.transpose(frames[headline], (1, 2, 0))
        ax_cam.imshow(base, cmap=None if rgb_mode else "gray", vmin=None if rgb_mode else 0,
                      vmax=None if rgb_mode else 1)
        ax_cam.imshow(cams[headline], cmap="jet", alpha=0.45)
        ax_cam.set_title(f"Grad-CAM · '{label_names[pred]}' · frame f{headline}", fontsize=9)
        ax_cam.axis("off")

        strip = np.linspace(0, n_frames - 1, num=min(n_frames, 6)).round().astype(int)
        for col, i in enumerate(strip):
            ax = fig.add_subplot(gs[2, col])
            base = as_display_gray(frames[i]) if not rgb_mode else np.transpose(frames[i], (1, 2, 0))
            ax.imshow(base, cmap=None if rgb_mode else "gray", vmin=None if rgb_mode else 0,
                      vmax=None if rgb_mode else 1)
            ax.imshow(cams[i], cmap="jet", alpha=0.45)
            ax.set_title(f"CAM f{i}", fontsize=7)
            ax.axis("off")
    else:
        ax_note = fig.add_subplot(gs[1:, 4:6])
        ax_note.axis("off")
        ax_note.text(0.5, 0.5, "Grad-CAM not available\nfor this backbone",
                     ha="center", va="center", fontsize=10, color="#8a97a8")

    return fig_to_data_uri(fig)


def _sub(fig, gs, row, i, n_show):
    # Frames beyond 6 spill onto a shared axis grid by splitting the row into n_show cols.
    subgs = gs[row, :].subgridspec(1, n_show)
    return fig.add_subplot(subgs[0, i])


def render_summary_figure(conf: np.ndarray, label_names: list[str], counts: np.ndarray,
                          title: str) -> str:
    fig, (ax_cm, ax_ct) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    k = len(label_names)
    im = ax_cm.imshow(conf, cmap="Blues")
    ax_cm.set_xticks(range(k), labels=label_names, fontsize=9)
    ax_cm.set_yticks(range(k), labels=label_names, fontsize=9)
    ax_cm.set_xlabel("predicted")
    ax_cm.set_ylabel("true")
    ax_cm.set_title("Confusion (full-data-fit probe — optimistic)", fontsize=10)
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
def build_dicom_index(excel: str, base_dir: str) -> dict[tuple[str, str], str]:
    ap_records, lat_records, _ = build_manifests(excel_path=excel, base_dir=base_dir)
    index = {}
    for record in [*ap_records, *lat_records]:
        index[(record.accession, record.run_column)] = record.dicom_path
    return index


def choose_samples(labels: np.ndarray, preds: np.ndarray, n_samples: int, n_classes: int,
                   seed: int) -> list[int]:
    """Stratify across true classes; within each class prefer to surface a wrong case."""
    rng = np.random.default_rng(seed)
    per_class = max(1, n_samples // n_classes)
    chosen: list[int] = []
    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        wrong = idx[preds[idx] != labels[idx]]
        right = idx[preds[idx] == labels[idx]]
        rng.shuffle(wrong)
        rng.shuffle(right)
        ordered = np.concatenate([wrong, right])[:per_class]
        chosen.extend(int(i) for i in ordered)
    remaining = [i for i in range(len(labels)) if i not in set(chosen)]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, n_samples - len(chosen))])
    return chosen[:n_samples]


def reconstruct_input(dicom_path: str, cfg: dict) -> dict:
    sequence = torch.from_numpy(load_dicom_sequence(dicom_path))
    args = types.SimpleNamespace(
        input_variant=cfg["input_variant"],
        image_size=cfg["image_size"],
        n_frames=cfg["n_frames"],
    )
    frames_rgb = prepare_model_images(sequence, args, dicom_path)  # (T,3,H,W) in [0,1]
    total_frames = int(sequence.shape[0])
    if cfg["input_variant"] == "top_contrast":
        selected = select_top_contrast_frames(normalize_sequence(sequence), cfg["n_frames"]).tolist()
    elif cfg["input_variant"] in ("uniform", "hflip", "border90", "multicrop"):
        selected = sample_frame_indices(total_frames, cfg["n_frames"]).tolist()
    else:
        selected = []
    return {"frames_rgb": frames_rgb, "total_frames": total_frames, "selected_frames": selected}


# --------------------------------------------------------------------------------------
# HTML assembly
# --------------------------------------------------------------------------------------
def build_html(cfg: dict, task_name: str, summary_uri: str, cards: list[dict],
               overall_acc: float) -> str:
    variant_note = {
        "uniform": "Uniform temporal sampling: 16 frames evenly spaced across the run.",
        "top_contrast": "Temporal-change selection: frame 0 plus the frames with the "
                        "largest adjacent-frame change (where contrast moves).",
        "hflip": "Every frame mirrored left-to-right.",
    }.get(cfg["input_variant"], cfg["input_variant"])
    esc = html.escape
    card_html = []
    for card in cards:
        sel = card["selected_frames"]
        sel_txt = (f"frames {sel} of {card['total_frames']}" if sel
                   else f"{card['total_frames']} frames")
        card_html.append(f"""
        <figure class="card">
          <img src="{card['uri']}" alt="sample"/>
          <figcaption>
            <b>{esc(card['accession'])} · {esc(card['run_column'])}</b>
            &nbsp;|&nbsp; study {esc(card['study_key'])}<br/>
            raw label: <code>{esc(card['label_text'])}</code>
            &nbsp;→&nbsp; task class <b>{esc(card['true_name'])}</b><br/>
            selected: {esc(sel_txt)}
          </figcaption>
        </figure>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>DSA model I/O — {esc(task_name)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #0f1216; color: #e6e9ee; }}
  header {{ padding: 22px 28px; background: #161b22; border-bottom: 1px solid #2a323d; }}
  h1 {{ margin: 0 0 6px; font-size: 20px; }}
  .meta {{ color: #9aa4b2; font-size: 13px; line-height: 1.7; }}
  .meta code {{ color: #d6b36a; }}
  main {{ padding: 20px 28px 60px; }}
  .summary img {{ max-width: 100%; background: #fff; border-radius: 8px; }}
  .cards {{ display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 24px; }}
  .card {{ margin: 0; background: #fff; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,.4); }}
  .card img {{ width: 100%; display: block; }}
  figcaption {{ padding: 10px 14px; font-size: 13px; color: #1a1f26; background: #eef1f5;
                line-height: 1.6; }}
  figcaption code {{ background: #dfe4ec; padding: 1px 5px; border-radius: 4px; }}
  .banner {{ background: #3a2e12; color: #f0d9a0; padding: 10px 14px; border-radius: 6px;
             font-size: 13px; margin-top: 14px; }}
</style></head>
<body>
<header>
  <h1>What the DSA occlusion model sees and predicts — task: {esc(task_name)}</h1>
  <div class="meta">
    <b>Latest experiment:</b> anatomy-task probes (<code>runs/anatomy_tasks</code>)<br/>
    <b>Backbone:</b> frozen <code>{esc(cfg['model'])}</code> @ {cfg['image_size']} px,
      {cfg['n_frames']} frames/run, temporal mean pooling<br/>
    <b>Probe:</b> <code>{esc(cfg['recipe'])}</code> on feature source
      <code>{esc(cfg['feature_source'])}</code> · classes: {esc(', '.join(cfg['labels']))}<br/>
    <b>Augmentation / input construction:</b> {esc(variant_note)}<br/>
    <b>Grad-CAM:</b> class-discriminative map on the penultimate transformer block,
      targeting the probe's decision for the predicted class.
  </div>
  <div class="banner">
    Predictions come from the saved <b>full-data-fit</b> probe, so accuracy on these same
    runs ({overall_acc:.1%} over all {cfg['n_samples']} task runs) is optimistic and is
    <b>not</b> held-out performance. This view is for inspecting the pipeline, not scoring it.
  </div>
</header>
<main>
  <div class="summary"><img src="{summary_uri}" alt="summary"/></div>
  <div class="cards">
    {''.join(card_html)}
  </div>
</main>
</body></html>"""


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="m2_m3_pca_strict",
                   help="Anatomy task to visualize (default: m2_m3_pca_strict). Use --list.")
    p.add_argument("--list", action="store_true", help="List available tasks and exit.")
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--results", default=str(ANATOMY_RESULTS))
    p.add_argument("--excel", default=str(default_excel_path()))
    p.add_argument("--base-dir", default=str(default_base_dir()))
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM (faster).")
    p.add_argument("--out", default="runs/viz_model_io")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_experiment(Path(args.results))

    if args.list or args.task not in catalog:
        print("Available tasks in the latest experiment:\n")
        for name, cfg in catalog.items():
            cam = "Grad-CAM ✓" if cfg["supports_gradcam"] else "no Grad-CAM (video backbone)"
            print(f"  {name:24s}  {cfg['model'] or '(fusion)':28s}  "
                  f"variant={cfg['input_variant']:12s}  {cam}")
        if args.list:
            return 0
        print(f"\nTask '{args.task}' not found or unsupported. Pick one above.")
        return 1

    cfg = catalog[args.task]
    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    # Cached features + metadata (fixed sample order) and the task's membership/labels.
    cache = np.load(cfg["cache_path"], allow_pickle=True)
    meta = list(cache["meta"])
    features = cache["mean"]
    task = {t.name: t for t in build_tasks([m["label_text"] for m in meta])}[args.task]
    label_names = cfg["labels"]
    n_classes = len(label_names)

    # Saved probe -> predictions/probabilities on the exact cached features it was fit on.
    artifact = joblib.load(cfg["probe_path"])
    pipeline = artifact["pipeline"]
    X = features[task.indices]
    y_true = task.labels
    probs_all = pipeline.predict_proba(X)
    preds_all = probs_all.argmax(axis=1)
    overall_acc = float((preds_all == y_true).mean())

    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, preds_all):
        conf[t, p] += 1
    counts = np.bincount(y_true, minlength=n_classes)
    print(f"Task {args.task}: {len(y_true)} runs, classes {label_names}, "
          f"full-data-fit accuracy {overall_acc:.3f}")

    summary_uri = render_summary_figure(
        conf, label_names, counts,
        f"{args.task} — frozen {cfg['model'].split('/')[-1]} @ {cfg['image_size']}px + {cfg['recipe']}")

    # Grad-CAM engine + effective linear head.
    cam_engine = None
    w_eff_t = b_eff_t = channel_mean = channel_std = None
    if cfg["supports_gradcam"] and not args.no_gradcam:
        cam_engine = BackboneCAM(cfg["model"], DINOV2_REVISION, device)
        w_eff, b_eff = probe_linear_head(pipeline)
        w_eff_t = torch.tensor(w_eff, dtype=torch.float32, device=device)
        b_eff_t = torch.tensor(b_eff, dtype=torch.float32, device=device)
        channel_mean = torch.tensor(cfg["image_mean"], dtype=torch.float32,
                                    device=device).view(1, 3, 1, 1)
        channel_std = torch.tensor(cfg["image_std"], dtype=torch.float32,
                                   device=device).view(1, 3, 1, 1)

    dicom_index = build_dicom_index(args.excel, args.base_dir)
    chosen = choose_samples(y_true, preds_all, args.n_samples, n_classes, args.seed)

    cards = []
    for rank, local_i in enumerate(chosen, 1):
        row = task.indices[local_i]
        m = meta[row]
        key = (m["accession"], m["run_column"])
        dicom_path = dicom_index.get(key)
        if not dicom_path:
            print(f"  [skip] no DICOM for {key}")
            continue
        pred = int(preds_all[local_i])
        true = int(y_true[local_i])
        probs = probs_all[local_i]
        print(f"  [{rank}/{len(chosen)}] {key} true={label_names[true]} pred={label_names[pred]}")

        sample = reconstruct_input(dicom_path, cfg)
        cams = None
        if cam_engine is not None:
            cams, _ = cam_engine.gradcam(
                sample["frames_rgb"], channel_mean, channel_std, w_eff_t, b_eff_t, pred)

        uri = render_sample_figure(
            {"frames_rgb": sample["frames_rgb"], "accession": m["accession"],
             "run_column": m["run_column"]},
            cams, label_names, probs, pred, true)

        # Also drop the standalone PNG for offline viewing.
        png_path = out / f"sample_{rank:02d}_{m['accession']}_{m['run_column']}.png"
        png_path.write_bytes(base64.b64decode(uri.split(",", 1)[1]))

        cards.append({
            "uri": uri, "accession": m["accession"], "run_column": m["run_column"],
            "study_key": m["study_key"], "label_text": m["label_text"],
            "true_name": label_names[true],
            "total_frames": sample["total_frames"], "selected_frames": sample["selected_frames"],
        })

    html_doc = build_html(cfg, args.task, summary_uri, cards, overall_acc)
    html_path = out / f"model_io_{args.task}.html"
    html_path.write_text(html_doc)
    print(f"\nWrote {len(cards)} sample cards.")
    print(f"Open: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
