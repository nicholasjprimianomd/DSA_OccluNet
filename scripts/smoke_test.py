"""End-to-end smoke test: proves the data pipeline + training loop run on this machine.

It generates a tiny synthetic dataset (positives-only, like the current abnormal-only
data), stubs the V-JEPA 2 backbone with a lightweight module (so no ~1GB checkpoint is
downloaded and no GPU is required), then exercises the real loader, preprocessing,
class-weighting, freeze logic, and train/eval loops.

Run:
    python scripts/smoke_test.py

Requires: numpy, pandas, pydicom, openpyxl, torch (CPU is fine). It does NOT require
`transformers` or a GPU, because the backbone is stubbed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_COLS = ["AP_1", "Lateral_1", "AP_2", "Lateral_2", "AP_3", "Lateral_3"]

# Positives-only studies with varied location labels, mirroring the real spreadsheet.
SYNTH_STUDIES = [
    ("ACC001", "study-1", "train", {"AP_1": "L M2", "Lateral_1": "L M2", "AP_2": "R M3"}),
    ("ACC002", "study-2", "train", {"AP_1": "R M2", "Lateral_1": "R M2"}),
    ("ACC003", "study-3", "train", {"AP_1": "L M3", "Lateral_1": "L A2"}),
    ("ACC004", "study-4", "train", {"AP_1": "R P1", "Lateral_1": "L M2 and A3"}),
    ("ACC005", "study-5", "val", {"AP_1": "L M2", "Lateral_1": "R M2", "AP_2": "L M3"}),
    ("ACC006", "study-6", "val", {"AP_1": "R M2", "Lateral_1": "L M2"}),
]


def _make_dcm(path: Path, frames: int = 8, h: int = 32, w: int = 32) -> None:
    arr = (np.random.rand(frames, h, w) * 1000).astype(np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns, ds.NumberOfFrames = h, w, frames
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:  # older pydicom
        ds.save_as(str(path), write_like_original=False)


def _write_synth_dataset(base: Path) -> None:
    rows = []
    for acc, key, split, labels in SYNTH_STUDIES:
        acc_dir = base / acc
        acc_dir.mkdir(parents=True, exist_ok=True)
        row = {"Accession": acc, "MRN": key, "Study_Key": key, "Split": split}
        for col in RUN_COLS:
            if col in labels:
                _make_dcm(acc_dir / f"{col}_img.dcm")
                row[col] = f"{col}_img"
                row[f"{col}_Location"] = labels[col]
            else:
                row[col] = ""
                row[f"{col}_Location"] = ""
            row[f"{col}_Review_Flag"] = ""
        rows.append(row)
    pd.DataFrame(rows).to_excel(base / "AP_Lateral_Labels_Split.xlsx", index=False)


class _DummyBackbone(nn.Module):
    """Stands in for V-JEPA 2: (B,T,3,H,W) -> object with last_hidden_state (B,T,hidden)."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.fc = nn.Linear(3, hidden_size)

    def forward(self, **kwargs):
        x = next(iter(kwargs.values()))
        return types.SimpleNamespace(last_hidden_state=self.fc(x.mean(dim=(3, 4))))


def _run_case(T, view: str, stage: str, freeze: bool, device: torch.device,
              viz_dir=None) -> None:
    args = types.SimpleNamespace(
        excel=str(T.default_excel_path()), base_dir=str(T.default_base_dir()),
        view=view, stage=stage, treat_blank_as_negative=(stage == "binary_detection"),
        batch_size=2, epochs=1, learning_rate=1e-3, weight_decay=1e-4, num_workers=0,
        freeze_backbone=freeze, dry_run=False, save_path="",
    )
    records = T.build_records(args)
    label_names = T.stage_label_names(stage)
    train_records, val_records = T.split_records(records)
    amp = device.type == "cuda"
    class_weights = T.compute_class_weights(train_records, stage, label_names).to(device)

    train_ds = T.BackboneReadyDataset(
        T.build_base_dataset(view, train_records), stage, label_names, args.treat_blank_as_negative
    )
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=T.collate_task_samples)
    model = T.DsaVideoClassifier(num_labels=len(label_names), freeze_backbone=freeze).to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    clip_shape = tuple(train_ds[0].pixel_values.shape)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    metrics = T.train_one_epoch(model, loader, opt, device, class_weights, amp)
    assert np.isfinite(metrics["loss"]), f"non-finite loss for {view}/{stage}"
    assert clip_shape == (T.BACKBONE.clip_length, 3, T.BACKBONE.image_size, T.BACKBONE.image_size)
    if freeze:
        assert trainable < total, "freeze should reduce trainable params"

    # Exercise the visualization path (inputs + outputs) on one case.
    if viz_dir is not None and val_records:
        import viz
        val_ds = T.BackboneReadyDataset(
            T.build_base_dataset(view, val_records), stage, label_names, args.treat_blank_as_negative
        )
        val_loader = DataLoader(val_ds, batch_size=2, collate_fn=T.collate_task_samples)
        sample = T.collate_task_samples([train_ds[i] for i in range(min(4, len(train_ds)))])
        viz.save_input_samples(sample, viz_dir / "inputs", label_names, tag="train")
        out = T.collect_outputs(model, val_loader, device, amp)
        viz.save_confusion_matrix(out["labels"], out["preds"], label_names,
                                  viz_dir / "confusion_matrix.png")
        viz.save_embedding_scatter(out["features"], out["labels"], label_names,
                                   viz_dir / "embeddings_pca.png")
        viz.save_predictions_csv(out["labels"], out["preds"], out["probs"], label_names,
                                 out["metadata"], viz_dir / "predictions.csv")
        made = sorted(p.name for p in viz_dir.rglob("*") if p.is_file())
        print(f"  viz artifacts: {made}")

    print(f"  [{view:<7} {stage:<16} freeze={freeze!s:<5}] dev={device.type} amp={amp} "
          f"clip={clip_shape} trainable={trainable}/{total} loss={metrics['loss']:.3f}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "M2_M3_data"
        _write_synth_dataset(base)
        os.environ["DSA_BASE_DIR"] = str(base)

        import train_dsa_backbone as T
        T.DsaVideoClassifier._load_backbone = staticmethod(lambda name: _DummyBackbone())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)} — training on CUDA")
        else:
            print("No CUDA GPU visible — running on CPU (pipeline still validated)")

        viz_dir = Path(tmp) / "viz"
        print("\nData pipeline + training loop:")
        _run_case(T, "AP", "positive_subtype", freeze=True, device=device, viz_dir=viz_dir)
        _run_case(T, "Lateral", "positive_subtype", freeze=True, device=device)
        _run_case(T, "AP", "positive_subtype", freeze=False, device=device)  # full fine-tune path
        _run_case(T, "AP", "binary_detection", freeze=True, device=device)   # future path (needs normals)

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
