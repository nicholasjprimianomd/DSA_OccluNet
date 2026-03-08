from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from occlusion_loader import (
    APOcclusionDataset,
    LateralOcclusionDataset,
    OcclusionRecord,
    build_manifests,
    default_base_dir,
    default_excel_path,
    detect_split_column,
    location_column_for,
    normalize_value,
    resolve_dicom_path,
    review_flag_column_for,
    validate_dataframe_columns,
    validate_inputs,
    VIEW_COLUMNS,
)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    pretrained_name: str
    input_key: str
    clip_length: int
    image_size: int


BACKBONE = BackboneSpec(
    name="vjepa2",
    pretrained_name="facebook/vjepa2-vitl-fpc64-256",
    input_key="pixel_values_videos",
    clip_length=64,
    image_size=256,
)

TRAINING_STAGES = ("positive_subtype", "binary_detection")
VIEW_CHOICES = tuple(VIEW_COLUMNS.keys())
POSITIVE_ONLY_LABELS = ("m2", "m3", "other_positive")


@dataclass(frozen=True)
class TaskSample:
    pixel_values: torch.Tensor
    label: int
    label_name: str
    metadata: dict[str, Any]


def sample_frame_indices(num_frames: int, clip_length: int) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError("Sequence must contain at least one frame.")
    if num_frames == 1:
        return torch.zeros(clip_length, dtype=torch.long)
    return torch.linspace(0, num_frames - 1, steps=clip_length).round().to(dtype=torch.long)


def normalize_sequence(sequence: torch.Tensor) -> torch.Tensor:
    sequence = sequence.to(dtype=torch.float32)
    flattened = sequence.flatten()
    max_quantile_samples = 262144
    if flattened.numel() > max_quantile_samples:
        step = max(1, flattened.numel() // max_quantile_samples)
        sample_indices = torch.arange(
            0,
            flattened.numel(),
            step,
            device=flattened.device,
            dtype=torch.long,
        )[:max_quantile_samples]
        quantile_source = flattened.index_select(0, sample_indices)
    else:
        quantile_source = flattened

    lower = torch.quantile(quantile_source, 0.01)
    upper = torch.quantile(quantile_source, 0.99)
    if torch.isclose(upper, lower):
        max_value = sequence.max()
        min_value = sequence.min()
        if torch.isclose(max_value, min_value):
            return torch.zeros_like(sequence, dtype=torch.float32)
        sequence = (sequence - min_value) / (max_value - min_value)
    else:
        sequence = (sequence - lower) / (upper - lower)
    return sequence.clamp(0.0, 1.0)


def preprocess_clip(sequence: torch.Tensor, spec: BackboneSpec) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence shape (T, H, W), got {tuple(sequence.shape)}")

    sequence = normalize_sequence(sequence)
    frame_indices = sample_frame_indices(sequence.shape[0], spec.clip_length)
    clip = sequence.index_select(0, frame_indices)
    clip = clip.unsqueeze(1)
    clip = F.interpolate(
        clip,
        size=(spec.image_size, spec.image_size),
        mode="bilinear",
        align_corners=False,
    )
    clip = clip.repeat(1, 3, 1, 1)
    return clip


def normalize_location_label(label_text: str) -> str:
    return re.sub(r"\s+", "", label_text.strip().upper())


def positive_subtype_from_label(label_text: str) -> str:
    compact = normalize_location_label(label_text)
    if "M2" in compact:
        return "m2"
    if "M3" in compact:
        return "m3"
    return "other_positive"


def binary_target_from_label(
    label_text: str,
    treat_blank_as_negative: bool,
) -> tuple[bool, str]:
    cleaned = normalize_value(label_text)
    if cleaned:
        return True, "positive"
    if treat_blank_as_negative:
        return True, "negative"
    return False, ""


class BackboneReadyDataset(Dataset[TaskSample]):
    def __init__(
        self,
        base_dataset: Dataset[dict[str, Any]],
        stage: str,
        label_names: Sequence[str],
        treat_blank_as_negative: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.spec = BACKBONE
        self.stage = stage
        self.label_names = list(label_names)
        self.label_to_index = {name: index for index, name in enumerate(self.label_names)}
        self.treat_blank_as_negative = treat_blank_as_negative

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> TaskSample:
        sample = self.base_dataset[index]
        sequence = sample["sequence"]
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence)

        if self.stage == "positive_subtype":
            label_name = positive_subtype_from_label(sample["label_text"])
        else:
            is_valid, label_name = binary_target_from_label(
                sample["label_text"],
                treat_blank_as_negative=self.treat_blank_as_negative,
            )
            if not is_valid:
                raise ValueError("Binary detection sample missing a usable label.")

        return TaskSample(
            pixel_values=preprocess_clip(sequence, self.spec),
            label=self.label_to_index[label_name],
            label_name=label_name,
            metadata=sample["metadata"],
        )


class DsaVideoClassifier(nn.Module):
    def __init__(self, num_labels: int, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.spec = BACKBONE
        self.backbone = self._load_backbone(self.spec.pretrained_name)
        hidden_size = self._hidden_size(self.backbone)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _load_backbone(pretrained_name: str) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise ImportError(
                "transformers is required for the training backbone. Install it with: pip install transformers"
            ) from exc

        return AutoModel.from_pretrained(pretrained_name)

    @staticmethod
    def _hidden_size(backbone: nn.Module) -> int:
        config = getattr(backbone, "config", None)
        for attribute in ("hidden_size", "projection_dim", "embed_dim"):
            value = getattr(config, attribute, None)
            if isinstance(value, int):
                return value
        raise ValueError("Unable to infer hidden size from backbone config.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(**{self.spec.input_key: pixel_values})
        hidden_state = outputs.last_hidden_state
        pooled = hidden_state.mean(dim=1)
        return self.classifier(pooled)


def collate_task_samples(batch: Sequence[TaskSample]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item.pixel_values for item in batch], dim=0),
        "labels": torch.tensor([item.label for item in batch], dtype=torch.long),
        "label_names": [item.label_name for item in batch],
        "metadata": [item.metadata for item in batch],
    }


def build_base_dataset(view: str, records: Sequence[OcclusionRecord]) -> Dataset[dict[str, Any]]:
    if view == "AP":
        return APOcclusionDataset(records)
    if view == "Lateral":
        return LateralOcclusionDataset(records)
    raise ValueError(f"Unsupported view: {view}")


def detection_records_from_spreadsheet(
    excel_path: str | Path,
    base_dir: str | Path,
    view: str,
    treat_blank_as_negative: bool,
    study_key_column: str = "Study_Key",
) -> list[OcclusionRecord]:
    excel_path = Path(excel_path)
    base_dir = Path(base_dir)
    validate_inputs(excel_path, base_dir)

    df = pd.read_excel(excel_path)
    validate_dataframe_columns(df, require_location_columns=True)
    split_column = detect_split_column(df.columns)

    records: list[OcclusionRecord] = []
    for row_index, row in df.iterrows():
        accession = normalize_value(row.get("Accession"))
        if not accession:
            continue

        study_key = normalize_value(row.get(study_key_column)) or accession
        for run_index, run_column in enumerate(VIEW_COLUMNS[view], start=1):
            label_text = normalize_value(row.get(location_column_for(run_column)))
            if not label_text and not treat_blank_as_negative:
                continue

            dicom_value = normalize_value(row.get(run_column))
            resolved_path = resolve_dicom_path(base_dir, accession, dicom_value) if dicom_value else None
            if resolved_path is None:
                continue

            records.append(
                OcclusionRecord(
                    row_index=int(row_index),
                    study_key=study_key,
                    accession=accession,
                    view=view,
                    run_column=run_column,
                    run_index=run_index,
                    dicom_value=dicom_value,
                    dicom_path=str(resolved_path),
                    label_text=label_text,
                    location_column=location_column_for(run_column),
                    review_flag=normalize_value(row.get(review_flag_column_for(run_column))),
                    split=normalize_value(row.get(split_column)) if split_column else "",
                    resolved=True,
                )
            )
    return records


def filter_records_for_stage(
    records: Sequence[OcclusionRecord],
    stage: str,
    treat_blank_as_negative: bool,
) -> list[OcclusionRecord]:
    filtered: list[OcclusionRecord] = []
    for record in records:
        if stage == "positive_subtype":
            filtered.append(record)
            continue

        is_valid, _ = binary_target_from_label(
            record.label_text,
            treat_blank_as_negative=treat_blank_as_negative,
        )
        if is_valid:
            filtered.append(record)
    return filtered


def split_records(records: Sequence[OcclusionRecord]) -> tuple[list[OcclusionRecord], list[OcclusionRecord]]:
    train_records: list[OcclusionRecord] = []
    val_records: list[OcclusionRecord] = []
    for record in records:
        split_value = normalize_value(record.split).lower()
        if split_value in {"val", "valid", "validation", "dev"}:
            val_records.append(record)
        else:
            train_records.append(record)
    return train_records, val_records


def stage_label_names(stage: str) -> list[str]:
    if stage == "positive_subtype":
        return list(POSITIVE_ONLY_LABELS)
    return ["negative", "positive"]


def summarize_stage(records: Sequence[OcclusionRecord], stage: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if stage == "positive_subtype":
            label_name = positive_subtype_from_label(record.label_text)
        else:
            label_name = "positive" if normalize_value(record.label_text) else "negative"
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def evaluate(model: nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.size(0))

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * labels.size(0)
        total_examples += int(labels.size(0))
        total_correct += int((predictions == labels).sum().item())

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def build_records(args: argparse.Namespace) -> list[OcclusionRecord]:
    if args.stage == "binary_detection":
        records = detection_records_from_spreadsheet(
            excel_path=args.excel,
            base_dir=args.base_dir,
            view=args.view,
            treat_blank_as_negative=args.treat_blank_as_negative,
        )
    else:
        ap_records, lateral_records, _ = build_manifests(
            excel_path=args.excel,
            base_dir=args.base_dir,
        )
        records = ap_records if args.view == "AP" else lateral_records

    records = filter_records_for_stage(
        records,
        stage=args.stage,
        treat_blank_as_negative=args.treat_blank_as_negative,
    )
    if not records:
        raise ValueError("No usable records found for the selected training stage and view.")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a DSA video backbone using the AP/lateral occlusion loader with V-JEPA 2."
        )
    )
    parser.add_argument("--excel", default=str(default_excel_path()), help="Path to the Excel sheet.")
    parser.add_argument("--base-dir", default=str(default_base_dir()), help="Base directory for accession folders.")
    parser.add_argument("--view", choices=VIEW_CHOICES, required=True, help="Train either the AP or lateral model.")
    parser.add_argument(
        "--stage",
        choices=TRAINING_STAGES,
        default="positive_subtype",
        help="Positive-only pipeline validation now, or binary detection once negatives are available.",
    )
    parser.add_argument(
        "--treat-blank-as-negative",
        action="store_true",
        help="For binary detection, interpret blank location labels as negatives once that annotation policy is valid.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the encoder and train only the task head. Useful for the positive-only warm-up stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the datasets and model, print shapes and label counts, then stop before training.",
    )
    parser.add_argument(
        "--save-path",
        default="",
        help="Optional path to save the trained model checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = BACKBONE
    records = build_records(args)
    label_names = stage_label_names(args.stage)
    label_counts = summarize_stage(records, args.stage)
    train_records, val_records = split_records(records)

    if args.stage == "binary_detection" and label_counts.get("negative", 0) == 0:
        raise ValueError(
            "Binary detection requires negatives. Re-run with --treat-blank-as-negative once blank labels are valid negatives."
        )

    base_train_dataset = build_base_dataset(args.view, train_records)
    train_dataset = BackboneReadyDataset(
        base_dataset=base_train_dataset,
        stage=args.stage,
        label_names=label_names,
        treat_blank_as_negative=args.treat_blank_as_negative,
    )

    val_dataset: BackboneReadyDataset | None = None
    if val_records:
        base_val_dataset = build_base_dataset(args.view, val_records)
        val_dataset = BackboneReadyDataset(
            base_dataset=base_val_dataset,
            stage=args.stage,
            label_names=label_names,
            treat_blank_as_negative=args.treat_blank_as_negative,
        )

    print(f"Backbone: {spec.pretrained_name}")
    print(f"View: {args.view}")
    print(f"Stage: {args.stage}")
    print(f"Clip length: {spec.clip_length}")
    print(f"Image size: {spec.image_size}")
    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Label counts: {label_counts}")

    preview = train_dataset[0]
    print(f"Preview clip shape: {tuple(preview.pixel_values.shape)}")
    print(f"Preview label: {preview.label_name}")

    if args.dry_run:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_task_samples,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_task_samples,
        )
        if val_dataset is not None
        else None
    )

    model = DsaVideoClassifier(
        num_labels=len(label_names),
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f}"
        )
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch}: val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "backbone": BACKBONE.name,
                "view": args.view,
                "stage": args.stage,
                "label_names": label_names,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
