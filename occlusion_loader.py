from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pydicom

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - handled at runtime when dataset classes are used
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        pass


DEFAULT_COLAB_EXCEL_PATH = Path("/content/drive/MyDrive/M2_M3_data/AP_Lateral_Labels_Split.xlsx")
DEFAULT_COLAB_BASE_DIR = Path("/content/drive/MyDrive/M2_M3_data")
DEFAULT_LOCAL_EXCEL_PATH = Path(r"H:\My Drive\M2_M3_data\AP_Lateral_Labels_Split.xlsx")
DEFAULT_LOCAL_BASE_DIR = Path(r"H:\My Drive\M2_M3_data")

VIEW_COLUMNS: dict[str, list[str]] = {
    "AP": ["AP_1", "AP_2", "AP_3"],
    "Lateral": ["Lateral_1", "Lateral_2", "Lateral_3"],
}
RUN_COLUMNS = [column for columns in VIEW_COLUMNS.values() for column in columns]
MANIFEST_COLUMNS = [
    "row_index",
    "study_key",
    "accession",
    "view",
    "run_column",
    "run_index",
    "dicom_value",
    "dicom_path",
    "label_text",
    "location_column",
    "review_flag",
    "split",
    "resolved",
    "missing_reason",
]
SPLIT_COLUMN_CANDIDATES = (
    "Split",
    "split",
    "Dataset_Split",
    "dataset_split",
    "Partition",
    "partition",
)


@dataclass(frozen=True)
class OcclusionRecord:
    row_index: int
    study_key: str
    accession: str
    view: str
    run_column: str
    run_index: int
    dicom_value: str
    dicom_path: str
    label_text: str
    location_column: str
    review_flag: str = ""
    split: str = ""
    resolved: bool = True
    missing_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def running_in_colab() -> bool:
    return "google.colab" in sys.modules


def default_excel_path() -> Path:
    return DEFAULT_COLAB_EXCEL_PATH if running_in_colab() else DEFAULT_LOCAL_EXCEL_PATH


def default_base_dir() -> Path:
    return DEFAULT_COLAB_BASE_DIR if running_in_colab() else DEFAULT_LOCAL_BASE_DIR


def normalize_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def location_column_for(column: str) -> str:
    return f"{column}_Location"


def review_flag_column_for(column: str) -> str:
    return f"{column}_Review_Flag"


def candidate_paths(base_dir: Path, accession: str, image_value: str) -> list[Path]:
    accession_dir = base_dir / accession
    if image_value.lower().endswith(".dcm"):
        return [accession_dir / image_value]
    return [accession_dir / f"{image_value}.dcm", accession_dir / image_value]


def resolve_dicom_path(base_dir: Path, accession: str, image_value: str) -> Path | None:
    return next((path for path in candidate_paths(base_dir, accession, image_value) if path.exists()), None)


def detect_split_column(columns: Sequence[str]) -> str | None:
    for candidate in SPLIT_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def validate_inputs(excel_path: Path, base_dir: Path) -> None:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")


def validate_dataframe_columns(df: pd.DataFrame, require_location_columns: bool = True) -> None:
    required_columns = {"Accession", *RUN_COLUMNS}
    if require_location_columns:
        required_columns.update(location_column_for(column) for column in RUN_COLUMNS)

    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required column(s): {missing_list}")


def _new_view_summary() -> dict[str, Any]:
    return {
        "labeled_entries": 0,
        "resolved_samples": 0,
        "missing_samples": 0,
        "distinct_labels": set(),
        "missing_examples": [],
    }


def _append_missing_example(summary: dict[str, Any], record: OcclusionRecord) -> None:
    if len(summary["missing_examples"]) >= 5:
        return
    summary["missing_examples"].append(
        {
            "study_key": record.study_key,
            "accession": record.accession,
            "run_column": record.run_column,
            "dicom_value": record.dicom_value,
            "label_text": record.label_text,
            "missing_reason": record.missing_reason,
        }
    )


def build_manifests(
    excel_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    study_key_column: str = "Study_Key",
    include_missing: bool = False,
) -> tuple[list[OcclusionRecord], list[OcclusionRecord], dict[str, Any]]:
    excel_path = Path(excel_path) if excel_path is not None else default_excel_path()
    base_dir = Path(base_dir) if base_dir is not None else default_base_dir()
    validate_inputs(excel_path, base_dir)

    df = pd.read_excel(excel_path)
    validate_dataframe_columns(df, require_location_columns=True)

    split_column = detect_split_column(df.columns)
    missing_study_key_count = 0
    summaries = {view: _new_view_summary() for view in VIEW_COLUMNS}
    resolved_by_view: dict[str, list[OcclusionRecord]] = {view: [] for view in VIEW_COLUMNS}
    missing_by_view: dict[str, list[OcclusionRecord]] = {view: [] for view in VIEW_COLUMNS}

    for row_index, row in df.iterrows():
        accession = normalize_value(row.get("Accession"))
        if not accession:
            continue

        study_key = normalize_value(row.get(study_key_column))
        if not study_key:
            study_key = accession
            missing_study_key_count += 1

        for view, columns in VIEW_COLUMNS.items():
            for run_index, run_column in enumerate(columns, start=1):
                label_text = normalize_value(row.get(location_column_for(run_column)))
                if not label_text:
                    continue

                summaries[view]["labeled_entries"] += 1
                summaries[view]["distinct_labels"].add(label_text)
                dicom_value = normalize_value(row.get(run_column))
                resolved_path = resolve_dicom_path(base_dir, accession, dicom_value) if dicom_value else None

                missing_reason = ""
                if not dicom_value:
                    missing_reason = "missing_dicom_value"
                elif resolved_path is None:
                    missing_reason = "file_not_found"

                record = OcclusionRecord(
                    row_index=int(row_index),
                    study_key=study_key,
                    accession=accession,
                    view=view,
                    run_column=run_column,
                    run_index=run_index,
                    dicom_value=dicom_value,
                    dicom_path=str(resolved_path) if resolved_path is not None else "",
                    label_text=label_text,
                    location_column=location_column_for(run_column),
                    review_flag=normalize_value(row.get(review_flag_column_for(run_column))),
                    split=normalize_value(row.get(split_column)) if split_column else "",
                    resolved=resolved_path is not None,
                    missing_reason=missing_reason,
                )

                if record.resolved:
                    resolved_by_view[view].append(record)
                    summaries[view]["resolved_samples"] += 1
                else:
                    missing_by_view[view].append(record)
                    summaries[view]["missing_samples"] += 1
                    _append_missing_example(summaries[view], record)

    ap_records = list(resolved_by_view["AP"])
    lateral_records = list(resolved_by_view["Lateral"])
    if include_missing:
        ap_records.extend(missing_by_view["AP"])
        lateral_records.extend(missing_by_view["Lateral"])

    summary = {
        "excel_path": str(excel_path),
        "base_dir": str(base_dir),
        "study_key_column": study_key_column,
        "split_column": split_column or "",
        "missing_study_key_count": missing_study_key_count,
        "total_resolved_samples": len(resolved_by_view["AP"]) + len(resolved_by_view["Lateral"]),
        "total_missing_samples": len(missing_by_view["AP"]) + len(missing_by_view["Lateral"]),
        "views": {},
    }

    for view, view_summary in summaries.items():
        summary["views"][view] = {
            "labeled_entries": view_summary["labeled_entries"],
            "resolved_samples": view_summary["resolved_samples"],
            "missing_samples": view_summary["missing_samples"],
            "distinct_labels": sorted(view_summary["distinct_labels"]),
            "missing_examples": view_summary["missing_examples"],
        }

    return ap_records, lateral_records, summary


def build_label_mapping(records: Sequence[OcclusionRecord]) -> dict[str, int]:
    labels = sorted({record.label_text for record in records})
    return {label: index for index, label in enumerate(labels)}


def load_dicom_sequence(path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(Path(path))
    sequence = ds.pixel_array.astype(np.float32)

    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, ...]
    elif sequence.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D DICOM pixel array, got shape {sequence.shape} from {path}"
        )

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    sequence = sequence * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        sequence = np.max(sequence) - sequence
    return sequence


def _ensure_torch_available() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the dataset classes. Install it with: pip install torch"
        )


class BaseOcclusionDataset(TorchDataset):
    def __init__(
        self,
        records: Sequence[OcclusionRecord],
        view: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        _ensure_torch_available()

        if any(not record.resolved for record in records):
            raise ValueError("Dataset records must all be resolved. Use build_manifests(...) output.")

        self.records = list(records)
        self.view = view
        self.transform = transform
        self.label_to_index = label_to_index or build_label_mapping(self.records)

        for record in self.records:
            if record.view != view:
                raise ValueError(f"Expected only {view} records, got {record.view} in {record.run_column}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sequence = load_dicom_sequence(record.dicom_path)
        sequence_tensor = torch.from_numpy(sequence)
        label_id = self.label_to_index.get(record.label_text)

        sample = {
            "sequence": sequence_tensor,
            "label_text": record.label_text,
            "label_id": label_id,
            "study_key": record.study_key,
            "accession": record.accession,
            "view": record.view,
            "run_column": record.run_column,
            "run_index": record.run_index,
            "dicom_path": record.dicom_path,
            "metadata": record.to_dict(),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class APOcclusionDataset(BaseOcclusionDataset):
    def __init__(
        self,
        records: Sequence[OcclusionRecord],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        super().__init__(records=records, view="AP", transform=transform, label_to_index=label_to_index)


class LateralOcclusionDataset(BaseOcclusionDataset):
    def __init__(
        self,
        records: Sequence[OcclusionRecord],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            records=records,
            view="Lateral",
            transform=transform,
            label_to_index=label_to_index,
        )


def records_to_dataframe(records: Sequence[OcclusionRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.DataFrame([record.to_dict() for record in records], columns=MANIFEST_COLUMNS)


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Excel path: {summary['excel_path']}",
        f"Base dir: {summary['base_dir']}",
    ]
    if summary["split_column"]:
        lines.append(f"Detected split column: {summary['split_column']}")
    if summary["missing_study_key_count"]:
        lines.append(
            f"Missing Study_Key values replaced with Accession: {summary['missing_study_key_count']}"
        )

    for view in ("AP", "Lateral"):
        view_summary = summary["views"][view]
        labels = ", ".join(view_summary["distinct_labels"]) if view_summary["distinct_labels"] else "(none)"
        lines.extend(
            [
                f"{view}: {view_summary['resolved_samples']} resolved / {view_summary['labeled_entries']} labeled",
                f"{view} missing labeled runs: {view_summary['missing_samples']}",
                f"{view} labels: {labels}",
            ]
        )
        for missing_example in view_summary["missing_examples"]:
            lines.append(
                "Missing labeled run: "
                f"{view} study_key={missing_example['study_key']} "
                f"accession={missing_example['accession']} "
                f"column={missing_example['run_column']} "
                f"dicom={missing_example['dicom_value'] or '(blank)'} "
                f"label={missing_example['label_text']} "
                f"reason={missing_example['missing_reason']}"
            )
    lines.append(f"Total missing labeled samples: {summary['total_missing_samples']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build validated AP and lateral occlusion manifests from the spreadsheet and report summary counts."
        )
    )
    parser.add_argument(
        "--excel",
        default=str(default_excel_path()),
        help="Path to the Excel file with AP/Lateral labels.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(default_base_dir()),
        help="Base directory containing accession folders.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional path to write the combined resolved manifest CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ap_records, lateral_records, summary = build_manifests(
        excel_path=args.excel,
        base_dir=args.base_dir,
    )
    print(format_summary(summary))

    if args.out_csv:
        out_csv = Path(args.out_csv)
        manifest_df = records_to_dataframe([*ap_records, *lateral_records])
        manifest_df.to_csv(out_csv, index=False)
        print(f"Wrote {len(manifest_df)} resolved samples to {out_csv}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
