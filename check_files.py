import argparse
from pathlib import Path

import pandas as pd


def normalize_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def candidate_paths(base_dir: Path, accession: str, ap_value: str) -> list[Path]:
    accession_dir = base_dir / accession
    if ap_value.lower().endswith(".dcm"):
        return [accession_dir / ap_value]
    return [accession_dir / f"{ap_value}.dcm", accession_dir / ap_value]


def dcm_display_path(base_dir: Path, accession: str, ap_value: str) -> Path:
    accession_dir = base_dir / accession
    if ap_value.lower().endswith(".dcm"):
        return accession_dir / ap_value
    return accession_dir / f"{ap_value}.dcm"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AP/Lateral dicom files for each accession listed in an Excel sheet."
        )
    )
    parser.add_argument(
        "--excel",
        default=r"H:\My Drive\M2_M3_data\Accession_MRN_AP_Lateral_Labels_Split.xlsx",
        help="Path to the Excel file.",
    )
    parser.add_argument(
        "--base-dir",
        default=r"H:\My Drive\M2_M3_data",
        help="Base directory containing accession folders.",
    )
    parser.add_argument(
        "--out-csv",
        default="found_files.csv",
        help="Path to write the list of found .dcm filenames.",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    base_dir = Path(args.base_dir)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    df = pd.read_excel(excel_path)
    columns_to_check = [
        "AP_1",
        "AP_2",
        "AP_3",
        "Lateral_1",
        "Lateral_2",
        "Lateral_3",
    ]
    required_columns = {"Accession", *columns_to_check}
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required column(s): {missing_list}")

    error_count = 0
    checked_count = 0
    out_rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        accession = normalize_value(row.get("Accession"))
        if not accession:
            continue
        out_row: dict[str, str] = {"Accession": accession}
        for col in columns_to_check:
            out_row[col] = ""

        for column in columns_to_check:
            image_value = normalize_value(row.get(column))
            if not image_value:
                continue
            checked_count += 1
            paths = candidate_paths(base_dir, accession, image_value)
            display_path = dcm_display_path(base_dir, accession, image_value)
            display_name = display_path.name
            if not any(path.exists() for path in paths):
                print(
                    "ERROR: Missing file for Accession "
                    f"{accession} ({column}): {display_name}"
                )
                error_count += 1
            else:
                out_row[column] = display_name

        out_rows.append(out_row)

    if out_rows:
        out_csv = Path(args.out_csv)
        out_df = pd.DataFrame(out_rows)
        out_df = out_df[["Accession", "AP_1", "AP_2", "AP_3", "Lateral_1", "Lateral_2", "Lateral_3"]]
        out_df.to_csv(out_csv, index=False)
        found_count = sum(1 for r in out_rows for c in columns_to_check if r.get(c))
        print(f"Wrote {len(out_rows)} rows ({found_count} found filenames) to {out_csv}.")
    print(f"Checked {checked_count} AP/Lateral entries.")
    if error_count:
        print(f"Found {error_count} missing file(s).")
        return 1
    print("All files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
