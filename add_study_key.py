import argparse
import uuid
from pathlib import Path

import pandas as pd


def normalize_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a Study_Key column with UUIDs to the Excel file."
    )
    parser.add_argument(
        "--excel",
        default=r"H:\My Drive\M2_M3_data\Accession_MRN_AP_Lateral_Labels_Split.xlsx",
        help="Path to the Excel file to update.",
    )
    parser.add_argument(
        "--key-column",
        default="Study_Key",
        help="Name of the key column to add or update.",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if "Accession" not in df.columns:
        raise KeyError("Missing required column: Accession")

    key_col = args.key_column
    if key_col not in df.columns:
        df[key_col] = ""

    # Build a stable mapping of accession -> key (reuse existing when present)
    accession_to_key: dict[str, str] = {}
    for _, row in df.iterrows():
        accession = normalize_value(row.get("Accession"))
        if not accession:
            continue
        existing_key = normalize_value(row.get(key_col))
        if existing_key:
            accession_to_key.setdefault(accession, existing_key)

    # Fill missing keys with UUIDs
    for idx, row in df.iterrows():
        accession = normalize_value(row.get("Accession"))
        if not accession:
            continue
        if accession not in accession_to_key:
            accession_to_key[accession] = str(uuid.uuid4())
        df.at[idx, key_col] = accession_to_key[accession]

    df.to_excel(excel_path, index=False)
    print(f"Updated {excel_path} with column {key_col}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
