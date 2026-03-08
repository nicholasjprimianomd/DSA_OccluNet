import argparse
from pathlib import Path

from occlusion_loader import (
    build_manifests,
    default_base_dir,
    default_excel_path,
    format_summary,
    records_to_dataframe,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate labeled AP/Lateral occlusion runs and optionally export the resolved manifest."
        )
    )
    parser.add_argument(
        "--excel",
        default=str(default_excel_path()),
        help="Path to the Excel file.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(default_base_dir()),
        help="Base directory containing accession folders.",
    )
    parser.add_argument(
        "--out-csv",
        default="found_files.csv",
        help="Path to write the resolved manifest as CSV.",
    )
    args = parser.parse_args()

    ap_records, lateral_records, summary = build_manifests(
        excel_path=args.excel,
        base_dir=args.base_dir,
    )
    print(format_summary(summary))

    out_csv = Path(args.out_csv)
    manifest_df = records_to_dataframe([*ap_records, *lateral_records])
    manifest_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(manifest_df)} resolved samples to {out_csv}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
