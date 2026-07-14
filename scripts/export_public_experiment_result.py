#!/usr/bin/env python3
"""Export an identifier-free public manifest from a local experiment result.

Raw run JSON files may contain accession-level audit examples and per-case out-of-fold
outputs. This exporter keeps the protocol, aggregate metrics, comparisons, fold selections,
and seed-level metrics while removing identifiers and per-case predictions/scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DROP_KEYS = {
    "accession",
    "study_key",
    "identities",
    "oof_predictions",
    "oof_scores",
}
IDENTIFIER_LIST_KEYS = {"discordant", "discordant_strict_pairs"}
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(value: str) -> str:
    marker = "/DSA_OccluNet/"
    if marker in value:
        return value.split(marker, 1)[1]
    return value


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key in DROP_KEYS:
                continue
            if key in IDENTIFIER_LIST_KEYS and isinstance(item, list):
                output[f"{key}_count"] = len(item)
                continue
            output[key] = sanitize(item)
        return output
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return public_path(value)
    return value


def export_public_result(source: Path) -> dict[str, Any]:
    raw = json.loads(source.read_text())
    public = sanitize(raw)
    public["public_export"] = {
        "source_filename": source.name,
        "source_sha256": file_sha256(source),
        "removed": [
            "accession and Study_Key audit examples",
            "sample identities",
            "per-case OOF predictions and scores",
            "local absolute path prefixes",
        ],
        "note": "The full raw result remains local because the GitHub repository is public.",
    }
    encoded = json.dumps(public, sort_keys=True)
    if UUID_PATTERN.search(encoded):
        raise ValueError("Public export still contains a UUID-like identifier.")
    if "/home/" in encoded:
        raise ValueError("Public export still contains an absolute home-directory path.")
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    public = export_public_result(arguments.source)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(f"Wrote identifier-free public manifest: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
