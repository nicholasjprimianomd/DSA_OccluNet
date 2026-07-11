"""Reconstruct a Google Drive folder (e.g. M2_M3_data) from the offline Google DriveFS
stream cache — no Windows boot, no re-download.

Google Drive for Desktop streams files and caches their content under
`…/AppData/Local/Google/DriveFS/<account_id>/content_cache/` as plaintext blobs named
by a numeric content-entry id. `metadata_sqlite_db` holds the folder tree and, per file,
the content-entry id (in the `content-entry` property or the `inactive-content-entries`
protobuf). This script walks that tree and rebuilds `<folder>/<accession>/<file>` by
symlinking (default, instant) or copying each cached blob into place.

Example:
    python scripts/extract_drive_cache.py \
        --drivefs "/run/media/nick/<vol>/Users/nprim/AppData/Local/Google/DriveFS/108952070282071817452" \
        --folder M2_M3_data --out ~/M2_M3_data --mode symlink

Then:  export DSA_BASE_DIR=~/M2_M3_data
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


# ---- minimal protobuf reader (varint + field walk) -------------------------------------
def _read_varint(b: bytes, i: int) -> tuple[int, int]:
    shift = val = 0
    while True:
        byte = b[i]
        i += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, i
        shift += 7


def _iter_fields(b: bytes, start: int = 0, end: int | None = None):
    i, end = start, (len(b) if end is None else end)
    while i < end:
        key, i = _read_varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _read_varint(b, i)
            yield field, "varint", val
        elif wire == 2:
            ln, i = _read_varint(b, i)
            yield field, "bytes", b[i:i + ln]
            i += ln
        elif wire == 1:
            yield field, "fixed64", b[i:i + 8]; i += 8
        elif wire == 5:
            yield field, "fixed32", b[i:i + 4]; i += 4
        else:
            raise ValueError(f"unknown wire type {wire}")


def _entries_from_blob(blob: bytes) -> list[tuple[int, int | None]]:
    """From an `inactive-content-entries` blob: [(content_id, size), ...]."""
    out = []
    for field, wt, val in _iter_fields(blob):
        if field == 1 and wt == "bytes":
            cid = size = None
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == "varint":
                    cid = v2
                elif f2 == 4 and w2 == "varint":
                    size = v2
            if cid is not None:
                out.append((cid, size))
    return out


def _candidate_ids(props: dict[str, tuple]) -> list[tuple[int, int | None]]:
    """All (content_id, size) candidates from a file item's properties."""
    cands: list[tuple[int, int | None]] = []
    ce = props.get("content-entry")
    if ce is not None:
        val, _vt = ce
        if isinstance(val, int):
            cands.append((val, None))
        elif isinstance(val, (bytes, bytearray)):
            cid = size = None
            for f, w, v in _iter_fields(bytes(val)):
                if f == 1 and w == "varint":
                    cid = v
                elif f == 4 and w == "varint":
                    size = v
            if cid is not None:
                cands.append((cid, size))
            cands.extend(_entries_from_blob(bytes(val)))
    inact = props.get("inactive-content-entries")
    if inact is not None and isinstance(inact[0], (bytes, bytearray)):
        cands.extend(_entries_from_blob(bytes(inact[0])))
    return cands


# ---- cache index -----------------------------------------------------------------------
def build_cache_index(content_cache: Path) -> dict[int, Path]:
    """Map numeric content-entry id -> cache file path (scandir, no reads)."""
    index: dict[int, Path] = {}
    stack = [content_cache]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    elif e.name.isdigit():
                        index[int(e.name)] = Path(e.path)
        except OSError:
            continue
    return index


# ---- drive tree ------------------------------------------------------------------------
def load_props(con: sqlite3.Connection, stable_id: int) -> dict[str, tuple]:
    return {
        r["key"]: (r["value"], r["value_type"])
        for r in con.execute(
            "SELECT key,value,value_type FROM item_properties WHERE item_stable_id=?", (stable_id,)
        )
    }


def find_folder_id(con: sqlite3.Connection, name: str) -> int:
    row = con.execute(
        "SELECT stable_id FROM items WHERE local_title=? AND is_folder=1 AND trashed=0 "
        "ORDER BY stable_id LIMIT 1", (name,)
    ).fetchone()
    if row is None:
        raise SystemExit(f"Drive folder {name!r} not found in metadata db.")
    return row["stable_id"]


def walk(con: sqlite3.Connection, root_id: int):
    """Yield (relative_path, stable_id, file_size) for every file under root_id."""
    stack = [(root_id, Path())]
    while stack:
        pid, prel = stack.pop()
        for c in con.execute(
            "SELECT i.stable_id,i.local_title,i.is_folder,i.file_size FROM stable_parents sp "
            "JOIN items i ON i.stable_id=sp.item_stable_id "
            "WHERE sp.parent_stable_id=? AND i.trashed=0", (pid,)
        ):
            rel = prel / c["local_title"]
            if c["is_folder"]:
                stack.append((c["stable_id"], rel))
            else:
                yield rel, c["stable_id"], c["file_size"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drivefs", required=True, help="Path to the DriveFS <account_id> directory.")
    ap.add_argument("--folder", default="M2_M3_data", help="Drive folder name to reconstruct.")
    ap.add_argument("--out", required=True, help="Output base dir (folder is created inside).")
    ap.add_argument("--mode", choices=("symlink", "copy"), default="symlink",
                    help="symlink = instant, needs the cache drive mounted; copy = durable.")
    ap.add_argument("--referenced-xlsx", default="",
                    help="Only reconstruct DICOMs referenced by this label sheet's run columns "
                    "(plus any .xlsx). Great with --mode copy to pull just the ~700 labeled runs.")
    ap.add_argument("--limit-accessions", type=int, default=0, help="Only the first N accession folders (testing).")
    ap.add_argument("--verify", action="store_true", help="Check size and DICM magic on each reconstructed file.")
    args = ap.parse_args()

    acct = Path(args.drivefs)
    md = acct / "metadata_sqlite_db"
    content_cache = acct / "content_cache"
    if not md.exists() or not content_cache.exists():
        raise SystemExit(f"metadata_sqlite_db / content_cache not found under {acct}")

    print(f"Indexing content cache under {content_cache} …")
    index = build_cache_index(content_cache)
    print(f"  {len(index)} cached content blobs indexed")

    con = sqlite3.connect(f"file:{md}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    root_id = find_folder_id(con, args.folder)
    out_root = Path(args.out).expanduser()

    referenced: set[tuple[str, str]] | None = None
    if args.referenced_xlsx:
        import pandas as pd

        df = pd.read_excel(args.referenced_xlsx)
        run_cols = [c for c in df.columns if c.split("_")[0] in ("AP", "Lateral") and c[-1].isdigit()]
        referenced = set()
        for _, row in df.iterrows():
            acc = str(row.get("Accession", "")).strip()
            if not acc or acc.lower() == "nan":
                continue
            for c in run_cols:
                v = str(row.get(c, "")).strip()
                if v and v.lower() != "nan":
                    referenced.add((acc, v if v.lower().endswith(".dcm") else f"{v}.dcm"))
        print(f"Referenced filter: {len(referenced)} labeled runs from {args.referenced_xlsx}")

    made = missing = mismatch = 0
    seen_accessions: set[str] = set()
    for rel, sid, size in walk(con, root_id):
        accession = rel.parts[0] if len(rel.parts) > 1 else ""
        if referenced is not None and len(rel.parts) >= 2:
            if (accession, rel.name) not in referenced and not rel.name.lower().endswith(".xlsx"):
                continue
        if args.limit_accessions and accession not in seen_accessions:
            if len(seen_accessions) >= args.limit_accessions:
                continue
            seen_accessions.add(accession)

        cache_path = None
        for cid, csize in _candidate_ids(load_props(con, sid)):
            p = index.get(cid)
            if p is None:
                continue
            if size and csize is not None and csize != size:
                continue
            cache_path = p
            break
        if cache_path is None:
            missing += 1
            continue

        dest = out_root / args.folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if args.mode == "symlink":
            dest.symlink_to(cache_path)
        else:
            shutil.copy2(cache_path, dest)

        if args.verify:
            real = dest.resolve()
            ok_size = (real.stat().st_size == size) if size else True
            with open(real, "rb") as fh:
                fh.seek(128)
                ok_magic = fh.read(4) == b"DICM"
            if not (ok_size and ok_magic):
                mismatch += 1
        made += 1
        if made % 200 == 0:
            print(f"  … {made} files")

    con.close()
    print(f"\nDone: {made} files reconstructed into {out_root/args.folder}")
    print(f"  missing from cache: {missing}" + (f" | failed verify: {mismatch}" if args.verify else ""))
    print(f"\nNext:  export DSA_BASE_DIR={out_root/args.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
