"""Convert raw Synthea CSVs (data/raw/*.csv) to Parquet (data/parquet/).

Uses Polars lazy scan + streaming sink so the big tables (claims_transactions ~424MB,
observations ~134MB) never have to fit in memory at once.

Run:  make ingest      (or)   ./venv/bin/python src/etl/ingest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "parquet"

# Columns that look numeric but must stay strings (codes, ids, zips with leading zeros).
FORCE_STR = {"ZIP", "SSN", "FIPS", "CODE", "PROCEDURECODE", "REASONCODE"}


def convert(csv_path: Path, out_path: Path) -> int:
    """Stream one CSV to Parquet. Returns row count. Falls back to all-utf8 on type errors."""
    # keep code/id/zip-like columns as strings (leading zeros, mixed formats)
    header = pl.read_csv(csv_path, n_rows=0, infer_schema_length=0).columns
    overrides = {c: pl.Utf8 for c in header if c in FORCE_STR}
    try:
        lf = pl.scan_csv(
            csv_path,
            infer_schema_length=10_000,
            schema_overrides=overrides,
            try_parse_dates=False,
            ignore_errors=False,
        )
        lf.sink_parquet(out_path, compression="zstd")
    except Exception as e:  # noqa: BLE001 - inference can fail on mixed-type cols
        # most failures are int-then-float money columns; full-file inference fixes them
        # while preserving numeric types. Fall to all-string only if that still fails.
        print(f"    10k-row inference failed ({type(e).__name__}); retrying full-file", file=sys.stderr)
        try:
            lf = pl.scan_csv(
                csv_path,
                infer_schema_length=None,  # scan whole file
                schema_overrides=overrides,
                try_parse_dates=False,
            )
            lf.sink_parquet(out_path, compression="zstd")
        except Exception as e2:  # noqa: BLE001
            print(f"    full-file inference failed ({type(e2).__name__}); falling back to all-string", file=sys.stderr)
            lf = pl.scan_csv(csv_path, infer_schema_length=0)  # everything as Utf8
            lf.sink_parquet(out_path, compression="zstd")
    # cheap row count off the parquet without loading it
    return pl.scan_parquet(out_path).select(pl.len()).collect().item()


def main() -> int:
    if not RAW.exists():
        print(f"no raw dir: {RAW}", file=sys.stderr)
        return 1
    csvs = sorted(RAW.glob("*.csv"))
    if not csvs:
        print(f"no CSVs in {RAW}", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ingesting {len(csvs)} files: {RAW} -> {OUT}\n")
    total_rows = 0
    for csv_path in csvs:
        out_path = OUT / (csv_path.stem + ".parquet")
        t0 = time.time()
        rows = convert(csv_path, out_path)
        total_rows += rows
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {csv_path.name:<26} {rows:>10,} rows -> {size_mb:6.1f} MB  ({time.time()-t0:4.1f}s)")
    print(f"\ndone: {len(csvs)} tables, {total_rows:,} rows total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
