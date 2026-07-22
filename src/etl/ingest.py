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

# --- 300k bundle configuration ------------------------------------------------
# RAW is the EXTERNAL 300k CSV bundle; OUT is a SEPARATE parquet dir so the 1k
# parquet in data/parquet is left untouched. (1k originals were RAW=data/raw,
# OUT=data/parquet — flip back to re-ingest the 1k set.)
RAW = Path(r"C:\Users\Chris\Documents\RIAI\300k_20260508_bundle")
OUT = ROOT / "data" / "parquet_300k"

# Convert ALL tables in the bundle. observations.csv (~48GB) and
# claims_transactions.csv (~153GB) are handled via streaming sink (memory-safe),
# but expect a long run and tens of GB of parquet output.

# Tables to skip entirely (team decision — not needed for the pipeline).
EXCLUDE_TABLES = {"devices", "claims_transactions", "imaging_studies", "claims", "supplies"}

# These very large / mixed-type tables are converted as all-Utf8 (streaming) to skip
# the risky, slow full-file type-inference fallback. Downstream code casts as needed.
ALL_UTF8_TABLES = {"observations"}

# In addition to the full observations table, emit a small BP-only subset
# (observations_bp.parquet) that the memory-bound feature pipeline reads.
OBSERVATIONS_BP_CODES = {"8480-6", "8462-4"}  # systolic / diastolic LOINC

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


def convert_all_utf8(csv_path: Path, out_path: Path) -> int:
    """Stream a CSV to Parquet with every column as Utf8.

    Used for observations.csv (~48GB, mixed VALUE types) where per-column type
    inference is both risky and expensive. Downstream code parses/coerces as needed.
    """
    pl.scan_csv(csv_path, infer_schema_length=0).sink_parquet(out_path, compression="zstd")
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
    print(f"ingesting from {RAW} -> {OUT}\n")
    total_rows = 0
    n_done = 0
    for csv_path in csvs:
        stem = csv_path.stem
        if stem in EXCLUDE_TABLES:
            continue
        out_path = OUT / (stem + ".parquet")
        t0 = time.time()
        # huge/mixed-type tables -> all-Utf8; everything else keeps typed convert.
        rows = convert_all_utf8(csv_path, out_path) if stem in ALL_UTF8_TABLES else convert(csv_path, out_path)
        total_rows += rows
        n_done += 1
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {out_path.name:<26} {rows:>12,} rows -> {size_mb:7.1f} MB  ({time.time()-t0:5.1f}s)")

        # Also emit a small BP-only subset for the memory-bound feature pipeline.
        if stem == "observations":
            bp_path = OUT / "observations_bp.parquet"
            (pl.scan_parquet(out_path)
               .filter(pl.col("CODE").is_in(list(OBSERVATIONS_BP_CODES)))
               .sink_parquet(bp_path, compression="zstd"))
            print(f"  {bp_path.name:<26} {'(BP subset)':>12} -> {bp_path.stat().st_size/1e6:7.1f} MB")
    print(f"\ndone: {n_done} tables, {total_rows:,} rows total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
