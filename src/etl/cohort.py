"""Extract the treated-hypertensive cohort from data/parquet/*.parquet.

Cohort definition (per BP Cascade RI master plan):
    treated-hypertensive = patient has Essential hypertension (SNOMED 59621000)
                            AND >=1 antihypertensive medication fill (RxNorm, see below)

Output: one row per patient -> data/snapshots/cohort_patients.parquet (gitignored)

Run:  ./venv/bin/python src/etl/cohort.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
# Input parquet dir + output snapshot are env-overridable so the same extraction
# can be run against another dataset (e.g. the 1k data/parquet holdout) without
# editing code: COHORT_PARQUET_DIR / COHORT_OUT.
PARQUET = Path(os.environ.get("COHORT_PARQUET_DIR", ROOT / "data" / "parquet_300k"))
SNAPSHOTS = ROOT / "data" / "snapshots"
OUT_PATH = Path(os.environ.get("COHORT_OUT", SNAPSHOTS / "cohort_patients.parquet"))

# SNOMED-CT condition code for Essential hypertension (confirmed against this dataset's
# conditions.parquet on 2026-06-29 — single code, no synonyms found).
HTN_CONDITION_CODE = "59621000"

# RxNorm codes for antihypertensive medications, confirmed against this dataset's
# medications.parquet (32 codes spanning ACE-I, ARB, CCB, beta-blocker, thiazide, combos).
# NOTE: includes sacubitril/valsartan (Entresto) — primarily a heart-failure agent that
# also lowers BP; kept in-class per plan, but expect it to skew toward CHF-comorbid patients.
ANTIHYPERTENSIVE_CODES = {
    # beta-blockers
    "866412", "866419", "866427", "866429", "866436",  # metoprolol succinate ER
    "866511", "866924", "866514",                       # metoprolol tartrate
    "197379", "197381",                                  # atenolol
    "200032", "200033", "686924",                        # carvedilol
    # thiazide diuretics
    "310798",                                            # hydrochlorothiazide (mono)
    # CCB
    "308136",                                            # amlodipine (mono)
    "897685",                                            # verapamil
    # ACE-I
    "858804", "858810",                                  # enalapril
    "314076", "311353", "314077", "197884", "311354",    # lisinopril
    # ARB
    "979480", "979485", "979492",                        # losartan
    "349200",                                            # valsartan (mono)
    # combination products
    "999970",   # amlodipine/HCTZ/olmesartan
    "200285",   # HCTZ/valsartan
    "197887",   # HCTZ/lisinopril
    "979471",   # HCTZ/losartan
    "1656356",  # sacubitril/valsartan (Entresto) — see note above
}

REFERENCE_DATE = "2026-06-29"  # snapshot date for age calc; bump if re-run later


def load(table: str) -> pl.LazyFrame:
    path = PARQUET / f"{table}.parquet"
    if not path.exists():
        print(f"missing parquet: {path}", file=sys.stderr)
        sys.exit(1)
    return pl.scan_parquet(path)


def main() -> int:
    patients = load("patients")
    conditions = load("conditions")
    medications = load("medications")

    # --- Step 1: patients with the HTN diagnosis ---
    htn_patients = (
        conditions
        .filter(pl.col("CODE") == HTN_CONDITION_CODE)
        .select("PATIENT")
        .unique()
    )

    # --- Step 2: patients with >=1 antihypertensive fill ---
    treated_patients = (
        medications
        .filter(pl.col("CODE").is_in(list(ANTIHYPERTENSIVE_CODES)))
        .select("PATIENT")
        .unique()
    )

    # --- Step 3: intersection = treated-hypertensive cohort ---
    cohort_ids = (
        htn_patients
        .join(treated_patients, on="PATIENT", how="inner")
        .rename({"PATIENT": "Id"})
    )

    # --- Step 4: attach patient-level demographics for equity strata (held aside, not modeling features) ---
    cohort = (
        patients
        .join(cohort_ids, on="Id", how="inner")
        .with_columns(
            # empty strings -> null, consistent with ingest's "dates stay strings" decision
            pl.col("DEATHDATE").replace("", None).alias("DEATHDATE"),
            pl.col("DEATHDATE").replace("", None).is_not_null().alias("is_deceased"),
        )
        .with_columns(
            # age in years as of REFERENCE_DATE (or at death, if deceased)
            (
                (
                    pl.coalesce([pl.col("DEATHDATE"), pl.lit(REFERENCE_DATE)])
                    .str.to_date("%Y-%m-%d")
                    - pl.col("BIRTHDATE").str.to_date("%Y-%m-%d")
                ).dt.total_days()
                / 365.25
            )
            .floor()
            .cast(pl.Int32)
            .alias("age_years")
        )
        .select(
            "Id", "RACE", "ETHNICITY", "GENDER", "age_years", "is_deceased",
            "CITY", "STATE", "ZIP", "LAT", "LON",
            "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME",
        )
    )

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.sink_parquet(out_path, compression="zstd")

    n_htn = htn_patients.select(pl.len()).collect().item()
    n_treated = treated_patients.select(pl.len()).collect().item()
    n_cohort = pl.scan_parquet(out_path).select(pl.len()).collect().item()

    print(f"HTN-diagnosed patients:        {n_htn:,}")
    print(f"Antihypertensive-fill patients: {n_treated:,}")
    print(f"Treated-hypertensive cohort:    {n_cohort:,}  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
