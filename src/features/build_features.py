"""build_features.py — master feature-engineering pipeline for BP Cascade RI.

Orchestrates the individual feature modules into a single wide feature panel:

    load raw parquet  ->  compute PDC / BP-trajectory / SDOH features
                      ->  left-merge each onto the cohort by patient id
                      ->  export feature_panel.parquet (repo root)

Run
---
    ./venv/Scripts/python src/features/build_features.py

Design notes / gotchas baked into this script (see project memory):
  * PDC is the OUTCOME label (180-day forward window after the first-fill index),
    NOT a feature. It is written to labels.parquet and deliberately kept OUT of
    feature_panel.parquet to avoid target leakage.
  * SDOH flags read from conditions.parquet (SNOMED, date col START), NOT
    observations.parquet (LOINC labs/vitals) — otherwise every flag is 0.
  * BP readings are the LOINC subset of observations.parquet; there is no
    separate vitals table in this repo.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Make the sibling feature modules importable regardless of CWD (`from pdc import ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdc import calculate_pdc_outcome  # noqa: E402
from trajectories import compute_bp_trajectory  # noqa: E402
from sdoh import compute_sdoh_flags  # noqa: E402

# =============================================================================
# CONFIG  — edit paths / column names here
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"

# Inputs.
COHORT_PATH = ROOT / "data" / "cohort_patients_1k.parquet"   # must have PATIENT_ID_COL + INDEX_DATE_COL
MEDS_PATH = PARQUET / "medications.parquet"
VITALS_PATH = PARQUET / "observations.parquet"               # BP = LOINC subset of observations
CONDITIONS_PATH = PARQUET / "conditions.parquet"             # SDOH SNOMED source (NOT observations)
CODE_DICT_PATH = ROOT / "code_dictionary.yaml"

# Output.
OUTPUT_PATH = ROOT / "feature_panel.parquet"   # model features (X)
LABELS_PATH = ROOT / "labels.parquet"          # model target (y): PDC outcome, kept separate

# Column conventions.
PATIENT_ID_COL = "patient_id"       # cohort key column
INDEX_DATE_COL = "index_date"       # first-fill index: BP/SDOH pre-index cutoff + PDC forward window
CONDITIONS_DATE_COL = "START"       # conditions.parquet date column

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("build_features")


def _read_parquet(path: Path, label: str) -> pd.DataFrame:
    """Load one parquet file with a friendly error if it's missing."""
    if not path.exists():
        log.error("missing %s input: %s", label, path)
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    log.info("  loaded %-12s %8d rows x %2d cols  (%s)", label, len(df), df.shape[1], path.name)
    return df


def _normalize_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Ensure the cohort has the expected id + index-date columns.

    Tolerates the existing snapshot's ``Id`` column by renaming it to
    ``PATIENT_ID_COL``. Fails fast if the index date is absent, since every
    feature depends on it.
    """
    cohort = cohort.copy()
    if PATIENT_ID_COL not in cohort.columns and "Id" in cohort.columns:
        cohort = cohort.rename(columns={"Id": PATIENT_ID_COL})
        log.info("  normalized cohort id column 'Id' -> '%s'", PATIENT_ID_COL)
    if PATIENT_ID_COL not in cohort.columns:
        raise KeyError(f"cohort is missing patient id column '{PATIENT_ID_COL}'")
    return cohort


def main() -> int:
    # ----- Load ------------------------------------------------------------
    log.info("Loading data...")
    cohort = _normalize_cohort(_read_parquet(COHORT_PATH, "cohort"))
    meds = _read_parquet(MEDS_PATH, "medications")

    # TEMP: cohort has no index_date — derive it as each patient's very first medication
    # fill date (min START in medications.parquet). ISO-8601 strings sort chronologically,
    # so a string min is correct and avoids parsing the whole table here.
    if INDEX_DATE_COL not in cohort.columns:
        log.info("  deriving %s = first medication fill per patient", INDEX_DATE_COL)
        first_fill = (
            meds.groupby("PATIENT", as_index=False)["START"].min()
            .rename(columns={"PATIENT": PATIENT_ID_COL, "START": INDEX_DATE_COL})
        )
        cohort = cohort.merge(first_fill, on=PATIENT_ID_COL, how="left")
        n_missing = cohort[INDEX_DATE_COL].isna().sum()
        if n_missing:
            log.warning("  %d patient(s) have no fills; index_date is null -> NaN labels", n_missing)
    vitals = _read_parquet(VITALS_PATH, "vitals")
    conditions = _read_parquet(CONDITIONS_PATH, "conditions")
    with open(CODE_DICT_PATH) as f:
        code_dictionary = yaml.safe_load(f)
    log.info("  loaded code_dictionary.yaml")

    # ----- Compute outcome labels (kept OUT of the feature matrix) ---------
    log.info("Calculating PDC outcome labels (180d forward, post-index)...")
    labels_df = calculate_pdc_outcome(
        cohort, meds,
        cohort_patient_col=PATIENT_ID_COL,
        index_date_col=INDEX_DATE_COL,
    )

    # ----- Compute features ------------------------------------------------
    log.info("Calculating BP trajectories (systolic/diastolic)...")
    bp_df = compute_bp_trajectory(
        cohort, vitals,
        cohort_patient_col=PATIENT_ID_COL,
        index_date_col=INDEX_DATE_COL,
    )

    log.info("Calculating SDOH barrier flags...")
    sdoh_df = compute_sdoh_flags(
        cohort, conditions, code_dictionary,
        cohort_patient_col=PATIENT_ID_COL,
        index_date_col=INDEX_DATE_COL,
        date_col=CONDITIONS_DATE_COL,
    )

    # ----- Merge -----------------------------------------------------------
    log.info("Merging features onto cohort...")
    panel = cohort
    for name, feats in (("BP", bp_df), ("SDOH", sdoh_df)):
        before = panel.shape[1]
        panel = panel.merge(feats, on=PATIENT_ID_COL, how="left")
        log.info("  merged %-4s (+%d cols) -> %d rows x %d cols",
                 name, panel.shape[1] - before, len(panel), panel.shape[1])

    # ----- Export ----------------------------------------------------------
    log.info("Saving feature_panel.parquet...")
    panel.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %s  (%d patients x %d features)", OUTPUT_PATH, len(panel), panel.shape[1])

    log.info("Saving labels.parquet (PDC outcome — separate from features)...")
    labels_df.to_parquet(LABELS_PATH, index=False)
    n_gap = int(labels_df["has_30_day_gap"].fillna(0).sum())
    log.info("Saved %s  (%d patients, %d with a >=30d gap)", LABELS_PATH, len(labels_df), n_gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
