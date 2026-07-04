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

from pdc import calculate_pdc_outcome, ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL  # noqa: E402
from trajectories import compute_bp_trajectory  # noqa: E402
from sdoh import compute_sdoh_flags  # noqa: E402

# =============================================================================
# CONFIG  — edit paths / column names here
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet_300k"   # 300k parquet (1k was data/parquet)

# Inputs.
COHORT_PATH = ROOT / "data" / "snapshots" / "cohort_patients.parquet"  # output of src/etl/cohort.py
MEDS_PATH = PARQUET / "medications.parquet"
VITALS_PATH = PARQUET / "observations_bp.parquet"            # BP-only subset written by ingest.py
CONDITIONS_PATH = PARQUET / "conditions.parquet"             # SDOH SNOMED source (NOT observations)
PATIENTS_PATH = PARQUET / "patients.parquet"                 # demographics for selection-bias report
CODE_DICT_PATH = ROOT / "code_dictionary.yaml"

# Output.
OUTPUT_PATH = ROOT / "feature_panel.parquet"   # model features (X)
LABELS_PATH = ROOT / "labels.parquet"          # model target (y): PDC outcome, kept separate

# Column conventions.
PATIENT_ID_COL = "patient_id"       # cohort key column
INDEX_DATE_COL = "index_date"       # incident-fill index: BP/SDOH pre-index cutoff + PDC forward window
CONDITIONS_DATE_COL = "START"       # conditions.parquet date column

# Incident / new-user cohort design.
WASHOUT_DAYS = 365          # index_date = first antihypertensive fill with NO prior fill in the preceding WASHOUT_DAYS
BASELINE_WINDOW_DAYS = 1095  # keep only patients with >=1 observation in [index_date - BASELINE_WINDOW_DAYS, index_date]
REFERENCE_DATE = "2026-07-01"  # "today" for the age calc in the selection-bias report
HTN_DX_CODE = "59621000"    # SNOMED essential hypertension, for years-since-first-dx
SYSTOLIC_BP_CODE = "8480-6" # LOINC systolic BP; baseline eligibility requires one of these

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


def _parse_dt(series: pd.Series) -> pd.Series:
    """ISO strings -> tz-naive Timestamps (NaT on failure)."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def _incident_index_date(meds: pd.DataFrame) -> pd.Series:
    """Incident/new-user index date per patient (Series: PATIENT -> Timestamp).

    index_date = the earliest antihypertensive fill that has NO prior antihypertensive
    fill within the preceding WASHOUT_DAYS. This is the treatment-initiation point for a
    genuine new user rather than an arbitrary refill of a long-standing regimen.
    """
    ah = meds[meds["CODE"].astype(str).isin(ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL)].copy()
    ah["_s"] = _parse_dt(ah["START"])
    ah = ah.dropna(subset=["_s"]).sort_values(["PATIENT", "_s"])
    prev = ah.groupby("PATIENT")["_s"].shift()
    gap_days = (ah["_s"] - prev).dt.days
    incident = ah[prev.isna() | (gap_days > WASHOUT_DAYS)]  # no fill in the prior washout
    return incident.groupby("PATIENT")["_s"].min()


def _baseline_eligible_ids(cohort: pd.DataFrame, observations: pd.DataFrame) -> set:
    """Patients with >=1 systolic BP reading in [index_date - BASELINE_WINDOW_DAYS, index_date]."""
    obs = observations[observations["CODE"].astype(str) == SYSTOLIC_BP_CODE][["PATIENT", "DATE"]].copy()
    obs["_d"] = _parse_dt(obs["DATE"])
    obs = obs.dropna(subset=["_d"]).merge(
        cohort[[PATIENT_ID_COL, INDEX_DATE_COL]],
        left_on="PATIENT", right_on=PATIENT_ID_COL, how="inner",
    )
    lo = obs[INDEX_DATE_COL] - pd.to_timedelta(BASELINE_WINDOW_DAYS, unit="D")
    in_baseline = (obs["_d"] >= lo) & (obs["_d"] < obs[INDEX_DATE_COL])
    return set(obs.loc[in_baseline, PATIENT_ID_COL].unique())


def _group_profile(sub: pd.DataFrame, conditions: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Mean age / comorbidity-count / years-since-first-HTN-dx for a cohort subset."""
    ids = set(sub[PATIENT_ID_COL])
    cond = conditions[conditions["PATIENT"].isin(ids)]
    comorbidities = cond.groupby("PATIENT")["CODE"].nunique().reindex(ids).fillna(0)

    dx = cond[cond["CODE"].astype(str) == HTN_DX_CODE].copy()
    dx["_s"] = _parse_dt(dx["START"])
    first_dx = dx.groupby("PATIENT")["_s"].min()
    idx = sub.set_index(PATIENT_ID_COL)[INDEX_DATE_COL]
    years_since_dx = ((idx - first_dx).dt.days / 365.25).dropna()

    pts = patients[patients["Id"].isin(ids)]
    age = (pd.Timestamp(REFERENCE_DATE) - _parse_dt(pts["BIRTHDATE"])).dt.days / 365.25

    return {
        "n": len(ids),
        "age": age.mean(),
        "comorbidities": comorbidities.mean(),
        "yrs_since_dx": years_since_dx.mean() if len(years_since_dx) else float("nan"),
    }


def _log_selection_bias(included: pd.DataFrame, excluded: pd.DataFrame,
                        conditions: pd.DataFrame, patients: pd.DataFrame) -> None:
    inc, exc = (_group_profile(included, conditions, patients),
                _group_profile(excluded, conditions, patients))
    log.info("  selection-bias profile (included vs excluded):")
    log.info("    %-14s %8s %8s", "", "included", "excluded")
    for k, label in (("n", "n"), ("age", "mean age"),
                     ("comorbidities", "mean #cond"), ("yrs_since_dx", "yrs since dx")):
        log.info("    %-14s %8.1f %8.1f", label, inc[k], exc[k])


def main() -> int:
    # ----- Load ------------------------------------------------------------
    log.info("Loading data...")
    cohort = _normalize_cohort(_read_parquet(COHORT_PATH, "cohort"))
    meds = _read_parquet(MEDS_PATH, "medications")
    vitals = _read_parquet(VITALS_PATH, "vitals")
    conditions = _read_parquet(CONDITIONS_PATH, "conditions")
    patients = _read_parquet(PATIENTS_PATH, "patients")
    with open(CODE_DICT_PATH) as f:
        code_dictionary = yaml.safe_load(f)
    log.info("  loaded code_dictionary.yaml")

    # ----- Incident index_date + baseline eligibility ----------------------
    # index_date = first antihypertensive fill preceded by a WASHOUT_DAYS clean period
    # (new-user design). The prior first-ever-fill logic pulled index dates 10+ years
    # back for chronic patients, predating observation coverage and zeroing out baseline BP.
    if INDEX_DATE_COL not in cohort.columns:
        log.info("Deriving incident %s (washout=%dd)...", INDEX_DATE_COL, WASHOUT_DAYS)
        idx = _incident_index_date(meds).rename(INDEX_DATE_COL)
        cohort = cohort.merge(idx, left_on=PATIENT_ID_COL, right_index=True, how="left")

    n_total = len(cohort)
    no_fill = cohort[INDEX_DATE_COL].isna().sum()

    # Baseline eligibility: require >=1 observation in the baseline window. Patients who
    # fail are EXCLUDED (no imputation of missing baseline BP).
    log.info("Applying baseline eligibility (>=1 obs in prior %dd)...", BASELINE_WINDOW_DAYS)
    eligible = _baseline_eligible_ids(cohort.dropna(subset=[INDEX_DATE_COL]), vitals)
    keep = cohort[INDEX_DATE_COL].notna() & cohort[PATIENT_ID_COL].isin(eligible)
    included, excluded = cohort[keep], cohort[~keep]
    log.info("  eligible %d / %d (%.1f%%); excluded %d (%d no fill, %d no baseline obs)",
             len(included), n_total, 100 * len(included) / n_total,
             len(excluded), no_fill, len(excluded) - no_fill)
    _log_selection_bias(included, excluded, conditions, patients)
    cohort = included

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
