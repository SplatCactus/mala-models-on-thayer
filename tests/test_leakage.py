"""test_leakage.py — the index-date leakage rule, enforced end to end.

Rule under test: every PRE-index feature computed in src/features/*.py must
be built only from source rows whose timestamp is strictly BEFORE the
patient's index_date. A row dated on or after index_date must never move a
feature value, even though it is present in the raw input DataFrame -- both
feature modules claim to filter such rows out internally; these tests prove
that filtering actually happens at the boundary (on index_date, and after
it), not just "in general".

Modules under test:
    src/features/trajectories.py -- BP readings (vitals) must be < index_date
    src/features/sdoh.py         -- condition codes must be < index_date

The flip side of the rule (outcome labels must look >= index_date, forward)
is also checked for src/features/pdc.py, since a pre-index fill leaking
into the forward PDC window would be the same bug in the other direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FEATURES_DIR = Path(__file__).resolve().parents[1] / "src" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from trajectories import compute_bp_trajectory, BP_LOINC_CODES  # noqa: E402
from sdoh import compute_sdoh_flags  # noqa: E402
from pdc import calculate_pdc_outcome, ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL  # noqa: E402

SYSTOLIC_CODE = BP_LOINC_CODES["systolic"]
CONTAMINANT_VALUE = 999.0  # obviously out of physiological range -- easy to spot if leaked


# ---------------------------------------------------------------------------
# BP trajectory (trajectories.py): vitals strictly before index_date
# ---------------------------------------------------------------------------

def _cohort(patient_id: str, index_date: str) -> pd.DataFrame:
    return pd.DataFrame({"Id": [patient_id], "index_date": [index_date]})


def test_bp_trajectory_excludes_reading_on_index_date():
    cohort = _cohort("p1", "2025-06-01")
    vitals = pd.DataFrame({
        "PATIENT": ["p1", "p1"],
        "CODE": [SYSTOLIC_CODE, SYSTOLIC_CODE],
        "DATE": ["2025-05-30", "2025-06-01"],  # one day before, one ON index_date
        "VALUE": [120, CONTAMINANT_VALUE],
    })
    out = compute_bp_trajectory(cohort, vitals)
    row = out.iloc[0]
    assert row["sbp_mean"] == 120
    assert row["sbp_max"] == 120
    assert row["sbp_latest_reading"] == 120


def test_bp_trajectory_excludes_reading_after_index_date():
    cohort = _cohort("p1", "2025-06-01")
    vitals = pd.DataFrame({
        "PATIENT": ["p1", "p1"],
        "CODE": [SYSTOLIC_CODE, SYSTOLIC_CODE],
        "DATE": ["2025-05-30", "2025-06-15"],  # one before, one clearly after
        "VALUE": [120, CONTAMINANT_VALUE],
    })
    out = compute_bp_trajectory(cohort, vitals)
    row = out.iloc[0]
    assert row["sbp_mean"] == 120
    assert row["sbp_max"] == 120


def test_bp_trajectory_includes_reading_one_day_before_index_date():
    """Sanity check the other direction: a legitimately pre-index reading IS used."""
    cohort = _cohort("p1", "2025-06-01")
    vitals = pd.DataFrame({
        "PATIENT": ["p1"],
        "CODE": [SYSTOLIC_CODE],
        "DATE": ["2025-05-31"],
        "VALUE": [142],
    })
    out = compute_bp_trajectory(cohort, vitals)
    assert out.iloc[0]["sbp_mean"] == 142


def test_bp_trajectory_property_no_leakage_across_many_patients():
    """No sbp feature should ever reflect an on/after-index reading, for any patient."""
    rng = np.random.default_rng(0)
    n_patients = 50
    patient_ids = [f"p{i}" for i in range(n_patients)]
    index_dates = pd.date_range("2025-01-01", periods=n_patients, freq="3D")
    cohort = pd.DataFrame({"Id": patient_ids, "index_date": index_dates.astype(str)})

    rows = []
    for pid, idx in zip(patient_ids, index_dates):
        legit_value = float(rng.integers(100, 160))
        rows.append({"PATIENT": pid, "CODE": SYSTOLIC_CODE,
                      "DATE": str((idx - pd.Timedelta(days=10)).date()), "VALUE": legit_value})
        rows.append({"PATIENT": pid, "CODE": SYSTOLIC_CODE,
                      "DATE": str(idx.date()), "VALUE": CONTAMINANT_VALUE})            # on index_date
        rows.append({"PATIENT": pid, "CODE": SYSTOLIC_CODE,
                      "DATE": str((idx + pd.Timedelta(days=5)).date()), "VALUE": CONTAMINANT_VALUE})  # after
    vitals = pd.DataFrame(rows)

    out = compute_bp_trajectory(cohort, vitals)
    assert (out["sbp_max"] < CONTAMINANT_VALUE).all()
    assert out["sbp_max"].notna().all()


# ---------------------------------------------------------------------------
# SDOH flags (sdoh.py): conditions strictly before index_date
# ---------------------------------------------------------------------------

CODE_DICT = {
    "composite_flags": {
        "sdoh_isolation_any": {
            "routing_action": "bilingual_chw_call",
            "codes": ["422650009", "423315002"],
        }
    }
}


def test_sdoh_flag_excludes_code_on_index_date():
    cohort = _cohort("p1", "2025-06-01")
    conditions = pd.DataFrame({
        "PATIENT": ["p1"],
        "CODE": ["422650009"],
        "START": ["2025-06-01"],  # ON index_date, not before
    })
    out = compute_sdoh_flags(cohort, conditions, CODE_DICT, date_col="START")
    assert out.iloc[0]["flag_sdoh_isolation_any"] == 0
    assert out.iloc[0]["flag_sdoh_any"] == 0


def test_sdoh_flag_excludes_code_after_index_date():
    cohort = _cohort("p1", "2025-06-01")
    conditions = pd.DataFrame({
        "PATIENT": ["p1"],
        "CODE": ["422650009"],
        "START": ["2025-07-01"],
    })
    out = compute_sdoh_flags(cohort, conditions, CODE_DICT, date_col="START")
    assert out.iloc[0]["flag_sdoh_isolation_any"] == 0
    assert out.iloc[0]["flag_sdoh_any"] == 0


def test_sdoh_flag_includes_code_before_index_date():
    """Sanity check the other direction: a legitimately pre-index code IS used."""
    cohort = _cohort("p1", "2025-06-01")
    conditions = pd.DataFrame({
        "PATIENT": ["p1"],
        "CODE": ["422650009"],
        "START": ["2025-05-31"],
    })
    out = compute_sdoh_flags(cohort, conditions, CODE_DICT, date_col="START")
    assert out.iloc[0]["flag_sdoh_isolation_any"] == 1
    assert out.iloc[0]["flag_sdoh_any"] == 1


def test_sdoh_flag_property_no_leakage_across_many_patients():
    rng = np.random.default_rng(1)
    n_patients = 50
    patient_ids = [f"p{i}" for i in range(n_patients)]
    index_dates = pd.date_range("2025-01-01", periods=n_patients, freq="3D")
    cohort = pd.DataFrame({"Id": patient_ids, "index_date": index_dates.astype(str)})

    rows = []
    for pid, idx in zip(patient_ids, index_dates):
        # every patient gets the SDOH code on/after their index_date only --
        # if the leakage filter is broken, every flag would read 1.
        offset_days = int(rng.integers(0, 30))
        rows.append({"PATIENT": pid, "CODE": "422650009",
                      "START": str((idx + pd.Timedelta(days=offset_days)).date())})
    conditions = pd.DataFrame(rows)

    out = compute_sdoh_flags(cohort, conditions, CODE_DICT, date_col="START")
    assert (out["flag_sdoh_isolation_any"] == 0).all()
    assert (out["flag_sdoh_any"] == 0).all()


# ---------------------------------------------------------------------------
# PDC outcome (pdc.py): the flip side -- pre-index fills must not count
# toward the forward-looking coverage window.
# ---------------------------------------------------------------------------

def test_pdc_outcome_excludes_fill_before_index_date():
    antihypertensive_code = next(iter(ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL))
    cohort = _cohort("p1", "2025-06-01")
    meds = pd.DataFrame({
        "PATIENT": ["p1"],
        "CODE": [antihypertensive_code],
        "START": ["2025-05-01"],   # entirely before index_date
        "STOP": ["2025-05-31"],
    })
    out = calculate_pdc_outcome(cohort, meds, cohort_patient_col="Id")
    row = out.iloc[0]
    # No qualifying fill inside the forward window -> complete non-persistence.
    assert row["pdc_180d"] == 0.0
    assert row["has_30_day_gap"] == 1


def test_pdc_outcome_includes_fill_on_index_date():
    antihypertensive_code = next(iter(ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL))
    cohort = _cohort("p1", "2025-06-01")
    meds = pd.DataFrame({
        "PATIENT": ["p1"],
        "CODE": [antihypertensive_code],
        "START": ["2025-06-01"],   # exactly on index_date -- start of forward window
        "STOP": ["2025-07-01"],
    })
    out = calculate_pdc_outcome(cohort, meds, cohort_patient_col="Id")
    row = out.iloc[0]
    assert row["pdc_180d"] > 0.0
    assert row["has_30_day_gap"] == 1  # gap after the single 30-day fill, within the 180d window


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
