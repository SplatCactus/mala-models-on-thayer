"""test_pre_index_leakage.py — the index-date leakage rule for pre_index.py.

Same rule as test_leakage.py, extended to the src/features/pre_index.py feature
family (comorbidity / regimen / prior-adherence / engagement / payer churn):
every feature must be built only from source rows strictly BEFORE the patient's
index_date. A row dated ON or AFTER index_date must never move any feature value.

Each test feeds one contaminant row dated on/after index and asserts the feature
stays at its no-history default, plus a mirror "one day before" case proving the
legitimate pre-index signal IS counted (so the tests can't pass by the feature
being dead — the silent-zero failure mode the project memory warns about).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

FEATURES_DIR = Path(__file__).resolve().parents[1] / "src" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from pre_index import compute_pre_index_features  # noqa: E402
from pdc import ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL  # noqa: E402

AH_CODE = next(iter(ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL))
SYSTOLIC_CODE = "8480-6"
INDEX = "2025-06-01"


def _cohort(index_date: str = INDEX) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": ["p1"], "index_date": [index_date]})


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in cols})


def _run(*, conditions=None, meds=None, encounters=None, vitals=None, payers=None,
         index_date: str = INDEX) -> pd.Series:
    """Run compute_pre_index_features with any omitted table left empty."""
    conditions = conditions if conditions is not None else _empty(["PATIENT", "CODE", "START"])
    meds = meds if meds is not None else _empty(["PATIENT", "CODE", "START", "STOP"])
    encounters = encounters if encounters is not None else _empty(["PATIENT", "START"])
    vitals = vitals if vitals is not None else _empty(["PATIENT", "CODE", "DATE"])
    payers = payers if payers is not None else _empty(["PATIENT", "PAYER", "START_DATE"])
    out = compute_pre_index_features(
        _cohort(index_date), conditions, meds, encounters, vitals, payers)
    return out.iloc[0]


# --- comorbidity: cmb_n_conditions --------------------------------------------

def test_comorbidity_excludes_condition_on_and_after_index():
    conditions = pd.DataFrame({"PATIENT": ["p1", "p1"], "CODE": ["A", "B"],
                               "START": [INDEX, "2025-07-01"]})
    assert _run(conditions=conditions)["cmb_n_conditions"] == 0


def test_comorbidity_includes_condition_before_index():
    conditions = pd.DataFrame({"PATIENT": ["p1", "p1"], "CODE": ["A", "B"],
                               "START": ["2020-01-01", "2025-05-31"]})
    assert _run(conditions=conditions)["cmb_n_conditions"] == 2


# --- regimen: rx_n_active_meds / rx_n_meds_started_1y -------------------------

def test_regimen_excludes_med_started_on_index():
    meds = pd.DataFrame({"PATIENT": ["p1"], "CODE": ["m1"],
                         "START": [INDEX], "STOP": ["2025-12-01"]})
    row = _run(meds=meds)
    assert row["rx_n_active_meds"] == 0
    assert row["rx_n_meds_started_1y"] == 0


def test_regimen_counts_med_active_at_index():
    # started before index, still running at index -> active; also within 1y.
    meds = pd.DataFrame({"PATIENT": ["p1"], "CODE": ["m1"],
                         "START": ["2025-05-01"], "STOP": ["2025-12-01"]})
    row = _run(meds=meds)
    assert row["rx_n_active_meds"] == 1
    assert row["rx_n_meds_started_1y"] == 1


# --- prior adherence: adh_prior_n_med_fills / adh_prior_med_pdc ----------------

def test_prior_adherence_excludes_fill_on_index():
    meds = pd.DataFrame({"PATIENT": ["p1"], "CODE": [AH_CODE],
                         "START": [INDEX], "STOP": ["2025-07-01"]})
    row = _run(meds=meds)
    assert row["adh_prior_n_med_fills"] == 0
    assert row["adh_prior_med_pdc"] == 0.0


def test_prior_adherence_counts_prior_fill_and_coverage():
    # a 30-day fill entirely inside the prior-365d window
    meds = pd.DataFrame({"PATIENT": ["p1"], "CODE": ["m1"],
                         "START": ["2025-05-01"], "STOP": ["2025-05-31"]})
    row = _run(meds=meds)
    assert row["adh_prior_n_med_fills"] == 1
    assert 0.0 < row["adh_prior_med_pdc"] <= 1.0


# --- engagement: engage_n_encounters / engage_n_bp_readings -------------------

def test_engagement_excludes_encounter_and_reading_on_index():
    encounters = pd.DataFrame({"PATIENT": ["p1"], "START": [INDEX]})
    vitals = pd.DataFrame({"PATIENT": ["p1"], "CODE": [SYSTOLIC_CODE], "DATE": [INDEX]})
    row = _run(encounters=encounters, vitals=vitals)
    assert row["engage_n_encounters"] == 0
    assert row["engage_n_bp_readings"] == 0


def test_engagement_counts_prior_encounter_and_reading():
    encounters = pd.DataFrame({"PATIENT": ["p1"], "START": ["2025-05-15"]})
    vitals = pd.DataFrame({"PATIENT": ["p1"], "CODE": [SYSTOLIC_CODE], "DATE": ["2024-01-01"]})
    row = _run(encounters=encounters, vitals=vitals)
    assert row["engage_n_encounters"] == 1
    assert row["engage_n_bp_readings"] == 1  # within the 1095d BP lookback


# --- payer churn: payer_n_switches -------------------------------------------

def test_payer_switch_excludes_transition_on_index():
    # a real switch (A->B) but the second record starts ON index -> not counted
    payers = pd.DataFrame({"PATIENT": ["p1", "p1"], "PAYER": ["A", "B"],
                           "START_DATE": ["2020-01-01", INDEX]})
    assert _run(payers=payers)["payer_n_switches"] == 0


def test_payer_switch_counts_prior_change():
    payers = pd.DataFrame({"PATIENT": ["p1", "p1", "p1"], "PAYER": ["A", "A", "B"],
                           "START_DATE": ["2019-01-01", "2020-01-01", "2021-01-01"]})
    # one change (A->B) among pre-index transitions; A->A is not a switch
    assert _run(payers=payers)["payer_n_switches"] == 1


# --- cross-drug mechanics: xdrug_* -------------------------------------------

def _rich_meds(rows: list[dict]) -> pd.DataFrame:
    """Build a meds frame with the optional ENCOUNTER / cost columns populated."""
    base = {"PATIENT": "p1", "STOP": "", "ENCOUNTER": "e1",
            "TOTALCOST": 100.0, "PAYER_COVERAGE": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_xdrug_excludes_fills_on_and_after_index():
    meds = _rich_meds([
        {"CODE": "statin", "START": INDEX},           # on index
        {"CODE": "statin", "START": "2025-07-01"},    # after index
    ])
    row = _run(meds=meds)
    assert row["xdrug_n_distinct_meds"] == 0
    assert row["xdrug_n_prescribers"] == 0
    assert row["xdrug_tenure_days"] == 0


def test_xdrug_excludes_antihypertensive_from_cross_drug():
    # The index drug class must not count toward cross-drug behavior.
    meds = _rich_meds([{"CODE": AH_CODE, "START": "2024-01-01"}])
    assert _run(meds=meds)["xdrug_n_distinct_meds"] == 0


def test_xdrug_tenure_and_distinct_and_prescribers():
    meds = _rich_meds([
        {"CODE": "statin", "START": "2023-06-01", "ENCOUNTER": "e1"},
        {"CODE": "metformin", "START": "2024-06-01", "ENCOUNTER": "e2"},
    ])
    row = _run(meds=meds)
    assert row["xdrug_n_distinct_meds"] == 2
    assert row["xdrug_n_prescribers"] == 2
    # tenure = index (2025-06-01) minus first rx (2023-06-01) = 731 days (incl. leap day)
    assert row["xdrug_tenure_days"] == 731


def test_xdrug_refill_gap_same_drug_only():
    meds = _rich_meds([
        {"CODE": "statin", "START": "2025-01-01"},
        {"CODE": "statin", "START": "2025-02-01"},   # +31d
        {"CODE": "statin", "START": "2025-04-02"},   # +60d
    ])
    row = _run(meds=meds)
    assert row["xdrug_refill_gap_max"] == 60
    assert row["xdrug_refill_gap_mean"] == pytest.approx((31 + 60) / 2)


def test_xdrug_refill_gap_nan_when_no_repeat():
    meds = _rich_meds([{"CODE": "statin", "START": "2025-01-01"}])
    row = _run(meds=meds)
    assert pd.isna(row["xdrug_refill_gap_mean"])
    assert pd.isna(row["xdrug_refill_gap_max"])


def test_xdrug_out_of_pocket_cost_mean():
    meds = _rich_meds([
        {"CODE": "statin", "START": "2025-01-01", "TOTALCOST": 100.0, "PAYER_COVERAGE": 40.0},
        {"CODE": "metformin", "START": "2025-02-01", "TOTALCOST": 50.0, "PAYER_COVERAGE": 50.0},
    ])
    # out-of-pocket = TOTALCOST - PAYER_COVERAGE = 60 and 0 -> mean 30
    assert _run(meds=meds)["xdrug_oop_cost_mean"] == pytest.approx(30.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
