"""test_retrain_labels.py — the retraining seam, and its leakage guard.

Asserts (not trusts) that the retrain metadata columns closed_on_round / refill_source
can never be selected as model features, that the demo harvest carries the required
columns, and that refreshing labels flips the observed labels while introducing NO
feature or metadata columns (feature_panel stays strictly pre-index).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import retrain_labels as rl  # noqa: E402
from src.models.common import select_feature_columns  # noqa: E402


def test_retrain_metadata_never_a_model_feature():
    """closed_on_round / refill_source (post-index metadata) must never be features."""
    frame = pd.DataFrame({
        "patient_id": ["p1"],
        "age_years": [61.0],                 # real allowlisted feature
        "flag_sdoh_housing_barrier": [1],    # real allowlisted feature
        "closed_on_round": [1],              # retrain metadata — must be excluded
        "refill_source": ["ae_pharmacy_claims_export"],  # metadata — must be excluded
        "label_adherent": [1],               # a label — must be excluded
        "source_event_date": ["2026-07-01"], # metadata — must be excluded
        "loop_closed": [True],               # metadata — must be excluded
        "has_30_day_gap": [1],               # outcome — must be excluded
    })
    selected = set(select_feature_columns(frame))
    assert "age_years" in selected and "flag_sdoh_housing_barrier" in selected
    for meta_col in ("closed_on_round", "refill_source", "label_adherent",
                     "source_event_date", "loop_closed"):
        assert meta_col not in selected, f"{meta_col} leaked into model features"


def test_harvest_carries_required_columns_and_refresh_flips_labels(tmp_path):
    # Synthetic escalation state: p1 closed on round 1 (adherent), p2 confirmed break.
    esc = {"patients": {
        "p1": {"objective_outcome": {"observed": True, "on_time_refill": True,
                                     "event_date": "2026-07-01",
                                     "refill_source": "ae_pharmacy_claims_export",
                                     "closed_on_round": 1}},
        "p2": {"objective_outcome": {"observed": True, "on_time_refill": False,
                                     "event_date": "2026-07-02",
                                     "refill_source": "ae_pharmacy_claims_export",
                                     "closed_on_round": None}},
        "p3": {"objective_outcome": None},  # not observed -> not harvested
    }}
    esc_path = tmp_path / "escalation_state.json"
    esc_path.write_text(json.dumps(esc))
    retrain_path = tmp_path / "retrain_labels.parquet"

    rl.build_retraining_labels(mode="demo", escalation_state_path=esc_path,
                               loop_outcomes_path=tmp_path / "nope.json", out_path=retrain_path)
    harvested = pd.read_parquet(retrain_path)
    assert set(rl.RETRAIN_COLUMNS) == set(harvested.columns)
    assert len(harvested) == 2  # p3 (unobserved) excluded
    assert dict(zip(harvested["patient_id"], harvested["label_adherent"])) == {"p1": 1, "p2": 0}
    assert dict(zip(harvested["patient_id"], harvested["closed_on_round"]))["p1"] == 1

    # Refresh: p1 -> gap 0, p2 -> gap 1; p4 (unobserved) keeps its label; NO feature
    # or metadata columns appear in the refreshed labels file.
    labels = pd.DataFrame({"patient_id": ["p1", "p2", "p4"],
                           "pdc_180d": [0.9, 0.2, 0.5], "has_30_day_gap": [1, 0, 1]})
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)
    out = tmp_path / "labels_retrained.parquet"
    rl.refresh_labels(retrain_labels_path=retrain_path, labels_path=labels_path, out_path=out)
    refreshed = pd.read_parquet(out)
    assert list(refreshed.columns) == ["patient_id", "pdc_180d", "has_30_day_gap"]
    gap = dict(zip(refreshed["patient_id"], refreshed["has_30_day_gap"]))
    assert gap == {"p1": 0, "p2": 1, "p4": 1}  # observed flipped, unobserved unchanged
    for meta_col in ("closed_on_round", "refill_source"):
        assert meta_col not in refreshed.columns


def test_production_mode_raises():
    with pytest.raises(NotImplementedError):
        rl.build_retraining_labels(mode="production")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
