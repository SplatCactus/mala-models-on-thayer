"""test_consent.py — two-scope consent gating + the "never a model feature" guard.

Covers the fail-closed gating rules (granted/denied/unknown/stale, unknown distinct
from denied, missing record = unknown) and — the assertion the plan explicitly asks
for rather than trusting the docstring — that a consent-named column can never be
selected as a model feature by common.py's allowlist guard.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing.consent import (  # noqa: E402
    DENIED, EXTERNAL, GRANTED, INTERNAL, UNKNOWN,
    ConsentRecord, gate, is_stale, scope_status,
)

TODAY = date(2026, 7, 1)


def _rec(internal=GRANTED, external=GRANTED, as_of="2026-06-01"):
    return ConsentRecord(patient_id="p1", internal=internal, external=external,
                         source="AE (synthetic)", as_of=as_of)


# --- external gating: the scope that actually blocks ---------------------------

def test_external_granted_fresh_is_allowed():
    d = gate("transit_voucher", _rec(external=GRANTED), TODAY)
    assert d.scope == EXTERNAL and d.allowed and d.reason.startswith("authorized:")


def test_external_denied_is_gated_and_reason_says_denied():
    d = gate("transit_voucher", _rec(external=DENIED), TODAY)
    assert not d.allowed
    assert d.reason.startswith("denied:")


def test_external_unknown_is_gated_and_distinct_from_denied():
    d = gate("transit_voucher", _rec(external=UNKNOWN), TODAY)
    assert not d.allowed
    # "not received" must be distinguishable from "declined"
    assert d.reason.startswith("unknown_not_received:")
    assert "denied" not in d.reason


def test_external_granted_but_stale_fails_closed():
    d = gate("transit_voucher", _rec(external=GRANTED, as_of="2024-01-01"), TODAY)
    assert not d.allowed
    assert d.reason.startswith("stale_consent:")


def test_missing_record_is_unknown_not_denied_fail_closed():
    d = gate("transit_voucher", None, TODAY)
    assert not d.allowed
    assert d.reason.startswith("unknown_not_received:")


def test_missing_as_of_is_stale():
    assert is_stale(None, TODAY) is True


# --- internal scope: default-granted, blocks only when not authorized ----------

def test_internal_granted_allowed():
    d = gate("social_worker", _rec(internal=GRANTED), TODAY)
    assert d.scope == INTERNAL and d.allowed


def test_internal_unknown_fails_closed():
    d = gate("prescriber", _rec(internal=UNKNOWN), TODAY)
    assert not d.allowed and d.reason.startswith("unknown_not_received:")


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        gate("mail_the_patient", _rec(), TODAY)  # no such action / scope


def test_scope_status_surfaces_state_and_staleness_for_api():
    st = scope_status(_rec(external=GRANTED, as_of="2024-01-01"), EXTERNAL, TODAY)
    assert st["state"] == GRANTED and st["stale"] is True and st["allowed"] is False


# --- the guard the plan insists we assert, not trust ---------------------------

def test_consent_is_never_a_model_feature():
    """A consent-named column must never be admitted by select_feature_columns."""
    from src.models.common import select_feature_columns

    frame = pd.DataFrame({
        "patient_id": ["p1"],
        "age_years": [61.0],                     # a real allowlisted feature
        "flag_sdoh_housing_barrier": [1],        # a real allowlisted feature
        "internal_care_coordination": [1],       # consent — must be excluded
        "external_disclosure": [0],              # consent — must be excluded
        "consent_internal": [1],                 # consent — must be excluded
        "consent_external": [0],                 # consent — must be excluded
        "consent_as_of": ["2026-06-01"],         # consent — must be excluded
        "has_30_day_gap": [1],                   # outcome — must be excluded
    })
    selected = set(select_feature_columns(frame))
    assert "age_years" in selected and "flag_sdoh_housing_barrier" in selected
    for consent_col in ("internal_care_coordination", "external_disclosure",
                        "consent_internal", "consent_external", "consent_as_of"):
        assert consent_col not in selected, f"{consent_col} leaked into model features"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
