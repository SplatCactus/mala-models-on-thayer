"""test_connectors.py — the pharmacy dispense-data connector layer.

Surescripts refuses batch queries (naming the contractual restriction), every
adapter declares a full SourceProfile, the fallback chain records every attempt,
and the local synthetic adapter reproduces the pre-change closure decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sync.connectors import (  # noqa: E402
    ACCESS_BATCH_PERMITTED, ACCESS_PRESCRIBER_INITIATED,
    ConnectorAccessError, ConnectorAuthError,
    AeClaimsConnector, LocalFileConnector, RxfillPushConnector, SurescriptsConnector,
)
from src.sync.connectors.base import EncounterContext  # noqa: E402
from src.sync.connectors.factory import select_connector  # noqa: E402
from src.models.common import LABELS_PATH  # noqa: E402


# --- Surescripts refuses batch; access-mode contract is enforced ------------------

def test_surescripts_refuses_batch_query_naming_the_restriction():
    ss = SurescriptsConnector(env={})
    with pytest.raises(ConnectorAccessError) as exc:
        ss.fetch_dispense_events(["p1", "p2", "p3"], "2026-01-01", "2026-07-01")
    msg = str(exc.value).lower()
    # the error must name WHY it's refused (prescriber-initiated, encounter, batch)
    assert "prescriber-initiated" in msg or "prescriber" in msg
    assert "encounter" in msg
    assert "batch" in msg and ("prohibit" in msg or "cannot" in msg)


def test_surescripts_single_patient_requires_matching_encounter():
    ss = SurescriptsConnector(env={})
    enc = EncounterContext(prescriber_spi="SPI123", encounter_id="E1", patient_id="p1")
    # even with an encounter, a live query is not implemented (intentional stub) --
    # but it must get PAST the access-mode guard for the matching single patient.
    with pytest.raises(NotImplementedError):
        ss.fetch_dispense_events(["p1"], "2026-01-01", "2026-07-01", encounter=enc)
    # a mismatched / multi-patient query is still refused
    with pytest.raises(ConnectorAccessError):
        ss.fetch_dispense_events(["p1", "p2"], "2026-01-01", "2026-07-01", encounter=enc)


def test_surescripts_auth_fails_without_credentials_naming_requirements():
    ss = SurescriptsConnector(env={})
    with pytest.raises(ConnectorAuthError) as exc:
        ss.authenticate()
    msg = str(exc.value).lower()
    assert "surescripts" in msg and ("certification" in msg or "credential" in msg or "missing" in msg)


# --- every adapter declares a full, coherent SourceProfile ------------------------

def test_every_adapter_declares_full_profile():
    for conn in (SurescriptsConnector(env={}), AeClaimsConnector(), RxfillPushConnector(env={}), LocalFileConnector()):
        p = conn.source_profile
        assert isinstance(p.source_name, str) and p.source_name
        assert p.access_mode in (ACCESS_BATCH_PERMITTED, ACCESS_PRESCRIBER_INITIATED)
        assert p.min_latency_days <= p.typical_latency_days <= p.max_latency_days
        assert isinstance(p.requires_encounter, bool)
        assert isinstance(p.confirms_dispense, bool)          # dispense-vs-prescription capability
        d = p.to_dict()
        assert d["latency_days"] == {"min": p.min_latency_days,
                                     "typical": p.typical_latency_days, "max": p.max_latency_days}


def test_access_modes_are_as_specified():
    assert SurescriptsConnector(env={}).source_profile.access_mode == ACCESS_PRESCRIBER_INITIATED
    assert SurescriptsConnector(env={}).source_profile.requires_encounter is True
    assert AeClaimsConnector().source_profile.access_mode == ACCESS_BATCH_PERMITTED
    assert AeClaimsConnector().source_profile.max_latency_days == 90
    assert LocalFileConnector().source_profile.max_latency_days == 0


# --- fallback chain records every attempt + which one served ----------------------

def test_fallback_chain_records_all_attempts_and_server():
    conn, trace = select_connector(env={})  # no creds, no AE export -> local serves
    assert conn.source_profile.source_name == "local_file_synthetic"
    adapters = [t["adapter"] for t in trace]
    assert adapters == ["surescripts", "ae_claims", "local"]
    assert [t["outcome"] for t in trace] == ["auth_failed", "auth_failed", "served"]
    # every failed attempt carries a non-empty reason; the server carries None
    assert all(t["reason"] for t in trace if t["outcome"] == "auth_failed")
    assert trace[-1]["reason"] is None
    assert sum(t["outcome"] == "served" for t in trace) == 1  # exactly one serves


# --- local adapter reproduces the pre-change closure decisions --------------------

def test_local_adapter_matches_baseline_closure_decisions():
    """LocalFileConnector's on-time decision == the old SyntheticOutcomeSource's
    (has_30_day_gap == 0). The source label intentionally changed; the DECISION did not."""
    from src.sync.loop_closure import SyntheticOutcomeSource, ConnectorOutcomeSource
    ids = [str(x) for x in pd.read_parquet(LABELS_PATH)["patient_id"].head(300)]
    baseline = SyntheticOutcomeSource().outcomes_for(ids)
    via_connector = ConnectorOutcomeSource(LocalFileConnector()).outcomes_for(ids)
    assert set(baseline) == set(via_connector)  # same covered set
    for pid in ids:
        assert baseline[pid].on_time_refill == via_connector[pid].on_time_refill
    # and the new provenance fields are populated
    sample = next(iter(via_connector.values()))
    assert sample.refill_source == "local_file_synthetic" and sample.refill_latency_days == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
