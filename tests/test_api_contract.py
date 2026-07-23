"""test_api_contract.py — the served worklist + escalation contract (API_CONTRACT.md).

Uses direct function calls (no HTTP server / httpx dependency). Asserts valid JSON,
the UNCHANGED pre-escalation row/top-level shapes the frontend depends on, the new
escalation block + endpoints, funnel reconciliation, a per-patient round-trip against
the state machine, and graceful serving when all optional snapshots are absent.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.api.main as api  # noqa: E402
from src.routing.escalation import (  # noqa: E402
    EscalationState, evaluate_patient, read_active_source_latency,
)
from src.routing.consent import load_consent  # noqa: E402

# Frozen "old contract": row keys the frontend rebuild depends on staying stable.
OLD_ROW_KEYS = {
    "patient_id", "display_name", "preferred_language", "break_window_start",
    "break_window_end", "risk_score", "top_driver", "driver_label_en", "driver_label_es",
    "driver_label_pt", "driver_label_ht", "routed_action", "outreach_script_en",
    "outreach_script_es", "outreach_script_pt", "outreach_script_ht",
    "requires_human_review", "is_safety_override", "priority_score", "loop_outcome",
}
OLD_TOP_KEYS = {"generated_at", "data_source", "last_synced", "cohort_size", "capacity",
                "role_caps_used", "role_capacity_expansions", "worklist"}


def test_worklist_is_valid_json_with_all_documented_fields():
    data = api.load_worklist()
    json.dumps(data)  # must be JSON-serializable
    assert OLD_TOP_KEYS <= set(data)
    assert {"escalation_funnel", "pharmacy_source"} <= set(data)  # new top-level
    row = data["worklist"][0]
    assert OLD_ROW_KEYS <= set(row)
    assert {"dispatch_message_en", "chw_read_aloud_script_en", "escalation"} <= set(row)


def test_existing_field_shapes_unchanged():
    row = api.load_worklist()["worklist"][0]
    assert isinstance(row["patient_id"], str)
    assert row["display_name"].startswith("Patient #")
    assert row["preferred_language"] == "en"
    assert isinstance(row["risk_score"], (int, float))
    assert row["routed_action"] in {"pharmacist", "social_worker", "chw_call"}
    for k in ("driver_label_en", "outreach_script_en"):
        assert isinstance(row[k], str) and row[k]
    assert isinstance(row["requires_human_review"], bool)
    assert isinstance(row["is_safety_override"], bool)
    assert row["loop_outcome"] is None or isinstance(row["loop_outcome"], dict)
    # deprecated alias points at the same value as the renamed key
    assert row["outreach_script_en"] == row["dispatch_message_en"]


def test_escalation_block_shape():
    row = next(r for r in api.load_worklist()["worklist"] if r["escalation"])
    e = row["escalation"]
    for k in ("current_round", "round_label", "status", "days_remaining",
              "predicted_break_date", "consent_scopes", "gated_actions",
              "dispatch_history", "current_dispatch", "objective_outcome",
              "unactionable_in_time"):
        assert k in e, f"escalation block missing {k}"
    assert set(e["round_label"]) == {"en", "es", "pt", "ht"}
    assert set(e["consent_scopes"]) == {"internal_care_coordination", "external_disclosure"}


def test_new_endpoints_shapes_and_404():
    pid = api.load_worklist()["worklist"][0]["patient_id"]
    rec = api.get_escalation(pid)
    assert {"current_round", "status", "rounds", "consent", "current_dispatch"} <= set(rec)
    summary = api.get_escalation_summary()
    assert summary["escalation_funnel"] is not None
    assert summary["pharmacy_source"] is not None
    assert len(summary["pharmacy_source"]["fallback_trace"]) >= 1
    with pytest.raises(HTTPException) as exc:
        api.get_escalation("no-such-patient")
    assert exc.value.status_code == 404


def test_funnel_counts_reconcile():
    meta = json.loads((ROOT / "data" / "snapshots" / "escalation_state.json").read_text())["meta"]
    n = meta["n_worklist"]
    # every patient is in exactly one status bucket and one round bucket
    assert sum(meta["n_by_status"].values()) == n
    assert sum(meta["n_by_current_round"].values()) == n
    # closed count reconciles with the per-round closed breakdown
    assert meta["n_closed"] == sum(meta["n_closed_by_round"].values())


def test_every_patient_round_trips_against_state_machine():
    """The served escalation state is a fixed point of the state machine: re-evaluating
    a patient from the same inputs (with its persisted state as prior) reproduces its
    current_round / status / closed_on_round."""
    rt = json.loads((ROOT / "data" / "snapshots" / "routing_table.json").read_text())
    cards = {c["patient_id"]: c for c in rt["capped_worklist"]}
    payload = json.loads((ROOT / "data" / "snapshots" / "escalation_state.json").read_text())
    meta, patients = payload["meta"], payload["patients"]
    today = date.fromisoformat(meta["today"])
    source_name, max_latency = read_active_source_latency()
    outcomes = json.loads((ROOT / "data" / "snapshots" / "loop_outcomes.json").read_text())["outcomes"]
    consent = load_consent()

    sample = list(patients.items())[:50]
    for pid, served in sample:
        prior = EscalationState.from_dict(served)
        recomputed = evaluate_patient(cards[pid], outcomes.get(pid), consent.get(pid),
                                      source_name, max_latency, today, prior)
        assert recomputed.current_round == served["current_round"]
        assert recomputed.status == served["status"]
        assert recomputed.closed_on_round == served["closed_on_round"]


def test_serves_with_all_optional_snapshots_absent(monkeypatch):
    missing = ROOT / "data" / "snapshots" / "does_not_exist.json"
    monkeypatch.setattr(api, "ESCALATION_STATE_PATH", missing)
    monkeypatch.setattr(api, "LOOP_OUTCOMES_PATH", missing)
    monkeypatch.setattr(api, "PHARMACY_SYNC_PATH", missing)
    data = api.load_worklist()  # routing_table.json still present -> must not crash
    assert data["escalation_funnel"] is None
    assert data["pharmacy_source"] is None
    row = data["worklist"][0]
    assert OLD_ROW_KEYS <= set(row)          # existing fields still served
    assert row["escalation"] is None
    assert row["loop_outcome"] is None
    # dispatch_message_* falls back to the routing rationale (no escalation state)
    assert isinstance(row["dispatch_message_en"], str) and row["dispatch_message_en"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
