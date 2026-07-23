"""test_escalation_machine.py — state-machine requirements not covered in test_escalation.py.

Structural decoupling (no connector/model imports), restart round-trip via the
persisted file, closure at each round, edge windows, rounds-advance-only-on-failure,
and safety-override-still-gates-consent.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing.consent import ConsentRecord, GRANTED, UNKNOWN  # noqa: E402
from src.routing import escalation as esc  # noqa: E402
from src.routing.escalation import (  # noqa: E402
    compute_round_schedule, evaluate_patient, build_escalation_state,
    STATUS_CLOSED, STATUS_GATED_ON_CONSENT, STATUS_WAITING, STATUS_PENDING,
    MIN_ROUND_DWELL_DAYS,
)

ENTRY = date(2026, 1, 1)


def _card(days=170.0, driver="housing_barrier", safety=False, pid="p1"):
    return {"patient_id": pid, "days_to_predicted_break": days, "top_driver": driver,
            "is_safety_override": safety, "predicted_risk": 0.3, "rank_in_role": 1}


def _consent(internal=GRANTED, external=GRANTED):
    return ConsentRecord("p1", internal, external, "AE", "2026-01-01")


# --- structural decoupling: no concrete connector / model class imported ----------

def test_escalation_module_imports_no_connector_or_model_class():
    src = Path(esc.__file__).read_text()
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [f"{node.module}.{a.name}" for a in node.names]
    joined = " ".join(imported).lower()
    for forbidden in ("connectors", "surescripts", "ae_claims", "local_file",
                      "classifier", "shap_runner", "histgradientboosting",
                      "logisticregression", "sklearn"):
        assert forbidden not in joined, f"escalation.py must not import {forbidden!r}"


# --- starts at Round 0; advances only on failure ----------------------------------

def test_no_history_starts_at_round_0():
    s = evaluate_patient(_card(170), None, _consent(), "src", 0, ENTRY, None)
    assert s.current_round == 0
    assert s.status in (STATUS_WAITING, STATUS_PENDING)


def test_refill_closes_and_does_not_advance_with_later_time():
    """A refill closes at its round; later wall-clock does NOT push a closed loop on."""
    s0 = evaluate_patient(_card(170), None, _consent(), "src", 0, ENTRY, None)
    early_refill = {"observed": True, "on_time_refill": True,
                    "event_date": (ENTRY + timedelta(days=20)).isoformat(), "source": "x"}
    # evaluate far in the future — closure round must still reflect the EARLY refill.
    s = evaluate_patient(_card(170), early_refill, _consent(), "src", 0,
                         ENTRY + timedelta(days=300), s0)
    assert s.status == STATUS_CLOSED
    assert s.closed_on_round == 0  # refilled during Round 0, not advanced by time


def test_closed_on_round_reflects_the_round_active_at_the_refill():
    s0 = evaluate_patient(_card(170), None, _consent(), "src", 0, ENTRY, None)
    sched, _, _ = compute_round_schedule(ENTRY, 170, 0)
    # pick an event date inside each round's active window and assert the closure round
    for target_round in (0, 1, 2):
        wait_until = sched[target_round][0]
        event = (wait_until + timedelta(days=1)).isoformat()
        outcome = {"observed": True, "on_time_refill": True, "event_date": event, "source": "x"}
        s = evaluate_patient(_card(170), outcome, _consent(), "src", 0,
                             date.fromisoformat(event), s0)
        assert s.status == STATUS_CLOSED
        assert s.closed_on_round == target_round, f"expected close on round {target_round}"


# --- edge windows -----------------------------------------------------------------

def test_past_break_window_acts_now():
    # break already at/behind entry -> wait_until(0) clamps to entry -> acting immediately
    sched, break_date, _ = compute_round_schedule(ENTRY, -10, 0)
    assert break_date == ENTRY
    assert sched[0][0] == ENTRY
    s = evaluate_patient(_card(-10), None, _consent(), "src", 0, ENTRY, None)
    assert s.wait_elapsed is True


def test_short_window_does_not_collapse_rounds():
    sched, _, _ = compute_round_schedule(ENTRY, 5, 0)
    # each round still gets its dwell floor before escalating (no all-at-once collapse)
    for r in (0, 1, 2):
        w, e, _eff = sched[r]
        assert (e - w).days >= MIN_ROUND_DWELL_DAYS
    # rounds remain ordered
    assert sched[0][1] <= sched[1][0] and sched[1][1] <= sched[2][0]


def test_long_window_spaces_rounds_by_lead_days():
    sched, break_date, _ = compute_round_schedule(ENTRY, 170, 0)
    # Round 0 acts ~60d before the break, Round 2 ~10d before (front-loaded ladder)
    assert (break_date - sched[0][0]).days == 60
    assert (break_date - sched[2][0]).days == 10


# --- restart round-trip: persist, reload, rebuild -> identical --------------------

def _write_scenario(tmp: Path):
    cards = [_card(170, "housing_barrier", pid="a"),
             _card(60, "transport_barrier", pid="b"),
             _card(30, "bp_trend", pid="c")]
    (tmp / "routing_table.json").write_text(json.dumps(
        {"meta": {}, "capped_worklist": cards}))
    (tmp / "loop_outcomes.json").write_text(json.dumps({"outcomes": {
        "a": {"observed": True, "on_time_refill": True, "event_date": "2026-02-01",
              "source": "local_file_synthetic", "refill_source": "local_file_synthetic",
              "refill_latency_days": 0}}}))
    (tmp / "consent.json").write_text(json.dumps({"patients": {
        "b": {"internal": "granted", "external": "denied", "source": "AE", "as_of": "2026-01-01"}}}))
    (tmp / "sync.json").write_text(json.dumps({"selected": {
        "source_name": "local_file_synthetic", "access_mode": "batch_permitted",
        "max_latency_days": 0, "latency_days": {"min": 0, "typical": 0, "max": 0}}}))


def test_state_survives_restart_identically(tmp_path):
    _write_scenario(tmp_path)
    state_path = tmp_path / "escalation_state.json"
    kw = dict(routing_table_path=tmp_path / "routing_table.json",
              loop_outcomes_path=tmp_path / "loop_outcomes.json",
              consent_path=tmp_path / "consent.json",
              sync_state_path=tmp_path / "sync.json",
              state_path=state_path)
    t, n = date(2026, 3, 1), datetime(2026, 3, 1, tzinfo=timezone.utc)

    warm1 = build_escalation_state(**kw, today=t, now=n)          # cold (no prior)
    reloaded = json.loads(state_path.read_text())                 # persisted file
    warm2 = build_escalation_state(**kw, today=t, now=n)          # warm (prior exists)
    state_path.unlink()
    cold = build_escalation_state(**kw, today=t, now=n)           # cold again

    assert warm1["patients"] == reloaded["patients"]
    assert warm1["patients"] == warm2["patients"]
    assert warm1["patients"] == cold["patients"]                  # restart-safe


# --- safety override changes routing but does NOT bypass consent ------------------

def test_safety_override_still_gates_consent():
    # trauma safety -> Round 1 social worker (internal). With internal consent UNKNOWN,
    # the internal action is hard-blocked: the override does not bypass the gate.
    c = _consent(internal=UNKNOWN, external=UNKNOWN)
    s0 = evaluate_patient(_card(170, "trauma_exposure", safety=True), None, c, "src", 0, ENTRY, None)
    assert s0.current_round == 1  # skipped Round 0 (routing changed) ...
    # ... and at its immediate wait, the unknown internal consent gates it
    s = evaluate_patient(_card(170, "trauma_exposure", safety=True), None, c, "src", 0, ENTRY, s0)
    assert s.status == STATUS_GATED_ON_CONSENT
    assert any(g["scope"] == "internal_care_coordination" for g in s.gated_actions)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
