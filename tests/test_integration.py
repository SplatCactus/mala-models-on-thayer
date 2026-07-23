"""test_integration.py — end-to-end: real model -> routing -> escalation timeline.

Part 1 fits the REAL primary model on a cohort subsample and routes it (proving the
model->routing path). Part 2 runs the escalation engine across a simulated
multi-week timeline over a controlled cohort whose patients refill at different
rounds, one is consent-gated, and one is held by data latency -- then asserts
closures, gating, latency, and funnel reconciliation all come out right together.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from src.models.common import PANEL_PATH, LABELS_PATH  # noqa: E402
from src.explain.shap_runner import SHAPRunner  # noqa: E402
from src.routing.rules import RoutingRuleEngine, VALID_ACTIONS  # noqa: E402
from src.routing.escalation import build_escalation_state, compute_round_schedule  # noqa: E402

ENTRY = date(2026, 1, 1)


# --- Part 1: the real model routes a real cohort subsample ------------------------

def test_real_model_routes_a_cohort_subsample():
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    frame = panel.merge(labels, on="patient_id", how="inner").dropna(
        subset=["has_30_day_gap"]).reset_index(drop=True).head(800)
    y = (frame["has_30_day_gap"].to_numpy() == 1).astype(int)

    runner = SHAPRunner()  # the deployed primary model
    runner.fit(frame, y)
    assert isinstance(runner.pipeline.named_steps["clf"], HistGradientBoostingClassifier)

    profiles = runner.explain_cohort(frame)
    decisions = RoutingRuleEngine().route_cohort(profiles)
    assert len(decisions) == len(frame)
    assert all(d.action in VALID_ACTIONS for d in decisions)          # every patient routed
    assert all(0.0 <= p.predicted_risk <= 1.0 for p in profiles)      # real risk scores


# --- Part 2: controlled multi-week escalation timeline ----------------------------

def _card(pid, driver, days=170.0):
    return {"patient_id": pid, "days_to_predicted_break": days, "top_driver": driver,
            "is_safety_override": False, "predicted_risk": 0.3, "rank_in_role": 1}


def _write_cohort(tmp: Path):
    """5 patients (fixed 170d window for deterministic round timing):
    three refill in rounds 0/1/2, one transport patient is external-consent-gated,
    one never refills (walks the ladder)."""
    sched, _, _ = compute_round_schedule(ENTRY, 170, 0)
    # event dates inside each round's active window
    ev0 = (sched[0][0] + timedelta(days=5)).isoformat()   # Round 0 window
    ev1 = (sched[1][0] + timedelta(days=5)).isoformat()   # Round 1 window
    ev2 = (sched[2][0] + timedelta(days=3)).isoformat()   # Round 2 window

    cards = [_card("close0", "housing_barrier"), _card("close1", "bp_trend"),
             _card("close2", "isolation"), _card("gate", "transport_barrier"),
             _card("open", "financial_barrier")]
    (tmp / "routing_table.json").write_text(json.dumps({"meta": {}, "capped_worklist": cards}))

    def refill(ev):
        return {"observed": True, "on_time_refill": True, "event_date": ev,
                "source": "local_file_synthetic", "refill_source": "local_file_synthetic",
                "refill_latency_days": 0}
    (tmp / "loop_outcomes.json").write_text(json.dumps({"outcomes": {
        "close0": refill(ev0), "close1": refill(ev1), "close2": refill(ev2)}}))
    # All patients hold internal consent (so internal actions aren't fail-closed);
    # only the transport patient's EXTERNAL consent is denied -> voucher gated -> fallback.
    granted = {"internal": "granted", "external": "granted", "source": "AE", "as_of": "2026-01-01"}
    (tmp / "consent.json").write_text(json.dumps({"patients": {
        "close0": granted, "close1": granted, "close2": granted, "open": granted,
        "gate": {"internal": "granted", "external": "denied", "source": "AE", "as_of": "2026-01-01"}}}))
    (tmp / "sync.json").write_text(json.dumps({"selected": {
        "source_name": "local_file_synthetic", "access_mode": "batch_permitted",
        "max_latency_days": 0, "latency_days": {"min": 0, "typical": 0, "max": 0}}}))


def _build(tmp, today):
    return build_escalation_state(
        routing_table_path=tmp / "routing_table.json",
        loop_outcomes_path=tmp / "loop_outcomes.json",
        consent_path=tmp / "consent.json",
        sync_state_path=tmp / "sync.json",
        state_path=tmp / "escalation_state.json",
        today=today, now=datetime(today.year, today.month, today.day, tzinfo=timezone.utc))


def test_end_to_end_timeline(tmp_path):
    _write_cohort(tmp_path)
    # First build at ENTRY freezes entry dates; then advance the simulated clock.
    open_rounds = []
    for offset in (0, 115, 145, 175):
        payload = _build(tmp_path, ENTRY + timedelta(days=offset))
        open_rounds.append(payload["patients"]["open"]["current_round"])

    final = payload["patients"]

    # closures land on the correct round
    assert final["close0"]["status"] == "CLOSED" and final["close0"]["closed_on_round"] == 0
    assert final["close1"]["status"] == "CLOSED" and final["close1"]["closed_on_round"] == 1
    assert final["close2"]["status"] == "CLOSED" and final["close2"]["closed_on_round"] == 2

    # consent gate: transport voucher gated, internal CHW fallback, never dropped
    gate = final["gate"]
    voucher = [g for g in gate["gated_actions"] if g["action"] == "transit_voucher"]
    assert voucher and voucher[0]["fallback_action"] == "chw_transport_support"
    assert gate["status"] in ("EXHAUSTED", "DISPATCHED", "WAIT_ELAPSED_DISPATCH_PENDING",
                              "WAITING_ON_DATA_LATENCY")

    # the never-refilling patient walks the ladder: round advances 0 -> 1 -> 2
    assert open_rounds == sorted(open_rounds)
    assert open_rounds[0] == 0 and open_rounds[-1] == 2
    assert final["open"]["status"] == "EXHAUSTED"

    # funnel reconciles: every patient in exactly one status and one round bucket
    meta = payload["meta"]
    assert meta["n_worklist"] == 5
    assert sum(meta["n_by_status"].values()) == 5
    assert sum(meta["n_by_current_round"].values()) == 5
    assert meta["n_closed"] == 3 == sum(meta["n_closed_by_round"].values())


def test_end_to_end_latency_holds_the_open_patient(tmp_path):
    """With a 90-day source, the open patient is held WAITING_ON_DATA_LATENCY inside the
    latency window rather than escalating."""
    _write_cohort(tmp_path)
    (tmp_path / "sync.json").write_text(json.dumps({"selected": {
        "source_name": "ae_pharmacy_claims_export", "access_mode": "batch_permitted",
        "max_latency_days": 90, "latency_days": {"min": 15, "typical": 30, "max": 90}}}))
    _build(tmp_path, ENTRY)  # freeze entry
    sched, _, _ = compute_round_schedule(ENTRY, 170, 90)
    _w0, escalate0, effective0 = sched[0]
    inside = escalate0 + timedelta(days=1)
    assert inside < effective0
    payload = _build(tmp_path, inside)
    openp = payload["patients"]["open"]
    assert openp["current_round"] == 0
    assert openp["status"] == "WAITING_ON_DATA_LATENCY"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
