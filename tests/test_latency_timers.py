"""test_latency_timers.py — latency-adjusted escalation timing.

No patient escalates inside the active source's latency window; the effective
boundary is exactly escalate_at + max_latency; swapping adapters (0 / 14 / 90 day
lag) yields three different escalation dates on the identical patient; and advancing
(compressed) time preserves round ordering.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing.consent import ConsentRecord, GRANTED  # noqa: E402
from src.routing.escalation import (  # noqa: E402
    compute_round_schedule, evaluate_patient, ROUNDS,
    STATUS_WAITING_ON_DATA_LATENCY,
)

ENTRY = date(2026, 1, 1)


def _card(days=170.0, driver="housing_barrier"):
    return {"patient_id": "p1", "days_to_predicted_break": days, "top_driver": driver,
            "is_safety_override": False, "predicted_risk": 0.3, "rank_in_role": 1}


def _consent():
    return ConsentRecord("p1", GRANTED, GRANTED, "AE", "2026-01-01")


def test_effective_boundary_is_exactly_escalate_plus_max_latency():
    for latency in (0, 14, 90):
        sched, _, _ = compute_round_schedule(ENTRY, 170, latency)
        for r in ROUNDS:
            _w, escalate_at, effective = sched[r]
            assert effective == escalate_at + timedelta(days=latency), \
                f"round {r} latency {latency}: effective != escalate + latency"


def test_adapter_swap_produces_three_different_escalation_dates():
    eff0 = {}
    for latency in (0, 14, 90):
        sched, _, _ = compute_round_schedule(ENTRY, 170, latency)
        eff0[latency] = sched[0][2]  # effective escalate of Round 0
    assert eff0[0] != eff0[14] != eff0[90] and eff0[0] != eff0[90], \
        f"latencies collapsed to the same escalation date: {eff0}"
    # and they order correctly: more lag -> later escalation
    assert eff0[0] < eff0[14] < eff0[90]


def test_no_escalation_inside_the_latency_window():
    latency = 60
    s0 = evaluate_patient(_card(170), None, _consent(), "src", latency, ENTRY, None)
    sched, _, _ = compute_round_schedule(ENTRY, 170, latency)
    _w0, escalate0, effective0 = sched[0]
    # a date strictly inside [escalate_at, effective_escalate_at) must NOT advance
    inside = escalate0 + timedelta(days=1)
    assert inside < effective0
    s = evaluate_patient(_card(170), None, _consent(), "src", latency, inside, s0)
    assert s.current_round == 0
    assert s.status == STATUS_WAITING_ON_DATA_LATENCY


def test_advancing_time_preserves_round_ordering():
    latency = 30
    s0 = evaluate_patient(_card(170), None, _consent(), "src", latency, ENTRY, None)
    rounds_seen = []
    prev_break = s0.predicted_break_date
    for offset in (0, 40, 80, 120, 160, 200, 260):
        s = evaluate_patient(_card(170), None, _consent(), "src", latency,
                             ENTRY + timedelta(days=offset), s0)
        rounds_seen.append(s.current_round)
        assert s.predicted_break_date == prev_break  # schedule is time-independent (frozen)
    # current_round is monotonic non-decreasing as time advances (never regresses)
    assert rounds_seen == sorted(rounds_seen)
    assert rounds_seen[0] == 0 and rounds_seen[-1] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
