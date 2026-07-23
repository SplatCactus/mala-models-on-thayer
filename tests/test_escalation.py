"""test_escalation.py — the escalation state machine's core transitions.

Pure, deterministic, `now`/`today` injected. Fixtures are synthetic routing cards
(dicts), never a live model — so this suite is unaffected by the LogisticRegression
-> HGB model swap (decision #9). Covers: wait/dispatch/latency-guard/advance,
objective closure + confirmed break, the trauma Round-0 skip, consent gating (hard
internal block vs. external fallback), the frozen break date across a restart, and
deterministic reconstruction with no prior state.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing.consent import ConsentRecord, DENIED, GRANTED, UNKNOWN  # noqa: E402
from src.routing.escalation import (  # noqa: E402
    compute_round_schedule, evaluate_patient,
    STATUS_WAITING, STATUS_PENDING, STATUS_DISPATCHED, STATUS_WAITING_ON_DATA_LATENCY,
    STATUS_GATED_ON_CONSENT, STATUS_CLOSED, STATUS_EXHAUSTED,
)

ENTRY = date(2026, 1, 1)


def _card(days=170.0, driver="housing_barrier", safety=False, pid="p1"):
    return {"patient_id": pid, "days_to_predicted_break": days, "top_driver": driver,
            "is_safety_override": safety, "predicted_risk": 0.3, "rank_in_role": 1}


def _consent(internal=GRANTED, external=GRANTED, as_of="2026-01-01"):
    return ConsentRecord("p1", internal, external, "AE (synthetic)", as_of)


def _iso(d):
    return date.fromisoformat(d)


def _init(card, latency=0, consent=None):
    """State at entry (today == entry), the frozen baseline for later ticks."""
    return evaluate_patient(card, None, consent or _consent(), "src", latency, ENTRY, None)


def test_waiting_before_wait_until():
    s0 = _init(_card(170))
    assert s0.current_round == 0 and s0.status == STATUS_WAITING


def test_pending_when_wait_elapses():
    s0 = _init(_card(170))
    sched, _, _ = compute_round_schedule(ENTRY, 170, 0)
    w0 = _iso(sched[0][0].isoformat())
    s = evaluate_patient(_card(170), None, _consent(), "src", 0, w0, s0)
    assert s.current_round == 0 and s.status == STATUS_PENDING
    # the round-0 attempt is present but not yet stamped (pending)
    assert s.rounds[0].round == 0 and s.rounds[0].dispatched_at is None


def test_latency_guard_holds_then_releases():
    latency = 60
    s0 = _init(_card(170), latency=latency)
    sched, _, _ = compute_round_schedule(ENTRY, 170, latency)
    _w0, e0, eff0 = sched[0]
    # Past the escalate boundary but before the latency-adjusted one: hold.
    held = evaluate_patient(_card(170), None, _consent(), "src", latency, e0 + timedelta(days=1), s0)
    assert held.status == STATUS_WAITING_ON_DATA_LATENCY and held.current_round == 0
    # After the latency guard clears: advance to Round 1.
    released = evaluate_patient(_card(170), None, _consent(), "src", latency, eff0 + timedelta(days=1), s0)
    assert released.current_round == 1


def test_objective_refill_closes_and_records_round():
    s0 = _init(_card(170))
    event = (ENTRY + timedelta(days=30)).isoformat()
    outcome = {"observed": True, "on_time_refill": True, "event_date": event, "source": "x"}
    s = evaluate_patient(_card(170), outcome, _consent(), "src", 0, ENTRY + timedelta(days=30), s0)
    assert s.status == STATUS_CLOSED
    assert s.closed_on_round == 0
    assert s.objective_outcome["closed_on_round"] == 0


def test_confirmed_break_exhausts():
    s0 = _init(_card(170))
    outcome = {"observed": True, "on_time_refill": False, "event_date": ENTRY.isoformat(), "source": "x"}
    s = evaluate_patient(_card(170), outcome, _consent(), "src", 0, ENTRY + timedelta(days=5), s0)
    assert s.status == STATUS_EXHAUSTED and s.closed_on_round is None


def test_trauma_skips_round_0():
    s = _init(_card(170, driver="trauma_exposure", safety=True))
    assert s.current_round == 1                      # never round 0
    assert all(r.round != 0 for r in s.rounds)       # no round-0 attempt
    assert s.rounds[0].round == 1 and s.rounds[0].action == "social_worker"


def test_frozen_break_date_survives_changed_card():
    s0 = _init(_card(170))
    # Later tick: the model re-scored and the card now says 50 days — but the frozen
    # entry break date must not move (else timers would jitter with the model).
    s1 = evaluate_patient(_card(50), None, _consent(), "src", 0, ENTRY + timedelta(days=10), s0)
    assert s1.predicted_break_date == s0.predicted_break_date
    assert s1.days_to_break_at_entry == s0.days_to_break_at_entry


def test_external_gate_falls_back_never_dropped():
    c = _consent(external=DENIED)
    s0 = _init(_card(170, driver="transport_barrier"), consent=c)
    sched, _, _ = compute_round_schedule(ENTRY, 170, 0)
    w1 = sched[1][0]
    s = evaluate_patient(_card(170, driver="transport_barrier"), None, c, "src", 0, w1, s0)
    assert s.current_round == 1
    voucher = [g for g in s.gated_actions if g["action"] == "transit_voucher"]
    assert voucher and voucher[0]["fallback_action"] == "chw_transport_support"
    # NOT dropped and NOT hard-blocked: an external gate substitutes the internal
    # fallback and keeps acting (PENDING/DISPATCHED) — unlike an internal gate,
    # which hard-blocks to GATED_ON_CONSENT (see test_internal_unknown_hard_blocks).
    r1 = next(r for r in s.rounds if r.round == 1)
    assert r1.gated and r1.fallback_action == "chw_transport_support"
    assert s.status in (STATUS_PENDING, STATUS_DISPATCHED)


def test_internal_unknown_hard_blocks_on_consent():
    c = _consent(internal=UNKNOWN, external=UNKNOWN)
    s0 = _init(_card(170), consent=c)
    sched, _, _ = compute_round_schedule(ENTRY, 170, 0)
    w0 = sched[0][0]
    s = evaluate_patient(_card(170), None, c, "src", 0, w0, s0)
    assert s.status == STATUS_GATED_ON_CONSENT
    # internal block dispatches nothing for the active round
    assert next(r for r in s.rounds if r.round == 0).dispatched_at is None


def test_unactionable_when_latency_exceeds_window():
    s = _init(_card(170), latency=200)  # 200d lag >= 170d runway
    assert s.unactionable_in_time is True


def test_deterministic_reconstruction_without_prior():
    a = _init(_card(170))
    b = _init(_card(170))
    assert a.to_dict() == b.to_dict()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
