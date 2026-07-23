"""
src/routing/escalation.py

The escalation-ladder state machine. Given a routed patient (a routing_table.json
worklist card), the objective refill outcome we have so far, the patient's consent,
and the current date, it computes which round the patient is in, whether the
current round's wait has elapsed, and (via the round -> action map) what action is
due -- resolving consent and the data-latency guard first.

MODEL-AGNOSTIC BY CONTRACT (constraint / decision #9)
-----------------------------------------------------
A separate workstream swaps the risk model (LogisticRegression -> HGB), changing
risk scores AND the SHAP driver mix. This module therefore depends ONLY on the
routing card's fields (`predicted_risk`, `top_driver`, `days_to_predicted_break`,
`is_safety_override`) -- it never imports a model, assumes no score range, and
tolerates any `top_driver` (unmapped drivers fall back to the social worker). It
also imports NO concrete connector: the active source's latency comes from the
persisted `pharmacy_sync_state.json` the connector factory wrote, so escalation is
decoupled from which pharmacy source served.

FULLY DERIVABLE FROM PERSISTED DATA
-----------------------------------
Everything here is a pure function of (routing_table.json, loop_outcomes.json,
consent.json, pharmacy_sync_state.json, and `today`). The schedule is deterministic
from the frozen entry date + break window, so if the process restarts with no
escalation_state.json at all, the same current round and history reconstruct from
those inputs. Prior state only preserves the real wall-clock dispatch timestamps
and the frozen entry/break dates. Nothing is client-side; `now`/`today` are
injected (matching loop_closure.py's DI style) so the machine is deterministic and
testable.

THE LADDER
----------
Round 0 (CHW -> pharmacy, all patients) -> Round 1 (SDOH-specific) -> Round 2
(escalate to the AE prescriber). Success at any round = an objective dispense
before the predicted break date -> CLOSED + closed_on_round. Rounds advance only
on failure (no refill) AND only after the data-latency guard clears. A trauma
safety override skips Round 0 and enters Round 1 with the social worker +
requires_human_review (preserving rules.py's behavior).

Public API
----------
compute_round_schedule()   frozen entry + break window + source latency -> per-round timers
RoundAttempt / EscalationState   dataclasses (JSON-round-trippable)
evaluate_patient()         pure, read-only: card + outcome + consent + latency + today -> state
build_escalation_state()   CLI/persist: routing_table.json (+ loop_outcomes/consent/sync-state)
                           -> data/snapshots/escalation_state.json (additive, merged)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.routing.consent import (  # noqa: E402
    ACTION_REQUIRED_SCOPE,
    EXTERNAL,
    INTERNAL,
    ConsentRecord,
    gate,
    load_consent,
    scope_status,
)
from src.routing.dispatch_messages import build_dispatch  # noqa: E402

ROUTING_TABLE_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
LOOP_OUTCOMES_PATH = ROOT / "data" / "snapshots" / "loop_outcomes.json"
CONSENT_PATH = ROOT / "data" / "snapshots" / "consent.json"
SYNC_STATE_PATH = ROOT / "data" / "snapshots" / "pharmacy_sync_state.json"
STATE_PATH = ROOT / "data" / "snapshots" / "escalation_state.json"

# ---------------------------------------------------------------------------
# WAIT-PERIOD POLICY (tunable). See compute_round_schedule() for the reasoning.
# ---------------------------------------------------------------------------
FORWARD_WINDOW_DAYS = 180
# Days before the predicted break each round should act. Front-loaded: the cheap
# Round 0 fires earliest, each later round moves closer to the break.
ROUND_LEAD_DAYS = {0: 60, 1: 30, 2: 10}
# Floor on how long a round stays open before it can escalate, so a short or
# already-past break window cannot collapse all three rounds into one instant.
MIN_ROUND_DWELL_DAYS = 7

# Rounds.
ROUND_0_CHW_PHARMACY = 0
ROUND_1_SDOH = 1
ROUND_2_PRESCRIBER = 2
ROUNDS = (ROUND_0_CHW_PHARMACY, ROUND_1_SDOH, ROUND_2_PRESCRIBER)
LAST_ROUND = ROUND_2_PRESCRIBER

# Statuses.
STATUS_WAITING = "WAITING"
STATUS_PENDING = "WAIT_ELAPSED_DISPATCH_PENDING"
STATUS_DISPATCHED = "DISPATCHED"
STATUS_GATED_ON_CONSENT = "GATED_ON_CONSENT"
STATUS_WAITING_ON_DATA_LATENCY = "WAITING_ON_DATA_LATENCY"
STATUS_CLOSED = "CLOSED"
STATUS_EXHAUSTED = "EXHAUSTED"

# Round -> action (for consent gating + history). Round 1 depends on the dominant
# SHAP driver; an unmapped driver falls back to the social worker rather than
# raising (model-agnostic -- a model swap may surface a new driver). transit_voucher
# is the only external action; if its consent is gated the engine substitutes the
# internal fallback below.
ROUND1_DRIVER_ACTION = {
    "transport_barrier": "transit_voucher",
    "financial_barrier": "pharmacist",
    "bp_trend": "pharmacist",
    "housing_barrier": "social_worker",
    "isolation": "social_worker",
    "trauma_exposure": "social_worker",
    "low_education": "bilingual_chw",
    "migrant_status": "bilingual_chw",
}
ROUND1_FALLBACK_ACTION = "social_worker"
EXTERNAL_FALLBACK_ACTION = {"transit_voucher": "chw_transport_support"}


def _round_action(round_num: int, top_driver: str, is_safety: bool) -> str:
    if round_num == ROUND_0_CHW_PHARMACY:
        return "chw_pharmacy"
    if round_num == ROUND_1_SDOH:
        if is_safety:  # trauma safety override -> social worker (see rules.py)
            return "social_worker"
        return ROUND1_DRIVER_ACTION.get(top_driver, ROUND1_FALLBACK_ACTION)
    if round_num == ROUND_2_PRESCRIBER:
        return "prescriber"
    raise ValueError(f"unknown escalation round {round_num!r}")


def _parse_date(value) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _iso_midnight(day: date) -> str:
    """Deterministic UTC-midnight stamp for a backfilled dispatch (the moment the
    round became dispatchable), used when reconstructing history with no prior state."""
    return f"{day.isoformat()}T00:00:00Z"


def read_active_source_latency(sync_state_path: Path = SYNC_STATE_PATH) -> Tuple[str, int]:
    """Active pharmacy source name + its MAX latency (days), from the persisted
    sync-state the connector factory wrote. No connector import -- escalation stays
    decoupled from the concrete source. Absent file -> (unknown, 0): degrade to no
    latency guard (the local synthetic source is genuinely zero-latency anyway)."""
    p = Path(sync_state_path)
    if not p.exists():
        return "unknown (no pharmacy_sync_state.json)", 0
    with open(p) as f:
        sel = json.load(f).get("selected", {})
    max_latency = sel.get("max_latency_days")
    if max_latency is None:
        max_latency = sel.get("latency_days", {}).get("max", 0)
    return sel.get("source_name", "unknown"), int(max_latency or 0)


def compute_round_schedule(
    entry_date: date, days_to_break: float, source_max_latency_days: int
) -> Tuple[Dict[int, Tuple[date, date, date]], date, bool]:
    """Derive each round's (wait_until, escalate_at, effective_escalate_at) + the
    frozen break date + an ``unactionable_in_time`` flag.

    The wait is NOT a fixed constant: it is anchored to the patient's own predicted
    break date (``entry_date + days_to_break``). Per round:
      * ``wait_until``  = break_date - ROUND_LEAD_DAYS[r]  (when round r acts)
      * ``escalate_at`` = the next round's ideal wait (Round 2 escalates at the
        break date), floored at ``wait_until + MIN_ROUND_DWELL_DAYS``
      * ``effective_escalate_at`` = escalate_at + the confirming source's MAX
        latency. This is the latency guard: we do not escalate for "no refill seen"
        until the source would plausibly have surfaced a refill (constraint C2).
    Rounds chain on the *effective* boundary, so a laggy source pushes the whole
    ladder out rather than escalating through a patient before their refill could
    have shown up.

    Edge cases, all explicit:
      * BREAK ALREADY PAST / TODAY: ``days_to_break`` clamped to >= 0, so the break
        date is never behind entry; lead-based waits clamp up to entry -> act now.
      * VERY SHORT WINDOW: leads exceed the window, so the MIN_ROUND_DWELL_DAYS floor
        spaces rounds a week apart instead of firing all at once.
      * VERY LONG WINDOW: leads fit; natural break-60/30/10 cadence.
      * LATENCY EXCEEDS THE WINDOW: if the source's max latency >= the whole runway,
        we cannot confirm a refill before the break -> ``unactionable_in_time=True``
        (surfaced honestly; we still send the immediate Round 0, but we do not
        pretend the latency-gated escalation can complete in time).
    """
    runway = max(days_to_break, 0.0)
    break_date = entry_date + timedelta(days=runway)

    schedule: Dict[int, Tuple[date, date, date]] = {}
    prev_boundary = entry_date  # the next round can't start before the prior's effective escalate
    for r in ROUNDS:
        ideal_wait = break_date - timedelta(days=ROUND_LEAD_DAYS[r])
        wait_until = max(ideal_wait, entry_date, prev_boundary)
        next_lead = ROUND_LEAD_DAYS.get(r + 1, 0)  # Round 2 escalates at the break date
        ideal_escalate = break_date - timedelta(days=next_lead)
        escalate_at = max(ideal_escalate, wait_until + timedelta(days=MIN_ROUND_DWELL_DAYS))
        effective_escalate_at = escalate_at + timedelta(days=source_max_latency_days)
        schedule[r] = (wait_until, escalate_at, effective_escalate_at)
        prev_boundary = effective_escalate_at

    # Un-actionable in time: the confirming source's max lag meets/exceeds the whole
    # runway to the predicted break, so a refill can't be confirmed before it.
    unactionable_in_time = runway > 0 and source_max_latency_days >= runway
    return schedule, break_date, unactionable_in_time


def _round_for(when: date, schedule: Dict[int, Tuple[date, date, date]], min_round: int) -> int:
    """Highest round reached by ``when``: advance past a round only once its
    latency-adjusted (effective) escalate boundary has passed."""
    r = min_round
    while r < LAST_ROUND and when >= schedule[r][2]:
        r += 1
    return r


@dataclass
class RoundAttempt:
    """One round on the ladder for one patient: its timers, action, consent, outcome."""
    round: int
    action: str
    required_scope: str
    wait_until: str
    escalate_at: str
    effective_escalate_at: str
    dispatched_at: Optional[str]
    gated: bool
    consent_reason: Optional[str]
    fallback_action: Optional[str]
    outcome: str  # pending | no_refill | refill_observed | gated | gated_internal
    # Provider-addressed dispatch payload for this round (recipient + 4-language body
    # + optional read-aloud script), from src/routing/dispatch_messages.build_dispatch.
    dispatch: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoundAttempt":
        return cls(
            round=d["round"], action=d["action"], required_scope=d["required_scope"],
            wait_until=d["wait_until"], escalate_at=d["escalate_at"],
            effective_escalate_at=d["effective_escalate_at"],
            dispatched_at=d.get("dispatched_at"), gated=d.get("gated", False),
            consent_reason=d.get("consent_reason"), fallback_action=d.get("fallback_action"),
            outcome=d.get("outcome", "pending"), dispatch=d.get("dispatch") or {},
        )


@dataclass
class EscalationState:
    """Full escalation state for one patient -- the persisted, restart-safe unit."""
    patient_id: str
    entry_date: str
    days_to_break_at_entry: float
    predicted_break_date: str
    is_safety_override: bool
    source_name: str
    source_max_latency_days: int
    unactionable_in_time: bool
    current_round: int
    status: str
    wait_elapsed: bool
    closed_on_round: Optional[int]
    objective_outcome: Optional[dict]
    consent: dict
    gated_actions: List[dict]
    current_dispatch: dict = field(default_factory=dict)
    rounds: List[RoundAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EscalationState":
        return cls(
            patient_id=d["patient_id"],
            entry_date=d["entry_date"],
            days_to_break_at_entry=d["days_to_break_at_entry"],
            predicted_break_date=d["predicted_break_date"],
            is_safety_override=d.get("is_safety_override", False),
            source_name=d.get("source_name", "unknown"),
            source_max_latency_days=d.get("source_max_latency_days", 0),
            unactionable_in_time=d.get("unactionable_in_time", False),
            current_round=d["current_round"],
            status=d["status"],
            wait_elapsed=d.get("wait_elapsed", False),
            closed_on_round=d.get("closed_on_round"),
            objective_outcome=d.get("objective_outcome"),
            consent=d.get("consent", {}),
            gated_actions=d.get("gated_actions", []),
            current_dispatch=d.get("current_dispatch") or {},
            rounds=[RoundAttempt.from_dict(r) for r in d.get("rounds", [])],
        )


def evaluate_patient(
    card: dict,
    outcome: Optional[dict],
    consent_record: Optional[ConsentRecord],
    source_name: str,
    source_max_latency_days: int,
    today: date,
    prior: Optional[EscalationState] = None,
) -> EscalationState:
    """Compute a patient's escalation state (pure, read-only, no I/O).

    A freshly wait-elapsed round is reported as PENDING (dispatched_at None);
    ``build_escalation_state`` records the dispatch and flips it to DISPATCHED.
    Consent is resolved at the active round: an INTERNAL action that isn't
    authorized hard-blocks (GATED_ON_CONSENT, nothing dispatched); an EXTERNAL
    action that isn't authorized substitutes the internal fallback (still
    dispatched) and is recorded in ``gated_actions`` -- never a silent drop.
    """
    pid = str(card["patient_id"])
    top_driver = card.get("top_driver", "")
    is_safety = bool(card.get("is_safety_override", False))

    # Freeze entry date + break window on first sight; reuse thereafter.
    if prior is not None:
        entry_date = _parse_date(prior.entry_date)
        days_at_entry = float(prior.days_to_break_at_entry)
    else:
        entry_date = today
        days_at_entry = float(card.get("days_to_predicted_break", 0.0))

    schedule, break_date, unactionable = compute_round_schedule(
        entry_date, days_at_entry, source_max_latency_days)
    if is_safety:
        # Trauma safety case skips Round 0 and must be seen NOW (mandatory human
        # review can't wait out a break-window countdown): Round 1 wait is immediate.
        _, e1, eff1 = schedule[ROUND_1_SDOH]
        schedule[ROUND_1_SDOH] = (entry_date, e1, eff1)
    min_round = ROUND_1_SDOH if is_safety else ROUND_0_CHW_PHARMACY

    prior_dispatch = {a.round: a.dispatched_at for a in prior.rounds} if prior else {}

    observed = bool(outcome and outcome.get("observed"))
    on_time = bool(outcome and outcome.get("on_time_refill"))
    event_date = _parse_date(outcome["event_date"]) if (outcome and outcome.get("event_date")) else None
    ref_date = (event_date or today) if observed else today

    current_round = _round_for(ref_date, schedule, min_round)
    w, e, eff = schedule[current_round]

    # ---- top-level status --------------------------------------------------
    if observed and on_time:
        # Objective dispense -> loop closed at the round active when it landed.
        status = STATUS_CLOSED
        closed_on_round = current_round
    elif observed and not on_time:
        # Objective confirmed break -> the ladder failed for this patient. Terminal.
        status = STATUS_EXHAUSTED
        closed_on_round = None
    else:
        closed_on_round = None
        if current_round == LAST_ROUND and today >= eff:
            status = STATUS_EXHAUSTED  # walked the whole ladder, still no refill
        elif today >= w:
            active_action = _round_action(current_round, top_driver, is_safety)
            active_decision = gate(active_action, consent_record, today)
            if not active_decision.allowed and active_decision.scope == INTERNAL:
                status = STATUS_GATED_ON_CONSENT  # internal hard-block: nothing dispatched
            elif today >= e and today < eff:
                status = STATUS_WAITING_ON_DATA_LATENCY  # dispatched; holding for latency
            else:
                status = STATUS_DISPATCHED if prior_dispatch.get(current_round) else STATUS_PENDING
        else:
            status = STATUS_WAITING

    wait_elapsed = ref_date >= schedule[current_round][0]

    # Per-scope consent summary + refill provenance, computed once so the round loop
    # can localize dispatch messages and flag external gating consistently.
    cr = consent_record or ConsentRecord.for_patient(pid, None)
    consent_summary = {
        INTERNAL: scope_status(cr, INTERNAL, today),
        EXTERNAL: scope_status(cr, EXTERNAL, today),
    }
    break_date_str = break_date.isoformat()
    refill_meta = {
        "days_to_break": days_at_entry,
        "predicted_break_date": break_date_str,
        "refill_source": (outcome or {}).get("refill_source") or source_name,
        "refill_latency_days": (outcome or {}).get("refill_latency_days", source_max_latency_days),
    }

    # ---- reconstruct round history min_round..current_round ----------------
    rounds: List[RoundAttempt] = []
    gated_actions: List[dict] = []
    prior_summaries: List[dict] = []   # feeds the Round-2 "what we already tried" history
    for r in range(min_round, current_round + 1):
        rw, re, reff = schedule[r]
        action = _round_action(r, top_driver, is_safety)
        scope = ACTION_REQUIRED_SCOPE[action]
        decision = gate(action, consent_record, today)
        is_external = scope == EXTERNAL
        gated = not decision.allowed
        fallback = EXTERNAL_FALLBACK_ACTION.get(action) if (gated and is_external) else None

        if r < current_round:
            wait_reached = True
            if gated and not is_external:
                acted, outcome_word = False, "gated_internal"  # internal block: never acted
            else:
                acted, outcome_word = True, "no_refill"        # dispatched (or fallback), moved on
        else:  # r == current_round
            wait_reached = ref_date >= rw
            if status == STATUS_CLOSED:
                outcome_word = "refill_observed"
            elif status == STATUS_GATED_ON_CONSENT:
                outcome_word = "gated"
            elif status == STATUS_EXHAUSTED:
                outcome_word = "no_refill"
            elif status == STATUS_WAITING:
                outcome_word = "pending"
            else:  # DISPATCHED / PENDING / WAITING_ON_DATA_LATENCY
                outcome_word = "pending"
            if gated and not is_external:
                acted = False
            else:
                acted = wait_reached and status != STATUS_WAITING

        if not acted:
            dispatched_at = None
        elif status == STATUS_PENDING and r == current_round:
            dispatched_at = None  # elapsed but not yet recorded (build flips this)
        else:
            dispatched_at = prior_dispatch.get(r) or _iso_midnight(rw)

        if gated:
            gated_actions.append({
                "round": r, "action": action, "scope": scope,
                "reason": decision.reason, "fallback_action": fallback,
            })

        # Provider-addressed dispatch payload; Round 2 composes the prior-round
        # history accumulated in prior_summaries below.
        dispatch = build_dispatch(r, card, prior_summaries, consent_summary, refill_meta)

        rounds.append(RoundAttempt(
            round=r, action=action, required_scope=scope,
            wait_until=rw.isoformat(), escalate_at=re.isoformat(),
            effective_escalate_at=reff.isoformat(),
            dispatched_at=dispatched_at, gated=gated,
            consent_reason=decision.reason if gated else None,
            fallback_action=fallback, outcome=outcome_word, dispatch=dispatch,
        ))
        prior_summaries.append({
            "round": r,
            "recipient_type": dispatch.get("recipient_type"),
            "dispatched_on": dispatched_at[:10] if dispatched_at else None,
            "outcome": outcome_word,
        })

    objective_outcome = None
    if observed:
        objective_outcome = dict(outcome)
        objective_outcome["closed_on_round"] = closed_on_round

    current_dispatch = rounds[-1].dispatch if rounds else {}

    return EscalationState(
        patient_id=pid,
        entry_date=entry_date.isoformat(),
        days_to_break_at_entry=round(days_at_entry, 2),
        predicted_break_date=break_date.isoformat(),
        is_safety_override=is_safety,
        source_name=source_name,
        source_max_latency_days=source_max_latency_days,
        unactionable_in_time=unactionable,
        current_round=current_round,
        status=status,
        wait_elapsed=wait_elapsed,
        closed_on_round=closed_on_round,
        objective_outcome=objective_outcome,
        consent=consent_summary,
        gated_actions=gated_actions,
        current_dispatch=current_dispatch,
        rounds=rounds,
    )


def _record_dispatch(state: EscalationState, now: datetime) -> None:
    """Emit the current round's pending dispatch: stamp it and flip to DISPATCHED.

    Recording the dispatch into escalation_state.json IS the dispatch record (there
    is no external send in this layer). Kept out of evaluate_patient so that stays
    read-only and a live consumer can still observe the PENDING window.
    """
    stamp = now.isoformat().replace("+00:00", "Z")
    for attempt in state.rounds:
        if attempt.round == state.current_round:
            attempt.dispatched_at = stamp
            break
    state.status = STATUS_DISPATCHED


def _capped_worklist(routing_table_path: Path) -> List[dict]:
    with open(routing_table_path) as f:
        return json.load(f).get("capped_worklist", [])


def _load_outcomes(loop_outcomes_path: Path) -> dict:
    if not Path(loop_outcomes_path).exists():
        return {}
    with open(loop_outcomes_path) as f:
        return json.load(f).get("outcomes", {})


def _load_prior_states(state_path: Path) -> Dict[str, EscalationState]:
    if not Path(state_path).exists():
        return {}
    with open(state_path) as f:
        payload = json.load(f)
    return {pid: EscalationState.from_dict(d) for pid, d in payload.get("patients", {}).items()}


def build_escalation_state(
    routing_table_path: Path = ROUTING_TABLE_PATH,
    loop_outcomes_path: Path = LOOP_OUTCOMES_PATH,
    consent_path: Path = CONSENT_PATH,
    sync_state_path: Path = SYNC_STATE_PATH,
    state_path: Path = STATE_PATH,
    *,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
    sim_days_per_tick: Optional[int] = None,
) -> dict:
    """Advance every worklist patient one step and persist escalation_state.json.

    ADDITIVE + MERGED, not overwritten: routing_table.json is rewritten wholesale
    by sync_job.py each run, so escalation state lives in its OWN file and MERGES
    each patient's prior frozen entry date / break window / dispatch timestamps
    forward. routing_table.json (and thus fairness.py's eligible_pool /
    capped_worklist) is never touched here.
    """
    today = today or date.today()
    now = now or datetime.now(timezone.utc)

    cards = _capped_worklist(routing_table_path)
    outcomes = _load_outcomes(loop_outcomes_path)
    consent_records = load_consent(consent_path)
    source_name, max_latency = read_active_source_latency(sync_state_path)
    prior_states = _load_prior_states(state_path)

    patients: Dict[str, dict] = {}
    by_status: Dict[str, int] = {}
    by_round: Dict[str, int] = {}
    closed_by_round: Dict[str, int] = {}
    n_gated = n_unactionable = 0
    for card in cards:
        pid = str(card["patient_id"])
        state = evaluate_patient(
            card, outcomes.get(pid), consent_records.get(pid),
            source_name, max_latency, today, prior_states.get(pid))
        if state.status == STATUS_PENDING:
            _record_dispatch(state, now)

        patients[pid] = state.to_dict()
        by_status[state.status] = by_status.get(state.status, 0) + 1
        by_round[str(state.current_round)] = by_round.get(str(state.current_round), 0) + 1
        if state.closed_on_round is not None:
            k = str(state.closed_on_round)
            closed_by_round[k] = closed_by_round.get(k, 0) + 1
        if state.gated_actions:
            n_gated += 1
        if state.unactionable_in_time:
            n_unactionable += 1

    payload = {
        "meta": {
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "schema_version": "1.0",
            "today": today.isoformat(),
            "pharmacy_source": source_name,
            "source_max_latency_days": max_latency,
            "forward_window_days": FORWARD_WINDOW_DAYS,
            "round_lead_days": {str(k): v for k, v in ROUND_LEAD_DAYS.items()},
            "min_round_dwell_days": MIN_ROUND_DWELL_DAYS,
            # Clearly-labeled demo affordance; None in a normal run.
            "demo_time_compression_days_per_tick": sim_days_per_tick,
            "n_worklist": len(cards),
            "n_by_status": by_status,
            "n_by_current_round": by_round,
            "n_closed": sum(closed_by_round.values()),
            "n_closed_by_round": closed_by_round,
            "n_gated_on_consent_or_fallback": n_gated,
            "n_unactionable_in_time": n_unactionable,
        },
        "patients": patients,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=None,
                    help="evaluate as of this ISO date (default: real today)")
    # DEMO TIME-COMPRESSION affordance (clearly labeled): so a multi-week ladder is
    # demonstrable live on stage in seconds. NOT production behavior.
    ap.add_argument("--simulate-days-per-tick", type=int, default=None,
                    help="[DEMO ONLY] advance simulated time by N days per tick")
    ap.add_argument("--ticks", type=int, default=1,
                    help="[DEMO ONLY] number of compressed ticks to run")
    args = ap.parse_args()

    base = _parse_date(args.as_of) if args.as_of else date.today()
    step = args.simulate_days_per_tick

    if step is None:
        payload = build_escalation_state(today=base)
        m = payload["meta"]
        print(f"escalation state @ {m['today']} (source={m['pharmacy_source']}, "
              f"max_latency={m['source_max_latency_days']}d): by round "
              f"{m['n_by_current_round']}; by status {m['n_by_status']}; "
              f"{m['n_closed']} closed; {m['n_gated_on_consent_or_fallback']} gated; "
              f"{m['n_unactionable_in_time']} unactionable-in-time -> {STATE_PATH}")
        return 0

    print(f"*** DEMO TIME-COMPRESSION: +{step} simulated day(s)/tick x {args.ticks} "
          f"ticks (NOT production cadence) ***")
    for t in range(args.ticks):
        sim_today = base + timedelta(days=step * t)
        # Freeze `now` at the simulated date too, so recorded dispatch timestamps
        # track the compressed clock rather than wall-clock.
        sim_now = datetime(sim_today.year, sim_today.month, sim_today.day, tzinfo=timezone.utc)
        payload = build_escalation_state(today=sim_today, now=sim_now, sim_days_per_tick=step)
        m = payload["meta"]
        print(f"  [sim {sim_today}] round {m['n_by_current_round']} | status {m['n_by_status']} "
              f"| closed {m['n_closed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
