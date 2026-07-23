"""
src/sync/escalation_job.py

Background job that TICKS the escalation state machine over (optionally compressed)
time. It is deliberately separate from src/sync/sync_job.py -- whose docstring is
emphatic that its only scope is the live worklist re-score demo -- but it REUSES
that module's fit-once/score-subset machinery rather than duplicating it.

WHAT EACH TICK DOES
-------------------
1. Reveal the next batch of patients from a PharmacyRefillSource (incremental
   cohort reveal, exactly as sync_job does) and accumulate the revealed set.
2. Re-score + re-route the revealed subset with the once-fitted model and rewrite
   data/snapshots/routing_table.json (via sync_job.score_and_route +
   build_worklist_payload -- the same functions sync_job uses).
3. Refresh objective outcomes for the current worklist through the connector layer
   (loop_closure.write_loop_outcomes -> loop_outcomes.json).
4. Advance the escalation state (src/routing/escalation.build_escalation_state),
   which MERGES each patient's prior frozen entry/break dates + dispatch timestamps
   forward rather than overwriting -- so incremental reveal is additive and a
   restart re-derives the same state.

WHY THE MODEL IS FIT ONCE (inherited from sync_job's reasoning)
---------------------------------------------------------------
A production system trains offline and scores online; refitting on a growing random
subset every tick would make risk scores swing for reasons unrelated to the patient.
We fit once on the full labeled cohort and only re-score/re-route the revealed subset.

DEMO TIME-COMPRESSION (clearly labeled; NOT production cadence)
---------------------------------------------------------------
Real escalation is paced by a 30-90 day claims feed, so a live audience can't watch
weeks pass. ``--simulate-days-per-tick`` advances a SIMULATED clock by N days each
tick so the ladder is demonstrable in seconds. Every tick logs the simulated date
and the written state records ``demo_time_compression_days_per_tick`` in its meta,
so nobody mistakes the compressed run for real-time behavior.

Run
---
    ./venv/bin/python -m src.sync.escalation_job --simulate-days-per-tick 30 --max-ticks 6
    ./venv/bin/python -m src.sync.escalation_job --interval 3 --batch-size 100 --max-ticks 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse sync_job's building blocks rather than re-implementing them.
from src.sync.sync_job import (  # noqa: E402
    load_full_frame, build_risk_label, score_and_route, DATA_SOURCE_LABEL,
)
from src.models.common import PANEL_PATH, LABELS_PATH  # noqa: E402
from src.explain.shap_runner import SHAPRunner  # noqa: E402
from src.routing.rules import RoutingRuleEngine  # noqa: E402
from src.routing.capacity import CapacityEngine  # noqa: E402
from src.worklist_builder import build_worklist_payload  # noqa: E402
from src.sync.pharmacy_source import PharmacyRefillSource, SyntheticRIAdapter  # noqa: E402
from src.sync.loop_closure import write_loop_outcomes  # noqa: E402
from src.routing.escalation import build_escalation_state  # noqa: E402
from src.routing.consent import CONSENT_PATH, CONSENT_VALIDITY_DAYS, load_consent  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
ESC_DATA_SOURCE_LABEL = "synthetic (escalation demo)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("escalation_job")


def run_escalation_loop(
    source: PharmacyRefillSource,
    *,
    interval_seconds: float = 5.0,
    sim_days_per_tick: Optional[int] = None,
    base_date: Optional[date] = None,
    role_caps: Optional[Dict[str, int]] = None,
    max_ticks: Optional[int] = None,
    output_path: Path = OUTPUT_PATH,
) -> int:
    """Poll ``source`` on a timer; each tick re-routes the revealed subset and advances
    the escalation state. Returns the number of patients revealed when the loop exits."""
    log.info("Loading full cohort + fitting risk model once (reused from sync_job)...")
    frame = load_full_frame()
    runner = SHAPRunner()  # defaults to classifier.py's deployed model (model-agnostic downstream)
    runner.fit(frame, build_risk_label(frame))
    engine = RoutingRuleEngine()
    cap_engine = CapacityEngine(role_caps=role_caps)

    base_date = base_date or date.today()
    if sim_days_per_tick is not None:
        log.info("*** DEMO TIME-COMPRESSION: +%d simulated day(s)/tick (NOT production cadence) ***",
                 sim_days_per_tick)

    revealed: list[str] = []
    tick = 0
    while max_ticks is None or tick < max_ticks:
        records = source.load_refills()
        if records:
            revealed.extend(r.patient_id for r in records)
            log.info("  tick %d: +%d patient(s), total revealed %d", tick, len(records), len(revealed))
        else:
            log.info("  tick %d: no new activity", tick)

        if revealed:
            # Simulated clock: freeze `now` at the simulated date so recorded dispatch
            # timestamps track the compressed clock, not wall-clock.
            sim_today = base_date + timedelta(days=(sim_days_per_tick or 0) * tick)
            sim_now = datetime(sim_today.year, sim_today.month, sim_today.day, tzinfo=timezone.utc)

            subset = frame[frame["patient_id"].isin(revealed)].reset_index(drop=True)
            decisions, days_to_break, predicted_risk = score_and_route(subset, runner, engine)
            cap_result = cap_engine.build(decisions, days_to_break, predicted_risk)
            payload = build_worklist_payload(
                cap_result, routing_table_version=engine.version,
                data_source=ESC_DATA_SOURCE_LABEL, last_synced=sim_now.isoformat(),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(payload, f, indent=2)

            # Objective outcomes via the connector layer, then advance escalation state
            # (merges prior frozen dates/timestamps -> additive across ticks).
            write_loop_outcomes()
            esc = build_escalation_state(today=sim_today, now=sim_now, sim_days_per_tick=sim_days_per_tick)
            m = esc["meta"]
            log.info("  [sim %s] worklist %d | round %s | status %s | closed %d | gated %d",
                     sim_today, m["n_worklist"], m["n_by_current_round"], m["n_by_status"],
                     m["n_closed"], m["n_gated_on_consent_or_fallback"])

        tick += 1
        if max_ticks is None or tick < max_ticks:
            time.sleep(interval_seconds)
    return len(revealed)


def _warn_if_consent_will_read_stale(
    base: date, sim_days_per_tick, max_ticks, *, consent_path: Path = CONSENT_PATH
) -> None:
    """Loudly warn if the simulated timeline will run past the consent validity window.

    Consent staleness is measured against the SIMULATED clock (src/routing/consent.py's
    CONSENT_VALIDITY_DAYS, {days}). If the last simulated tick lands more than that many
    days after the freshest consent as_of date, every record reads stale, fails closed,
    and gating over-triggers on the whole cohort -- a silent, misleading result. This is
    the sharp edge the near-today anchor avoids; we refuse to proceed quietly past it.
    """.format(days=CONSENT_VALIDITY_DAYS)
    if not sim_days_per_tick:
        return  # no compression -> runs at ~today, consent stays fresh
    records = load_consent(consent_path)
    as_ofs = [datetime.strptime(r.as_of[:10], "%Y-%m-%d").date()
              for r in records.values() if r.as_of]
    if not as_ofs:
        return  # no dated consent to go stale against
    freshest = max(as_ofs)
    unbounded = max_ticks is None
    end = None if unbounded else base + timedelta(days=sim_days_per_tick * max(max_ticks - 1, 0))
    over = unbounded or (end - freshest).days > CONSENT_VALIDITY_DAYS
    if over:
        end_desc = "unbounded (runs forever)" if unbounded else end.isoformat()
        log.warning(
            "*** CONSENT WILL READ STALE ***  simulated end date %s is more than "
            "%d days (CONSENT_VALIDITY_DAYS) after the freshest consent as_of %s. "
            "Every consent record will be treated as absent -> gating will OVER-TRIGGER "
            "and mark the whole cohort gated. Fix by lowering --simulate-days-per-tick "
            "/ --max-ticks, anchoring --as-of near the consent dates, or re-running "
            "`python -m src.routing.consent` to refresh the as_of dates first.",
            end_desc, CONSENT_VALIDITY_DAYS, freshest.isoformat())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between ticks")
    ap.add_argument("--batch-size", type=int, default=150, help="patients revealed per tick")
    ap.add_argument("--seed", type=int, default=0, help="reveal-order seed (reproducible demo)")
    ap.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks")
    ap.add_argument("--simulate-days-per-tick", type=int, default=None,
                    help="[DEMO ONLY] advance simulated time by N days per tick")
    ap.add_argument("--as-of", default=None, help="simulated base date (ISO); default real today")
    args = ap.parse_args()

    source = SyntheticRIAdapter(PANEL_PATH, LABELS_PATH, batch_size=args.batch_size, seed=args.seed)
    base = date.fromisoformat(args.as_of) if args.as_of else date.today()
    # Guard the sharp edge: a far-future simulated timeline reads all consent as stale.
    _warn_if_consent_will_read_stale(base, args.simulate_days_per_tick, args.max_ticks)
    run_escalation_loop(
        source, interval_seconds=args.interval, sim_days_per_tick=args.simulate_days_per_tick,
        base_date=base, max_ticks=args.max_ticks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
