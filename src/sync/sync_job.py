"""
src/sync/sync_job.py

Lightweight background job for the "automatic data sync" demo: polls a
PharmacyRefillSource (src/sync/pharmacy_source.py) on a timer, and for any
newly-revealed patients, re-scores + re-routes them and rewrites
data/snapshots/routing_table.json -- the same file src/api/main.py already
serves and re-reads fresh on every request, so no API changes are needed for
the dashboard to pick up an update; it already re-reads the file on each
`/worklist` call.

WHAT THIS DOES vs. WHAT IT DOESN'T
------------------------------------
Does: re-run SHAPRunner.explain_cohort -> RoutingRuleEngine.route_cohort ->
CapacityEngine.build -> build_worklist_payload for the CURRENTLY-REVEALED
patient subset, each tick, adding data_source/last_synced to the payload's
meta. This is the "data-loading layer + sync job" this feature is scoped to.

Does NOT: touch any feature-engineering or model-training code. The risk
model (SHAPRunner) is fit ONCE, at startup, on the full labeled cohort --
see "why fit once" below. Does NOT recompute fairness_report.json per tick
(see module docstring in src/run_routing_pipeline.py's sibling report --
that stays a full-batch, run-once artifact; recomputing disparity ratios on
a 25-patient subset every few seconds is noise, not signal, and wasn't part
of what this feature asked for).

WHY THE MODEL FITS ONCE, NOT PER TICK
----------------------------------------
A production system trains offline and scores online; refitting
SHAPRunner's LogisticRegression on a tiny, growing, arbitrarily-ordered
subset every few seconds would make risk scores swing around for reasons
that have nothing to do with the patient's actual data -- purely an
artifact of which random batch has been revealed so far. Fitting once on
the full 16,205-patient labeled cohort (same as src/run_routing_pipeline.py
does for the real batch run) and then only re-scoring/re-routing the
revealed subset each tick avoids that, and is what "loading a deployed
model" means here -- not model logic living in this file.

Run
---
    ./venv/bin/python -m src.sync.sync_job                     # demo defaults
    ./venv/bin/python -m src.sync.sync_job --interval 3 --batch-size 50
    ./venv/bin/python -m src.sync.sync_job --max-ticks 5        # bounded, for testing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.common import PANEL_PATH, LABELS_PATH  # noqa: E402
from src.explain.shap_runner import SHAPRunner, ALL_MODEL_FEATURES  # noqa: E402
from src.routing.rules import RoutingRuleEngine  # noqa: E402
from src.routing.capacity import CapacityEngine  # noqa: E402
from src.worklist_builder import build_worklist_payload  # noqa: E402
from src.sync.pharmacy_source import PharmacyRefillSource, SyntheticRIAdapter  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "snapshots" / "routing_table.json"

# Matches run_routing_pipeline.py's confirmed polarity/window constants --
# duplicated (not imported) because this is a separate, sync-specific entry
# point and importing run_routing_pipeline.py's module-level code would
# re-run its own script-style setup. Keep these in sync if either changes.
#
# UPDATED 2026-07-22 to match run_routing_pipeline.py's target correction:
# the at-risk positive class is has_30_day_gap == 1 directly (the event we
# route on), not the older pdc_180d < 0.80 proxy -- the two coincide for all
# but ~6/16205 patients, but the gap event is congruent with classifier.py's
# now-default polarity too. See that module's "RISK-LABEL POLARITY" note.
FORWARD_WINDOW_DAYS = 180
DATA_SOURCE_LABEL = "synthetic (demo)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("sync_job")


def load_full_frame(panel_path: Path = PANEL_PATH, labels_path: Path = LABELS_PATH) -> pd.DataFrame:
    """Same join/dropna contract as run_routing_pipeline.py's load_panel_and_labels."""
    panel = pd.read_parquet(panel_path)
    labels = pd.read_parquet(labels_path)
    frame = panel.merge(labels, on="patient_id", how="inner")
    frame = frame.dropna(subset=["has_30_day_gap"]).reset_index(drop=True)
    missing = [c for c in ALL_MODEL_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"feature panel is missing expected model columns: {missing}")
    return frame


def build_risk_label(frame: pd.DataFrame):
    """At-risk-positive label: y=1 iff has_30_day_gap == 1 (see
    run_routing_pipeline.py's "RISK-LABEL POLARITY" note -- same polarity)."""
    return (frame["has_30_day_gap"].to_numpy() == 1).astype(int)


def score_and_route(frame_subset: pd.DataFrame, runner: SHAPRunner, engine: RoutingRuleEngine):
    profiles = runner.explain_cohort(frame_subset)
    decisions = engine.route_cohort(profiles)
    days_to_break = {p.patient_id: (1.0 - p.predicted_risk) * FORWARD_WINDOW_DAYS for p in profiles}
    predicted_risk = {p.patient_id: p.predicted_risk for p in profiles}
    return decisions, days_to_break, predicted_risk


def run_sync_loop(
    source: PharmacyRefillSource,
    *,
    interval_seconds: float = 5.0,
    role_caps: Optional[Dict[str, int]] = None,
    max_ticks: Optional[int] = None,
    output_path: Path = OUTPUT_PATH,
) -> int:
    """Poll `source` on a timer, re-scoring/re-routing the revealed subset each tick.

    Returns the number of patients revealed by the time the loop exits
    (only exits at all if `max_ticks` is set -- otherwise runs forever,
    as a background job should).
    """
    log.info("Loading full cohort + fitting risk model once (features/model unchanged)...")
    frame = load_full_frame()
    y_risk = build_risk_label(frame)
    runner = SHAPRunner()
    runner.fit(frame, y_risk)
    log.info("  fitted on %d patients, %d at risk (has_30_day_gap == 1)",
              len(frame), int(y_risk.sum()))

    engine = RoutingRuleEngine()
    cap_engine = CapacityEngine(role_caps=role_caps)

    revealed_ids: list[str] = []
    tick = 0
    while max_ticks is None or tick < max_ticks:
        records = source.load_refills()
        if records:
            new_ids = [r.patient_id for r in records]
            revealed_ids.extend(new_ids)
            log.info("  tick %d: +%d new patient(s) (%s), total revealed: %d",
                     tick, len(new_ids), records[0].source, len(revealed_ids))
        else:
            log.info("  tick %d: no new activity", tick)

        if revealed_ids:
            subset = frame[frame["patient_id"].isin(revealed_ids)].reset_index(drop=True)
            decisions, days_to_break, predicted_risk = score_and_route(subset, runner, engine)
            capacity_result = cap_engine.build(decisions, days_to_break, predicted_risk)
            payload = build_worklist_payload(
                capacity_result,
                routing_table_version=engine.version,
                data_source=DATA_SOURCE_LABEL,
                last_synced=datetime.now(timezone.utc).isoformat(),
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(payload, f, indent=2)
            log.info("  wrote %s (%d revealed, %d in capped worklist)",
                     output_path, len(revealed_ids), len(capacity_result.capped_worklist))

        tick += 1
        if max_ticks is None or tick < max_ticks:
            time.sleep(interval_seconds)

    return len(revealed_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    ap.add_argument("--batch-size", type=int, default=25, help="patients revealed per tick")
    ap.add_argument("--seed", type=int, default=0, help="reveal-order seed (reproducible demo)")
    ap.add_argument("--max-ticks", type=int, default=None, help="stop after N ticks (omit to run forever)")
    args = ap.parse_args()

    source = SyntheticRIAdapter(PANEL_PATH, LABELS_PATH, batch_size=args.batch_size, seed=args.seed)
    run_sync_loop(source, interval_seconds=args.interval, max_ticks=args.max_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
