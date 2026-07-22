"""retrain_labels.py — turn CLOSED feedback loops into retraining labels (State 4 -> training signal).

The feedback loop's payoff for the model: every objectively CLOSED loop is a
fresh, real-world-observed label we can fold into the next training run. This
module is the seam that harvests them. It is intentionally a documented STUB
with a working demo path -- the production path needs infrastructure that does
not exist yet (see below).

What a "retraining label" is here
---------------------------------
The loop closes on an objective on-time refill (src/sync/loop_closure.py,
``on_time_refill``), NEVER on self-report. So a closed loop yields:

    adherent (y=1)      loop closed on an on-time refill (has_30_day_gap == 0)
    non-adherent (y=0)  confirmed break (has_30_day_gap == 1)

matching classifier.py's target polarity (y=1 == adherent). Feeding these back
lets the model learn from patients it has now *actually observed* post-outreach.

Why this is a STUB, not wired to training yet
---------------------------------------------
1. ACTION GATING IS CLIENT-SIDE. Per the demo's no-backend choice, State 3
   (Actioned) lives in the dashboard's localStorage, not on the server. A
   faithful retraining signal is "loops that were *actioned by a CHW* and then
   closed" -- but the server can't see the localStorage action log. So the
   ``demo`` path below emits candidate labels from the OBJECTIVE outcomes alone
   (every observed refill), and flags that it is NOT action-gated. Production
   must persist State 3 server-side (a real datastore replacing localStorage)
   and gate on it.
2. CONFOUNDING. On synthetic data the label is the ground-truth outcome
   regardless of intervention, so retraining on it cannot teach the model the
   *effect* of outreach -- only re-affirm the existing signal. On real data the
   closed-loop labels would be a biased sample (only the routed/actioned
   patients), which a production pipeline must correct for (e.g. propensity
   weighting) before retraining. Do not skip this.
3. IDENTITY / TIMELINESS. Real closures arrive over time from CurrentCare (see
   src/sync/loop_closure.CurrentCareOutcomeSource) and need the CurrentCare->cohort
   id crosswalk and a proper event high-water-mark, neither of which exists yet.

``build_retraining_labels(..., mode="production")`` therefore raises
NotImplementedError; ``mode="demo"`` writes a candidate label file so the seam
is exercisable and visible.

Public API
----------
build_retraining_labels(mode="demo") -> path to written labels (demo) / raises (production)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOOP_OUTCOMES_PATH = ROOT / "data" / "snapshots" / "loop_outcomes.json"
RETRAIN_LABELS_PATH = ROOT / "data" / "snapshots" / "retrain_labels.parquet"


def _load_outcomes(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m src.sync.loop_closure` first to "
            "produce objective loop outcomes.")
    with open(path) as f:
        return json.load(f).get("outcomes", {})


def build_retraining_labels(
    *,
    mode: str = "demo",
    loop_outcomes_path: Path = LOOP_OUTCOMES_PATH,
    out_path: Path = RETRAIN_LABELS_PATH,
    actioned_ids: Optional[set[str]] = None,
) -> Path:
    """Harvest closed-loop labels for the next training run.

    Parameters
    ----------
    mode : "demo" | "production"
        ``demo`` writes candidate labels from the objective outcomes (NOT
        action-gated -- see module docstring) so the seam is runnable today.
        ``production`` raises NotImplementedError: it requires server-side State-3
        action state, a real CurrentCare closure feed, and confounding correction.
    actioned_ids : set[str], optional
        If a caller CAN supply the set of server-side actioned patient ids (once
        State 3 is persisted), pass it to gate the demo labels to genuinely
        actioned+closed loops -- the shape the production path will use.

    Returns
    -------
    Path to the written retrain_labels.parquet (columns: patient_id,
    label_adherent, source_event_date, loop_closed).
    """
    if mode == "production":
        raise NotImplementedError(
            "Production retraining-label harvest is not wired: it needs (1) "
            "server-side State-3 action state (localStorage can't be read here), "
            "(2) a real CurrentCare closure feed + id crosswalk, and (3) "
            "selection-bias correction before the labels re-enter training. See "
            "src/models/retrain_labels.py module docstring.")
    if mode != "demo":
        raise ValueError(f"unknown mode {mode!r} (expected 'demo' or 'production')")

    outcomes = _load_outcomes(loop_outcomes_path)
    rows = []
    for pid, o in outcomes.items():
        if not o.get("observed"):
            continue
        if actioned_ids is not None and pid not in actioned_ids:
            continue  # gate on server-side action state when available
        rows.append({
            "patient_id": pid,
            "label_adherent": int(bool(o.get("on_time_refill"))),
            "source_event_date": o.get("event_date"),
            "loop_closed": True,
        })
    df = pd.DataFrame(rows, columns=["patient_id", "label_adherent", "source_event_date", "loop_closed"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> int:
    path = build_retraining_labels(mode="demo")
    df = pd.read_parquet(path)
    n_pos = int(df["label_adherent"].sum())
    print(f"[demo, NOT action-gated] wrote {len(df)} closed-loop retraining labels "
          f"({n_pos} adherent / {len(df) - n_pos} break) -> {path}")
    print("Production harvest is stubbed (NotImplementedError) -- see module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
