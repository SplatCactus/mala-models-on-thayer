"""
src/sync/loop_closure.py

State-4 detector for the feedback loop: "was an OBJECTIVE on-time refill
observed for this routed patient?"  This is the loop-closure counterpart to
src/sync/pharmacy_source.py.  pharmacy_source answers "which patients have NEW
activity" (drives State-1 arrival / the live-sync demo); this module answers
"did a ROUTED patient actually refill within their predicted window" (drives
State-4 outcome and the dashboard's Loop-Closure-Rate banner).

OBJECTIVE-ONLY CLOSURE (design rule, per chris-workstream-plan.md Phase 2)
--------------------------------------------------------------------------
The loop may close ONLY on an objective refill signal, NEVER on self-reported
patient feedback.  On this synthetic cohort the objective signal is the PDC
label already computed in src/features/pdc.py and committed to labels.parquet:

    on_time_refill  ==  has_30_day_gap == 0   (they stayed covered -> they kept
                                                refilling -> the loop closed)
    confirmed_break ==  has_30_day_gap == 1   (a >=30d uncovered stretch -> no
                                                on-time refill -> loop did NOT close)

labels.parquet is a *committed* artifact, so this detector runs with no
gitignored raw data (see pharmacy_source.py's "IMPORTANT CONSTRAINT").  The
label is derived from the patient's real fill history over the forward window,
so it is a faithful stand-in for "the pharmacy feed later confirmed a refill."

SEAM FOR THE REAL FEED
----------------------
``CurrentCareOutcomeSource`` is a documented NotImplementedError stub, mirroring
``CurrentCareAdapter`` in pharmacy_source.py: once a CurrentCare DUA exists, the
real refill/dispense events replace the label-derived outcome here and NOTHING
else in the pipeline changes.  ``RefillOutcomeSource`` is the swap point.

CAVEAT to state to judges
-------------------------
Because the label is the ground-truth outcome, the closure signal reflects the
cohort's underlying adherence, NOT a causal effect of the CHW intervention --
synthetic data cannot demonstrate that lift.  The banner shows the *funnel and
mechanism* (routed -> reached -> objectively confirmed), not proof of impact.

Public API
----------
RefillOutcome           dataclass: per-patient objective outcome
RefillOutcomeSource     ABC: .outcomes_for(patient_ids) -> dict[pid, RefillOutcome]
SyntheticOutcomeSource  reads labels.parquet (has_30_day_gap)
CurrentCareOutcomeSource documented stub for the real RI CurrentCare HIE feed
write_loop_outcomes()   CLI entry: routing_table.json -> loop_outcomes.json
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "labels.parquet"
ROUTING_TABLE_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
OUTCOMES_PATH = ROOT / "data" / "snapshots" / "loop_outcomes.json"


@dataclass(frozen=True)
class RefillOutcome:
    """One objective loop-closure outcome for one routed patient.

    ``observed`` is whether the objective source has an outcome for this patient
    at all; ``on_time_refill`` is the closure signal (True = loop closed on an
    on-time refill, False = confirmed break). ``event_date`` is when the outcome
    was observed (detection time on synthetic data; the real dispense date once
    CurrentCare is wired). Field names mirror pharmacy_source.RefillRecord so the
    two sources stay shape-compatible.
    """
    patient_id: str
    observed: bool
    on_time_refill: bool
    event_date: Optional[str]
    source: str


class RefillOutcomeSource(ABC):
    """Something the closure detector can ask 'did these patients refill on time?'."""

    @abstractmethod
    def outcomes_for(self, patient_ids: Iterable[str]) -> dict[str, RefillOutcome]:
        """Return one RefillOutcome per patient id that has an objective outcome.

        Ids with no outcome are simply absent from the returned dict (the caller
        treats a missing id as 'not yet observed').
        """
        raise NotImplementedError


class SyntheticOutcomeSource(RefillOutcomeSource):
    """Objective outcome from the committed PDC labels (has_30_day_gap)."""

    def __init__(self, labels_path: Path = LABELS_PATH, *, patient_id_col: str = "patient_id"):
        df = pd.read_parquet(labels_path)
        df = df.dropna(subset=["has_30_day_gap"])
        # has_30_day_gap == 0  ->  no >=30d uncovered stretch  ->  on-time refill.
        self._on_time = {
            str(pid): (int(gap) == 0)
            for pid, gap in zip(df[patient_id_col], df["has_30_day_gap"])
        }

    def outcomes_for(self, patient_ids: Iterable[str]) -> dict[str, RefillOutcome]:
        now = datetime.now(timezone.utc).date().isoformat()
        out: dict[str, RefillOutcome] = {}
        for pid in patient_ids:
            pid = str(pid)
            if pid not in self._on_time:
                continue
            out[pid] = RefillOutcome(
                patient_id=pid,
                observed=True,
                on_time_refill=self._on_time[pid],
                event_date=now,
                source="synthetic_ri (labels-derived)",
            )
        return out


class CurrentCareOutcomeSource(RefillOutcomeSource):
    """Real-integration stub: objective refill outcomes from RI's CurrentCare HIE.

    NOT IMPLEMENTED -- requires a data-use agreement, same as
    pharmacy_source.CurrentCareAdapter. Intended behavior: a routed patient's
    loop closes when an antihypertensive dispense (NDC/RxNorm in
    ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL) is confirmed within their predicted
    break window; ``on_time_refill`` then comes from that real dispense timestamp
    vs. the window, not from the PDC label.

    REALITY CHECK (verified July 2026 -- do not assume this feed exists yet)
    -----------------------------------------------------------------------
    * CurrentCare's consent model flipped from opt-in to **opt-out in April
      2025** (broader denominator now), so the older "opt-in" framing is stale.
    * BUT CurrentCare does NOT expose a clean antihypertensive *dispense/refill*
      feed. Its PDMP data covers **controlled substances only** -- BP meds are
      not controlled, so PDMP will not see them -- and its broader medication
      content is largely **EHR prescription lists (what was ordered), not
      confirmed pharmacy fills.** So the objective "did they refill?" signal is
      the weakest link in the whole loop and is likely **claims/pharmacy-based,
      not CurrentCare**, in a real build. Confirm the available dispense feed
      with RIQI before relying on this class.
    * CurrentCare's most automatable surface is **CMAD** (real-time ADT / ED &
      hospital alerts, EHR-integrable) -- useful for a *hospitalization* signal,
      not for refill closure. There is no public self-serve API; access is
      arranged directly with RIQI.

    See pharmacy_source.CurrentCareAdapter for the full field/consent checklist.
    """

    def outcomes_for(self, patient_ids: Iterable[str]) -> dict[str, RefillOutcome]:
        raise NotImplementedError(
            "CurrentCareOutcomeSource is not wired: RI's CurrentCare HIE has no "
            "clean antihypertensive refill feed (PDMP = controlled substances "
            "only; the rest is prescription lists, not confirmed fills), and "
            "access needs a RIQI data-use agreement. A real build likely sources "
            "the refill signal from pharmacy claims. Use SyntheticOutcomeSource "
            "for the demo."
        )


def detect_closures(
    routed_patient_ids: Iterable[str],
    source: Optional[RefillOutcomeSource] = None,
) -> dict[str, RefillOutcome]:
    """Objective outcome for each routed patient (missing == not yet observed)."""
    source = source or SyntheticOutcomeSource()
    return source.outcomes_for(routed_patient_ids)


def _routed_ids(routing_table_path: Path) -> list[str]:
    with open(routing_table_path) as f:
        payload = json.load(f)
    return [c["patient_id"] for c in payload.get("capped_worklist", [])]


def write_loop_outcomes(
    routing_table_path: Path = ROUTING_TABLE_PATH,
    outcomes_path: Path = OUTCOMES_PATH,
    source: Optional[RefillOutcomeSource] = None,
) -> dict:
    """Compute objective outcomes for the current worklist and write a JSON map.

    Output shape (consumed read-only by src/api/main.py):
        {
          "meta": {"generated_at", "source", "n_routed", "n_observed",
                   "n_on_time", "n_break"},
          "outcomes": { patient_id: {observed, on_time_refill, event_date, source} }
        }
    """
    routed = _routed_ids(routing_table_path)
    src = source or SyntheticOutcomeSource()
    outcomes = detect_closures(routed, src)

    n_on_time = sum(1 for o in outcomes.values() if o.on_time_refill)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": src.__class__.__name__,
            "closure_rule": "objective on-time refill (has_30_day_gap == 0); never self-report",
            "n_routed": len(routed),
            "n_observed": len(outcomes),
            "n_on_time": n_on_time,
            "n_break": len(outcomes) - n_on_time,
        },
        "outcomes": {pid: {k: v for k, v in asdict(o).items() if k != "patient_id"}
                     for pid, o in outcomes.items()},
    }
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(outcomes_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> int:
    payload = write_loop_outcomes()
    m = payload["meta"]
    print(f"loop outcomes: {m['n_observed']}/{m['n_routed']} routed patients observed; "
          f"{m['n_on_time']} on-time refills, {m['n_break']} confirmed breaks "
          f"-> {OUTCOMES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
