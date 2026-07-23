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
``RefillOutcomeSource`` is the swap point. ``ConnectorOutcomeSource`` wires it to
the connector layer (src/sync/connectors/): the factory's fallback chain
(Surescripts -> AE claims -> local) picks a source, and its dispense events become
the outcomes here -- so replacing the demo's local synthetic source with a real AE
claims export or a certified Surescripts connection changes nothing in this file.
``CurrentCareOutcomeSource`` is a DEPRECATED stub (CurrentCare/RIQI was sunset;
CRISP Shared Services is now RI's RHIO) kept only to document that dead end.

CAVEAT to state to judges
-------------------------
Because the label is the ground-truth outcome, the closure signal reflects the
cohort's underlying adherence, NOT a causal effect of the CHW intervention --
synthetic data cannot demonstrate that lift.  The banner shows the *funnel and
mechanism* (routed -> reached -> objectively confirmed), not proof of impact.

Public API
----------
RefillOutcome            dataclass: per-patient objective outcome (+ refill_source/latency)
RefillOutcomeSource      ABC: .outcomes_for(patient_ids) -> dict[pid, RefillOutcome]
SyntheticOutcomeSource   reads labels.parquet (has_30_day_gap) directly
ConnectorOutcomeSource   dispense events from the connector layer (the real path)
CurrentCareOutcomeSource DEPRECATED stub (CurrentCare/RIQI sunset; CRISP is now the RHIO)
write_loop_outcomes()    CLI entry: routing_table.json -> loop_outcomes.json
"""
from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "labels.parquet"
ROUTING_TABLE_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
OUTCOMES_PATH = ROOT / "data" / "snapshots" / "loop_outcomes.json"

# Match pdc.py's forward outcome window: the lookback the connector-backed source
# scans for a qualifying dispense (the "before the predicted break" test itself is
# escalation.py's job; this module only reports whether an objective refill was seen).
FORWARD_WINDOW_DAYS = 180

sys.path.insert(0, str(ROOT))
from src.sync.connectors.base import PharmacyConnector  # noqa: E402
from src.sync.connectors.factory import get_connector  # noqa: E402


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
    # EXTENDED (2026-07-23, connector layer): which pharmacy source produced this
    # observation and that source's typical latency in days. Additive with
    # defaults so every existing consumer (retrain_labels.py, api/main.py) keeps
    # reading the original keys unchanged; the escalation timers use the latency to
    # avoid escalating a patient whose refill simply hasn't surfaced yet.
    refill_source: Optional[str] = None
    refill_latency_days: Optional[int] = None


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
                # zero latency is a synthetic-data property only (see LocalFileConnector).
                refill_source="synthetic_ri (labels-derived)",
                refill_latency_days=0,
            )
        return out


class ConnectorOutcomeSource(RefillOutcomeSource):
    """Objective outcomes from a :class:`PharmacyConnector` (the real integration).

    Turns the connector's dispense events into RefillOutcomes and records WHICH
    source produced them and HOW LATE that source runs (``refill_source`` /
    ``refill_latency_days``). Depends only on the connector INTERFACE -- the
    concrete adapter is chosen by ``get_connector()`` (factory fallback chain
    Surescripts -> AE claims -> local), so swapping the real feed in changes
    nothing here.

    Closure rule (unchanged in spirit): a patient the source *covers* who has at
    least one confirmed **dispense** in the scan window is an on-time refill;
    covered-but-no-dispense is a confirmed break; not covered at all is omitted
    (not yet observed). Whether the refill beat the patient's predicted break DATE
    is escalation.py's decision, not this module's -- here we only report the
    objective observation and its date.
    """

    def __init__(
        self,
        connector: Optional[PharmacyConnector] = None,
        *,
        lookback_days: int = FORWARD_WINDOW_DAYS,
        as_of: Optional[str] = None,
    ):
        if connector is None:
            # Factory already authenticated the selected source (and wrote the
            # sync-state trace); don't re-authenticate it.
            self._connector = get_connector()
        else:
            self._connector = connector
            self._connector.authenticate()
        self._profile = self._connector.source_profile
        self._lookback_days = lookback_days
        self._as_of = as_of

    def outcomes_for(self, patient_ids: Iterable[str]) -> dict[str, RefillOutcome]:
        ids = [str(p) for p in patient_ids]
        end = self._as_of or datetime.now(timezone.utc).date().isoformat()
        start = (datetime.fromisoformat(end).date() - timedelta(days=self._lookback_days)).isoformat()

        events = self._connector.fetch_dispense_events(ids, start, end)
        covered = self._connector.covered_patient_ids(ids)

        # Earliest confirmed dispense per patient (prescription-only events, if any
        # source ever emits them, never close the loop).
        first_dispense: dict[str, str] = {}
        for e in events:
            if not e.is_dispense:
                continue
            prior = first_dispense.get(e.patient_id)
            if prior is None or e.dispense_date < prior:
                first_dispense[e.patient_id] = e.dispense_date

        src_name = self._profile.source_name
        latency = self._profile.typical_latency_days
        out: dict[str, RefillOutcome] = {}
        for pid in ids:
            if pid not in covered:
                continue  # source has no record for this patient -> not yet observed
            on_time = pid in first_dispense
            out[pid] = RefillOutcome(
                patient_id=pid,
                observed=True,
                on_time_refill=on_time,
                # dispense date if refilled; else the detection ("as of") date.
                event_date=first_dispense[pid] if on_time else end,
                source=src_name,
                refill_source=src_name,
                refill_latency_days=latency,
            )
        return out


class CurrentCareOutcomeSource(RefillOutcomeSource):
    """DEPRECATED real-integration stub: objective refill outcomes from RI's CurrentCare HIE.

    DEPRECATED (2026-07-23): RIQI/CurrentCare was **sunset**; Rhode Island's RHIO
    is now **CRISP Shared Services**. Do not build on this class. Third-party HIE
    access requires state designation plus a multi-party DUA, and even then CRISP's
    surface (like CurrentCare's) is ADT/encounter and prescription-list data, not a
    clean antihypertensive *dispense* feed. The dispense/refill signal now comes
    from the connector layer (src/sync/connectors/, via ``ConnectorOutcomeSource``
    above): AE pharmacy claims for the automated backbone, Surescripts only
    opportunistically. A CRISP adapter is intentionally NOT built; its seam would
    be a new ``PharmacyConnector`` in the connectors package (see factory.py).

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

    Defaults to the connector-backed source (``ConnectorOutcomeSource`` → factory
    fallback chain, which selects the local synthetic source in the demo), so the
    real dispense-data path is exercised end to end. Pass an explicit ``source`` to
    override (e.g. the dependency-free ``SyntheticOutcomeSource``).

    Output shape (consumed read-only by src/api/main.py); EXTENDED 2026-07-23 with
    ``refill_source`` / ``refill_latency_days`` on each outcome and in meta, keeping
    every prior key:
        {
          "meta": {"generated_at", "source", "refill_source", "refill_latency_days",
                   "closure_rule", "n_routed", "n_observed", "n_on_time", "n_break"},
          "outcomes": { patient_id: {observed, on_time_refill, event_date, source,
                                     refill_source, refill_latency_days} }
        }
    """
    routed = _routed_ids(routing_table_path)
    src = source or ConnectorOutcomeSource()
    outcomes = detect_closures(routed, src)

    n_on_time = sum(1 for o in outcomes.values() if o.on_time_refill)
    # Surface the source + its latency at the top level so the API's sync badge can
    # show which feed produced these outcomes and how laggy it is. Read off a
    # representative outcome (all share one source per run); None-safe if empty.
    sample = next(iter(outcomes.values()), None)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": src.__class__.__name__,
            "refill_source": sample.refill_source if sample else None,
            "refill_latency_days": sample.refill_latency_days if sample else None,
            "closure_rule": "objective confirmed dispense; never self-report",
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
