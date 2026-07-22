"""
src/sync/pharmacy_source.py

Data-loading abstraction for "new refill data arrived" -- the interface the
sync job (src/sync/sync_job.py) polls to find out which patients have new
pharmacy activity since the last check.

WHY THIS EXISTS
---------------
For the judge demo we want to show the worklist growing/updating live, as if
a real pharmacy feed were streaming in, WITHOUT pretending our synthetic
cohort is a live feed. So: one real interface, two implementations -- a
synthetic one that actually runs today, and a documented stub for the real
integration this project would need next.

IMPORTANT CONSTRAINT (read before extending this file)
--------------------------------------------------------
SyntheticRIAdapter does NOT run src/features/build_features.py's feature
engineering on a live claims stream. It can't: build_features.py's raw
inputs (data/parquet_300k/*.parquet -- patients/conditions/medications/
encounters/observations/payer_transitions) are gitignored source data, not
part of this repo's committed artifacts. Only the already-computed
feature_panel.parquet/labels.parquet (the output of a build_features.py run)
are checked in. So "new data arriving" here means revealing already-built
patient rows from that static panel in batches -- simulating feed *cadence*
and the resulting incremental *scoring*, not incremental *feature
computation*. Do not present this as "recomputing features live" to judges;
say "simulating new pharmacy activity arriving" instead. See sync_job.py's
module docstring for how the revealed subset is re-scored.

Public API
----------
RefillRecord            dataclass: one new-activity record for one patient
PharmacyRefillSource    ABC: .load_refills() -> list[RefillRecord]
SyntheticRIAdapter      wraps feature_panel.parquet/labels.parquet, reveals
                        patient batches in a fixed, seeded, non-repeating order
CurrentCareAdapter      documented NotImplementedError stub for the real
                        RI CurrentCare HIE integration
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RefillRecord:
    """One "new pharmacy activity" record for one patient.

    Field names are deliberately close to what a real fill/dispense event
    looks like (see CurrentCareAdapter's docstring) even though
    SyntheticRIAdapter can only populate a subset of them from data we
    already have -- the interface is shaped for the real integration, not
    shrunk to fit today's simulation.
    """
    patient_id: str
    event_date: str          # ISO date (YYYY-MM-DD) this activity was observed
    source: str               # adapter name, e.g. "synthetic_ri" / "currentcare_hie"


class PharmacyRefillSource(ABC):
    """Something the sync job can poll for newly-available patient activity."""

    @abstractmethod
    def load_refills(self) -> list[RefillRecord]:
        """Return records for patients with new activity since the last call.

        Must be safe to call repeatedly on a timer; an empty list means
        "nothing new this tick," not an error.
        """
        raise NotImplementedError


class SyntheticRIAdapter(PharmacyRefillSource):
    """Simulates a pharmacy feed by revealing feature_panel.parquet rows in batches.

    See this module's docstring "IMPORTANT CONSTRAINT" section -- this does
    NOT compute features live; it reveals already-computed patient rows from
    the static panel this repo already has, in a fixed pseudo-random order
    (seeded, so a demo re-run is reproducible) and a configurable batch size.
    Once every patient has been revealed, subsequent calls return `[]`
    forever (the synthetic cohort is finite; a real feed wouldn't be).
    """

    def __init__(
        self,
        panel_path: Path,
        labels_path: Path,
        *,
        batch_size: int = 25,
        seed: int = 0,
        patient_id_col: str = "patient_id",
    ):
        panel = pd.read_parquet(panel_path, columns=[patient_id_col])
        labels = pd.read_parquet(labels_path, columns=[patient_id_col])
        # Only patients present in both -- same inner-join contract every
        # other consumer of these two files uses (see common.py/classifier.py).
        ids = sorted(set(panel[patient_id_col]) & set(labels[patient_id_col]))
        rng = random.Random(seed)
        rng.shuffle(ids)
        self._ids = ids
        self.batch_size = batch_size
        self._cursor = 0

    def load_refills(self) -> list[RefillRecord]:
        batch = self._ids[self._cursor: self._cursor + self.batch_size]
        self._cursor += len(batch)
        today = datetime.now(timezone.utc).date().isoformat()
        return [
            RefillRecord(patient_id=pid, event_date=today, source="synthetic_ri")
            for pid in batch
        ]

    @property
    def remaining(self) -> int:
        """Patients not yet revealed -- lets the sync job log/stop sensibly."""
        return max(len(self._ids) - self._cursor, 0)


class CurrentCareAdapter(PharmacyRefillSource):
    """Real-integration stub: Rhode Island's CurrentCare HIE.

    NOT IMPLEMENTED -- requires a signed data-use agreement (DUA) with
    CurrentCare before any of this can be built. This class exists so the
    integration point is visible in the codebase and so the fields we'd
    need are written down now, while the synthetic-vs-real boundary is
    fresh, rather than re-derived later from scratch.

    Fields we would request once a DUA exists
    ------------------------------------------
      - A CurrentCare-issued patient identifier, plus whatever crosswalk
        CurrentCare provides (or we build) to our cohort's `patient_id`
        (from src/etl/cohort.py) -- these are not the same identifier space.
      - NDC or RxNorm product code for the dispensed medication (to
        classify antihypertensive vs. other, same as
        ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL in src/features/pdc.py).
      - Fill/dispense date and days-supply (to build coverage intervals the
        same way src/features/pdc.py and src/features/pre_index.py do).
      - Dispensing pharmacy identifier (NPI or CurrentCare's own pharmacy
        ID) -- not currently a feature, but useful context.
      - Payer/plan identifier at time of fill (feeds
        src/features/pre_index.py's `payer_n_switches`).
      - A dispense-event timestamp suitable for a real "since last sync"
        cursor -- unlike SyntheticRIAdapter's simple batch-index cursor,
        a real feed needs a proper high-water-mark (e.g. "give me
        everything after event_id/timestamp X") to avoid re-fetching or
        missing records across restarts.
      - Whatever CurrentCare's own rate limits / pagination contract
        requires (batch size, polling interval, backoff behavior) -- this
        adapter's `load_refills()` would need to respect those, not just
        the demo's fixed polling interval.

    Everything else in this pipeline (feature computation, the risk model,
    routing rules) is unaffected by which adapter is in use -- only this
    class and the sync job's polling loop would change.
    """

    def load_refills(self) -> list[RefillRecord]:
        raise NotImplementedError(
            "CurrentCareAdapter requires a signed data-use agreement with "
            "Rhode Island's CurrentCare HIE before a real feed can be "
            "requested. See this class's docstring for the fields to "
            "request once one exists."
        )
