"""
src/sync/connectors/local_file.py

The synthetic-demo source: the connector the pipeline actually runs on today.
Synthesizes dispense events from the committed ``labels.parquet`` so the whole
connector -> loop-closure path is exercisable with no gitignored data and no
network. It reproduces exactly the closure decision the previous
``SyntheticOutcomeSource`` made:

    has_30_day_gap == 0  ->  patient stayed covered -> emit ONE dispense event
    has_30_day_gap == 1  ->  a >=30d uncovered stretch -> emit NO dispense event

so switching loop_closure onto this connector does not change which patients are
counted as on-time refills.

``latency = 0`` with a deliberate caveat: **zero latency is a property of synthetic
data only.** No real dispense source is instantaneous (Surescripts 1-14d, AE claims
30-90d); the local adapter reports 0 solely because the label is already known. The
API surfaces this so a viewer sees the demo is running on the zero-latency synthetic
source, not a real feed. ``batch_permitted`` (a local file is trivially batchable),
``confirms_dispense = True`` (the label is derived from real fill history, a
faithful dispense stand-in).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .base import (
    ACCESS_BATCH_PERMITTED,
    ConnectorAuthError,
    DispenseEvent,
    EncounterContext,
    PharmacyConnector,
    SourceProfile,
)

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS = _ROOT / "labels.parquet"


class LocalFileConnector(PharmacyConnector):
    """Synthetic dispense feed derived from labels.parquet -- the demo source."""

    SOURCE_NAME = "local_file_synthetic"

    def __init__(self, labels_path: Optional[Path] = None, *, patient_id_col: str = "patient_id"):
        self._labels_path = Path(labels_path or _DEFAULT_LABELS)
        self._patient_id_col = patient_id_col
        self._on_time: Optional[dict] = None   # pid -> bool (has a synthetic on-time fill)
        self._last_synced: Optional[str] = None

    @property
    def source_profile(self) -> SourceProfile:
        return SourceProfile(
            source_name=self.SOURCE_NAME,
            access_mode=ACCESS_BATCH_PERMITTED,
            min_latency_days=0,
            typical_latency_days=0,   # synthetic-only; see module docstring
            max_latency_days=0,
            requires_encounter=False,
            confirms_dispense=True,
        )

    def authenticate(self) -> None:
        if not self._labels_path.exists():
            raise ConnectorAuthError(
                f"local synthetic source needs {self._labels_path} (run "
                "src/features/build_features.py). It is committed in this repo, so "
                "this should only fail in a stripped checkout."
            )
        df = pd.read_parquet(self._labels_path).dropna(subset=["has_30_day_gap"])
        self._on_time = {
            str(pid): (int(gap) == 0)
            for pid, gap in zip(df[self._patient_id_col], df["has_30_day_gap"])
        }

    def _ensure_loaded(self) -> None:
        if self._on_time is None:
            self.authenticate()

    def fetch_dispense_events(
        self,
        patient_ids: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        encounter: Optional[EncounterContext] = None,  # ignored: batch source
    ) -> List[DispenseEvent]:
        """One synthetic dispense per on-time patient, dated at the query's end_date.

        On-time patients (has_30_day_gap == 0) get a single antihypertensive fill;
        confirmed-break patients get none. The event is dated ``end_date`` -- the
        query's as-of moment -- because the synthetic label carries no real fill
        date; escalation applies the "before the predicted break date" test.
        """
        self._ensure_loaded()
        events: List[DispenseEvent] = []
        for pid in patient_ids:
            pid = str(pid)
            if self._on_time.get(pid):
                events.append(DispenseEvent(
                    patient_id=pid,
                    dispense_date=end_date,
                    product_description="antihypertensive (synthetic dispense)",
                    ndc=None,
                    rxnorm=None,
                    days_supply=30,
                    pharmacy_ncpdp=None,
                    source=self.SOURCE_NAME,
                    is_dispense=True,
                    latency_days=0,
                ))
        self._last_synced = pd.Timestamp.utcnow().isoformat()
        return events

    def covered_patient_ids(self, patient_ids: Iterable[str]) -> set:
        """Every requested patient present in labels.parquet -- the synthetic feed
        'covers' the whole labeled cohort, so a break (found, no fill) is
        distinguishable from an unknown patient."""
        self._ensure_loaded()
        ids = {str(p) for p in patient_ids}
        return ids & set(self._on_time.keys())
