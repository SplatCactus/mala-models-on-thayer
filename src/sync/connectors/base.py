"""
src/sync/connectors/base.py

The pharmacy-data connector abstraction: one interface every dispense-data source
implements, plus the metadata that makes the escalation timers honest.

WHY THIS EXISTS (see ESCALATION_PLAN.md Part 5 / constraint C2)
--------------------------------------------------------------
Escalation timers must account for how late a refill signal arrives, so we never
escalate a patient whose refill probably already happened but has not surfaced in
the data yet. That means every source has to *declare* its latency and how it may
legally be queried -- not just return events. So a connector is two things:

  1. a ``SourceProfile`` -- name, access mode (prescriber-initiated vs.
     batch-permitted), min/typical/max latency in days, whether a clinical
     encounter is required to query it, and whether it confirms a TRUE dispense
     (the drug left the pharmacy) or only a PRESCRIPTION (it was ordered); and
  2. ``fetch_dispense_events`` over a set of patient ids and a date range.

DEPENDENCY DIRECTION (enforced by construction)
-----------------------------------------------
The escalation logic depends ONLY on this module's interface, never on a concrete
adapter. No escalation code imports ``surescripts`` / ``ae_claims`` / ``local_file``;
it takes a ``PharmacyConnector`` (or reads the already-materialized outcomes) and
asks it for its ``source_profile``. Swapping adapters therefore changes nothing
downstream. The concrete-adapter choice lives only in ``factory.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

# Access modes. A source is either queryable in an automated batch (a flat feed we
# may poll) or only by a licensed prescriber in the context of treating a specific
# patient (Surescripts) -- the escalation backbone can only be paced by the former.
ACCESS_PRESCRIBER_INITIATED = "prescriber_initiated"
ACCESS_BATCH_PERMITTED = "batch_permitted"
_VALID_ACCESS_MODES = frozenset({ACCESS_PRESCRIBER_INITIATED, ACCESS_BATCH_PERMITTED})


class ConnectorError(Exception):
    """Base class for connector failures."""


class ConnectorAuthError(ConnectorError):
    """Raised by ``authenticate()`` when a source can't be used (missing creds/DUA/file).

    This is the signal the factory catches to fall back to the next source in the
    chain -- it means "this source is unavailable here," not "the query was illegal."
    """


class ConnectorAccessError(ConnectorError):
    """Raised when a query violates the source's access mode.

    Distinct from ``ConnectorAuthError``: the source exists and we may be
    authenticated, but the *kind* of query is not permitted (e.g. an automated
    batch pull against a prescriber-initiated, encounter-tied feed). This is a
    contractual/legal boundary, not a fallback trigger -- it should surface, not
    be silently swallowed.
    """


@dataclass(frozen=True)
class SourceProfile:
    """Declared metadata for one dispense-data source.

    ``confirms_dispense`` is the honesty flag the loop-closure logic needs: a true
    fill/dispense event means the drug was actually picked up (a valid closure
    signal), whereas a prescription event means it was only ordered (NOT a
    closure). ``*_latency_days`` is how long after the real-world dispense the
    event becomes visible in this source -- the number the escalation latency
    guard adds before it is willing to escalate on "no refill seen yet."
    """
    source_name: str
    access_mode: str
    min_latency_days: int
    typical_latency_days: int
    max_latency_days: int
    requires_encounter: bool
    confirms_dispense: bool

    def __post_init__(self) -> None:
        if self.access_mode not in _VALID_ACCESS_MODES:
            raise ValueError(
                f"access_mode must be one of {sorted(_VALID_ACCESS_MODES)}, "
                f"got {self.access_mode!r}"
            )
        if not (self.min_latency_days <= self.typical_latency_days <= self.max_latency_days):
            raise ValueError(
                f"latency profile must satisfy min<=typical<=max, got "
                f"{self.min_latency_days}/{self.typical_latency_days}/{self.max_latency_days}"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Nest the latency figures the way the API/plan expose them (a "latency
        # profile"), while keeping the flat fields for any direct reader.
        d["latency_days"] = {
            "min": self.min_latency_days,
            "typical": self.typical_latency_days,
            "max": self.max_latency_days,
        }
        return d


@dataclass(frozen=True)
class EncounterContext:
    """The clinical-encounter identity a prescriber-initiated source requires.

    Surescripts Medication History may only be queried by a licensed prescriber in
    connection with treating a specific patient. This carries the prescriber's
    Surescripts Prescriber ID (SPI) and the encounter the query is attached to, so
    a source can enforce that a bare batch query (no prescriber, no encounter) is
    refused. Batch sources ignore it.
    """
    prescriber_spi: str
    encounter_id: str
    patient_id: str


@dataclass(frozen=True)
class DispenseEvent:
    """One dispense (or prescription) event for one patient.

    Field names track a real fill/dispense record (NDC + RxNorm product code, fill
    date, days supply, dispensing pharmacy NCPDP id). ``is_dispense`` mirrors the
    source's ``confirms_dispense``: True == a confirmed fill (a closure signal),
    False == a prescription/order only (not a closure). ``source`` is the emitting
    source's ``source_name`` and ``latency_days`` is that source's typical latency,
    carried on the event so a downstream consumer records *how late* this
    observation is without re-deriving it.
    """
    patient_id: str
    dispense_date: str            # ISO date (YYYY-MM-DD)
    product_description: str
    ndc: Optional[str]
    rxnorm: Optional[str]
    days_supply: Optional[int]
    pharmacy_ncpdp: Optional[str]
    source: str
    is_dispense: bool
    latency_days: int

    def to_dict(self) -> dict:
        return asdict(self)


class PharmacyConnector(ABC):
    """A source the pipeline can ask for dispense events + its own latency profile.

    Contract:
      * ``source_profile`` is available WITHOUT authenticating (the factory needs it
        to record an attempt even when a source is unavailable).
      * ``authenticate()`` raises ``ConnectorAuthError`` if the source can't be used
        here (missing credentials / DUA / export file). Safe to call repeatedly.
      * ``fetch_dispense_events`` returns events for the requested patients whose
        ``dispense_date`` falls in ``[start_date, end_date]``. A prescriber-initiated
        source raises ``ConnectorAccessError`` unless a valid ``encounter`` is given.
      * ``covered_patient_ids`` reports which of the requested patients this source
        has *any* record for -- so a caller can tell "found, no dispense" (a
        confirmed break) apart from "not in this feed at all" (not yet observed).
    """

    @property
    @abstractmethod
    def source_profile(self) -> SourceProfile:
        raise NotImplementedError

    @abstractmethod
    def authenticate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch_dispense_events(
        self,
        patient_ids: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        encounter: Optional[EncounterContext] = None,
    ) -> List[DispenseEvent]:
        raise NotImplementedError

    @abstractmethod
    def covered_patient_ids(self, patient_ids: Iterable[str]) -> set:
        raise NotImplementedError

    @property
    def last_synced(self) -> Optional[str]:
        """ISO timestamp of the last successful fetch, or None. Set by adapters."""
        return getattr(self, "_last_synced", None)
