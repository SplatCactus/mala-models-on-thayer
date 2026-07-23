"""
src/sync/connectors/rxfill.py

RXFILL (NCPDP SCRIPT) push -- documented, LOW-COVERAGE source.

RXFILL is a real-time notification a pharmacy can send back to a prescriber's
system when a prescription is dispensed (or not picked up). When present it is the
best possible refill signal: near-zero latency and a true dispense confirmation,
with no prescriber pull required (the pharmacy pushes it), so it is
``batch_permitted`` from our polling perspective, latency 0-1 days,
``confirms_dispense = True``.

THE CATCH (stated plainly): RXFILL enrollment for **chronic generic** medications
like antihypertensives is **rare in practice** -- pharmacies most reliably send
RXFILL for controlled substances and specialty drugs, not $4 generic lisinopril.
So even where a connection exists, coverage of our cohort would be sparse, which is
why this is NOT the demo's timer backbone (that is ae_claims.py) and NOT in the
default fallback chain. It is implemented as a documented seam: if a partner ever
turns on RXFILL for BP meds, insert it ahead of AE claims in the factory chain.

There is no RXFILL feed configured in this repo, so ``authenticate`` raises and the
factory never selects it unless one is explicitly wired in.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .base import (
    ACCESS_BATCH_PERMITTED,
    ConnectorAuthError,
    DispenseEvent,
    EncounterContext,
    PharmacyConnector,
    SourceProfile,
)

_ENV_ENDPOINT = "RXFILL_FEED_ENDPOINT"


class RxfillPushConnector(PharmacyConnector):
    """NCPDP SCRIPT RXFILL dispense-notification feed -- documented, low coverage."""

    SOURCE_NAME = "ncpdp_rxfill_push"

    def __init__(self, env: Optional[dict] = None):
        self._env = dict(os.environ if env is None else env)

    @property
    def source_profile(self) -> SourceProfile:
        return SourceProfile(
            source_name=self.SOURCE_NAME,
            access_mode=ACCESS_BATCH_PERMITTED,
            min_latency_days=0,
            typical_latency_days=0,
            max_latency_days=1,
            requires_encounter=False,
            confirms_dispense=True,
        )

    def authenticate(self) -> None:
        if not self._env.get(_ENV_ENDPOINT):
            raise ConnectorAuthError(
                "No RXFILL feed configured "
                f"(${_ENV_ENDPOINT} unset). RXFILL is rarely enabled for chronic "
                "generics like antihypertensives, so this source is not expected to "
                "be available; the factory falls back."
            )
        raise ConnectorAuthError(
            "RXFILL endpoint configured but the live NCPDP SCRIPT subscription is "
            "not implemented in this repo."
        )

    def fetch_dispense_events(
        self,
        patient_ids: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        encounter: Optional[EncounterContext] = None,
    ) -> List[DispenseEvent]:  # pragma: no cover - never authenticates in this repo
        raise NotImplementedError("RXFILL live subscription is not implemented.")

    def covered_patient_ids(self, patient_ids: Iterable[str]) -> set:  # pragma: no cover
        raise NotImplementedError("RXFILL live subscription is not implemented.")
