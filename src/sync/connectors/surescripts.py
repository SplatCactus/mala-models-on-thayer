"""
src/sync/connectors/surescripts.py

PRIMARY source, intentionally STUBBED: Surescripts Medication History over NCPDP
SCRIPT. No real credentials, and -- critically -- no fake batch path. This adapter
models the real request/response SHAPES and then REFUSES the one thing our
pipeline would want but is not allowed to do: an automated background batch query.

WHY IT CAN'T BE THE TIMER BACKBONE (constraint C2)
--------------------------------------------------
Surescripts Medication History is **prescriber-initiated and treatment-tied**: a
licensed prescriber may pull a specific patient's medication history in connection
with a real clinical encounter. It is contractually NOT a background feed a
third-party platform may batch-poll for a panel of patients. So:
  * ``access_mode = prescriber_initiated`` and ``requires_encounter = True``;
  * ``fetch_dispense_events`` raises ``ConnectorAccessError`` for any call lacking
    a prescriber identity + encounter context -- i.e. exactly the batch query our
    escalation timers would otherwise want. We do not pretend that works.
It therefore contributes *opportunistic* observations (when a clinician happens to
query during an encounter), never the automated cadence -- that is AE claims
(see ae_claims.py). Latency 1-14 days (PBM/history propagation).

WHAT GOING LIVE REQUIRES (do not assume any of this exists yet)
---------------------------------------------------------------
  * Surescripts certification -- a 12-18 month process -- OR integration through
    an already-certified middleware / EHR partner that fronts the connection.
  * A licensed prescriber with an ACTIVE Surescripts Prescriber ID (SPI); queries
    are attributable to that prescriber.
  * Each query permitted ONLY in connection with treatment of a specific patient
    in a healthcare setting (no bulk/background panel pulls).
  * Transport + auth: production is a mutually-authenticated (client-cert / signed)
    NCPDP SCRIPT exchange to the Surescripts endpoint, not a public REST API.
  * NOTE: RXFILL (NCPDP SCRIPT) push WOULD give near-real-time dispense
    confirmation without a prescriber pull -- but it is rarely implemented for
    chronic generics like antihypertensives, so it is not a reliable backbone
    either (see rxfill.py).

When a certified connection exists, only this file changes: ``authenticate()``
establishes the session and ``_send`` posts the ``RxHistoryRequest`` below and
parses the ``RxHistoryResponse`` into ``DispenseEvent``s. Nothing downstream moves.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .base import (
    ACCESS_PRESCRIBER_INITIATED,
    ConnectorAccessError,
    ConnectorAuthError,
    DispenseEvent,
    EncounterContext,
    PharmacyConnector,
    SourceProfile,
)

# Environment variables a real (or middleware-fronted) connection would read.
_ENV_ACCOUNT = "SURESCRIPTS_ACCOUNT_ID"
_ENV_SPI = "SURESCRIPTS_PRESCRIBER_SPI"
_ENV_SECRET = "SURESCRIPTS_CLIENT_SECRET"
_ENV_ENDPOINT = "SURESCRIPTS_ENDPOINT"


class SurescriptsConnector(PharmacyConnector):
    """Surescripts Medication History (NCPDP SCRIPT) -- primary, stubbed, no batch."""

    SOURCE_NAME = "surescripts_medication_history"

    def __init__(self, env: Optional[dict] = None):
        # Read config lazily from the environment; construction never fails so the
        # factory can read source_profile and record an attempt even with no creds.
        self._env = dict(os.environ if env is None else env)

    @property
    def source_profile(self) -> SourceProfile:
        return SourceProfile(
            source_name=self.SOURCE_NAME,
            access_mode=ACCESS_PRESCRIBER_INITIATED,
            min_latency_days=1,
            typical_latency_days=7,
            max_latency_days=14,
            requires_encounter=True,
            # Medication History includes dispensed/fill records, so a positive is a
            # true dispense (not merely a prescription order).
            confirms_dispense=True,
        )

    def authenticate(self) -> None:
        """Establish a session, or raise ConnectorAuthError naming what's missing.

        In the demo this always raises (no credentials / no certified connection),
        which is the signal for the factory to fall back to the next source.
        """
        missing = [k for k in (_ENV_ACCOUNT, _ENV_SPI, _ENV_SECRET, _ENV_ENDPOINT)
                   if not self._env.get(k)]
        if missing:
            raise ConnectorAuthError(
                "Surescripts is not connected: missing "
                f"{', '.join(missing)}. Going live requires Surescripts "
                "certification (12-18 months) or a certified middleware partner, "
                "an active prescriber Surescripts Prescriber ID (SPI), and "
                "treatment-context authorization -- see module docstring."
            )
        # A real build would open the mutually-authenticated NCPDP SCRIPT session
        # here. We never reach this in the demo (no creds), and we deliberately do
        # not fabricate a session.
        raise ConnectorAuthError(
            "Surescripts credentials present but live NCPDP SCRIPT transport is not "
            "implemented in this repo (no certified connection). This is the "
            "intentional credential stub."
        )

    def _build_rx_history_request(
        self, patient_id: str, encounter: EncounterContext,
        start_date: str, end_date: str,
    ) -> dict:
        """Shape a single-patient NCPDP SCRIPT RxHistoryRequest.

        Modeled on the real message structure (MessageHeader / Patient / Prescriber
        / RxHistoryRequest body). One patient per request, attributed to a
        prescriber + encounter -- there is deliberately no "list of patients" field,
        because the protocol has no batch-panel concept.
        """
        return {
            "Message": {
                "Header": {
                    "To": {"Qualifier": "SureScripts", "Value": self._env.get(_ENV_ENDPOINT)},
                    "From": {"Qualifier": "AccountId", "Value": self._env.get(_ENV_ACCOUNT)},
                    "MessageID": f"rxhist-{encounter.encounter_id}-{patient_id}",
                    "SentTime": None,  # set at send time
                    "PrescriberOrderNumber": None,
                },
                "Body": {
                    "RxHistoryRequest": {
                        "Patient": {"HumanPatient": {"Identification": {"PatientId": patient_id}}},
                        "Prescriber": {"Identification": {"SPI": encounter.prescriber_spi}},
                        "BenefitsCoordination": {},
                        # Treatment context is mandatory and audited.
                        "RequestParameters": {
                            "EncounterId": encounter.encounter_id,
                            "BeginDate": start_date,
                            "EndDate": end_date,
                        },
                    }
                },
            }
        }

    @staticmethod
    def _parse_rx_history_response(response: dict) -> List[DispenseEvent]:  # pragma: no cover
        """Parse an NCPDP SCRIPT RxHistoryResponse into DispenseEvents.

        Real responses carry MedicationDispensed and MedicationPrescribed segments;
        only MedicationDispensed (a confirmed fill: DrugCoded NDC/RxNorm, Quantity,
        DaysSupply, LastFillDate, Pharmacy) becomes an ``is_dispense=True`` event.
        Unreachable in the demo (we never obtain a live response); kept as the
        documented parse seam so wiring the real feed is mechanical.
        """
        raise NotImplementedError(
            "RxHistoryResponse parsing runs only against a live Surescripts "
            "connection, which this repo does not have."
        )

    def fetch_dispense_events(
        self,
        patient_ids: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        encounter: Optional[EncounterContext] = None,
    ) -> List[DispenseEvent]:
        """Refuse a batch query outright; a real single-patient query is unimplemented.

        The escalation pipeline wants "give me dispenses for this whole panel" --
        which is precisely the automated batch pull Surescripts prohibits. We raise
        ``ConnectorAccessError`` for it rather than build a path that pretends it
        works. A legitimate single-patient, encounter-attributed query would proceed
        to ``_build_rx_history_request`` / ``_send`` (unimplemented: no live conn).
        """
        ids = [str(p) for p in patient_ids]
        if encounter is None:
            raise ConnectorAccessError(
                "Surescripts Medication History cannot be batch-queried: it is "
                "prescriber-initiated and must be attached to a specific patient's "
                "clinical encounter (a licensed prescriber's SPI + an encounter "
                "context). An automated background panel pull "
                f"({len(ids)} patients, no encounter) is contractually prohibited. "
                "Use the AE claims feed for the automated refill-timer backbone."
            )
        if len(ids) != 1 or encounter.patient_id not in ids:
            raise ConnectorAccessError(
                "Surescripts queries are single-patient and treatment-scoped: the "
                "encounter's patient must be the one (and only) id queried."
            )
        # Legitimate single-patient, encounter-attributed query -- would send here.
        self._build_rx_history_request(ids[0], encounter, start_date, end_date)
        raise NotImplementedError(
            "Surescripts live query not implemented (no certified connection). "
            "The request shape is built above; wiring _send/_parse is all that "
            "remains once a connection exists."
        )

    def covered_patient_ids(self, patient_ids: Iterable[str]) -> set:
        # Coverage is only knowable per-encounter query; there is no panel-level
        # "who is in this feed" concept for a prescriber-initiated source.
        raise ConnectorAccessError(
            "Surescripts has no batch coverage concept: membership is only "
            "resolved per prescriber-initiated, single-patient query."
        )
