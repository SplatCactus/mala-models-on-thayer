"""
src/sync/connectors — pharmacy dispense-data connector layer.

One ABC (:class:`PharmacyConnector`) with a declared :class:`SourceProfile`
(access mode + latency + dispense-vs-prescription), several adapters, and a
:func:`get_connector` factory with a Surescripts -> AE claims -> local fallback
chain. Escalation/loop-closure depend on the interface here, never on a concrete
adapter (the adapter choice lives only in ``factory.py``).

This layer covers the *dispense/refill* signal (loop closure, escalation timers).
The sibling module ``src/sync/pharmacy_source.py`` covers the separate "new
patient activity arrived" inbound-feed demo; its public names are re-exported
here as a compatibility shim so a caller can import either concern from one place
and existing ``sync_job.py`` imports keep working.
"""
from __future__ import annotations

from .base import (
    ACCESS_BATCH_PERMITTED,
    ACCESS_PRESCRIBER_INITIATED,
    ConnectorAccessError,
    ConnectorAuthError,
    ConnectorError,
    DispenseEvent,
    EncounterContext,
    PharmacyConnector,
    SourceProfile,
)
from .ae_claims import AeClaimsConnector
from .local_file import LocalFileConnector
from .rxfill import RxfillPushConnector
from .surescripts import SurescriptsConnector
from .factory import (
    DEFAULT_CHAIN,
    SYNC_STATE_PATH,
    build_sync_state,
    get_connector,
    select_connector,
    write_sync_state,
)

# Compatibility shim: re-export the inbound-activity source names so existing
# imports (e.g. src/sync/sync_job.py) can resolve them via connectors too, and so
# the two source concerns are discoverable from one package.
from ..pharmacy_source import (  # noqa: E402
    PharmacyRefillSource,
    RefillRecord,
    SyntheticRIAdapter,
    CurrentCareAdapter,
)

__all__ = [
    "ACCESS_BATCH_PERMITTED",
    "ACCESS_PRESCRIBER_INITIATED",
    "ConnectorAccessError",
    "ConnectorAuthError",
    "ConnectorError",
    "DispenseEvent",
    "EncounterContext",
    "PharmacyConnector",
    "SourceProfile",
    "AeClaimsConnector",
    "LocalFileConnector",
    "RxfillPushConnector",
    "SurescriptsConnector",
    "DEFAULT_CHAIN",
    "SYNC_STATE_PATH",
    "build_sync_state",
    "get_connector",
    "select_connector",
    "write_sync_state",
    # inbound-activity shim
    "PharmacyRefillSource",
    "RefillRecord",
    "SyntheticRIAdapter",
    "CurrentCareAdapter",
]
