"""
src/sync/connectors/factory.py

Selects the active pharmacy-data source by trying an ordered fallback chain and
returning the first that authenticates. The chain default is
**Surescripts -> AE claims -> local** (override with ``$PHARMACY_SOURCE_CHAIN``,
a comma-separated list of adapter keys). In the demo Surescripts has no credentials
and there is no AE export file, so the chain falls through to the local synthetic
source.

Every attempt is recorded -- which adapter, whether it served or why it failed --
and the whole trace plus the selected source's profile is written to
``data/snapshots/pharmacy_sync_state.json`` so the API can expose a live "which
feed are we on, and how laggy is it" badge (constraint C2). The trace is plain
JSON-serializable dicts, never adapter objects.

CRISP SEAM (do not build here): Rhode Island's RHIO is now **CRISP Shared
Services** (CurrentCare/RIQI was sunset). A CRISP adapter is intentionally NOT
built -- third-party HIE access requires state designation plus a multi-party DUA
-- but it would slot into this chain as another ``PharmacyConnector`` (likely
``batch_permitted`` for an ADT/encounter feed, not a clean antihypertensive
dispense feed), inserted by name in ``_ADAPTERS`` / the chain env var.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .base import ConnectorAuthError, PharmacyConnector
from .ae_claims import AeClaimsConnector
from .local_file import LocalFileConnector
from .rxfill import RxfillPushConnector
from .surescripts import SurescriptsConnector

_ROOT = Path(__file__).resolve().parents[3]
SYNC_STATE_PATH = _ROOT / "data" / "snapshots" / "pharmacy_sync_state.json"

_ENV_CHAIN = "PHARMACY_SOURCE_CHAIN"
DEFAULT_CHAIN = ("surescripts", "ae_claims", "local")

# Adapter key -> zero-arg constructor. rxfill is registered so a chain override can
# insert it, but it is deliberately absent from DEFAULT_CHAIN (low coverage).
_ADAPTERS: Dict[str, Callable[[], PharmacyConnector]] = {
    "surescripts": SurescriptsConnector,
    "ae_claims": AeClaimsConnector,
    "rxfill": RxfillPushConnector,
    "local": LocalFileConnector,
}


def _chain_from_env(env: dict) -> List[str]:
    raw = env.get(_ENV_CHAIN)
    if not raw:
        return list(DEFAULT_CHAIN)
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in _ADAPTERS]
    if unknown:
        raise ValueError(f"unknown pharmacy source(s) {unknown}; known: {sorted(_ADAPTERS)}")
    return keys


def select_connector(
    env: Optional[dict] = None,
) -> Tuple[PharmacyConnector, List[dict]]:
    """Try the chain; return (first authenticating connector, attempt trace).

    The trace is a list of ``{adapter, access_mode, outcome, reason}`` dicts, in the
    order tried -- serializable as-is. Raises ConnectorError only if EVERY source
    fails (local should always succeed in this repo, so that is effectively
    unreachable).
    """
    env = dict(os.environ if env is None else env)
    trace: List[dict] = []
    for key in _chain_from_env(env):
        connector = _ADAPTERS[key]()
        profile = connector.source_profile
        try:
            connector.authenticate()
        except ConnectorAuthError as exc:
            trace.append({
                "adapter": key,
                "source_name": profile.source_name,
                "access_mode": profile.access_mode,
                "outcome": "auth_failed",
                "reason": str(exc),
            })
            continue
        trace.append({
            "adapter": key,
            "source_name": profile.source_name,
            "access_mode": profile.access_mode,
            "outcome": "served",
            "reason": None,
        })
        return connector, trace

    raise ConnectorAuthError(
        "no pharmacy source authenticated. Attempts: "
        + "; ".join(f"{t['adapter']}={t['reason']}" for t in trace)
    )


def build_sync_state(connector: PharmacyConnector, trace: List[dict], *, now: Optional[datetime] = None) -> dict:
    """Serializable sync-state: the selected source's profile + the full attempt trace."""
    now = now or datetime.now(timezone.utc)
    profile = connector.source_profile
    return {
        "selected": profile.to_dict(),
        "last_synced": connector.last_synced or now.isoformat().replace("+00:00", "Z"),
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "attempts": trace,
    }


def write_sync_state(
    connector: PharmacyConnector,
    trace: List[dict],
    *,
    path: Path = SYNC_STATE_PATH,
    now: Optional[datetime] = None,
) -> dict:
    """Write pharmacy_sync_state.json (source_name, access_mode, latency, last_synced)."""
    state = build_sync_state(connector, trace, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return state


def get_connector(*, env: Optional[dict] = None, write_state: bool = True) -> PharmacyConnector:
    """Select the active source via the fallback chain, writing the sync-state trace.

    This is the one place a concrete adapter is named; every other module takes the
    returned ``PharmacyConnector`` interface, so nothing downstream depends on which
    source served.
    """
    connector, trace = select_connector(env)
    if write_state:
        write_sync_state(connector, trace)
    return connector
