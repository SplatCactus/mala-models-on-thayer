"""
src/routing/consent.py

Two-scope, per-patient consent gating for the escalation ladder.

WHY TWO SCOPES, AND THE LEGAL BASIS (constraint C3)
---------------------------------------------------
Not every routed action has the same disclosure footprint, so one consent flag
is not enough. We model TWO scopes, evaluated independently per patient:

  * ``internal_care_coordination`` -- routing to staff INSIDE the Accountable
    Entity as a covered entity (CHW, social worker, pharmacist, prescriber).
    Legal basis: the Business Associate Agreement + the treatment relationship.
    Under HIPAA this is healthcare *operations* / treatment, which the BAA
    already authorizes -- it does NOT require a separately signed patient form.
    So in our synthetic data internal defaults to granted for everyone; it is
    still modeled (and can be explicitly *denied* for a patient who revoked all
    data sharing) because the system must be able to represent that.

  * ``external_disclosure`` -- routing to a NON-covered entity (a transit broker,
    a community-based organization). Legal basis: **R.I. Gen. Laws § 5-37.3-4**
    (the RI Confidentiality of Health Care Communications and Information Act),
    which is stricter than HIPAA and requires **explicit signed patient
    authorization** to disclose identifiable health information to a non-covered
    entity. The BAA does NOT cover this. So external must be affirmatively
    granted, and anything short of a fresh, signed "granted" fails closed.

CONSENT IS DATA, NEVER INFERRED
-------------------------------
Consent arrives FROM the AE with a ``source`` and an ``as_of`` date; we never
infer it from behavior. Three states per scope, and "unknown" is deliberately
distinct from "denied": a patient whose status we simply have not received has
NOT declined, and the API must be able to tell the two apart (e.g. to prompt the
AE to collect authorization vs. to respect a refusal).

STALENESS, FAIL-CLOSED
----------------------
A § 5-37.3-4 authorization is a point-in-time signed document, not a standing
state, so consent older than ``CONSENT_VALIDITY_DAYS`` is treated as **absent**.
A missing OR stale OR unknown OR denied status all fail **closed** (not
authorized). Fail-closed is the safe direction: the failure mode we must avoid is
an unauthorized disclosure, so ambiguity resolves to "do not disclose."

GATING NEVER DROPS A PATIENT
----------------------------
Every action declares a required scope (``ACTION_REQUIRED_SCOPE``). ``gate()``
returns a machine-readable :class:`ConsentDecision`; when an EXTERNAL action is
not authorized, the escalation engine substitutes a documented INTERNAL fallback
(e.g. transit-voucher -> CHW-mediated transport help) rather than silently
dropping the patient. The reason string is always populated so the UI can explain
why an action was gated.

NOT A MODEL FEATURE
-------------------
Consent is an operational field and must never influence the risk model. It has
no allowlisted prefix, so ``src/models/common.py::select_feature_columns`` already
excludes it by construction -- ``tests/test_consent.py`` asserts that rather than
trusting the docstring.

Public API
----------
ConsentRecord / ConsentDecision           dataclasses (JSON-round-trippable)
INTERNAL / EXTERNAL, GRANTED/DENIED/UNKNOWN  scope + state constants
ACTION_REQUIRED_SCOPE                      action -> required scope
scope_status(record, scope, today)         -> {state, stale, allowed, ...} for the API
gate(action, record, today)                -> ConsentDecision
load_consent(path)                         -> {patient_id: ConsentRecord}
generate_synthetic_consent(...) / main()   writes the documented synthetic consent.json
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
CONSENT_PATH = ROOT / "data" / "snapshots" / "consent.json"
ROUTING_TABLE_PATH = ROOT / "data" / "snapshots" / "routing_table.json"

# Scopes.
INTERNAL = "internal_care_coordination"
EXTERNAL = "external_disclosure"
SCOPES = (INTERNAL, EXTERNAL)

# Per-scope states. UNKNOWN ("not received") is distinct from DENIED ("declined").
GRANTED = "granted"
DENIED = "denied"
UNKNOWN = "unknown"

# Policy: how long a signed authorization stays valid before it must be re-obtained.
# A § 5-37.3-4 authorization is point-in-time; 365 days is a tunable default.
CONSENT_VALIDITY_DAYS = 365

# Each routing action's required scope. Everything inside the covered entity is
# internal; only disclosure to a non-covered entity (the transit voucher) is
# external. `chw_transport_support` is the internal fallback for a gated voucher.
ACTION_REQUIRED_SCOPE: Dict[str, str] = {
    "chw_pharmacy": INTERNAL,        # Round 0: CHW contacts the pharmacy
    "social_worker": INTERNAL,
    "pharmacist": INTERNAL,
    "bilingual_chw": INTERNAL,
    "prescriber": INTERNAL,          # Round 2
    "chw_transport_support": INTERNAL,  # internal fallback for a gated transit voucher
    "transit_voucher": EXTERNAL,     # Round 1 transport: discloses to a transit broker
}


@dataclass(frozen=True)
class ConsentRecord:
    """One patient's consent status per scope, as received from the AE.

    ``internal`` / ``external`` are states (GRANTED/DENIED/UNKNOWN). ``as_of`` is
    the date the AE captured the status (staleness is measured from it). A patient
    with no record at all is treated as UNKNOWN on both scopes (see ``for_patient``).
    """
    patient_id: str
    internal: str
    external: str
    source: Optional[str]
    as_of: Optional[str]

    def state(self, scope: str) -> str:
        if scope == INTERNAL:
            return self.internal
        if scope == EXTERNAL:
            return self.external
        raise ValueError(f"unknown consent scope {scope!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("patient_id")
        return d

    @classmethod
    def for_patient(cls, patient_id: str, raw: Optional[dict]) -> "ConsentRecord":
        """Build a record from a raw dict, or an all-UNKNOWN record if none exists.

        A missing patient is UNKNOWN (status not received) -- never DENIED.
        """
        raw = raw or {}
        return cls(
            patient_id=str(patient_id),
            internal=raw.get("internal", UNKNOWN),
            external=raw.get("external", UNKNOWN),
            source=raw.get("source"),
            as_of=raw.get("as_of"),
        )


@dataclass(frozen=True)
class ConsentDecision:
    """The result of gating one action against one patient's consent."""
    action: str
    scope: str
    state: str          # GRANTED / DENIED / UNKNOWN (raw state for this scope)
    stale: bool
    allowed: bool
    reason: str         # machine-readable, e.g. "authorized:internal_care_coordination"

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def is_stale(as_of: Optional[str], today: date, *, validity_days: int = CONSENT_VALIDITY_DAYS) -> bool:
    """A status with no date, or older than the validity window, is stale (absent)."""
    parsed = _parse_date(as_of)
    if parsed is None:
        return True  # no date == cannot prove currency == fail closed
    return (today - parsed).days > validity_days


def scope_status(record: ConsentRecord, scope: str, today: date) -> dict:
    """Per-scope status for the API: raw state, staleness, and effective allow."""
    state = record.state(scope)
    stale = is_stale(record.as_of, today)
    allowed = (state == GRANTED) and not stale
    return {
        "scope": scope,
        "state": state,
        "as_of": record.as_of,
        "source": record.source,
        "stale": stale,
        "allowed": allowed,
    }


def gate(action: str, record: Optional[ConsentRecord], today: date) -> ConsentDecision:
    """Decide whether ``action`` is authorized for this patient, fail-closed.

    Unknown, denied, or stale all yield ``allowed=False`` with a reason that keeps
    "unknown" (not received) distinct from "denied" (declined) so the caller can
    respond differently. A missing record is UNKNOWN, never DENIED.
    """
    if action not in ACTION_REQUIRED_SCOPE:
        raise ValueError(
            f"action {action!r} declares no required consent scope; add it to "
            f"ACTION_REQUIRED_SCOPE (known: {sorted(ACTION_REQUIRED_SCOPE)})."
        )
    scope = ACTION_REQUIRED_SCOPE[action]
    rec = record or ConsentRecord.for_patient("", None)
    state = rec.state(scope)
    stale = is_stale(rec.as_of, today)
    allowed = (state == GRANTED) and not stale

    if allowed:
        reason = f"authorized:{scope}"
    elif state == DENIED:
        reason = f"denied:{scope}"
    elif state == GRANTED and stale:
        reason = f"stale_consent:{scope}:as_of={rec.as_of}"
    else:  # UNKNOWN (incl. missing record)
        reason = f"unknown_not_received:{scope}"

    return ConsentDecision(
        action=action, scope=scope, state=state, stale=stale, allowed=allowed, reason=reason
    )


def load_consent(path: Path = CONSENT_PATH) -> Dict[str, ConsentRecord]:
    """Load the consent map (patient_id -> ConsentRecord), or {} if the file is absent.

    A missing file means every patient is UNKNOWN (fail-closed downstream), which is
    the honest default before any AE consent feed exists.
    """
    if not Path(path).exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    return {
        str(pid): ConsentRecord.for_patient(pid, raw)
        for pid, raw in payload.get("patients", {}).items()
    }


# ---------------------------------------------------------------------------
# Synthetic consent generation (demo only)
# ---------------------------------------------------------------------------

def _bucket(patient_id: str) -> int:
    """Deterministic 0-99 bucket from the patient id (stable, no RNG, reproducible)."""
    return int(hashlib.sha1(str(patient_id).encode()).hexdigest(), 16) % 100


def generate_synthetic_consent(
    patient_ids: Iterable[str],
    *,
    fresh_as_of: str = "2026-06-15",
    stale_as_of: str = "2024-01-01",  # >365d before mid-2026 -> stale
    source: str = "AE_intake (SYNTHETIC)",
) -> dict:
    """Build a SYNTHETIC consent map that exercises every gating path.

    internal is granted for everyone (BAA basis; default true per the plan);
    external is mixed by a deterministic bucket so gating is visibly exercised:
      ~55% granted+fresh (authorized), ~20% denied, ~15% unknown, ~10% granted+stale.
    A small deterministic slice of patients is OMITTED entirely so the
    missing-record -> UNKNOWN-on-both path is real too.
    """
    patients: Dict[str, dict] = {}
    for pid in patient_ids:
        pid = str(pid)
        b = _bucket(pid)
        if b < 3:
            continue  # ~3% omitted entirely -> no record -> unknown on both scopes
        if b < 58:
            external, as_of = GRANTED, fresh_as_of
        elif b < 78:
            external, as_of = DENIED, fresh_as_of
        elif b < 93:
            external, as_of = UNKNOWN, None
        else:
            external, as_of = GRANTED, stale_as_of  # granted-but-stale -> fails closed
        patients[pid] = {
            "internal": GRANTED,
            "external": external,
            "source": source,
            "as_of": as_of,
        }
    return {
        "_note": (
            "SYNTHETIC consent data for the demo -- NOT real patient authorization. "
            "Generated deterministically from patient ids by src/routing/consent.py to "
            "exercise the two-scope gating. Real values would arrive from the AE's "
            "consent feed with genuine signed-authorization dates."
        ),
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "synthetic": True,
            "validity_days": CONSENT_VALIDITY_DAYS,
            "n_patients": len(patients),
        },
        "patients": patients,
    }


def _routed_ids(routing_table_path: Path) -> list:
    with open(routing_table_path) as f:
        payload = json.load(f)
    return [c["patient_id"] for c in payload.get("capped_worklist", [])]


def main() -> int:
    ids = _routed_ids(ROUTING_TABLE_PATH)
    payload = generate_synthetic_consent(ids)
    CONSENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONSENT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    n = len(payload["patients"])
    print(f"wrote {n} SYNTHETIC consent records ({len(ids) - n} omitted -> unknown) "
          f"-> {CONSENT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
