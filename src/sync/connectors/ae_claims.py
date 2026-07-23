"""
src/sync/connectors/ae_claims.py

The demo's ACTUAL automated refill-timer backbone: a partner Accountable Entity's
flat claims/dispense export. Unlike Surescripts, an AE may share its own adjudicated
pharmacy-claims extract with us under the BAA, and we may read it in an automated
batch -- so ``access_mode = batch_permitted``. The cost is latency: pharmacy claims
adjudicate on a billing cycle, so a fill shows up ~30 days later and can lag up to
~90 days. That is the latency the escalation guard budgets for.

A paid/adjudicated pharmacy claim is a strong DISPENSE proxy -- the pharmacy billed
for a drug that was dispensed -- so ``confirms_dispense = True``.

EXPECTED FILE FORMAT
--------------------
A CSV or Parquet extract at ``$AE_CLAIMS_EXPORT`` (default
``data/ae_claims_export.parquet``). One row per pharmacy claim line, columns:

    patient_id        cohort patient id (crosswalked by the AE to our id space)
    ndc               dispensed product NDC (string; may be blank)
    rxnorm            RxNorm product code (string; may be blank)
    product_description  human-readable drug name
    fill_date         date the claim was filled/serviced (ISO YYYY-MM-DD)
    days_supply       integer days supply on the claim (may be blank)
    pharmacy_npi      dispensing pharmacy NPI/NCPDP id (may be blank)
    claim_status      "paid" | "reversed" | "denied"  (only "paid" counts as a fill)

Rows with a blank/unparseable ``fill_date`` or a non-"paid" ``claim_status`` are
ignored. The file does not exist in this repo (no real AE feed), so ``authenticate``
raises and the factory falls back to the local synthetic source; drop a real export
at that path and this adapter serves it with no code change.
"""
from __future__ import annotations

import os
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
_ENV_EXPORT = "AE_CLAIMS_EXPORT"
_DEFAULT_EXPORT = _ROOT / "data" / "ae_claims_export.parquet"

_REQUIRED_COLUMNS = ("patient_id", "fill_date", "claim_status")


class AeClaimsConnector(PharmacyConnector):
    """Partner-AE flat pharmacy-claims export -- batch-permitted, 30-90 day lag."""

    SOURCE_NAME = "ae_pharmacy_claims_export"

    def __init__(self, export_path: Optional[Path] = None, env: Optional[dict] = None):
        env = dict(os.environ if env is None else env)
        self._export_path = Path(export_path or env.get(_ENV_EXPORT, _DEFAULT_EXPORT))
        self._df: Optional[pd.DataFrame] = None
        self._last_synced: Optional[str] = None

    @property
    def source_profile(self) -> SourceProfile:
        return SourceProfile(
            source_name=self.SOURCE_NAME,
            access_mode=ACCESS_BATCH_PERMITTED,
            min_latency_days=15,
            typical_latency_days=30,
            max_latency_days=90,
            requires_encounter=False,
            confirms_dispense=True,  # a paid pharmacy claim is a dispense proxy
        )

    def authenticate(self) -> None:
        """Load the export, or raise ConnectorAuthError if it isn't present/valid."""
        if not self._export_path.exists():
            raise ConnectorAuthError(
                f"AE claims export not found at {self._export_path} (set "
                f"${_ENV_EXPORT} to point at a real extract). Falling back."
            )
        df = (pd.read_parquet(self._export_path)
              if self._export_path.suffix == ".parquet"
              else pd.read_csv(self._export_path, dtype=str))
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ConnectorAuthError(
                f"AE claims export {self._export_path} is missing required "
                f"column(s) {missing}; expected {list(_REQUIRED_COLUMNS)} (+ optional "
                "ndc/rxnorm/product_description/days_supply/pharmacy_npi)."
            )
        self._df = df

    def _paid_fills(self, patient_ids: set, start_date: str, end_date: str) -> pd.DataFrame:
        if self._df is None:
            self.authenticate()
        df = self._df
        pid = df["patient_id"].astype(str)
        status = df["claim_status"].astype(str).str.lower() if "claim_status" in df else "paid"
        fill = pd.to_datetime(df["fill_date"], errors="coerce")
        keep = (
            pid.isin(patient_ids)
            & (status == "paid")
            & fill.notna()
            & (fill >= pd.Timestamp(start_date))
            & (fill <= pd.Timestamp(end_date))
        )
        return df[keep].assign(_pid=pid[keep], _fill=fill[keep])

    def fetch_dispense_events(
        self,
        patient_ids: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        encounter: Optional[EncounterContext] = None,  # ignored: batch source
    ) -> List[DispenseEvent]:
        ids = {str(p) for p in patient_ids}
        rows = self._paid_fills(ids, start_date, end_date)
        latency = self.source_profile.typical_latency_days

        def _col(row, name):
            val = row.get(name)
            return None if val is None or (isinstance(val, float) and pd.isna(val)) or val == "" else val

        events: List[DispenseEvent] = []
        for _, row in rows.iterrows():
            days = _col(row, "days_supply")
            events.append(DispenseEvent(
                patient_id=str(row["_pid"]),
                dispense_date=row["_fill"].date().isoformat(),
                product_description=str(_col(row, "product_description") or "antihypertensive fill"),
                ndc=_col(row, "ndc"),
                rxnorm=_col(row, "rxnorm"),
                days_supply=int(days) if days is not None and str(days).strip() != "" else None,
                pharmacy_ncpdp=_col(row, "pharmacy_npi"),
                source=self.SOURCE_NAME,
                is_dispense=True,
                latency_days=latency,
            ))
        self._last_synced = pd.Timestamp.utcnow().isoformat()
        return events

    def covered_patient_ids(self, patient_ids: Iterable[str]) -> set:
        """Patients present in the export at all (any claim line), so a caller can
        tell 'in the feed, no qualifying fill' (a break) from 'not in the feed'."""
        if self._df is None:
            self.authenticate()
        ids = {str(p) for p in patient_ids}
        present = set(self._df["patient_id"].astype(str))
        return ids & present
