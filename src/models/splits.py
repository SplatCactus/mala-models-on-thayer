"""splits.py — temporal train/val/test partitioning for BP Cascade RI.

Patients are split into train/val/test by index_date order (earliest ->
train, middle -> val, latest -> test), NOT randomly. This mirrors real
deployment: the model trains on the past and is evaluated on the future, so
it can never benefit from later-indexed patients leaking into an earlier
split the way a shuffled/random split could.

Leakage rule enforced here (feature/outcome row-level leakage is already
enforced inside src/features/*.py via strictly-before/strictly-after
index_date filters -- this module enforces the *cohort-level* temporal
ordering instead):

    max(index_date in train) <= min(index_date in val)
    max(index_date in val)   <= min(index_date in test)

This is checked with an explicit runtime assertion after every split, not
left as a comment -- a future change to the cut logic (e.g. adding
stratification) that accidentally reorders patients across the boundary
fails loudly instead of silently letting a later patient train the model
that will be validated against an earlier one.

Public API
----------
make_temporal_splits(panel_df, ...) -> pandas.DataFrame
    Columns [patient_id_col, index_date_col, "split"], one row per patient
    with a parseable index_date. split in {"train", "val", "test"}.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "feature_panel.parquet"
OUTPUT_PATH = ROOT / "data" / "snapshots" / "splits.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("splits")


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO date/datetime strings to tz-naive Timestamps (NaT on failure)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _assert_no_temporal_leakage(ordered: pd.DataFrame, index_date_col: str) -> None:
    """Fail loudly if any split boundary is out of chronological order.

    ``ordered`` must already carry a "split" column. This re-checks the
    temporal ordering at runtime instead of trusting it holds "by
    construction" -- the actual code-level enforcement of the leakage rule.
    """
    bounds = ordered.groupby("split")[index_date_col].agg(["min", "max"])
    for earlier, later in (("train", "val"), ("val", "test")):
        if earlier not in bounds.index or later not in bounds.index:
            continue
        earlier_max = bounds.loc[earlier, "max"]
        later_min = bounds.loc[later, "min"]
        if earlier_max > later_min:
            raise AssertionError(
                f"temporal leakage: max({index_date_col}) in '{earlier}' "
                f"({earlier_max}) is after min({index_date_col}) in '{later}' "
                f"({later_min})"
            )


def make_temporal_splits(
    panel_df: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
    index_date_col: str = "index_date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Assign each patient to train/val/test by index_date order.

    Methodology
    -----------
    1. Parse ``index_date_col``; drop patients whose index_date is missing
       or unparseable (logged, not silently included -- they can't be placed
       in a chronological order).
    2. Sort ascending by ``(index_date, patient_id)``. patient_id is a
       deterministic tiebreak for patients sharing a date, which is common
       here since index_date is currently derived as "first medication fill
       day" (see build_features.py) and Synthea dates cluster.
    3. Cut by cumulative fraction of patients: earliest ``train_frac`` ->
       train, next ``val_frac`` -> val, remainder -> test. This is a
       quantile cut on patient count, not a fixed calendar-date cut, since
       the real spread/density of index_date hasn't been validated against
       this cohort yet.
    4. Re-verify chronological ordering across split boundaries at runtime
       (see :func:`_assert_no_temporal_leakage`).

    No stratification by outcome or demographics is applied -- this is a
    pure chronological cut. Per-split outcome/demographic balance is logged
    as a diagnostic only.

    Inputs
    ------
    panel_df : pandas.DataFrame
        Must contain ``patient_id_col`` and ``index_date_col``.

    Returns
    -------
    pandas.DataFrame
        Columns [patient_id_col, index_date_col, "split"]. Patients with a
        missing/unparseable index_date are excluded entirely.
    """
    fracs = (train_frac, val_frac, test_frac)
    if any(f < 0 for f in fracs) or abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must be >=0 and sum to 1.0, got {fracs}")

    cohort = panel_df[[patient_id_col, index_date_col]].copy()
    cohort["_index_date"] = _to_naive_datetime(cohort[index_date_col])

    n_total = len(cohort)
    dated = cohort.dropna(subset=["_index_date"]).copy()
    n_dropped = n_total - len(dated)
    if n_dropped:
        log.warning(
            "  dropping %d/%d patient(s) with missing/unparseable %s",
            n_dropped, n_total, index_date_col,
        )

    dated = dated.sort_values(["_index_date", patient_id_col]).reset_index(drop=True)

    n = len(dated)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val  # remainder, so rounding never drops a patient
    dated["split"] = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)

    _assert_no_temporal_leakage(dated, "_index_date")

    for name in ("train", "val", "test"):
        subset = dated[dated["split"] == name]
        if len(subset):
            log.info(
                "  %-5s %6d patients   index_date %s -> %s",
                name, len(subset), subset["_index_date"].min().date(), subset["_index_date"].max().date(),
            )
        else:
            log.info("  %-5s %6d patients", name, 0)

    return dated[[patient_id_col, index_date_col, "split"]]


def main() -> int:
    log.info("Loading %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)

    log.info("Assigning temporal train/val/test splits...")
    splits = make_temporal_splits(panel)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %s (%d patients)", OUTPUT_PATH, len(splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
