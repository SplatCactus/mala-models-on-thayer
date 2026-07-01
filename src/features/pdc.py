"""Antihypertensive PDC outcome labels (forward-looking, post-index).

This module produces the MODEL TARGET, not a pre-index feature. For each patient we
look at the fixed follow-up window immediately AFTER their index_date (the first
antihypertensive fill) and summarize how well they stayed on therapy:

    pdc_180d         Proportion of Days Covered over [index_date, index_date + 180d)
    has_30_day_gap   1 if there is any continuous uncovered stretch >= 30 days in
                     that window (a discontinuation signal), else 0

Window convention: the follow-up window is the half-open interval
``[index_date, index_date + 180d)`` — it *starts at* index_date (the first fill's
coverage begins there) and runs forward 180 days. Because index_date is the first
fill, there is no pre-index coverage to worry about; this is a clean forward label.

Overlap handling: early refills overlap. Overlapping days-supply is counted ONCE by
taking the union of per-fill coverage intervals (sort-and-sweep merge) before both
the PDC numerator and the gap scan. So a 30-day refill picked up 5 days early adds
25 net covered days, not 30.

Public API
----------
calculate_pdc_outcome(cohort_df, meds_df, ...) -> pandas.DataFrame
    Columns ``[patient_id, pdc_180d, has_30_day_gap]``, one row per cohort patient.

See code_dictionary.yaml (rxnorm:) for the antihypertensive code groupings and
SCHEMA.md for the medications table columns.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# RxNorm code sets
# ---------------------------------------------------------------------------
# INGREDIENT-LEVEL concepts from code_dictionary.yaml (rxnorm:). Kept as the
# documented clinical source of truth. NOTE: this dataset's medications.csv stores
# PRODUCT-level codes, so filtering fills on these ingredient codes matches 0 rows.
ANTIHYPERTENSIVE_RXNORM_INGREDIENT_LEVEL: frozenset[str] = frozenset({
    # ACE inhibitors
    "29046", "18867", "3827", "35208", "35296",
    # ARBs
    "73494", "69749", "83515", "73478", "321064",
    # Calcium channel blockers
    "17767",
    # Thiazide diuretics
    "5487",
    # Beta-blockers
    "6918", "19484", "20352", "33306", "52175", "7801", "6984",
})

# PRODUCT-LEVEL codes actually present in this dataset's medications.parquet,
# mirrored from src/etl/cohort.py. This is the DEFAULT used below because it is the
# set that actually matches fills — see [[rxnorm-product-vs-ingredient]] in memory.
ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL: frozenset[str] = frozenset({
    # beta-blockers
    "866412", "866419", "866427", "866429", "866436",  # metoprolol succinate ER
    "866511", "866924", "866514",                       # metoprolol tartrate
    "197379", "197381",                                 # atenolol
    "200032", "200033", "686924",                       # carvedilol
    # thiazide diuretics
    "310798",                                           # hydrochlorothiazide (mono)
    # CCB
    "308136",                                           # amlodipine (mono)
    "897685",                                           # verapamil
    # ACE-I
    "858804", "858810",                                 # enalapril
    "314076", "311353", "314077", "197884", "311354",   # lisinopril
    # ARB
    "979480", "979485", "979492",                       # losartan
    "349200",                                           # valsartan (mono)
    # combination products
    "999970",   # amlodipine/HCTZ/olmesartan
    "200285",   # HCTZ/valsartan
    "197887",   # HCTZ/lisinopril
    "979471",   # HCTZ/losartan
    "1656356",  # sacubitril/valsartan (Entresto)
})

# Default for calculate_pdc_outcome(): the product-level set that matches real fills.
ANTIHYPERTENSIVE_RXNORM = ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO date/datetime strings to tz-naive, day-normalized Timestamps (NaT on failure)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def _coverage_and_max_gap(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[int, int]:
    """Union coverage within a window and the longest uncovered stretch inside it.

    Given raw ``[start, end)`` coverage intervals (already clipped to the window),
    this:
      1. Sorts by start and sweeps left to right, merging overlapping/contiguous
         intervals so overlapping days-supply is counted once.
      2. Sums merged interval lengths -> ``covered_days`` (PDC numerator).
      3. Walks the window from ``window_start`` to ``window_end`` tracking the
         largest gap between consecutive covered blocks. Gaps at the very start
         (window_start -> first coverage) and end (last coverage -> window_end) are
         included, so a patient who stops refilling mid-window is caught by the
         trailing gap.

    Returns ``(covered_days, max_gap_days)``. With no intervals the whole window is
    one gap: ``(0, (window_end - window_start).days)``.
    """
    window_len = (window_end - window_start).days
    if not intervals:
        return 0, window_len

    intervals.sort(key=lambda iv: iv[0])

    # Merge overlapping intervals into disjoint covered blocks.
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    covered_days = sum((e - s).days for s, e in merged)

    # Scan for the longest uncovered stretch, including leading and trailing gaps.
    max_gap = 0
    cursor = window_start
    for s, e in merged:
        gap = (s - cursor).days
        if gap > max_gap:
            max_gap = gap
        if e > cursor:
            cursor = e
    trailing = (window_end - cursor).days
    if trailing > max_gap:
        max_gap = trailing

    return covered_days, max_gap


def calculate_pdc_outcome(
    cohort_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    antihypertensive_codes: frozenset[str] | set[str] | None = ANTIHYPERTENSIVE_RXNORM,
    *,
    cohort_patient_col: str = "patient_id",
    index_date_col: str = "index_date",
    med_patient_col: str = "PATIENT",
    code_col: str = "CODE",
    start_col: str = "START",
    stop_col: str = "STOP",
    forward_days: int = 180,
    gap_threshold_days: int = 30,
    default_days_supply: int = 30,
) -> pd.DataFrame:
    """Compute forward-looking PDC outcome labels per cohort patient.

    Methodology
    -----------
    For each patient, over the follow-up window ``[index_date, index_date + forward_days)``:

    1. Select that patient's antihypertensive fills (rows whose ``code_col`` is in
       ``antihypertensive_codes``; pass ``None`` to use every fill unfiltered).
    2. Turn each fill into a coverage interval ``[START, START + days_supply)``, where
       ``days_supply`` is ``(STOP - START)`` when a valid later STOP exists, else
       ``default_days_supply`` (30). Clip each interval to the follow-up window.
    3. Union the clipped intervals so overlapping early refills are not double-counted.
    4. ``pdc_180d = covered_days / forward_days`` (capped at 1.0).
    5. ``has_30_day_gap = 1`` if the longest uncovered stretch in the window is
       ``>= gap_threshold_days`` (default 30), else 0. Leading and trailing gaps count,
       so failing to refill and dropping off therapy both register.

    Inputs
    ------
    cohort_df : pandas.DataFrame
        One row per patient. Must contain ``cohort_patient_col`` and ``index_date_col``
        (the first-medication-fill date).
    meds_df : pandas.DataFrame
        Raw medication fill history. Must contain ``med_patient_col``, ``code_col``,
        ``start_col`` and ``stop_col``.
    antihypertensive_codes : set of str or None, optional
        RxNorm codes counting as antihypertensive. Defaults to the product-level set
        that matches this dataset's fills. ``None`` disables filtering.

    Returns
    -------
    pandas.DataFrame
        Columns ``[cohort_patient_col, "pdc_180d", "has_30_day_gap"]``, one row per
        cohort patient in input order. A patient with no qualifying fills in the
        window gets ``pdc_180d = 0.0`` and ``has_30_day_gap = 1`` (full-window gap =
        complete non-persistence). A patient with a missing/unparseable index_date
        gets NaN for both (cannot label — drop before training).

    Notes
    -----
    Pure function: no I/O, inputs not mutated. This is a TARGET; keep it out of the
    feature matrix to avoid leakage.
    """
    # --- Cohort side: id + parsed index date --------------------------------
    cohort = cohort_df[[cohort_patient_col, index_date_col]].copy()
    cohort["_index_date"] = _to_naive_datetime(cohort[index_date_col])

    # --- Medication side: filter + normalize --------------------------------
    meds = meds_df
    if antihypertensive_codes is not None:
        meds = meds[meds[code_col].astype(str).isin(set(antihypertensive_codes))]
    meds = meds.copy()
    meds["_start"] = _to_naive_datetime(meds[start_col])
    meds["_stop"] = _to_naive_datetime(meds[stop_col])
    meds = meds.dropna(subset=["_start"])

    meds = meds.merge(
        cohort[[cohort_patient_col, "_index_date"]],
        left_on=med_patient_col,
        right_on=cohort_patient_col,
        how="inner",
        suffixes=("", "_cohort"),
    )

    # Per-fill coverage end: real STOP when valid, else START + default supply.
    supply_end = meds["_start"] + pd.to_timedelta(default_days_supply, unit="D")
    valid_stop = meds["_stop"].notna() & (meds["_stop"] > meds["_start"])
    meds["_end"] = supply_end.where(~valid_stop, meds["_stop"])

    forward = pd.to_timedelta(forward_days, unit="D")

    # --- Per-patient outcome ------------------------------------------------
    results: dict[str, dict[str, float]] = {}
    for pid, grp in meds.groupby(med_patient_col, sort=False):
        index_date = grp["_index_date"].iloc[0]
        window_start = index_date
        window_end = index_date + forward  # exclusive

        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for start, end in zip(grp["_start"], grp["_end"]):
            clip_start = max(start, window_start)
            clip_end = min(end, window_end)
            if clip_start < clip_end:
                intervals.append((clip_start, clip_end))

        covered, max_gap = _coverage_and_max_gap(intervals, window_start, window_end)
        results[pid] = {
            "pdc_180d": min(covered / forward_days, 1.0),
            "has_30_day_gap": int(max_gap >= gap_threshold_days),
        }

    # --- Assemble output, one row per cohort patient ------------------------
    def _row(r: pd.Series) -> pd.Series:
        if pd.isna(r["_index_date"]):
            return pd.Series({"pdc_180d": float("nan"), "has_30_day_gap": float("nan")})
        stats = results.get(r[cohort_patient_col])
        if stats is None:
            # In-cohort patient with no antihypertensive fills in the window at all:
            # complete non-persistence -> 0% covered, full-window gap.
            return pd.Series({"pdc_180d": 0.0, "has_30_day_gap": 1})
        return pd.Series(stats)

    out = cohort[[cohort_patient_col]].copy()
    stats_df = cohort.apply(_row, axis=1)
    out = pd.concat([out, stats_df], axis=1)
    return out[[cohort_patient_col, "pdc_180d", "has_30_day_gap"]]
