"""Blood-pressure trajectory features over a pre-index lookback window.

For each cohort patient we summarize their systolic and diastolic readings in the
365 days *strictly prior* to index_date — i.e. the half-open interval
``[index_date - 365d, index_date)``. For each of the two channels we emit:

    mean, max, min, latest_reading (closest to index_date), trajectory_slope

``trajectory_slope`` is the slope of an ordinary-least-squares line fit of the
reading value against time (in days relative to index_date). A positive slope
means BP is trending *up* as the patient approaches index_date — the early-warning
signal called out in code_dictionary.yaml (loinc.blood_pressure). Units are
mmHg per day.

Public API
----------
compute_bp_trajectory(cohort_df, vitals_df, bp_codes=...) -> pandas.DataFrame
    One row per cohort patient with the ten BP features (5 per channel).

BP LOINC codes (from code_dictionary.yaml -> loinc.blood_pressure):
    systolic  = 8480-6
    diastolic = 8462-4
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default LOINC codes for the two BP channels (source: code_dictionary.yaml).
BP_LOINC_CODES: dict[str, str] = {
    "systolic": "8480-6",
    "diastolic": "8462-4",
}

# Short column prefixes for each channel in the output frame.
_PREFIX = {"systolic": "sbp", "diastolic": "dbp"}

# The five features produced per channel.
_FEATURES = ("mean", "max", "min", "latest_reading", "trajectory_slope")


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO date/datetime strings to tz-naive pandas Timestamps (NaT on failure)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _slope(days: np.ndarray, values: np.ndarray) -> float:
    """OLS slope of ``values`` regressed on ``days`` (mmHg per day).

    Returns NaN when a slope is undefined: fewer than two readings, or all
    readings on the same day (zero variance in the time axis -> vertical line).
    """
    if len(values) < 2:
        return float("nan")
    if np.ptp(days) == 0:  # all timestamps identical -> slope undefined
        return float("nan")
    # polyfit degree-1: returns [slope, intercept].
    slope, _ = np.polyfit(days, values, 1)
    return float(slope)


def _channel_stats(group: pd.DataFrame, index_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    """Compute the five features for one channel of one patient.

    ``group`` holds that patient's in-window readings for a single channel, with
    columns ``_date`` (Timestamp) and ``_value`` (float). ``latest_reading`` is the
    value at the row closest to (but still before) index_date.
    """
    keys = {f"{prefix}_{f}": float("nan") for f in _FEATURES}
    if group.empty:
        return keys

    values = group["_value"].to_numpy(dtype=float)
    dates = group["_date"]
    # Time axis: days relative to index_date (negative; nearer 0 == closer to index).
    days = (dates - index_date).dt.total_seconds().to_numpy(dtype=float) / 86400.0

    latest_idx = dates.to_numpy().argmax()  # most recent reading == closest to index

    keys[f"{prefix}_mean"] = float(np.mean(values))
    keys[f"{prefix}_max"] = float(np.max(values))
    keys[f"{prefix}_min"] = float(np.min(values))
    keys[f"{prefix}_latest_reading"] = float(values[latest_idx])
    keys[f"{prefix}_trajectory_slope"] = _slope(days, values)
    return keys


def compute_bp_trajectory(
    cohort_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    bp_codes: dict[str, str] = BP_LOINC_CODES,
    *,
    cohort_patient_col: str = "Id",
    index_date_col: str = "index_date",
    vitals_patient_col: str = "PATIENT",
    code_col: str = "CODE",
    date_col: str = "DATE",
    value_col: str = "VALUE",
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Compute systolic & diastolic BP trajectory features per cohort patient.

    Methodology
    -----------
    For each patient, restrict ``vitals_df`` to BP readings (``code_col`` matching
    the systolic/diastolic codes in ``bp_codes``) whose ``date_col`` falls in
    ``[index_date - lookback_days, index_date)`` — strictly before index_date.
    Reading values are coerced to float (non-numeric values dropped). For each
    channel we then compute:

      * ``mean`` / ``max`` / ``min`` — over the in-window readings.
      * ``latest_reading`` — the value of the reading closest to index_date
        (i.e. the most recent one in the window).
      * ``trajectory_slope`` — OLS slope of value vs. days-relative-to-index
        (mmHg/day). Positive == rising toward index_date. NaN if <2 readings or
        all readings share one date.

    Inputs
    ------
    cohort_df : pandas.DataFrame
        One row per cohort patient. Must contain ``cohort_patient_col`` and
        ``index_date_col``.
    vitals_df : pandas.DataFrame
        Long observation table (Synthea observations.csv, or a BP-filtered subset).
        Must contain ``vitals_patient_col``, ``code_col``, ``date_col``, ``value_col``.
    bp_codes : dict, optional
        Mapping ``{"systolic": <loinc>, "diastolic": <loinc>}``. Defaults to
        BP_LOINC_CODES (8480-6 / 8462-4).

    Keyword args remap column names and tune the lookback length.

    Returns
    -------
    pandas.DataFrame
        One row per cohort patient: ``cohort_patient_col`` plus, for each channel
        (``sbp_*`` systolic, ``dbp_*`` diastolic), the columns
        ``{prefix}_mean``, ``{prefix}_max``, ``{prefix}_min``,
        ``{prefix}_latest_reading``, ``{prefix}_trajectory_slope``.
        Patients with no readings in the window (or a missing index_date) get NaN
        across all features for that channel.

    Notes
    -----
    Pure function: no I/O, inputs not mutated.
    """
    # --- Cohort side: id + parsed index date --------------------------------
    cohort = cohort_df[[cohort_patient_col, index_date_col]].copy()
    cohort["_index_date"] = _to_naive_datetime(cohort[index_date_col])

    code_to_channel = {code: channel for channel, code in bp_codes.items()}

    # --- Vitals side: BP rows only, parsed + numeric ------------------------
    vit = vitals_df[vitals_df[code_col].astype(str).isin(code_to_channel)].copy()
    vit["_channel"] = vit[code_col].astype(str).map(code_to_channel)
    vit["_date"] = _to_naive_datetime(vit[date_col])
    vit["_value"] = pd.to_numeric(vit[value_col], errors="coerce")
    vit = vit.dropna(subset=["_date", "_value"])

    # Attach index date and window the readings (strictly before index_date).
    vit = vit.merge(
        cohort[[cohort_patient_col, "_index_date"]],
        left_on=vitals_patient_col,
        right_on=cohort_patient_col,
        how="inner",
    )
    lookback = pd.to_timedelta(lookback_days, unit="D")
    in_window = (
        (vit["_date"] >= vit["_index_date"] - lookback)
        & (vit["_date"] < vit["_index_date"])
    )
    vit = vit[in_window]

    # --- Aggregate per patient ----------------------------------------------
    all_feature_cols = [
        f"{_PREFIX[ch]}_{f}" for ch in bp_codes for f in _FEATURES
    ]

    grouped = {pid: g for pid, g in vit.groupby(vitals_patient_col, sort=False)}

    rows: list[dict[str, float]] = []
    for _, crow in cohort.iterrows():
        pid = crow[cohort_patient_col]
        index_date = crow["_index_date"]
        row: dict[str, float] = {cohort_patient_col: pid}

        if pd.isna(index_date) or pid not in grouped:
            row.update({col: float("nan") for col in all_feature_cols})
            rows.append(row)
            continue

        pgroup = grouped[pid]
        for channel, prefix in _PREFIX.items():
            if channel not in bp_codes:
                continue
            chan_group = pgroup[pgroup["_channel"] == channel]
            row.update(_channel_stats(chan_group, index_date, prefix))
        rows.append(row)

    return pd.DataFrame(rows, columns=[cohort_patient_col] + all_feature_cols)


# Backwards-compatible alias matching the build_features.py spec naming.
calculate_bp_trajectories = compute_bp_trajectory
