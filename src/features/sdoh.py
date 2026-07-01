"""SDOH barrier flag features from the clinical code dictionary.

Produces binary (0/1) indicator columns for social-determinant-of-health barriers,
one per routing-ready grouping defined in ``code_dictionary.yaml`` (composite_flags:),
plus a master ``flag_sdoh_any``. The SNOMED codes are read *from the parsed dictionary*
— this module never hardcodes or guesses clinical codes.

Groupings (auto-derived from composite_flags entries that carry a ``routing_action``
and a list of ``codes``):

    flag_sdoh_isolation_any      social isolation / limited contact
    flag_sdoh_financial_barrier  unemployed / not in labor force
    flag_sdoh_housing_barrier    housing unsatisfactory / homeless
    flag_sdoh_transport_barrier  lack of transport / transport problem
    flag_sdoh_low_education      primary / high-school-only education
    flag_migrant_any             refugee / social migrant
    flag_trauma_exposure         IPV / environmental violence
    flag_sdoh_any                1 if ANY of the above is 1

Data-leakage rule
-----------------
A code only counts toward a flag if its recorded date is **strictly prior** to the
patient's index_date (``date_recorded < index_date``). Anything on or after index_date
is ignored, so no post-index information leaks into the feature.

!! SOURCE-TABLE NOTE !!
These are SNOMED *findings/conditions*. In this Synthea dataset they live in
``conditions.csv`` (date column ``START``), NOT ``observations.csv`` (LOINC labs/vitals,
date column ``DATE``). Pass the conditions table with ``date_col="START"`` to actually
match codes; otherwise every flag will be 0. The parameter is named ``observations_df``
to match the requested signature, but any long [patient, code, date] table works.

Public API
----------
compute_sdoh_flags(cohort_df, observations_df, code_dictionary, ...) -> pandas.DataFrame
"""
from __future__ import annotations

import pandas as pd

# Default master-flag column name.
SDOH_ANY_FLAG = "flag_sdoh_any"


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO date/datetime strings to tz-naive pandas Timestamps (NaT on failure)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def extract_sdoh_barrier_groups(
    code_dictionary: dict,
    barrier_flags: list[str] | None = None,
) -> dict[str, list[str]]:
    """Pull SDOH barrier groupings (name -> list of SNOMED codes) from the parsed dict.

    Reads ``code_dictionary["composite_flags"]``. When ``barrier_flags`` is None, a
    grouping is auto-selected if it has a ``routing_action`` key AND a list-valued
    ``codes`` field — this picks exactly the routing-ready SDOH/psychosocial composites
    (isolation, financial, housing, transport, education, migrant, trauma) and excludes
    the purely clinical composites (mi_any, ckd_any, etc.) which have no routing_action.
    Pass an explicit list of composite-flag keys to override the selection.

    Codes are returned as strings to match the (string) CODE columns in the source data.
    """
    composites = code_dictionary.get("composite_flags", {})
    groups: dict[str, list[str]] = {}
    for key, spec in composites.items():
        if not isinstance(spec, dict):
            continue
        if barrier_flags is not None:
            if key not in barrier_flags:
                continue
        else:
            # Auto-select: routing-ready barriers only.
            if "routing_action" not in spec:
                continue
        codes = spec.get("codes")
        if not isinstance(codes, list):  # skips ckd_stage (dict) / htn_treated (none)
            continue
        groups[key] = [str(c) for c in codes]
    return groups


def compute_sdoh_flags(
    cohort_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    code_dictionary: dict,
    *,
    cohort_patient_col: str = "Id",
    index_date_col: str = "index_date",
    obs_patient_col: str = "PATIENT",
    code_col: str = "CODE",
    date_col: str = "DATE",
    barrier_flags: list[str] | None = None,
    flag_prefix: str = "flag_",
    any_flag_name: str = SDOH_ANY_FLAG,
) -> pd.DataFrame:
    """Build binary SDOH barrier flags per cohort patient, leakage-safe.

    Methodology
    -----------
    1. Read the SDOH barrier groupings (name -> SNOMED codes) from
       ``code_dictionary["composite_flags"]`` via :func:`extract_sdoh_barrier_groups`.
       No codes are hardcoded here.
    2. Restrict ``observations_df`` to rows whose ``code_col`` is one of those codes.
    3. Apply the leakage rule: keep only rows whose ``date_col`` is **strictly prior**
       to the patient's ``index_date`` (``date < index_date``; rows with a missing or
       on/after-index date are dropped).
    4. For each grouping, a patient's flag is 1 if any surviving row carries a code in
       that group, else 0. ``flag_sdoh_any`` is 1 if any individual flag is 1.

    Inputs
    ------
    cohort_df : pandas.DataFrame
        One row per cohort patient. Must contain ``cohort_patient_col`` (patient id)
        and ``index_date_col`` (per-patient index date; string or datetime).
    observations_df : pandas.DataFrame
        Long clinical-history table with ``obs_patient_col``, ``code_col`` and
        ``date_col``. NOTE: SDOH SNOMED codes live in conditions.csv on this dataset —
        pass it with ``date_col="START"`` (see module docstring).
    code_dictionary : dict
        Parsed code_dictionary.yaml (``yaml.safe_load`` output).

    Keyword args remap column names, override the grouping selection (``barrier_flags``),
    and customize flag naming (``flag_prefix``, ``any_flag_name``).

    Returns
    -------
    pandas.DataFrame
        One row per cohort patient, in ``cohort_df`` order. Columns:
        ``cohort_patient_col``, one ``{flag_prefix}{group}`` int column per grouping,
        and ``any_flag_name``. All flag columns are integer 0/1 — never NaN. Patients
        with no qualifying clinical history (or a missing index_date) default to 0
        across every flag.

    Notes
    -----
    Pure function: no I/O, inputs not mutated.
    """
    groups = extract_sdoh_barrier_groups(code_dictionary, barrier_flags)
    flag_names = {group: f"{flag_prefix}{group}" for group in groups}

    # --- Cohort side: id + parsed index date --------------------------------
    cohort = cohort_df[[cohort_patient_col, index_date_col]].copy()
    cohort["_index_date"] = _to_naive_datetime(cohort[index_date_col])

    # Start every patient at 0 for every flag (the no-history / no-leak default).
    out = cohort[[cohort_patient_col]].copy()
    for group in groups:
        out[flag_names[group]] = 0

    all_codes = {code for codes in groups.values() for code in codes}

    # --- Observation side: relevant codes, parsed, leakage-filtered ---------
    if all_codes:
        obs = observations_df[
            observations_df[code_col].astype(str).isin(all_codes)
        ].copy()
        obs["_code"] = obs[code_col].astype(str)
        obs["_date"] = _to_naive_datetime(obs[date_col])
        obs = obs.merge(
            cohort[[cohort_patient_col, "_index_date"]],
            left_on=obs_patient_col,
            right_on=cohort_patient_col,
            how="inner",
            suffixes=("", "_cohort"),
        )
        # Strictly-prior leakage rule. NaT comparisons are False -> dropped.
        obs = obs[obs["_date"] < obs["_index_date"]]

        # For each grouping, mark patients who carry any of its codes.
        out = out.set_index(cohort_patient_col)
        for group, codes in groups.items():
            patients_hit = obs.loc[obs["_code"].isin(codes), obs_patient_col].unique()
            if len(patients_hit):
                out.loc[out.index.intersection(patients_hit), flag_names[group]] = 1
        out = out.reset_index()

    # --- Master any-flag -----------------------------------------------------
    flag_cols = list(flag_names.values())
    if flag_cols:
        out[any_flag_name] = (out[flag_cols].sum(axis=1) > 0).astype(int)
    else:
        out[any_flag_name] = 0

    # Ensure clean integer dtype on every flag column.
    for col in flag_cols + [any_flag_name]:
        out[col] = out[col].astype(int)

    return out[[cohort_patient_col] + flag_cols + [any_flag_name]]


# Backwards-compatible alias matching the build_features.py spec naming.
calculate_sdoh_flags = compute_sdoh_flags
