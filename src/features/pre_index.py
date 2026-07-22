"""pre_index.py — leakage-safe pre-index feature modules for the adherence model.

Every feature here is computed *strictly before* each patient's ``index_date``
(the ``_date < index_date`` rule, identical to sdoh.py / trajectories.py). None
of them can see a fill, encounter, condition, or payer record dated on/after
index, so no post-index information — least of all the PDC label window
``[index_date, index_date + 180d)`` — can leak in.

Feature families (each function is pure: no I/O, inputs not mutated)
-------------------------------------------------------------------
comorbidity       cmb_n_conditions        # distinct condition codes pre-index
regimen           rx_n_active_meds        # distinct meds active *at* index
                  rx_n_meds_started_1y    # distinct meds first filled in prior 365d
prior adherence   adh_prior_n_med_fills   # medication fills in prior 365d (any drug)
                  adh_prior_med_pdc       union days-covered fraction, prior 365d
engagement        engage_n_encounters     # encounters in prior 365d
                  engage_n_bp_readings    # BP readings in prior 1095d
payer churn       payer_n_switches        # payer changes among pre-index transitions
cross-drug        xdrug_n_distinct_meds   # distinct non-AH chronic meds ever pre-index
mechanics         xdrug_refill_gap_mean   # mean same-drug refill gap, days (NaN if none)
                  xdrug_refill_gap_max    # worst same-drug refill gap, days (NaN if none)
                  xdrug_tenure_days       # days from first non-AH rx to index
                  xdrug_n_prescribers     # distinct pre-index encounters (prescriber proxy)
                  xdrug_oop_cost_mean     # mean out-of-pocket per non-AH fill (NaN if none)

Why the antihypertensive-specific pre-index metrics are NOT built
-----------------------------------------------------------------
The cohort is an *incident/new-user* design (index = first antihypertensive
fill after a washout), so by construction **zero** patients have any
antihypertensive fill strictly before index. Any AH-specific pre-index metric
(multi-drug flag, AH refill rate, AH gap mean/prob, AH regimen churn) would be
an all-zero dead column for all 16,205 patients — the silent-dead-feature
failure the project memory warns about — so they are dropped entirely, not
zero-filled.

The valid signal instead comes from **cross-drug** behavior: the ``xdrug_*``
family reads the patient's OTHER pre-index chronic medications (statins,
metformin, levothyroxine, …) as a leakage-safe proxy for how reliably they fill
prescriptions, how long they have been in treatment, how many prescribers they
see, and their out-of-pocket cost friction — all computed strictly before
index. (``rx_*`` / ``adh_*`` already capture general polypharmacy and prior-year
coverage; ``xdrug_*`` adds refill mechanics, tenure, prescriber breadth, and
cost over the full pre-index history.)

Column prefixes (``cmb_``, ``rx_``, ``adh_``, ``engage_``, ``payer_``,
``xdrug_``) are registered in src/models/common.py FEATURE_PREFIXES so they pass
the leakage allowlist guard.

Public API
----------
compute_pre_index_features(cohort_df, conditions_df, meds_df, encounters_df,
                           vitals_df, payer_transitions_df, ...) -> pandas.DataFrame
    One row per cohort patient (cohort order) with all eight feature columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Reuse the union-coverage sweep from the PDC module so "days covered by any med"
# is computed exactly the way the label's PDC is (overlapping fills counted once).
# ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL is imported to *exclude* the index drug
# class from the cross-drug features (they proxy adherence via OTHER meds).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdc import _coverage_and_max_gap, ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL  # noqa: E402

# Default lookback windows (days).
ADHERENCE_LOOKBACK_DAYS = 365    # prior-adherence + encounter engagement window
BP_LOOKBACK_DAYS = 1095          # BP-reading engagement window (matches trajectories.py)

# BP LOINC codes (systolic + diastolic), source: code_dictionary.yaml.
BP_LOINC_CODES = frozenset({"8480-6", "8462-4"})

DEFAULT_DAYS_SUPPLY = 30         # fallback coverage length when a fill has no valid STOP


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO date/datetime strings to tz-naive Timestamps (NaT on failure)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _prep_source(
    df: pd.DataFrame, cohort_ids: set, cohort_idx: pd.Series,
    *, patient_col: str = "PATIENT", date_cols: tuple[str, ...] = (),
    keep_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Slim a raw source table to the cohort, parsing its date columns ONCE.

    Performance matters here: the raw tables are 10-20M rows and are shared by
    several features, so we (1) drop to cohort patients with a cheap ``isin``
    BEFORE any datetime parsing, (2) keep only the needed columns, and (3) parse
    each date column a single time (``START`` -> ``_start`` etc.), attaching the
    patient's parsed index date as ``_idx``. Every downstream feature then works
    on this small, already-parsed frame instead of re-copying/re-parsing millions
    of rows per feature.
    """
    sub = df[df[patient_col].isin(cohort_ids)]
    # Keep the id, the date columns, and any *present* optional columns (a minimal
    # test frame may omit e.g. ENCOUNTER / cost columns).
    present_keep = [c for c in keep_cols if c in df.columns]
    cols = [patient_col, *present_keep, *date_cols]
    sub = sub[[c for c in dict.fromkeys(cols)]].copy()  # de-dup, preserve order
    for col in date_cols:
        sub["_" + col.lower()] = _to_naive_datetime(sub[col])
    sub["_idx"] = sub[patient_col].map(cohort_idx)
    return sub


def _comorbidity_counts(cond: pd.DataFrame) -> pd.Series:
    """cmb_n_conditions: distinct condition codes recorded strictly before index."""
    pre = cond[cond["_start"] < cond["_idx"]]               # strict pre-index
    return pre.groupby("PATIENT")["CODE"].nunique()


def _regimen_counts(
    meds: pd.DataFrame, *, lookback_days: int = ADHERENCE_LOOKBACK_DAYS,
) -> tuple[pd.Series, pd.Series]:
    """rx_n_active_meds and rx_n_meds_started_1y (meds already parsed/slimmed).

    ``rx_n_active_meds`` = distinct medications active *at* index: fill START is
    strictly before index and the fill has not yet ended (STOP missing, or STOP
    after index). STOP is the prescription's planned end, written at fill time
    (before index), so this uses no post-index information.

    ``rx_n_meds_started_1y`` = distinct medications filled in
    ``[index - lookback_days, index)`` — recent medication churn, START-only.
    """
    started_before = meds["_start"] < meds["_idx"]
    active = started_before & (meds["_stop"].isna() | (meds["_stop"] > meds["_idx"]))
    n_active = meds[active].groupby("PATIENT")["_code"].nunique()

    lb = pd.to_timedelta(lookback_days, unit="D")
    recent = started_before & (meds["_start"] >= meds["_idx"] - lb)
    n_recent = meds[recent].groupby("PATIENT")["_code"].nunique()
    return n_active, n_recent


def _prior_adherence(
    meds: pd.DataFrame, *, lookback_days: int = ADHERENCE_LOOKBACK_DAYS,
    default_days_supply: int = DEFAULT_DAYS_SUPPLY,
) -> tuple[pd.Series, pd.Series]:
    """adh_prior_n_med_fills and adh_prior_med_pdc over the pre-index window.

    Behavioral proxy: how engaged was the patient with *any* medication in the
    ``[index - lookback_days, index)`` window before starting BP therapy? PDC is
    the union-coverage fraction (days with >=1 active drug / window length),
    computed with the same overlap-merge sweep as the outcome label so overlapping
    fills are counted once. Strictly pre-index — the window ends at index.

    ``meds`` is the shared, cohort-slimmed frame (columns ``_start``/``_stop``/
    ``_idx``); the per-patient sweep therefore runs over a small filtered set.
    """
    lb = pd.to_timedelta(lookback_days, unit="D")
    supply_end = meds["_start"] + pd.to_timedelta(default_days_supply, unit="D")
    valid_stop = meds["_stop"].notna() & (meds["_stop"] > meds["_start"])
    end = supply_end.where(~valid_stop, meds["_stop"])
    window_start = meds["_idx"] - lb
    # A fill is relevant if it started before index AND its coverage reaches into
    # [window_start, index).
    rel = meds[(meds["_start"] < meds["_idx"]) & (end > window_start)].copy()
    rel["_end"] = end[rel.index]

    n_fills = rel.groupby("PATIENT").size()

    pdc: dict = {}
    for pid, grp in rel.groupby("PATIENT", sort=False):
        idx = grp["_idx"].iloc[0]
        w_start = idx - lb
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for start, e in zip(grp["_start"], grp["_end"]):
            clip_start = max(start, w_start)
            clip_end = min(e, idx)            # clip to [index-lookback, index)
            if clip_start < clip_end:
                intervals.append((clip_start, clip_end))
        covered, _ = _coverage_and_max_gap(intervals, w_start, idx)
        pdc[pid] = min(covered / lookback_days, 1.0)
    return n_fills, pd.Series(pdc, dtype="float64")


def _engagement(
    enc: pd.DataFrame, vit: pd.DataFrame,
    *, encounter_lookback_days: int = ADHERENCE_LOOKBACK_DAYS,
    bp_lookback_days: int = BP_LOOKBACK_DAYS,
) -> tuple[pd.Series, pd.Series]:
    """engage_n_encounters (prior 365d) and engage_n_bp_readings (prior 1095d).

    ``enc`` / ``vit`` are cohort-slimmed, date-parsed frames (``_start`` /
    ``_date`` + ``_idx``); ``vit`` is expected pre-restricted to BP codes.
    """
    enc_lb = pd.to_timedelta(encounter_lookback_days, unit="D")
    e = enc[(enc["_start"] >= enc["_idx"] - enc_lb) & (enc["_start"] < enc["_idx"])]
    n_enc = e.groupby("PATIENT").size()

    bp_lb = pd.to_timedelta(bp_lookback_days, unit="D")
    v = vit[(vit["_date"] >= vit["_idx"] - bp_lb) & (vit["_date"] < vit["_idx"])]
    n_bp = v.groupby("PATIENT").size()
    return n_enc, n_bp


def _payer_switches(pt: pd.DataFrame, *, payer_col: str = "PAYER") -> pd.Series:
    """payer_n_switches: number of times the payer changes across pre-index transitions.

    Synthea emits many payer-membership rows per patient; a *switch* is a
    consecutive pair (ordered by START_DATE) whose PAYER id differs. Only
    transitions beginning strictly before index count. ``pt`` is cohort-slimmed
    with ``_start_date`` / ``_idx`` already parsed.
    """
    pre = pt[pt["_start_date"] < pt["_idx"]].sort_values(["PATIENT", "_start_date"])
    prev = pre.groupby("PATIENT")[payer_col].shift()
    switch = prev.notna() & (pre[payer_col].astype(str) != prev.astype(str))
    return switch.groupby(pre["PATIENT"]).sum()


def _cross_drug_features(
    meds: pd.DataFrame, *, ah_codes: frozenset[str] = ANTIHYPERTENSIVE_RXNORM_PRODUCT_LEVEL,
    has_enc: bool = True, has_cost: bool = True,
    encounter_col: str = "ENCOUNTER", total_cost_col: str = "TOTALCOST",
    payer_cov_col: str = "PAYER_COVERAGE",
) -> pd.DataFrame:
    """Cross-drug refill mechanics / tenure / prescriber / cost, over non-AH fills.

    Built from the patient's *non-antihypertensive* fills dated strictly before
    index (there are no antihypertensive fills pre-index by cohort design, so
    excluding the AH set is belt-and-suspenders). ``meds`` is the shared,
    cohort-slimmed frame. Returns a frame indexed by ``PATIENT`` with the
    available ``xdrug_*`` columns; the caller reindexes and applies defaults.
    """
    m = meds[(meds["_start"] < meds["_idx"]) & (~meds["_code"].isin(ah_codes))]

    feats = pd.DataFrame(index=pd.Index([], name="PATIENT"))
    if m.empty:
        return feats

    grp = m.groupby("PATIENT")
    feats["xdrug_n_distinct_meds"] = grp["_code"].nunique()
    feats["xdrug_tenure_days"] = (grp["_idx"].first() - grp["_start"].min()).dt.days
    if has_enc and encounter_col in m.columns:
        feats["xdrug_n_prescribers"] = grp[encounter_col].nunique()
    if has_cost and total_cost_col in m.columns and payer_cov_col in m.columns:
        oop = pd.to_numeric(m[total_cost_col], errors="coerce") \
            - pd.to_numeric(m[payer_cov_col], errors="coerce")
        feats["xdrug_oop_cost_mean"] = oop.groupby(m["PATIENT"]).mean()

    # Same-drug refill gaps: consecutive fills of the SAME code, per patient.
    # A patient with no drug filled >=2x pre-index has no defined gap -> NaN.
    m_sorted = m.sort_values(["PATIENT", "_code", "_start"])
    prev = m_sorted.groupby(["PATIENT", "_code"])["_start"].shift()
    gap = (m_sorted["_start"] - prev).dt.days
    gap_stats = gap.groupby(m_sorted["PATIENT"]).agg(["mean", "max"])
    feats["xdrug_refill_gap_mean"] = gap_stats["mean"]
    feats["xdrug_refill_gap_max"] = gap_stats["max"]
    return feats


def compute_pre_index_features(
    cohort_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    payer_transitions_df: pd.DataFrame,
    *,
    cohort_patient_col: str = "patient_id",
    index_date_col: str = "index_date",
) -> pd.DataFrame:
    """Build all pre-index features, one leakage-safe row per cohort patient.

    Returns a frame with ``cohort_patient_col`` plus the feature columns
    (``cmb_*``, ``rx_*``, ``adh_*``, ``engage_*``, ``payer_*``, ``xdrug_*``), in
    ``cohort_df`` row order. Count / tenure features default to 0 for a patient
    with no qualifying pre-index history; genuinely-undefined quantities
    (``xdrug_refill_gap_*``, ``xdrug_oop_cost_mean``) default to NaN, which the
    gradient-boosted model consumes natively — the same convention as the
    ``sbp_*``/``dbp_*`` trajectory features.

    Each raw source is filtered to the cohort and date-parsed exactly once (see
    :func:`_prep_source`); the medication frame in particular is prepared a single
    time and shared across the regimen / prior-adherence / cross-drug features.
    """
    # Cohort key + index date, parsed once. cohort_idx maps PATIENT -> index_date.
    key = cohort_df[cohort_patient_col].reset_index(drop=True)
    idx_parsed = _to_naive_datetime(cohort_df[index_date_col]).reset_index(drop=True)
    cohort_idx = pd.Series(idx_parsed.to_numpy(), index=key.to_numpy())
    cohort_ids = set(key)

    # --- Prepare each raw source once (filter to cohort + parse dates) --------
    meds = _prep_source(
        meds_df, cohort_ids, cohort_idx,
        date_cols=("START", "STOP"),
        keep_cols=("CODE", "ENCOUNTER", "TOTALCOST", "PAYER_COVERAGE"),
    )
    meds = meds.dropna(subset=["_start"])
    meds["_code"] = meds["CODE"].astype(str)
    has_enc = "ENCOUNTER" in meds.columns
    has_cost = "TOTALCOST" in meds.columns and "PAYER_COVERAGE" in meds.columns

    cond = _prep_source(conditions_df, cohort_ids, cohort_idx,
                        date_cols=("START",), keep_cols=("CODE",))
    enc = _prep_source(encounters_df, cohort_ids, cohort_idx, date_cols=("START",))
    enc = enc.dropna(subset=["_start"])
    vit_all = _prep_source(vitals_df, cohort_ids, cohort_idx,
                           date_cols=("DATE",), keep_cols=("CODE",))
    vit = vit_all[vit_all["CODE"].astype(str).isin(BP_LOINC_CODES)].dropna(subset=["_date"])
    pay = _prep_source(payer_transitions_df, cohort_ids, cohort_idx,
                       date_cols=("START_DATE",), keep_cols=("PAYER",))
    pay = pay.dropna(subset=["_start_date"])

    # --- Compute features from the slim frames -------------------------------
    n_cond = _comorbidity_counts(cond)
    n_active, n_recent = _regimen_counts(meds)
    n_fills, prior_pdc = _prior_adherence(meds)
    n_enc, n_bp = _engagement(enc, vit)
    n_switch = _payer_switches(pay)
    xdrug = _cross_drug_features(meds, has_enc=has_enc, has_cost=has_cost)

    def _int_col(series: pd.Series) -> pd.Series:
        return key.map(series).fillna(0).astype("int64")

    def _float_col(series: pd.Series, default: float) -> pd.Series:
        mapped = key.map(series).astype("float64")
        return mapped.fillna(default) if default is not None else mapped

    out = pd.DataFrame({cohort_patient_col: key.to_numpy()})
    out["cmb_n_conditions"] = _int_col(n_cond)
    out["rx_n_active_meds"] = _int_col(n_active)
    out["rx_n_meds_started_1y"] = _int_col(n_recent)
    out["adh_prior_n_med_fills"] = _int_col(n_fills)
    out["adh_prior_med_pdc"] = _float_col(prior_pdc, 0.0)
    out["engage_n_encounters"] = _int_col(n_enc)
    out["engage_n_bp_readings"] = _int_col(n_bp)
    out["payer_n_switches"] = _int_col(n_switch)
    # Cross-drug family. Counts / tenure default to 0; genuinely-undefined
    # refill-gap and cost quantities stay NaN (native to the GBDT).
    out["xdrug_n_distinct_meds"] = _int_col(xdrug.get("xdrug_n_distinct_meds", pd.Series(dtype="float64")))
    out["xdrug_tenure_days"] = _int_col(xdrug.get("xdrug_tenure_days", pd.Series(dtype="float64")))
    out["xdrug_n_prescribers"] = _int_col(xdrug.get("xdrug_n_prescribers", pd.Series(dtype="float64")))
    out["xdrug_refill_gap_mean"] = _float_col(xdrug.get("xdrug_refill_gap_mean", pd.Series(dtype="float64")), None)
    out["xdrug_refill_gap_max"] = _float_col(xdrug.get("xdrug_refill_gap_max", pd.Series(dtype="float64")), None)
    out["xdrug_oop_cost_mean"] = _float_col(xdrug.get("xdrug_oop_cost_mean", pd.Series(dtype="float64")), None)
    return out
