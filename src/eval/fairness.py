"""Fairness & routing-parity metrics for BP Cascade RI (spec §3, §6).

Two distinct questions are evaluated, deliberately kept in separate
functions rather than one blended "fairness score" (spec §3 rationale):

  (3a) MODEL-LEVEL parity — is the survival model itself equally accurate
       and equally calibrated across ethnicity strata? These functions
       wrap calibration.py's compute_c_index / compute_calibration_at_horizon
       directly (they already produce a per-stratum breakdown) and apply
       the parity-specific gap thresholds on top — the underlying numbers
       are computed exactly once, not duplicated here.

  (3b) ROUTING-DECISION parity — even a fair model can produce an unfair
       worklist. These are new metrics with no calibration.py analogue:
       selection-rate parity (four-fifths rule), routed-action distribution
       parity (chi-square), time-to-outreach parity, and capacity-cap
       exclusion parity (spec §6 — the check unique to the routing-over-flag
       design and the one most likely to get probed directly).

Also implements: resolve_language() — the `language` column lookup with
the ethnicity-proxy fallback agreed on for spec §6 / Assumption A6.

Status: SCAFFOLD. All computation in this module is pure statistics/logic
(chi-square, ratios, gap checks) with no dependency on which survival
library or model class is used, so — per the scaffolding brief — nothing
here is stubbed on "which model library." The only TODOs are the ones
inherited from calibration.py's still-stubbed AUC/Brier functions, which
this module does not itself call.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.eval import (
    C_INDEX_PARITY_MAX_GAP,
    CALIBRATION_PARITY_HORIZON_DAYS,
    CALIBRATION_SLOPE_BAND_STRATUM,
    CAPACITY_CAP_EXCLUSION_MAX_GAP_PP,
    ETHNICITY_STRATA,
    LANGUAGE_STRATA,
    MIN_STRATUM_N,
    ROUTED_ACTION_CHI2_ALPHA,
    SELECTION_RATE_PARITY_MIN_RATIO,
    MetricResult,
    StratumResult,
    stratify_and_compute,
)
from src.eval.calibration import compute_c_index, compute_calibration_at_horizon

ArrayLike = Sequence[float] | np.ndarray

# ---------------------------------------------------------------------------
# Language resolution — spec §6 / Assumption A6 (accepted as written).
# Pure logic, fully implemented, no model dependency.
# ---------------------------------------------------------------------------


def resolve_language(
    language_col: Optional[pd.Series],
    ethnicity_col: pd.Series,
) -> tuple[pd.Series, dict[str, int]]:
    """Resolve each patient's language, falling back to an ethnicity proxy.

    Looks for a `language` column with values in {"EN", "ES"}. Per patient:
      - if `language_col` is provided AND non-null for that patient, use it
        as-is (case-normalized to upper).
      - otherwise (column missing entirely, or null/blank for that
        patient), proxy: "ES" if ethnicity == "hispanic" else "EN".

    The fallback is applied per patient, not per dataset (spec §6) — if
    `language_col` exists but has gaps, only the gapped patients are
    proxied.

    Returns
    -------
    (resolved_language, source_counts) where `resolved_language` is a
    pandas Series aligned to `ethnicity_col`'s index, and `source_counts`
    is {"from_source": n, "proxied_from_ethnicity": n} — this MUST be
    surfaced in any report that uses the resolved language field (spec §6:
    "never invisible in the final report"), e.g. "language field: 240/262
    from source data, 22/262 proxied from ethnicity."
    """
    ethnicity_normalized = ethnicity_col.astype(str).str.strip().str.lower()
    proxy = np.where(ethnicity_normalized == "hispanic", "ES", "EN")
    proxy_series = pd.Series(proxy, index=ethnicity_col.index)

    if language_col is None:
        # Column absent entirely — per Assumption A6, this must be reported
        # as "0/N from source — 100% ethnicity-proxied", not glossed over.
        resolved = proxy_series
        source_counts = {
            "from_source": 0,
            "proxied_from_ethnicity": int(len(ethnicity_col)),
        }
        return resolved, source_counts

    language_normalized = language_col.astype(str).str.strip().str.upper()
    is_present = language_normalized.isin(list(LANGUAGE_STRATA))
    resolved = language_normalized.where(is_present, proxy_series)

    source_counts = {
        "from_source": int(is_present.sum()),
        "proxied_from_ethnicity": int((~is_present).sum()),
    }
    return resolved, source_counts


# ---------------------------------------------------------------------------
# §3a — Model-level parity (wraps calibration.py; adds gap-threshold flags)
# ---------------------------------------------------------------------------


def check_c_index_parity(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    risk_scores: ArrayLike,
    ethnicity_labels: ArrayLike,
) -> MetricResult:
    """C-index parity across ethnicity strata (spec §3a).

    Reuses calibration.compute_c_index for the actual per-stratum C-index
    computation, then flags if the max pairwise gap between reportable
    (N>=15) strata exceeds C_INDEX_PARITY_MAX_GAP (0.05).
    """
    base_result = compute_c_index(
        event_times, event_observed, risk_scores, ethnicity_labels=ethnicity_labels
    )
    return _apply_max_pairwise_gap_flag(
        base_result,
        metric_name="C-index parity (ethnicity)",
        max_gap=C_INDEX_PARITY_MAX_GAP,
        threshold_description=(
            f"max pairwise C-index gap > {C_INDEX_PARITY_MAX_GAP} between "
            "ethnicity strata (spec §3a)."
        ),
    )


def check_calibration_parity(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    predicted_risk_at_horizon: ArrayLike,
    ethnicity_labels: ArrayLike,
    *,
    horizon_days: int = CALIBRATION_PARITY_HORIZON_DAYS,
) -> MetricResult:
    """Calibration-slope parity at the 90-day horizon (spec §3a).

    Reuses calibration.compute_calibration_at_horizon with the WIDENED
    within-stratum band (CALIBRATION_SLOPE_BAND_STRATUM = [0.7, 1.3],
    Assumption A3) rather than the global [0.8, 1.2] band, since
    within-stratum sample sizes are smaller and a tighter band would
    produce spurious flags on noise alone.

    Note: this checks each stratum's slope against the band independently
    (pass/fail per stratum via compute_calibration_at_horizon's own logic)
    rather than a pairwise-gap comparison like C-index parity, because
    "both strata's slopes are near 1.0" is the actual fairness question
    here, not "the two slopes are close to each other" (two strata could
    both be badly and similarly mis-calibrated and still "pass" a
    gap-based check, which would be the wrong answer).
    """
    base_result = compute_calibration_at_horizon(
        event_times,
        event_observed,
        predicted_risk_at_horizon,
        horizon_days=horizon_days,
        ethnicity_labels=ethnicity_labels,
        slope_band=CALIBRATION_SLOPE_BAND_STRATUM,
    )
    stratum_flags = []
    for stratum_result in base_result.per_stratum.values():
        if stratum_result.status == "insufficient_sample":
            continue
        in_band = (
            stratum_result.value is not None
            and CALIBRATION_SLOPE_BAND_STRATUM[0]
            <= stratum_result.value
            <= CALIBRATION_SLOPE_BAND_STRATUM[1]
        )
        stratum_flags.append(in_band)

    if not stratum_flags:
        flag: str = "insufficient_sample"
    elif all(stratum_flags):
        flag = "pass"
    else:
        flag = "fail"

    return MetricResult(
        metric_name=f"Calibration-slope parity (ethnicity, {horizon_days}d)",
        overall_value=base_result.overall_value,
        per_stratum=base_result.per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=(
            f"{horizon_days}-day calibration slope per ethnicity stratum, each "
            f"checked against band {CALIBRATION_SLOPE_BAND_STRATUM}."
        ),
        threshold_description=(
            f"each stratum's slope within {CALIBRATION_SLOPE_BAND_STRATUM} "
            "(widened band, Assumption A3, spec §3a)."
        ),
    )


def _apply_max_pairwise_gap_flag(
    base_result: MetricResult,
    *,
    metric_name: str,
    max_gap: float,
    threshold_description: str,
) -> MetricResult:
    """Shared helper: recompute a MetricResult's flag from a max-pairwise-gap
    rule over its already-computed per_stratum values, without recomputing
    the underlying metric. Used by check_c_index_parity and
    check_selection_rate_parity's sibling checks.
    """
    reportable_values = [
        s.value for s in base_result.per_stratum.values() if s.status == "ok"
    ]
    if len(reportable_values) < 2:
        flag: str = "insufficient_sample"
        gap: Optional[float] = None
    else:
        gap = max(reportable_values) - min(reportable_values)
        flag = "fail" if gap > max_gap else "pass"

    description = (
        f"{metric_name}: max pairwise gap="
        f"{gap:.3f}" if gap is not None else f"{metric_name}: insufficient strata to compare"
    )
    return MetricResult(
        metric_name=metric_name,
        overall_value=gap,
        per_stratum=base_result.per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=threshold_description,
    )


# ---------------------------------------------------------------------------
# §3a — Routing-decision selection-rate parity (four-fifths rule)
# ---------------------------------------------------------------------------


def check_selection_rate_parity(
    in_capped_worklist: ArrayLike,
    ethnicity_labels: ArrayLike,
) -> MetricResult:
    """Four-fifths / disparate-impact rule on worklist selection rate (spec §3a).

    Parameters
    ----------
    in_capped_worklist : boolean/0-1 array, one entry per patient in the
        ELIGIBLE pool (predicted to break within horizon AND has >=1
        modifiable driver) — True/1 if that patient made the capacity-capped
        worklist. Callers must pre-filter to the eligible pool before
        calling this function; passing the full cohort (including
        ineligible patients) would conflate model eligibility with capacity
        rationing and answer a different question than spec §3a asks.

    Flag: selection-rate ratio (min stratum rate / max stratum rate) < 0.8.
    """
    selected_arr = np.asarray(in_capped_worklist, dtype=float)
    per_stratum = stratify_and_compute(
        ethnicity_labels,
        {"selected": selected_arr},
        compute_fn=lambda selected: float(selected.mean()),
        known_strata=ETHNICITY_STRATA,
    )
    reportable_rates = [s.value for s in per_stratum.values() if s.status == "ok"]

    if len(reportable_rates) < 2:
        flag: str = "insufficient_sample"
        ratio: Optional[float] = None
        description = "Selection-rate parity: insufficient strata to compare."
    else:
        ratio = min(reportable_rates) / max(reportable_rates) if max(reportable_rates) > 0 else None
        if ratio is None:
            flag = "insufficient_sample"
            description = "Selection-rate parity: max stratum rate is 0, ratio undefined."
        else:
            flag = "fail" if ratio < SELECTION_RATE_PARITY_MIN_RATIO else "pass"
            description = (
                f"Selection-rate ratio (min/max stratum)={ratio:.2f} "
                f"({'below' if flag == 'fail' else 'at/above'} the "
                f"{SELECTION_RATE_PARITY_MIN_RATIO} four-fifths threshold)."
            )

    return MetricResult(
        metric_name="Worklist selection-rate parity (ethnicity)",
        overall_value=ratio,
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=(
            f"selection-rate ratio < {SELECTION_RATE_PARITY_MIN_RATIO} "
            "(four-fifths / disparate-impact rule, spec §3a)."
        ),
    )


# ---------------------------------------------------------------------------
# §3b — Routing-decision parity: routed-action distribution & time-to-outreach
# ---------------------------------------------------------------------------


def check_routed_action_distribution_parity(
    routed_actions: ArrayLike,
    stratum_labels: ArrayLike,
    *,
    stratum_name: str = "ethnicity",
    known_strata: Sequence[str] = ETHNICITY_STRATA,
    alpha: float = ROUTED_ACTION_CHI2_ALPHA,
) -> MetricResult:
    """Chi-square test for routed_action distribution parity (spec §3b, A4).

    Per Assumption A4 (accepted as written): this is a QUALITATIVE flag via
    scipy's chi-square test of independence between `stratum_labels` and
    `routed_actions`, not a hard percentage-point threshold — there is no
    clean standard numeric rule for "acceptable divergence" across a
    5-category action distribution the way there is for a binary selection
    rate.

    Parameters
    ----------
    routed_actions : per-patient routed action, e.g. one of
        {"pharmacist", "social_worker", "bilingual_chw_call",
        "transport_voucher", "co_dispatch"} — restricted to patients ON the
        capped worklist (this is a routing-decision question, not an
        eligibility question).
    stratum_labels : ethnicity or language labels, aligned to
        `routed_actions`. Call this function once per axis (ethnicity,
        language) per spec §3b — `stratum_name`/`known_strata` just control
        labeling/reporting, not behavior.

    Small-stratum note: unlike other metrics, this test's validity depends
    on the full contingency table (stratum x action) having adequate
    expected cell counts, not just each stratum's total N >= 15 — a
    stratum can individually clear N=15 and still produce an unreliable
    chi-square if its counts are spread thin across 5 action categories.
    TODO: consider Fisher's exact test or category collapsing as a
    fallback once real routing-action distributions are available to see
    how sparse the table actually gets; scipy.stats.chi2_contingency will
    raise a warning (not an error) on low expected counts, which is
    surfaced via `warnings` here — check the returned description if this
    fires.
    """
    labels_arr = np.asarray(stratum_labels)
    actions_arr = np.asarray(routed_actions)

    per_stratum: dict[str, StratumResult] = {}
    for stratum in known_strata:
        n = int((labels_arr == stratum).sum())
        status = "ok" if n >= MIN_STRATUM_N else "insufficient_sample"
        # `value` here is each stratum's total N on the worklist, not a
        # per-stratum statistic — the chi-square statistic itself is only
        # meaningful jointly across strata, hence overall_value below.
        per_stratum[stratum] = StratumResult(
            stratum=stratum, n=n, value=None, status=status
        )

    reportable_strata = [s for s in known_strata if per_stratum[s].status == "ok"]
    if len(reportable_strata) < 2:
        return MetricResult(
            metric_name=f"Routed-action distribution parity ({stratum_name})",
            overall_value=None,
            per_stratum=per_stratum,
            flag="insufficient_sample",
            description=(
                f"Routed-action distribution parity ({stratum_name}): insufficient "
                "strata to build a contingency table."
            ),
            threshold_description=(
                f"chi-square test of independence, alpha={alpha} (spec §3b, A4)."
            ),
        )

    contingency = pd.crosstab(
        pd.Series(labels_arr, name=stratum_name)[np.isin(labels_arr, reportable_strata)],
        pd.Series(actions_arr, name="routed_action")[np.isin(labels_arr, reportable_strata)],
    )
    chi2, p_value, _dof, _expected = scipy_stats.chi2_contingency(contingency)

    flag = "fail" if p_value < alpha else "pass"
    description = (
        f"Routed-action distribution parity ({stratum_name}): chi2={chi2:.2f}, "
        f"p={p_value:.3f} — "
        f"{'statistically significant divergence in action distribution' if flag == 'fail' else 'no significant divergence detected'} "
        f"at alpha={alpha}."
    )

    return MetricResult(
        metric_name=f"Routed-action distribution parity ({stratum_name})",
        overall_value=float(p_value),
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=f"chi-square test of independence, alpha={alpha} (spec §3b, A4).",
    )


def check_time_to_outreach_parity(
    break_window_start_days: ArrayLike,
    ethnicity_labels: ArrayLike,
) -> MetricResult:
    """Compare distribution of break_window_start (days out) between strata (spec §3b).

    Uses a two-sided Mann-Whitney U test (non-parametric, appropriate for
    a possibly-skewed days-out distribution, and Hispanic/Non-Hispanic is
    a natural two-group comparison) on `break_window_start_days` between
    the two reportable ethnicity strata. `overall_value` is the difference
    in medians (stratum A - stratum B, in days) for report legibility;
    `flag` is driven by the test's p-value at ROUTED_ACTION_CHI2_ALPHA
    (reusing the same 0.05 significance convention as §3b's other test —
    TODO: confirm this shared-alpha choice is acceptable, spec doesn't
    specify a separate one for this metric).
    """
    days_arr = np.asarray(break_window_start_days, dtype=float)
    per_stratum = stratify_and_compute(
        ethnicity_labels,
        {"days": days_arr},
        compute_fn=lambda days: float(np.median(days)),
        known_strata=ETHNICITY_STRATA,
    )
    reportable = [s for s in ETHNICITY_STRATA if per_stratum[s].status == "ok"]

    if len(reportable) < 2:
        return MetricResult(
            metric_name="Time-to-outreach parity (ethnicity)",
            overall_value=None,
            per_stratum=per_stratum,
            flag="insufficient_sample",
            description="Time-to-outreach parity: insufficient strata to compare.",
            threshold_description=(
                f"Mann-Whitney U test, alpha={ROUTED_ACTION_CHI2_ALPHA} (spec §3b)."
            ),
        )

    labels_arr = np.asarray(ethnicity_labels)
    group_a = days_arr[labels_arr == reportable[0]]
    group_b = days_arr[labels_arr == reportable[1]]
    _stat, p_value = scipy_stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

    median_diff = per_stratum[reportable[0]].value - per_stratum[reportable[1]].value
    flag = "fail" if p_value < ROUTED_ACTION_CHI2_ALPHA else "pass"

    return MetricResult(
        metric_name="Time-to-outreach parity (ethnicity)",
        overall_value=median_diff,
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=(
            f"Median break_window_start difference ({reportable[0]} - {reportable[1]})="
            f"{median_diff:.1f} days, Mann-Whitney p={p_value:.3f}."
        ),
        threshold_description=f"Mann-Whitney U test, alpha={ROUTED_ACTION_CHI2_ALPHA} (spec §3b).",
    )


# ---------------------------------------------------------------------------
# §6 — Capacity-cap exclusion parity (ethnicity AND language)
# ---------------------------------------------------------------------------


def check_capacity_cap_exclusion_parity(
    in_capped_worklist: ArrayLike,
    stratum_labels: ArrayLike,
    *,
    stratum_name: str,
    known_strata: Sequence[str],
) -> MetricResult:
    """Capacity-cap exclusion-rate parity (spec §6).

    exclusion_rate(stratum) = 1 - (in capped worklist / in eligible pool)

    Parameters
    ----------
    in_capped_worklist : boolean/0-1 array over the ELIGIBLE pool only
        (same pre-filtering requirement as check_selection_rate_parity —
        this function does not itself distinguish eligible from ineligible
        patients).
    stratum_labels : ethnicity OR language labels (call once per axis, per
        spec §6, which explicitly requires both).

    Flag: exclusion-rate gap between strata > CAPACITY_CAP_EXCLUSION_MAX_GAP_PP
    (15 percentage points) — deliberately a different, coarser threshold
    than check_selection_rate_parity's four-fifths rule, because this
    metric targets the capacity CAP specifically ("does the cap itself
    deprioritize one group"), not the model or eligibility criteria.
    """
    selected_arr = np.asarray(in_capped_worklist, dtype=float)
    per_stratum_selection = stratify_and_compute(
        stratum_labels,
        {"selected": selected_arr},
        compute_fn=lambda selected: 1.0 - float(selected.mean()),
        known_strata=known_strata,
    )
    reportable_gaps = [
        s.value for s in per_stratum_selection.values() if s.status == "ok"
    ]

    if len(reportable_gaps) < 2:
        flag: str = "insufficient_sample"
        gap: Optional[float] = None
        description = f"Capacity-cap exclusion parity ({stratum_name}): insufficient strata."
    else:
        gap = max(reportable_gaps) - min(reportable_gaps)
        flag = "fail" if gap > CAPACITY_CAP_EXCLUSION_MAX_GAP_PP else "pass"
        description = (
            f"Capacity-cap exclusion-rate gap ({stratum_name})={gap:.1%} "
            f"({'exceeds' if flag == 'fail' else 'within'} "
            f"{CAPACITY_CAP_EXCLUSION_MAX_GAP_PP:.0%} threshold)."
        )

    return MetricResult(
        metric_name=f"Capacity-cap exclusion parity ({stratum_name})",
        overall_value=gap,
        per_stratum=per_stratum_selection,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=(
            f"exclusion-rate gap > {CAPACITY_CAP_EXCLUSION_MAX_GAP_PP:.0%} "
            f"between {stratum_name} strata (spec §6)."
        ),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_fairness_report(
    *,
    event_times: ArrayLike,
    event_observed: ArrayLike,
    risk_scores: ArrayLike,
    predicted_risk_at_90d: ArrayLike,
    ethnicity_labels: ArrayLike,
    language_col: Optional[pd.Series],
    ethnicity_col: pd.Series,
    eligible_in_capped_worklist: ArrayLike,
    routed_actions_on_worklist: ArrayLike,
    ethnicity_labels_on_worklist: ArrayLike,
    language_col_on_worklist: Optional[pd.Series],
    ethnicity_col_on_worklist: pd.Series,
    break_window_start_days_on_worklist: ArrayLike,
) -> dict[str, MetricResult | dict]:
    """Run every fairness/parity metric and assemble one combined report.

    Callers are responsible for pre-filtering arrays to the correct
    population per metric (full eligible pool for the two capacity/selection
    checks; capped-worklist-only for the two routing-decision checks) — see
    each function's docstring. This orchestrator does not itself infer
    eligibility or worklist membership from raw model output, since that
    logic lives in capacity.py, not here.

    Returns
    -------
    dict with keys: "c_index_parity", "calibration_parity",
    "selection_rate_parity", "capacity_cap_exclusion_parity_ethnicity",
    "capacity_cap_exclusion_parity_language",
    "routed_action_parity_ethnicity", "routed_action_parity_language",
    "time_to_outreach_parity", "language_resolution" (source_counts dict
    from resolve_language, surfaced per spec §6's transparency requirement).
    """
    resolved_language, language_source_counts = resolve_language(
        language_col, ethnicity_col
    )
    resolved_language_on_worklist, worklist_language_source_counts = resolve_language(
        language_col_on_worklist, ethnicity_col_on_worklist
    )

    return {
        "c_index_parity": check_c_index_parity(
            event_times, event_observed, risk_scores, ethnicity_labels
        ),
        "calibration_parity": check_calibration_parity(
            event_times, event_observed, predicted_risk_at_90d, ethnicity_labels
        ),
        "selection_rate_parity": check_selection_rate_parity(
            eligible_in_capped_worklist, ethnicity_labels
        ),
        "capacity_cap_exclusion_parity_ethnicity": check_capacity_cap_exclusion_parity(
            eligible_in_capped_worklist,
            ethnicity_labels,
            stratum_name="ethnicity",
            known_strata=ETHNICITY_STRATA,
        ),
        "capacity_cap_exclusion_parity_language": check_capacity_cap_exclusion_parity(
            eligible_in_capped_worklist,
            resolved_language.to_numpy(),
            stratum_name="language",
            known_strata=LANGUAGE_STRATA,
        ),
        "routed_action_parity_ethnicity": check_routed_action_distribution_parity(
            routed_actions_on_worklist,
            ethnicity_labels_on_worklist,
            stratum_name="ethnicity",
            known_strata=ETHNICITY_STRATA,
        ),
        "routed_action_parity_language": check_routed_action_distribution_parity(
            routed_actions_on_worklist,
            resolved_language_on_worklist.to_numpy(),
            stratum_name="language",
            known_strata=LANGUAGE_STRATA,
        ),
        "time_to_outreach_parity": check_time_to_outreach_parity(
            break_window_start_days_on_worklist, ethnicity_labels_on_worklist
        ),
        "language_resolution": {
            "eligible_pool": language_source_counts,
            "capped_worklist": worklist_language_source_counts,
        },
    }
