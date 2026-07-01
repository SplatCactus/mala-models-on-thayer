"""Discrimination & calibration metrics for BP Cascade RI's survival model.

Implements /docs/eval_metric_spec.md §1 (discrimination) and §2
(calibration). Every function here is generic over model implementation —
it consumes plain arrays (event times, censoring indicators, risk scores /
predicted survival probabilities), not a fitted `lifelines` model object —
so it can be exercised with synthetic fixtures today and wired to the real
CoxPH/AFT model's output later without changing this module (spec
requirement: decouple the harness from the model library).

Every function returns a `MetricResult` (see src/eval/__init__.py) that
always includes a per-stratum breakdown by ethnicity, even though
discrimination/calibration are primarily "is the model good" questions —
this is what lets fairness.py re-use these functions directly for the
model-level parity checks in spec §3a instead of recomputing C-index or
calibration slope a second time.

Status: SCAFFOLD. C-index and calibration-at-horizon are fully implemented
(they're generic statistics with no lifelines-utils gap). Time-windowed AUC
and Brier score are stubbed with TODOs — see the module-level docstring in
the sibling fairness.py... actually see the summary at the bottom of the
PR/response that generated this file: `lifelines.utils` (v0.30.3, confirmed
by inspection) has neither a dynamic/time-dependent AUC estimator nor a
brier_score / integrated_brier_score function, unlike scikit-survival. Both
need either a hand-rolled IPCW estimator or a scoped scikit-survival
dependency — a real decision, not just plumbing, so they are left as TODOs
rather than guessed at.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.eval import (
    AUC_HORIZON_DAYS,
    BRIER_HORIZONS_DAYS,
    BRIER_RELATIVE_IMPROVEMENT_THRESHOLD,
    C_INDEX_MEANINGFUL_THRESHOLD,
    C_INDEX_WEAK_THRESHOLD,
    CALIBRATION_N_BINS,
    CALIBRATION_SLOPE_BAND_GLOBAL,
    ETHNICITY_STRATA,
    MetricResult,
    stratify_and_compute,
)

ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# §1 — Discrimination
# ---------------------------------------------------------------------------


def compute_c_index(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    risk_scores: ArrayLike,
    *,
    ethnicity_labels: Optional[ArrayLike] = None,
) -> MetricResult:
    """Harrell's C-index — primary discrimination metric (spec §1).

    Parameters
    ----------
    event_times : days from index_date to break (or censoring).
    event_observed : 1 if the patient broke persistence (event observed),
        0 if right-censored.
    risk_scores : higher = predicted higher risk of an EARLIER break.
        Sign convention matters and is model-dependent: for a fitted
        `lifelines.CoxPHFitter`, this is the partial hazard /
        `predict_partial_hazard(X)`, which is already "higher = riskier"
        by construction. For a `lifelines` AFT model (e.g.
        `LogNormalAFTFitter`), `predict_expectation(X)` returns an expected
        *time*, which is inversely related to risk — callers must pass
        `-predict_expectation(X)` (or an equivalent monotonic-decreasing
        transform), NOT the raw expected time, or this function will
        silently report the wrong sign of discrimination. TODO(model
        adapter): once CoxPH vs. Log-Normal AFT is finalized, this
        conversion belongs in a single shared adapter function, not
        repeated at every call site.
    ethnicity_labels : optional per-patient ethnicity ("hispanic" /
        "nonhispanic"). If provided, per-stratum C-index is computed via
        the same N>=15 guard as every other metric (used directly by
        fairness.check_c_index_parity — no separate computation there).

    Returns
    -------
    MetricResult with `overall_value` = C-index computed over the full
    cohort, and `per_stratum` populated iff `ethnicity_labels` was given.
    """
    from lifelines.utils import concordance_index  # local import: see TODO below

    # TODO(model library): concordance_index is generic — it only needs a
    # risk score with a documented direction, so this is NOT actually
    # blocked on CoxPH vs. AFT. It IS implicitly blocked on the adapter
    # described above existing so callers pass a correctly-signed score.

    event_times_arr = np.asarray(event_times, dtype=float)
    event_observed_arr = np.asarray(event_observed, dtype=float)
    risk_scores_arr = np.asarray(risk_scores, dtype=float)

    overall_value = float(
        concordance_index(event_times_arr, -risk_scores_arr, event_observed_arr)
    )
    # NOTE: lifelines' concordance_index expects predicted_scores where
    # LOWER = predicted to survive longer (i.e. it's built for the
    # "predicted time" convention). We negate risk_scores_arr here because
    # this function's own documented convention is "higher risk_scores =
    # higher risk of an EARLIER break" — the opposite of predicted time.
    # TODO: cover this sign convention explicitly in a unit test once real
    # model output is wired in, since a silent sign flip is the single
    # easiest way this metric could be wrong without erroring.

    per_stratum = {}
    if ethnicity_labels is not None:
        per_stratum = stratify_and_compute(
            ethnicity_labels,
            {
                "event_times": event_times_arr,
                "event_observed": event_observed_arr,
                "risk_scores": risk_scores_arr,
            },
            compute_fn=lambda event_times, event_observed, risk_scores: float(
                concordance_index(event_times, -risk_scores, event_observed)
            ),
            known_strata=ETHNICITY_STRATA,
        )

    if overall_value >= C_INDEX_MEANINGFUL_THRESHOLD:
        flag: str = "pass"
        description = (
            f"C-index={overall_value:.3f}: model shows meaningful discrimination "
            "beyond chance (0.5)."
        )
    elif overall_value >= C_INDEX_WEAK_THRESHOLD:
        flag = "pass"
        description = (
            f"C-index={overall_value:.3f}: weak but non-trivial discrimination — "
            "expected given synthetic Synthea data has limited behavioral realism."
        )
    else:
        flag = "fail"
        description = (
            f"C-index={overall_value:.3f}: model does not yet discriminate "
            "meaningfully between patients."
        )

    return MetricResult(
        metric_name="C-index",
        overall_value=overall_value,
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=(
            f">= {C_INDEX_MEANINGFUL_THRESHOLD} meaningful, "
            f">= {C_INDEX_WEAK_THRESHOLD} weak-but-nontrivial, else fail (spec §1)."
        ),
    )


def compute_time_windowed_auc(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    risk_scores: ArrayLike,
    *,
    horizon_days: int = AUC_HORIZON_DAYS,
    ethnicity_labels: Optional[ArrayLike] = None,
) -> MetricResult:
    """Time-dependent AUC at a fixed horizon (spec §1, secondary metric).

    "Will this patient have broken persistence within `horizon_days` days:
    yes/no" — reported ALONGSIDE, never instead of, compute_c_index(). Per
    spec §1, this exists purely as a judge-legible landmark; the report
    must present it labeled as "a single fixed-horizon read of the same
    model... the underlying prediction is a continuous break window, not
    this cutoff," never as the headline metric.

    STATUS: STUB. `lifelines.utils` (v0.30.3) does not expose a
    time-dependent/dynamic AUC estimator (no `cumulative_dynamic_auc`
    equivalent) — confirmed by inspecting the module. Real implementation
    requires one of:
      (a) hand-rolling the Uno / Chambless-Diao IPCW time-dependent AUC
          estimator directly against `risk_scores`, `event_times`,
          `event_observed`, and a Kaplan-Meier estimate of the censoring
          distribution (buildable with `lifelines.KaplanMeierFitter` on the
          flipped event indicator), or
      (b) taking a scoped dependency on `scikit-survival`
          (`sksurv.metrics.cumulative_dynamic_auc`) for this one function
          only.
    This is a genuine open decision (not just plumbing) — TODO(model
    library / dependency scope): confirm (a) vs (b) before implementing.

    Returns a MetricResult with `flag="not_applicable"` and
    `overall_value=None` until implemented, so downstream report assembly
    doesn't have to special-case "this metric doesn't exist yet" — it just
    renders the not_applicable state like any other.
    """
    # TODO(blocked): see docstring. Implementing this correctly requires
    # picking (a) hand-rolled IPCW estimator or (b) scikit-survival, which
    # is a dependency-scope decision, not something to guess at here.
    per_stratum = (
        stratify_and_compute(
            ethnicity_labels,
            {"event_times": np.asarray(event_times, dtype=float)},
            compute_fn=lambda **_: None,
            known_strata=ETHNICITY_STRATA,
        )
        if ethnicity_labels is not None
        else {}
    )
    return MetricResult(
        metric_name=f"{horizon_days}-day time-windowed AUC",
        overall_value=None,
        per_stratum=per_stratum,
        flag="not_applicable",
        description=(
            f"{horizon_days}-day AUC not yet implemented — blocked on IPCW "
            "estimator vs. scikit-survival dependency decision (see TODO)."
        ),
        threshold_description="Not yet defined pending implementation decision.",
    )


# ---------------------------------------------------------------------------
# §2 — Calibration
# ---------------------------------------------------------------------------


def compute_brier_score(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    predicted_survival_at_horizon: ArrayLike,
    *,
    horizon_days: int,
    ethnicity_labels: Optional[ArrayLike] = None,
) -> MetricResult:
    """IPCW Brier score at a fixed horizon vs. the no-skill reference (spec §2a).

    Parameters
    ----------
    predicted_survival_at_horizon : model-predicted P(survive past
        horizon_days) per patient — i.e. S_hat_i(horizon_days). For a
        fitted `lifelines` CoxPH or AFT model this is
        `model.predict_survival_function(X, times=[horizon_days])`,
        transposed to one value per patient. TODO(model adapter): this
        extraction is model-object-specific and belongs in the same
        adapter module noted in compute_c_index, not duplicated here.

    "Well calibrated" per spec §2a is defined RELATIVE to the no-skill
    (population Kaplan-Meier) reference at the same horizon:
    Brier(model) must be >= BRIER_RELATIVE_IMPROVEMENT_THRESHOLD (10%)
    lower than Brier(no-skill).

    STATUS: STUB. `lifelines.utils` has no `brier_score` /
    `integrated_brier_score` function (confirmed by inspection — this is
    NOT a function that exists and was just missed). A correct IPCW Brier
    score needs a Kaplan-Meier estimate of the *censoring* distribution
    G(t) (buildable via `lifelines.KaplanMeierFitter` fit on the flipped
    event indicator) to weight each patient's squared-error term. This is
    the same open (a) hand-roll vs. (b) scikit-survival decision as
    compute_time_windowed_auc — TODO(model library / dependency scope).
    """
    # TODO(blocked): see docstring — needs IPCW weighting via a
    # censoring-distribution KM estimate; not implemented here.
    per_stratum = (
        stratify_and_compute(
            ethnicity_labels,
            {"event_times": np.asarray(event_times, dtype=float)},
            compute_fn=lambda **_: None,
            known_strata=ETHNICITY_STRATA,
        )
        if ethnicity_labels is not None
        else {}
    )
    return MetricResult(
        metric_name=f"{horizon_days}-day Brier score",
        overall_value=None,
        per_stratum=per_stratum,
        flag="not_applicable",
        description=(
            f"{horizon_days}-day Brier score not yet implemented — blocked on "
            "IPCW censoring-weight estimator (see TODO)."
        ),
        threshold_description=(
            f">= {BRIER_RELATIVE_IMPROVEMENT_THRESHOLD:.0%} relative improvement "
            "over the no-skill (population Kaplan-Meier) reference (spec §2a)."
        ),
    )


def compute_calibration_at_horizon(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    predicted_risk_at_horizon: ArrayLike,
    *,
    horizon_days: int,
    n_bins: int = CALIBRATION_N_BINS,
    ethnicity_labels: Optional[ArrayLike] = None,
    slope_band: tuple[float, float] = CALIBRATION_SLOPE_BAND_GLOBAL,
) -> MetricResult:
    """Quintile-binned calibration plot + slope at a fixed horizon (spec §2b).

    Parameters
    ----------
    predicted_risk_at_horizon : model-predicted P(break by horizon_days)
        per patient — i.e. 1 - S_hat_i(horizon_days). Same extraction
        caveat as compute_brier_score's `predicted_survival_at_horizon`
        (this is just its complement).
    n_bins : defaults to 5 (spec §2 rationale: ~262-patient cohort makes
        10 bins too sparse for a stable per-bin observed rate).
    slope_band : acceptance band for the fitted slope of observed vs.
        predicted bin rates (1.0 = perfect calibration). Global default is
        [0.8, 1.2] (spec §2b); fairness.py passes the widened
        CALIBRATION_SLOPE_BAND_STRATUM band when calling this per-stratum
        for the parity check (spec §3a / Assumption A3), since
        within-stratum bins are smaller and noisier.

    Bin assignment and each bin's observed rate:
      - Patients are bucketed into `n_bins` quantile bins by
        `predicted_risk_at_horizon`.
      - Each bin's OBSERVED event rate at `horizon_days` is estimated via
        Kaplan-Meier within that bin (1 - S_hat_bin(horizon_days)), not a
        naive event-count ratio, so censored patients are handled
        correctly rather than dropped or miscounted as non-events.

    STATUS: mostly implementable now (Kaplan-Meier-per-bin is generic, not
    model-library-blocked) — TODO items below are narrow, not structural.
    """
    from lifelines import KaplanMeierFitter  # local import: see TODO below

    event_times_arr = np.asarray(event_times, dtype=float)
    event_observed_arr = np.asarray(event_observed, dtype=float)
    predicted_risk_arr = np.asarray(predicted_risk_at_horizon, dtype=float)

    def _bin_calibration_slope(
        event_times: np.ndarray, event_observed: np.ndarray, predicted_risk: np.ndarray
    ) -> Optional[float]:
        n = len(predicted_risk)
        if n < n_bins:
            # Not enough patients to form n_bins non-degenerate quantile
            # bins. TODO: decide whether to fall back to fewer bins
            # automatically or report insufficient_sample-style None; for
            # now, conservatively return None rather than a bin count
            # smaller than the spec's stated design.
            return None
        bin_edges = np.quantile(predicted_risk, np.linspace(0, 1, n_bins + 1))
        bin_idx = np.clip(
            np.digitize(predicted_risk, bin_edges[1:-1], right=True), 0, n_bins - 1
        )
        predicted_means = []
        observed_rates = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            predicted_means.append(float(predicted_risk[mask].mean()))
            kmf = KaplanMeierFitter()
            kmf.fit(event_times[mask], event_observed=event_observed[mask])
            # TODO: confirm horizon_days falls within the bin's observed
            # follow-up range; extrapolating a KM curve past the last
            # observed time is a known failure mode worth guarding
            # explicitly once this runs against real data.
            survival_at_horizon = float(
                kmf.survival_function_at_times(horizon_days).iloc[0]
            )
            observed_rates.append(1.0 - survival_at_horizon)
        if len(predicted_means) < 2:
            return None
        # Least-squares slope of observed vs. predicted bin rates.
        slope, _intercept = np.polyfit(predicted_means, observed_rates, deg=1)
        return float(slope)

    overall_value = _bin_calibration_slope(
        event_times_arr, event_observed_arr, predicted_risk_arr
    )

    per_stratum = {}
    if ethnicity_labels is not None:
        per_stratum = stratify_and_compute(
            ethnicity_labels,
            {
                "event_times": event_times_arr,
                "event_observed": event_observed_arr,
                "predicted_risk": predicted_risk_arr,
            },
            compute_fn=lambda event_times, event_observed, predicted_risk: (
                _bin_calibration_slope(event_times, event_observed, predicted_risk)
            ),
            known_strata=ETHNICITY_STRATA,
        )

    if overall_value is None:
        flag: str = "insufficient_sample"
        description = (
            f"{horizon_days}-day calibration slope: insufficient patients to form "
            f"{n_bins} stable bins."
        )
    elif slope_band[0] <= overall_value <= slope_band[1]:
        flag = "pass"
        description = (
            f"{horizon_days}-day calibration slope={overall_value:.2f}: within "
            f"acceptance band {slope_band}."
        )
    else:
        flag = "fail"
        description = (
            f"{horizon_days}-day calibration slope={overall_value:.2f}: outside "
            f"acceptance band {slope_band}."
        )

    return MetricResult(
        metric_name=f"{horizon_days}-day calibration slope",
        overall_value=overall_value,
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=f"slope within {slope_band} (spec §2b).",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_calibration_report(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    risk_scores: ArrayLike,
    predicted_survival_by_horizon: dict[int, ArrayLike],
    *,
    ethnicity_labels: Optional[ArrayLike] = None,
) -> dict[str, MetricResult | list[MetricResult]]:
    """Run every calibration/discrimination metric and assemble one report.

    Parameters
    ----------
    predicted_survival_by_horizon : {horizon_days: predicted_survival_array}
        for each horizon in BRIER_HORIZONS_DAYS (30/90/180). Kept as an
        explicit mapping rather than re-deriving inside this function
        because the extraction from a fitted model is horizon-specific and
        belongs in the model adapter (see TODOs above), not here.

    Returns
    -------
    dict with keys:
      "c_index": MetricResult
      "time_windowed_auc": MetricResult (currently not_applicable — stub)
      "brier_scores": list[MetricResult], one per BRIER_HORIZONS_DAYS
          (currently all not_applicable — stub)
      "calibration_slopes": list[MetricResult], one per BRIER_HORIZONS_DAYS
    This is the shape src/eval's future report generator consumes — do not
    change these keys without updating that consumer too.
    """
    c_index_result = compute_c_index(
        event_times, event_observed, risk_scores, ethnicity_labels=ethnicity_labels
    )
    auc_result = compute_time_windowed_auc(
        event_times, event_observed, risk_scores, ethnicity_labels=ethnicity_labels
    )

    brier_results = []
    calibration_results = []
    for horizon in BRIER_HORIZONS_DAYS:
        predicted_survival = predicted_survival_by_horizon.get(horizon)
        if predicted_survival is None:
            # TODO: decide whether a missing horizon should hard-error once
            # this is wired to a real model, rather than silently skipping.
            continue
        brier_results.append(
            compute_brier_score(
                event_times,
                event_observed,
                predicted_survival,
                horizon_days=horizon,
                ethnicity_labels=ethnicity_labels,
            )
        )
        predicted_risk = 1.0 - np.asarray(predicted_survival, dtype=float)
        calibration_results.append(
            compute_calibration_at_horizon(
                event_times,
                event_observed,
                predicted_risk,
                horizon_days=horizon,
                ethnicity_labels=ethnicity_labels,
            )
        )

    return {
        "c_index": c_index_result,
        "time_windowed_auc": auc_result,
        "brier_scores": brier_results,
        "calibration_slopes": calibration_results,
    }
