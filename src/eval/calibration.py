"""Discrimination & calibration metrics for BP Cascade RI's survival model.

Implements /docs/eval_metric_spec.md §1 (discrimination) and §2
(calibration). Every function here is generic over model implementation —
it consumes plain arrays (event times, censoring indicators, risk scores /
predicted survival probabilities), not a fitted model object — so it can
be wired to any survival model's output without changing this module's
core functions (spec requirement: decouple the harness from the model
library).

REFACTOR DECISION (delegate to scikit-survival, drop hand-rolled IPCW):
  `src/models/survival.py` (Umar's model) already takes a hard
  `scikit-survival` dependency (`RandomSurvivalForest`,
  `concordance_index_censored`, `cumulative_dynamic_auc`) to fit and
  evaluate the model itself. Since that dependency is unavoidable either
  way, this module now DELEGATES its discrimination/Brier metrics
  straight to `sksurv.metrics` instead of maintaining a second, hand-rolled
  IPCW implementation that had to be kept numerically consistent with it.
  This removes ~150 lines of custom censoring-weight math
  (`_fit_censoring_km`, `_censoring_survival_at`, `_ipcw_brier_at_horizon`,
  `_uno_time_dependent_auc`) that a prior revision hand-rolled and
  cross-validated against `sksurv` — see git history for that
  cross-validation (our hand-rolled Uno AUC matched
  `cumulative_dynamic_auc` to the decimal on the real model and real
  test-split data), which is what made this refactor safe to do with
  confidence rather than a guess.

  Concretely:
    - `compute_c_index` now calls `sksurv.metrics.concordance_index_censored`
      directly (this ALSO simplifies the sign convention: unlike lifelines'
      `concordance_index`, sksurv's expects "higher risk score = higher
      risk" — this module's own documented convention — so the negation
      hand-rolled IPCW code used to need is gone too).
    - `compute_time_windowed_auc` now calls
      `sksurv.metrics.cumulative_dynamic_auc` directly.
    - `compute_brier_score` now calls `sksurv.metrics.brier_score` directly.
    - `compute_calibration_at_horizon` (spec §2b's quintile-bin/slope
      metric) is UNCHANGED and still uses `lifelines.KaplanMeierFitter` —
      sksurv has no equivalent single-call "survival probability at an
      arbitrary query time, per bin" convenience, and hand-rolling that
      one piece is a few lines against a stable, well-tested KM
      implementation, not a second copy of IPCW censoring-weight math.
      This is the one remaining, deliberate `lifelines` dependency in this
      module — not an oversight.

  API consequence: `sksurv.metrics.brier_score` / `cumulative_dynamic_auc`
  need a `survival_train` structured array (from the model's TRAINING
  split) to estimate the censoring distribution — the same convention
  `src/models/survival.py`'s own `evaluate_time_dependent_auc` already
  uses (it takes `y_train` for exactly this reason). `compute_brier_score`
  and `compute_time_windowed_auc` below therefore now require
  `event_times_train` / `event_observed_train` parameters; this is a
  breaking signature change from the prior hand-rolled revision (which
  estimated censoring only from the evaluation set itself — sksurv's
  train-based convention is the more defensible one, so this is a
  correctness improvement, not just a refactor).

Every function returns a `MetricResult` (see src/eval/__init__.py) that
always includes a per-stratum breakdown by ethnicity, even though
discrimination/calibration are primarily "is the model good" questions —
this is what lets fairness.py re-use these functions directly for the
model-level parity checks in spec §3a instead of recomputing C-index or
calibration slope a second time.

Status: fully implemented and integration-tested against a real fitted
RandomSurvivalForest (src/models/survival.py) on the actual 1K-cohort
feature_panel.parquet / labels.parquet / splits.parquet artifacts.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import brier_score as _sksurv_brier_score
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc as _sksurv_cumulative_dynamic_auc
from sksurv.util import Surv

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
    StratumResult,
    stratify_and_compute,
)

ArrayLike = Sequence[float] | np.ndarray


def _survival_struct(event_times: np.ndarray, event_observed: np.ndarray) -> np.ndarray:
    """Build an sksurv structured (event: bool, time: float) array.

    Thin wrapper around `sksurv.util.Surv.from_arrays` so every call site
    in this module builds these the same way `src/models/survival.py`
    (`build_survival_labels`) already does — one source of truth for the
    array shape sksurv's own metric functions expect.
    """
    return Surv.from_arrays(event=np.asarray(event_observed, dtype=bool), time=np.asarray(event_times, dtype=float))


def _downgrade_undefined_to_insufficient(
    per_stratum: dict[str, StratumResult]
) -> dict[str, StratumResult]:
    """Reclassify a stratum whose metric came back `None` despite N>=min_n.

    `stratify_and_compute` only guards on sample SIZE (N>=MIN_STRATUM_N) —
    it has no way to know that a stratum meeting that size bar can still
    make a metric mathematically undefined (e.g. a stratum with zero
    observed events, or zero patients past a given horizon, which sksurv's
    AUC/Brier functions raise ValueError on). Rather than let `compute_fn`
    propagate that exception, `compute_time_windowed_auc` /
    `compute_brier_score` below catch it and return None; this helper then
    reclassifies any such "ok status but None value" stratum as
    `insufficient_sample`, preserving the invariant documented on
    `StratumResult` (value is None only when status is
    "insufficient_sample") instead of silently violating it.
    """
    fixed = {}
    for stratum, result in per_stratum.items():
        if result.status == "ok" and result.value is None:
            fixed[stratum] = StratumResult(
                stratum=result.stratum,
                n=result.n,
                value=None,
                status="insufficient_sample",
            )
        else:
            fixed[stratum] = result
    return fixed


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

    Delegates to `sksurv.metrics.concordance_index_censored` (the same
    function `src/models/survival.py`'s own `evaluate_survival_model` uses
    for its reported concordance index — this module's overall C-index on
    the real test split is therefore guaranteed to match that number
    exactly, by construction, not just by coincidence).

    Parameters
    ----------
    event_times : days from index_date to break (or censoring).
    event_observed : 1 if the patient broke persistence (event observed),
        0 if right-censored.
    risk_scores : higher = predicted higher risk of an EARLIER break.
        sksurv's convention (a pair is concordant if the higher-risk-score
        patient has the shorter survival time) matches this module's own
        documented convention directly — no sign transform needed, for
        RandomSurvivalForest's `predict_risk()` (see `rsf_risk_scores()`
        at the bottom of this file) or any other model's risk score that
        follows the same "higher = riskier" convention.
    ethnicity_labels : optional per-patient ethnicity ("hispanic" /
        "nonhispanic"). If provided, per-stratum C-index is computed via
        the same N>=15 guard as every other metric (used directly by
        fairness.check_c_index_parity — no separate computation there).

    Returns
    -------
    MetricResult with `overall_value` = C-index computed over the full
    cohort, and `per_stratum` populated iff `ethnicity_labels` was given.
    """
    event_times_arr = np.asarray(event_times, dtype=float)
    event_observed_arr = np.asarray(event_observed, dtype=bool)
    risk_scores_arr = np.asarray(risk_scores, dtype=float)

    overall_value = float(
        concordance_index_censored(event_observed_arr, event_times_arr, risk_scores_arr)[0]
    )

    def _safe_c_index(
        event_times: np.ndarray, event_observed: np.ndarray, risk_scores: np.ndarray
    ) -> Optional[float]:
        # A stratum can clear the N>=MIN_STRATUM_N size guard in
        # stratify_and_compute and still be mathematically undefined here —
        # concordance_index_censored raises ValueError if the slice has zero
        # observed events (all censored) or zero comparable pairs. That's a
        # real, reachable case for a rare-event outcome sliced by ethnicity
        # (unlike the N=5 Hispanic cohort, which never reaches this function
        # at all — it's caught by the size guard first). Caught here and
        # downgraded via _downgrade_undefined_to_insufficient below, exactly
        # like compute_time_windowed_auc/compute_brier_score already do, so
        # this metric doesn't crash the whole report the first time it hits
        # a reportable-but-degenerate stratum.
        try:
            return float(
                concordance_index_censored(event_observed.astype(bool), event_times, risk_scores)[0]
            )
        except ValueError:
            return None

    per_stratum = {}
    if ethnicity_labels is not None:
        per_stratum = stratify_and_compute(
            ethnicity_labels,
            {
                "event_times": event_times_arr,
                "event_observed": event_observed_arr,
                "risk_scores": risk_scores_arr,
            },
            compute_fn=_safe_c_index,
            known_strata=ETHNICITY_STRATA,
        )
        per_stratum = _downgrade_undefined_to_insufficient(per_stratum)

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
    event_times_train: ArrayLike,
    event_observed_train: ArrayLike,
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

    Delegates to `sksurv.metrics.cumulative_dynamic_auc` (see module
    docstring's REFACTOR DECISION — this replaces a hand-rolled Uno et al.
    (2007) IPCW estimator that was cross-validated to match this exact
    function's output on real model data before being retired in favor of
    calling it directly).

    Parameters
    ----------
    event_times_train, event_observed_train : the survival OUTCOME data
        (NOT features) from the model's TRAINING split — required to
        estimate the censoring distribution used for IPCW weighting,
        matching the same convention `src/models/survival.py`'s own
        `evaluate_time_dependent_auc` uses (`y_train`). This is a
        DIFFERENT array than `event_times`/`event_observed` (the
        evaluation-split outcomes being scored) — passing the same split
        for both would estimate the censoring distribution from the very
        data being evaluated, which is what the earlier hand-rolled
        revision did and which this refactor deliberately corrects.

    Note on `flag`: the spec does not define an independent numeric bar
    for this metric (it's explicitly a secondary, judge-legible landmark,
    not a metric with its own pass/fail criterion). The flag here is a
    bare sanity check against the 0.5 chance baseline only.

    Note on failure handling: sksurv raises `ValueError` if a stratum (or
    the whole cohort) has zero cases, zero controls, or `horizon_days`
    falls outside the range `survival_train` can support. These are caught
    and reported as `insufficient_sample` rather than propagating a raw
    sksurv traceback — consistent with `src/models/survival.py`'s own
    stated preference for a clear, cohort-count-specific error over a
    generic library exception.
    """
    event_times_arr = np.asarray(event_times, dtype=float)
    event_observed_arr = np.asarray(event_observed, dtype=bool)
    risk_scores_arr = np.asarray(risk_scores, dtype=float)
    survival_train = _survival_struct(event_times_train, event_observed_train)

    def _safe_auc(times_: np.ndarray, observed_: np.ndarray, risk_: np.ndarray) -> Optional[float]:
        n_cases = int(((times_ <= horizon_days) & observed_).sum())
        n_controls = int((times_ > horizon_days).sum())
        if n_cases == 0 or n_controls == 0:
            return None
        try:
            survival_test = _survival_struct(times_, observed_)
            auc_arr, _mean_auc = _sksurv_cumulative_dynamic_auc(
                survival_train, survival_test, risk_, np.asarray([horizon_days], dtype=float)
            )
            return float(auc_arr[0])
        except ValueError:
            return None

    overall_value = _safe_auc(event_times_arr, event_observed_arr, risk_scores_arr)

    per_stratum = {}
    if ethnicity_labels is not None:
        per_stratum = stratify_and_compute(
            ethnicity_labels,
            {
                "event_times": event_times_arr,
                "event_observed": event_observed_arr,
                "risk_scores": risk_scores_arr,
            },
            compute_fn=lambda event_times, event_observed, risk_scores: _safe_auc(
                event_times, event_observed, risk_scores
            ),
            known_strata=ETHNICITY_STRATA,
        )
        per_stratum = _downgrade_undefined_to_insufficient(per_stratum)

    if overall_value is None:
        flag: str = "insufficient_sample"
        description = (
            f"{horizon_days}-day AUC: insufficient cases/controls in the cohort, "
            "or horizon falls outside the training split's supported range."
        )
    else:
        flag = "pass" if overall_value > 0.5 else "fail"
        description = (
            f"{horizon_days}-day AUC={overall_value:.3f} — a single fixed-horizon "
            "read of the same model, provided for interpretability; the underlying "
            "prediction is a continuous break window, not this cutoff."
        )

    return MetricResult(
        metric_name=f"{horizon_days}-day time-windowed AUC",
        overall_value=overall_value,
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=(
            "No independent spec-defined bar (secondary/interpretability metric); "
            "flag is a bare >0.5 chance-baseline sanity check only (spec §1)."
        ),
        extra={
            "n_cases": int(((event_times_arr <= horizon_days) & event_observed_arr).sum()),
            "n_controls": int((event_times_arr > horizon_days).sum()),
        },
    )


# ---------------------------------------------------------------------------
# §2 — Calibration
# ---------------------------------------------------------------------------


def compute_brier_score(
    event_times: ArrayLike,
    event_observed: ArrayLike,
    predicted_survival_at_horizon: ArrayLike,
    event_times_train: ArrayLike,
    event_observed_train: ArrayLike,
    *,
    horizon_days: int,
    ethnicity_labels: Optional[ArrayLike] = None,
) -> MetricResult:
    """IPCW Brier score at a fixed horizon vs. the no-skill reference (spec §2a).

    Delegates to `sksurv.metrics.brier_score` (see module docstring's
    REFACTOR DECISION).

    Parameters
    ----------
    predicted_survival_at_horizon : model-predicted P(survive past
        horizon_days) per patient — i.e. Ŝ_i(horizon_days). For the fitted
        `sksurv.ensemble.RandomSurvivalForest` (`src/models/survival.py`)
        this is extracted via `rsf_predicted_survival_at_horizon()` at the
        bottom of this file.
    event_times_train, event_observed_train : TRAINING-split outcome data
        for the censoring-distribution estimate — see
        `compute_time_windowed_auc`'s docstring for why this must be the
        training split, not the split being scored.

    "Well calibrated" per spec §2a is defined RELATIVE to the no-skill
    (population Kaplan-Meier) reference at the same horizon: Brier(model)
    must be >= BRIER_RELATIVE_IMPROVEMENT_THRESHOLD (10%) lower than
    Brier(no-skill), where Brier(no-skill) uses the population marginal KM
    survival estimate (same value for every patient, estimated via
    `lifelines.KaplanMeierFitter` on the evaluation split) in place of the
    model's per-patient prediction, run through the SAME
    `sksurv.metrics.brier_score` call (same `survival_train` reference) so
    the comparison isolates the model's added value.
    """
    event_times_arr = np.asarray(event_times, dtype=float)
    event_observed_arr = np.asarray(event_observed, dtype=bool)
    predicted_survival_arr = np.asarray(predicted_survival_at_horizon, dtype=float)
    survival_train = _survival_struct(event_times_train, event_observed_train)

    def _safe_brier(
        times_: np.ndarray, observed_: np.ndarray, predicted_survival_: np.ndarray
    ) -> Optional[float]:
        try:
            survival_test = _survival_struct(times_, observed_)
            _times_out, brier_scores = _sksurv_brier_score(
                survival_train,
                survival_test,
                predicted_survival_.reshape(-1, 1),
                np.asarray([horizon_days], dtype=float),
            )
            return float(brier_scores[0])
        except ValueError:
            return None

    overall_value = _safe_brier(event_times_arr, event_observed_arr, predicted_survival_arr)

    # No-skill reference: population marginal KM survival at the horizon
    # (evaluation split), broadcast to every patient, run through the SAME
    # sksurv brier_score call (same survival_train).
    event_km = KaplanMeierFitter()
    event_km.fit(event_times_arr, event_observed=event_observed_arr)
    population_survival_at_horizon = float(
        event_km.survival_function_at_times(horizon_days).iloc[0]
    )
    no_skill_predicted_survival = np.full(
        len(event_times_arr), population_survival_at_horizon
    )
    no_skill_value = _safe_brier(
        event_times_arr, event_observed_arr, no_skill_predicted_survival
    )

    per_stratum = {}
    if ethnicity_labels is not None:
        per_stratum = stratify_and_compute(
            ethnicity_labels,
            {
                "event_times": event_times_arr,
                "event_observed": event_observed_arr,
                "predicted_survival": predicted_survival_arr,
            },
            compute_fn=lambda event_times, event_observed, predicted_survival: (
                _safe_brier(event_times, event_observed, predicted_survival)
            ),
            known_strata=ETHNICITY_STRATA,
        )
        per_stratum = _downgrade_undefined_to_insufficient(per_stratum)
        # NOTE: per-stratum values here are raw Brier scores (each using the
        # SAME shared survival_train censoring reference) — NOT
        # relative-improvement-over-no-skill. The relative-improvement
        # pass/fail bar below is a global-cohort statement only; per spec
        # §3a, stratified CALIBRATION QUALITY is checked via the slope
        # metric (compute_calibration_at_horizon / check_calibration_parity).

    if overall_value is None or no_skill_value is None or no_skill_value == 0:
        flag: str = "insufficient_sample"
        description = (
            f"{horizon_days}-day Brier score: could not compute a valid no-skill "
            "reference (insufficient events/controls, population event rate is 0 "
            "at this horizon, or horizon falls outside the training split's "
            "supported range)."
        )
        relative_improvement: Optional[float] = None
    else:
        relative_improvement = 1.0 - (overall_value / no_skill_value)
        flag = "pass" if relative_improvement >= BRIER_RELATIVE_IMPROVEMENT_THRESHOLD else "fail"
        description = (
            f"{horizon_days}-day Brier score={overall_value:.4f} vs. no-skill "
            f"reference={no_skill_value:.4f} "
            f"({relative_improvement:+.1%} relative improvement; "
            f"{'meets' if flag == 'pass' else 'does not meet'} the "
            f"{BRIER_RELATIVE_IMPROVEMENT_THRESHOLD:.0%} bar)."
        )

    return MetricResult(
        metric_name=f"{horizon_days}-day Brier score",
        overall_value=overall_value,
        per_stratum=per_stratum,
        flag=flag,  # type: ignore[arg-type]
        description=description,
        threshold_description=(
            f">= {BRIER_RELATIVE_IMPROVEMENT_THRESHOLD:.0%} relative improvement "
            "over the no-skill (population Kaplan-Meier) reference (spec §2a)."
        ),
        extra={
            "no_skill_brier": no_skill_value if no_skill_value is not None else float("nan"),
            "relative_improvement": (
                relative_improvement if relative_improvement is not None else float("nan")
            ),
        },
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

    UNCHANGED by the sksurv-delegation refactor — see module docstring.
    Still uses `lifelines.KaplanMeierFitter` for the per-bin observed
    event-rate estimate, since sksurv has no single-call equivalent for
    "survival probability at an arbitrary query time, per bin" and
    hand-rolling that step here is a few lines against a stable KM
    implementation, not a duplicate of the censoring-weight math this
    refactor removed elsewhere in this file.

    Parameters
    ----------
    predicted_risk_at_horizon : model-predicted P(break by horizon_days)
        per patient — i.e. 1 - Ŝ_i(horizon_days). Same extraction as
        compute_brier_score's `predicted_survival_at_horizon` (this is
        just its complement).
    n_bins : defaults to 5 (spec §2 rationale: ~262-patient cohort makes
        10 bins too sparse for a stable per-bin observed rate).
    slope_band : acceptance band for the fitted slope of observed vs.
        predicted bin rates (1.0 = perfect calibration). Global default is
        [0.8, 1.2] (spec §2b); fairness.py passes the widened
        CALIBRATION_SLOPE_BAND_STRATUM band when calling this per-stratum
        for the parity check (spec §3a / Assumption A3).

    Bin assignment and each bin's observed rate:
      - Patients are bucketed into `n_bins` quantile bins by
        `predicted_risk_at_horizon`.
      - Each bin's OBSERVED event rate at `horizon_days` is estimated via
        Kaplan-Meier within that bin (1 - Ŝ_bin(horizon_days)), not a
        naive event-count ratio, so censored patients are handled
        correctly rather than dropped or miscounted as non-events.
    """
    event_times_arr = np.asarray(event_times, dtype=float)
    event_observed_arr = np.asarray(event_observed, dtype=float)
    predicted_risk_arr = np.asarray(predicted_risk_at_horizon, dtype=float)

    def _bin_calibration_slope(
        event_times: np.ndarray, event_observed: np.ndarray, predicted_risk: np.ndarray
    ) -> Optional[float]:
        n = len(predicted_risk)
        if n < n_bins:
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
            survival_at_horizon = float(
                kmf.survival_function_at_times(horizon_days).iloc[0]
            )
            observed_rates.append(1.0 - survival_at_horizon)
        if len(predicted_means) < 2:
            return None
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
        per_stratum = _downgrade_undefined_to_insufficient(per_stratum)

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
    event_times_train: ArrayLike,
    event_observed_train: ArrayLike,
    *,
    ethnicity_labels: Optional[ArrayLike] = None,
) -> dict[str, MetricResult | list[MetricResult]]:
    """Run every calibration/discrimination metric and assemble one report.

    Parameters
    ----------
    predicted_survival_by_horizon : {horizon_days: predicted_survival_array}
        for each horizon in BRIER_HORIZONS_DAYS (30/90/180). Kept as an
        explicit mapping rather than re-deriving inside this function
        because the extraction from a fitted model is horizon-specific —
        see `rsf_predicted_survival_at_horizon()` below for the RSF
        extraction this is wired to.
    event_times_train, event_observed_train : TRAINING-split outcome data,
        now required by both `compute_time_windowed_auc` and
        `compute_brier_score` for their sksurv-delegated IPCW censoring
        estimate (see module docstring's REFACTOR DECISION) — this is a
        breaking signature change from the prior hand-rolled revision.

    Returns
    -------
    dict with keys:
      "c_index": MetricResult
      "time_windowed_auc": MetricResult
      "brier_scores": list[MetricResult], one per BRIER_HORIZONS_DAYS
      "calibration_slopes": list[MetricResult], one per BRIER_HORIZONS_DAYS
    This is the shape src/eval's future report generator consumes — do not
    change these keys without updating that consumer too.
    """
    c_index_result = compute_c_index(
        event_times, event_observed, risk_scores, ethnicity_labels=ethnicity_labels
    )
    auc_result = compute_time_windowed_auc(
        event_times,
        event_observed,
        risk_scores,
        event_times_train,
        event_observed_train,
        ethnicity_labels=ethnicity_labels,
    )

    brier_results = []
    calibration_results = []
    for horizon in BRIER_HORIZONS_DAYS:
        predicted_survival = predicted_survival_by_horizon.get(horizon)
        if predicted_survival is None:
            continue
        brier_results.append(
            compute_brier_score(
                event_times,
                event_observed,
                predicted_survival,
                event_times_train,
                event_observed_train,
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


# ---------------------------------------------------------------------------
# RandomSurvivalForest model adapters — the ONLY functions in this module
# that know a specific model class/library. Every metric function above
# takes generic arrays; these two functions are the seam between a fitted
# `sksurv.ensemble.RandomSurvivalForest` (via `src/models/survival.py`,
# Umar's model) and that generic contract. Unaffected by this revision's
# sksurv-delegation refactor.
# ---------------------------------------------------------------------------


def rsf_risk_scores(fitted_model, X: pd.DataFrame) -> np.ndarray:
    """Extract risk_scores from a fitted RSF pipeline, in this module's
    "higher = riskier" convention.

    `fitted_model` is the `sklearn.pipeline.Pipeline` returned by
    `src.models.survival.fit_survival_model` (impute -> RandomSurvivalForest).
    Delegates to `src.models.survival.predict_risk`, whose docstring already
    documents "higher = higher risk of a sustained gap" — the same
    convention this module's `compute_c_index` / `compute_time_windowed_auc`
    (and sksurv's own metric functions) expect, so no sign transform is
    needed here; this function exists purely so every call site shares one
    import/extraction point rather than reaching into `src.models.survival`
    directly.
    """
    from src.models.survival import predict_risk

    return np.asarray(predict_risk(fitted_model, X), dtype=float)


def rsf_predicted_survival_at_horizon(
    fitted_model, X: pd.DataFrame, horizon_days: float
) -> np.ndarray:
    """Extract per-patient Ŝ_i(horizon_days) from a fitted RSF pipeline.

    `src.models.survival.predict_survival_function` returns one
    `sksurv.functions.StepFunction` per patient (each callable at an
    arbitrary time via sksurv's own step-interpolation); this evaluates
    every patient's step function at `horizon_days` and returns one value
    per patient, in the shape `compute_brier_score` /
    `compute_calibration_at_horizon` expect (via `1 - predicted_survival`
    for the latter — see run_calibration_report).

    Raises whatever `StepFunction.__call__` raises if `horizon_days` falls
    outside a given patient's domain (sksurv extrapolates flat by default
    for RandomSurvivalForest, so this is not expected to raise in practice,
    but is not defensively caught here — a silent wrong number would be
    worse than a loud error surfacing a real out-of-domain horizon).
    """
    from src.models.survival import predict_survival_function

    survival_functions = predict_survival_function(fitted_model, X)
    return np.array([fn(horizon_days) for fn in survival_functions], dtype=float)

