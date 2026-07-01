"""BP Cascade RI evaluation harness — shared types and report-language constants.

This package holds the credibility layer of the pitch: calibration.py and
fairness.py. Every metric function in this package returns the same
StratumResult / MetricResult shapes defined here, and every report-facing
sentence about synthetic data comes from the disclaimer helpers below —
NOT from ad hoc strings written inside individual metric functions. See
/docs/eval_metric_spec.md, which is the single source of truth this module
implements. Any change to a threshold, bin count, or disclaimer wording
must be made in that spec doc first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds locked in /docs/eval_metric_spec.md — do not edit ad hoc.
# ---------------------------------------------------------------------------

#: §4 — minimum patients in a stratum before a metric is reported as a
#: number rather than "insufficient_sample".
MIN_STRATUM_N: int = 15

#: §1 — C-index above this is "meaningful discrimination beyond chance".
C_INDEX_MEANINGFUL_THRESHOLD: float = 0.65
#: §1 — C-index above this (but below MEANINGFUL) is "weak but non-trivial".
C_INDEX_WEAK_THRESHOLD: float = 0.55

#: §2a — Brier score must be at least this fraction lower than the
#: no-skill (Kaplan-Meier marginal) reference to count as "well calibrated".
BRIER_RELATIVE_IMPROVEMENT_THRESHOLD: float = 0.10
#: §2a — horizons (days) at which Brier score is evaluated.
BRIER_HORIZONS_DAYS: tuple[int, ...] = (30, 90, 180)

#: §2b — number of quintile bins for the calibration-at-horizon plot.
#: Fixed at 5, not 10, given the ~262-patient cohort (see spec §2 rationale).
CALIBRATION_N_BINS: int = 5
#: §2b — global calibration slope acceptance band (1.0 = perfect).
CALIBRATION_SLOPE_BAND_GLOBAL: tuple[float, float] = (0.8, 1.2)
#: §3a Assumption A3 — widened within-stratum band (smaller N -> more noise).
CALIBRATION_SLOPE_BAND_STRATUM: tuple[float, float] = (0.7, 1.3)
#: §2b / §3a — horizon used for the (single, report-legible) parity check.
CALIBRATION_PARITY_HORIZON_DAYS: int = 90

#: §1 — fixed landmark horizon for the secondary, judge-legible AUC.
AUC_HORIZON_DAYS: int = 90

#: §3a — max pairwise C-index gap between ethnicity strata before flagging.
C_INDEX_PARITY_MAX_GAP: float = 0.05
#: §3a — four-fifths / disparate-impact rule for worklist selection rate.
SELECTION_RATE_PARITY_MIN_RATIO: float = 0.8
#: §6 — capacity-cap exclusion-rate gap (percentage points, i.e. 0-1 scale)
#: between strata before flagging the capacity cap itself as disparate.
CAPACITY_CAP_EXCLUSION_MAX_GAP_PP: float = 0.15
#: §3b / A4 — significance level for the routed-action chi-square test.
ROUTED_ACTION_CHI2_ALPHA: float = 0.05

#: §6 — the two ethnicity/language axes this harness evaluates (A1: race and
#: migrant status are explicitly out of scope for this spec/sprint).
ETHNICITY_STRATA: tuple[str, ...] = ("hispanic", "nonhispanic")
LANGUAGE_STRATA: tuple[str, ...] = ("EN", "ES")


# ---------------------------------------------------------------------------
# Synthetic-data framing language — §5. Used verbatim, never improvised
# per-metric. Every MetricResult.disclaimer is populated via
# synthetic_data_disclaimer(); every top-level report calls
# report_framing_paragraph() exactly once.
# ---------------------------------------------------------------------------

_SYNTHETIC_DATA_DISCLAIMER_TEMPLATE: str = (
    "This {metric_name} is computed on Synthea-derived synthetic Rhode Island "
    "data and demonstrates the evaluation methodology BP Cascade RI will apply "
    "to real patient data — it is not a validated clinical finding. Real-RI "
    "validation, run against production EHR/claims data under an IRB-approved "
    "protocol, is the named next step before any deployment decision."
)

_REPORT_FRAMING_PARAGRAPH_TEMPLATE: str = (
    "All results in this report are generated from a synthetic {cohort_n}-patient "
    "Synthea cohort ({treated_htn_n} treated-hypertensive patients) built to "
    "demonstrate BP Cascade RI's modeling, explanation, and routing methodology "
    "end to end. Synthetic data lets us validate that the pipeline — survival "
    "model \u2192 SHAP-based routing \u2192 fairness audit — runs correctly and "
    "produces internally consistent, checkable numbers. It cannot and does not "
    "establish real-world model performance, calibration, or equity on actual "
    "Rhode Island patients. Real-RI validation is the explicit, named next "
    "step, not an afterthought."
)


def synthetic_data_disclaimer(metric_name: str) -> str:
    """Return the §5 disclaimer sentence for a specific metric, verbatim.

    Every MetricResult produced by this package must populate its
    ``disclaimer`` field via this function (see MetricResult below) so the
    wording is identical everywhere it appears in the final report.
    """
    return _SYNTHETIC_DATA_DISCLAIMER_TEMPLATE.format(metric_name=metric_name)


def report_framing_paragraph(cohort_n: int = 1124, treated_htn_n: int = 262) -> str:
    """Return the §5 top-of-report framing paragraph.

    Defaults reflect the current committed feature_panel.parquet cohort
    (262 treated-hypertensive patients). Callers building a report against
    the 1K or 300K Synthea bundle should pass the actual counts explicitly
    rather than relying on these defaults.
    """
    return _REPORT_FRAMING_PARAGRAPH_TEMPLATE.format(
        cohort_n=cohort_n, treated_htn_n=treated_htn_n
    )


# ---------------------------------------------------------------------------
# Shared result types. Every metric function in calibration.py and
# fairness.py returns a MetricResult; every stratum inside it is a
# StratumResult. This is requirement (2) from the scaffolding brief — do
# not let individual metric functions invent their own shapes.
# ---------------------------------------------------------------------------

#: Overall pass/fail/insufficient-sample/not-applicable flag for a metric,
#: evaluated against the numeric threshold named in /docs/eval_metric_spec.md.
Flag = Literal["pass", "fail", "insufficient_sample", "not_applicable"]

#: Per-stratum status. "insufficient_sample" is set purely by the
#: N < MIN_STRATUM_N guard (stratify_by_label below) and NEVER co-occurs
#: with a populated numeric `value` — see spec §4.
StratumStatus = Literal["ok", "insufficient_sample"]


@dataclass
class StratumResult:
    """One stratum's worth of a single metric.

    `value` is None whenever `status == "insufficient_sample"`. This is
    enforced by stratify_by_label()/finalize_stratum() below, not left to
    each metric function to remember.
    """

    stratum: str
    n: int
    value: Optional[float]
    status: StratumStatus


@dataclass
class MetricResult:
    """Structured result for exactly one metric, per requirement (2).

    Every stratum the cohort contains for the relevant axis (ethnicity or
    language) MUST appear in `per_stratum`, even if its status is
    "insufficient_sample" — strata are never silently dropped (spec §4).
    """

    metric_name: str
    overall_value: Optional[float]
    per_stratum: dict[str, StratumResult]
    flag: Flag
    description: str
    threshold_description: str
    disclaimer: str = field(default="")

    def __post_init__(self) -> None:
        if not self.disclaimer:
            self.disclaimer = synthetic_data_disclaimer(self.metric_name)


# ---------------------------------------------------------------------------
# Small-stratum guard — pure logic, no model dependency, fully implemented
# now per requirement (3). Both calibration.py and fairness.py build their
# per-stratum breakdowns through this single function so the N=15 rule is
# enforced identically everywhere.
# ---------------------------------------------------------------------------


def stratify_and_compute(
    stratum_labels: Sequence[str] | np.ndarray,
    arrays: Mapping[str, Sequence[float] | np.ndarray],
    *,
    compute_fn,
    min_n: int = MIN_STRATUM_N,
    known_strata: Optional[Sequence[str]] = None,
) -> dict[str, StratumResult]:
    """Split one or more aligned arrays by `stratum_labels`, apply the
    N>=min_n guard, and call `compute_fn` on each stratum's slice.

    `arrays` is a name->array mapping of every patient-level array a metric
    needs (e.g. {"event_times": ..., "event_observed": ..., "risk_scores":
    ...}), all aligned to the same patient order as `stratum_labels`.

    For each stratum in `known_strata` (falls back to the unique labels
    observed if not given — but callers should almost always pass
    ETHNICITY_STRATA or LANGUAGE_STRATA explicitly, see below):
      - if the stratum has fewer than `min_n` members: StratumResult with
        status="insufficient_sample", value=None. compute_fn is NOT called
        — a too-small group never produces a number (spec §4).
      - otherwise: compute_fn(**{name: sliced_array, ...}) -> Optional[float]
        is called with every array in `arrays` sliced to that stratum's
        patients, and the result is wrapped with status="ok".

    Passing `known_strata` explicitly (e.g. ETHNICITY_STRATA) ensures a
    stratum with zero patients in this particular slice still appears in
    the output as insufficient_sample rather than being silently absent
    from the returned dict (spec §4: "never silently dropped").
    """
    labels_arr = np.asarray(stratum_labels)
    array_items = {name: np.asarray(arr) for name, arr in arrays.items()}
    strata = list(known_strata) if known_strata is not None else sorted(set(labels_arr))

    results: dict[str, StratumResult] = {}
    for stratum in strata:
        mask = labels_arr == stratum
        n = int(mask.sum())
        if n < min_n:
            results[stratum] = StratumResult(
                stratum=stratum, n=n, value=None, status="insufficient_sample"
            )
            continue
        sliced = {name: arr[mask] for name, arr in array_items.items()}
        stratum_value = compute_fn(**sliced)
        results[stratum] = StratumResult(
            stratum=stratum, n=n, value=stratum_value, status="ok"
        )
    return results
