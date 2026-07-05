"""
src/eval/calibration.py

Computes model discrimination/calibration metrics — Harrell's C-index and
ROC-AUC — overall and stratified by a grouping column (ethnicity, in this
pipeline). Consumed by fairness.py to build the parity report, and callable
standalone for model-monitoring purposes.

WORKFLOW
--------
1. Take the scored cohort (predicted_risk, actual outcome label, and a
   "time to event" proxy — here `days_to_predicted_break` — plus a group
   column).
2. Compute Harrell's C-index (pairwise concordance between predicted risk
   ranking and observed time-to-break-window / outcome ordering) and
   ROC-AUC, overall and within each stratum of `group_col`.
3. Return a CalibrationReport with per-stratum sample sizes so downstream
   consumers can flag strata too small for a reliable estimate.

We implement C-index and AUC directly (O(n^2) pairwise comparison, fine
for cohort sizes in the thousands) rather than pulling in `lifelines` or
`scikit-survival`, to keep this evaluation module dependency-light and
fully auditable — reviewers should be able to read the comparison loop
and see exactly what "concordant" means here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_RELIABLE_STRATUM_SIZE = 30


@dataclass
class StratumMetrics:
    group: str
    n: int
    n_events: int
    c_index: Optional[float]
    auc: Optional[float]
    reliable: bool

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "n": self.n,
            "n_events": self.n_events,
            "c_index": None if self.c_index is None else round(self.c_index, 4),
            "auc": None if self.auc is None else round(self.auc, 4),
            "reliable": self.reliable,
        }


@dataclass
class CalibrationReport:
    overall: StratumMetrics
    by_group: List[StratumMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "by_group": [s.to_dict() for s in self.by_group],
        }


def harrell_c_index(risk_scores: np.ndarray, time_to_event: np.ndarray, event: np.ndarray) -> Optional[float]:
    """Harrell's concordance index.

    A pair (i, j) is comparable if one of them had the event and their
    event time is shorter than the other's observed time. The pair is
    concordant if the one with the shorter time also has the higher
    predicted risk.

    Vectorized rewrite of the original O(n^2) double loop -- numerically
    identical, just fast enough for the 16K-patient cohort. Only a patient
    who actually had the event can be the "failed-first" member of a
    comparable pair (the `event[k] == 1` guard in the original), so we loop
    over just the event rows and vectorize the inner comparison across the
    whole cohort with numpy -- O(n_events * n) work, O(n) memory, no
    n-by-n pair matrix. Comparability and the tied-risk = 0.5 credit are
    preserved exactly, including the original's rule that an event-vs-
    censored pair is comparable whenever the other row is censored,
    independent of the two times.
    """
    risk = np.asarray(risk_scores, dtype=float)
    time = np.asarray(time_to_event, dtype=float)
    event = np.asarray(event)

    event_idx = np.flatnonzero(event == 1)
    if event_idx.size == 0:
        return None

    censored = event == 0  # the "or event[longer] == 0" branch, vectorized

    permissible = 0
    concordant = 0.0
    for k in event_idx:
        # Rows m that k (which failed) is comparable-and-earlier-than:
        # time[k] < time[m], OR m is censored (matches the original exactly).
        comparable = (time[k] < time) | censored
        comparable[k] = False  # never pair a row with itself
        r_m = risk[comparable]
        if r_m.size == 0:
            continue
        permissible += r_m.size
        concordant += float(np.count_nonzero(risk[k] > r_m))
        concordant += 0.5 * float(np.count_nonzero(risk[k] == r_m))

    if permissible == 0:
        return None
    return concordant / permissible


def roc_auc(risk_scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Rank-based ROC-AUC (Mann-Whitney U formulation), no sklearn dependency
    needed for this specific computation, avoids issues with single-class strata."""
    pos = risk_scores[labels == 1]
    neg = risk_scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None

    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    pos_ranks_sum = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _compute_stratum(df: pd.DataFrame, group_label: str) -> StratumMetrics:
    n = len(df)
    n_events = int(df["actual_discontinued"].sum())
    reliable = n >= MIN_RELIABLE_STRATUM_SIZE and 0 < n_events < n

    c_idx = None
    auc = None
    if n_events > 0 and n_events < n:
        c_idx = harrell_c_index(
            df["predicted_risk"].to_numpy(),
            df["days_to_predicted_break"].to_numpy(),
            df["actual_discontinued"].to_numpy(),
        )
        auc = roc_auc(df["predicted_risk"].to_numpy(), df["actual_discontinued"].to_numpy())

    return StratumMetrics(
        group=group_label, n=n, n_events=n_events, c_index=c_idx, auc=auc, reliable=reliable
    )


def compute_calibration_report(
    scored_cohort: pd.DataFrame, group_col: str = "ethnicity"
) -> CalibrationReport:
    """scored_cohort must contain: predicted_risk, actual_discontinued (0/1),
    days_to_predicted_break, and `group_col`."""
    overall = _compute_stratum(scored_cohort, "overall")

    by_group = []
    for group_value, sub_df in scored_cohort.groupby(group_col):
        by_group.append(_compute_stratum(sub_df, str(group_value)))

    return CalibrationReport(overall=overall, by_group=sorted(by_group, key=lambda s: s.group))
