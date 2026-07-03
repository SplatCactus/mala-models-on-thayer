"""
src/eval/fairness.py

Computes ethnicity-stratum fairness/parity metrics by comparing the
uncapped `eligible_pool` (src/routing/capacity.py) against the
capacity-capped `capped_worklist`. This is why capacity.py exposes both
sets separately: if the eligible pool is demographically balanced but the
capped worklist isn't, that is a capacity-cap fairness problem, not a
model or routing-rule problem, and the report needs to distinguish them.

WORKFLOW
--------
1. Selection-rate parity: for each ethnicity group, what fraction of that
   group's eligible-pool members made it into the capped worklist
   ("selection rate"). Report the min/max ratio across groups (a common
   80%-rule-style disparity ratio) both before capping (should be ~1.0
   trivially, included for completeness) and after capping.
2. Action-distribution parity: within the capped worklist, does the
   mix of actions (pharmacist / social_worker / chw_call) differ by
   ethnicity group in a way that suggests differential routing rather
   than differential need?
3. Discrimination parity: delegates to calibration.py to get C-index/AUC
   per ethnicity stratum, so a reviewer can see in one report both
   "are we selecting people fairly" and "is the model equally accurate
   across groups."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.eval.calibration import CalibrationReport, compute_calibration_report


@dataclass
class SelectionParity:
    group: str
    eligible_n: int
    capped_n: int
    selection_rate: float

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "eligible_n": self.eligible_n,
            "capped_n": self.capped_n,
            "selection_rate": round(self.selection_rate, 4),
        }


@dataclass
class FairnessReport:
    selection_parity: List[SelectionParity]
    min_max_disparity_ratio: float
    action_distribution_by_group: Dict[str, Dict[str, float]]
    calibration: CalibrationReport

    def to_dict(self) -> dict:
        return {
            "selection_parity_by_group": [s.to_dict() for s in self.selection_parity],
            "min_max_disparity_ratio": round(self.min_max_disparity_ratio, 4),
            "flags_disparate_impact_80pct_rule": self.min_max_disparity_ratio < 0.8,
            "action_distribution_by_group": {
                g: {a: round(p, 4) for a, p in dist.items()}
                for g, dist in self.action_distribution_by_group.items()
            },
            "calibration_by_group": self.calibration.to_dict(),
        }


def compute_fairness_report(
    eligible_pool_df: pd.DataFrame,
    capped_worklist_df: pd.DataFrame,
    scored_cohort_df: pd.DataFrame,
    group_col: str = "ethnicity",
) -> FairnessReport:
    """
    eligible_pool_df / capped_worklist_df: one row per (patient, decision),
        must contain patient_id, action, and group_col.
    scored_cohort_df: full cohort scoring table for calibration, must
        contain predicted_risk, actual_discontinued, days_to_predicted_break,
        and group_col.
    """
    selection_rows: List[SelectionParity] = []
    rates = []
    for group_value, elig_sub in eligible_pool_df.groupby(group_col):
        capped_sub = capped_worklist_df[capped_worklist_df[group_col] == group_value]
        eligible_n = len(elig_sub)
        capped_n = len(capped_sub)
        rate = capped_n / eligible_n if eligible_n > 0 else 0.0
        selection_rows.append(
            SelectionParity(
                group=str(group_value), eligible_n=eligible_n, capped_n=capped_n, selection_rate=rate
            )
        )
        rates.append(rate)

    selection_rows.sort(key=lambda s: s.group)
    disparity_ratio = (min(rates) / max(rates)) if rates and max(rates) > 0 else 1.0

    action_dist: Dict[str, Dict[str, float]] = {}
    for group_value, sub in capped_worklist_df.groupby(group_col):
        counts = sub["action"].value_counts(normalize=True)
        action_dist[str(group_value)] = counts.to_dict()

    calibration = compute_calibration_report(scored_cohort_df, group_col=group_col)

    return FairnessReport(
        selection_parity=selection_rows,
        min_max_disparity_ratio=disparity_ratio,
        action_distribution_by_group=action_dist,
        calibration=calibration,
    )
