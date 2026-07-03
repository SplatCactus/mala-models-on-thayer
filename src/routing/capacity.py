"""
src/routing/capacity.py

Takes the full set of routing decisions (one per patient, from rules.py)
and produces:
  1. `eligible_pool`   — every routed patient, uncapped, with a computed
                          priority score. This is what fairness.py needs
                          to compute pre-capacity parity metrics.
  2. `capped_worklist` — the eligible pool after applying role-specific
                          maximum capacity caps, i.e. what actually gets
                          worked this cycle. This is what dashboard.html
                          renders and what fairness.py compares the
                          eligible pool against to detect cap-induced bias.

PRIORITIZATION
--------------
Within each action role, patients are ranked by:

    priority_score = urgency_weight * modifiability_weight

  - urgency_weight = 1 / (1 + days_to_predicted_break)
    Patients closer to their predicted medication "break window"
    (predicted discontinuation date) are prioritized — intervening
    earlier has less value than catching someone right before they lapse,
    but catching them AFTER they've already lapsed has none, so recency
    to the break window is what we optimize.

  - modifiability_weight = normalized |driver_value| for the resolved
    top driver, min-max scaled to [0.1, 1.0] within the role's candidate
    pool. A higher-magnitude driver signal means the modifiable factor is
    more clearly the dominant lever for that patient, so intervention is
    more likely to move the needle.

Safety-override patients (requires_human_review=True) are always placed
at the top of their role's queue regardless of score, and are never
excluded by the capacity cap — clinical safety cases are never dropped
for capacity reasons. If a cap would be exceeded by including all
safety-override cases, the cap is expanded for that role and the excess
is logged, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.routing.rules import RoutingDecision

DEFAULT_ROLE_CAPS: Dict[str, int] = {
    "pharmacist": 40,
    "social_worker": 30,
    "chw_call": 50,
}


@dataclass
class WorklistEntry:
    patient_id: str
    action: str
    priority_score: float
    days_to_predicted_break: float
    predicted_risk: float
    rank_in_role: int
    included_in_capped_worklist: bool
    capacity_expanded_for_safety: bool
    decision: RoutingDecision

    def to_dict(self) -> dict:
        d = self.decision.to_dict()
        d.update(
            {
                "priority_score": round(self.priority_score, 6),
                "days_to_predicted_break": round(self.days_to_predicted_break, 2),
                "predicted_risk": round(self.predicted_risk, 6),
                "rank_in_role": self.rank_in_role,
                "included_in_capped_worklist": self.included_in_capped_worklist,
                "capacity_expanded_for_safety": self.capacity_expanded_for_safety,
            }
        )
        return d


@dataclass
class CapacityResult:
    eligible_pool: List[WorklistEntry] = field(default_factory=list)
    capped_worklist: List[WorklistEntry] = field(default_factory=list)
    role_caps_used: Dict[str, int] = field(default_factory=dict)
    role_expansions: Dict[str, int] = field(default_factory=dict)


class CapacityEngine:
    def __init__(self, role_caps: Dict[str, int] = None):
        self.role_caps = dict(role_caps) if role_caps else dict(DEFAULT_ROLE_CAPS)

    @staticmethod
    def _min_max_scale(values: List[float], lo: float = 0.1, hi: float = 1.0) -> List[float]:
        if not values:
            return []
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [hi for _ in values]
        return [lo + (hi - lo) * (v - vmin) / (vmax - vmin) for v in values]

    def build(
        self,
        decisions: List[RoutingDecision],
        days_to_break: Dict[str, float],
        predicted_risk: Dict[str, float],
    ) -> CapacityResult:
        result = CapacityResult()

        by_role: Dict[str, List[RoutingDecision]] = {}
        for d in decisions:
            by_role.setdefault(d.action, []).append(d)

        for role, role_decisions in by_role.items():
            magnitudes = [abs(d.driver_value) for d in role_decisions]
            mod_weights = self._min_max_scale(magnitudes)

            scored: List[WorklistEntry] = []
            for decision, mod_w in zip(role_decisions, mod_weights):
                dtb = days_to_break.get(decision.patient_id, 999.0)
                urgency_w = 1.0 / (1.0 + max(dtb, 0.0))
                score = urgency_w * mod_w
                scored.append(
                    WorklistEntry(
                        patient_id=decision.patient_id,
                        action=role,
                        priority_score=score,
                        days_to_predicted_break=dtb,
                        predicted_risk=predicted_risk.get(decision.patient_id, float("nan")),
                        rank_in_role=-1,
                        included_in_capped_worklist=False,
                        capacity_expanded_for_safety=False,
                        decision=decision,
                    )
                )

            # Safety-override patients always sort first, then by score desc.
            scored.sort(
                key=lambda e: (not e.decision.requires_human_review, -e.priority_score)
            )
            for rank, entry in enumerate(scored, start=1):
                entry.rank_in_role = rank

            cap = self.role_caps.get(role, len(scored))
            n_safety = sum(1 for e in scored if e.decision.requires_human_review)
            effective_cap = max(cap, n_safety)
            expanded = effective_cap > cap
            if expanded:
                result.role_expansions[role] = effective_cap - cap

            for entry in scored:
                entry.capacity_expanded_for_safety = expanded and entry.decision.requires_human_review
                entry.included_in_capped_worklist = entry.rank_in_role <= effective_cap

            result.role_caps_used[role] = effective_cap
            result.eligible_pool.extend(scored)
            result.capped_worklist.extend([e for e in scored if e.included_in_capped_worklist])

        return result
