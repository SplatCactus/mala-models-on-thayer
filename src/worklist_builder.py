"""
src/routing/worklist_builder.py

Assembles the final worklist JSON payload that dashboard.html consumes.
Gathers routing decisions (rules.py) + capacity ranking (capacity.py) into
one schema, keeping `eligible_pool` and `capped_worklist` as clearly
separate top-level arrays (per requirement: fairness.py needs both,
and the dashboard should only ever render `capped_worklist`).

SCHEMA (top-level keys)
------------------------
{
  "meta": {
      "generated_at": ISO8601 str,
      "routing_table_version": int,
      "cohort_size": int,
      "role_caps_used": {role: int},
      "role_capacity_expansions": {role: int}   # safety-driven cap expansions
  },
  "capped_worklist": [ WorklistCard, ... ],   # what dashboard.html renders
  "eligible_pool_summary": {                  # NOT rendered by the dashboard;
      "total": int,                           # consumed by fairness.py /
      "by_action": {action: int}              # analytics only
  },
  "eligible_pool": [ WorklistCard, ... ]      # full pool, for fairness.py
}

WorklistCard fields (per patient):
  patient_id, action, secondary_action, requires_human_review,
  is_safety_override, priority_score, rank_in_role,
  days_to_predicted_break, predicted_risk,
  script: { en: {label, rationale}, es: {label, rationale} }
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

from src.routing.capacity import CapacityResult, WorklistEntry

ACTION_LABELS = {
    "pharmacist": {"en": "Pharmacist Review", "es": "Revisión de Farmacéutico"},
    "social_worker": {"en": "Social Worker Outreach", "es": "Contacto de Trabajo Social"},
    "chw_call": {"en": "CHW Call", "es": "Llamada de CHW"},
}


def _card(entry: WorklistEntry) -> dict:
    d = entry.decision
    return {
        "patient_id": entry.patient_id,
        "action": entry.action,
        "secondary_action": d.secondary_action,
        "requires_human_review": d.requires_human_review,
        "is_safety_override": d.is_safety_override,
        "used_tie_break_hierarchy": d.used_tie_break_hierarchy,
        "top_driver": d.top_driver,
        "priority_score": round(entry.priority_score, 6),
        "rank_in_role": entry.rank_in_role,
        "days_to_predicted_break": round(entry.days_to_predicted_break, 2),
        "predicted_risk": round(entry.predicted_risk, 6),
        "capacity_expanded_for_safety": entry.capacity_expanded_for_safety,
        "script": {
            "en": {
                "label": ACTION_LABELS[entry.action]["en"],
                "rationale": d.rationale_en,
            },
            "es": {
                "label": ACTION_LABELS[entry.action]["es"],
                "rationale": d.rationale_es,
            },
        },
    }


def build_worklist_payload(
    capacity_result: CapacityResult, routing_table_version: int
) -> dict:
    capped_cards = [_card(e) for e in capacity_result.capped_worklist]
    eligible_cards = [_card(e) for e in capacity_result.eligible_pool]

    by_action: Dict[str, int] = {}
    for e in capacity_result.eligible_pool:
        by_action[e.action] = by_action.get(e.action, 0) + 1

    payload = {
        "meta": {
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "routing_table_version": routing_table_version,
            "cohort_size": len(capacity_result.eligible_pool),
            "role_caps_used": capacity_result.role_caps_used,
            "role_capacity_expansions": capacity_result.role_expansions,
        },
        "capped_worklist": capped_cards,
        "eligible_pool_summary": {
            "total": len(eligible_cards),
            "by_action": by_action,
        },
        "eligible_pool": eligible_cards,
    }
    return payload
