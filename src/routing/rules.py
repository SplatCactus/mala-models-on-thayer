"""
src/routing/rules.py

The lookup engine that turns a patient's SHAP driver profile
(src/explain/shap_runner.PatientDriverProfile) into exactly one UI action:
`pharmacist`, `social_worker`, or `chw_call`.

WORKFLOW
--------
1. Load routing_table.yaml (versioned, clinician-reviewable rules).
2. For each patient:
   a. SAFETY OVERRIDE CHECK — if `trauma_exposure` is present in the
      patient's driver set at all (any positive attribution), force-route
      to the override action and set requires_human_review. This check
      runs before anything else and cannot be bypassed by tie-break logic.
   b. FALLBACK REMAP — apply `driver_fallbacks` (e.g. regimen_complexity
      -> bp_trend) to the ranked driver list before evaluating margins.
   c. TIE-BREAK MARGIN — if the top driver's attribution doesn't exceed
      the second-ranked driver's by `tie_break_margin` (relative, i.e.
      top >= margin * second), fall back to the strict
      `clinical_hierarchy` order and pick the first hierarchy driver
      present in the patient's candidate set.
   d. LOOKUP — map the resolved driver to an action via `drivers` table,
      carrying forward `secondary_action` into the background data layer
      (never surfaced as a second primary action).
3. Emit a RoutingDecision with a full bilingual (EN/ES) audit rationale
   string so a human reviewer can see exactly why a patient was routed
   where they were, without needing to inspect model internals.

No ML happens in this file. Every decision is fully reproducible from
routing_table.yaml + the ranked driver list handed in by shap_runner.py.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.explain.shap_runner import PatientDriverProfile

DEFAULT_ROUTING_TABLE_PATH = Path(__file__).parent / "routing_table.yaml"

VALID_ACTIONS = {"pharmacist", "social_worker", "chw_call"}


@dataclass
class RoutingDecision:
    patient_id: str
    action: str
    secondary_action: Optional[str]
    top_driver: str
    driver_value: float
    used_tie_break_hierarchy: bool
    requires_human_review: bool
    is_safety_override: bool
    routing_table_version: int
    rationale_en: str
    rationale_es: str
    all_candidate_drivers: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "action": self.action,
            "secondary_action": self.secondary_action,
            "top_driver": self.top_driver,
            "driver_value": round(self.driver_value, 6),
            "used_tie_break_hierarchy": self.used_tie_break_hierarchy,
            "requires_human_review": self.requires_human_review,
            "is_safety_override": self.is_safety_override,
            "routing_table_version": self.routing_table_version,
            "rationale": {
                "en": self.rationale_en,
                "es": self.rationale_es,
            },
            "candidate_drivers": [
                {"driver": d, "value": round(v, 6)} for d, v in self.all_candidate_drivers
            ],
        }


class RoutingRuleEngine:
    """Loads routing_table.yaml and resolves patient driver profiles to actions."""

    def __init__(self, table_path: Path = DEFAULT_ROUTING_TABLE_PATH):
        self.table_path = Path(table_path)
        with open(self.table_path, "r", encoding="utf-8") as f:
            self.table = yaml.safe_load(f)

        self.version: int = self.table["version"]
        self.tie_break_margin: float = self.table["tie_break_margin"]
        self.clinical_hierarchy: List[str] = self.table["clinical_hierarchy"]
        self.driver_fallbacks: Dict[str, str] = self.table.get("driver_fallbacks", {}) or {}
        self.safety_overrides: Dict[str, dict] = self.table.get("safety_overrides", {}) or {}
        self.drivers: Dict[str, dict] = self.table["drivers"]

        self._validate_table()

    def _validate_table(self) -> None:
        for name, cfg in self.drivers.items():
            if cfg["action"] not in VALID_ACTIONS:
                raise ValueError(
                    f"routing_table.yaml: driver '{name}' maps to invalid "
                    f"action '{cfg['action']}'. Must be one of {VALID_ACTIONS}."
                )
        for name, cfg in self.safety_overrides.items():
            if cfg["action"] not in VALID_ACTIONS:
                raise ValueError(
                    f"routing_table.yaml: safety_override '{name}' maps to "
                    f"invalid action '{cfg['action']}'."
                )
        for hierarchy_driver in self.clinical_hierarchy:
            if hierarchy_driver not in self.drivers:
                raise ValueError(
                    f"routing_table.yaml: clinical_hierarchy references "
                    f"unknown driver '{hierarchy_driver}'."
                )

    def _apply_fallback_remap(
        self, ranked: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Remap out-of-scope drivers (e.g. regimen_complexity) to their
        fallback driver, merging attribution values if both are present."""
        remapped: Dict[str, float] = {}
        for driver, value in ranked:
            target = self.driver_fallbacks.get(driver, driver)
            remapped[target] = remapped.get(target, 0.0) + value
        return sorted(remapped.items(), key=lambda kv: kv[1], reverse=True)

    def _check_safety_override(
        self, profile: PatientDriverProfile
    ) -> Optional[RoutingDecision]:
        for override_driver, cfg in self.safety_overrides.items():
            value = profile.driver_attributions.get(override_driver, 0.0)
            if value > 0:
                return RoutingDecision(
                    patient_id=profile.patient_id,
                    action=cfg["action"],
                    secondary_action=cfg.get("secondary_action"),
                    top_driver=override_driver,
                    driver_value=value,
                    used_tie_break_hierarchy=False,
                    requires_human_review=bool(cfg.get("requires_human_review", False)),
                    is_safety_override=True,
                    routing_table_version=self.version,
                    rationale_en=cfg["rationale_en"],
                    rationale_es=cfg["rationale_es"],
                    all_candidate_drivers=profile.ranked_modifiable_drivers(),
                )
        return None

    def route_patient(self, profile: PatientDriverProfile) -> RoutingDecision:
        # Step 1: hard safety override, bypasses everything else.
        override_decision = self._check_safety_override(profile)
        if override_decision is not None:
            return override_decision

        # Step 2: rank candidates, apply out-of-scope fallback remap.
        ranked = profile.ranked_modifiable_drivers()
        # trauma_exposure never reaches here with positive value (caught above),
        # but strip it defensively in case attribution is exactly 0 or negative.
        ranked = [(d, v) for d, v in ranked if d != "trauma_exposure"]
        ranked = self._apply_fallback_remap(ranked)

        if not ranked:
            # No positive modifiable driver at all — default to lowest-risk
            # touchpoint. This is a deliberate, logged default rather than
            # a silent fallback.
            return self._build_decision(
                profile,
                resolved_driver="bp_trend",
                resolved_value=0.0,
                ranked=ranked,
                used_hierarchy=True,
            )

        # Step 3: tie-break margin check.
        top_driver, top_value = ranked[0]
        used_hierarchy = False
        resolved_driver, resolved_value = top_driver, top_value

        if len(ranked) > 1:
            _, second_value = ranked[1]
            beats_margin = (
                second_value <= 0
                or top_value >= self.tie_break_margin * second_value
            )
            if not beats_margin:
                used_hierarchy = True
                candidate_set = {d for d, _ in ranked}
                for hierarchy_driver in self.clinical_hierarchy:
                    if hierarchy_driver in candidate_set:
                        resolved_driver = hierarchy_driver
                        resolved_value = dict(ranked)[hierarchy_driver]
                        break

        return self._build_decision(
            profile,
            resolved_driver=resolved_driver,
            resolved_value=resolved_value,
            ranked=ranked,
            used_hierarchy=used_hierarchy,
        )

    def _build_decision(
        self,
        profile: PatientDriverProfile,
        resolved_driver: str,
        resolved_value: float,
        ranked: List[Tuple[str, float]],
        used_hierarchy: bool,
    ) -> RoutingDecision:
        cfg = self.drivers[resolved_driver]
        rationale_en = cfg["rationale_en"].format(value=resolved_value, patient_id=profile.patient_id)
        rationale_es = cfg["rationale_es"].format(value=resolved_value, patient_id=profile.patient_id)

        if used_hierarchy:
            hierarchy_note_en = (
                " [Tie-break: top driver did not exceed 2nd-ranked driver by "
                f"{self.tie_break_margin}x margin; resolved via clinical hierarchy.]"
            )
            hierarchy_note_es = (
                " [Desempate: el factor principal no superó al segundo por el "
                f"margen de {self.tie_break_margin}x; resuelto mediante jerarquía clínica.]"
            )
            rationale_en += hierarchy_note_en
            rationale_es += hierarchy_note_es

        return RoutingDecision(
            patient_id=profile.patient_id,
            action=cfg["action"],
            secondary_action=cfg.get("secondary_action"),
            top_driver=resolved_driver,
            driver_value=resolved_value,
            used_tie_break_hierarchy=used_hierarchy,
            requires_human_review=False,
            is_safety_override=False,
            routing_table_version=self.version,
            rationale_en=rationale_en,
            rationale_es=rationale_es,
            all_candidate_drivers=ranked,
        )

    def route_cohort(
        self, profiles: List[PatientDriverProfile]
    ) -> List[RoutingDecision]:
        return [self.route_patient(p) for p in profiles]
