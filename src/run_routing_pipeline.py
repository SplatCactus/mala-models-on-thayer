"""
src/run_routing_pipeline.py

End-to-end glue for today's checklist: "generate the final SHAP-based
routing table" and "push the final routing table (JSON)". Ties together
five modules that each work in isolation but were never wired into one
pipeline before this:

    src/models/common.py       (PANEL_PATH / LABELS_PATH)
    src/explain/shap_runner.py (SHAPRunner)
    src/routing/rules.py       (RoutingRuleEngine)
    src/routing/capacity.py    (CapacityEngine)
    src/worklist_builder.py    (build_worklist_payload)
    src/eval/fairness.py       (compute_fairness_report)

Run
---
    ./venv/bin/python -m src.run_routing_pipeline

Writes data/snapshots/routing_table.json (served by src/api/main.py) and
data/snapshots/fairness_report.json (mic-drop-slide numbers).

RISK-LABEL POLARITY (confirmed by Chris, 2026-07-04)
-----------------------------------------------------
The model's predicted_risk must be "higher number = higher risk," using
the standard 0.80 PDC adherence threshold to define the at-risk positive
class: y=1 iff pdc_180d < 0.80 (below the CMS/PQA adherence cutoff).

This is the OPPOSITE polarity from classifier.py's own default
(`load_classification_frame`'s default binarizes pdc_180d >= 0.80 as the
positive class, i.e. y=1 there means *adherent*). We deliberately do NOT
call that default here -- see `build_risk_label` below, which builds the
label directly from pdc_180d with the confirmed at-risk-positive polarity.
See also the matching note on SHAPRunner.fit() in shap_runner.py.

DAYS-TO-PREDICTED-BREAK (new derivation -- not computed anywhere else)
------------------------------------------------------------------------
capacity.py's CapacityEngine.build() requires a `days_to_break` value per
patient, but nothing upstream produces one -- the model predicts a
180-day forward-window risk *probability* (pdc.py's outcome window), not
a specific break date. Absent a real survival/hazard model (survival.py
exists but is superseded per MODEL_CARD.md, and isn't fit for the current
n), we use a documented, reviewable linear proxy:

    days_to_predicted_break = (1 - predicted_risk) * FORWARD_WINDOW_DAYS

A patient at 90% predicted risk is assumed ~18 days out; a patient at 10%
risk ~162 days out, both within the model's own 180-day forward window.
This is a proxy, not a clinical estimate, and should be replaced by a
real hazard curve once the cohort is large enough to support one -- same
caveat MODEL_CARD.md already logs for survival.py's duration proxy. It is
the single most important open modeling assumption in this pipeline and
should be called out as such wherever break-window dates are shown.

ETHNICITY CASING FIX + DEMOGRAPHICS JOIN
------------------------------------------
src/eval/fairness.py and calibration.py default to a lowercase
`group_col="ethnicity"`, but src/etl/cohort.py emits uppercase
`ETHNICITY` (held aside from model features on purpose, per
src/models/common.py's HELD_ASIDE_DEMOGRAPHIC_COLUMNS -- it's excluded
from X, not from the feature panel file itself). Nothing previously
joined demographics back onto routing output for fairness reporting at
all, so this would have KeyError'd the moment fairness.py actually ran.
Fixed at the single join site in `_compute_fairness` below (renames
ETHNICITY -> ethnicity once) rather than changing fairness.py's or
calibration.py's public default -- that keeps their API the same for any
other caller that already expects "ethnicity" lowercase.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.common import PANEL_PATH, LABELS_PATH  # noqa: E402
from src.explain.shap_runner import SHAPRunner, ALL_MODEL_FEATURES  # noqa: E402
from src.routing.rules import RoutingRuleEngine  # noqa: E402
from src.routing.capacity import CapacityEngine  # noqa: E402
from src.worklist_builder import build_worklist_payload  # noqa: E402
from src.eval.fairness import compute_fairness_report, FairnessReport  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
FAIRNESS_OUTPUT_PATH = ROOT / "data" / "snapshots" / "fairness_report.json"

# Chris-confirmed: below this pdc_180d cutoff = "at risk" (y=1).
PDC_RISK_THRESHOLD = 0.80
# Matches pdc.py's forward outcome window -- see module docstring re:
# days_to_predicted_break derivation.
FORWARD_WINDOW_DAYS = 180

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("run_routing_pipeline")


def load_panel_and_labels(
    panel_path: Path = PANEL_PATH, labels_path: Path = LABELS_PATH
) -> pd.DataFrame:
    """Load + inner-join the feature panel and labels on patient_id.

    Same join key/shape classifier.py's load_classification_frame uses,
    duplicated here (not imported) because we need pdc_180d itself for
    the at-risk label, not classifier.py's own (opposite-polarity)
    binarization of it.
    """
    panel = pd.read_parquet(panel_path)
    labels = pd.read_parquet(labels_path)
    frame = panel.merge(labels, on="patient_id", how="inner")

    n_before = len(frame)
    frame = frame.dropna(subset=["pdc_180d"]).reset_index(drop=True)
    n_dropped = n_before - len(frame)
    if n_dropped:
        log.warning(
            "  dropping %d/%d patient(s) with unlabeled pdc_180d", n_dropped, n_before
        )
    if frame.empty:
        raise ValueError("no patients with a labeled pdc_180d outcome -- nothing to route")
    return frame


def build_risk_label(frame: pd.DataFrame) -> np.ndarray:
    """At-risk-positive binary label: y=1 iff pdc_180d < PDC_RISK_THRESHOLD.

    See module docstring "RISK-LABEL POLARITY" -- this is deliberately the
    opposite of classifier.py's default binarization.
    """
    return (frame["pdc_180d"].to_numpy() < PDC_RISK_THRESHOLD).astype(int)


def _compute_fairness(
    frame: pd.DataFrame,
    decisions_df: pd.DataFrame,
    capped_ids: set,
    predicted_risk: Dict[str, float],
    days_to_break: Dict[str, float],
    y_risk: np.ndarray,
) -> FairnessReport:
    if "ETHNICITY" not in frame.columns:
        raise KeyError(
            "feature panel has no ETHNICITY column -- cannot compute fairness "
            "strata. Confirm src/etl/cohort.py's demographic columns survived "
            "the build_features.py merge."
        )

    demo = frame[["patient_id", "ETHNICITY"]].rename(columns={"ETHNICITY": "ethnicity"})
    demo["ethnicity"] = demo["ethnicity"].fillna("unknown")

    eligible_pool_df = decisions_df.merge(demo, on="patient_id", how="left")
    capped_worklist_df = eligible_pool_df[eligible_pool_df["patient_id"].isin(capped_ids)]

    scored_cohort_df = demo.copy()
    scored_cohort_df["predicted_risk"] = scored_cohort_df["patient_id"].map(predicted_risk)
    scored_cohort_df["days_to_predicted_break"] = scored_cohort_df["patient_id"].map(days_to_break)
    scored_cohort_df["actual_discontinued"] = y_risk

    return compute_fairness_report(
        eligible_pool_df, capped_worklist_df, scored_cohort_df, group_col="ethnicity"
    )


def run(
    panel_path: Path = PANEL_PATH,
    labels_path: Path = LABELS_PATH,
    role_caps: Optional[Dict[str, int]] = None,
    routing_table_path: Optional[Path] = None,
) -> tuple[dict, FairnessReport]:
    log.info("Loading feature panel + labels...")
    frame = load_panel_and_labels(panel_path, labels_path)
    log.info("  %d patients loaded", len(frame))

    missing = [c for c in ALL_MODEL_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(
            f"feature panel is missing expected model columns: {missing} -- "
            "check that src/features/build_features.py was re-run against "
            "the intended (1K vs 300K) parquet source."
        )

    y_risk = build_risk_label(frame)
    log.info(
        "  %d/%d patients at risk (pdc_180d < %.2f)",
        int(y_risk.sum()), len(y_risk), PDC_RISK_THRESHOLD,
    )

    log.info("Fitting SHAPRunner (linear risk model, at-risk-positive polarity)...")
    runner = SHAPRunner()
    runner.fit(frame, y_risk)
    profiles = runner.explain_cohort(frame)
    log.info("  explained %d patients", len(profiles))

    log.info("Routing cohort via routing_table.yaml...")
    engine = (
        RoutingRuleEngine(table_path=routing_table_path)
        if routing_table_path
        else RoutingRuleEngine()
    )
    decisions = engine.route_cohort(profiles)

    days_to_break = {
        p.patient_id: (1.0 - p.predicted_risk) * FORWARD_WINDOW_DAYS for p in profiles
    }
    predicted_risk = {p.patient_id: p.predicted_risk for p in profiles}

    log.info("Applying capacity caps...")
    cap_engine = CapacityEngine(role_caps=role_caps)
    capacity_result = cap_engine.build(decisions, days_to_break, predicted_risk)
    for role, cap in cap_engine.role_caps.items():
        log.info("  %-14s cap=%d", role, cap)
    if capacity_result.role_expansions:
        log.info("  safety-driven cap expansions: %s", capacity_result.role_expansions)

    log.info("Building worklist payload...")
    payload = build_worklist_payload(capacity_result, routing_table_version=engine.version)

    log.info("Computing fairness report (ethnicity strata)...")
    decisions_df = pd.DataFrame([d.to_dict() for d in decisions])
    capped_ids = {e.patient_id for e in capacity_result.capped_worklist}
    fairness_report = _compute_fairness(
        frame, decisions_df, capped_ids, predicted_risk, days_to_break, y_risk
    )

    return payload, fairness_report


def main() -> int:
    payload, fairness_report = run()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s", OUTPUT_PATH)

    with open(FAIRNESS_OUTPUT_PATH, "w") as f:
        json.dump(fairness_report.to_dict(), f, indent=2)
    log.info("Wrote %s", FAIRNESS_OUTPUT_PATH)

    log.info(
        "min/max selection-rate disparity ratio: %.4f (%s)",
        fairness_report.min_max_disparity_ratio,
        "FLAGGED (<0.8, 80%% rule)"
        if fairness_report.min_max_disparity_ratio < 0.8
        else "within 80%% rule",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
