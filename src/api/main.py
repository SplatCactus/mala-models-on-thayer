"""API serving the CHW routing worklist.

Serves data/snapshots/routing_table.json (produced by
src/run_routing_pipeline.py) -- swapped over from the hand-written
src/api/mock_data/mock_routing.json now that the real pipeline exists,
per the comment this file used to carry pointing at exactly this path.

SCHEMA ADAPTER (2026-07-04)
----------------------------
src/worklist_builder.py's output schema (meta/capped_worklist/
eligible_pool) is the source of truth for every consumer -- fairness.py
in particular needs eligible_pool vs. capped_worklist kept separate, and
priority_score/rank_in_role/is_safety_override for audit purposes. That
schema does NOT match what dashboard.html's JS expects (a flat `worklist`
array with display_name/break_window_start/break_window_end/risk_score/
routed_action/etc, plus top-level cohort_size/capacity).

Decision: adapt here, in the API layer, rather than reshaping
worklist_builder.py's output or dashboard.html's expectations. Reasoning:
worklist_builder.py's schema carries information other consumers need
that the flat dashboard shape would throw away; dashboard.html is
UI-only display logic with no other downstream consumer, so it's the
cheaper and lower-risk side to adapt. See _to_dashboard_view below.

DE-IDENTIFICATION (datathon demo requirement, confirmed 2026-07-04)
----------------------------------------------------------------------
display_name is set to "Patient #<patient_id>" -- there is no real name
anywhere in this pipeline (RACE/ETHNICITY/etc. are demographic, not
identity, fields, and are deliberately held aside from modeling; no name
field is ever read from cohort.py). preferred_language defaults to "en"
for every row, since no real language-preference field exists in the
pipeline yet -- this is a known simplification, not a claim that every
patient's actual preferred language is English.

Run:  ./venv/bin/python -m uvicorn src.api.main:app --reload --port 8000
Then visit http://localhost:8000/worklist in a browser or curl it.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]

WORKLIST_PATH = ROOT / "data" / "snapshots" / "routing_table.json"

app = FastAPI(title="BP Cascade RI — Worklist API")

# Allow the dashboard (likely running on a different port during dev) to call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Short bilingual display labels per driver, for the dashboard's "top
# driver" column. routing_table.yaml only carries full audit-rationale
# sentences (rationale_en/es), not short labels, so these are defined
# here rather than invented in routing_table.yaml (which is meant to stay
# a clinical-rules document, not a UI-copy document). Keys must match
# routing_table.yaml's `drivers:` + `safety_overrides:` keys exactly.
DRIVER_LABELS = {
    "housing_barrier": {"en": "Housing instability", "es": "Inestabilidad de vivienda"},
    "financial_barrier": {"en": "Financial barrier", "es": "Barrera financiera"},
    "transport_barrier": {"en": "Transportation barrier", "es": "Barrera de transporte"},
    "isolation": {"en": "Social isolation", "es": "Aislamiento social"},
    "low_education": {"en": "Health literacy / education", "es": "Alfabetización en salud"},
    "migrant_status": {"en": "Migrant status", "es": "Estatus migratorio"},
    "bp_trend": {"en": "Blood pressure trend", "es": "Tendencia de presión arterial"},
    "trauma_exposure": {"en": "Trauma exposure (safety)", "es": "Exposición a trauma (seguridad)"},
}


def _to_dashboard_view(payload: dict) -> dict:
    """Translate worklist_builder.py's payload into dashboard.html's flat shape."""
    today = dt.date.today()
    rows = []
    for card in payload["capped_worklist"]:
        dtb = card["days_to_predicted_break"]
        # break_window_start/end: a +/- 3-day window around the predicted
        # break day, purely for display -- the model outputs a point
        # estimate (see run_routing_pipeline.py's days_to_predicted_break
        # derivation note), not a start/end range.
        start = today + dt.timedelta(days=max(dtb - 3, 0))
        end = today + dt.timedelta(days=dtb + 3)
        driver = card["top_driver"]
        labels = DRIVER_LABELS.get(driver, {"en": driver, "es": driver})

        rows.append({
            "patient_id": card["patient_id"],
            "display_name": f"Patient #{card['patient_id'][:8]}",
            "preferred_language": "en",
            "break_window_start": start.isoformat(),
            "break_window_end": end.isoformat(),
            "risk_score": card["predicted_risk"],
            "top_driver": driver,
            "driver_label_en": labels["en"],
            "driver_label_es": labels["es"],
            "routed_action": card["action"],
            "outreach_script_en": card["script"]["en"]["rationale"],
            "outreach_script_es": card["script"]["es"]["rationale"],
            "requires_human_review": card["requires_human_review"],
            "is_safety_override": card["is_safety_override"],
            "priority_score": card["priority_score"],
        })

    return {
        "generated_at": payload["meta"]["generated_at"],
        "cohort_size": payload["meta"]["cohort_size"],
        "capacity": sum(payload["meta"]["role_caps_used"].values()),
        "role_caps_used": payload["meta"]["role_caps_used"],
        "role_capacity_expansions": payload["meta"]["role_capacity_expansions"],
        "worklist": rows,
    }


def load_worklist() -> dict:
    if not WORKLIST_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"worklist file not found: {WORKLIST_PATH}. Run "
                   f"`python -m src.run_routing_pipeline` first.",
        )
    with open(WORKLIST_PATH) as f:
        payload = json.load(f)
    return _to_dashboard_view(payload)


@app.get("/worklist")
def get_worklist():
    """Full worklist: metadata + every flagged patient, ranked by break window."""
    data = load_worklist()
    data["worklist"] = sorted(data["worklist"], key=lambda p: p["break_window_start"])
    return data


@app.get("/worklist/{patient_id}")
def get_patient(patient_id: str):
    """Single patient detail, for a click-through / detail view."""
    data = load_worklist()
    for patient in data["worklist"]:
        if patient["patient_id"] == patient_id:
            return patient
    raise HTTPException(status_code=404, detail=f"patient not found: {patient_id}")


@app.get("/health")
def health():
    return {"status": "ok", "source": str(WORKLIST_PATH.relative_to(ROOT))}
    
