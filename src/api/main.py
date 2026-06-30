"""Mock API serving the CHW routing worklist.

This serves data/mock/mock_routing.json so the dashboard can be built and
tested before Annie's real routing pipeline exists. Swapping mock -> real
later is a one-line change: point WORKLIST_PATH at the real output file.

Run:  ./venv/bin/python -m uvicorn src.api.main:app --reload --port 8000
Then visit http://localhost:8000/worklist in a browser or curl it.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]

# Swap this single line once Annie's real routing table exists, e.g.:
#   WORKLIST_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
WORKLIST_PATH = ROOT / "src" / "api" / "mock_data" / "mock_routing.json"

app = FastAPI(title="BP Cascade RI — Worklist API")

# Allow the dashboard (likely running on a different port during dev) to call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_worklist() -> dict:
    if not WORKLIST_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"worklist file not found: {WORKLIST_PATH}",
        )
    with open(WORKLIST_PATH) as f:
        return json.load(f)


@app.get("/worklist")
def get_worklist():
    """Full worklist: metadata + every flagged patient, ranked by break window."""
    data = load_worklist()
    # sort by nearest break_window_start so the dashboard doesn't have to
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
