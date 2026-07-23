"""API serving the CHW routing worklist + escalation state.

Serves data/snapshots/routing_table.json (produced by src/run_routing_pipeline.py)
translated into the flat row shape the frontend consumes, PLUS an additive
per-patient escalation block and top-level escalation/pharmacy-source summaries
sourced from data/snapshots/escalation_state.json and pharmacy_sync_state.json.

ADDITIVE BY CONTRACT (the frontend is being rebuilt in parallel)
----------------------------------------------------------------
Every field that existed before this change keeps its exact shape. New fields are
added, never renamed in place, and every new read uses ``.get()`` fallbacks so a
pre-migration snapshot (no escalation_state.json, older routing_table.json) still
serves without crashing. See API_CONTRACT.md for the full field-by-field contract.

RENAMED KEY (with a deprecated alias)
-------------------------------------
The old ``outreach_script_*`` keys implied patient-facing copy. The provider-
addressed dispatch message is now ``dispatch_message_*`` and the worker's read-aloud
tool is ``chw_read_aloud_script_*`` (distinct concerns). ``outreach_script_*`` is
kept as a DEPRECATED alias of ``dispatch_message_*`` so nothing in flight breaks.

DE-IDENTIFICATION (datathon demo requirement)
---------------------------------------------
display_name is "Patient #<id8>"; there is no name field anywhere in the pipeline.
preferred_language defaults to "en" (no language-preference field exists yet).

Run:  ./venv/bin/python -m uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]

WORKLIST_PATH = ROOT / "data" / "snapshots" / "routing_table.json"
# All optional + read-only; absent files degrade gracefully (see the loaders).
LOOP_OUTCOMES_PATH = ROOT / "data" / "snapshots" / "loop_outcomes.json"
ESCALATION_STATE_PATH = ROOT / "data" / "snapshots" / "escalation_state.json"
PHARMACY_SYNC_PATH = ROOT / "data" / "snapshots" / "pharmacy_sync_state.json"

app = FastAPI(title="BP Cascade RI — Worklist + Escalation API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)

# Short multilingual driver labels for the dashboard's "top driver" column. Keys
# match routing_table.yaml's drivers:/safety_overrides: keys exactly.
DRIVER_LABELS = {
    "housing_barrier": {"en": "Housing instability", "es": "Inestabilidad de vivienda",
                        "pt": "Instabilidade habitacional", "ht": "Enstabilite lojman"},
    "financial_barrier": {"en": "Financial barrier", "es": "Barrera financiera",
                          "pt": "Barreira financeira", "ht": "Baryè finansye"},
    "transport_barrier": {"en": "Transportation barrier", "es": "Barrera de transporte",
                          "pt": "Barreira de transporte", "ht": "Baryè transpò"},
    "isolation": {"en": "Social isolation", "es": "Aislamiento social",
                  "pt": "Isolamento social", "ht": "Izolman sosyal"},
    "low_education": {"en": "Health literacy / education", "es": "Alfabetización en salud",
                      "pt": "Literacia em saúde / educação", "ht": "Konesans sou sante / edikasyon"},
    "migrant_status": {"en": "Migrant status", "es": "Estatus migratorio",
                       "pt": "Estatuto migratório", "ht": "Estati imigran"},
    "bp_trend": {"en": "Blood pressure trend", "es": "Tendencia de presión arterial",
                 "pt": "Tendência da pressão arterial", "ht": "Tandans tansyon"},
    "trauma_exposure": {"en": "Trauma exposure (safety)", "es": "Exposición a trauma (seguridad)",
                        "pt": "Exposição a trauma (segurança)", "ht": "Ekspozisyon a chòk (sekirite)"},
}

# Round labels (UI copy belongs in the API layer, like DRIVER_LABELS).
ROUND_LABELS = {
    0: {"en": "Round 0 — CHW → Pharmacy", "es": "Ronda 0 — CHW → Farmacia",
        "pt": "Ronda 0 — ACS → Farmácia", "ht": "Faz 0 — CHW → Famasi"},
    1: {"en": "Round 1 — SDOH routing", "es": "Ronda 1 — Enrutamiento SDOH",
        "pt": "Ronda 1 — Encaminhamento SDOH", "ht": "Faz 1 — Woutaj SDOH"},
    2: {"en": "Round 2 — Prescriber escalation", "es": "Ronda 2 — Escalamiento al prescriptor",
        "pt": "Ronda 2 — Escalonamento ao prescritor", "ht": "Faz 2 — Eskalad bay preskriptè"},
}


def _read_json(path: Path) -> dict:
    """Read a JSON file, or {} if absent/unreadable (graceful-absent everywhere)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_loop_outcomes() -> dict:
    """Objective per-patient closure outcomes (patient_id -> outcome dict), or {}."""
    return _read_json(LOOP_OUTCOMES_PATH).get("outcomes", {})


def _load_escalation() -> tuple[dict, dict]:
    """(patients {pid -> state}, meta) from escalation_state.json, or ({}, {})."""
    payload = _read_json(ESCALATION_STATE_PATH)
    return payload.get("patients", {}), payload.get("meta", {})


def _pharmacy_source_view() -> dict | None:
    """Active source + access-mode + latency profile + last_synced + fallback trace."""
    sync = _read_json(PHARMACY_SYNC_PATH)
    if not sync:
        return None
    sel = sync.get("selected", {})
    return {
        "name": sel.get("source_name"),
        "access_mode": sel.get("access_mode"),
        "latency_profile": sel.get("latency_days", {
            "min": sel.get("min_latency_days"),
            "typical": sel.get("typical_latency_days"),
            "max": sel.get("max_latency_days"),
        }),
        "confirms_dispense": sel.get("confirms_dispense"),
        "last_synced": sync.get("last_synced"),
        "fallback_trace": sync.get("attempts", []),
    }


def _days_between(iso_from: str, iso_to: str) -> int | None:
    try:
        return (dt.date.fromisoformat(iso_to[:10]) - dt.date.fromisoformat(iso_from[:10])).days
    except (ValueError, TypeError):
        return None


def _escalation_block(esc: dict, ref_today: str) -> dict:
    """Build the additive per-row escalation block from one patient's state dict.

    ``ref_today`` is the date the escalation state reflects (its meta.today), so
    day-countdowns are measured against the state's clock, not the API host's.
    """
    cur = esc.get("current_round")
    status = esc.get("status")
    rounds = esc.get("rounds", [])
    cur_attempt = next((r for r in rounds if r.get("round") == cur), None)

    days_remaining = days_until_latency = None
    if cur_attempt and status not in ("CLOSED", "EXHAUSTED"):
        if status == "WAITING":
            days_remaining = _days_between(ref_today, cur_attempt.get("wait_until", ""))
        elif status == "WAITING_ON_DATA_LATENCY":
            days_until_latency = _days_between(ref_today, cur_attempt.get("effective_escalate_at", ""))
            days_remaining = days_until_latency
        else:  # DISPATCHED / WAIT_ELAPSED_DISPATCH_PENDING / GATED_ON_CONSENT
            days_remaining = _days_between(ref_today, cur_attempt.get("escalate_at", ""))

    dispatch_history = [{
        "round": r.get("round"),
        "recipient_type": r.get("dispatch", {}).get("recipient_type"),
        "recipient_label": r.get("dispatch", {}).get("recipient_label"),
        "mediated_by": r.get("dispatch", {}).get("mediated_by"),
        "dispatched_at": r.get("dispatched_at"),
        "outcome": r.get("outcome"),
        "body": r.get("dispatch", {}).get("body"),
    } for r in rounds]

    # current_dispatch is surfaced once the wait has elapsed (a dispatch is due/sent).
    cd = esc.get("current_dispatch") or {}
    current_dispatch = None
    if esc.get("wait_elapsed") and cd:
        current_dispatch = {
            "recipient_type": cd.get("recipient_type"),
            "recipient_label": cd.get("recipient_label"),
            "mediated_by": cd.get("mediated_by"),
            "addressed_to": cd.get("addressed_to"),
            "body": cd.get("body"),
            "read_aloud_script": cd.get("read_aloud_script"),
        }

    return {
        "current_round": cur,
        "round_label": ROUND_LABELS.get(cur, {}),
        "status": status,
        "days_remaining": days_remaining,
        "days_until_latency_clears": days_until_latency,
        "predicted_break_date": esc.get("predicted_break_date"),
        "unactionable_in_time": esc.get("unactionable_in_time", False),
        "consent_scopes": esc.get("consent", {}),
        "gated_actions": esc.get("gated_actions", []),
        "dispatch_history": dispatch_history,
        "current_dispatch": current_dispatch,
        "objective_outcome": esc.get("objective_outcome"),
    }


def _escalation_funnel(esc_meta: dict) -> dict | None:
    """Top-level funnel counts, or None if no escalation state exists yet."""
    if not esc_meta:
        return None
    by_round = esc_meta.get("n_by_current_round", {})
    by_status = esc_meta.get("n_by_status", {})
    return {
        "n_by_round": by_round,
        "n_closed": esc_meta.get("n_closed", 0),
        "n_closed_by_round": esc_meta.get("n_closed_by_round", {}),
        "n_round2": by_round.get("2", 0),
        "n_gated": esc_meta.get("n_gated_on_consent_or_fallback", 0),
        "n_waiting_on_latency": by_status.get("WAITING_ON_DATA_LATENCY", 0),
        "n_unactionable_in_time": esc_meta.get("n_unactionable_in_time", 0),
    }


def _to_dashboard_view(payload: dict) -> dict:
    """Translate worklist_builder.py's payload into the flat row shape + escalation."""
    today = dt.date.today()
    outcomes = _load_loop_outcomes()
    esc_patients, esc_meta = _load_escalation()
    # Countdowns are measured against the escalation state's own clock (its meta.today),
    # which may be a simulated date under the demo time-compression job.
    ref_today = esc_meta.get("today", today.isoformat())

    rows = []
    for card in payload["capped_worklist"]:
        dtb = card["days_to_predicted_break"]
        start = today + dt.timedelta(days=max(dtb - 3, 0))
        end = today + dt.timedelta(days=dtb + 3)
        driver = card["top_driver"]
        labels = DRIVER_LABELS.get(driver, {"en": driver, "es": driver, "pt": driver, "ht": driver})
        script = card["script"]
        pid = card["patient_id"]
        esc = esc_patients.get(pid)

        # Provider-addressed dispatch message: the current round's body when we have
        # escalation state; otherwise fall back to the routing rationale (pre-migration).
        cd = (esc or {}).get("current_dispatch") or {}
        dispatch_msg = cd.get("body") or {
            "en": script["en"]["rationale"], "es": script["es"]["rationale"],
            "pt": script.get("pt", script["en"])["rationale"],
            "ht": script.get("ht", script["en"])["rationale"],
        }
        read_aloud = cd.get("read_aloud_script")

        row = {
            # ---- existing fields (unchanged shape) ----
            "patient_id": pid,
            "display_name": f"Patient #{pid[:8]}",
            "preferred_language": "en",
            "break_window_start": start.isoformat(),
            "break_window_end": end.isoformat(),
            "risk_score": card["predicted_risk"],
            "top_driver": driver,
            "driver_label_en": labels["en"],
            "driver_label_es": labels["es"],
            "driver_label_pt": labels.get("pt", labels["en"]),
            "driver_label_ht": labels.get("ht", labels["en"]),
            "routed_action": card["action"],
            # DEPRECATED alias of dispatch_message_* (kept so nothing in flight breaks):
            "outreach_script_en": dispatch_msg["en"],
            "outreach_script_es": dispatch_msg["es"],
            "outreach_script_pt": dispatch_msg.get("pt", dispatch_msg["en"]),
            "outreach_script_ht": dispatch_msg.get("ht", dispatch_msg["en"]),
            "requires_human_review": card["requires_human_review"],
            "is_safety_override": card["is_safety_override"],
            "priority_score": card["priority_score"],
            "loop_outcome": outcomes.get(pid),
            # ---- new: provider-addressed dispatch message (renamed from outreach_script_*) ----
            "dispatch_message_en": dispatch_msg["en"],
            "dispatch_message_es": dispatch_msg["es"],
            "dispatch_message_pt": dispatch_msg.get("pt", dispatch_msg["en"]),
            "dispatch_message_ht": dispatch_msg.get("ht", dispatch_msg["en"]),
            # ---- new: the worker's read-aloud script (null unless CHW/prescriber) ----
            "chw_read_aloud_script_en": (read_aloud or {}).get("en"),
            "chw_read_aloud_script_es": (read_aloud or {}).get("es"),
            "chw_read_aloud_script_pt": (read_aloud or {}).get("pt"),
            "chw_read_aloud_script_ht": (read_aloud or {}).get("ht"),
            # ---- new: full escalation block (null if no escalation state yet) ----
            "escalation": _escalation_block(esc, ref_today) if esc else None,
        }
        rows.append(row)

    return {
        "generated_at": payload["meta"]["generated_at"],
        "data_source": payload["meta"].get("data_source", "unknown"),
        "last_synced": payload["meta"].get("last_synced", payload["meta"]["generated_at"]),
        "cohort_size": payload["meta"]["cohort_size"],
        "capacity": sum(payload["meta"]["role_caps_used"].values()),
        "role_caps_used": payload["meta"]["role_caps_used"],
        "role_capacity_expansions": payload["meta"]["role_capacity_expansions"],
        # ---- new top-level: escalation funnel + active pharmacy source ----
        "escalation_funnel": _escalation_funnel(esc_meta),
        "pharmacy_source": _pharmacy_source_view(),
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
    """Full worklist: metadata + escalation funnel + pharmacy source + every flagged
    patient (each with an additive `escalation` block), ranked by break window."""
    data = load_worklist()
    data["worklist"] = sorted(data["worklist"], key=lambda p: p["break_window_start"])
    return data


@app.get("/worklist/{patient_id}")
def get_patient(patient_id: str):
    """Single patient detail (same enriched row shape), for a click-through view."""
    data = load_worklist()
    for patient in data["worklist"]:
        if patient["patient_id"] == patient_id:
            return patient
    raise HTTPException(status_code=404, detail=f"patient not found: {patient_id}")


@app.get("/escalation/summary")
def get_escalation_summary():
    """Funnel counts + active pharmacy source, for a dashboard header.

    Declared BEFORE /escalation/{patient_id} so the literal 'summary' path wins over
    the path parameter (FastAPI matches routes in definition order)."""
    _, esc_meta = _load_escalation()
    return {
        "generated_at": esc_meta.get("generated_at"),
        "today": esc_meta.get("today"),
        "demo_time_compression_days_per_tick": esc_meta.get("demo_time_compression_days_per_tick"),
        "escalation_funnel": _escalation_funnel(esc_meta),
        "pharmacy_source": _pharmacy_source_view(),
    }


@app.get("/escalation/{patient_id}")
def get_escalation(patient_id: str):
    """Full escalation record for one patient (the raw state, incl. per-round timers,
    dispatch history with message bodies, consent scopes, and objective outcome)."""
    patients, _ = _load_escalation()
    esc = patients.get(patient_id)
    if esc is None:
        raise HTTPException(
            status_code=404,
            detail=f"no escalation state for patient {patient_id} (run "
                   f"`python -m src.routing.escalation` after producing the worklist).",
        )
    return esc


@app.get("/health")
def health():
    return {"status": "ok", "source": str(WORKLIST_PATH.relative_to(ROOT))}
