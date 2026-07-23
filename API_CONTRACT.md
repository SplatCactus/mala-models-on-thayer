# BP Cascade RI — API Contract (for the frontend rebuild)

This is the complete contract for the worklist + escalation API. If you are
building the UI, you can build against this document alone — no other context is
assumed. The API is **read-only** (`GET` only), CORS-open, and serves JSON.

Base URL in dev: `http://localhost:8000` (run:
`./venv/bin/python -m uvicorn src.api.main:app --reload --port 8000`).

**Conventions**
- All money/score numbers are JSON numbers; dates are ISO `YYYY-MM-DD`; timestamps
  are ISO 8601 (often UTC `...Z`).
- Multilingual text always appears as four sibling keys or a `{en,es,pt,ht}` object.
  Languages are always exactly `en`, `es`, `pt`, `ht`.
- **Provider-only:** every dispatch is addressed to a provider/organization. There
  is no patient recipient anywhere; `addressed_to` is always
  `"provider_or_organization"`. A "read-aloud script" is a worker's tool, never a
  message the system sends.
- **Graceful degradation:** if the escalation/pharmacy snapshots don't exist yet,
  `escalation` is `null` per row and `escalation_funnel` / `pharmacy_source` are
  `null` at the top level. Every existing field still returns. Guard for `null`.
- **"Always present" vs "conditional"** is called out per field below.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/worklist` | Full worklist: top-level summary + every capped patient row (each with an `escalation` block). Rows sorted by `break_window_start`. |
| GET | `/worklist/{patient_id}` | One patient row (same shape as a `/worklist` row). 404 if not in the capped worklist. |
| GET | `/escalation/{patient_id}` | The raw, complete escalation record for one patient (richer than the row's `escalation` block). 404 if no escalation state for that patient. |
| GET | `/escalation/summary` | Funnel counts + active pharmacy source, for a dashboard header. Never 404s. |
| GET | `/health` | `{"status":"ok","source":"data/snapshots/routing_table.json"}` |

---

## `GET /worklist` — top level

| Field | Type | Always present? | Notes |
|---|---|---|---|
| `generated_at` | string (ISO ts) | yes | When the routing snapshot was built. |
| `data_source` | string | yes | e.g. `"synthetic (batch)"`, `"synthetic (escalation demo)"`. |
| `last_synced` | string (ISO ts) | yes | Falls back to `generated_at`. |
| `cohort_size` | int | yes | Eligible pool size (e.g. 16205). |
| `capacity` | int | yes | Sum of role caps used. |
| `role_caps_used` | object {role: int} | yes | e.g. `{"pharmacist":40,"social_worker":852,"chw_call":50}`. |
| `role_capacity_expansions` | object {role: int} | yes | Safety-driven cap expansions. |
| `escalation_funnel` | object \| null | conditional | `null` if no escalation state yet. Shape below. |
| `pharmacy_source` | object \| null | conditional | `null` if no pharmacy sync state yet. Shape below. |
| `worklist` | array<Row> | yes | Sorted by `break_window_start`. Row shape below. |

### `escalation_funnel`
```json
{
  "n_by_round": {"0": 90, "1": 852},
  "n_closed": 689,
  "n_closed_by_round": {"1": 688, "0": 1},
  "n_round2": 0,
  "n_gated": 249,
  "n_waiting_on_latency": 0,
  "n_unactionable_in_time": 0
}
```
`n_by_round` / `n_closed_by_round` keys are round numbers **as strings**. A round
absent from a map means zero.

### `pharmacy_source`
```json
{
  "name": "local_file_synthetic",
  "access_mode": "batch_permitted",
  "latency_profile": {"min": 0, "typical": 0, "max": 0},
  "confirms_dispense": true,
  "last_synced": "2026-07-23T05:47:54.789646Z",
  "fallback_trace": [
    {"adapter":"surescripts","source_name":"surescripts_medication_history",
     "access_mode":"prescriber_initiated","outcome":"auth_failed","reason":"..."},
    {"adapter":"ae_claims","source_name":"ae_pharmacy_claims_export",
     "access_mode":"batch_permitted","outcome":"auth_failed","reason":"..."},
    {"adapter":"local","source_name":"local_file_synthetic",
     "access_mode":"batch_permitted","outcome":"served","reason":null}
  ]
}
```
- `access_mode` ∈ `"batch_permitted" | "prescriber_initiated"`.
- `latency_profile` is in **days**; the demo's synthetic source is 0 (a real source
  is not — Surescripts 1–14d, AE claims 30–90d). Use it to show a "data lags ~N days"
  badge.
- `fallback_trace[*].outcome` ∈ `"served" | "auth_failed"`. Exactly one is `"served"`.

---

## `GET /worklist` — a Row

Every field below is **always present** unless marked conditional. Existing fields
(top of the table) keep the exact shape they had before escalation existed.

| Field | Type | Notes |
|---|---|---|
| `patient_id` | string (uuid) | Stable id; use for the detail endpoints. |
| `display_name` | string | `"Patient #<first 8 of id>"`. No real names exist. |
| `preferred_language` | string | Currently always `"en"` (no preference field yet). |
| `break_window_start` / `break_window_end` | string (date) | ±3-day display window around the predicted break. |
| `risk_score` | float 0–1 | Model P(gap). |
| `top_driver` | string | e.g. `"transport_barrier"`, `"bp_trend"`, `"trauma_exposure"`. |
| `driver_label_{en,es,pt,ht}` | string | Short localized driver label. |
| `routed_action` | string | `"pharmacist" \| "social_worker" \| "chw_call"` (the base routing action). |
| `dispatch_message_{en,es,pt,ht}` | string | **Provider-addressed** message for the current round. Renamed from `outreach_script_*`. |
| `outreach_script_{en,es,pt,ht}` | string | **DEPRECATED alias** of `dispatch_message_*` (same value). Do not build new UI on it. |
| `chw_read_aloud_script_{en,es,pt,ht}` | string \| null | The worker's read-aloud tool; `null` unless the current recipient is a CHW or prescriber. Never a message the system sends. |
| `requires_human_review` | bool | True for trauma safety overrides. |
| `is_safety_override` | bool | True for trauma safety overrides. |
| `priority_score` | float | Ranking score within role. |
| `loop_outcome` | object \| null | **Back-compat.** Objective outcome or `null`. `{observed, on_time_refill, event_date, source, refill_source, refill_latency_days}`. |
| `escalation` | object \| null | The escalation block (below). `null` if no escalation state for this patient. |

### Row `escalation` block
| Field | Type | Always present (when `escalation`≠null)? | Notes |
|---|---|---|---|
| `current_round` | int (0/1/2) | yes | 0 = CHW→Pharmacy, 1 = SDOH routing, 2 = Prescriber. |
| `round_label` | {en,es,pt,ht} | yes | Localized round label. |
| `status` | string enum | yes | See status enum below. |
| `days_remaining` | int \| null | yes | Days to the next boundary for the current round (measured against the state's clock). `null` when closed/exhausted. |
| `days_until_latency_clears` | int \| null | conditional | Set only when `status == "WAITING_ON_DATA_LATENCY"`. |
| `predicted_break_date` | string (date) | yes | Frozen at entry; does not drift with model re-scores. |
| `unactionable_in_time` | bool | yes | True when the confirming source's max latency ≥ the whole break runway (we can't confirm a refill before the break). |
| `consent_scopes` | object | yes | Two scopes, shape below. |
| `gated_actions` | array | yes | May be empty. Each: `{round, action, scope, reason, fallback_action}`. |
| `dispatch_history` | array | yes | One entry per round attempted so far; shape below. |
| `current_dispatch` | object \| null | conditional | Present once the wait has elapsed (a dispatch is due/sent); `null` while `WAITING`. Shape below. |
| `objective_outcome` | object \| null | yes | `null` until an objective outcome is observed; shape below. |

**Status enum** (`escalation.status`):
`"WAITING"` · `"WAIT_ELAPSED_DISPATCH_PENDING"` · `"DISPATCHED"` ·
`"WAITING_ON_DATA_LATENCY"` · `"GATED_ON_CONSENT"` · `"CLOSED"` · `"EXHAUSTED"`.

**`consent_scopes`** (both scopes always present):
```json
{
  "internal_care_coordination": {"scope":"internal_care_coordination","state":"granted",
    "as_of":"2026-06-15","source":"AE_intake (SYNTHETIC)","stale":false,"allowed":true},
  "external_disclosure": {"scope":"external_disclosure","state":"denied",
    "as_of":"2026-06-15","source":"AE_intake (SYNTHETIC)","stale":false,"allowed":false}
}
```
`state` ∈ `"granted" | "denied" | "unknown"` (unknown = status not received — NOT a
refusal). `allowed` is the effective decision (granted AND not stale).

**`dispatch_history[*]`**:
```json
{"round":0,"recipient_type":"ae_chw","recipient_label":{"en":"the patient's Community Health Worker (CHW)","es":"...","pt":"...","ht":"..."},
 "mediated_by":"pharmacy","dispatched_at":"2026-07-14T00:00:00Z","outcome":"no_refill",
 "body":{"en":"To the patient's Community Health Worker: ...","es":"...","pt":"...","ht":"..."}}
```
`recipient_type` ∈ `"ae_chw" | "social_worker" | "pharmacist" | "bilingual_chw" |
"transit_voucher_external" | "prescriber"`. `mediated_by` is the org the recipient
contacts on the patient's behalf (`"pharmacy"` for Round 0, `"transit_broker"` /
`"internal_transportation"` for transport, else `null`). `outcome` ∈ `"pending" |
"no_refill" | "refill_observed" | "gated" | "gated_internal"`. `dispatched_at` is
`null` if not yet dispatched.

**`current_dispatch`** (when present):
```json
{"recipient_type":"prescriber","recipient_label":{"en":"the patient's prescriber (within the AE)","...":"..."},
 "mediated_by":null,"addressed_to":"provider_or_organization",
 "body":{"en":"To the patient's prescriber (within the AE): ...","...":"..."},
 "read_aloud_script":{"en":"[READ-ALOUD SCRIPT -- the CHW's/prescriber's own tool ...]","...":"..."}}
```
`read_aloud_script` is `null` unless the recipient is a CHW or prescriber.
`addressed_to` is **always** `"provider_or_organization"`.

**`objective_outcome`** (when non-null):
```json
{"observed":true,"on_time_refill":true,"event_date":"2026-07-01","source":"local_file_synthetic",
 "refill_source":"local_file_synthetic","refill_latency_days":0,"closed_on_round":1}
```

---

## `GET /escalation/{patient_id}`

The **raw** escalation record (superset of the row block — includes per-round timer
fields the row omits). 404 with a `{"detail": "..."}` body if there is no state for
that id. Keys: `patient_id, entry_date, days_to_break_at_entry, predicted_break_date,
is_safety_override, source_name, source_max_latency_days, unactionable_in_time,
current_round, status, wait_elapsed, closed_on_round, objective_outcome, consent,
gated_actions, current_dispatch, rounds`. Each `rounds[*]` adds `wait_until`,
`escalate_at`, `effective_escalate_at`, `action`, `required_scope`, `gated`,
`consent_reason`, `fallback_action`, and the full `dispatch` payload.

## `GET /escalation/summary`
```json
{"generated_at":"...","today":"2026-07-23","demo_time_compression_days_per_tick":null,
 "escalation_funnel":{...},"pharmacy_source":{...}}
```
`demo_time_compression_days_per_tick` is non-null only when the escalation job ran
in demo time-compression mode.

---

## Complete example row per escalation `status`

Only the `escalation` block is shown; the surrounding row fields are as above.

**WAITING** (wait not yet elapsed — no dispatch shown):
```json
{"current_round":0,"round_label":{"en":"Round 0 — CHW → Pharmacy","es":"Ronda 0 — CHW → Farmacia","pt":"Ronda 0 — ACS → Farmácia","ht":"Faz 0 — CHW → Famasi"},
 "status":"WAITING","days_remaining":48,"days_until_latency_clears":null,
 "predicted_break_date":"2026-11-01","unactionable_in_time":false,
 "consent_scopes":{"internal_care_coordination":{"state":"granted","allowed":true,"stale":false,"as_of":"2026-06-15","source":"AE_intake (SYNTHETIC)","scope":"internal_care_coordination"},
   "external_disclosure":{"state":"granted","allowed":true,"stale":false,"as_of":"2026-06-15","source":"AE_intake (SYNTHETIC)","scope":"external_disclosure"}},
 "gated_actions":[],"dispatch_history":[{"round":0,"recipient_type":"ae_chw","recipient_label":{"en":"the patient's Community Health Worker (CHW)"},"mediated_by":"pharmacy","dispatched_at":null,"outcome":"pending","body":{"en":"To the patient's Community Health Worker: ..."}}],
 "current_dispatch":null,"objective_outcome":null}
```

**WAIT_ELAPSED_DISPATCH_PENDING** — `status:"WAIT_ELAPSED_DISPATCH_PENDING"`,
`current_dispatch` populated (Round 0 `ae_chw`), `dispatch_history[0].dispatched_at`
still `null` (recorded by the job on the next tick), `objective_outcome:null`.

**DISPATCHED** — same as pending but `dispatch_history[current_round].dispatched_at`
is a timestamp and `status:"DISPATCHED"`.

**WAITING_ON_DATA_LATENCY** (dispatched, holding for the source's latency before
escalating): `status:"WAITING_ON_DATA_LATENCY"`, `days_until_latency_clears: 22`,
`days_remaining: 22`, current round already dispatched.

**GATED_ON_CONSENT** (transport patient, external not authorized → CHW fallback):
```json
{"current_round":1,"round_label":{"en":"Round 1 — SDOH routing","...":"..."},
 "status":"WAIT_ELAPSED_DISPATCH_PENDING","days_remaining":15,"days_until_latency_clears":null,
 "predicted_break_date":"2026-09-10","unactionable_in_time":false,
 "consent_scopes":{"external_disclosure":{"state":"denied","allowed":false,"stale":false,"...":"..."},"internal_care_coordination":{"state":"granted","allowed":true,"...":"..."}},
 "gated_actions":[{"round":1,"action":"transit_voucher","scope":"external_disclosure","reason":"denied:external_disclosure","fallback_action":"chw_transport_support"}],
 "dispatch_history":[{"round":0,"recipient_type":"ae_chw","outcome":"no_refill","...":"..."},
   {"round":1,"recipient_type":"ae_chw","mediated_by":"internal_transportation","outcome":"pending","body":{"en":"To the patient's Community Health Worker: Patient #... has a transportation barrier ... NOT authorized for external disclosure ... arrange transportation support internally ..."}}],
 "current_dispatch":{"recipient_type":"ae_chw","addressed_to":"provider_or_organization","body":{"en":"..."},"read_aloud_script":{"en":"[READ-ALOUD SCRIPT ...]"}},
 "objective_outcome":null}
```
Note: for an **external** gate the top-level `status` stays an acting status and the
gate shows up in `gated_actions` with the internal `fallback_action`. Top-level
`status:"GATED_ON_CONSENT"` occurs when an **internal** action is unauthorized (a
hard block — nothing dispatched).

**CLOSED** (objective refill observed):
```json
{"current_round":1,"round_label":{"en":"Round 1 — SDOH routing","...":"..."},
 "status":"CLOSED","days_remaining":null,"days_until_latency_clears":null,
 "predicted_break_date":"2026-08-20","unactionable_in_time":false,
 "consent_scopes":{"...":"..."},"gated_actions":[],
 "dispatch_history":[{"round":0,"...":"..."},{"round":1,"outcome":"refill_observed","...":"..."}],
 "current_dispatch":{"...":"..."},
 "objective_outcome":{"observed":true,"on_time_refill":true,"event_date":"2026-07-01","source":"local_file_synthetic","refill_source":"local_file_synthetic","refill_latency_days":0,"closed_on_round":1}}
```

**EXHAUSTED** (confirmed break, or walked the whole ladder with no refill):
```json
{"current_round":2,"round_label":{"en":"Round 2 — Prescriber escalation","...":"..."},
 "status":"EXHAUSTED","days_remaining":null,"days_until_latency_clears":null,
 "predicted_break_date":"2026-07-20","unactionable_in_time":false,
 "consent_scopes":{"...":"..."},"gated_actions":[],
 "dispatch_history":[{"round":0,"outcome":"no_refill","...":"..."},{"round":1,"outcome":"no_refill","...":"..."},{"round":2,"recipient_type":"prescriber","outcome":"no_refill","body":{"en":"To the patient's prescriber (within the AE): ... Interventions already attempted: Round 0 (the patient's Community Health Worker (CHW)) — 2026-05-21 — no refill observed; Round 1 (...) — ... ."}}],
 "current_dispatch":{"recipient_type":"prescriber","...":"..."},
 "objective_outcome":{"observed":true,"on_time_refill":false,"event_date":"2026-07-02","source":"local_file_synthetic","refill_source":"local_file_synthetic","refill_latency_days":0,"closed_on_round":null}}
```

---

## Producing the data the API serves
```bash
python -m src.run_routing_pipeline      # -> routing_table.json (worklist + fairness)
python -m src.sync.loop_closure         # -> loop_outcomes.json + pharmacy_sync_state.json
python -m src.routing.escalation        # -> escalation_state.json (adds the escalation block)
# live demo (compressed time): python -m src.sync.escalation_job --simulate-days-per-tick 30 --max-ticks 6
```
If `escalation_state.json` / `pharmacy_sync_state.json` are absent, `/worklist` still
returns every existing field with `escalation: null` and the two top-level summaries
`null`. Build for that case first, then light up the escalation UI when present.
