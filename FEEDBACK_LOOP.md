# Closed-Loop Feedback Architecture (States 1–4)

How a model risk score becomes a patient actually getting their medication —
and how that outcome flows back into the model. This is the answer to *"how does
routing someone actually result in them getting their meds?"*

```
State 1 Routed ──▶ State 2 Acknowledged ──▶ State 3 Actioned ──▶ State 4 Outcome Observed
 (model)             (CHW opens record)        (intervention logged)   (objective refill)
                                                                              │
                                                                              ▼
                                                              retraining label (State 4 → model)
```

## The four states

| State | Trigger | Where it lives | Code |
|---|---|---|---|
| **1 Routed** | Model flags gap risk + top SHAP driver → one CHW action | `routing_table.json` (server) | `run_routing_pipeline.py` → `routing/rules.py` |
| **2 Acknowledged** | CHW/social worker opens the record | Dashboard **localStorage** | `dashboard.html` status dropdown |
| **3 Actioned** | Worker logs a concrete RI intervention (+ ref #) | Dashboard **localStorage** | "Mark actioned" modal |
| **4 Outcome Observed** | System detects an **objective on-time refill** in the predicted window | `loop_outcomes.json` (server) | `sync/loop_closure.py` |

**Design rule (non-negotiable):** the loop closes **only** on the objective
refill signal — **never** on self-reported patient feedback. States 1 & 4 are
server-side/objective; States 2 & 3 are the CHW's subjective progress and are
client-side for this demo (no backend by design — swap localStorage for a real
datastore to persist and to gate retraining on State 3).

## Driver → RI intervention → objective closure

When a patient is routed, their top modifiable driver maps to a concrete Rhode
Island mechanism. The **"Mark actioned" modal records these manually** today;
the option keys are stable so a real adapter can slot in later.

| Driver | RI mechanism (intervention) | Real integration surface (verified Jul 2026) |
|---|---|---|
| Transport barrier | **MTM** NEMT ride (Medicaid) or **RIPTA** bus pass | MTM is the statewide broker (**not** Modivcare). Portal + phone **1-855-330-9131** (facility line 855-330-9133); **no public API**. RIPTA ½-mile rule → pass instead of door-to-door. Manual. |
| Language / literacy | Bilingual CHW teach-back | **CPO-HEZ / ONE Neighborhood Builders CHW Collective**, embedded at **PCHC** & **Clínica Esperanza** (Spanish, Cape Verdean Kriolu, Haitian Creole). Human/manual. |
| Food / basic needs | SNAP + food-bank referral | **RI Community Food Bank** + HEZ; routes through **Unite Rhode Island (Unite Us)** closed-loop referral network. |
| Refill / financial gap | 90-day mail / auto-refill sync; copay help | Community pharmacy; manual. |
| Any of the above | **Unite Rhode Island (Unite Us)** | The **statewide closed-loop referral rail** (United Way of RI coordination center). Send → accept → **outcome status returns** = a real closed loop. This is the backbone a production feedback loop should ride. |

## The closure signal — the honest weak link

State 4 needs an objective *"did they refill?"* event. In this repo it comes
from the committed **`labels.parquet`** (`has_30_day_gap == 0` ⇒ on-time refill),
which `pdc.py` derives from real fill history — a faithful stand-in.

For a **real** build, the refill feed is the hardest part:
- **CurrentCare (RI HIE)** flipped to **opt-out (April 2025)** — good for coverage
  — but **has no clean antihypertensive refill feed**: its PDMP covers
  *controlled substances only* (BP meds aren't controlled), and the rest is
  *prescription lists, not confirmed dispenses*. So the objective refill signal
  is most likely **pharmacy-claims-based**, not CurrentCare.
- CurrentCare's automatable surface (**CMAD** real-time ADT alerts) is useful for
  a **hospitalization** signal, not refill closure.

`sync/loop_closure.CurrentCareOutcomeSource` is the documented seam for whatever
real feed replaces the label-derived outcome; nothing upstream changes.

## Escalation ladder (Rounds 0/1/2) — implemented 2026-07-23

State 1 "Routed" is now the entry to a **timed, latency-adjusted, consent-gated
escalation ladder**, all server-side and provider-only (we never message a patient):

- **Round 0 — CHW → pharmacy** (all patients): dispatch goes to the AE's CHW with
  an instruction to contact the pharmacy. We never contact a pharmacy directly.
- **Round 1 — SDOH-specific** (only if no refill): dominant SHAP driver →
  social worker / pharmacist / bilingual CHW, or a **transit voucher** (the only
  *external* disclosure). Transport is **consent-gated**: without external
  authorization it falls back to CHW-mediated internal transport help — never dropped.
- **Round 2 — escalate to the AE prescriber** (only if Round 1 fails): the dispatch
  body composes the **complete prior history** (every round, recipient, date, outcome).
- **Success at any round** = an objective dispense before the predicted break date →
  loop CLOSED, `closed_on_round` recorded.

What's in code now:
- `src/routing/escalation.py` — pure, deterministic state machine (`now` injected).
  Statuses: WAITING · WAIT_ELAPSED_DISPATCH_PENDING · DISPATCHED ·
  WAITING_ON_DATA_LATENCY · GATED_ON_CONSENT · CLOSED · EXHAUSTED. State is fully
  derivable from persisted data → `data/snapshots/escalation_state.json` (additive,
  merges across ticks; restart-safe). Timers derive from the frozen predicted break
  date and are **adjusted upward by the active source's max latency** so we never
  escalate for "no refill" before the confirming feed would have shown one.
- `src/routing/consent.py` — two scopes: `internal_care_coordination` (BAA + treatment)
  and `external_disclosure` (R.I. Gen. Laws § 5-37.3-4 signed authorization). Fail-closed;
  "unknown" is distinct from "denied"; a staleness window (365d) treats old consent as
  absent. Synthetic `data/snapshots/consent.json` (documented synthetic).
- `src/routing/dispatch_messages.py` — provider-addressed, four-language dispatch
  bodies + a labeled CHW/prescriber read-aloud script. `addressed_to` is always
  `provider_or_organization`.
- `src/sync/connectors/` — the dispense-data connector layer (Surescripts stub →
  AE claims → local fallback chain) with per-source latency + access-mode metadata,
  written to `data/snapshots/pharmacy_sync_state.json`.
- `src/sync/escalation_job.py` — ticks the machine (reuses sync_job's fit-once/score
  machinery), with a clearly-labeled `--simulate-days-per-tick` demo affordance.
- `src/api/main.py` — serves the per-patient escalation block + funnel + pharmacy
  source; see `API_CONTRACT.md`.

The old **States 2/3** (Acknowledged/Actioned) remain the CHW's client-side subjective
progress; the loop still CLOSES only on the objective refill, never on self-report.

## Why n_round2 is 0 on synthetic data

On the synthetic cohort the escalation funnel always reports **`n_round2 = 0`**, and
this is **structural, not a tuning problem**. The `local_file` connector derives each
patient's outcome from the fixed `has_30_day_gap` label committed in `labels.parquet`,
so a patient resolves **permanently** to `CLOSED` (label = adherent) or `EXHAUSTED`
(label = a confirmed break) at their **first** observation. There is never an open
"no refill *yet*" interval for the ladder to climb through — the outcome is known up
front, so no synthetic patient is ever left unresolved long enough to reach Round 2.

Round 2 (escalate to the AE prescriber) is **fully implemented and unit-tested**
(`src/routing/escalation.py`; `tests/test_escalation.py`,
`tests/test_escalation_machine.py`, `tests/test_dispatch_messages.py::test_round2_*`
exercise the transition, the latency-adjusted timer, and the full-history dispatch
body). What it *requires* to fire is a **longitudinal feed with genuinely unresolved
patients** — a patient who is open, gets Round 0/1 dispatches, and only later either
refills or doesn't. **Real AE claims data provides exactly that** (a 30–90-day
rolling window of fills that resolve over time), which is the demo's intended
production timer backbone. We deliberately do **not** fabricate Round 2 patients to
make the funnel look fuller; `n_round2 = 0` is the honest reading of a
label-derived synthetic feed.

## State 4 → retraining

Closed loops are fresh, real-world-observed labels: `on_time_refill` → `y=1`
(adherent → `has_30_day_gap=0`), confirmed break → `y=0`.
`models/retrain_labels.py` now harvests each observed loop into
`data/snapshots/retrain_labels.parquet` as
`(patient_id, label_adherent, source_event_date, closed_on_round, refill_source,
loop_closed)`, and `refresh_labels()` merges the labels into
`data/snapshots/labels_retrained.parquet` — a drop-in replacement for
`labels.parquet` in a retraining run. **`closed_on_round` and `refill_source` are
metadata, never features** (they describe post-index events); `feature_panel.parquet`
is never touched, so features stay strictly pre-index (asserted by
`tests/test_retrain_labels.py`).

**What this proves / doesn't:** it demonstrates the *architecture* supports
continuous learning — observed outcomes flow back into a training-ready label set
with the leakage guard intact. It does **not** claim retraining is validated on real
patients; on synthetic data the objective label is the ground-truth outcome
regardless of intervention, so retraining re-affirms existing signal rather than
demonstrating *lift*. The production path (`mode="production"`) still raises
`NotImplementedError`: it first needs (1) server-side State-3 action gating,
(2) a real dispense feed + id crosswalk, and (3) **selection-bias correction**
(closed-loop labels are only the routed/actioned subset).

## Dependencies / open items
- **ROI (Release of Information) protocol** for the State 3 → 4 human handoff —
  owned by **Alec (Privacy Lead)**. Left as a clean seam; not implemented here.
- **Unverified claim:** the planning doc's *"$12,348 lifetime savings per
  participant"* could **not** be sourced (real CHW-ROI figures are ratios, e.g.
  Penn IMPaCT ~$2.47 per $1). **Do not cite externally** until the origin is found.
- Confirm live routing of the 855-330-913x numbers (MTM vs. a residual Modivcare
  page) and Unite Us RI API availability before building real adapters.

## Run it

```bash
python -m src.run_routing_pipeline          # State 1: routing_table.json (worklist + fairness)
python -m src.sync.loop_closure             # State 4: loop_outcomes.json + pharmacy_sync_state.json
python -m src.routing.consent               # synthetic consent.json (demo)
python -m src.routing.escalation            # escalation_state.json (Rounds 0/1/2 + dispatch bodies)
python -m uvicorn src.api.main:app --port 8000   # serve worklist + escalation (see API_CONTRACT.md)
# live compressed-time demo (optional):
python -m src.sync.escalation_job --simulate-days-per-tick 30 --max-ticks 6
# State 4 → retraining labels + refreshed labels file (demo):
python -m src.models.retrain_labels
# frontend is rebuilt separately; it consumes only the API (src/ui/dashboard.html is legacy)
```
