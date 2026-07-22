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

## State 4 → retraining

Closed loops are fresh, real-world-observed labels: `on_time_refill` → `y=1`
(adherent), confirmed break → `y=0`. `models/retrain_labels.py` harvests them.
It is a **stub with a runnable demo path** — the production path
(`mode="production"`) raises `NotImplementedError` because it first needs:
(1) State-3 action state persisted **server-side** (localStorage can't be read),
(2) a real closure feed + CurrentCare↔cohort id crosswalk, and
(3) **selection-bias correction** (closed-loop labels are only the routed/actioned
subset). Retraining on synthetic labels re-affirms existing signal; it cannot
prove intervention *lift*.

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
python -m src.run_routing_pipeline          # State 1: produce routing_table.json
python -m src.sync.loop_closure             # State 4: produce loop_outcomes.json
python -m uvicorn src.api.main:app --port 8000   # serve worklist + outcomes
# open src/ui/dashboard.html  → States 2/3 tracked in localStorage
python -m src.models.retrain_labels         # State 4 → retraining labels (demo stub)
```
