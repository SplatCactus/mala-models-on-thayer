# Escalation Ladder — Implementation Plan (v3, APPROVED)

Status: **ARCHITECTURE APPROVED 2026-07-23. Still PLAN ONLY — do not implement yet.**
All 8 open assumptions and 3 additional decisions are resolved below (see
"Approved decisions"). No code has been written for the escalation build; no
existing file has been modified for it.

This models the legal + technical due-diligence constraints **honestly in code**,
because a demo that shows a gated step *with its reason* is more credible than one
that pretends the constraint isn't there.

> History: an earlier draft (provider-direct "notify the pharmacy" Round 0, an
> AE-CHW Round 2, no consent/latency model) was reverted when due diligence
> overturned its assumptions. This v3 is the approved basis.

---

## 0. Ground truth — what the repo actually does today

| Area | Reality in code |
|---|---|
| **Feedback loop** | `src/sync/loop_closure.py` yields a **one-shot binary** outcome (`on_time_refill = has_30_day_gap==0`, `loop_closure.py:104`). `data/snapshots/loop_outcomes.json` **does not exist**; `api/main.py:108` always returns `{}`. API row shape for `loop_outcome`: `{observed, on_time_refill, event_date, source}` or `null` (`api/main.py:164`). |
| **Time / escalation** | **None.** No timers, no state machine. States 2/3 are client-side localStorage (`FEEDBACK_LOOP.md:21`). |
| **Pharmacy data** | ABC `PharmacyRefillSource` (`pharmacy_source.py:66`) → `SyntheticRIAdapter` (reveals panel rows) + `CurrentCareAdapter` (stub). Parallel `RefillOutcomeSource` in `loop_closure.py:84`. **No Surescripts, no CRISP.** |
| **Routing** | Actions hard-limited to `pharmacist / social_worker / chw_call` (`rules.py:53`). **No "pharmacy" action.** Trauma is a hard safety override gating on the raw flag (`rules.py:155`). Tie-break margin 1.25 → `clinical_hierarchy`. |
| **Retraining** | `retrain_labels.py` demo path + `NotImplementedError` production path. Reads the absent `loop_outcomes.json`. Not wired to training. |
| **Model served** | **UPDATED 2026-07-23:** the routing pipeline now scores with the **HistGradientBoostingClassifier** primary model (`shap_runner.py` fits classifier.py's `DEPLOYED_MODEL`, TreeSHAP); out-of-fold ROC-AUC 0.857 (`auc_report.json`). This is decision #9's dependency, and it has **landed**. Escalation must stay model-agnostic (see Decision 9): it consumes `predicted_risk` / `top_driver` / `days_to_predicted_break` from `routing_table.json` and never imports the model. |
| **API** | `GET /worklist`, `/worklist/{id}`, `/health`. `loop_outcome` + `preferred_language` effectively mocked. |
| **Tests** | 31 passing (`test_leakage.py` + `test_pre_index_leakage.py`). MODEL_CARD's "10 passing" is stale. Leakage guard = `common.py::select_feature_columns` (allowlist + outcome-column block). |
| **Packaging** | Only `src/eval/__init__.py`; everything else is namespace-imported via `sys.path.insert(0, ROOT)` + `from src.x import y`. |
| **Committed artifacts** | `feature_panel.parquet`, `labels.parquet` (16,205 patients; 2,362 with `has_30_day_gap==1`), `routing_table.json` (942 capped / 16,205 eligible), `fairness_report.json`. |

---

## CORE PRINCIPLE — provider-only, never patient-facing

Every dispatch is addressed to a provider or organization **inside the covered
entity**, who then contacts the patient. The system never messages a patient.
The one patient-directed artifact is a **read-aloud script for a CHW** — the
worker's tool, never a message we send. See §Reframing for every site to fix.

## The three corrected constraints (these drive the whole design)

- **C1 — We never contact a pharmacy directly.** Retail pharmacies won't act on
  an uncontracted third party's alert, and third-party refill steering raises
  anti-kickback / steering concerns. **Every pharmacy-directed action is
  CHW-mediated:** our dispatch goes to the AE's CHW with an instruction to
  contact the pharmacy. The dispatch payload must make the mediating human
  explicit (`mediated_by: "ae_chw"`).
- **C2 — Refill data is latent, not real-time.** Surescripts Medication History
  is **prescriber-initiated, encounter-tied, no automated batch** (contractually),
  1–14 day PBM lag; RXFILL push is rare for chronic generics; AE claims lag
  30–90 days. So **the connector carries a per-source latency profile and an
  access-mode flag**, every observation records its source + latency, and
  **escalation timers are latency-adjusted** so we never escalate a patient whose
  refill probably already happened but hasn't surfaced. This is the single most
  important honesty point in the build.
- **C3 — Consent is per-patient and scoped, stricter than HIPAA.** R.I. Gen.
  Laws § 5-37.3-4 requires explicit signed consent to disclose PHI to a
  **non-covered entity** (transit provider, CBO). The BAA covers ingestion +
  risk stratification as healthcare operations but **not** downstream external
  disclosure. So the system models **two consent scopes** — `internal` (CHW, SW,
  prescriber, pharmacist — all inside the covered entity) and `external`
  (transit, CBO) — and **gates each action against the right scope.** Consent
  arrives from the AE as a per-patient field; it is never assumed.

---

## THE LADDER (target model)

| Round | Dispatch (all provider-addressed) | Consent scope needed |
|---|---|---|
| **0 — CHW → pharmacy** (all patients) | To the **AE CHW**: contact the patient's pharmacy about the refill. Pharmacy never contacted by us. `mediated_by: ae_chw`. | `internal` |
| **1 — SDOH-specific** (if no refill) | trauma/isolation/housing → **social worker**; financial → **pharmacist** (refill sync / 90-day mail order); low_education/migrant → **bilingual CHW**; transport → **transit voucher (EXTERNAL)** | internal for all except transport → **`external`; if absent, action is GATED and falls back to a CHW-mediated internal action** |
| **2 — Escalate to prescriber** (if R1 fails) | To the patient's **AE prescriber** (full record access); they contact the patient, or route onward if the real provider is outside the AE. Payload **MUST carry the full prior-round history**. | `internal` |

**Success at any round** = an objective **dispense event before the predicted
break date** in the pharmacy feed. On success: close, record `closed_on_round`,
emit a training label. No self-report, ever.

**bp_trend** (a driver with no SDOH branch above) → **pharmacist** in Round 1
(medication review) — **APPROVED (decision #2):** medication review is the right
action for a blood-pressure-trajectory driver. Internal scope.

---

## 1. Escalation state machine (rounds, latency-adjusted timers, transitions)

### NEW `src/routing/escalation.py`
Pure, deterministic, `now` injected (testable; matches `loop_closure.py`'s DI
style). Round enum `ROUND_0_CHW_PHARMACY | ROUND_1_SDOH | ROUND_2_PRESCRIBER`.

**Per-patient state** (persisted, see §8):
```
current_round, status ∈ {WAITING, WAIT_ELAPSED_DISPATCH_PENDING, DISPATCHED,
                         WAITING_ON_DATA_LATENCY, GATED_ON_CONSENT,
                         CLOSED, EXHAUSTED},
predicted_break_date (frozen at entry), refill_source, refill_latency_days,
closed_on_round, consent_scopes, gated_actions[]
```

**Transitions (per tick):** (1) enter Round 0, freeze break date; (2) objective
dispense before break date → `CLOSED` + `closed_on_round`; (3) `now ≥
wait_until(round)` and undispatched → dispatch (§4), resolving consent (§3)
first — if the round's action needs a scope the patient lacks (or has only
stale consent), status `GATED_ON_CONSENT` and fall back to the internal
CHW-mediated action; (4) `now ≥ escalate_at(round)` but the **latency guard has
NOT cleared** (`now < effective_escalate_at`, §2) → status
`WAITING_ON_DATA_LATENCY` (a first-class, API-visible state — decision #4; it is
a feature: it shows we know dispense data lags); (5) `now ≥ effective_escalate_at`
and still no refill → advance; (6) Round 2 elapsed with no refill → `EXHAUSTED`
(logged). **Trauma safety override** skips Round 0 → enters Round 1 social worker
with `requires_human_review` (preserves `rules.py`).

**Files:** NEW `escalation.py`; NEW `data/snapshots/escalation_state.json`
(runtime, gitignored). Its driver is **NEW `src/sync/escalation_job.py`**
(decision #5 — not growing `sync_job.py`, whose docstring is emphatic about
scope); reuses the fit-once/score-subset machinery. The job takes a
`--simulate-days-per-tick` **demo time-compression** flag (decision #6, required)
so a multi-week ladder is demonstrable live in seconds; the flag is explicitly
labeled a demo affordance in `--help`, in log output, and in the written state's
`meta` so it can never be mistaken for production cadence.

**What could break:** break-date drift (mitigate: freeze at entry); `routing_table.json`
is rewritten wholesale each run (`sync_job.py:167`) → escalation state must be a
**separate, merged** file; incremental cohort reveal (state merge must be additive);
**time-compression vs. latency guard must share one clock** (see New Risks) — the
compressed "day" must scale both the wait timers and the source latency windows,
or the guard either never fires or never clears in the demo.

## 2. Break-window-derived wait, latency-adjusted

Single documented function, policy constants at top of `escalation.py`.
```
predicted_break_date = entry_date + days_to_predicted_break          # frozen
wait_until(r)   = predicted_break_date − ROUND_LEAD_DAYS[r]           # e.g. {0:60,1:30,2:10}
escalate_at(r)  = max(wait_until(r)+MIN_ROUND_DWELL_DAYS, next round's wait)
# LATENCY GUARD (C2): do not escalate until the confirming source WOULD have shown a refill
effective_escalate_at(r) = escalate_at(r) + refill_source.max_latency_days
```
`days_to_predicted_break` is today's linear proxy `(1−risk)·180`
(`run_routing_pipeline.py:214`) — kept, with its "not a clinical estimate"
caveat. **Edge cases:** past/negative window → clamp to entry (act now); short
window → `MIN_ROUND_DWELL_DAYS` floor stops all rounds collapsing into one
instant; long window → natural break-60/30/10 spacing. **The latency guard is
the C2 teeth:** with an AE-claims source (30–90d lag) a patient is never
escalated for "no refill" until ≥ max-latency after the boundary, so we don't
punish a refill that simply hasn't surfaced.

**What could break:** if `lead ≥ window` every wait lands in the past → validate
and clamp; a mis-set latency profile silently stalls escalation → surface it in
the API and log it.

## 3. Two-scope consent model + per-action gating

### NEW `src/routing/consent.py`
```
ConsentScopes(internal: bool, external: bool, source: str, as_of: date)
ACTION_REQUIRED_SCOPE = { chw_pharmacy:internal, social_worker:internal,
  pharmacist:internal, bilingual_chw:internal, prescriber:internal,
  transit_voucher:external }
gate(action, scopes, today) -> ALLOWED | GATED(reason) ; when GATED, escalation
uses the documented internal fallback (transit_voucher → CHW-mediated transport help).
```
Consent arrives **as a per-patient field from the AE**, not assumed. It is an
**operational** field, **never a model feature** — it must be excluded from
`select_feature_columns` (it has no allowlisted prefix, so the guard already
blocks it; `test_consent.py` asserts that).

**Data source — APPROVED (decision #1):** the synthetic cohort has no consent
field, so the demo joins a **synthetic** consent table
`data/snapshots/consent.json`, per-patient `{internal, external, as_of}`.
Requirements folded in:
  * A header/`_note` field in `consent.json` **and** a section in `SCHEMA.md`
    both state plainly that this file is **synthetic**; real values arrive from
    the AE feed. This is a compliance-sensitive artifact — it must never be
    mistaken for real consent (see New Risks).
  * Default `internal = true` (covered by the BAA / treatment relationship);
    `external` is **mixed** across patients so the transit-voucher gate is
    visibly exercised on stage.

**Consent staleness — APPROVED (decision #7), fail-closed:** each record carries
`as_of`; consent older than `CONSENT_VALIDITY_DAYS` (proposed **365 days**, a
tunable policy constant documented in `consent.py` and `SCHEMA.md`) is treated as
**absent** for its scope. Reasoning: § 5-37.3-4 consent is a point-in-time signed
authorization, not a standing state; an unbounded-age consent would let a
years-old signature authorize a disclosure today. A **missing OR stale** record
fails **closed** (no external scope → gate + internal fallback), never open.

## 4. Round-specific dispatch messages (en/es/pt/ht, provider-addressed)

### NEW `src/routing/dispatch_messages.py`
`build_dispatch(round, card, prior_rounds, consent, refill_meta) -> payload`.
Recipient types: `ae_chw` (R0, `mediated_by` explicit), `social_worker`,
`pharmacist`, `bilingual_chw`, `transit_voucher_external`, `prescriber` (R2).
Every payload carries `addressed_to: "provider_or_organization"` (no patient
recipient exists) and, for CHW/prescriber, a clearly labeled read-aloud script
("the worker's tool, not a message we send"). **Round 0 body is addressed to the
CHW and says "contact the pharmacy," never the pharmacy directly** (C1). **Round 2
body composes the full history** — every prior round, recipient, date, outcome —
the "tell them what we already tried" requirement. Four-language dicts follow the
existing `rationale_{en,es,pt,ht}` / `DRIVER_LABELS` pattern; duplicated (not
imported from `api/main.py`) so a routing module doesn't depend on the API layer.

## 5. Pharmacy connector — per-source latency + access-mode metadata

**APPROVED framing (decision #3), stated prominently — not buried:** Surescripts
Medication History is **prescriber-initiated and encounter-tied, contractually
not a batch background feed**, so it is an *opportunistic* source only (an
observation lands when a clinician queries during a real encounter). The
**automated refill-timer backbone for the demo is the AE claims feed (30–90-day
lag)**. This is the honest architecture: our escalation cadence is paced by the
slowest-but-only-batch-legal source, and the API surfaces that latency so the
lag is visible, not hidden. The docs (MODEL_CARD/SCHEMA/README as relevant) must
say this in plain sight.

### NEW `src/sync/connectors/` (base, surescripts, ae_claims, rxfill, local_file, factory)
```
@dataclass SourceProfile: name; access_mode ∈ {prescriber_initiated, batch_permitted};
           min_latency_days; typical_latency_days; max_latency_days; requires_encounter: bool
class PharmacyConnector(ABC):
    def authenticate(self) -> None
    def fetch_dispense_events(patient_ids, start, end) -> list[DispenseEvent]
    def covered_patient_ids(patient_ids) -> set[str]     # "history found" per patient
    @property source_profile -> SourceProfile ; last_synced -> str
DispenseEvent: patient_id, rxnorm/ndc, product, dispense_date, days_supply, pharmacy_ncpdp, source
```
- **SurescriptsConnector** (primary, **stubbed**): realistic OAuth2 auth + NCPDP
  medication-history request shape; `access_mode=prescriber_initiated`,
  `requires_encounter=True`, latency 1–14d. `fetch` raises `NotImplementedError`
  if creds absent, with a message naming the DUA + credentials + endpoint needed,
  and **the factory falls back**. Because it's prescriber-initiated + no-batch, it
  is **not** the automated timer backbone — it contributes observations only when
  an encounter context is supplied (modeled, documented).
- **AeClaimsConnector** (fallback, `batch_permitted`, latency 30–90d) — the actual
  automated batch refill detector for the demo's timers.
- **RxfillPushConnector** (fallback, `batch_permitted`, latency ~0–1d, rarely
  available) — documented, low-coverage.
- **LocalFileConnector** (synthetic demo) — synthesizes dispense events from
  `labels.parquet` (`has_30_day_gap==0` → an in-window dispense; ==1 → none),
  `batch_permitted`, latency 0. **This is what runs today and must keep working
  byte-for-byte.**
- **factory.py**: fallback chain **Surescripts → AE claims → local**, selected by
  env; returns the first that authenticates. Writes **`data/snapshots/pharmacy_sync_state.json`**
  = `{source_name, access_mode, latency_profile, last_synced}` for the API (C2 badge).

**CurrentCare:** existing `CurrentCareAdapter`/`CurrentCareOutcomeSource` marked
**DEPRECATED** with a comment: CurrentCare/RIQI was sunset; **CRISP Shared
Services** is now RI's RHIO. **Do not build a CurrentCare adapter.** No CRISP
adapter built either unless asked — noted as the real RHIO seam.

**Refill detection is real, not hypothetical:** `loop_closure.py` gains a
`ConnectorOutcomeSource(connector)` (implements the existing `RefillOutcomeSource`
ABC) that turns dispense events into `on_time_refill` + `event_date` + `source` +
`latency_days`, writes the existing `loop_outcomes.json` shape (extended, not
replaced), which `escalation.py` reads. Escalation depends only on the ABC /
`loop_outcomes.json`, never on a concrete adapter (proves plug-and-play).

## 6. API changes (`src/api/main.py`)

Per-row **additive** `escalation` block (existing fields unchanged in shape):
`current_round` + `round_label{en,es,pt,ht}`; `status` (incl.
`WAITING_ON_DATA_LATENCY`); `days_remaining` for the current wait AND, when the
status is latency-held, `days_until_latency_clears` so the UI can render "waiting
on data latency (~N days)"; `dispatch_history[]` (round, recipient_type,
recipient_label, dispatched_at, outcome, message body); `current_dispatch`
(recipient + body in 4 languages, only if wait elapsed); `consent_scopes` (incl.
`as_of` and a `stale` flag) + `gated_actions[]` (action + reason: `no_external` /
`stale_consent`); `objective_outcome` (incl. `closed_on_round`, `refill_source`,
`refill_latency_days`). Keep the existing `loop_outcome` field derived from it
(back-compat). Top-level: **funnel counts** (`n_by_round`, `n_closed`,
`n_closed_by_round`, `n_round2`, `n_gated`, `n_waiting_on_latency`) and the
**active pharmacy source + access-mode + latency profile + last_synced** from
`pharmacy_sync_state.json`. **All three surfaces are built (decision #8):** the
enriched `GET /worklist`, plus new `GET /escalation/{id}` (full history) and
`GET /escalation/summary` (funnel). All reads graceful-absent (`.get` fallbacks,
like `main.py:174`). The **complete consumable contract** is enumerated in
Decision 10 below, so the frontend can build against it before this is finished.

## 7. Retraining pipeline (design)

Extend `retrain_labels.py`: each closed loop → `(patient_id, label_adherent,
source_event_date, closed_on_round, refill_source)`. `closed_on_round`/source are
**metadata, never features** (leakage guard blocks them). Merge harvested labels
into a refreshed `labels.parquet` for observed patients; features
(`feature_panel.parquet`) stay strictly pre-index; re-run `classifier.py`. Gate
on server-side action state (now available via `escalation_state.json`).
Confounding/propensity caveat stays loud: on synthetic data this re-affirms
signal, it cannot prove lift. NEW `src/models/apply_retrain_labels.py` (production
path documented, demo path runnable).

## 8. Schema migration + backward compatibility

`routing_table.json` **unchanged** → `fairness.py`'s `eligible_pool` /
`capped_worklist` untouched. Escalation state is a **separate, additive**
`escalation_state.json`; consent in `consent.json`; sync state in
`pharmacy_sync_state.json`. `loop_outcomes.json` is **extended** (adds
`refill_source`, `refill_latency_days`, keeps all existing keys) so `retrain_labels`
and the API keep reading it. New `connectors/` re-exports the old
`pharmacy_source` names as shims so `sync_job.py` imports don't break. Every new
API read uses `.get(...)` fallbacks so a pre-migration snapshot still serves.

## 9. Tests needed

`test_escalation.py` (round advance, latency guard blocks premature escalation,
frozen break date, trauma skip-to-R1, EXHAUSTED); `test_wait_period.py` (math +
latency adjustment + edge clamps); `test_consent.py` (gating per scope, external
fallback, fail-closed on missing record, **consent field rejected as a model
feature**); `test_dispatch_messages.py` (4 languages, R0 addresses CHW not
pharmacy, R2 carries full history, `mediated_by` present); `test_connectors.py`
(factory fallback chain, Surescripts raises without creds, local unchanged,
per-source latency surfaced); `test_api_escalation.py` (additive block,
graceful-absent, funnel + sync source). Existing 31 leakage tests must stay green.

---

## Reframing debt (every patient-facing implication)

| Site | Issue | Fix |
|---|---|---|
| `routing_table.yaml:62,67,72,77` | trauma rationale says "automated **outreach**/contacto automatizado" — reads as automated patient contact | reframe to "automated **provider dispatch**"; bump version |
| `api/main.py:157-160` | `outreach_script_*` keys carry the audit rationale and imply patient-facing copy | rename to provider-`dispatch_message` vs. read-aloud `chw_script`; **(API-layer change — batched with §6)** |
| Round 0 concept | must be **CHW→pharmacy**, never pharmacy-direct (C1) | dispatch addressed to CHW, `mediated_by` explicit |
| `chw_call` action name | implies *we* call | keep key; document target = dispatch to CHW |
| `pharmacy_source.py` / `loop_closure.py` CurrentCare | sunset | mark DEPRECATED, name CRISP as the RHIO |

## 4-state loop ↔ Round 0/1/2 mapping

State 1 Routed → **Round 0 entry**. States 2/3 (Acknowledged/Actioned,
client-side) → **per-round** subjective progress; still never close the loop.
State 4 (objective refill) → **success at any round** + `closed_on_round`. **Net
new:** the timed, latency-adjusted, consent-gated transitions — the old model had
no server-side timer, no consent scope, no source latency.

## Approved decisions (2026-07-23)

All eight open assumptions are resolved as approved:

1. **Consent data** — synthetic `data/snapshots/consent.json`, labeled synthetic
   in-file and in `SCHEMA.md`, `internal=true` default, `external` mixed. (§3)
2. **`bp_trend → pharmacist`** in Round 1 — approved (medication review). (Ladder)
3. **AE claims (30–90d) is the automated refill-timer backbone; Surescripts is
   opportunistic/encounter-tied only** — approved, framed prominently. (§5)
4. **Latency guard + visible `WAITING_ON_DATA_LATENCY` status** — approved; it's a
   feature surfaced in the API, not hidden. (§1, §2, §6)
5. **New `src/sync/escalation_job.py`** as the ticker — approved. (§1)
6. **Demo time-compression `--simulate-days-per-tick`** — approved and required;
   explicitly labeled a demo affordance. (§1)
7. **Consent date + staleness, fail-closed** — approved; `CONSENT_VALIDITY_DAYS`
   (proposed 365) documented; stale ⇒ absent. (§3)
8. **All three API surfaces** (enriched `/worklist` + `/escalation/{id}` +
   `/escalation/summary`) — approved, build all. (§6, Decision 10)

## Decision 9 — model-family independence (hard requirement)

A separate workstream rewired the routing pipeline from the runtime
LogisticRegression to the **HistGradientBoostingClassifier** (that rewire has now
landed; see the Ground-truth "Model served" row). It changes risk scores, the
score distribution, and per-driver SHAP magnitudes. **Escalation must hardcode
none of that.** Concretely:
  * Escalation reads `predicted_risk`, `top_driver`, `days_to_predicted_break`,
    `is_safety_override` **from `routing_table.json`** — the stable contract — and
    never imports `shap_runner`, `classifier`, or any model object.
  * No constant may assume a score *range* (e.g. HGB with `class_weight=None`
    produces lower probabilities, so break dates skew later; timers derive from
    `days_to_predicted_break` and therefore self-adjust — do not bake in a
    threshold tuned to the old logistic distribution).
  * No constant may assume a *fixed driver set*; iterate whatever drivers appear.
  * `test_escalation.py` fixtures use synthetic cards, not a live model, so a
    future model swap cannot break escalation tests.

## Decision 10 — complete API surface (frontend contract)

The UI is being rebuilt from scratch and is not in the repo; **assume it can read
nothing but HTTP responses.** Everything below must be reachable via the API so
the frontend can build against it before the backend is done. (Field names are
the contract; shapes are additive and `.get`-guarded.)

- **`GET /worklist`** → existing top-level keys UNCHANGED (`generated_at`,
  `data_source`, `last_synced`, `cohort_size`, `capacity`, `role_caps_used`,
  `role_capacity_expansions`, `worklist[]`), PLUS new top-level:
  `pharmacy_source {name, access_mode, latency_days:{min,typical,max}, last_synced}`
  and `escalation_funnel {n_by_round, n_closed, n_closed_by_round, n_round2,
  n_gated, n_waiting_on_latency}`. Each `worklist[]` row keeps its current fields
  and gains `escalation { current_round, round_label{en,es,pt,ht}, status,
  days_remaining, days_until_latency_clears?, consent_scopes{internal,external,
  as_of,stale}, gated_actions[{action,reason}], current_dispatch{recipient_type,
  recipient_label,mediated_by,addressed_to,body{en,es,pt,ht},read_aloud_script?},
  dispatch_history[{round,recipient_type,recipient_label,mediated_by,dispatched_at,
  outcome,body{en,es,pt,ht}}], objective_outcome{observed,on_time_refill,event_date,
  refill_source,refill_latency_days,closed_on_round}|null }`. Existing
  `loop_outcome` retained (back-compat).
- **`GET /worklist/{patient_id}`** → one enriched row (same shape) or 404.
- **`GET /escalation/{patient_id}`** → the full escalation record for one patient
  (the `escalation` block above plus the frozen `predicted_break_date`,
  `entry_date`, and per-round `wait_until`/`escalate_at`/`effective_escalate_at`).
- **`GET /escalation/summary`** → `{escalation_funnel{...}, pharmacy_source{...},
  by_round[], generated_at}` for a dashboard header.
- **`GET /health`** → unchanged.
- **Enumerations the UI can rely on:** `status ∈ {WAITING,
  WAIT_ELAPSED_DISPATCH_PENDING, DISPATCHED, WAITING_ON_DATA_LATENCY,
  GATED_ON_CONSENT, CLOSED, EXHAUSTED}`; `current_round ∈ {0,1,2}`;
  `recipient_type ∈ {ae_chw, social_worker, pharmacist, bilingual_chw,
  transit_voucher_external, prescriber}`; `addressed_to == "provider_or_organization"`
  always. Languages always `{en,es,pt,ht}`.

## Decision 11 — `src/ui/dashboard.html` is read-only

Nothing in the escalation build may write to `src/ui/dashboard.html`. The teammate
owns the rebuild; our job is only to serve the Decision-10 contract. This is
already reflected in the plan (no dashboard edits anywhere); called out here so it
is unambiguous for the whole build.

## New risks these answers introduce

1. **Time-compression ↔ latency-guard shared clock (highest).** With
   `--simulate-days-per-tick`, the AE-claims 30–90d latency window and the round
   timers must be measured in the **same simulated clock**. If compression scales
   the timers but not the latency window (or vice versa), the guard either never
   fires (escalates through everything instantly) or never clears (nothing ever
   escalates) on stage. Mitigation: one simulated-time function feeds both; a
   test asserts a compressed run still transitions WAITING → WAITING_ON_DATA_LATENCY
   → advance.
2. **AE-claims cadence is honestly slow.** Pacing real escalation on a 30–90d feed
   means production escalation takes *months*; the demo compression hides that.
   The docs must state the real cadence plainly so the compressed demo isn't
   mistaken for real-time behavior.
3. **Synthetic `consent.json` is compliance-sensitive.** A file named "consent"
   that is actually fabricated could be misread as real authorization. Mitigation:
   loud synthetic labeling in-file + SCHEMA.md + API responses never imply the
   consent is real; fail-closed keeps the *unsafe* direction (over-disclosure)
   from being the default.
4. **Score-distribution shift from the HGB rewire.** Lower HGB probabilities push
   `days_to_predicted_break` later, so break dates (and thus every wait) skew
   further out than under the old logistic model. Timers self-adjust because they
   derive from that value, but the *absolute* demo cadence now depends on the
   model — another reason the timers must not hardcode any score threshold
   (Decision 9) and another reason time-compression is required for a live demo.
5. **Fail-closed consent raises CHW-fallback load.** Defaulting stale/missing
   external consent to gated means more transit-voucher cases fall back to
   CHW-mediated transport help. Correct for compliance, but the funnel/`n_gated`
   count must be surfaced so the added internal workload is visible, not silent.

## Revised build order (dependency-correct)

Decision #9's model rewire has **already landed**, so it no longer blocks. Build:
1. **Connectors** (`connectors/` + `SourceProfile` latency/access-mode; local +
   AE-claims + Surescripts stub + factory + `pharmacy_sync_state.json`), then wire
   `loop_closure.ConnectorOutcomeSource` → extended `loop_outcomes.json`.
2. **Consent** (`consent.py` + synthetic `consent.json` + staleness + fail-closed).
3. **Escalation state machine** (`escalation.py`) with latency-adjusted,
   consent-gated, break-window timers on one simulated clock.
4. **Dispatch messages** (`dispatch_messages.py`, provider-only, 4 languages, R2 history).
5. **Job** (`escalation_job.py`) with `--simulate-days-per-tick`.
6. **API** (enriched `/worklist` + `/escalation/{id}` + `/escalation/summary`) to
   the Decision-10 contract; plus the §Reframing fixes (incl. `outreach_script_*`).
7. **Retraining seam** (`retrain_labels` extension + `apply_retrain_labels.py`).
8. **Tests** (§9) + `SCHEMA.md`/docs updates. Existing 31 tests stay green.

---

**Architecture approved; nothing is built yet. Awaiting the go-ahead to start at
step 1 of the revised build order above.**
