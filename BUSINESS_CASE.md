# BP Cascade RI — Business Case

The commercial and legal model behind the product, written so the claims are
defensible from the repository rather than only from a slide. Every number traces
to a source; every legal constraint traces to code. Figures on synthetic data are
labeled as such.

---

## 1. Customer and the money mechanism

**Customer.** Rhode Island Medicaid **Accountable Entities (AEs)** and the Medicaid
**Managed Care Organizations (MCOs)** that contract them. Named targets:
Providence Community Health Centers, Blackstone Valley Community Health Care,
Thundermist Health Center. Roughly **208,800 attributed Medicaid lives** statewide
across the AE program (RI EOHHS Accountable Entity attribution).

**Why they pay — the quality-score lever.** **Controlling High Blood Pressure**
(CBP) is a mandatory core measure in the **OHIC Aligned Measure Set** used across
RI commercial and Medicaid contracts. An AE's **overall quality score gates and
multiplies its shared-savings pool**: miss quality thresholds and the AE forfeits
a share of savings it would otherwise earn. A sustained antihypertensive
medication gap is a direct hit to CBP performance. Preventing that gap therefore
**protects the shared-savings pool** — the AE's own revenue — which is the
willingness-to-pay. We sell gap-prevention that shows up in the CBP numerator, not
a generic "risk score."

---

## 2. Cost-avoidance formula (every input + source + confidence)

```
gross cost avoidance / targeted patient / year
  = annual_cost_differential × absolute_adherence_lift
  = $1,176 × 5%
  ≈ $59
```

| Input | Value | Source | Confidence |
|---|---|---|---|
| Annual cost differential, adherent vs non-adherent hypertensive patient | **$1,176** | AHA cost-of-non-adherence literature, cohort n≈4.8M | Medium — real-world claims cohort, not RI-specific; direction robust, level approximate |
| Absolute adherence lift from the intervention | **5 percentage points** | Conservative planning assumption vs. CHW/pharmacy adherence-intervention literature (typically 3–12 pts) | Low–Medium — deliberately conservative; **not** yet demonstrated on RI data |
| Gross cost avoidance / patient / year | **≈ $59** | Product of the above | Inherits the weakest input (Low–Medium) |

This is a **gross** figure. It excludes program cost (CHW time, platform,
referral overhead), so net ROI requires a caseload/cost model an AE supplies. The
value proposition is a **portfolio** effect across a capacity-capped worklist
(942 routed patients in the current snapshot), not a per-patient windfall.

### What we do NOT claim (and why it matters)

We explicitly **refuse the $20,000-stroke-avoided framing.** The inflated pitch
multiplies a full stroke cost by an adherence lift and books it as savings. It is
wrong because **baseline stroke incidence in this population is only ~1.25–2% per
year** — closing one medication gap does not prevent a $20,000 event for that
patient in expectation; at 5% relative risk reduction on a ~1.5% base rate the
expected stroke-cost avoidance is well under $20 per patient-year, an order of
magnitude below the noise. Anchoring on the honest **~$59 medication-and-utilization
differential** is both defensible and still compelling at portfolio scale.
Refusing the inflated number is a credibility asset, not a weakness.

---

## 3. The legal ground: CHCCIA and the two-scope consent model

**Constraint.** Rhode Island's **Confidentiality of Health Care Communications and
Information Act (CHCCIA), R.I. Gen. Laws § 5-37.3-4**, is **stricter than HIPAA**.
Disclosing identifiable health information to a **non-covered entity** (a transit
broker, a community-based organization) requires the **patient's own signed
authorization**. An Accountable Entity's Business Associate Agreement covers
ingestion and risk stratification as *healthcare operations*, but it does **not**
let the AE consent to external disclosure **on the patient's behalf**.

**How the code implements it.** Two independently-evaluated scopes, per patient,
fail-closed:

- `internal_care_coordination` — routing to staff inside the covered entity (CHW,
  social worker, pharmacist, prescriber). Basis: BAA + treatment relationship.
- `external_disclosure` — routing to a non-covered entity (transit broker, CBO).
  Basis: a signed § 5-37.3-4 authorization; nothing less clears it.

Code references:
- `src/routing/consent.py` — `INTERNAL` / `EXTERNAL` scopes, `ACTION_REQUIRED_SCOPE`
  (each action declares its scope), `gate()` (fail-closed; **unknown ≠ denied**;
  `CONSENT_VALIDITY_DAYS = 365` staleness → treated as absent).
- `src/routing/escalation.py` — Round 1 `transport_barrier` resolves to
  `transit_voucher` (external). If external consent is missing/denied/stale, the
  engine substitutes the **internal CHW-mediated fallback** (`chw_transport_support`)
  and records the gate + reason — the patient is never silently dropped.
- Enforced-out-of-model: consent is an operational field with no allowlisted
  prefix, so `src/models/common.py::select_feature_columns` cannot admit it;
  `tests/test_consent.py` asserts this.

On synthetic data the consent map is generated by `src/routing/consent.py`
(`data/snapshots/consent.json`, flagged synthetic); real values arrive from the AE
consent feed. In the current snapshot the four gating states are all exercised:
595 granted/granted, 185 granted/denied, 140 granted/unknown, 22 no-record.

---

## 4. Data reality: why Surescripts is not the timer backbone

**Surescripts Medication History** is **prescriber-initiated and encounter-tied**:
a licensed prescriber may pull a specific patient's history in connection with a
real clinical encounter. It **contractually prohibits automated background batch
queries** by a third-party platform, and carries a **1–14 day PBM propagation
lag**. Going live requires Surescripts certification (a 12–18 month process) or a
certified middleware partner, plus an active prescriber Surescripts Prescriber ID
(SPI). It is therefore an **opportunistic** source, not a panel-scanning timer.

**Consequence in code.** The demo's automated refill-timer backbone is the **AE
pharmacy-claims feed** (`batch_permitted`, ~30–90 day billing lag), not
Surescripts. Every source declares an `access_mode` and a latency profile, and
**escalation timers are latency-adjusted** so a patient is never escalated for a
refill that simply has not surfaced yet.

Code references:
- `src/sync/connectors/base.py` — `SourceProfile` (`access_mode`,
  min/typical/max latency, `requires_encounter`, dispense-vs-prescription).
- `src/sync/connectors/surescripts.py` — refuses a batch query, naming the
  contractual restriction; `authenticate()` fails clean without credentials.
- `src/sync/connectors/ae_claims.py` — the batch backbone (30/90-day latency).
- `src/sync/connectors/factory.py` — Surescripts → AE claims → local fallback
  chain, trace persisted to `data/snapshots/pharmacy_sync_state.json`.
- `src/routing/escalation.py::compute_round_schedule` — `effective_escalate_at =
  escalate_at + source_max_latency_days`.

**RXFILL** (NCPDP SCRIPT push) would give near-real-time dispense confirmation
without a prescriber pull, but is rarely implemented for chronic generics like
antihypertensives, so it is documented (`connectors/rxfill.py`) but not the
backbone. CurrentCare/RIQI is **deprecated** (sunset; CRISP Shared Services is now
RI's RHIO).

---

## 5. Why we do not purchase transport or pay patient incentives

A deliberate design boundary, not a missing feature. **Paying for a Medicaid
beneficiary's transport, or paying the patient an incentive, raises federal
Anti-Kickback Statute and beneficiary-inducement (Civil Monetary Penalty)
exposure.** Rhode Island **already funds non-emergency medical transport (NEMT)**
through its statewide broker (MTM Link); the benefit exists. Our role is to make
sure the **existing** benefit is actually *used* — a CHW-mediated request into the
established broker — not to create a new payment we fund. This keeps the product on
the safe side of the kickback line while still removing the transport barrier. (See
`bizNonGoal` copy on the site and the `transit_voucher` → CHW-mediated fallback in
`escalation.py`.)

---

## 6. What the model actually delivers (honest, current)

- Primary model: `HistGradientBoostingClassifier`, 33 leakage-safe features,
  out-of-fold **ROC-AUC 0.857** (5-fold stratified CV; logistic baseline 0.681).
  Source of truth: `data/snapshots/auc_report.json` (reproduce: `make eval`).
- Fairness: capped-worklist selection-rate min/max ratio **0.9933** — passes the
  80% disparate-impact rule (`data/snapshots/fairness_report.json`).
- Synthetic-data ceiling: Synthea models adherence as a stable per-patient trait,
  so 0.857 is a synthetic ceiling, not a real-world estimate.
- Round 2 of the escalation ladder is fully implemented and unit-tested but
  **unreachable on synthetic data** (`n_round2 = 0`) — the label-derived local feed
  resolves every patient at first observation, so there is no unresolved interval
  to climb. It requires a longitudinal claims feed. See
  `FEEDBACK_LOOP.md` → "Why n_round2 is 0 on synthetic data."

---

## 7. Honest open questions for a real pilot

- **Agreements needed:** an AE data-sharing agreement + BAA; a Surescripts
  certification path or certified middleware partner; a mechanism to receive
  per-patient § 5-37.3-4 consent from the AE; MTM Link and Unite Rhode Island
  integration terms.
- **The refill signal is the weak link.** A real "did they refill?" feed is most
  likely pharmacy-claims-based (30–90 day lag); confirm the available dispense feed
  before committing to escalation cadences.
- **Lift is unproven.** The 5-point adherence assumption and the $59 figure need a
  real evaluation (ideally a staggered rollout) before either is quoted as fact.
- **Realistic timeline:** ~3–6 months to a data-ready pilot with one AE
  (agreements + claims feed + consent field), ~12–18 months if Surescripts
  certification is on the critical path. A first outcomes read follows one to two
  claims-lag cycles after go-live (i.e. not before ~6–9 months post-integration).

---

*Sources: OHIC Aligned Measure Set (Controlling High Blood Pressure); RI EOHHS
Accountable Entity attribution; AHA hypertension cost-of-non-adherence literature
(n≈4.8M); MTM Link (RI statewide NEMT broker); Unite Rhode Island (closed-loop
referral network); R.I. Gen. Laws § 5-37.3-4. Model/fairness figures are
reproducible from `data/snapshots/auc_report.json` and `fairness_report.json`.*
