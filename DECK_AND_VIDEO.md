# DECK FIXES + VIDEO SCRIPT — due tonight 11:59 PM

## Part 1 — Deck fixes (paste-ready)

### FIX 1 — Title slide + any slide saying "survival model" (CRITICAL)

The model pivoted to binary classification on July 2 (MODEL_CARD.md). The
deck must not claim a survival model — a judge who opens the repo will see
the mismatch immediately.

**Title slide tagline, change from:**
> Survival-Based Hypertension Persistence & Bilingual CHW Routing

**to:**
> Predicting Hypertension Medication Breaks & Bilingual CHW Routing

**Slide 4 ("BP Cascade RI -") subtitle, change from:**
> A survival model that anticipates a break window, predicting a date range
> where coverage is expected to lapse.

**to:**
> A risk model that flags which patients are likely to have a 30+ day
> medication gap in the next 6 months — and routes each one to the right
> helper before it happens.

(If asked at the Datathon why not survival: "We started there — the
incident-user cohort on the 1K dev set was too small for stable temporal
survival splits, so we shipped the defensible classifier and kept the
survival infrastructure in the repo for the full-scale build. It's
documented in our model card." That answer *gains* you credibility.)

### FIX 2 — Replace the lorem ipsum "Project Timeline" slide

Retitle it **"What We Built in 12 Days"** — four columns:

1. **Data (Jun 24–30):** 300K-patient SyntheticRI via Globus → streaming
   Polars ETL → 16,205-patient treated-hypertensive cohort
2. **Features + Model (Jun 30–Jul 3):** PDC, BP trajectories, SDOH barrier
   flags → gap-risk classifier; leakage test caught a real bug — all
   results are post-fix
3. **Fairness + Routing (Jul 3–5):** subgroup calibration audit; SHAP
   drivers → capacity-capped routing to CHW / social worker / pharmacist
4. **Last mile (Jul 5–6):** bilingual worklist dashboard + API + SMS-nudge
   mock, running end-to-end on the routing snapshot

### FIX 3 — The conclusion slide (currently empty, and fix the typo
"Conclusio n")

Title: **"A Tuesday-Morning Action List"**

Body:
> Every other team can tell you who has high blood pressure today. BP
> Cascade RI tells you **which patient's treatment is likely to break in
> the next six months — and whether to send a pharmacist, a social worker,
> or a phone call in Spanish.**
>
> - 16,205-patient cohort → 942-patient worklist, capped to real staffing
> - Fairness audited: passes the 80% disparate-impact rule (ratio 0.87),
>   with equal-or-better model performance for Hispanic patients
> - Built with leakage tests, a model card, and open documentation
>
> **Next step: validation on real Rhode Island clinical data.** Everything
> here is a workflow demonstration on synthetic data — and we say that out
> loud, because the workflow is the product.

### FIX 4 — SDOH Barriers slide

There's a dangling bullet that just says "Among" — either finish the
sentence or delete the bullet. Suggested finish:
> Among the capped worklist, the dominant modifiable drivers were trauma
> exposure, migrant status, and BP trend — each mapped to a different
> routed action.

### FIX 5 — Numbers slide (the 54.2% / 47.2% / etc. stat cards)

Keep it, but add a footer line:
> Prevalences from SyntheticRI (synthetic); shown to illustrate the
> barrier burden the routing layer is designed around.

---

## Part 2 — 3-minute video script

**Target: ~430 words ≈ 3 minutes at presentation pace. One speaker per
section, or Andres narrates all with teammates' slides.**

---

**[0:00–0:25 — The problem — Andres]**

Hypertension is controllable with cheap generic drugs — but roughly half
of treated patients lose control, and for Hispanic and Latino Rhode
Islanders the drop is steeper. The failure point usually isn't the
prescription. It's persistence: a copay surprise, a confusing regimen
change, a missed refill nobody notices until there's a crisis. The system
is reactive. We built BP Cascade RI to make it anticipatory.

**[0:25–0:55 — What it does — Andres, over dashboard screen recording]**

This is a community health worker's Tuesday-morning action list. Each row
is a real output of our pipeline: a patient flagged as high-risk for a
30-plus-day medication gap, the dominant modifiable barrier behind that
risk, and a routed action — a pharmacist for a regimen problem, a social
worker for housing instability, a phone call in the patient's preferred
language for isolation. English and Spanish, one toggle. Capped at real
staffing capacity — 942 patients out of a 16,000-patient cohort — because
a list a clinic can't work through isn't a tool, it's noise.

**[0:55–1:35 — How it works — Umar/Chris]**

We processed the 300,000-patient SyntheticRI dataset with a streaming ETL
pipeline into a treated-hypertensive cohort of 16,205 incident users. From
medications, encounters, blood-pressure trajectories, and coded SDOH
findings we built leakage-disciplined features — everything the model sees
comes strictly from before each patient's index date, and an automated
test enforces it. That test earned its keep: it caught a real lookback bug
in our BP features mid-build, and every number we report is post-fix. The
model is a deliberately simple, auditable classifier; SHAP attribution on
that model is the routing logic — we explain a real model's real drivers
instead of inventing barrier labels synthetic data can't support.

**[1:35–2:15 — Fairness — Annie]**

Because this tool decides who gets outreach, we audited who gets selected.
Worklist selection passes the 80% disparate-impact rule across ethnicity
groups, and model discrimination is slightly better for Hispanic patients
than non-Hispanic — checked per subgroup, published in the repo, not
asserted.

**[2:15–2:45 — Honesty + next step — Alec/Andres]**

Everything you've seen is a methodology demonstration on synthetic data —
Synthea generates these patterns from module logic, so we make no claims
about real Rhode Islanders. What we're demonstrating is the workflow: from
raw records to a capacity-capped, fairness-audited, bilingual action list.
The next step is validation with real RI clinical partners — Clínica
Esperanza, Providence Community Health Centers, Thundermist.

**[2:45–3:00 — Close — Andres]**

Every team here can tell you who has high blood pressure today. BP Cascade
RI tells you whose treatment is about to break — and who should knock on
the door. Thank you.

---

## Part 3 — What NOT to say in the video

- Don't say "survival model" or "predicted break date" — say "risk of a
  30-day gap in the next six months"
- Don't cite the AUC unprompted in a 3-min pitch; if a slide shows it,
  frame as "modest discrimination, expected on synthetic data — the
  validated workflow is the deliverable"
- Don't say "277 GB" as a flex — say "300,000 patients" once and move on
- Don't claim the SMS nudge sends anything — it's a mocked UI component
