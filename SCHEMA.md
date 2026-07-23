# SCHEMA

Source data: Synthea synthetic EHR/claims CSVs (the "1k" bundle, `1k_20260505_bundle`).
Landing zone: `data/raw/` (gitignored). Parquet conversions: `data/parquet/` (via `src/etl/ingest.py`).

Cohort size: **1,124 patients**, 2,344,790 rows across 18 tables.

## Tables (columns + row counts)

### patients.csv — 1,124 rows (one row per patient)
`Id, BIRTHDATE, DEATHDATE, SSN, DRIVERS, PASSPORT, PREFIX, FIRST, MIDDLE, LAST, SUFFIX, MAIDEN, MARITAL, RACE, ETHNICITY, GENDER, BIRTHPLACE, ADDRESS, CITY, STATE, COUNTY, FIPS, ZIP, LAT, LON, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE, INCOME`

### encounters.csv — 63,431 rows
`Id, START, STOP, PATIENT, ORGANIZATION, PROVIDER, PAYER, ENCOUNTERCLASS, CODE, DESCRIPTION, BASE_ENCOUNTER_COST, TOTAL_CLAIM_COST, PAYER_COVERAGE, REASONCODE, REASONDESCRIPTION`

### conditions.csv — 39,932 rows
`START, STOP, PATIENT, ENCOUNTER, SYSTEM, CODE, DESCRIPTION`

### observations.csv — 798,213 rows (labs/vitals, long format)
`DATE, PATIENT, ENCOUNTER, CATEGORY, CODE, DESCRIPTION, VALUE, UNITS, TYPE`

### medications.csv — 48,350 rows
`START, STOP, PATIENT, PAYER, ENCOUNTER, CODE, DESCRIPTION, BASE_COST, PAYER_COVERAGE, DISPENSES, TOTALCOST, REASONCODE, REASONDESCRIPTION`

### procedures.csv — 173,187 rows
`START, STOP, PATIENT, ENCOUNTER, SYSTEM, CODE, DESCRIPTION, BASE_COST, REASONCODE, REASONDESCRIPTION`

### immunizations.csv — 16,465 rows
`DATE, PATIENT, ENCOUNTER, CODE, DESCRIPTION, BASE_COST`

### allergies.csv — 950 rows
`START, STOP, PATIENT, ENCOUNTER, CODE, SYSTEM, DESCRIPTION, TYPE, CATEGORY, REACTION1, DESCRIPTION1, SEVERITY1, REACTION2, DESCRIPTION2, SEVERITY2`

### careplans.csv — 3,764 rows
`Id, START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION, REASONCODE, REASONDESCRIPTION`

### devices.csv — 6,018 rows
`START, STOP, PATIENT, ENCOUNTER, CODE, DESCRIPTION, UDI`

### imaging_studies.csv — 34,508 rows
`Id, DATE, PATIENT, ENCOUNTER, SERIES_UID, BODYSITE_CODE, BODYSITE_DESCRIPTION, MODALITY_CODE, MODALITY_DESCRIPTION, INSTANCE_UID, SOP_CODE, SOP_DESCRIPTION, PROCEDURE_CODE`

### supplies.csv — 27,869 rows
`DATE, PATIENT, ENCOUNTER, CODE, DESCRIPTION, QUANTITY`

### claims.csv — 111,781 rows
`Id, PATIENTID, PROVIDERID, PRIMARYPATIENTINSURANCEID, SECONDARYPATIENTINSURANCEID, DEPARTMENTID, PATIENTDEPARTMENTID, DIAGNOSIS1..8, REFERRINGPROVIDERID, APPOINTMENTID, CURRENTILLNESSDATE, SERVICEDATE, SUPERVISINGPROVIDERID, STATUS1, STATUS2, STATUSP, OUTSTANDING1, OUTSTANDING2, OUTSTANDINGP, LASTBILLEDDATE1, LASTBILLEDDATE2, LASTBILLEDDATEP, HEALTHCARECLAIMTYPEID1, HEALTHCARECLAIMTYPEID2`

### claims_transactions.csv — 976,986 rows (largest table)
`ID, CLAIMID, CHARGEID, PATIENTID, TYPE, AMOUNT, METHOD, FROMDATE, TODATE, PLACEOFSERVICE, PROCEDURECODE, MODIFIER1, MODIFIER2, DIAGNOSISREF1..4, UNITS, DEPARTMENTID, NOTES, UNITAMOUNT, TRANSFEROUTID, TRANSFERTYPE, PAYMENTS, ADJUSTMENTS, TRANSFERS, OUTSTANDING, APPOINTMENTID, LINENOTE, PATIENTINSURANCEID, FEESCHEDULEID, PROVIDERID, SUPERVISINGPROVIDERID`

### payer_transitions.csv — 41,678 rows
`PATIENT, MEMBERID, START_DATE, END_DATE, PAYER, SECONDARY_PAYER, PLAN_OWNERSHIP, OWNER_NAME`

### payers.csv — 10 rows
`Id, NAME, OWNERSHIP, ADDRESS, CITY, STATE_HEADQUARTERED, ZIP, PHONE, AMOUNT_COVERED, AMOUNT_UNCOVERED, REVENUE, COVERED_ENCOUNTERS, UNCOVERED_ENCOUNTERS, COVERED_MEDICATIONS, UNCOVERED_MEDICATIONS, COVERED_PROCEDURES, UNCOVERED_PROCEDURES, COVERED_IMMUNIZATIONS, UNCOVERED_IMMUNIZATIONS, UNIQUE_CUSTOMERS, QOLS_AVG, MEMBER_MONTHS`

### organizations.csv — 262 rows
`Id, NAME, ADDRESS, CITY, STATE, ZIP, LAT, LON, PHONE, REVENUE, UTILIZATION`

### providers.csv — 262 rows
`Id, ORGANIZATION, NAME, GENDER, SPECIALITY, ADDRESS, CITY, STATE, ZIP, LAT, LON, ENCOUNTERS, PROCEDURES`

## Join keys

- `patients.Id` ← `*.PATIENT` / `claims.PATIENTID` / `claims_transactions.PATIENTID`
- `encounters.Id` ← `*.ENCOUNTER`
- `organizations.Id` ← `encounters.ORGANIZATION`, `providers.ORGANIZATION`
- `providers.Id` ← `encounters.PROVIDER`, `claims.PROVIDERID`
- `payers.Id` ← `encounters.PAYER`, `medications.PAYER`, `payer_transitions.PAYER`
- `claims.Id` ← `claims_transactions.CLAIMID`

## Notes
- Code/id/zip-like columns (ZIP, SSN, FIPS, CODE, PROCEDURECODE, REASONCODE) are forced to string in ingest to preserve leading zeros.
- Dates kept as strings on ingest; parse in the features stage.


## "Cohort validation (Jun 30): 0 patients have HTN diagnosis without an antihypertensive fill; 103 patients have a fill without the HTN diagnosis code (likely off-label use of BP meds for other conditions, or a Synthea documentation quirk). The treated-hypertensive AND-filter is working correctly — this population's diagnosis and treatment are essentially coincident."

## Cohort validation (Andres, 2026-06-30)

Ran cohort.py on the 1K dataset. Results:
- HTN-diagnosed patients: 262
- Antihypertensive-fill patients: 365
- Treated-hypertensive cohort (intersection): 262

Note: every HTN-diagnosed patient also has an antihypertensive fill — 0
diagnosed-untreated patients on this dataset. 103 patients have a fill
but no HTN diagnosis code (likely off-label use of BP meds for other
conditions, e.g. beta-blockers for arrhythmia, or a Synthea quirk).

This is expected behavior on this dataset, not a bug in the cohort
logic. The AND-filter (diagnosis AND treatment) is working correctly —
this particular synthetic population happens to have 100% treatment-
on-diagnosis. Worth flagging in the pitch as a synthetic-data property,
not a real-RI finding.

Cohort snapshot: data/snapshots/cohort_patients.parquet (gitignored,
regenerate with `./venv/bin/python src/etl/cohort.py`)

## Feedback-loop & escalation artifacts (data/snapshots/, gitignored)

Produced by the routing → closure → escalation pipeline; consumed read-only by
`src/api/main.py` (see `API_CONTRACT.md` for the served shapes).

- **routing_table.json** — worklist (`meta`, `capped_worklist`, `eligible_pool`,
  `eligible_pool_summary`). Unchanged by the escalation work; `fairness.py` consumes
  `eligible_pool` vs `capped_worklist`. Produced by `run_routing_pipeline.py`.
- **loop_outcomes.json** — objective per-patient outcomes from
  `sync/loop_closure.py` via the connector layer. `outcomes[pid] = {observed,
  on_time_refill, event_date, source, refill_source, refill_latency_days}`. The last
  two are additive (2026-07-23); every prior key is unchanged.
- **pharmacy_sync_state.json** — active dispense source + fallback trace, written by
  `sync/connectors/factory.py`: `{selected:{source_name, access_mode
  (prescriber_initiated|batch_permitted), min/typical/max_latency_days,
  requires_encounter, confirms_dispense, latency_days:{min,typical,max}},
  last_synced, attempts:[{adapter, outcome (served|auth_failed), reason}]}`.
- **consent.json** — **SYNTHETIC** two-scope consent (`src/routing/consent.py`).
  `patients[pid] = {internal, external (granted|denied|unknown), source, as_of}`;
  a top-level `_note` and `meta.synthetic:true` flag it as not-real. `internal_care_
  coordination` rests on the BAA + treatment relationship; `external_disclosure`
  requires signed authorization per **R.I. Gen. Laws § 5-37.3-4**. Consent older than
  `meta.validity_days` (365) is treated as absent (fail-closed). NOT a model feature.
- **escalation_state.json** — per-patient ladder state (`src/routing/escalation.py`),
  additive and merged across ticks (restart-safe). `patients[pid]` carries
  `current_round (0/1/2)`, `status`, frozen `predicted_break_date`,
  `source_max_latency_days`, `unactionable_in_time`, per-scope `consent`,
  `gated_actions`, `objective_outcome` (incl. `closed_on_round`), `current_dispatch`,
  and `rounds[]` (each with `wait_until`/`escalate_at`/`effective_escalate_at`,
  `dispatched_at`, `outcome`, and the provider-addressed 4-language `dispatch` body).
- **retrain_labels.parquet** — harvested closed-loop labels (`src/models/retrain_labels.py`):
  `patient_id, label_adherent, source_event_date, closed_on_round, refill_source,
  loop_closed`. `closed_on_round`/`refill_source` are **metadata, never features**.
- **labels_retrained.parquet** — a refreshed label file (`patient_id, pdc_180d,
  has_30_day_gap`) with observed outcomes folded in; a drop-in replacement for
  `labels.parquet` in a retraining run. `feature_panel.parquet` is never modified, so
  features stay strictly pre-index.

