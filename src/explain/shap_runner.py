"""
src/explain/shap_runner.py

Produces the per-patient "top modifiable driver" decomposition that
src/routing/rules.py consumes.

WORKFLOW
--------
1. Fit (or load) a persistence-risk model on the patient feature matrix.
2. For each patient, decompose the model's log-odds prediction into
   additive per-feature attributions.
3. Aggregate raw features into the clinical driver categories that
   routing_table.yaml knows about (e.g. `sbp_slope` + `dbp_slope` ->
   `bp_trend`).
4. Split attributions into MODIFIABLE_DRIVERS (eligible for routing) vs.
   non-modifiable context features (age, sex, baseline_adherence, ...)
   which inform the model but must never drive a UI action or appear in
   routing_table.yaml.
5. Return a ranked list of (driver, attribution_value) per patient so
   rules.py can apply tie-break / hierarchy / safety-override logic on
   top of it.

ATTRIBUTION METHOD
-------------------
We use exact Shapley-value decomposition for a linear (logistic
regression) model: for a linear predictor f(x) = b0 + sum_i(coef_i * x_i),
the Shapley value of feature i relative to the dataset mean baseline is
exactly coef_i * (x_i - mean_i). This is mathematically exact (not an
approximation) for linear/logistic models and keeps this module free of
heavy/optional native dependencies (e.g. the `shap` package's C/numba
build chain), which matters for a routing-adjacent module that clinical
reviewers need to be able to audit line-by-line.

If a future sprint swaps in a tree ensemble or the `shap` library
directly, only `_fit_model` and `_attribute` need to change — everything
downstream (rules.py, capacity.py) consumes the same
`PatientDriverProfile` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Driver taxonomy
# ---------------------------------------------------------------------------

# Raw feature -> clinical driver category. Multiple raw features can roll
# up into one driver (e.g. blood pressure trend features).
#
# CORRECTED 2026-07-04: this taxonomy previously referenced placeholder
# column names (housing_barrier_score, isolation_score, sbp_slope, ...)
# that no feature module actually produces -- SHAPRunner.fit() would raise
# a KeyError against the real feature_panel.parquet. Replaced with the
# actual columns emitted by src/features/sdoh.py (flag_sdoh_*, flag_*) and
# src/features/trajectories.py (*_trajectory_slope). The driver *names*
# on the right-hand side are unchanged and still match routing_table.yaml
# (drivers:/clinical_hierarchy:/safety_overrides: keys) -- only the raw
# input-column mapping was wrong, not the taxonomy itself or anything
# downstream in rules.py/capacity.py.
FEATURE_TO_DRIVER: Dict[str, str] = {
    "flag_sdoh_housing_barrier": "housing_barrier",
    "flag_sdoh_financial_barrier": "financial_barrier",
    "flag_sdoh_transport_barrier": "transport_barrier",
    "flag_sdoh_isolation_any": "isolation",
    "flag_sdoh_low_education": "low_education",
    "flag_migrant_any": "migrant_status",
    "sbp_trajectory_slope": "bp_trend",
    "dbp_trajectory_slope": "bp_trend",
    "flag_trauma_exposure": "trauma_exposure",
}
# NOTE: "regimen_complexity" (routing_table.yaml's driver_fallbacks target)
# intentionally has no entry here -- no feature module in src/features/
# computes a regimen-complexity signal. The fallback remap in rules.py is
# defensive/no-op until such a feature exists; it is not a missing case,
# it's an honestly-empty one. Do not fabricate a placeholder column to
# fill it.

# Drivers that are eligible to be routed on. `trauma_exposure` is included
# here because shap_runner must still surface it — rules.py is what treats
# it as a hard override rather than a ranked candidate.
MODIFIABLE_DRIVERS = set(FEATURE_TO_DRIVER.values())

# Context features that inform the risk model but are never routing
# candidates. Kept out of FEATURE_TO_DRIVER on purpose.
#
# CORRECTED 2026-07-04: age/baseline_adherence/months_on_therapy/
# num_comorbidities were placeholders -- only age_years is actually
# produced anywhere in src/features/*.py or src/etl/cohort.py today.
# Re-add the others here (and build a producer for them under
# src/features/) if/when they exist; do not carry forward columns that
# don't exist just to keep this list looking more complete than the data
# actually is.
NON_MODIFIABLE_FEATURES = [
    "age_years",
]

ALL_MODEL_FEATURES = list(FEATURE_TO_DRIVER.keys()) + NON_MODIFIABLE_FEATURES


@dataclass
class PatientDriverProfile:
    """Per-patient output of the SHAP decomposition step."""

    patient_id: str
    predicted_risk: float  # model-predicted probability of non-persistence
    driver_attributions: Dict[str, float]  # driver -> signed attribution
    raw_feature_attributions: Dict[str, float] = field(default_factory=dict)

    def ranked_modifiable_drivers(self) -> List[Tuple[str, float]]:
        """Modifiable drivers ranked by |attribution| descending.

        Only positive-risk-contributing drivers are candidates for
        intervention routing — a driver that is *protective* (negative
        attribution) is not something we'd route an intervention against.
        """
        candidates = [
            (driver, val)
            for driver, val in self.driver_attributions.items()
            if driver in MODIFIABLE_DRIVERS and val > 0
        ]
        return sorted(candidates, key=lambda kv: kv[1], reverse=True)


class SHAPRunner:
    """Fits a persistence-risk model and produces driver decompositions."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self._fitted = False
        self._feature_means: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SHAPRunner":
        """Fit the underlying linear risk model.

        X must contain exactly ALL_MODEL_FEATURES columns (order-independent;
        we reindex). y is a binary label: 1 = discontinued/non-persistent
        (i.e. AT RISK), 0 = persistent/adherent.

        RISK POLARITY (confirmed by Chris, 2026-07-04): predicted_risk must
        be "higher number = higher risk," using the standard 0.80 PDC
        adherence threshold to define the positive class -- y=1 iff
        pdc_180d < 0.80 (below the CMS/PQA adherence cutoff = at risk).
        This is the OPPOSITE polarity from classifier.py's own default
        (`load_classification_frame`'s default binarizes pdc_180d >= 0.80
        as the positive class, i.e. y=1 there means *adherent*). Callers
        of this class (see src/run_routing_pipeline.py) must build y with
        the at-risk-positive polarity documented here -- this class does
        not re-derive or flip the label itself, so passing the wrong
        polarity in silently inverts every downstream routing decision.
        """
        X = X[ALL_MODEL_FEATURES]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._feature_means = X_scaled.mean(axis=0)
        self._fitted = True
        return self

    def _attribute(self, x_scaled_row: np.ndarray) -> Dict[str, float]:
        """Exact linear-model Shapley attribution for one patient row.

        attribution_i = coef_i * (x_i - mean_i), which sums exactly to
        (log-odds prediction - baseline log-odds).
        """
        coefs = self.model.coef_[0]
        deltas = x_scaled_row - self._feature_means
        contributions = coefs * deltas
        return dict(zip(ALL_MODEL_FEATURES, contributions))

    def explain_patient(self, patient_row: pd.Series) -> PatientDriverProfile:
        if not self._fitted:
            raise RuntimeError("SHAPRunner.fit() must be called before explain_patient().")

        x = patient_row[ALL_MODEL_FEATURES].to_frame().T
        x_scaled = self.scaler.transform(x)[0]

        raw_attrib = self._attribute(x_scaled)
        predicted_risk = float(self.model.predict_proba(x_scaled.reshape(1, -1))[0, 1])

        driver_attrib: Dict[str, float] = {}
        for feature, value in raw_attrib.items():
            driver = FEATURE_TO_DRIVER.get(feature)
            if driver is None:
                continue  # non-modifiable context feature, not a driver
            driver_attrib[driver] = driver_attrib.get(driver, 0.0) + value

        return PatientDriverProfile(
            patient_id=str(patient_row["patient_id"]),
            predicted_risk=predicted_risk,
            driver_attributions=driver_attrib,
            raw_feature_attributions=raw_attrib,
        )

    def explain_cohort(self, df: pd.DataFrame) -> List[PatientDriverProfile]:
        """Convenience batch wrapper over explain_patient."""
        return [self.explain_patient(row) for _, row in df.iterrows()]
