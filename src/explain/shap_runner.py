"""
src/explain/shap_runner.py

Produces the per-patient "top modifiable driver" decomposition that
src/routing/rules.py consumes.

WORKFLOW
--------
1. Fit the persistence-risk model on the patient feature matrix. The model is
   the PRIMARY model defined in src/models/classifier.py (a
   HistGradientBoostingClassifier by default), fit here on the SAME feature set
   classifier.py trains on (src/models/common.py::select_feature_columns).
2. For each patient, decompose the model's prediction into additive per-feature
   attributions using the SHAP explainer appropriate to the model family.
3. Aggregate raw features into the clinical driver categories that
   routing_table.yaml knows about (e.g. `sbp_trajectory_slope` +
   `dbp_trajectory_slope` -> `bp_trend`).
4. Split attributions into MODIFIABLE_DRIVERS (eligible for routing) vs.
   non-modifiable context features (age, comorbidity/utilization/refill-behavior
   history, ...) which inform the model but must never drive a UI action or
   appear in routing_table.yaml.
5. Return a ranked list of (driver, attribution_value) per patient so
   rules.py can apply tie-break / hierarchy / safety-override logic on top of it.

ATTRIBUTION METHOD — DISPATCHES ON MODEL FAMILY (corrected 2026-07-23)
---------------------------------------------------------------------
This module previously fit an ad-hoc LogisticRegression on a hand-picked
10-feature subset and used exact linear Shapley. That meant the entire routing
pipeline was scored by a WEAK model (out-of-fold ROC-AUC ~0.68) while the repo
documented the STRONG one (HGB ~0.85 on the enriched 33-feature panel). The two
had drifted completely apart. The fix: score with classifier.py's primary model
and explain it with the correct SHAP computation for its family.

  * TREE model (HistGradientBoostingClassifier, the default/deployed):
    ``shap.TreeExplainer``. TreeSHAP is exact for tree ensembles and returns
    attributions in **log-odds (margin) space** that sum to the model's raw
    decision-function output minus a base value (verified additive against
    ``clf.decision_function``). That is the SAME space, sign convention, and
    additive structure the old linear path used -- positive attribution = pushes
    toward the positive class = higher gap risk -- so the ``val > 0`` filter in
    ``ranked_modifiable_drivers`` and every downstream comparison in rules.py /
    capacity.py keep their exact meaning. What changes is only that the numbers
    now come from the real model (and reflect feature interactions the linear
    model could not), which is the entire point.

  * LINEAR model (LogisticRegression), kept as an interpretable baseline:
    exact Shapley ``coef_i * (x_i - mean_i)``, mathematically exact for a linear
    predictor and auditable line-by-line. This path builds its OWN
    impute->scale->LR pipeline (NOT classifier.build_logistic_regression, whose
    ``add_indicator=True`` appends missingness-indicator columns that would
    desync the coef<->feature 1:1 mapping this exact decomposition relies on).

Model choice is configurable via ``SHAPRunner(model_name=...)``; the default is
classifier.py's ``DEPLOYED_MODEL`` so the pipeline uses the real primary model
unless a caller deliberately asks for the baseline.

SEMANTIC-STABILITY NOTE (read before changing the driver taxonomy)
------------------------------------------------------------------
Switching from linear to tree SHAP does NOT change the driver taxonomy or the
sign convention (both are margin-space, positive = higher risk). It DOES change
which driver is dominant for some patients, because the strong model weights the
same barrier flags differently in the presence of the new context features, and
because HGB's probabilities (with ``class_weight=None``) run lower than the old
balanced-logistic ones -- so ``predicted_risk`` and the downstream
``days_to_predicted_break`` shift. That shift is expected and correct (it is the
whole reason for this change); the before/after worklist/fairness delta is
reported alongside the regenerated artifacts, never silently absorbed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# classifier.py is the single source of truth for the model family + its
# hyperparameters and the shared leakage-safe feature selection. Import it the
# same way run_auc.py does (ROOT + src/models on path) so the served model and
# the evaluated model are literally the same builder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "models"))

from src.models.common import select_feature_columns  # noqa: E402
from src.models.classifier import MODEL_BUILDERS, DEPLOYED_MODEL  # noqa: E402

# ---------------------------------------------------------------------------
# Driver taxonomy
# ---------------------------------------------------------------------------

# Raw feature -> clinical driver category. Multiple raw features can roll up into
# one driver (e.g. blood pressure trend features). These are the ONLY features
# that can trigger a routed action; see the "modifiable vs. context" note below.
#
# CORRECTED 2026-07-04: this taxonomy previously referenced placeholder column
# names that no feature module produces; replaced with the actual columns emitted
# by src/features/sdoh.py (flag_sdoh_*, flag_*) and src/features/trajectories.py
# (*_trajectory_slope). The driver *names* on the right still match
# routing_table.yaml (drivers:/clinical_hierarchy:/safety_overrides: keys).
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
# intentionally has no entry here -- no feature module computes it. The fallback
# remap in rules.py is defensive/no-op until such a feature exists.

# Drivers eligible to be routed on. `trauma_exposure` is included because
# shap_runner must still surface it -- rules.py treats it as a hard override
# rather than a ranked candidate.
MODIFIABLE_DRIVERS = set(FEATURE_TO_DRIVER.values())

# MODIFIABLE vs. NON-MODIFIABLE CONTEXT (the model uses all 33; only these 8
# drivers can route). Decision 2026-07-23, argued here because it governs what a
# CHW is ever asked to act on:
#
# The enriched panel (src/features/pre_index.py) adds cross-drug refill mechanics
# (xdrug_*), prior-adherence (adh_*), engagement (engage_*), comorbidity (cmb_*),
# payer churn (payer_*), regimen (rx_*), plus BP *level* summaries (sbp_mean/max/
# min/latest, dbp_*). Every one of these is fed to the model and improves the risk
# score -- but NONE is a routing candidate, for one reason: routing maps a driver
# to a concrete human intervention (routing_table.yaml), and these features are
# either non-actionable predictors or clinical context, not social barriers a CHW
# can address. A high refill-gap history (xdrug_refill_gap_max) or a payer switch
# (payer_n_switches) is PREDICTIVE of a future gap, but "the patient has missed
# refills before" is not itself a barrier you dispatch a social worker or
# pharmacist against -- the actionable barrier is *why* (housing/financial/
# transport/...), which the flag_* drivers capture. Surfacing "refill-gap history"
# as a dominant driver would produce a routed action with no matching intervention
# and would let a pure risk marker masquerade as a modifiable cause. So they are
# CONTEXT: they sharpen P(gap), never appear as a `top_driver`, and never reach
# routing_table.yaml. Likewise BP *level* stays context -- only the BP *trend*
# (slope, worsening over time) is the pharmacist-actionable `bp_trend` signal, so
# folding level in would change which patients route to a med review on a static
# reading rather than a trajectory. Anything not in FEATURE_TO_DRIVER is context
# by construction; NON_MODIFIABLE_FEATURES names the one explicitly-allowlisted
# non-prefixed context column so the intent is greppable.
NON_MODIFIABLE_FEATURES = [
    "age_years",
]

# Routing-critical features that MUST be present for driver attribution to work
# (the 9 driver raw columns + age). This is a subset of the fitted model's full
# feature set (select_feature_columns, currently 33) -- kept as an explicit
# contract that run_routing_pipeline.py / sync_job.py check the panel against, so
# a panel missing a driver column fails loudly instead of silently dropping a
# routable barrier. Named ALL_MODEL_FEATURES for backward-compat with those
# importers; it is the routing contract, not the model's entire input.
ALL_MODEL_FEATURES = list(FEATURE_TO_DRIVER.keys()) + NON_MODIFIABLE_FEATURES

# Model families this module knows how to explain. Explicit dispatch, no guessing:
# a model_name outside this map raises rather than silently mis-attributing.
_TREE_MODELS = frozenset({"hist_gradient_boosting"})
_LINEAR_MODELS = frozenset({"logistic_regression"})


@dataclass
class PatientDriverProfile:
    """Per-patient output of the SHAP decomposition step."""

    patient_id: str
    predicted_risk: float  # model-predicted probability of the >=30d gap event
    driver_attributions: Dict[str, float]  # driver -> signed attribution (margin space)
    raw_feature_attributions: Dict[str, float] = field(default_factory=dict)
    # Actual (raw, pre-imputation) model-input feature values for this patient,
    # keyed by feature name. This is the GROUND-TRUTH data, distinct from the
    # signed SHAP attributions above -- safety overrides must gate on this, never
    # on an attribution's sign. See driver_data_value().
    raw_feature_values: Dict[str, float] = field(default_factory=dict)

    def driver_data_value(self, driver: str) -> float:
        """Max ACTUAL (raw, pre-imputation) feature value among the raw features
        that map to ``driver``; 0.0 if none are present.

        This is the ground-truth data signal a hard safety override must gate on.
        It is deliberately NOT the SHAP attribution: an attribution's sign flips
        with the learned effect and the baseline, so a driver whose presence
        lowers the model output would attribute positively for patients WITHOUT
        the flag -- firing the override on the wrong patients. Reading the raw
        flag value avoids that entirely. NaN (missing data) is treated as
        not-present, so a missing flag never triggers a safety override.
        """
        present = [
            v
            for feat, drv in FEATURE_TO_DRIVER.items()
            if drv == driver
            for v in (self.raw_feature_values.get(feat, 0.0),)
            if v is not None and v > 0  # NaN > 0 is False -> missing excluded
        ]
        return max(present) if present else 0.0

    def ranked_modifiable_drivers(self) -> List[Tuple[str, float]]:
        """Modifiable drivers ranked by attribution descending.

        Only positive-risk-contributing drivers are candidates for intervention
        routing -- a driver that is *protective* (negative attribution) is not
        something we'd route an intervention against. Positive == raises the gap
        risk, in the same margin space for both the tree and linear explainers.
        """
        candidates = [
            (driver, val)
            for driver, val in self.driver_attributions.items()
            if driver in MODIFIABLE_DRIVERS and val > 0
        ]
        return sorted(candidates, key=lambda kv: kv[1], reverse=True)


class SHAPRunner:
    """Fits classifier.py's primary model and produces driver decompositions.

    ``model_name`` selects the model family (default: the deployed primary model
    from classifier.py). ``"hist_gradient_boosting"`` uses TreeSHAP;
    ``"logistic_regression"`` uses exact linear Shapley on an internal
    impute->scale->LR pipeline (interpretable baseline).
    """

    def __init__(self, model_name: str = DEPLOYED_MODEL, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.feature_columns: Optional[List[str]] = None
        self._fitted = False

        if model_name in _TREE_MODELS:
            self._family = "tree"
            # The real primary model, from classifier.py -- same builder, same
            # hyperparameters as training/eval, so served == evaluated model.
            self.pipeline: Pipeline = MODEL_BUILDERS[model_name]()
            self.explainer = None
        elif model_name in _LINEAR_MODELS:
            self._family = "linear"
            # Internal exact-Shapley pipeline (NOT classifier.build_logistic_regression:
            # its add_indicator=True appends columns that desync the coef<->feature
            # 1:1 mapping the exact decomposition needs). keep_empty_features=True so
            # an all-NaN column (e.g. a slope with no readings) never drops a column.
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            self.scaler = StandardScaler()
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
            self._feature_means: Optional[np.ndarray] = None
        else:
            raise ValueError(
                f"SHAPRunner has no SHAP dispatch for model '{model_name}'. "
                f"Supported: {sorted(_TREE_MODELS | _LINEAR_MODELS)}. (Tree models "
                f"with column-changing preprocessing, e.g. the RF baseline's "
                f"add_indicator imputer, are intentionally not wired -- they break "
                f"the 1:1 feature<->attribution mapping routing relies on.)"
            )

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "SHAPRunner":
        """Fit the risk model on the leakage-safe feature set.

        ``frame`` is the merged panel+labels frame (same object run_auc.py /
        classifier.py load); the model features are picked from it by
        ``select_feature_columns`` -- the SAME allowlist classifier.py uses, so
        the demographic hold-out and outcome-leakage guard apply identically here.
        ``y`` is the event-positive label (1 == has_30_day_gap, the >=30-day gap
        we predict); callers build it with that polarity and this class does not
        re-derive or flip it, so passing the wrong polarity silently inverts every
        downstream routing decision.
        """
        self.feature_columns = select_feature_columns(frame)
        X = frame[self.feature_columns].astype("float64")

        if self._family == "tree":
            self.pipeline.fit(X, y)
            clf = self.pipeline.named_steps["clf"]
            try:
                import shap
            except ImportError as exc:  # pragma: no cover - shap is a listed dep
                raise ImportError(
                    "shap is required for the tree model's SHAP attributions "
                    "(pip install shap). Or use model_name='logistic_regression' "
                    "for the dependency-free exact-linear baseline."
                ) from exc
            # TreeSHAP, no background dataset -> tree_path_dependent margin-space
            # attributions that sum to decision_function - base_value.
            self.explainer = shap.TreeExplainer(clf)
        else:  # linear
            X_imp = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imp)
            self.model.fit(X_scaled, y)
            self._feature_means = X_scaled.mean(axis=0)

        self._fitted = True
        return self

    def _attributions_and_risk(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (attributions [n, n_features], predicted_risk [n]) for a batch.

        Attributions are in margin (log-odds) space for BOTH families, positive ==
        higher gap risk; predicted_risk is P(gap event) = predict_proba[:, 1].
        """
        if self._family == "tree":
            # Feed the explainer exactly what the final estimator sees. The HGB
            # pipeline has no preprocessing steps, but transform through any that
            # exist so this stays correct if the pipeline ever gains a step.
            pre_steps = self.pipeline.steps[:-1]
            X_model = Pipeline(pre_steps).transform(X) if pre_steps else X
            raw = self.explainer.shap_values(X_model)
            if isinstance(raw, list):
                # older shap returns one array per class -> positive class
                sv = np.asarray(raw[1] if len(raw) > 1 else raw[0])
            else:
                sv = np.asarray(raw)
                if sv.ndim == 3:  # (n, features, classes) -> positive class
                    sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
            proba = self.pipeline.predict_proba(X)[:, 1]
            return sv, proba

        # linear: exact Shapley coef_i * (x_scaled_i - mean_i)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        sv = self.model.coef_[0] * (X_scaled - self._feature_means)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return sv, proba

    def explain_cohort(self, df: pd.DataFrame) -> List[PatientDriverProfile]:
        """Attribute every patient in ``df`` in one batched pass.

        Batched on purpose: TreeSHAP over the whole cohort at once is ~2 orders of
        magnitude faster than the old row-by-row ``iterrows`` loop, and the linear
        path vectorizes trivially. Driver attributions are the summed margin-space
        contributions of that driver's raw features (FEATURE_TO_DRIVER); all other
        selected features stay as context in raw_feature_attributions and never
        become a driver.
        """
        if not self._fitted:
            raise RuntimeError("SHAPRunner.fit() must be called before explain_cohort().")

        cols = self.feature_columns
        X = df[cols].astype("float64")
        sv, proba = self._attributions_and_risk(X)

        values = X.to_numpy(dtype=float)  # raw, pre-imputation (NaN preserved)
        ids = df["patient_id"].astype(str).to_numpy()

        profiles: List[PatientDriverProfile] = []
        for i in range(len(df)):
            raw_attrib = {c: float(sv[i, j]) for j, c in enumerate(cols)}
            raw_values = {c: float(values[i, j]) for j, c in enumerate(cols)}
            driver_attrib: Dict[str, float] = {}
            for feat, driver in FEATURE_TO_DRIVER.items():
                if feat in raw_attrib:  # driver features are always selected, but be safe
                    driver_attrib[driver] = driver_attrib.get(driver, 0.0) + raw_attrib[feat]
            profiles.append(PatientDriverProfile(
                patient_id=ids[i],
                predicted_risk=float(proba[i]),
                driver_attributions=driver_attrib,
                raw_feature_attributions=raw_attrib,
                raw_feature_values=raw_values,
            ))
        return profiles

    def explain_patient(self, patient_row: pd.Series) -> PatientDriverProfile:
        """Attribute a single patient (thin wrapper over the batched path)."""
        return self.explain_cohort(patient_row.to_frame().T)[0]
