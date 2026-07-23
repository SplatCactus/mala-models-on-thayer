"""test_model_wiring.py — the routing pipeline uses the real primary model, and the
leakage allowlist keeps demographics/consent/outcomes out of every model path.

The out-of-fold AUC test runs real 5-fold CV on the full panel (~30s) and asserts
RELATIONSHIPS (HGB reproducible + materially beats logistic), never a brittle exact
figure.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "models"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.models.common import (  # noqa: E402
    PANEL_PATH, LABELS_PATH, select_feature_columns,
    HELD_ASIDE_DEMOGRAPHIC_COLUMNS, OUTCOME_COLUMNS,
)
from src.explain.shap_runner import SHAPRunner  # noqa: E402
from src.models.classifier import DEPLOYED_MODEL  # noqa: E402
import src.run_routing_pipeline as rrp  # noqa: E402

DEMOGRAPHICS = ["RACE", "ETHNICITY", "GENDER", "INCOME", "ZIP", "LAT", "LON",
                "CITY", "STATE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"]


def _small_frame(n=200):
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    frame = panel.merge(labels, on="patient_id", how="inner").dropna(
        subset=["has_30_day_gap"]).reset_index(drop=True).head(n)
    y = (frame["has_30_day_gap"].to_numpy() == 1).astype(int)
    return frame, y


# --- the pipeline fits the PRIMARY model, not an ad-hoc LogisticRegression --------

def test_shap_runner_default_fits_hist_gradient_boosting():
    frame, y = _small_frame()
    runner = SHAPRunner()  # default model_name == classifier.DEPLOYED_MODEL
    runner.fit(frame, y)
    clf = runner.pipeline.named_steps["clf"]
    assert isinstance(clf, HistGradientBoostingClassifier)
    assert not isinstance(clf, LogisticRegression)
    assert runner._family == "tree"
    assert DEPLOYED_MODEL == "hist_gradient_boosting"


def test_routing_pipeline_defaults_to_deployed_model():
    # run() must default to the deployed primary model, not a hardcoded baseline.
    assert inspect.signature(rrp.run).parameters["model_name"].default == DEPLOYED_MODEL


def test_logistic_baseline_still_available_and_is_linear():
    frame, y = _small_frame()
    runner = SHAPRunner(model_name="logistic_regression")
    runner.fit(frame, y)
    assert runner._family == "linear"
    assert isinstance(runner.model, LogisticRegression)


# --- demographics / consent / outcomes cannot reach the model ---------------------

def test_demographics_cannot_reach_the_model():
    """Held-aside demographics are excluded by the allowlist under the real path."""
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    frame = panel.merge(labels, on="patient_id", how="inner")
    selected = set(select_feature_columns(frame))
    for demo in DEMOGRAPHICS:
        assert demo in frame.columns, f"expected {demo} present in panel to test exclusion"
        assert demo not in selected, f"{demo} reached the model feature set"
    # belt-and-suspenders: nothing selected is in the held-aside demographic set
    assert not (selected & set(HELD_ASIDE_DEMOGRAPHIC_COLUMNS))


def test_outcome_and_consent_columns_never_selected():
    frame = pd.DataFrame({
        "patient_id": ["p1"], "age_years": [61.0], "flag_sdoh_housing_barrier": [1],
        "sbp_mean": [140.0], "xdrug_tenure_days": [30],
        # things that must never be features:
        "pdc_180d": [0.9], "has_30_day_gap": [1],          # outcomes
        "internal_care_coordination": [1], "external_disclosure": [0],  # consent
        "RACE": ["x"], "INCOME": [50000],                  # demographics
    })
    selected = set(select_feature_columns(frame))
    assert {"age_years", "flag_sdoh_housing_barrier", "sbp_mean", "xdrug_tenure_days"} <= selected
    for banned in ("pdc_180d", "has_30_day_gap", "internal_care_coordination",
                   "external_disclosure", "RACE", "INCOME"):
        assert banned not in selected


def test_outcome_exclusion_dominates_even_a_misconfigured_allowlist(monkeypatch):
    """Defense-in-depth: outcome columns are excluded even if their prefix is (wrongly)
    added to the allowlist, because select_feature_columns filters OUTCOME_COLUMNS
    unconditionally before the prefix check.

    (Note: the function's `AssertionError` backstop is therefore unreachable through any
    input -- the exclusion prevents leakage rather than raising. We assert the real,
    reachable guarantee: outcomes stay out even when the allowlist is misconfigured.)"""
    import src.models.common as common
    monkeypatch.setattr(common, "FEATURE_PREFIXES", common.FEATURE_PREFIXES + ("pdc_", "has_"))
    frame = pd.DataFrame({"patient_id": ["p1"], "age_years": [61.0],
                          "pdc_180d": [0.9], "has_30_day_gap": [1]})
    selected = set(common.select_feature_columns(frame))
    assert "age_years" in selected
    assert "pdc_180d" not in selected and "has_30_day_gap" not in selected


# --- out-of-fold AUC: reproducible + materially beats logistic (slow, ~30s) --------

def test_oof_auc_reproducible_and_beats_logistic():
    from sklearn.metrics import roc_auc_score
    from src.eval.run_auc import out_of_fold_proba
    from src.models.classifier import build_hist_gradient_boosting, build_logistic_regression, load_classification_frame

    X, y, _ = load_classification_frame(PANEL_PATH, LABELS_PATH)
    hgb1 = out_of_fold_proba(build_hist_gradient_boosting, X, y)
    hgb2 = out_of_fold_proba(build_hist_gradient_boosting, X, y)
    log = out_of_fold_proba(build_logistic_regression, X, y)

    roc_hgb, roc_hgb2, roc_log = (roc_auc_score(y, hgb1), roc_auc_score(y, hgb2), roc_auc_score(y, log))
    # Reproducible from the fixed seed (deterministic): identical to floating tolerance.
    assert abs(roc_hgb - roc_hgb2) < 1e-9
    # Materially exceeds the logistic baseline (relationship, not an exact figure).
    assert roc_hgb > roc_log + 0.10
    # Sanity band (wide, non-brittle): the synthetic-data ceiling is ~0.85 / ~0.68.
    assert 0.78 < roc_hgb < 0.92
    assert 0.60 < roc_log < 0.74


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
