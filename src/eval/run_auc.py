"""run_auc.py — honest, reproducible out-of-fold evaluation for the classifiers.

This is the *single source of truth* for how well the adherence models actually
discriminate. It scores every model **strictly out-of-fold**: for each of 5
stratified folds we fit on the training rows and predict only the held-out rows,
so no patient is ever scored by a model that saw them in training. That is the
difference between the honest ROC-AUC (~0.65) and the misleading in-sample
number (~0.93) a refit-on-everything RandomForest reports by memorization.

What it reports, per model
--------------------------
  * ROC-AUC            — out-of-fold discrimination on the full cohort.
  * PR-AUC (FAILURE)   — average precision for the *non-adherent* minority, the
                         class we actually care about. Because y=1 means
                         *adherent*, the failure class is ``1 - y`` scored by
                         ``1 - proba`` (``average_precision_score(1-y, 1-p)``),
                         reported next to its no-skill baseline ``(1 - y).mean()``
                         so a lift over chance is visible at a glance.
  * ROC-AUC by ETHNICITY — the same out-of-fold predictions sliced by the
                         ETHNICITY column held aside in feature_panel.parquet
                         (never a model input), to check for a fairness gap.

Results are written to data/snapshots/auc_report.json.

Run
---
    ./venv/Scripts/python src/eval/run_auc.py      (or: make eval)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import the model builders + loader from src/models/. classifier.py inserts its
# own directory on sys.path (so its ``from common import ...`` resolves); we add
# it here too so ``import classifier`` works regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import classifier  # noqa: E402
from classifier import (  # noqa: E402
    MODEL_BUILDERS,
    PANEL_PATH,
    LABELS_PATH,
    load_classification_frame,
)

N_SPLITS = 5
RANDOM_STATE = 0
MIN_GROUP_N = 50  # don't report a by-ethnicity ROC-AUC on a tiny, unstable slice
REPORT_PATH = ROOT / "data" / "snapshots" / "auc_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("run_auc")


def out_of_fold_proba(
    build_model_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Return P(y=1) for every row, each predicted by a fold that did NOT train on it.

    A fresh model is built per fold; the held-out rows get that fold's
    ``predict_proba``. Every row is in exactly one validation fold, so the
    returned array has one honest prediction per row and never scores a patient
    on a model that saw them. ``shuffle=True`` is required — the cohort is
    implicitly ordered, so unshuffled folds would be non-representative chunks.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = np.full(len(y), np.nan)
    for train_idx, val_idx in cv.split(X, y):
        model = build_model_fn()
        model.fit(X.iloc[train_idx], y[train_idx])
        proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    if np.isnan(proba).any():
        # Every row belongs to exactly one fold, so this should be unreachable;
        # guard anyway so a silent NaN never poisons a downstream AUC.
        raise RuntimeError("some rows received no out-of-fold prediction")
    return proba


def ethnicity_series(patient_ids: pd.Series) -> pd.Series | None:
    """ETHNICITY per patient, aligned to ``patient_ids`` order (or None if absent).

    ETHNICITY is held aside in feature_panel.parquet (never a model feature); we
    read it straight from the panel and reindex onto the evaluation row order so
    the slice lines up with the out-of-fold predictions.
    """
    panel = pd.read_parquet(PANEL_PATH, columns=["patient_id", "ETHNICITY"])
    mapping = panel.set_index("patient_id")["ETHNICITY"]
    aligned = patient_ids.map(mapping)
    if aligned.isna().all():
        return None
    return aligned.reset_index(drop=True)


def by_ethnicity_roc(y: np.ndarray, proba: np.ndarray, ethnicity: pd.Series) -> dict[str, dict]:
    """Out-of-fold ROC-AUC within each ethnicity group with >= MIN_GROUP_N patients.

    A group is skipped (with a logged reason) if it is too small to trust or if
    its held-out rows are single-class (ROC-AUC undefined).
    """
    out: dict[str, dict] = {}
    y = np.asarray(y)
    for group, mask in ethnicity.groupby(ethnicity).groups.items():
        idx = np.asarray(mask)
        n = len(idx)
        yg = y[idx]
        if n < MIN_GROUP_N:
            out[str(group)] = {"n": int(n), "roc_auc": None, "note": "n<%d" % MIN_GROUP_N}
            continue
        if len(np.unique(yg)) < 2:
            out[str(group)] = {"n": int(n), "roc_auc": None, "note": "single-class slice"}
            continue
        out[str(group)] = {
            "n": int(n),
            "n_adherent": int(yg.sum()),
            "roc_auc": float(roc_auc_score(yg, proba[idx])),
        }
    return out


def evaluate() -> dict:
    log.info("Loading classification frame...")
    X, y, patient_ids = load_classification_frame(PANEL_PATH, LABELS_PATH)
    n_pos, n = int(y.sum()), len(y)
    log.info("  %d patients x %d features, %d adherent (%.1f%%), %d non-adherent",
             n, X.shape[1], n_pos, 100 * y.mean(), n - n_pos)

    fail_baseline = float(1 - y.mean())  # no-skill PR-AUC for the failure (non-adherent) class
    log.info("  no-skill PR-AUC baseline for the FAILURE class: %.3f", fail_baseline)

    ethnicity = ethnicity_series(patient_ids)
    if ethnicity is None:
        log.warning("  ETHNICITY not found in panel — skipping by-ethnicity ROC-AUC")

    report: dict = {
        "n_patients": n,
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "prevalence_adherent": float(y.mean()),
        "pr_auc_failure_baseline": fail_baseline,
        "cv": {"n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE},
        "models": {},
    }

    for name, build_fn in MODEL_BUILDERS.items():
        log.info("Out-of-fold %d-fold CV: %s", N_SPLITS, name)
        proba = out_of_fold_proba(build_fn, X, y)
        roc = float(roc_auc_score(y, proba))
        pr_fail = float(average_precision_score(1 - y, 1 - proba))
        log.info("  %-24s ROC-AUC %.3f | PR-AUC(fail) %.3f (baseline %.3f, lift %+.3f)",
                 name, roc, pr_fail, fail_baseline, pr_fail - fail_baseline)

        entry = {
            "roc_auc": roc,
            "pr_auc_failure": pr_fail,
            "pr_auc_failure_lift": pr_fail - fail_baseline,
        }
        if ethnicity is not None:
            entry["roc_auc_by_ethnicity"] = by_ethnicity_roc(y, proba, ethnicity)
            for grp, stats in entry["roc_auc_by_ethnicity"].items():
                if stats["roc_auc"] is not None:
                    log.info("      ethnicity=%-12s n=%5d  ROC-AUC %.3f",
                             grp, stats["n"], stats["roc_auc"])
                else:
                    log.info("      ethnicity=%-12s n=%5d  (skipped: %s)",
                             grp, stats["n"], stats["note"])
        report["models"][name] = entry

    return report


def main() -> int:
    report = evaluate()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
