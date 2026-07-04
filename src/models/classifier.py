"""classifier.py — binary classification baseline for antihypertensive persistence.

Team decision (2026-07-02): go with Chris's classification framing instead of
the survival-analysis approach in survival.py, now that the incident-user
cohort fix (build_features.py, commit fee55b9) shrank the cohort from 262 to
136 patients. Rationale from his email:

  * Validation: stratified K-fold CV (not a single held-out split) -- 136
    patients is too small for a single temporal test set to give a stable
    estimate (the 262-patient survival run already hit 1 event in its val
    split; 136 patients would be worse).
  * Models: simple, low-variance models only -- Logistic Regression or a
    shallow Random Forest. Hold off on XGBoost/neural nets until more data.
  * Metrics: PR-AUC / F1 instead of accuracy, given class imbalance.
    class_weight='balanced'.

This also sidesteps survival.py's biggest documented limitation for free:
that module needed a duration/event proxy derived from pdc_180d because
pdc.py has no real gap-onset timestamp (see its "KNOWN LIMITATION" section).
A plain classifier just uses has_30_day_gap directly -- no proxy needed.

Leakage rule enforced here
---------------------------
Imports ``select_feature_columns`` from common.py (same allowlist, same
outcome-column guard survival.py uses) rather than re-implementing it -- one
leakage rule, one place it can break, shared across both model families.
common.py has no scikit-survival dependency, so this module can run without
it -- importing straight from survival.py instead would transitively pull
in scikit-survival just for this one function, which defeats the point of a
lighter classification pipeline (this was a real bug during development,
not a hypothetical -- see git history). Row-level feature/outcome date
leakage (features strictly before index_date, outcome strictly after) is
enforced upstream in src/features/*.py and is unaffected by this module;
see tests/test_leakage.py, which still applies unchanged.

What's different from splits.py's temporal design, on purpose
-----------------------------------------------------------------
splits.py enforces chronological train < val < test ordering by index_date.
Stratified K-fold CV does not preserve that -- folds are randomly shuffled
(stratified only on the outcome label), so a given fold's "train" rows can
have a later index_date than its "val" rows. This is an accepted trade-off,
not an oversight: at n=136 a temporal split's per-split event counts are too
small to trust, and Chris's ask was explicitly for CV instead. splits.py is
left as-is (unused by this module) for whenever the cohort is large enough
to support a temporal split again.

Public API
----------
load_classification_frame(panel_path, labels_path, ...) -> (X, y, patient_ids)
build_logistic_regression(**overrides) -> sklearn Pipeline
build_shallow_random_forest(**overrides) -> sklearn Pipeline
run_stratified_cv(build_model_fn, X, y, n_splits=5) -> dict of per-fold + mean/std metrics
save_model(model, path) / load_model(path)  -- re-exported from common.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    PANEL_PATH,
    LABELS_PATH,
    select_feature_columns,
    save_model,
    load_model,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
N_SPLITS = 5

# Standard PDC adherence cutoff (CMS/PQA quality measures): a patient is
# "adherent" over the 180-day window iff pdc_180d >= 0.80. This is the model
# target -- note the positive class is *adherent*, the opposite polarity of the
# has_30_day_gap label (where 1 meant a discontinuation gap).
PDC_ADHERENCE_THRESHOLD = 0.80

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("classifier")


def load_classification_frame(
    panel_path: Path,
    labels_path: Path,
    *,
    patient_id_col: str = "patient_id",
    label_col: str = "pdc_180d",
    pdc_threshold: float = PDC_ADHERENCE_THRESHOLD,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Load X (features) and y (binary label) for the whole cohort, no split file.

    Unlike survival.py's load_training_frame, this doesn't consume
    data/snapshots/splits.parquet -- K-fold CV re-splits the full cohort at
    training time instead of relying on a frozen partition.

    Target
    ------
    Defaults to a PDC-thresholded adherence label: ``y = (pdc_180d >=
    pdc_threshold)``, with ``pdc_threshold`` the standard 0.80 cutoff, so
    ``y=1`` means *adherent*. Any continuous label column (values not already
    in {0, 1}) is binarized the same way; an already-binary column such as
    ``has_30_day_gap`` is taken as-is -- pass ``label_col="has_30_day_gap"``
    to recover the old discontinuation-gap target (where ``y=1`` means a gap,
    the opposite polarity).

    Returns
    -------
    X : pandas.DataFrame
        Feature columns only (see :func:`survival.select_feature_columns`).
    y : numpy.ndarray of int
        Binary target as 0/1 (see Target above).
    patient_ids : pandas.Series
        Same row order as X/y.
    """
    panel = pd.read_parquet(panel_path)
    labels = pd.read_parquet(labels_path)
    frame = panel.merge(labels, on=patient_id_col, how="inner")

    n_before = len(frame)
    frame = frame.dropna(subset=[label_col]).reset_index(drop=True)
    n_dropped = n_before - len(frame)
    if n_dropped:
        log.warning("  dropping %d/%d patient(s) with an unlabeled outcome", n_dropped, n_before)
    if frame.empty:
        raise ValueError(f"all {n_before} patient(s) have an unlabeled {label_col!r} -- nothing to train on")

    feature_cols = select_feature_columns(frame, patient_id_col=patient_id_col)
    X = frame[feature_cols].copy()

    raw = frame[label_col]
    # Binarize a continuous target (e.g. pdc_180d) at the adherence threshold;
    # leave an already-0/1 column untouched. NaNs are gone (dropna above), so
    # the comparison is well-defined for every remaining row.
    is_binary = raw.dropna().isin((0, 1)).all()
    if is_binary:
        y = raw.astype(int).to_numpy()
    else:
        y = (raw.to_numpy() >= pdc_threshold).astype(int)
    patient_ids = frame[patient_id_col].reset_index(drop=True)
    return X, y, patient_ids


def build_logistic_regression(**overrides) -> Pipeline:
    """L2-penalized logistic regression, regularized harder than sklearn's default.

    C=0.1 (vs. sklearn's default 1.0) per Chris's "expect to need stronger
    regularization than usual" -- deliberately conservative given n=136.
    Pass ``penalty="l1", solver="liblinear"`` via overrides for L1 instead.
    """
    params = dict(penalty="l2", C=0.1, class_weight="balanced", max_iter=2000, random_state=0)
    params.update(overrides)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def build_shallow_random_forest(**overrides) -> Pipeline:
    """Shallow (max_depth=4) Random Forest -- deliberately low-variance for n=136."""
    params = dict(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=0, n_jobs=-1,
    )
    params.update(overrides)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(**params)),
    ])


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "shallow_random_forest": build_shallow_random_forest,
}


def run_stratified_cv(
    build_model_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = 0,
    threshold: float = 0.35,
) -> dict[str, np.ndarray | float]:
    """Stratified K-fold CV, reporting PR-AUC and F1 per fold plus mean/std.

    A fresh, unfit model is built per fold via ``build_model_fn()`` -- folds
    never share a fitted estimator. ``shuffle=True`` is required: the cohort
    is implicitly ordered (by patient_id / index_date), and StratifiedKFold
    without shuffling would carve contiguous, non-representative chunks
    instead of a random stratified sample.

    Raises
    ------
    ValueError
        If the minority class has fewer patients than ``n_splits`` --
        StratifiedKFold cannot guarantee >=1 minority-class example per fold
        otherwise. Raised with the actual counts instead of surfacing
        sklearn's own less specific error.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < n_splits:
        raise ValueError(
            f"cannot run {n_splits}-fold CV: minority class has only "
            f"{min(n_pos, n_neg)} patient(s) (n_pos={n_pos}, n_neg={n_neg})"
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pr_aucs: list[float] = []
    f1s: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        model = build_model_fn()
        model.fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        pred = (proba >= threshold).astype(int)
        pr_aucs.append(float(average_precision_score(y[val_idx], proba)))
        f1s.append(float(f1_score(y[val_idx], pred, zero_division=0)))

    pr_aucs_arr = np.array(pr_aucs)
    f1s_arr = np.array(f1s)
    return {
        "pr_auc_per_fold": pr_aucs_arr,
        "pr_auc_mean": float(pr_aucs_arr.mean()),
        "pr_auc_std": float(pr_aucs_arr.std()),
        "f1_per_fold": f1s_arr,
        "f1_mean": float(f1s_arr.mean()),
        "f1_std": float(f1s_arr.std()),
    }


def main() -> int:
    log.info("Loading classification frame...")
    X, y, _ = load_classification_frame(PANEL_PATH, LABELS_PATH)
    log.info("  %d patients x %d features, %d/%d positive (adherent, pdc_180d >= %.2f, %.1f%%)",
              *X.shape, int(y.sum()), len(y), PDC_ADHERENCE_THRESHOLD, 100 * y.mean())

    baseline_pr_auc = float(y.mean())  # a no-skill classifier's expected PR-AUC
    log.info("  no-skill baseline PR-AUC (positive rate): %.3f", baseline_pr_auc)

    for name, build_fn in MODEL_BUILDERS.items():
        log.info("Running %d-fold stratified CV: %s", N_SPLITS, name)
        metrics = run_stratified_cv(build_fn, X, y, n_splits=N_SPLITS)
        log.info(
            "  %s: PR-AUC %.3f +/- %.3f | F1 %.3f +/- %.3f",
            name, metrics["pr_auc_mean"], metrics["pr_auc_std"],
            metrics["f1_mean"], metrics["f1_std"],
        )

        # Refit on the full cohort for the deployed artifact -- CV above is
        # for evaluation only, none of these fitted folds are kept.
        model = build_fn()
        model.fit(X, y)
        out_path = MODEL_DIR / f"{name}.joblib"
        save_model(model, out_path)
        log.info("  saved %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
