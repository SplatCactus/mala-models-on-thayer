"""classifier.py — antihypertensive-adherence classifiers (GBDT primary).

Team decision (2026-07-02): go with Chris's classification framing instead of
the survival-analysis approach in survival.py.

Model choice, updated 2026-07-22
--------------------------------
The primary model is now a **gradient-boosted decision tree**
(:func:`build_hist_gradient_boosting`, sklearn ``HistGradientBoostingClassifier``).
For tabular EHR data — mixed-scale numeric features, binary flags, native
missingness, non-linear interactions — GBDT is the correct default, and it wins
empirically. On the enriched 33-feature panel (BP trajectory + SDOH + the
pre-index cross-drug behavioral features in src/features/pre_index.py), honest
out-of-fold ROC-AUC is **HGB 0.848 vs random forest 0.837 / logistic 0.681**
(see src/eval/run_auc.py). Logistic regression is kept as a lightweight,
interpretable baseline; the shallow random forest is kept only as a labeled
baseline — it is dominated by HGB and its ``max_depth=12`` overfits badly
(in-sample ROC-AUC 0.93).

The jump from the BP/SDOH-only panel (HGB 0.643) is driven by feature richness,
NOT architecture: the pre-index cross-drug refill/tenure/engagement features are
each individually weak (max univariate |AUC-0.5| ~0.15, so no single-feature
outcome leak) but combine strongly. Caveat: Synthea models medication adherence
as a stable per-patient trait, so pre-index non-antihypertensive refill behavior
predicts post-index antihypertensive adherence more cleanly than it would on
real EHR data — treat 0.848 as a synthetic-data ceiling, not a real-world one.

The original 2026-07-02 rationale said "simple, low-variance models only --
Logistic Regression or a shallow Random Forest; hold off on XGBoost/neural nets
until more data." That was written for the **n=136** incident-user cohort. The
current feature panel has **16,205** labeled patients, so the small-n rationale
is obsolete: GBDT (and, if wanted, XGBoost/LightGBM) is now appropriate.
``HistGradientBoostingClassifier`` ships with sklearn — no new dependency.

Validation stays as Chris specified: stratified K-fold CV with
``shuffle=True``, PR-AUC / F1 over accuracy given class imbalance,
``class_weight='balanced'``. Honest evaluation lives in src/eval/run_auc.py,
which scores strictly out-of-fold (never on training rows).

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
build_hist_gradient_boosting(**overrides) -> sklearn Pipeline  (primary / deployed)
build_logistic_regression(**overrides) -> sklearn Pipeline     (interpretable baseline)
build_shallow_random_forest(**overrides) -> sklearn Pipeline   (dominated baseline)
run_stratified_cv(build_model_fn, X, y, n_splits=5) -> dict of per-fold + mean/std metrics
save_model(model, path) / load_model(path)  -- re-exported from common.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
    label_col: str = "has_30_day_gap",
    pdc_threshold: float = PDC_ADHERENCE_THRESHOLD,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Load X (features) and y (binary label) for the whole cohort, no split file.

    Unlike survival.py's load_training_frame, this doesn't consume
    data/snapshots/splits.parquet -- K-fold CV re-splits the full cohort at
    training time instead of relying on a frozen partition.

    Target (updated 2026-07-22)
    ---------------------------
    Defaults to ``has_30_day_gap`` -- the discontinuation event we actually
    predict: ``y=1`` iff the patient has a >=30-day uncovered stretch in the
    180-day post-index window (an already-binary column, taken as-is). This is
    the EVENT-positive polarity (y=1 = the bad outcome), congruent with the
    routing pipeline's at-risk label and with predicted_risk = P(event).

    Why not ``pdc_180d >= 0.80``? That older default made ``y=1`` mean *adherent*
    (the opposite polarity), which forced every downstream metric to flip via
    ``1-y``. In this cohort the two coincide for all but 6/16205 patients (a
    low-PDC patient on 30-day fills essentially always has one >=30-day gap), so
    the numbers are unchanged -- but the gap target is the congruent framing.
    Pass ``label_col="pdc_180d"`` to recover the continuous-PDC threshold label
    (binarized at ``pdc_threshold``, ``y=1`` = adherent).

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


def build_hist_gradient_boosting(**overrides) -> Pipeline:
    """Histogram-based gradient-boosted trees -- the primary / deployed model.

    GBDT is the right default for tabular EHR data: it handles mixed-scale
    numeric + binary-flag features, native missingness, and non-linear feature
    interactions, and it is the empirical winner (out-of-fold ROC-AUC 0.848 vs
    random forest 0.837 / logistic 0.681 on the 33-feature panel; see
    src/eval/run_auc.py).

    No ``SimpleImputer``/``StandardScaler`` in the pipeline, unlike the logistic
    baseline: ``HistGradientBoostingClassifier`` bins features internally, so it
    is scale-invariant and treats NaN as a first-class split direction. Imputing
    would only discard the "value is missing" signal it already exploits.
    """
    params = dict(
        max_depth=6, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
        class_weight="balanced", random_state=0,
    )
    params.update(overrides)
    return Pipeline([
        ("clf", HistGradientBoostingClassifier(**params)),
    ])


def build_shallow_random_forest(**overrides) -> Pipeline:
    """Random Forest (max_depth=12) -- kept only as a labeled, dominated baseline.

    Strictly dominated by :func:`build_hist_gradient_boosting` and prone to
    overfitting at this depth (in-sample ROC-AUC ~0.93 vs out-of-fold ~0.63).
    Retained for comparison in the eval report, not for deployment.
    """
    params = dict(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=0, n_jobs=-1,
    )
    params.update(overrides)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(**params)),
    ])


# HGB first: it is the primary model and the deployed artifact (main() saves
# every builder's fit, but HGB is the one to serve). Logistic and RF follow as
# an interpretable baseline and a dominated baseline respectively.
MODEL_BUILDERS = {
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "logistic_regression": build_logistic_regression,
    "shallow_random_forest": build_shallow_random_forest,
}

# The model family served in production; main() logs this explicitly.
DEPLOYED_MODEL = "hist_gradient_boosting"


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
    log.info("  %d patients x %d features, %d/%d positive (has_30_day_gap = the >=30d "
             "discontinuation event, %.1f%%)", *X.shape, int(y.sum()), len(y), 100 * y.mean())

    baseline_pr_auc = float(y.mean())  # a no-skill classifier's expected PR-AUC (event rate)
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
        log.info("  saved %s%s", out_path, "  <- deployed" if name == DEPLOYED_MODEL else "")

    log.info("Deployed model artifact: %s.joblib", DEPLOYED_MODEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
