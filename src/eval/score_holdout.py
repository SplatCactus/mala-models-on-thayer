"""score_holdout.py — apply an ALREADY-TRAINED model to a holdout panel. No fit.

Inference-only generalization check: load a frozen joblib model artifact
(trained elsewhere, e.g. models/shallow_random_forest.joblib) and score a
separately-built feature panel (e.g. the 1k data/parquet cohort) with its
weights untouched. This never calls ``fit`` — same features, same weights,
different patients — so it measures how the existing model generalizes, not a
newly-fit one.

Feature alignment
-----------------
A saved sklearn Pipeline remembers the exact columns it was trained on
(``feature_names_in_``). We select precisely those from the holdout panel, in
training order, so a panel that happens to carry newer engineered columns still
scores against the model's original feature contract (extra columns ignored, a
missing one is a hard error rather than a silent zero).

Run
---
    ./venv/Scripts/python src/eval/score_holdout.py \
        --model models/shallow_random_forest.joblib \
        --panel feature_panel_1k.parquet --labels labels_1k.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from common import load_model  # noqa: E402

PDC_ADHERENCE_THRESHOLD = 0.80
MIN_GROUP_N = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("score_holdout")


def _model_feature_names(model) -> list[str]:
    """The exact feature columns the pipeline was trained on, in order."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:  # fall back to the first fitted step
        names = getattr(model[0], "feature_names_in_", None)
    if names is None:
        raise ValueError(
            "model does not expose feature_names_in_ — cannot align holdout columns "
            "to the training contract without risking silent misalignment")
    return list(names)


def score(model_path: Path, panel_path: Path, labels_path: Path,
          *, patient_id_col: str = "patient_id", label_col: str = "pdc_180d") -> dict:
    model = load_model(model_path)
    feat_cols = _model_feature_names(model)
    log.info("Model %s trained on %d features", model_path.name, len(feat_cols))

    panel = pd.read_parquet(panel_path)
    labels = pd.read_parquet(labels_path)
    frame = panel.merge(labels, on=patient_id_col, how="inner")
    frame = frame.dropna(subset=[label_col]).reset_index(drop=True)

    missing = [c for c in feat_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"holdout panel is missing model feature column(s): {missing}")

    X = frame[feat_cols].copy()          # exact training columns, exact order
    raw = frame[label_col]
    is_binary = raw.dropna().isin((0, 1)).all()
    y = raw.astype(int).to_numpy() if is_binary \
        else (raw.to_numpy() >= PDC_ADHERENCE_THRESHOLD).astype(int)

    proba = model.predict_proba(X)[:, 1]  # inference only — no fit
    roc = float(roc_auc_score(y, proba))
    fail_baseline = float(1 - y.mean())
    pr_fail = float(average_precision_score(1 - y, 1 - proba))
    log.info("  n=%d, adherent=%d (%.1f%%)", len(y), int(y.sum()), 100 * y.mean())
    log.info("  ROC-AUC %.3f | PR-AUC(fail) %.3f (baseline %.3f, lift %+.3f)",
             roc, pr_fail, fail_baseline, pr_fail - fail_baseline)

    result = {
        "model": model_path.name,
        "panel": panel_path.name,
        "n_patients": int(len(y)),
        "prevalence_adherent": float(y.mean()),
        "roc_auc": roc,
        "pr_auc_failure": pr_fail,
        "pr_auc_failure_baseline": fail_baseline,
        "pr_auc_failure_lift": pr_fail - fail_baseline,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
    }

    if "ETHNICITY" in frame.columns:
        result["roc_auc_by_ethnicity"] = {}
        for grp, sub in frame.groupby("ETHNICITY").groups.items():
            idx = np.asarray(sub)
            yg = y[idx]
            if len(idx) < MIN_GROUP_N or len(np.unique(yg)) < 2:
                result["roc_auc_by_ethnicity"][str(grp)] = {"n": int(len(idx)), "roc_auc": None}
                continue
            g_roc = float(roc_auc_score(yg, proba[idx]))
            result["roc_auc_by_ethnicity"][str(grp)] = {"n": int(len(idx)), "roc_auc": g_roc}
            log.info("      ethnicity=%-12s n=%4d  ROC-AUC %.3f", grp, len(idx), g_roc)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "shallow_random_forest.joblib")
    ap.add_argument("--panel", type=Path, default=ROOT / "feature_panel_1k.parquet")
    ap.add_argument("--labels", type=Path, default=ROOT / "labels_1k.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "snapshots" / "holdout_1k_report.json")
    args = ap.parse_args()

    result = score(args.model, args.panel, args.labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
