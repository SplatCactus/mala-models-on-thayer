"""common.py — shared, model-family-agnostic utilities for src/models/.

Extracted out of survival.py (2026-07-02) when classifier.py needed
``select_feature_columns``/``save_model``/``load_model`` but importing them
straight from survival.py transitively required scikit-survival -- a heavy,
unrelated dependency for a module that's deliberately just sklearn. None of
what lives here touches scikit-survival; keep it that way, so any future
model family in this package can depend on common.py without dragging in
every other family's libraries.

Leakage rule
------------
:func:`select_feature_columns` is the single place the feature/outcome
leakage rule is enforced for every model in this package -- it raises
``AssertionError`` if an outcome column (:data:`OUTCOME_COLUMNS`) ever ends
up selected, and only returns columns from an explicit allowlist
(:data:`FEATURE_PREFIXES` / :data:`EXPLICIT_FEATURE_COLUMNS`), holding aside
demographic columns (:data:`HELD_ASIDE_DEMOGRAPHIC_COLUMNS`) by construction.
Both survival.py and classifier.py call this same function -- there is
exactly one leakage rule to keep correct, not one per model family.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "feature_panel.parquet"
LABELS_PATH = ROOT / "labels.parquet"

# Outcome columns from labels.parquet (src/features/pdc.py). These must
# never appear among the selected feature columns.
OUTCOME_COLUMNS = frozenset({"pdc_180d", "has_30_day_gap"})

# Identifier / non-feature columns.
NON_FEATURE_ID_COLUMNS = frozenset({"patient_id", "index_date", "Id", "split"})

# Demographic columns attached in src/etl/cohort.py "for equity strata" --
# held aside from every model by design. Not an exhaustive schema
# validation, just the columns cohort.py is known to emit.
HELD_ASIDE_DEMOGRAPHIC_COLUMNS = frozenset({
    "RACE", "ETHNICITY", "GENDER", "CITY", "STATE", "ZIP", "LAT", "LON",
    "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME", "is_deceased",
})

# Clinical/SDOH feature prefixes emitted by src/features/trajectories.py
# (sbp_*, dbp_*) and src/features/sdoh.py (flag_*).
FEATURE_PREFIXES = ("sbp_", "dbp_", "flag_")

# Non-prefixed columns explicitly allowed as features.
EXPLICIT_FEATURE_COLUMNS = frozenset({"age_years"})


def select_feature_columns(
    panel_df: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
) -> list[str]:
    """Pick the modeling feature columns out of a merged panel+labels frame.

    Only columns matching :data:`FEATURE_PREFIXES` or listed in
    :data:`EXPLICIT_FEATURE_COLUMNS` are selected. Identifier columns
    (:data:`NON_FEATURE_ID_COLUMNS`), outcome columns
    (:data:`OUTCOME_COLUMNS`), and held-aside demographics
    (:data:`HELD_ASIDE_DEMOGRAPHIC_COLUMNS`) are always excluded.

    Raises
    ------
    AssertionError
        If an outcome column ever ends up selected -- this is the explicit
        code-level enforcement of the feature/outcome leakage rule, not just
        a naming convention.
    """
    selected = [
        col for col in panel_df.columns
        if col != patient_id_col
        and col not in NON_FEATURE_ID_COLUMNS
        and col not in OUTCOME_COLUMNS
        and col not in HELD_ASIDE_DEMOGRAPHIC_COLUMNS
        and (col.startswith(FEATURE_PREFIXES) or col in EXPLICIT_FEATURE_COLUMNS)
    ]

    leaked = OUTCOME_COLUMNS & set(selected)
    if leaked:
        raise AssertionError(
            f"outcome column(s) {sorted(leaked)} selected as model features -- leakage"
        )
    return selected


def save_model(model: Pipeline, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> Pipeline:
    return joblib.load(Path(path))
