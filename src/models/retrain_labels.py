"""retrain_labels.py — turn CLOSED feedback loops into retraining labels (State 4 -> training signal).

Every objectively observed loop is a fresh, real-world label we can fold into the
next training run. This module harvests them and produces a REFRESHED labels file
that can retrain the model alongside the UNCHANGED, strictly-pre-index feature panel.

What a "retraining label" is here
---------------------------------
The loop closes on an objective dispense (src/sync/loop_closure.py via the connector
layer), never on self-report. Each observed patient yields:

    label_adherent = 1   an on-time refill was observed  (-> has_30_day_gap = 0)
    label_adherent = 0   a confirmed break               (-> has_30_day_gap = 1)

plus metadata: ``source_event_date`` (when the dispense/observation landed),
``closed_on_round`` (which escalation round the loop closed on), and
``refill_source`` (which pharmacy source confirmed it, e.g. ae_pharmacy_claims_export).

LEAKAGE (non-negotiable): closed_on_round and refill_source are METADATA, never
features. They describe what happened AFTER index, so feeding them to the model
would be textbook target leakage. They live only in retrain_labels.parquet; the
refreshed labels.parquet carries only ``has_30_day_gap`` (+ the existing pdc_180d),
and feature_panel.parquet is never touched, so it stays strictly pre-index.
src/models/common.py::select_feature_columns has no allowlisted prefix for either
column and would drop them anyway -- tests/test_retrain_labels.py asserts that.

WHAT THIS PROVES, AND WHAT IT DOES NOT
--------------------------------------
Proves: the ARCHITECTURE supports continuous learning -- observed outcomes flow
back into a training-ready label set through a stable seam, with the leakage guard
intact. Does NOT prove: that retraining has been validated on real patients. On
this SYNTHETIC cohort the objective label IS the ground-truth outcome regardless of
whether we intervened, so retraining on it re-affirms the existing signal; it cannot
demonstrate intervention LIFT. A real build would need (1) server-side State-3
action gating, (2) a real dispense feed + id crosswalk, and (3) selection-bias
correction (closed-loop labels are only the routed/actioned subset) before these
labels re-enter training. ``mode="production"`` therefore still raises.

Public API
----------
build_retraining_labels(mode="demo") -> path to retrain_labels.parquet (or raises for production)
refresh_labels()                     -> path to labels_retrained.parquet (features untouched)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOOP_OUTCOMES_PATH = ROOT / "data" / "snapshots" / "loop_outcomes.json"
ESCALATION_STATE_PATH = ROOT / "data" / "snapshots" / "escalation_state.json"
RETRAIN_LABELS_PATH = ROOT / "data" / "snapshots" / "retrain_labels.parquet"
LABELS_PATH = ROOT / "labels.parquet"
REFRESHED_LABELS_PATH = ROOT / "data" / "snapshots" / "labels_retrained.parquet"

# Columns of retrain_labels.parquet. label_adherent is the y; the rest are metadata
# that must NEVER become features (see the LEAKAGE note above).
RETRAIN_COLUMNS = ["patient_id", "label_adherent", "source_event_date",
                   "closed_on_round", "refill_source", "loop_closed"]


def _load_escalation_outcomes(path: Path) -> dict:
    """{patient_id -> objective_outcome (+ closed_on_round)} from escalation_state.json.

    escalation_state.json is the richest source: its per-patient objective_outcome
    already carries closed_on_round. Returns {} if absent (fall back to loop_outcomes).
    """
    if not Path(path).exists():
        return {}
    payload = json.load(open(path))
    out = {}
    for pid, st in payload.get("patients", {}).items():
        oc = st.get("objective_outcome")
        if oc and oc.get("observed"):
            out[str(pid)] = oc
    return out


def _load_loop_outcomes(path: Path) -> dict:
    """{patient_id -> outcome} from loop_outcomes.json (no closed_on_round). Fallback."""
    if not Path(path).exists():
        return {}
    return json.load(open(path)).get("outcomes", {})


def _harvest(escalation_path: Path, loop_outcomes_path: Path, actioned_ids: Optional[set]) -> list[dict]:
    """Collect one label row per OBSERVED patient. Prefers escalation state (has
    closed_on_round); falls back to loop_outcomes.json when escalation state is absent."""
    esc = _load_escalation_outcomes(escalation_path)
    src = esc if esc else _load_loop_outcomes(loop_outcomes_path)
    if not src:
        raise FileNotFoundError(
            f"neither {escalation_path} nor {loop_outcomes_path} found -- run "
            "`python -m src.sync.loop_closure` and `python -m src.routing.escalation` first.")

    rows = []
    for pid, oc in src.items():
        if not oc.get("observed"):
            continue
        if actioned_ids is not None and pid not in actioned_ids:
            continue  # gate on server-side action state when available (production shape)
        rows.append({
            "patient_id": str(pid),
            "label_adherent": int(bool(oc.get("on_time_refill"))),
            "source_event_date": oc.get("event_date"),
            "closed_on_round": oc.get("closed_on_round"),   # metadata, never a feature
            "refill_source": oc.get("refill_source") or oc.get("source"),  # metadata, never a feature
            "loop_closed": True,
        })
    return rows


def build_retraining_labels(
    *,
    mode: str = "demo",
    escalation_state_path: Path = ESCALATION_STATE_PATH,
    loop_outcomes_path: Path = LOOP_OUTCOMES_PATH,
    out_path: Path = RETRAIN_LABELS_PATH,
    actioned_ids: Optional[set] = None,
) -> Path:
    """Harvest closed-loop labels for the next training run.

    ``demo`` writes candidate labels from the objective outcomes (NOT action-gated
    unless ``actioned_ids`` is supplied). ``production`` raises NotImplementedError:
    it needs server-side State-3 action state, a real closure feed + id crosswalk,
    and selection-bias correction (see module docstring).
    """
    if mode == "production":
        raise NotImplementedError(
            "Production retraining-label harvest is not wired: it needs (1) server-side "
            "State-3 action state, (2) a real dispense feed + id crosswalk, and (3) "
            "selection-bias correction before the labels re-enter training. See the "
            "module docstring.")
    if mode != "demo":
        raise ValueError(f"unknown mode {mode!r} (expected 'demo' or 'production')")

    rows = _harvest(escalation_state_path, loop_outcomes_path, actioned_ids)
    df = pd.DataFrame(rows, columns=RETRAIN_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def refresh_labels(
    retrain_labels_path: Path = RETRAIN_LABELS_PATH,
    labels_path: Path = LABELS_PATH,
    out_path: Path = REFRESHED_LABELS_PATH,
) -> Path:
    """Merge harvested labels into a refreshed labels file for retraining.

    For each OBSERVED patient, ``has_30_day_gap`` is set from the observed outcome
    (adherent -> 0, break -> 1); unobserved patients keep their existing label. Only
    the label column is touched -- **feature_panel.parquet is never read or written**,
    so features stay strictly pre-index. The metadata columns (closed_on_round,
    refill_source) are deliberately NOT carried into this file. The result is a
    drop-in replacement for labels.parquet in a retraining run
    (``classifier.load_classification_frame(PANEL_PATH, labels_retrained.parquet)``).
    """
    harvested = pd.read_parquet(retrain_labels_path)
    labels = pd.read_parquet(labels_path).copy()
    # observed patient -> new has_30_day_gap (1 - adherent)
    new_gap = {str(pid): 1 - int(adh)
               for pid, adh in zip(harvested["patient_id"], harvested["label_adherent"])}
    pid_str = labels["patient_id"].astype(str)
    labels["has_30_day_gap"] = [
        new_gap.get(p, g) for p, g in zip(pid_str, labels["has_30_day_gap"])
    ]
    # Only the identity + label columns survive; NO feature/metadata columns added.
    keep = [c for c in ("patient_id", "pdc_180d", "has_30_day_gap") if c in labels.columns]
    refreshed = labels[keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refreshed.to_parquet(out_path, index=False)
    return out_path


def main() -> int:
    path = build_retraining_labels(mode="demo")
    df = pd.read_parquet(path)
    n_pos = int(df["label_adherent"].sum())
    by_round = df["closed_on_round"].value_counts(dropna=False).to_dict()
    print(f"[demo, NOT action-gated] wrote {len(df)} closed-loop retraining labels "
          f"({n_pos} adherent / {len(df) - n_pos} break) -> {path}")
    print(f"  closed_on_round distribution (metadata, NOT a feature): {by_round}")
    refreshed = refresh_labels()
    rc = pd.read_parquet(refreshed)
    print(f"  refreshed labels for retraining: {len(rc)} rows, columns {list(rc.columns)} "
          f"-> {refreshed}")
    print("  feature_panel.parquet was NOT touched (features stay strictly pre-index).")
    print("Production harvest is stubbed (NotImplementedError) -- see module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
