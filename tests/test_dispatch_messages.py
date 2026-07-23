"""test_dispatch_messages.py — provider-addressed, four-language dispatch payloads.

Includes an ADVERSARIAL scan that actively tries to find patient-directed phrasing
in every generated body/recipient field: the provider-only claim is the core product
promise, so it gets a test that tries to break it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing.dispatch_messages import build_dispatch, RECIPIENT_TYPES, LANGS  # noqa: E402

# One card per Round-1 driver branch + safety, exercising every message kind.
DRIVERS = ["transport_barrier", "financial_barrier", "bp_trend", "housing_barrier",
           "isolation", "low_education", "migrant_status"]


def _card(driver="housing_barrier", safety=False):
    return {"patient_id": "abcd1234ef", "top_driver": driver, "is_safety_override": safety,
            "predicted_risk": 0.4, "rank_in_role": 2, "days_to_predicted_break": 40}


def _consent(external_allowed):
    return {"external_disclosure": {"allowed": external_allowed},
            "internal_care_coordination": {"allowed": True}}


PRIOR = [
    {"round": 0, "recipient_type": "ae_chw", "dispatched_on": "2026-05-01", "outcome": "no_refill"},
    {"round": 1, "recipient_type": "social_worker", "dispatched_on": "2026-06-01", "outcome": "no_refill"},
]


def _all_dispatches():
    """Every dispatch we can generate: each round x driver x external-consent state."""
    out = []
    for driver in DRIVERS:
        for ext in (True, False):
            c = _consent(ext)
            out.append(build_dispatch(0, _card(driver), [], c, {"predicted_break_date": "2026-09-01"}))
            out.append(build_dispatch(1, _card(driver), [], c, {"predicted_break_date": "2026-09-01"}))
            out.append(build_dispatch(2, _card(driver), PRIOR, c, {"predicted_break_date": "2026-09-01"}))
    out.append(build_dispatch(1, _card("trauma_exposure", safety=True), [], _consent(True), {}))
    return out


def test_all_four_languages_present_nonempty_no_silent_english_fallback():
    for d in _all_dispatches():
        body = d["body"]
        assert set(body.keys()) == set(LANGS)
        for lang in LANGS:
            assert body[lang] and body[lang].strip(), f"empty {lang} body for {d['recipient_type']}"
        # es/pt/ht must be genuine translations, not silent copies of en
        for lang in ("es", "pt", "ht"):
            assert body[lang] != body["en"], f"{lang} silently fell back to English"


def test_round0_addressed_to_chw_not_pharmacy():
    d = build_dispatch(0, _card(), [], _consent(True), {})
    assert d["recipient_type"] == "ae_chw"
    assert d["mediated_by"] == "pharmacy"          # the CHW contacts the pharmacy
    en = d["body"]["en"]
    assert en.startswith("To the patient's Community Health Worker")
    assert "contact the patient's pharmacy" in en
    assert not en.lower().startswith("to the pharmacy")


def test_round2_body_contains_full_prior_history():
    d = build_dispatch(2, _card(), PRIOR, _consent(True), {})
    assert d["recipient_type"] == "prescriber"
    en = d["body"]["en"]
    # every prior round, its recipient, date, and outcome must be in the body
    assert "Round 0" in en and "Round 1" in en
    assert "Community Health Worker" in en and "Social Worker" in en
    assert "2026-05-01" in en and "2026-06-01" in en
    assert "no refill observed" in en
    assert d["intervention_history"] == PRIOR


def test_read_aloud_script_only_for_chw_or_prescriber_and_labeled():
    # bilingual CHW gets a labeled read-aloud script...
    d_chw = build_dispatch(1, _card("low_education"), [], _consent(True), {})
    assert d_chw["recipient_type"] == "bilingual_chw"
    assert d_chw["read_aloud_script"] is not None
    assert "READ-ALOUD SCRIPT" in d_chw["read_aloud_script"]["en"]
    # ...a pharmacist does not
    d_ph = build_dispatch(1, _card("financial_barrier"), [], _consent(True), {})
    assert d_ph["recipient_type"] == "pharmacist"
    assert d_ph["read_aloud_script"] is None


# ---------------------------------------------------------------------------
# ADVERSARIAL: no dispatch body or recipient field may address the patient.
# ---------------------------------------------------------------------------

# Provider-only closing clause each body MUST contain (per language; matches the
# "not (to) the patient" variants used across templates).
_PROVIDER_ONLY_MARKER = {
    "en": re.compile(r"not (to )?the patient", re.I),
    "es": re.compile(r"no al paciente", re.I),
    "pt": re.compile(r"não ao paciente", re.I),
    "ht": re.compile(r"pa bay pasyan an", re.I),
}
# Phrases that would betray patient-directed messaging; none may appear in a body.
_PATIENT_CONTACT_DENYLIST = [
    r"dear patient", r"estimado paciente", r"querido paciente", r"caro paciente",
    r"prezado paciente", r"chè pasyan",
    r"we('| wi)ll (text|call|email|message|sms|notify) you",
    r"\btext you\b", r"\bsms\b", r"reply stop", r"click here", r"unsubscribe",
]
_OPENERS = ("To the", "Para ", "Pou ")


def test_adversarial_no_dispatch_addresses_the_patient():
    denylist = [re.compile(p, re.I) for p in _PATIENT_CONTACT_DENYLIST]
    for d in _all_dispatches():
        # structural promise
        assert d["addressed_to"] == "provider_or_organization"
        assert d["recipient_type"] in RECIPIENT_TYPES  # never a patient recipient
        for lang, text in d["body"].items():
            # positive: the body carries a provider address opener + provider-only clause
            assert text.startswith(_OPENERS), f"body not addressed to a provider ({lang}): {text[:40]}"
            assert _PROVIDER_ONLY_MARKER[lang].search(text), \
                f"body missing provider-only clause ({lang}): {text[-60:]}"
            # negative: no patient-directed phrasing
            for pat in denylist:
                assert not pat.search(text), f"patient-directed phrasing '{pat.pattern}' in {lang} body"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
