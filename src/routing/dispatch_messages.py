"""
src/routing/dispatch_messages.py

Turns "this patient is at round N of the escalation ladder" into the concrete
dispatch payload: a recipient (always a PROVIDER or ORGANIZATION, never the
patient), a human-readable message body in all four languages, and, for a CHW or
prescriber, a clearly-labeled read-aloud script (the worker's own tool -- never a
message this system sends).

PROVIDER-ONLY, ENFORCED IN THE DATA
-----------------------------------
Every payload carries ``addressed_to = "provider_or_organization"``. There is no
patient recipient type anywhere. Round 0 is addressed to the patient's CHW with an
instruction to contact the pharmacy -- it is NEVER addressed to the pharmacy (that
would be third-party steering) and NEVER to the patient. Round 2 is addressed to
the AE prescriber and composes the COMPLETE prior history (every round, recipient,
date, outcome) -- the "tell them everything we already tried" requirement that is
the whole point of Round 2.

WHY THE STRINGS LIVE HERE (not imported from api/main.py)
---------------------------------------------------------
A routing module must not depend on the API layer, so the recipient/driver labels
and message templates are defined here, in the same field-per-language dict shape
the rest of the repo uses (see routing_table.yaml's rationale_{en,es,pt,ht}). The
recipient resolution mirrors src/routing/escalation.py's round->action map; they
are the same clinical decisions expressed in two vocabularies (escalation's consent
*action* keys vs. this module's display *recipient* types) -- keep them consistent.

Public API
----------
build_dispatch(round_num, card, prior_rounds, consent, refill_meta) -> payload dict
RECIPIENT_TYPES  the six provider/org recipient constants
"""
from __future__ import annotations

from typing import List, Optional

LANGS = ("en", "es", "pt", "ht")

# The one value addressed_to can take: a machine-checkable promise that no dispatch
# targets the patient.
ADDRESSED_TO = "provider_or_organization"

# Recipient types (task-specified). transit_voucher_external is the only one that
# discloses outside the covered entity; a gated voucher falls back to ae_chw.
RECIPIENT_AE_CHW = "ae_chw"
RECIPIENT_SOCIAL_WORKER = "social_worker"
RECIPIENT_PHARMACIST = "pharmacist"
RECIPIENT_BILINGUAL_CHW = "bilingual_chw"
RECIPIENT_TRANSIT_EXTERNAL = "transit_voucher_external"
RECIPIENT_PRESCRIBER = "prescriber"
RECIPIENT_TYPES = frozenset({
    RECIPIENT_AE_CHW, RECIPIENT_SOCIAL_WORKER, RECIPIENT_PHARMACIST,
    RECIPIENT_BILINGUAL_CHW, RECIPIENT_TRANSIT_EXTERNAL, RECIPIENT_PRESCRIBER,
})

# Round-1 dominant-driver -> recipient (mirrors escalation.ROUND1_DRIVER_ACTION).
# Unmapped drivers fall back to the social worker (model-agnostic: a model swap may
# surface a new driver).
ROUND1_DRIVER_RECIPIENT = {
    "transport_barrier": RECIPIENT_TRANSIT_EXTERNAL,
    "financial_barrier": RECIPIENT_PHARMACIST,
    "bp_trend": RECIPIENT_PHARMACIST,
    "housing_barrier": RECIPIENT_SOCIAL_WORKER,
    "isolation": RECIPIENT_SOCIAL_WORKER,
    "trauma_exposure": RECIPIENT_SOCIAL_WORKER,
    "low_education": RECIPIENT_BILINGUAL_CHW,
    "migrant_status": RECIPIENT_BILINGUAL_CHW,
}

# Recipients who get a read-aloud script (they speak with the patient directly).
_READ_ALOUD_RECIPIENTS = frozenset({RECIPIENT_AE_CHW, RECIPIENT_BILINGUAL_CHW, RECIPIENT_PRESCRIBER})

RECIPIENT_LABELS = {
    RECIPIENT_AE_CHW: {
        "en": "the patient's Community Health Worker (CHW)",
        "es": "el Trabajador Comunitario de Salud (CHW) del paciente",
        "pt": "o Agente Comunitário de Saúde (ACS) do paciente",
        "ht": "Ajan Kominotè Sante (CHW) pasyan an",
    },
    RECIPIENT_SOCIAL_WORKER: {
        "en": "Social Worker", "es": "Trabajador(a) Social",
        "pt": "Assistente Social", "ht": "Travayè Sosyal",
    },
    RECIPIENT_PHARMACIST: {
        "en": "Pharmacist", "es": "Farmacéutico(a)",
        "pt": "Farmacêutico(a)", "ht": "Famasyen",
    },
    RECIPIENT_BILINGUAL_CHW: {
        "en": "Bilingual Community Health Worker",
        "es": "Trabajador(a) Comunitario(a) de Salud bilingüe",
        "pt": "Agente Comunitário de Saúde bilingue",
        "ht": "Ajan Kominotè Sante bileng",
    },
    RECIPIENT_TRANSIT_EXTERNAL: {
        "en": "RIPTA / MTM transportation broker",
        "es": "gestor de transporte RIPTA / MTM",
        "pt": "operador de transporte RIPTA / MTM",
        "ht": "kourtye transpò RIPTA / MTM",
    },
    RECIPIENT_PRESCRIBER: {
        "en": "the patient's prescriber (within the AE)",
        "es": "el prescriptor del paciente (dentro de la ER)",
        "pt": "o prescritor do paciente (dentro da ER)",
        "ht": "preskriptè pasyan an (andedan AE a)",
    },
}

# Short driver labels per language (mirrors api/main.py's DRIVER_LABELS keys; defined
# here so a routing module doesn't import the API layer).
DRIVER_LABELS = {
    "housing_barrier": {"en": "housing instability", "es": "inestabilidad de vivienda",
                        "pt": "instabilidade habitacional", "ht": "enstabilite lojman"},
    "financial_barrier": {"en": "financial barrier", "es": "barrera financiera",
                          "pt": "barreira financeira", "ht": "baryè finansye"},
    "transport_barrier": {"en": "transportation barrier", "es": "barrera de transporte",
                          "pt": "barreira de transporte", "ht": "baryè transpò"},
    "isolation": {"en": "social isolation", "es": "aislamiento social",
                  "pt": "isolamento social", "ht": "izolman sosyal"},
    "low_education": {"en": "health literacy / education", "es": "alfabetización en salud",
                      "pt": "literacia em saúde / educação", "ht": "konesans sou sante / edikasyon"},
    "migrant_status": {"en": "migrant status", "es": "estatus migratorio",
                       "pt": "estatuto migratório", "ht": "estati imigran"},
    "bp_trend": {"en": "blood pressure trend", "es": "tendencia de presión arterial",
                 "pt": "tendência da pressão arterial", "ht": "tandans tansyon"},
    "trauma_exposure": {"en": "trauma exposure (safety)", "es": "exposición a trauma (seguridad)",
                        "pt": "exposição a trauma (segurança)", "ht": "ekspozisyon a chòk (sekirite)"},
}

OUTCOME_WORDS = {
    "pending": {"en": "in progress", "es": "en curso", "pt": "em curso", "ht": "an kou"},
    "no_refill": {"en": "no refill observed", "es": "sin recarga observada",
                  "pt": "sem recarga observada", "ht": "pa gen rechaj ki obsève"},
    "refill_observed": {"en": "refill observed", "es": "recarga observada",
                        "pt": "recarga observada", "ht": "rechaj obsève"},
    "gated": {"en": "gated on consent (fallback used)", "es": "bloqueado por consentimiento (se usó alternativa)",
              "pt": "bloqueado por consentimento (usada alternativa)", "ht": "bloke sou konsantman (itilize altènatif)"},
    "gated_internal": {"en": "gated on consent (held)", "es": "bloqueado por consentimiento (en espera)",
                       "pt": "bloqueado por consentimento (em espera)", "ht": "bloke sou konsantman (an atant)"},
}
_ROUND_WORD = {"en": "Round", "es": "Ronda", "pt": "Ronda", "ht": "Faz"}

# message_kind -> {lang: template}. Placeholders: {ref} {days} {break_date}
# {barrier} {rank} {risk_pct} {history}. Each body names the provider it is
# addressed to and, where it could be misread, states the request is not to the patient.
BODY_TEMPLATES = {
    "chw_pharmacy": {
        "en": ("To the patient's Community Health Worker: {ref} is approaching a predicted "
               "antihypertensive refill gap (~{days} days, around {break_date}). Please contact "
               "the patient's pharmacy about their refill. This dispatch is addressed to you, the "
               "CHW -- not to the patient, and not to the pharmacy directly."),
        "es": ("Para el Trabajador Comunitario de Salud del paciente: {ref} se acerca a una "
               "interrupción prevista en la recarga de antihipertensivos (~{days} días, alrededor "
               "del {break_date}). Comuníquese con la farmacia del paciente sobre su recarga. Esta "
               "solicitud se dirige a usted, el CHW, no al paciente ni directamente a la farmacia."),
        "pt": ("Para o Agente Comunitário de Saúde do paciente: {ref} aproxima-se de uma interrupção "
               "prevista na recarga de anti-hipertensivos (~{days} dias, por volta de {break_date}). "
               "Contacte a farmácia do paciente sobre a recarga. Este pedido destina-se a si, o ACS, "
               "não ao paciente nem diretamente à farmácia."),
        "ht": ("Pou Ajan Kominotè Sante pasyan an: {ref} ap pwoche yon koupe nou prevwa nan rechaj "
               "medikaman tansyon (~{days} jou, ozalantou {break_date}). Tanpri kontakte famasi pasyan "
               "an sou rechaj li. Demann sa a ale ba ou, CHW a -- pa bay pasyan an, ni dirèkteman bay famasi a."),
    },
    "social_worker": {
        "en": ("To the Social Worker: {ref} did not refill after the pharmacy round and remains at "
               "risk of an antihypertensive gap (predicted around {break_date}). Dominant barrier: "
               "{barrier}. Please connect this patient with resources for it. Addressed to you as a "
               "provider, not to the patient."),
        "es": ("Para el/la Trabajador(a) Social: {ref} no recargó tras la ronda de farmacia y sigue "
               "en riesgo de una interrupción (prevista alrededor del {break_date}). Barrera principal: "
               "{barrier}. Conecte a este paciente con recursos. Dirigido a usted como proveedor, no al paciente."),
        "pt": ("Para o/a Assistente Social: {ref} não recarregou após a ronda da farmácia e continua "
               "em risco de uma interrupção (prevista por volta de {break_date}). Barreira principal: "
               "{barrier}. Ligue este paciente a recursos. Dirigido a si como profissional, não ao paciente."),
        "ht": ("Pou Travayè Sosyal la: {ref} pa t rechaje apre faz famasi a e li toujou nan risk yon "
               "koupe (nou prevwa ozalantou {break_date}). Pi gwo baryè a: {barrier}. Konekte pasyan an "
               "ak resous. Voye ba ou kòm founisè, pa bay pasyan an."),
    },
    "pharmacist": {
        "en": ("To the Pharmacist: {ref} did not refill after the pharmacy round and remains at risk "
               "(predicted around {break_date}). Dominant barrier: {barrier}. Please review the regimen "
               "and consider refill synchronization or a 90-day mail-order supply. Addressed to you as a "
               "provider, not to the patient."),
        "es": ("Para el/la Farmacéutico(a): {ref} no recargó tras la ronda de farmacia y sigue en riesgo "
               "(prevista alrededor del {break_date}). Barrera principal: {barrier}. Revise el tratamiento y "
               "considere sincronización de recargas o suministro de 90 días por correo. Dirigido a usted como "
               "proveedor, no al paciente."),
        "pt": ("Para o/a Farmacêutico(a): {ref} não recarregou após a ronda da farmácia e continua em risco "
               "(prevista por volta de {break_date}). Barreira principal: {barrier}. Reveja o regime e pondere "
               "sincronização de recargas ou fornecimento de 90 dias por correio. Dirigido a si como profissional, "
               "não ao paciente."),
        "ht": ("Pou Famasyen an: {ref} pa t rechaje apre faz famasi a e li toujou nan risk (nou prevwa ozalantou "
               "{break_date}). Pi gwo baryè a: {barrier}. Revize tretman an epi konsidere senkronize rechaj yo oswa "
               "yon rezèv 90 jou pa lapòs. Voye ba ou kòm founisè, pa bay pasyan an."),
    },
    "bilingual_chw": {
        "en": ("To the Bilingual Community Health Worker: {ref} remains at risk (predicted around "
               "{break_date}); dominant barrier {barrier}. Please provide literacy-adapted, teach-back "
               "medication counseling in the patient's preferred language. Addressed to you as a provider, "
               "not to the patient."),
        "es": ("Para el/la Trabajador(a) Comunitario(a) de Salud bilingüe: {ref} sigue en riesgo (prevista "
               "alrededor del {break_date}); barrera principal {barrier}. Brinde asesoría de medicación adaptada "
               "a la alfabetización, con teach-back, en el idioma preferido del paciente. Dirigido a usted como "
               "proveedor, no al paciente."),
        "pt": ("Para o/a Agente Comunitário de Saúde bilingue: {ref} continua em risco (prevista por volta de "
               "{break_date}); barreira principal {barrier}. Preste aconselhamento adaptado à literacia, com "
               "teach-back, no idioma preferido do paciente. Dirigido a si como profissional, não ao paciente."),
        "ht": ("Pou Ajan Kominotè Sante bileng lan: {ref} toujou nan risk (nou prevwa ozalantou {break_date}); pi "
               "gwo baryè {barrier}. Bay konsèy sou medikaman ki adapte ak nivo konesans, ak teach-back, nan lang "
               "pasyan an pi renmen. Voye ba ou kòm founisè, pa bay pasyan an."),
    },
    "transit_voucher": {
        "en": ("To the RIPTA / MTM transportation broker: {ref} remains at risk (predicted around "
               "{break_date}); the barrier is {barrier}. Please arrange a bus voucher or non-emergency "
               "medical transport (Medicaid-eligible) so this patient can reach the pharmacy. This is an "
               "external disclosure made under signed patient authorization; addressed to the broker, not the patient."),
        "es": ("Para el gestor de transporte RIPTA / MTM: {ref} sigue en riesgo (prevista alrededor del "
               "{break_date}); la barrera es {barrier}. Gestione un vale de autobús o transporte médico no urgente "
               "(elegible por Medicaid) para que el paciente llegue a la farmacia. Divulgación externa bajo "
               "autorización firmada; dirigido al gestor, no al paciente."),
        "pt": ("Para o operador de transporte RIPTA / MTM: {ref} continua em risco (prevista por volta de "
               "{break_date}); a barreira é {barrier}. Organize um vale de autocarro ou transporte médico não urgente "
               "(elegível pelo Medicaid) para o paciente chegar à farmácia. Divulgação externa sob autorização "
               "assinada; dirigido ao operador, não ao paciente."),
        "ht": ("Pou kourtye transpò RIPTA / MTM lan: {ref} toujou nan risk (nou prevwa ozalantou {break_date}); "
               "baryè a se {barrier}. Ranje yon bon otobis oswa transpò medikal ki pa ijans (kalifye pou Medicaid) "
               "pou pasyan an ka rive nan famasi a. Se yon divilgasyon deyò sou otorizasyon siyen; voye bay kourtye a, "
               "pa bay pasyan an."),
    },
    "chw_transport_fallback": {
        "en": ("To the patient's Community Health Worker: {ref} has a transportation barrier ({barrier}) but is "
               "NOT authorized for external disclosure to a transit broker, so the transit-voucher route is gated. "
               "Please arrange transportation support internally (within the AE) instead. Addressed to you, the CHW, "
               "not to the patient."),
        "es": ("Para el Trabajador Comunitario de Salud del paciente: {ref} tiene una barrera de transporte ({barrier}) "
               "pero NO está autorizada la divulgación externa a un gestor de transporte, así que la vía del vale queda "
               "bloqueada. Gestione apoyo de transporte internamente (dentro de la ER). Dirigido a usted, el CHW, no al paciente."),
        "pt": ("Para o Agente Comunitário de Saúde do paciente: {ref} tem uma barreira de transporte ({barrier}) mas "
               "NÃO tem autorização para divulgação externa a um operador de transporte, pelo que a via do vale fica "
               "bloqueada. Organize apoio de transporte internamente (dentro da ER). Dirigido a si, o ACS, não ao paciente."),
        "ht": ("Pou Ajan Kominotè Sante pasyan an: {ref} gen yon baryè transpò ({barrier}) men li PA otorize pou "
               "divilgasyon deyò bay yon kourtye transpò, kidonk wout bon transpò a bloke. Ranje sipò transpò anndan "
               "(nan AE a) pito. Voye ba ou, CHW a, pa bay pasyan an."),
    },
    "prescriber": {
        "en": ("To the patient's prescriber (within the AE): we flagged {ref} for elevated risk of a sustained "
               "antihypertensive gap (predicted around {break_date}; priority rank {rank}, predicted risk {risk_pct}). "
               "Dominant barrier: {barrier}. Interventions already attempted: {history}. Earlier rounds did not result "
               "in an observed refill, so we are escalating to you -- you have full record access (address, emergency "
               "contacts, employer), so please reach this patient through whatever channel works, or route onward if "
               "their provider is outside our AE. Addressed to you as a provider, not to the patient."),
        "es": ("Para el prescriptor del paciente (dentro de la ER): identificamos a {ref} por riesgo elevado de una "
               "interrupción sostenida (prevista alrededor del {break_date}; rango de prioridad {rank}, riesgo {risk_pct}). "
               "Barrera principal: {barrier}. Intervenciones ya intentadas: {history}. Las rondas anteriores no lograron "
               "una recarga observada, por lo que escalamos a usted: tiene acceso completo al expediente (dirección, "
               "contactos de emergencia, empleador); comuníquese por el canal que funcione o derive si su proveedor está "
               "fuera de nuestra ER. Dirigido a usted como proveedor, no al paciente."),
        "pt": ("Para o prescritor do paciente (dentro da ER): sinalizámos {ref} por risco elevado de uma interrupção "
               "sustentada (prevista por volta de {break_date}; posição de prioridade {rank}, risco {risk_pct}). Barreira "
               "principal: {barrier}. Intervenções já tentadas: {history}. As rondas anteriores não resultaram numa recarga "
               "observada, pelo que escalamos para si -- tem acesso completo ao registo (morada, contactos de emergência, "
               "entidade patronal); contacte pelo canal que funcionar ou encaminhe se o profissional estiver fora da nossa "
               "ER. Dirigido a si como profissional, não ao paciente."),
        "ht": ("Pou preskriptè pasyan an (andedan AE a): nou make {ref} pou gwo risk yon koupe kontinyèl (nou prevwa ozalantou "
               "{break_date}; ran priyorite {rank}, risk {risk_pct}). Pi gwo baryè: {barrier}. Entèvansyon nou deja eseye: "
               "{history}. Faz anvan yo pa t bay yon rechaj ki obsève, kidonk n ap eskalade ba ou -- ou gen aksè konplè ak "
               "dosye a (adrès, kontak ijans, anplwayè); kontakte pasyan an nan nenpòt kanal ki mache, oswa voye l pi lwen si "
               "founisè li deyò AE nou an. Voye ba ou kòm founisè, pa bay pasyan an."),
    },
}

# One labeled read-aloud script (the worker adapts it); the same tool for any CHW or
# prescriber recipient. Explicitly NOT a message the system sends.
_READ_ALOUD_SCRIPT = {
    "en": ("[READ-ALOUD SCRIPT -- the CHW's/prescriber's own tool to use with the patient in person or by "
           "phone; NOT a message sent by this system.] \"Hi, I'm reaching out from your care team about your "
           "blood-pressure medication. I want to make sure nothing is getting in the way of your refill -- can "
           "we talk about what would help?\""),
    "es": ("[GUION PARA LEER EN VOZ ALTA -- herramienta del CHW/prescriptor para usar con el paciente en persona "
           "o por teléfono; NO es un mensaje enviado por este sistema.] \"Hola, le llamo de su equipo de atención "
           "por su medicamento para la presión. Quiero asegurarme de que nada dificulte su recarga; ¿podemos hablar "
           "de lo que le ayudaría?\""),
    "pt": ("[GUIÃO PARA LER EM VOZ ALTA -- ferramenta do ACS/prescritor para usar com o paciente presencialmente ou "
           "por telefone; NÃO é uma mensagem enviada por este sistema.] \"Olá, contacto-o da sua equipa de cuidados "
           "sobre o seu medicamento para a tensão. Quero garantir que nada dificulta a sua recarga; podemos falar "
           "sobre o que ajudaria?\""),
    "ht": ("[ESKRIP POU LI AWOTE -- zouti CHW/preskriptè a pou itilize ak pasyan an an pèsòn oswa nan telefòn; se PA "
           "yon mesaj sistèm nan voye.] \"Bonjou, m ap rele w soti nan ekip swen ou pou medikaman tansyon ou. Mwen vle "
           "asire anyen pa bloke rechaj ou -- èske nou ka pale sou sa ki ta ede w?\""),
}


def _patient_ref(card: dict) -> str:
    """De-identified reference, matching api/main.py's 'Patient #<id8>' convention."""
    return f"Patient #{str(card['patient_id'])[:8]}"


def _resolve(round_num: int, top_driver: str, is_safety: bool, external_allowed: bool):
    """Resolve (recipient_type, message_kind, mediated_by) for a round.

    ``mediated_by`` names the org the recipient will contact on the patient's behalf
    (Round 0: the pharmacy; a gated transport fallback: internal transportation) --
    it makes the CHW-mediated hop explicit and is None when the recipient acts directly.
    """
    if round_num == 0:
        return RECIPIENT_AE_CHW, "chw_pharmacy", "pharmacy"
    if round_num == 1:
        if is_safety:  # trauma safety override -> social worker (see rules.py / escalation.py)
            return RECIPIENT_SOCIAL_WORKER, "social_worker", None
        rec = ROUND1_DRIVER_RECIPIENT.get(top_driver, RECIPIENT_SOCIAL_WORKER)
        if rec == RECIPIENT_TRANSIT_EXTERNAL:
            if external_allowed:
                return RECIPIENT_TRANSIT_EXTERNAL, "transit_voucher", "transit_broker"
            # external disclosure not authorized -> CHW-mediated internal fallback
            return RECIPIENT_AE_CHW, "chw_transport_fallback", "internal_transportation"
        return rec, rec, None
    if round_num == 2:
        return RECIPIENT_PRESCRIBER, "prescriber", None
    raise ValueError(f"unknown escalation round {round_num!r}")


def _history_strings(prior_rounds: List[dict]) -> dict:
    """The Round-2 'what we already tried' sentence, one per language."""
    if not prior_rounds:
        return {"en": "none recorded", "es": "ninguna registrada",
                "pt": "nenhuma registada", "ht": "pa gen okenn ki anrejistre"}
    out = {}
    for lang in LANGS:
        parts = []
        for a in prior_rounds:
            rec = a.get("recipient_type", a.get("action", "?"))
            label = RECIPIENT_LABELS.get(rec, {}).get(lang, rec)
            outcome = OUTCOME_WORDS.get(a.get("outcome", "pending"), OUTCOME_WORDS["pending"])[lang]
            when = a.get("dispatched_on") or "-"
            parts.append(f"{_ROUND_WORD[lang]} {a.get('round')} ({label}) — {when} — {outcome}")
        out[lang] = "; ".join(parts)
    return out


def build_dispatch(
    round_num: int,
    card: dict,
    prior_rounds: Optional[List[dict]] = None,
    consent: Optional[dict] = None,
    refill_meta: Optional[dict] = None,
) -> dict:
    """Build the provider-addressed dispatch payload for one patient at one round.

    ``card`` supplies patient_id / top_driver / is_safety_override / predicted_risk /
    rank_in_role. ``prior_rounds`` (round summaries with recipient_type/dispatched_on/
    outcome) is composed into the Round-2 history. ``consent`` is the per-scope consent
    summary ({external: {allowed: bool}, ...}); a gated external voucher resolves to the
    CHW fallback. ``refill_meta`` (refill_source / refill_latency_days) is carried through
    for provenance. Returns a dict with recipient_type, recipient_label{lang},
    mediated_by, addressed_to (always provider_or_organization), body{lang}, and -- for a
    CHW/prescriber only -- read_aloud_script{lang} labeled as the worker's tool.
    """
    top_driver = card.get("top_driver", "")
    is_safety = bool(card.get("is_safety_override", False))
    external_allowed = bool((consent or {}).get("external_disclosure", (consent or {}).get("external", {})).get("allowed", False)) \
        if consent else False

    recipient, kind, mediated_by = _resolve(round_num, top_driver, is_safety, external_allowed)
    history = _history_strings(prior_rounds or [])

    predicted_risk = card.get("predicted_risk")
    risk_pct = f"{predicted_risk * 100:.0f}%" if isinstance(predicted_risk, (int, float)) else "n/a"
    days = max(int(round(float(refill_meta.get("days_to_break", card.get("days_to_predicted_break", 0)) if refill_meta else card.get("days_to_predicted_break", 0)))), 0)
    break_date = (refill_meta or {}).get("predicted_break_date", "the predicted break date")

    body = {}
    for lang in LANGS:
        fields = {
            "ref": _patient_ref(card),
            "days": days,
            "break_date": break_date,
            "barrier": DRIVER_LABELS.get(top_driver, {}).get(lang, top_driver or "n/a"),
            "rank": card.get("rank_in_role", "n/a"),
            "risk_pct": risk_pct,
            "history": history[lang],
        }
        body[lang] = BODY_TEMPLATES[kind][lang].format(**fields)

    payload = {
        "round": round_num,
        "recipient_type": recipient,
        "recipient_label": dict(RECIPIENT_LABELS[recipient]),
        "mediated_by": mediated_by,
        "addressed_to": ADDRESSED_TO,
        "top_driver": top_driver,
        "body": body,
        "refill_source": (refill_meta or {}).get("refill_source"),
    }
    if recipient in _READ_ALOUD_RECIPIENTS:
        payload["read_aloud_script"] = dict(_READ_ALOUD_SCRIPT)
    else:
        payload["read_aloud_script"] = None
    if round_num == 2:
        payload["intervention_history"] = list(prior_rounds or [])
    return payload
