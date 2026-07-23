/*
 * BP Cascade RI — landing page + operations console (faithful design port).
 *
 * The prototype's renderVals() logic is ported to vanilla JS and wired to REAL
 * pipeline data: the live API when running (:8000/worklist), otherwise the
 * bundled snapshot generated from data/snapshots/routing_table.json. No
 * fabricated numbers — every stat, chart, and copilot answer is computed from
 * the served worklist. The ENTIRE interface localizes across EN/ES/PT/HT.
 */

import { COPILOT, ESC_L, INTERVENTIONS, MKT, ROUTE_LABELS, SITE_L, UI_STRINGS } from "./content.js";

const API_URL = `${location.protocol}//${location.hostname || "localhost"}:8000/worklist`;
const BUNDLED_URL = "./assets/worklist.sample.json";
const PAGE_SIZE = 30;
const POLL_INTERVAL = 30000;
const WORKFLOW_KEY = "bp.workflow.v3";
const LOCALES = { en: "en-US", es: "es-ES", pt: "pt-PT", ht: "fr-HT" };
const LANGS = ["en", "es", "pt", "ht"];
const LANG_NAMES = { en: "English", es: "Español", pt: "Português", ht: "Kreyòl Ayisyen" };

const ICONS = {
  arrow: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 10h11m-4-4 4 4-4 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  chevron: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 8 5 5 5-5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  filter: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M3 5h14M6 10h8M8.5 15h3" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`,
  person: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="6.5" r="3" stroke="currentColor" stroke-width="1.5"/><path d="M4.5 16c.6-3 2.9-4.5 5.5-4.5S15 13 15.6 16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  command: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 5h12M4 10h12M4 15h7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  shield: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 2.5 4 5v4.2c0 3.6 2.5 6.3 6 8.3 3.5-2 6-4.7 6-8.3V5l-6-2.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="m7.6 10 1.7 1.7L13 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  target: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><circle cx="10" cy="10" r="6.5" stroke="currentColor" stroke-width="1.5"/><circle cx="10" cy="10" r="2.3" fill="currentColor"/></svg>`,
  lock: `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="4.5" y="8.8" width="11" height="7.7" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M7 8.8V6.6a3 3 0 0 1 6 0v2.2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
};

const state = {
  lang: localStorage.getItem("bp.language") || "en",
  data: null, source: null,
  query: "", route: "all", esc: "all", minRisk: 0, visible: PAGE_SIZE,
  sortKey: "priority", sortDir: -1,
  expanded: new Set(), justExpanded: null,
  workflowMode: "after", activeTab: "triage", copilot: null,
  activePatient: null, dialogReturn: null, commandReturn: null,
  commandItems: [], commandIndex: 0, lastSignature: "",
  evidenceAnimated: false, statsAnimated: false,
};

const dom = {};
const $ = (id) => document.getElementById(id);

/* ------------------------------- helpers ------------------------------- */
const FALLBACK = UI_STRINGS.en;
function S() { return UI_STRINGS[state.lang] || FALLBACK; }
function M() { return MKT[state.lang] || MKT.en; }
function SL() { return SITE_L[state.lang] || SITE_L.en; }
function E() { return ESC_L[state.lang] || ESC_L.en; }
function escapeHtml(v) {
  return String(v ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
function debounce(fn, ms = 160) {
  let t = 0;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
function fmt(v) { return new Intl.NumberFormat(LOCALES[state.lang] || "en-US").format(Number(v) || 0); }
function routeLabel(k) { return (ROUTE_LABELS[state.lang] && ROUTE_LABELS[state.lang][k]) || ROUTE_LABELS.en[k] || k; }
function driverLabel(r) { return r[`driver_label_${state.lang}`] || r.driver_label_en || r.top_driver || ""; }
function rationale(r) { return r[`outreach_script_${state.lang}`] || r.outreach_script_en || ""; }
function reducedMotion() { try { return matchMedia("(prefers-reduced-motion: reduce)").matches; } catch { return false; } }
function interp(str, tk) { return String(str ?? "").replace(/\{(\w+)\}/g, (_, k) => (tk[k] != null ? tk[k] : "")); }

function relTime(v) {
  if (!v) return "…";
  const u = M().units || MKT.en.units;
  const sec = Math.max(0, Math.round((Date.now() - new Date(v).getTime()) / 1000));
  if (sec < 60) return `${sec}${u.s}`;
  if (sec < 3600) return `${Math.round(sec / 60)}${u.m}`;
  return `${Math.round(sec / 3600)}${u.h}`;
}
function syncedText() { return state.data ? `${M().metaSynced} ${relTime(state.data.last_synced)} ${M().metaAgo}` : ""; }
function sourceWord() { return state.source === "live" ? M().metaLive : (state.data?.data_source || M().metaSnapshot); }
function formatRange(a, b) {
  const loc = LOCALES[state.lang] || "en-US";
  const o = { month: "short", day: "numeric" };
  const sep = M().rangeTo;
  try {
    return `${new Date(`${a}T00:00:00`).toLocaleDateString(loc, o)} ${sep} ${new Date(`${b}T00:00:00`).toLocaleDateString(loc, o)}`;
  } catch { return `${a} ${sep} ${b}`; }
}

function readWorkflow() { try { return JSON.parse(localStorage.getItem(WORKFLOW_KEY)) || {}; } catch { return {}; } }
function writeWorkflow(id, patch) {
  const all = readWorkflow();
  all[id] = { ...(all[id] || {}), ...patch, at: new Date().toISOString() };
  localStorage.setItem(WORKFLOW_KEY, JSON.stringify(all));
}
function statusOf(id) { return (readWorkflow()[id] || {}).status || "routed"; }

function fadeSwap(el, dy = 6) {
  if (!el || reducedMotion() || typeof el.animate !== "function") return;
  el.animate([{ opacity: 0, transform: `translateY(${dy}px)` }, { opacity: 1, transform: "none" }], { duration: 260, easing: "cubic-bezier(.2,.75,.2,1)" });
}

/* ------------------------------- derived ------------------------------- */
function rows() { return state.data?.worklist || []; }
function maxPriority() {
  const vals = rows().map((r) => Number(r.priority_score) || 0);
  const m = vals.length ? Math.max(...vals) : 0;
  return m > 0 ? m : 1;
}
function heatColor(pct) {
  // Green (calm) → amber → red (critical). Same stops as the CSS bar gradient.
  const stops = [
    [0, [14, 138, 107]],
    [28, [79, 155, 78]],
    [52, [201, 162, 39]],
    [76, [196, 92, 42]],
    [100, [180, 35, 24]],
  ];
  const t = Math.max(0, Math.min(100, Number(pct) || 0));
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i += 1;
  const [aPos, a] = stops[i];
  const [bPos, b] = stops[i + 1];
  const u = (t - aPos) / Math.max(bPos - aPos, 1);
  const rgb = a.map((c, n) => Math.round(c + (b[n] - c) * u));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
function urgencyOf(r) { return Math.max(1, Math.round(((Number(r.priority_score) || 0) / maxPriority()) * 100)); }

function summary() {
  const list = rows(); const wf = readWorkflow();
  const riskPct = list.map((r) => (Number(r.risk_score) || 0) * 100);
  // State-4 closure is an objective on-time refill only — confirmed breaks are
  // observed outcomes but do not count as closed loops / observed refills.
  const onTime = list.filter((r) => r.loop_outcome?.on_time_refill).length;
  return {
    cohort: state.data?.cohort_size ?? state.data?.cohort ?? list.length,
    routed: list.length,
    safety: list.filter((r) => r.is_safety_override || r.requires_human_review).length,
    observed: onTime,
    confirmedBreaks: list.filter((r) => r.loop_outcome?.observed && !r.loop_outcome?.on_time_refill).length,
    acknowledged: list.filter((r) => ["acknowledged", "actioned"].includes(wf[r.patient_id]?.status)).length,
    actioned: list.filter((r) => wf[r.patient_id]?.status === "actioned").length,
    pharmacist: list.filter((r) => r.routed_action === "pharmacist").length,
    social: list.filter((r) => r.routed_action === "social_worker").length,
    chw: list.filter((r) => r.routed_action === "chw_call").length,
    peak: riskPct.length ? Math.round(Math.max(...riskPct)) : 0,
  };
}
function tokens() {
  const sm = summary(); const total = sm.routed || 1;
  const dominant = [["pharmacist", sm.pharmacist], ["social_worker", sm.social], ["chw_call", sm.chw]].sort((a, b) => b[1] - a[1])[0] || ["social_worker", 0];
  return {
    routed: fmt(sm.routed), safety: fmt(sm.safety), peakRisk: `${sm.peak}%`,
    social: fmt(sm.social), chw: fmt(sm.chw), pharmacist: fmt(sm.pharmacist),
    observed: fmt(sm.observed), acknowledged: fmt(sm.acknowledged), actioned: fmt(sm.actioned),
    openLoops: fmt(Math.max(sm.routed - sm.observed, 0)),
    dominantRoute: routeLabel(dominant[0]), dominantPct: `${Math.round((dominant[1] / total) * 100)}%`, __dominantRoute: dominant[0],
    ...escTokens(),
  };
}

/* ----------------------------- escalation ----------------------------- */
function pharmacySource() { return state.data?.pharmacy_source || null; }
function hasEscalation() { return rows().some((r) => r.escalation); }
function escStatusLabel(st) { return (E().statusLabels && E().statusLabels[st]) || st; }
function escRoundInfo(n) { const rl = E().roundLabels || ESC_L.en.roundLabels; return rl[n] || rl[0]; }
function escOptionLabel(v) { const o = (E().filterOptions || []).find(([k]) => k === v); return o ? o[1] : v; }

// Per-rung and terminal counts derived straight from the served worklist, so the
// cascade always matches exactly what the table below it shows.
function escSummary() {
  const rung = [0, 1, 2].map(() => ({ total: 0, closed: 0, exhausted: 0, inflight: 0 }));
  const byRound = { 0: 0, 1: 0, 2: 0 };
  let closed = 0, exhausted = 0, inflight = 0, gated = 0, withEsc = 0;
  for (const r of rows()) {
    const e = r.escalation; if (!e) continue;
    withEsc += 1;
    const cr = Math.max(0, Math.min(2, Number(e.current_round) || 0));
    byRound[cr] += 1;
    const rr = rung[cr];
    rr.total += 1;
    if (e.status === "CLOSED") { closed += 1; rr.closed += 1; }
    else if (e.status === "EXHAUSTED") { exhausted += 1; rr.exhausted += 1; }
    else { inflight += 1; rr.inflight += 1; }
    if ((e.gated_actions && e.gated_actions.length) || e.status === "GATED_ON_CONSENT") gated += 1;
  }
  return { rung, byRound, closed, exhausted, inflight, gated, withEsc, total: rows().length };
}
function escTokens() {
  if (!state.data || !hasEscalation()) return {};
  const s = escSummary(); const ps = pharmacySource();
  return {
    escClosed: fmt(s.closed), escExhausted: fmt(s.exhausted), escGated: fmt(s.gated),
    escInflight: fmt(s.inflight), escRound2: fmt(s.byRound[2] || 0),
    pharmacySource: ps?.name || sourceWord(),
    pharmacyAccess: ps ? (E().accessModes[ps.access_mode] || ps.access_mode) : "",
  };
}
function matchEsc(r, f) {
  const e = r.escalation; if (!e) return false;
  switch (f) {
    case "r0": return Number(e.current_round) === 0;
    case "r1": return Number(e.current_round) === 1;
    case "r2": return Number(e.current_round) === 2;
    case "closed": return e.status === "CLOSED";
    case "exhausted": return e.status === "EXHAUSTED";
    case "gated": return Boolean((e.gated_actions && e.gated_actions.length) || e.status === "GATED_ON_CONSENT");
    case "inflight": return !["CLOSED", "EXHAUSTED"].includes(e.status);
    default: return true;
  }
}

/* ------------------------------ language ------------------------------ */
function renderLangSwitch() {
  const html = LANGS.map((c) =>
    `<button type="button" data-lang="${c}" aria-pressed="${c === state.lang}">${c.toUpperCase()}</button>`).join("");
  dom.langSwitch.innerHTML = html;
  if (dom.mobileLangSwitch) dom.mobileLangSwitch.innerHTML = html;
}
function renderHeroTitle() {
  const el = $("hero-title");
  if (!el) return;
  el.setAttribute("aria-label", M().heroAria);
  el.innerHTML = M().heroTitle.map((w) =>
    w.br ? "<br>" : `<span class="word-mask"><span data-hero-word${w.a ? ' class="accent-word"' : ""}>${escapeHtml(w.t)}</span></span>`).join(" ");
}
function applyMarketing() {
  const m = M();
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = m[el.dataset.i18n];
    if (typeof v === "string") el.textContent = v;
  });
  const aria = m.aria || MKT.en.aria;
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => {
    const v = aria[el.dataset.i18nAria];
    if (typeof v === "string") el.setAttribute("aria-label", v);
  });
  document.title = `BP Cascade RI \u2014 ${m.heroAria}`;
  renderHeroTitle();
}
function applyConsoleLabels() {
  const s = S();
  $("lbl-search").textContent = s.search;
  dom.q.placeholder = s.searchPlaceholder;
  $("lbl-route").textContent = s.route;
  $("lbl-risk").textContent = s.minRisk;
  dom.resetFilters.textContent = s.resetFilters;
  dom.loadMore.textContent = s.loadMore;
  $("console-live").textContent = s.liveSnapshot;
  dom.commandInput.placeholder = s.commandPlaceholder;
  $("dialog-title").textContent = s.markActioned;
  $("lbl-intervention").textContent = s.intervention;
  $("lbl-reference").textContent = s.reference;
  $("lbl-notes").textContent = s.notes;
  dom.dialogReference.placeholder = s.referencePlaceholder;
  dom.dialogNotes.placeholder = s.notesPlaceholder;
  $("dialog-note").textContent = s.dialogNote;
  dom.dialogCancel.textContent = s.cancel;
  dom.dialogSave.textContent = s.save;
  dom.routeFilter.innerHTML = [
    ["all", s.allRoutes], ["pharmacist", routeLabel("pharmacist")],
    ["social_worker", routeLabel("social_worker")], ["chw_call", routeLabel("chw_call")],
  ].map(([v, l]) => `<option value="${v}"${v === state.route ? " selected" : ""}>${escapeHtml(l)}</option>`).join("");
  const lblEsc = $("lbl-esc");
  if (lblEsc) lblEsc.textContent = E().filterLabel;
  if (dom.escFilter) dom.escFilter.innerHTML = (E().filterOptions || []).map(([v, l]) => `<option value="${v}"${v === state.esc ? " selected" : ""}>${escapeHtml(l)}</option>`).join("");
  // Pre-load placeholders (overwritten by renderHero/renderSources once data arrives)
  if (!state.data) {
    dom.heroQueueCount.textContent = M().loading;
    dom.heroSource.textContent = M().metaConnecting;
    dom.heroSync.textContent = "";
    dom.consoleMeta.textContent = `${M().metaConnecting}\u2026`;
  }
}
function setLanguage(lang) {
  if (!UI_STRINGS[lang]) return;
  state.lang = lang; state.copilot = null;
  localStorage.setItem("bp.language", lang);
  document.documentElement.lang = lang;
  renderLangSwitch();
  applyMarketing();
  applyConsoleLabels();
  renderEvidence(); renderRail(); renderWorkflow(state.workflowMode); renderTrust(); renderMethod();
  renderProductTabs(); renderTabCopy();
  if (state.data) { renderHero(); renderConsoleStats(); renderWorklist(); renderEscalation(); renderShowcase(state.activeTab, { animate: false }); renderSources(); }
  renderCopilot();
  if (!dom.commandDialog.hidden) renderCommand(dom.commandInput.value);
  // Subtle, professional fade across the freshly re-rendered sections on language swap.
  fadeSwap($("hero-title"), 4);
  [dom.evidenceGrid, dom.careRail, dom.workflowCards, dom.trustStack, dom.methodGrid, dom.tabCopy, dom.screenBody, dom.copilotPrompts, dom.consoleStats]
    .forEach((el) => fadeSwap(el, 5));
  if (state.data) fadeSwap(dom.wlBody, 6);
}

/* --------------------------- rendered sections --------------------------- */
function renderEvidence() {
  dom.evidenceGrid.innerHTML = SL().evidence.map((it) => {
    const num = Number(String(it.value).replace(/[^0-9.]/g, ""));
    const countable = Number.isFinite(num) && /[0-9]/.test(it.value);
    return `<div class="evidence-item">
      <div class="evidence-value tabular"${countable ? ` data-count="${num}"` : ""}>${escapeHtml(fmtValue(it.value, countable, num))}</div>
      <div class="evidence-label">${escapeHtml(it.label)}</div>
      <div class="evidence-note">${escapeHtml(it.note)}</div>
    </div>`;
  }).join("");
  // Count-up runs once on scroll-in (see observeCount); re-renders (language/poll) show final values without replaying.
}
function fmtValue(raw, countable, num) { return countable ? fmt(num) : raw; }

function renderRail() {
  dom.careRail.innerHTML = SL().rail.map(([n, title, body, kind], i) => {
    const cls = kind === "complete" ? " is-complete" : kind === "active" ? " is-active" : "";
    const line = i < SL().rail.length - 1 ? `<span class="rail-line"></span>` : "";
    return `<div class="rail-step${cls}">
      <div class="rail-col"><span class="rail-dot">${n}</span>${line}</div>
      <div class="rail-card"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(body)}</span></div>
    </div>`;
  }).join("");
}

function renderWorkflow(mode) {
  state.workflowMode = mode;
  dom.wfBefore.setAttribute("aria-selected", String(mode === "before"));
  dom.wfAfter.setAttribute("aria-selected", String(mode === "after"));
  dom.workflowCards.innerHTML = SL().workflow[mode].map(([n, title, body, tag]) => `
    <article class="wf-card ${mode}">
      <span class="wf-num">${escapeHtml(n)}</span>
      <div class="wf-body"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(body)}</p></div>
      <span class="wf-tag">${escapeHtml(tag)}</span>
    </article>`).join("");
  if (!reducedMotion() && typeof dom.workflowCards.firstElementChild?.animate === "function") {
    dom.workflowCards.querySelectorAll(".wf-card").forEach((el, i) =>
      el.animate([{ opacity: 0, transform: "translateX(-10px)" }, { opacity: 1, transform: "none" }],
        { duration: 460, delay: i * 90, easing: "cubic-bezier(.2,.75,.2,1)", fill: "backwards" }));
  }
}

function renderTrust() {
  dom.trustStack.innerHTML = SL().trustPoints.map((it) => `
    <article class="trust-card" data-reveal>
      <span class="t-eyebrow">${escapeHtml(it.eyebrow)}</span>
      <h3>${escapeHtml(it.title)}</h3>
      <p>${escapeHtml(it.body)}</p>
      <span class="code-chip">${escapeHtml(it.code)}</span>
    </article>`).join("");
  dom.trustStack.querySelectorAll("[data-reveal]").forEach((el) => el.classList.add("is-visible"));
}

function renderMethod() {
  dom.methodGrid.innerHTML = SL().method.map(([n, title, body]) => `
    <div class="method-item"><span class="method-num">${escapeHtml(n)}</span>
      <h3>${escapeHtml(title)}</h3><p>${escapeHtml(body)}</p></div>`).join("");
}

/* ---------------------------- escalation UI ---------------------------- */
function renderEscalation() {
  if (!dom.escSection) return;
  const show = Boolean(state.data && hasEscalation());
  dom.escSection.hidden = !show;
  if (!show) return;
  const e = E(); const s = escSummary();
  renderCascade(e, s);
  renderPharmacySource(e);
  renderGuardrails(e);
}

function renderCascade(e, s) {
  const rungs = e.roundLabels.map((rl, i) => {
    const rr = s.rung[i]; const empty = rr.total === 0;
    const seg = (n, cls) => (n > 0 ? `<span class="${cls}" style="width:${(n / rr.total) * 100}%"></span>` : "");
    const legend = [];
    if (rr.closed) legend.push(`<span><span class="dot closed"></span>${escapeHtml(e.closedHere)} <b>${fmt(rr.closed)}</b></span>`);
    if (rr.exhausted) legend.push(`<span><span class="dot exhausted"></span>${escapeHtml(e.noRefill)} <b>${fmt(rr.exhausted)}</b></span>`);
    if (rr.inflight) legend.push(`<span><span class="dot inflight"></span>${escapeHtml(e.inFlight)} <b>${fmt(rr.inflight)}</b></span>`);
    const line = i < e.roundLabels.length - 1 ? `<span class="esc-rung-line"></span>` : "";
    const aria = `${rl.name}: ${e.closedHere} ${rr.closed}, ${e.noRefill} ${rr.exhausted}, ${e.inFlight} ${rr.inflight}`;
    return `<div class="esc-rung${empty ? " is-empty" : ""}">
      <div class="esc-rung-col"><span class="esc-rung-dot">${i}</span>${line}</div>
      <div class="esc-rung-body">
        <div class="esc-rung-top">
          <span class="esc-rung-name">${escapeHtml(rl.name)}<span>${escapeHtml(rl.desc)}</span></span>
          <span class="esc-rung-total">${fmt(rr.total)}</span>
        </div>
        <div class="esc-bar" role="img" aria-label="${escapeHtml(aria)}">${empty ? "" : seg(rr.closed, "seg-closed") + seg(rr.exhausted, "seg-exhausted") + seg(rr.inflight, "seg-inflight")}</div>
        <div class="esc-rung-legend">${legend.join("") || `<span>${escapeHtml(e.atRung)}</span>`}</div>
      </div>
    </div>`;
  }).join("");
  const term = [
    ["CLOSED", s.closed, e.terminalClosed],
    ["EXHAUSTED", s.exhausted, e.terminalExhausted],
    ["GATED_ON_CONSENT", s.gated, e.terminalGated],
  ].map(([k, n, l]) => `<div class="esc-term" data-esc="${k}"><strong class="tabular">${fmt(n)}</strong><span>${escapeHtml(l)}</span></div>`).join("");
  dom.escCascade.innerHTML = `
    <div class="esc-card-head"><h3>${escapeHtml(e.cascadeTitle)}</h3><span class="mono">${fmt(s.withEsc)} ${escapeHtml(e.routedWord)}</span></div>
    <div class="esc-rungs">${rungs}</div>
    <div class="esc-terminals">${term}</div>`;
  growEscBars();
}
function growEscBars() {
  if (reducedMotion()) return;
  dom.escCascade.querySelectorAll(".esc-bar > span").forEach((el, i) => {
    if (typeof el.animate !== "function") return;
    el.animate([{ width: "0%" }, { width: el.style.width }], { duration: 620, delay: i * 45, easing: "cubic-bezier(.2,.75,.2,1)", fill: "backwards" });
  });
}

function renderPharmacySource(e) {
  const ps = pharmacySource();
  if (!ps) { dom.escSource.hidden = true; return; }
  dom.escSource.hidden = false;
  const lat = ps.latency_profile || {};
  const latStr = [lat.min, lat.typical, lat.max].every((v) => v != null) ? `${lat.min}/${lat.typical}/${lat.max}${e.latencyUnit}` : "";
  const access = e.accessModes[ps.access_mode] || ps.access_mode || "";
  const trace = (ps.fallback_trace || []).map((tr) => {
    const status = tr.outcome === "served" ? e.outcomeServed : e.outcomeAuthFailed;
    return `<div class="esc-trace-row" data-outcome="${escapeHtml(tr.outcome)}">
      <span class="esc-trace-dot"></span>
      <span class="esc-trace-name" title="${escapeHtml(tr.reason || tr.source_name || "")}">${escapeHtml(tr.source_name || tr.adapter)}</span>
      <span class="esc-trace-status">${escapeHtml(status)}</span>
    </div>`;
  }).join("");
  dom.escSource.innerHTML = `
    <div class="esc-card-head"><h3>${escapeHtml(e.sourceTitle)}</h3><span class="mono">${escapeHtml(e.sourceSynced)} ${escapeHtml(relTime(ps.last_synced))}</span></div>
    <div class="esc-source-name">${escapeHtml(ps.name)}<span class="tag">${escapeHtml(e.sourceServing)}</span></div>
    <div class="esc-source-meta">
      <span>${escapeHtml(e.sourceAccess)} <b>${escapeHtml(access)}</b></span>
      ${latStr ? `<span>${escapeHtml(e.sourceLatency)} <b>${escapeHtml(latStr)}</b></span>` : ""}
    </div>
    <div class="esc-trace-title">${escapeHtml(e.fallbackTitle)}</div>
    ${trace}`;
}

function renderGuardrails(e) {
  const icons = [ICONS.shield, ICONS.target, ICONS.lock];
  dom.escGuardrails.innerHTML = `<h3>${escapeHtml(e.guardrailsTitle)}</h3>` + (e.guardrails || []).map((g, i) => `
    <div class="esc-guard">
      <span class="esc-guard-icon">${icons[i] || ICONS.shield}</span>
      <div><strong>${escapeHtml(g.title)}</strong><p>${escapeHtml(g.body)}</p></div>
    </div>`).join("");
}

function escPill(r) {
  const e = r.escalation; if (!e) return "";
  const ri = escRoundInfo(e.current_round);
  const title = `${ri.name} · ${ri.desc} · ${escStatusLabel(e.status)}`;
  return `<span class="esc-pill" data-esc="${escapeHtml(e.status)}" title="${escapeHtml(title)}">R${Number(e.current_round) || 0} · ${escapeHtml(escStatusLabel(e.status))}</span>`;
}

function escDateFmt(iso) {
  if (!iso) return "";
  try { return new Date(`${iso}T00:00:00`).toLocaleDateString(LOCALES[state.lang] || "en-US", { month: "short", day: "numeric", year: "numeric" }); }
  catch { return iso; }
}

function escalationDetail(r) {
  const e = r.escalation; const t = E(); const lang = state.lang;
  if (!e) return `<div class="esc-detail"><p class="esc-body">${escapeHtml(t.noEscalation)}</p></div>`;
  const ri = escRoundInfo(e.current_round);
  const statusL = escStatusLabel(e.status);
  let countdown = "";
  if (e.status === "WAITING_ON_DATA_LATENCY" && e.days_until_latency_clears != null) countdown = `${fmt(e.days_until_latency_clears)} ${t.daysToData}`;
  else if (e.days_remaining != null && !["CLOSED", "EXHAUSTED"].includes(e.status)) countdown = `${fmt(e.days_remaining)} ${t.daysLeft}`;
  const breakDate = escDateFmt(e.predicted_break_date);

  const cd = e.current_dispatch || (e.dispatch_history && e.dispatch_history[e.dispatch_history.length - 1]) || null;
  const body = r[`dispatch_message_${lang}`] || r.dispatch_message_en || (cd && cd.body && (cd.body[lang] || cd.body.en)) || "";
  const recip = cd ? (cd.recipient_label?.[lang] || cd.recipient_label?.en || cd.recipient_type) : "";
  const med = cd && cd.mediated_by ? ` <span class="mono">${escapeHtml(t.via)} ${escapeHtml(t.mediators[cd.mediated_by] || cd.mediated_by)}</span>` : "";
  const readAloud = r[`chw_read_aloud_script_${lang}`] || r.chw_read_aloud_script_en || "";
  const dispatchBlock = `<div class="esc-block">
    <h4>${escapeHtml(t.currentDispatch)}</h4>
    ${recip ? `<div class="esc-dispatch-to"><b>${escapeHtml(t.dispatchTo)}:</b> ${escapeHtml(recip)}${med}</div>` : ""}
    <div class="esc-provider-badge">${ICONS.shield}<span>${escapeHtml(t.providerBadge)}</span></div>
    ${body ? `<p class="esc-body">${escapeHtml(body)}</p>` : ""}
    ${readAloud ? `<div class="esc-readaloud"><span class="lbl">${escapeHtml(t.readAloud)}</span>${escapeHtml(readAloud)}</div>` : ""}
  </div>`;

  const cs = e.consent_scopes || {};
  const scopeRow = (key) => {
    const sc = cs[key]; if (!sc) return "";
    const scopeLabel = t.consent[key] || key;
    const stateLabel = t.consent[sc.state] || sc.state;
    const stale = sc.stale ? ` · ${t.consent.stale}` : "";
    return `<div class="esc-consent-row"><span>${escapeHtml(scopeLabel)}</span><span class="esc-tag" data-ok="${Boolean(sc.allowed)}">${escapeHtml(stateLabel)}${escapeHtml(stale)}</span></div>`;
  };
  const consentBlock = (cs.internal_care_coordination || cs.external_disclosure)
    ? `<div class="esc-block"><h4>${escapeHtml(t.consentTitle)}</h4>${scopeRow("internal_care_coordination")}${scopeRow("external_disclosure")}</div>`
    : "";

  const gated = (e.gated_actions || []).map((g) => {
    const act = escRoundInfo(g.round);
    const fb = g.fallback_action ? `${t.fallbackWord}: ${g.fallback_action}` : t.hardBlock;
    return `<div class="esc-gate">${escapeHtml(act.name)} · <code>${escapeHtml(g.reason || "")}</code> · ${escapeHtml(fb)}</div>`;
  }).join("");
  const gatedBlock = gated ? `<div class="esc-block wide"><h4>${escapeHtml(t.gateTitle)}</h4>${gated}</div>` : "";

  const hist = (e.dispatch_history || []).map((h) => {
    const hr = escRoundInfo(h.round);
    const hrecip = h.recipient_label?.[lang] || h.recipient_label?.en || h.recipient_type || "";
    const out = t.outcomes[h.outcome] || h.outcome || "";
    return `<div class="esc-hist-row"><span class="esc-hist-round">${escapeHtml(hr.name)}</span><span class="esc-hist-recip">${escapeHtml(hrecip)}</span><span class="esc-hist-outcome">${escapeHtml(out)}</span></div>`;
  }).join("");
  const histBlock = hist ? `<div class="esc-block wide"><h4>${escapeHtml(t.historyTitle)}</h4>${hist}</div>` : "";

  return `<div class="esc-detail">
    <div class="esc-detail-head">
      <span class="esc-badge" data-esc="${escapeHtml(e.status)}">${escapeHtml(ri.name)} · ${escapeHtml(statusL)}</span>
      ${breakDate ? `<span class="esc-detail-meta">${escapeHtml(t.predictedBreak)}: <b>${escapeHtml(breakDate)}</b></span>` : ""}
      ${countdown ? `<span class="esc-detail-meta mono">${escapeHtml(countdown)}</span>` : ""}
    </div>
    <div class="esc-blocks">${dispatchBlock}${consentBlock}${gatedBlock}${histBlock}</div>
  </div>`;
}

/* ------------------------------- hero ------------------------------- */
function renderHero() {
  if (!state.data) return;
  const top = [...rows()].sort((a, b) => (Number(b.priority_score) || 0) - (Number(a.priority_score) || 0)).slice(0, 3);
  dom.heroQueueCount.textContent = `${fmt(rows().length)} ${M().metaRouted}`;
  dom.heroSource.textContent = sourceWord();
  dom.heroSync.textContent = syncedText();
  dom.heroSignalList.innerHTML = top.map((r, i) => {
    const risk = Math.round((Number(r.risk_score) || 0) * 100);
    return `<div class="signal-row${i === 0 ? " is-first" : ""}" data-queue-row>
      <span class="signal-index">0${i + 1}</span>
      <span class="signal-person"><strong>${escapeHtml(r.display_name)}</strong><span>${escapeHtml(driverLabel(r))} · ${escapeHtml(routeLabel(r.routed_action))}</span></span>
      <span class="signal-risk"><span class="mono">${risk}%</span><span class="signal-bar"><span style="width:${risk}%;--pct:${risk}"></span></span></span>
    </div>`;
  }).join("");
  if (!reducedMotion()) {
    dom.heroSignalList.querySelectorAll("[data-queue-row]").forEach((el, i) =>
      el.animate([{ opacity: 0, transform: "translateX(-10px)" }, { opacity: 1, transform: "none" }],
        { duration: 520, delay: i * 85, easing: "cubic-bezier(.2,.75,.2,1)", fill: "backwards" }));
  }
}

/* ------------------------------ product ------------------------------ */
function renderProductTabs() {
  dom.productTabs.innerHTML = SL().showcaseTabs.map((tab) =>
    `<button class="p-tab" type="button" role="tab" data-tab="${tab.id}" aria-selected="${tab.id === state.activeTab}">${escapeHtml(tab.label)}</button>`).join("");
}
function renderTabCopy() {
  const tab = SL().showcaseTabs.find((t) => t.id === state.activeTab) || SL().showcaseTabs[0];
  dom.tabCopy.innerHTML = `<span class="tab-eyebrow">${escapeHtml(tab.eyebrow)}</span><h3>${escapeHtml(tab.title)}</h3><p>${escapeHtml(tab.body)}</p>`;
}
function charts() { return (COPILOT[state.lang] || COPILOT.en).charts; }

function renderShowcase(tab, { animate = true } = {}) {
  const changed = tab !== state.activeTab;
  state.activeTab = tab;
  renderProductTabs();
  renderTabCopy();
  const ch = charts();
  dom.screenBody.innerHTML = tab === "route" ? buildRoute(ch) : tab === "close" ? buildClose(ch) : buildTriage(ch);
  if (animate) growCharts();
  if (changed) { fadeSwap(dom.tabCopy, 4); fadeSwap(dom.screenBody, 8); }
}

function buildTriage(ch) {
  const list = rows();
  const bands = [0, 0, 0, 0, 0];
  list.forEach((r) => { bands[Math.min(4, Math.max(0, Math.floor(((Number(r.risk_score) || 0) * 100) / 20)))]++; });
  const max = Math.max(1, ...bands);
  const labels = M().riskBands;
  const cols = bands.map((c, i) => `
    <div class="tri-col">
      <span class="tri-count">${fmt(c)}</span>
      <span class="tri-track"><span class="tri-fill" style="height:${c ? Math.max(4, Math.round((c / max) * 100)) : 2}%;background:${heatColor(i * 25 + 12)}"></span></span>
      <span class="tri-label">${labels[i]}</span>
    </div>`).join("");
  return `
    <div class="chart-head"><span class="chart-title">${escapeHtml(ch.risk)}</span><span class="chart-legend">n = ${fmt(list.length)} ${escapeHtml(ch.unit)}</span></div>
    <div class="tri-bars" role="img" aria-label="${escapeHtml(ch.risk)}: ${labels.map((l, i) => `${l} ${bands[i]}`).join(", ")}">${cols}</div>
    <p class="chart-foot">${escapeHtml(ch.foot.triage)}</p>`;
}
function buildRoute(ch) {
  const sm = summary();
  const nodes = [["pharmacist", sm.pharmacist], ["social_worker", sm.social], ["chw_call", sm.chw]];
  const total = Math.max(1, sm.pharmacist + sm.social + sm.chw);
  const bars = nodes.map(([route, n]) => `
    <div class="route-row">
      <div class="route-top"><span class="route-name"><span class="rdot" data-route="${route}"></span>${escapeHtml(routeLabel(route))}</span><span class="route-count">${fmt(n)}</span></div>
      <div class="route-track"><span class="route-fill" data-route="${route}" style="width:${n ? Math.max(3, Math.round((n / total) * 100)) : 0}%"></span></div>
    </div>`).join("");
  return `
    <div class="chart-head"><span class="chart-title">${escapeHtml(ch.flow)}</span><span class="chart-legend">${fmt(total)} ${escapeHtml(M().metaRouted)}</span></div>
    <div class="route-rows" role="img" aria-label="${escapeHtml(ch.flow)}: ${nodes.map(([r, n]) => `${routeLabel(r)} ${n}`).join(", ")}">${bars}</div>
    <p class="chart-foot">${escapeHtml(ch.foot.route)}</p>`;
}
function buildClose(ch) {
  const s = S(); const sm = summary(); const base = Math.max(1, sm.routed);
  const stages = [[s.routed, sm.routed, "ink"], [s.acknowledged, sm.acknowledged, "bg2"], [s.actioned, sm.actioned, "bg2"], [ch.observed, sm.observed, "accent"]];
  const bars = stages.map(([label, n, tone]) => {
    const pct = Math.max(9, Math.round((n / base) * 100));
    const bg = tone === "accent" ? "var(--accent)" : tone === "ink" ? "var(--ink)" : "var(--bg2)";
    const color = tone === "accent" || tone === "ink" ? "#fff" : "var(--ink)";
    const border = tone === "bg2" ? "border:1px solid var(--line);" : "";
    return `<div class="funnel-bar" style="width:${pct}%">
      <span class="funnel-fill" style="background:${bg};${border}"></span>
      <span class="funnel-text"><span style="color:${color}">${escapeHtml(label)}</span><span class="funnel-count" style="color:${color}">${fmt(n)}</span></span>
    </div>`;
  }).join("");
  return `
    <div class="chart-head"><span class="chart-title">${escapeHtml(ch.funnel)}</span><span class="chart-legend">${fmt(sm.observed)} / ${fmt(sm.routed)}</span></div>
    <div class="funnel-rows" role="img" aria-label="${escapeHtml(ch.funnel)}: ${stages.map(([l, n]) => `${l} ${n}`).join(", ")}">${bars}</div>
    <p class="chart-foot">${escapeHtml(ch.foot.close)}</p>`;
}
function growCharts() {
  if (reducedMotion()) return;
  dom.screenBody.querySelectorAll(".tri-fill").forEach((el, i) => grow(el, "scaleY", i));
  dom.screenBody.querySelectorAll(".route-fill,.funnel-fill").forEach((el, i) => grow(el, "scaleX", i));
}
function grow(el, axis, i) {
  if (typeof el.animate !== "function") return;
  el.animate([{ transform: `${axis}(0)` }, { transform: `${axis}(1)` }], { duration: 600, delay: i * 55, easing: "cubic-bezier(.2,.75,.2,1)", fill: "backwards" });
}

/* ------------------------------ copilot ------------------------------ */
function renderCopilot() {
  const cp = COPILOT[state.lang] || COPILOT.en;
  $("copilot-title").textContent = cp.title;
  dom.copilotNote.textContent = cp.note;
  const prompts = cp.prompts.filter((p) => p.tab === state.activeTab);
  const tk = state.data ? tokens() : {};
  dom.copilotPrompts.innerHTML = prompts.map((p) =>
    `<button class="copilot-chip" type="button" data-prompt="${p.id}" aria-pressed="${state.copilot?.id === p.id}">${escapeHtml(interp(p.chip, tk))}</button>`).join("");
  if (state.copilot && prompts.some((p) => p.id === state.copilot.id)) {
    dom.copilotAnswer.hidden = false;
    dom.copilotAnswer.innerHTML = state.copilot.html;
  } else {
    dom.copilotAnswer.hidden = true; dom.copilotAnswer.innerHTML = "";
  }
}
function runPrompt(id) {
  const cp = COPILOT[state.lang] || COPILOT.en;
  const p = cp.prompts.find((x) => x.id === id);
  if (!p) return;
  if (state.activeTab !== p.tab) renderShowcase(p.tab);
  const tk = tokens();
  let html = `<span class="copilot-line">${interp(p.answer, tk)}</span>`;
  if (p.action) {
    const route = p.action === "__dominant__" ? tk.__dominantRoute : p.action;
    html += `<div><button class="copilot-apply" type="button" data-apply="${route}">${ICONS.filter}<span>${escapeHtml(interp(p.actionLabel, tk))}</span></button></div>`;
  }
  if (p.escAction) {
    html += `<div><button class="copilot-apply" type="button" data-apply-esc="${escapeHtml(p.escAction)}">${ICONS.filter}<span>${escapeHtml(interp(p.escActionLabel, tk))}</span></button></div>`;
  }
  state.copilot = { id, html };
  renderCopilot();
  fadeSwap(dom.copilotAnswer, 4);
}
function applyCopilotRoute(route) {
  resetFilters();
  state.route = route; dom.routeFilter.value = route;
  renderWorklist();
  scrollTo("console");
  toast(`${S().route} · ${routeLabel(route)}`);
}
function applyCopilotEsc(v) {
  resetFilters();
  state.esc = v; if (dom.escFilter) dom.escFilter.value = v;
  renderWorklist();
  scrollTo("console");
  toast(`${E().filterLabel} · ${escOptionLabel(v)}`);
}

/* ------------------------------ console ------------------------------ */
function renderConsoleStats() {
  if (!state.data) return;
  const s = S(); const sm = summary();
  const tiles = [
    [sm.cohort, s.modeledCohort, false], [sm.routed, s.routedToday, false],
    [sm.safety, s.humanReview, true], [sm.observed, s.observedRefills, false],
  ];
  dom.consoleStats.innerHTML = tiles.map(([v, l, safety]) =>
    `<div class="console-stat${safety ? " is-safety" : ""}"><strong class="tabular" data-count="${v}">${fmt(v)}</strong><span>${escapeHtml(l)}</span></div>`).join("");
  dom.consoleMeta.textContent = `${sourceWord()} · ${syncedText()}`;
  // Count-up runs once on scroll-in (see observeCount); re-renders (language/poll) show final values without replaying.
}

function filtered() {
  const q = state.query.trim().toLowerCase();
  const out = rows().filter((r) => {
    const risk = (Number(r.risk_score) || 0) * 100;
    const hay = [r.display_name, r.patient_id, driverLabel(r), routeLabel(r.routed_action)].join(" ").toLowerCase();
    return (!q || hay.includes(q))
      && (state.route === "all" || r.routed_action === state.route)
      && (state.esc === "all" || matchEsc(r, state.esc))
      && risk >= state.minRisk;
  });
  const dir = state.sortDir;
  out.sort((a, b) => {
    if (state.sortKey === "window") return dir * String(a.break_window_start).localeCompare(String(b.break_window_start));
    const key = state.sortKey === "risk" ? "risk_score" : "priority_score";
    return dir * ((Number(a[key]) || 0) - (Number(b[key]) || 0));
  });
  return out;
}

function headCell(key, label) {
  const active = state.sortKey === key;
  const caret = active ? (state.sortDir === 1 ? "↑" : "↓") : "↕";
  return `<button type="button" data-sort="${key}" aria-label="${escapeHtml(label)}">${escapeHtml(label)}<span class="caret">${caret}</span></button>`;
}
function statusOptions(sel) {
  const s = S();
  return [["routed", s.routed], ["acknowledged", s.acknowledged], ["actioned", s.actioned]]
    .map(([v, l]) => `<option value="${v}"${v === sel ? " selected" : ""}>${escapeHtml(l)}</option>`).join("");
}

function renderWorklist() {
  if (!state.data) return;
  const s = S();
  dom.worklist.setAttribute("aria-busy", "false");
  dom.wlHead.innerHTML = `<div class="wl-cols">
    <div>${escapeHtml(s.patient)}</div>
    ${headCell("priority", s.urgency)}
    ${headCell("window", s.window)}
    ${headCell("risk", s.riskShort)}
    <div>${escapeHtml(s.barrier)}</div>
    <div>${escapeHtml(s.route)}</div>
    <div>${escapeHtml(s.status)}</div>
    <div></div>
  </div>`;

  const all = filtered();
  const vis = all.slice(0, state.visible);

  if (!all.length) {
    dom.wlBody.innerHTML = `<div class="wl-empty"><strong>${escapeHtml(s.noResults)}</strong><button class="button button-outline" type="button" data-reset>${escapeHtml(s.resetFilters)}</button></div>`;
    dom.consoleFoot.hidden = true;
    return;
  }

  dom.wlBody.innerHTML = vis.map((r) => {
    const id = r.patient_id; const status = statusOf(id);
    const risk = Math.round((Number(r.risk_score) || 0) * 100); const urg = urgencyOf(r);
    const open = state.expanded.has(id);
    const outcome = r.loop_outcome;
    const onTime = Boolean(outcome?.on_time_refill);
    const confirmedBreak = Boolean(outcome?.observed && !outcome?.on_time_refill);
    const outcomeLabel = onTime ? s.outcomeObserved : confirmedBreak ? s.confirmedBreak : s.loopOpen;
    const outcomeClass = onTime ? "observed" : confirmedBreak ? "confirmed-break" : "";
    return `<div class="wl-group${open ? " is-open" : ""}" data-id="${escapeHtml(id)}">
      <div class="wl-cols wl-row">
        <div class="wl-patient" data-col="patient"><strong>${escapeHtml(r.display_name)}</strong><span class="pid">${escapeHtml(String(r.patient_id).slice(0, 14))}</span>${r.is_safety_override ? `<span class="safety-pill">${escapeHtml(s.humanReview)}</span>` : ""}${escPill(r)}</div>
        <div data-col="urgency" data-col-label="${escapeHtml(s.urgency)}"><div class="metric-top"><strong>${urg}</strong><span>/100</span></div><div class="metric-track"><span class="metric-fill urgency" style="width:${urg}%;--pct:${urg}"></span></div></div>
        <div class="wl-window" data-col="window" data-col-label="${escapeHtml(s.window)}">${escapeHtml(formatRange(r.break_window_start, r.break_window_end))}</div>
        <div data-col="risk" data-col-label="${escapeHtml(s.riskShort)}"><div class="metric-top"><strong>${risk}%</strong></div><div class="metric-track"><span class="metric-fill risk" style="width:${risk}%;--pct:${Math.max(risk, 1)}"></span></div></div>
        <div class="wl-driver" data-col="driver" data-col-label="${escapeHtml(s.barrier)}">${escapeHtml(driverLabel(r))}</div>
        <div data-col="route" data-col-label="${escapeHtml(s.route)}"><span class="route-pill"><span class="rdot" data-route="${escapeHtml(r.routed_action)}"></span>${escapeHtml(routeLabel(r.routed_action))}</span></div>
        <div data-col="status" data-col-label="${escapeHtml(s.status)}"><select class="status-select" data-patient="${escapeHtml(id)}" aria-label="${escapeHtml(`${r.display_name} ${s.status}`)}">${statusOptions(status)}</select></div>
        <div class="wl-expand" data-col="expand"><button class="expand-btn" type="button" data-expand="${escapeHtml(id)}" aria-expanded="${open}" aria-label="${escapeHtml(open ? s.hideRationale : s.viewRationale)}">${ICONS.chevron}</button></div>
      </div>
      ${open ? `<div class="wl-detail">
        <p>${escapeHtml(rationale(r))}</p>
        <div class="wl-detail-meta"><strong>${escapeHtml(routeLabel(r.routed_action))}</strong><span class="${outcomeClass}">${escapeHtml(outcomeLabel)}</span></div>
        ${r.escalation ? escalationDetail(r) : ""}
      </div>` : ""}
    </div>`;
  }).join("");

  if (state.justExpanded) {
    const d = dom.wlBody.querySelector(`.wl-group[data-id="${(window.CSS && CSS.escape) ? CSS.escape(state.justExpanded) : state.justExpanded}"] .wl-detail`);
    fadeSwap(d, 4);
    state.justExpanded = null;
  }

  dom.resultCount.textContent = s.showing(fmt(vis.length), fmt(all.length));
  dom.loadMore.hidden = vis.length >= all.length;
  dom.consoleFoot.hidden = false;
}

function resetFilters() {
  state.query = ""; state.route = "all"; state.esc = "all"; state.minRisk = 0; state.visible = PAGE_SIZE;
  dom.q.value = ""; dom.routeFilter.value = "all"; dom.riskFilter.value = "0"; dom.riskValue.textContent = "0%";
  if (dom.escFilter) dom.escFilter.value = "all";
  renderWorklist();
  fadeSwap(dom.wlBody, 6);
}
function patientName(id) { return rows().find((r) => r.patient_id === id)?.display_name || id; }

function onWorklistClick(e) {
  const sort = e.target.closest("[data-sort]");
  if (sort) {
    const k = sort.dataset.sort;
    if (state.sortKey === k) state.sortDir *= -1;
    else { state.sortKey = k; state.sortDir = k === "window" ? 1 : -1; }
    renderWorklist(); return;
  }
  const exp = e.target.closest("[data-expand]");
  if (exp) {
    const id = exp.dataset.expand;
    if (state.expanded.has(id)) state.expanded.delete(id);
    else { state.expanded.add(id); state.justExpanded = id; }
    renderWorklist(); return;
  }
  if (e.target.closest("[data-reset]")) resetFilters();
}
function onWorklistChange(e) {
  const sel = e.target.closest("[data-patient]");
  if (!sel) return;
  const id = sel.dataset.patient;
  if (sel.value === "actioned") openDialog(id, sel);
  else { writeWorkflow(id, { status: sel.value }); afterWorkflow(); toast(`${patientName(id)} · ${sel.options[sel.selectedIndex].text}`); }
}
function afterWorkflow() { renderConsoleStats(); renderWorklist(); renderShowcase(state.activeTab); }

/* ------------------------------- dialog ------------------------------- */
function defaultIntervention(driver) {
  if (driver === "transport_barrier" || driver === "transport") return "transport_mtm";
  if (["migrant_status", "low_education", "isolation", "language"].includes(driver)) return "language_chw";
  if (["housing_barrier", "financial_barrier", "food"].includes(driver)) return "referral_uniteus";
  return "refill_sync";
}
function openDialog(id, ret) {
  const r = rows().find((x) => x.patient_id === id);
  if (!r) return;
  state.activePatient = id; state.dialogReturn = ret || document.activeElement;
  const saved = readWorkflow()[id] || {};
  dom.dialogPatient.textContent = `${r.display_name} · ${driverLabel(r)}`;
  dom.dialogIntervention.innerHTML = Object.entries(INTERVENTIONS).map(([k, l]) => `<option value="${k}">${escapeHtml(l[state.lang] || l.en)}</option>`).join("");
  dom.dialogIntervention.value = saved.intervention || defaultIntervention(r.top_driver);
  dom.dialogReference.value = saved.reference || "";
  dom.dialogNotes.value = saved.note || "";
  dom.actionDialog.hidden = false; document.body.classList.add("is-locked");
  setTimeout(() => dom.dialogIntervention.focus(), 20);
}
function closeDialog() {
  if (dom.actionDialog.hidden) return;
  dom.actionDialog.hidden = true; document.body.classList.remove("is-locked");
  const ret = state.dialogReturn;
  // If the dialog was opened from a status <select> and the user cancelled,
  // re-sync the control to the stored status so UI and state never diverge.
  if (ret && ret.classList?.contains("status-select") && state.activePatient) ret.value = statusOf(state.activePatient);
  ret?.focus?.(); state.activePatient = null;
}
function saveDialog(e) {
  e.preventDefault();
  const id = state.activePatient; if (!id) return;
  writeWorkflow(id, { status: "actioned", intervention: dom.dialogIntervention.value, reference: dom.dialogReference.value.trim(), note: dom.dialogNotes.value.trim() });
  const name = patientName(id);
  closeDialog(); afterWorkflow(); toast(`${name} · ${S().actioned}`);
}

/* ---------------------------- command menu ---------------------------- */
function commandDefs() {
  const s = S();
  const defs = [
    { group: s.commands, icon: ICONS.command, title: s.jumpConsole, run: () => scrollTo("console") },
    { group: s.commands, icon: ICONS.command, title: s.resetFilters, run: () => { resetFilters(); scrollTo("console"); } },
  ];
  if (state.data && hasEscalation()) {
    defs.push({ group: s.commands, icon: ICONS.command, title: M().navEscalation, subtitle: E().cascadeTitle, run: () => scrollTo("escalation") });
    ["gated", "exhausted", "closed"].forEach((v) =>
      defs.push({ group: E().filterLabel, icon: ICONS.filter, title: escOptionLabel(v), subtitle: E().filterLabel, run: () => applyCopilotEsc(v) }));
  }
  LANGS.filter((c) => c !== state.lang).forEach((c) => defs.push({ group: s.commands, icon: ICONS.command, title: LANG_NAMES[c], subtitle: M().cmdLanguage, run: () => setLanguage(c) }));
  return defs;
}
function renderCommand(query) {
  const s = S(); const q = query.trim().toLowerCase();
  const patients = rows().filter((r) => !q || `${r.display_name} ${driverLabel(r)}`.toLowerCase().includes(q)).slice(0, 6)
    .map((r) => ({ group: s.patients, icon: ICONS.person, title: r.display_name, subtitle: `${driverLabel(r)} · ${routeLabel(r.routed_action)}`, run: () => focusPatient(r.patient_id) }));
  const cmds = commandDefs().filter((c) => !q || `${c.title} ${c.subtitle || ""}`.toLowerCase().includes(q));
  state.commandItems = [...patients, ...cmds]; state.commandIndex = 0;
  if (!state.commandItems.length) { dom.commandResults.innerHTML = `<div class="cmd-empty">${escapeHtml(s.noCommands)}</div>`; return; }
  let prev = "";
  dom.commandResults.innerHTML = state.commandItems.map((it, i) => {
    const g = it.group !== prev ? `<div class="cmd-group">${escapeHtml(it.group)}</div>` : ""; prev = it.group;
    return `${g}<button class="cmd-item" type="button" role="option" data-index="${i}" aria-selected="${i === 0}">
      <span class="cmd-icon">${it.icon}</span>
      <span class="cmd-copy"><strong>${escapeHtml(it.title)}</strong>${it.subtitle ? `<span>${escapeHtml(it.subtitle)}</span>` : ""}</span>
      <kbd>&#8629;</kbd></button>`;
  }).join("");
}
function highlightCommand() {
  dom.commandResults.querySelectorAll("[data-index]").forEach((el, i) => {
    const sel = i === state.commandIndex; el.setAttribute("aria-selected", String(sel));
    if (sel) el.scrollIntoView({ block: "nearest" });
  });
}
function runCommand(i) { const it = state.commandItems[i]; if (it) { closeCommand(); it.run(); } }
function openCommand() {
  state.commandReturn = document.activeElement; dom.commandInput.value = "";
  dom.commandDialog.hidden = false; document.body.classList.add("is-locked");
  renderCommand(""); setTimeout(() => dom.commandInput.focus(), 20);
}
function closeCommand() {
  if (dom.commandDialog.hidden) return;
  dom.commandDialog.hidden = true; document.body.classList.remove("is-locked"); state.commandReturn?.focus?.();
}
function focusPatient(id) {
  resetFilters();
  const r = rows().find((x) => x.patient_id === id); if (!r) return;
  state.query = r.display_name; dom.q.value = r.display_name; state.expanded.add(id); state.justExpanded = id;
  renderWorklist(); scrollTo("console");
}

/* ------------------------------ chrome ------------------------------ */
function scrollTo(id) {
  const el = $(id); if (!el) return;
  window.scrollTo({ top: el.getBoundingClientRect().top + scrollY - 74, behavior: reducedMotion() ? "auto" : "smooth" });
}
function toast(msg) {
  const el = document.createElement("div"); el.className = "toast"; el.textContent = msg;
  dom.toastRegion.append(el); setTimeout(() => el.remove(), 3200);
}
function toggleMobile() {
  const open = dom.mobilePanel.dataset.open !== "true";
  dom.mobilePanel.dataset.open = String(open); dom.mobileToggle.setAttribute("aria-expanded", String(open));
}
function closeMobile() { dom.mobilePanel.dataset.open = "false"; dom.mobileToggle.setAttribute("aria-expanded", "false"); }
function renderSources() {
  if (!state.data) return;
  dom.consoleMeta.textContent = `${sourceWord()} · ${syncedText()}`;
  dom.heroSource.textContent = sourceWord();
  dom.heroSync.textContent = syncedText();
}

function trapFocus(e, container) {
  const f = [...container.querySelectorAll('button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])')].filter((el) => el.offsetParent !== null);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}
function onKeydown(e) {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); dom.commandDialog.hidden ? openCommand() : closeCommand(); return; }
  if (e.key === "Escape") { if (!dom.actionDialog.hidden) closeDialog(); else if (!dom.commandDialog.hidden) closeCommand(); else closeMobile(); return; }
  if (!dom.commandDialog.hidden) {
    const n = state.commandItems.length;
    if (e.key === "ArrowDown") { e.preventDefault(); state.commandIndex = (state.commandIndex + 1) % Math.max(n, 1); highlightCommand(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); state.commandIndex = (state.commandIndex - 1 + n) % Math.max(n, 1); highlightCommand(); }
    else if (e.key === "Enter") { e.preventDefault(); runCommand(state.commandIndex); }
    else if (e.key === "Tab") trapFocus(e, dom.commandDialog.firstElementChild);
  } else if (!dom.actionDialog.hidden && e.key === "Tab") trapFocus(e, dom.actionDialog.firstElementChild);
}

/* ------------------------------ motion ------------------------------ */
function heroEntrance() {
  if (reducedMotion()) return;
  requestAnimationFrame(() => {
    document.querySelectorAll("[data-hero-word]").forEach((el, i) =>
      el.animate?.([{ opacity: 0, transform: "translateY(105%)" }, { opacity: 1, transform: "none" }], { duration: 760, delay: 90 + i * 65, easing: "cubic-bezier(.2,.75,.2,1)", fill: "both" }));
    document.querySelectorAll("[data-hero-rise]").forEach((el, i) =>
      el.animate?.([{ opacity: 0, transform: "translateY(16px)" }, { opacity: 1, transform: "none" }], { duration: 680, delay: 460 + i * 100, easing: "cubic-bezier(.2,.75,.2,1)", fill: "both" }));
  });
}
function initReveal() {
  const els = document.querySelectorAll("[data-reveal]");
  if (reducedMotion() || !("IntersectionObserver" in window)) { els.forEach((el) => el.classList.add("is-visible")); return; }
  const io = new IntersectionObserver((ents) => ents.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("is-visible"); io.unobserve(en.target); } }), { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
  els.forEach((el) => io.observe(el));
}
function countUp(scope) {
  scope.querySelectorAll("[data-count]:not([data-done])").forEach((el) => {
    el.dataset.done = "1";
    const to = Number(el.dataset.count) || 0;
    const suffix = /%$/.test(el.textContent) ? "%" : "";
    if (reducedMotion()) { el.textContent = fmt(to) + suffix; return; }
    const dur = 1050, st = performance.now();
    const tick = (now) => { const p = Math.min((now - st) / dur, 1); const e = 1 - Math.pow(1 - p, 3); el.textContent = fmt(Math.round(to * e)) + suffix; if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  });
}
function observeCount(el, onSeen) {
  if (!el) return;
  if (!("IntersectionObserver" in window)) { onSeen(); return; }
  const io = new IntersectionObserver((ents) => ents.forEach((en) => { if (en.isIntersecting) { io.unobserve(en.target); onSeen(); } }), { threshold: 0.4 });
  io.observe(el);
}
function initNavHighlight() {
  const links = [...document.querySelectorAll(".desktop-nav a")];
  const sections = links.map((a) => $(a.hash.slice(1))).filter(Boolean);
  if (!("IntersectionObserver" in window) || !sections.length) return;
  const io = new IntersectionObserver((ents) => {
    const vis = ents.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!vis) return;
    links.forEach((a) => { if (a.hash === `#${vis.target.id}`) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
  }, { rootMargin: "-20% 0px -65% 0px", threshold: [0, 0.25, 0.6] });
  sections.forEach((sec) => io.observe(sec));
}

/* --------------------------- data + boot --------------------------- */
let fetchAbort = null;
async function fetchData() {
  if (fetchAbort) fetchAbort.abort();
  fetchAbort = new AbortController();
  const { signal } = fetchAbort;
  try {
    const res = await fetch(API_URL, { cache: "no-store", signal });
    if (!res.ok) throw new Error(String(res.status));
    return { data: await res.json(), source: "live" };
  } catch (err) {
    if (err?.name === "AbortError") throw err;
    const res = await fetch(BUNDLED_URL, { cache: "no-store", signal });
    if (!res.ok) throw new Error(`snapshot ${res.status}`);
    return { data: await res.json(), source: "bundled" };
  }
}
async function loadData({ initial = false } = {}) {
  try {
    const { data, source } = await fetchData();
    // Include outcome fingerprint so loop_outcomes updates re-render even when
    // routing_table meta (generated_at / last_synced / length) is unchanged.
    let onTime = 0, observed = 0;
    for (const r of data.worklist || []) {
      const o = r.loop_outcome;
      if (!o) continue;
      if (o.observed) observed += 1;
      if (o.on_time_refill) onTime += 1;
    }
    const sig = `${data.generated_at}|${data.last_synced}|${data.worklist?.length || 0}|${source}|o${observed}|t${onTime}`;
    if (!initial && sig === state.lastSignature) { renderSources(); return; }
    state.lastSignature = sig; state.data = data; state.source = source;
    renderHero(); renderConsoleStats(); renderWorklist(); renderEscalation(); renderCopilot(); renderShowcase(state.activeTab); renderSources();
  } catch (err) {
    if (err?.name === "AbortError") return;
    if (state.data) return;
    dom.worklist.setAttribute("aria-busy", "false");
    dom.wlHead.innerHTML = "";
    dom.wlBody.innerHTML = `<div class="wl-error"><strong>${escapeHtml(S().dataUnavailable)}</strong><p>${escapeHtml(S().apiHint)}</p><code>uvicorn src.api.main:app --port 8000</code><br><button class="button button-outline" type="button" data-retry>${escapeHtml(S().retry)}</button></div>`;
    dom.wlBody.querySelector("[data-retry]")?.addEventListener("click", () => loadData({ initial: true }));
    dom.consoleFoot.hidden = true;
  }
}

function cacheDom() {
  ["site-nav", "lang-switch", "mobile-lang-switch", "command-trigger", "mobile-toggle", "mobile-panel",
    "hero-signal-list", "hero-queue-count", "hero-source", "hero-sync", "care-rail",
    "evidence-grid", "workflow-cards", "wf-before", "wf-after",
    "product-tabs", "tab-copy", "screen-body", "copilot-prompts", "copilot-answer", "copilot-note",
    "trust-stack", "method-grid", "console-stats", "console-meta", "console-controls",
    "esc-cascade", "esc-source", "esc-guardrails", "esc-filter",
    "q", "route-filter", "risk-filter", "risk-value", "reset-filters",
    "worklist", "wl-head", "wl-body", "console-foot", "result-count", "load-more",
    "command-dialog", "command-input", "command-results",
    "action-dialog", "dialog-close", "dialog-cancel", "dialog-patient", "action-form",
    "dlg-intervention", "dlg-reference", "dlg-notes", "toast-region",
  ].forEach((id) => { dom[id.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = $(id); });
  dom.dialogIntervention = dom.dlgIntervention;
  dom.dialogReference = dom.dlgReference;
  dom.dialogNotes = dom.dlgNotes;
  dom.dialogCancel = $("dialog-cancel");
  dom.dialogSave = $("dialog-save");
  dom.escSection = $("escalation");
}

function boot() {
  cacheDom();
  document.documentElement.lang = state.lang;
  renderLangSwitch();
  applyMarketing();
  applyConsoleLabels();
  renderEvidence(); renderRail(); renderWorkflow("after"); renderTrust(); renderMethod();
  renderProductTabs(); renderTabCopy(); renderCopilot();
  heroEntrance();
  initReveal();
  initNavHighlight();

  dom.langSwitch.addEventListener("click", (e) => { const b = e.target.closest("[data-lang]"); if (b) setLanguage(b.dataset.lang); });
  dom.mobileLangSwitch?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-lang]");
    if (b) { setLanguage(b.dataset.lang); closeMobile(); }
  });
  dom.commandTrigger.addEventListener("click", openCommand);
  dom.mobileToggle.addEventListener("click", toggleMobile);
  dom.mobilePanel.querySelectorAll("a").forEach((a) => a.addEventListener("click", closeMobile));
  dom.wfBefore.addEventListener("click", () => renderWorkflow("before"));
  dom.wfAfter.addEventListener("click", () => renderWorkflow("after"));
  dom.productTabs.addEventListener("click", (e) => { const t = e.target.closest("[data-tab]"); if (t) { renderShowcase(t.dataset.tab); renderCopilot(); } });
  dom.copilotPrompts.addEventListener("click", (e) => { const c = e.target.closest("[data-prompt]"); if (c) runPrompt(c.dataset.prompt); });
  dom.copilotAnswer.addEventListener("click", (e) => {
    const a = e.target.closest("[data-apply]"); if (a) { applyCopilotRoute(a.dataset.apply); return; }
    const ae = e.target.closest("[data-apply-esc]"); if (ae) applyCopilotEsc(ae.dataset.applyEsc);
  });

  dom.q.addEventListener("input", debounce((e) => {
    state.query = e.target.value; state.visible = PAGE_SIZE; renderWorklist();
  }, 180));
  dom.routeFilter.addEventListener("change", (e) => { state.route = e.target.value; state.visible = PAGE_SIZE; renderWorklist(); fadeSwap(dom.wlBody, 6); });
  dom.escFilter?.addEventListener("change", (e) => { state.esc = e.target.value; state.visible = PAGE_SIZE; renderWorklist(); fadeSwap(dom.wlBody, 6); });
  const onRiskInput = debounce(() => {
    state.visible = PAGE_SIZE; renderWorklist();
  }, 90);
  dom.riskFilter.addEventListener("input", (e) => {
    state.minRisk = Number(e.target.value);
    dom.riskValue.textContent = `${state.minRisk}%`;
    onRiskInput();
  });
  // Fade once on release so continuous slider input stays snappy.
  dom.riskFilter.addEventListener("change", () => fadeSwap(dom.wlBody, 6));
  dom.consoleControls.addEventListener("reset", (e) => { e.preventDefault(); resetFilters(); });
  dom.consoleControls.addEventListener("submit", (e) => e.preventDefault());
  dom.loadMore.addEventListener("click", () => { state.visible += PAGE_SIZE; renderWorklist(); fadeSwap(dom.wlBody, 4); });
  dom.worklist.addEventListener("click", onWorklistClick);
  dom.worklist.addEventListener("change", onWorklistChange);

  dom.dialogClose.addEventListener("click", closeDialog);
  dom.dialogCancel.addEventListener("click", closeDialog);
  dom.actionDialog.addEventListener("click", (e) => { if (e.target === dom.actionDialog) closeDialog(); });
  dom.actionForm.addEventListener("submit", saveDialog);

  dom.commandDialog.addEventListener("click", (e) => { if (e.target === dom.commandDialog) closeCommand(); });
  dom.commandInput.addEventListener("input", () => renderCommand(dom.commandInput.value));
  dom.commandResults.addEventListener("mousemove", (e) => { const it = e.target.closest("[data-index]"); if (it) { state.commandIndex = Number(it.dataset.index); highlightCommand(); } });
  dom.commandResults.addEventListener("click", (e) => { const it = e.target.closest("[data-index]"); if (it) runCommand(Number(it.dataset.index)); });
  addEventListener("keydown", onKeydown);

  const onScroll = () => {
    dom.siteNav?.classList.toggle("is-scrolled", scrollY > 8);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const kbd = $("command-kbd");
  if (kbd) {
    const mac = /Mac|iPhone|iPad|iPod/i.test(navigator.platform || "") || navigator.userAgentData?.platform === "macOS";
    kbd.textContent = mac ? "⌘K" : "Ctrl K";
  }

  observeCount(dom.evidenceGrid, () => { state.evidenceAnimated = true; countUp(dom.evidenceGrid); });
  observeCount($("console"), () => { state.statsAnimated = true; countUp(dom.consoleStats); });

  loadData({ initial: true });
  setInterval(() => { if (document.visibilityState === "visible") loadData(); }, POLL_INTERVAL);
  setInterval(() => { if (state.data && document.visibilityState === "visible") renderSources(); }, 15000);
}

if (document.readyState === "loading") addEventListener("DOMContentLoaded", boot);
else boot();
