const API = "/api/v1/research";
const KEYS = { run: "refine.activeRun", query: "refine.activeQuery", job: "refine.pipelineJob", question: "refine.question" };
const stages = ["acquire", "parse", "sections", "chunks", "represent", "kg", "gaps", "validate"];
const pipelineSequence = ["query", "collect", ...stages];
const pipelinePrerequisites = {
  parse: ["acquire"], sections: ["parse"], chunks: ["sections"], represent: ["chunks"],
  kg: ["chunks", "represent"], gaps: ["kg"], validate: ["gaps"],
};
const stageNames = { acquire: "Acquire full text", parse: "Parse documents", sections: "Extract sections", chunks: "Build chunks", represent: "Create embeddings", kg: "Build knowledge graph", gaps: "Identify candidate gaps", validate: "Validate gaps" };
const sequenceNames = { query: "Understand research question", collect: "Collect literature", ...stageNames };
const nav = [
  ["workspace", "⌂", "Workspace"], ["new", "+", "Full investigation"], ["pipeline", "↗", "Run pipeline"],
  ["literature", "▤", "Literature"], ["gaps", "◈", "Candidate gaps"], ["validation", "✓", "Validation"],
  ["evidence", "◎", "Evidence map"], ["report", "▧", "Research report"],
];
const state = {
  runId: localStorage.getItem(KEYS.run) || "", queryId: localStorage.getItem(KEYS.query) || "",
  question: localStorage.getItem(KEYS.question) || "", route: location.hash.slice(1) || "workspace",
  health: null, runs: [], runsLoaded: false, loadingRuns: false, papers: [], papersLoaded: false, paperOffset: 0, paperSort: "downloaded", gaps: null, validation: null, loadingValidation: false, graph: null, loadingGraph: false, busy: false,
  pipeline: JSON.parse(localStorage.getItem("refine.pipeline") || "{}"), queryResult: null,
  gapOptions: JSON.parse(localStorage.getItem("refine.gapOptions") || "{\"llmWording\":false,\"llmCandidates\":false}"),
  selectedGapIds: new Set(),
  latestApiResponse: null,
  responseDisclosureState: new Map(),
  fullWorkflow: JSON.parse(localStorage.getItem("refine.fullWorkflow") || "null"),
};
const app = document.querySelector("#app");

function esc(value) { return String(value ?? "").replace(/[&<>\"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]); }
function setStored(key, value) { if (value) localStorage.setItem(key, value); else localStorage.removeItem(key); }
function activeRun(id, question = state.question, queryId = state.queryId) {
  if ((id || "") !== state.runId) state.selectedGapIds.clear();
  state.runId = id || ""; state.question = question || ""; state.queryId = queryId || "";
  setStored(KEYS.run, state.runId); setStored(KEYS.question, state.question); setStored(KEYS.query, state.queryId);
}
function toast(message, kind = "info") {
  const host = document.querySelector("#toast-host"); if (!host) return;
  const item = document.createElement("div"); item.className = `toast ${kind}`; item.textContent = message; host.append(item); setTimeout(() => item.remove(), 5200);
}
async function api(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  let response;
  try {
    response = await fetch(`${API}${path}`, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  } catch (error) {
    updateLatestApiResponse({ method, path, status: null, ok: false, body: { error: error.message } });
    throw error;
  }
  const raw = await response.text(); let data;
  try { data = raw ? JSON.parse(raw) : {}; } catch { data = { detail: raw }; }
  updateLatestApiResponse({ method, path, status: response.status, ok: response.ok, body: data });
  if (!response.ok) throw new Error(data.detail || data.error || data.message || `Request failed (${response.status})`);
  return data;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });
function card(title, body, extra = "") { return `<section class="card ${extra}"><div class="card-body">${title}${body}</div></section>`; }
function stat(value, label, note = "") { return `<div class="stat"><strong>${esc(value)}</strong><span>${esc(label)}</span>${note ? `<small>${esc(note)}</small>` : ""}</div>`; }
function badge(value) { const style = String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-"); return `<span class="badge ${style}">${esc(value || "Unknown")}</span>`; }
function formatJson(value) { try { return esc(JSON.stringify(value ?? {}, null, 2)); } catch { return esc(String(value)); } }
function responseDetails(label, value, defaultOpen = false, key = label) {
  if (value == null) return "";
  const open = state.responseDisclosureState.has(key) ? state.responseDisclosureState.get(key) : defaultOpen;
  return `<details class="response-details" data-response-key="${esc(key)}" ${open ? "open" : ""}><summary>${esc(label)}</summary><pre><code>${formatJson(value)}</code></pre></details>`;
}
function latestApiResponseHtml() {
  const response = state.latestApiResponse;
  if (!response) return `<div class="latest-api-response empty-response"><b>API responses</b><small>Responses from recent requests will appear here as formatted JSON.</small></div>`;
  const status = response.status == null ? "Network error" : `HTTP ${response.status}`;
  return `<details class="latest-api-response" id="latest-api-response"><summary><span><b>Latest API response</b><small>${esc(response.method)} ${esc(response.path)} · ${status}</small></span><span class="response-toggle">View JSON⌄</span></summary><pre><code>${formatJson(response.body)}</code></pre></details>`;
}
function updateLatestApiResponse(response) {
  state.latestApiResponse = { ...response, receivedAt: new Date().toISOString() };
  const slot = document.querySelector("#latest-api-response-slot");
  if (slot) slot.innerHTML = latestApiResponseHtml();
}
function shell(content) {
  app.innerHTML = `<div class="layout"><aside class="sidebar"><a class="brand" href="#workspace"><span class="brand-mark">r</span><span>refine<small>RESEARCH WORKSPACE</small></span></a><div class="side-label">WORKSPACE</div><div class="nav-list">${nav.map(([id, icon, label]) => `<a class="nav-link ${state.route === id ? "active" : ""}" href="#${id}"><span class="nav-icon">${icon}</span>${label}</a>`).join("")}</div><div class="sidebar-spacer"></div><div class="side-run"><div class="side-label">ACTIVE RUN</div>${state.runId ? `<b>${esc(state.runId.slice(0, 8))}…</b><small>${esc(state.question || "Research run")}</small>` : `<small>No run selected</small>`}<a href="#pipeline">Open pipeline →</a></div><div class="sidebar-foot"><span class="health-dot ${state.health?.status === "ok" ? "online" : "offline"}"></span>${state.health?.status === "ok" ? "API connected" : "API unavailable"}<a href="/docs" target="_blank" rel="noreferrer">API docs ↗</a></div></aside><main class="main"><header class="topbar"><div><span class="eyebrow">EVIDENCE-WEIGHTED RESEARCH</span><span class="top-title">${state.runId ? `Run <code>${esc(state.runId)}</code>` : "Workspace"}</span></div><div class="top-actions"><button class="button subtle small" data-action="refresh">↻ Refresh</button><a class="button dark small" href="#new">＋ New investigation</a></div></header><div class="page">${content}<div id="latest-api-response-slot">${latestApiResponseHtml()}</div></div></main></div><div id="toast-host" class="toast-host"></div>`;
 }
function heading(kicker, title, desc, action = "") { return `<div class="page-heading"><div><div class="eyebrow">${esc(kicker)}</div><h1>${title}</h1><p>${desc}</p></div>${action}</div>`; }
function empty(title, detail, action = "") { return `<div class="empty"><div class="empty-icon">✳</div><h3>${title}</h3><p>${detail}</p>${action}</div>`; }
function requireRun() { return state.runId ? "" : empty("Choose a research run", "Start an investigation or load an existing run to see its results.", `<a class="button dark" href="#new">Start an investigation</a>`); }
function rowRun(run) { return `<button class="run-row" data-action="select-run" data-run="${esc(run.run_id)}" data-question="${esc(run.research_question)}" data-query="${esc(run.query_id)}" data-paper-count="${esc(run.paper_count ?? 0)}"><span class="run-glyph">◈</span><span class="run-main"><b>${esc(run.research_question || "Untitled investigation")}</b><small>${esc(run.run_id)} · ${run.paper_count ?? 0} papers · ${esc(run.created_at || "")}</small></span><span class="arrow">↗</span></button>`; }
function workspace() {
  return `${heading("RESEARCH WORKSPACE", "Your research workspace.", "Investigations, literature, candidate gaps, and validation results.", `<a class="button dark" href="#new">＋ New investigation</a>`)}<div class="welcome-grid"><div class="welcome-card"><span class="eyebrow">YOUR RESEARCH, ORGANIZED</span><h2>Research<br><em>workspace.</em></h2><p>Collect literature, map its concepts, surface candidate gaps, and validate each claim against sources.</p><a class="text-link" href="#new">Begin an investigation <span>→</span></a><div class="welcome-art"><span>QUESTION</span><i>→</i><span>EVIDENCE</span><i>→</i><span>GAP</span></div></div><div class="overview-card"><div class="section-head"><div><span class="eyebrow">CURRENT SESSION</span><h3>${state.runId ? "Active investigation" : "Ready when you are"}</h3></div>${state.runId ? badge("ACTIVE") : badge("IDLE")}</div>${state.runId ? `<p class="muted">${esc(state.question || "Research run")}</p><code class="run-code">${esc(state.runId)}</code><div class="overview-links"><a href="#pipeline">Continue pipeline →</a><a href="#literature">Browse literature →</a></div>` : `<p class="muted">Create a query to begin, or load a run ID you already have.</p><a class="button outline" href="#new">Create a research query</a>`}<form id="load-run-form" class="load-form"><label for="load-run-id">Load run by ID</label><div class="inline-form"><input id="load-run-id" value="${esc(state.runId)}"><button class="button outline small">Load</button></div></form></div></div><div class="section-head recent-heading"><div><span class="eyebrow">HISTORY</span><h2>Recent investigations</h2></div><button class="text-button" data-action="refresh">Refresh list ↻</button></div>${state.runs.length ? `<div class="run-list">${state.runs.map(rowRun).join("")}</div>` : empty("No saved runs", "")}`;
}
function newInvestigation() {
  const workflow = state.fullWorkflow;
  const phases = [
    ...pipelineSequence.map(stage => [stage, sequenceNames[stage]]),
  ];
  const done = phases.filter(([key]) => workflow?.phases?.[key] === "COMPLETED").length;
  const progress = Math.round(done / phases.length * 100);
  const results = state.validation?.validation_results || state.validation?.results || [];
  const candidates = state.gaps?.candidates || [];
  const resultCards = results.map((result, index) => {
    const candidate = candidates.find(gap => gap.gap_id === result.gap_id);
    const evidence = result.evidence || [];
    const statement = result.gap_statement || candidate?.gap_statement || result.gap_id || "Validated candidate";
    return `<article class="validated-gap">
      <div class="section-head"><span class="eyebrow">GAP ${String(result.gap_rank || index + 1).padStart(2, "0")}</span>${badge(result.classification)}</div>
      <h3>${esc(statement)}</h3>
      <p>${esc(result.explanation || result.reason || "")}</p>
      <div class="confidence"><span>Confidence</span><b>${result.confidence == null ? "—" : `${Math.round(result.confidence * 100)}%`}</b></div>
      <details open><summary>Evidence (${evidence.length})</summary>
        ${evidence.length ? evidence.map(item => {
          const link = /^https?:\/\//i.test(item.url || "") ? `<a href="${esc(item.url)}" target="_blank" rel="noreferrer">Open source ↗</a>` : "";
          return `<blockquote><b>${esc(item.title || item.source || "Source")}</b>${item.year ? ` · ${esc(item.year)}` : ""}<span>${esc(item.snippet || item.text || item.quote || item.why || "")}</span><small>${esc(item.doi || item.source || "")} ${link}</small></blockquote>`;
        }).join("") : "<p>No source excerpts were returned for this candidate.</p>"}
      </details>
    </article>`;
  }).join("");
  return `${heading("FULL INVESTIGATION", "From question to evidence.", "Enter a research question. Refine will run the complete workflow and present validated gaps with their supporting evidence.")}
    <section class="card full-query-card"><div class="card-body">
      <form id="query-form">
        <label for="research-question">Research question</label>
        <textarea id="research-question" name="query" rows="4" required ${state.busy ? "disabled" : ""}>${esc(state.question)}</textarea>
        <div class="form-foot"><small>One action runs query understanding, search expansion, collection, processing, gap detection, and validation.</small>
          <button class="button dark" ${state.busy ? "disabled" : ""}>${state.busy ? "Investigation running…" : "▶ Run full investigation"}</button>
        </div>
      </form>
    </div></section>
    ${workflow ? `<section class="card workflow-card"><div class="card-body">
      <div class="section-head"><div><span class="eyebrow">INVESTIGATION PROGRESS</span><h2>${workflow.status === "COMPLETED" ? "Results are ready" : workflow.status === "FAILED" ? "Workflow stopped" : "Running your investigation"}</h2></div>${badge(workflow.status)}</div>
      <p class="workflow-question">${esc(workflow.question || state.question)}</p>
      ${workflow.runId ? `<p class="workflow-run">Run ID <code>${esc(workflow.runId)}</code> · ${esc(workflow.paperCount ?? 0)} papers collected${workflow.collectionStatus === "partial" ? " (partial collection)" : ""}</p>` : ""}
      <div class="progress-wrap workflow-progress"><div class="progress"><i style="width:${progress}%"></i></div><small>${progress}%</small></div>
      <div class="workflow-steps">${phases.map(([key, label]) => {
        const status = workflow.phases?.[key] || (workflow.current === key ? "RUNNING" : "PENDING");
        return `<div class="workflow-step ${status.toLowerCase()}"><span>${status === "COMPLETED" ? "✓" : status === "FAILED" ? "!" : "·"}</span><b>${esc(label)}</b>${badge(status)}</div>`;
      }).join("")}</div>
      ${workflow.error ? `<div class="inline-error">${esc(workflow.error)}</div>` : ""}
      ${workflow.expandedQueries?.length ? `<details class="expanded-searches"><summary>Expanded searches (${workflow.expandedQueries.length})</summary><ol>${workflow.expandedQueries.map(query => `<li>${esc(query)}</li>`).join("")}</ol></details>` : ""}
    </div></section>` : ""}
    ${workflow?.status === "COMPLETED" ? `<section class="validated-results">
      <div class="section-head"><div><span class="eyebrow">VALIDATED FINDINGS</span><h2>Candidate gaps and evidence</h2></div>${badge(results.length ? "COMPLETE" : "NO CANDIDATES")}</div>
      ${resultCards || `<div class="empty-compact">Validation completed, but no candidate gap results were returned.</div>`}
    </section>` : ""}
  `;
}
function pipeline() {
  const workflow = state.fullWorkflow;
  const workflowActive = workflow?.status === "RUNNING";
  const phaseStatus = (phase, stage) => {
    const pipelineStatus = stage ? state.pipeline[stage]?.status : null;
    return pipelineStatus || workflow?.phases?.[phase] || (state.runId && ["query", "collect"].includes(phase) ? "COMPLETED" : "NOT_STARTED");
  };
  const completed = pipelineSequence.filter(step => {
    if (step === "query" || step === "collect") return workflow?.phases?.[step] === "COMPLETED" || Boolean(state.runId && !workflowActive);
    return state.pipeline[step]?.status === "COMPLETED" || workflow?.phases?.[step] === "COMPLETED";
  }).length;
  const percent = Math.round(completed / pipelineSequence.length * 100);
  const gapOptionsPanel = `<section class="card gap-options"><div class="card-body"><div class="gap-options-heading"><b>LLM assistance for gap detection</b><small>Both options are off by default. LLM results remain candidate hypotheses and still need EWRGV validation.</small></div><label><input id="llm-wording" type="checkbox" ${state.gapOptions.llmWording ? "checked" : ""}> Refine signal-based gap wording</label><small>Rephrase existing signal candidates using their cited passages; keep the original wording and scores.</small><label><input id="llm-candidates" type="checkbox" ${state.gapOptions.llmCandidates ? "checked" : ""}> Add LLM-proposed candidates</label><small>Generate additional proposals from reranked corpus passages and include their cited evidence in Candidate Gaps.</small><p>Requires a configured LLM API key. Candidates without valid corpus citations are discarded.</p></div></section>`;
  const action = state.runId ? `<button class="button green" data-action="run-pipeline" ${state.busy ? "disabled" : ""}>${state.busy ? "Pipeline running…" : "▶ Run remaining stages"}</button>` : "";
  const intro = heading("RUN PIPELINE", "Move the evidence forward.", "Follow the complete sequence from query understanding and literature collection through processing, candidate identification, and validation. Stages are gated by their prerequisites.", action);
  const workflowSummary = workflow && (workflowActive || !state.runId)
    ? `<div class="pipeline-summary"><div><span class="eyebrow">INVESTIGATION PROGRESS</span><h3>${workflow.status === "FAILED" ? "Investigation stopped" : workflow.status === "COMPLETED" ? "Investigation complete" : `Step: ${sequenceNames[workflow.current] || "Investigation"}`}</h3>${workflow.runId ? `<small>Run ${esc(workflow.runId)} · ${esc(workflow.paperCount ?? 0)} papers collected</small>` : ""}</div><div class="progress-wrap"><div class="progress"><i style="width:${percent}%"></i></div><small>${percent}%</small></div></div>` : "";
  if (!state.runId) {
    if (!workflow) return `${intro}${requireRun()}`;
    const earlyStages = ["query", "collect"].map((step, i) => {
      const status = phaseStatus(step);
      const detail = step === workflow.current ? workflow.error || "In progress…" : status === "COMPLETED" ? "Completed." : "Waiting to start.";
      return `<article class="stage-card ${status.toLowerCase()}"><span class="stage-number">${String(i + 1).padStart(2, "0")}</span><div class="stage-copy"><b>${sequenceNames[step]}</b><small>${esc(detail)}</small></div>${badge(status.replaceAll("_", " "))}</article>`;
    }).join("");
    return `${intro}${workflowSummary}<div class="stage-list">${earlyStages}</div>`;
  }
  const runStages = `<div class="pipeline-summary"><div><span class="eyebrow">PIPELINE PROGRESS</span><h3>${completed} of ${pipelineSequence.length} complete</h3></div><div class="progress-wrap"><div class="progress"><i style="width:${percent}%"></i></div><small>${percent}%</small></div></div>${gapOptionsPanel}<div class="stage-list">${pipelineSequence.map((step, i) => {
    if (step === "query" || step === "collect") {
      const details = step === "query"
        ? (state.question ? `Question: ${state.question}` : "Research question is attached to this run.")
        : `Run ${state.runId.slice(0, 8)}… exists; its collected literature is the input to acquisition.`;
      const status = phaseStatus(step, step === "query" ? null : "collect");
      return `<article class="stage-card ${status.toLowerCase()} pipeline-context-step"><span class="stage-number">${String(i + 1).padStart(2, "0")}</span><div class="stage-copy"><b>${sequenceNames[step]}</b><small>${esc(details)}</small></div>${badge(status.replaceAll("_", " "))}</article>`;
    }
    const s = step;
    const result = state.pipeline[s] || {};
    const status = result.status || "NOT_STARTED";
    const prerequisites = pipelinePrerequisites[s] || [];
    const missing = prerequisites.filter(required => state.pipeline[required]?.status !== "COMPLETED");
    const gateHint = missing.length ? `Complete first: ${missing.map(required => sequenceNames[required]).join(", ")}.` : stageHint(s);
    const disabled = state.busy || missing.length > 0;
    return `<article class="stage-card ${status.toLowerCase()} ${missing.length ? "stage-blocked" : ""}"><span class="stage-number">${String(i + 1).padStart(2, "0")}</span><div class="stage-copy"><b>${stageNames[s]}</b><small>${esc(result.message || gateHint)}</small>${result.job_id ? `<small class="job-id">Job ${esc(result.job_id)}</small>` : ""}</div>${badge(status.replaceAll("_", " "))}<div class="stage-actions">${result.job_id ? `<button class="button subtle small" data-action="check-job-status" data-stage="${s}" ${state.busy ? "disabled" : ""}>Check status</button>` : ""}<button class="button outline small" data-action="run-stage" data-stage="${s}" ${disabled ? "disabled" : ""}>${status === "COMPLETED" ? "Run again" : "Run"}</button></div>${responseDetails("Job start response (POST)", result.startResponse, false, `${s}:start`)}${responseDetails("Latest job status response (GET)", result.statusResponse, result.showStatusResponse, `${s}:status`)}${responseDetails("Job result response (GET)", result.resultResponse, false, `${s}:result`)}</article>`;
  }).join("")}</div><div class="pipeline-log" id="pipeline-log">${state.busy ? `<span class="pulse"></span> ${esc(state.pipeline.currentMessage || "Working…")}` : ""}</div>`;
  return `${intro}${workflowSummary}${runStages}`;
}
function stageHint(s) { return ({ acquire: "Download available full text for papers in this run.", parse: "Extract text from acquired PDFs.", sections: "Classify document sections such as methods and results.", chunks: "Create retrieval-sized text chunks.", represent: "Create vector representations for semantic search.", kg: "Extract entities and relationships into the knowledge graph.", gaps: "Identify gaps with signal detectors and reranked evidence; optionally add LLM-proposed candidates or refine signal-based wording.", validate: "Check candidate gaps against literature sources." })[s]; }
function invalidateDownstreamStages(stage) {
  const stageIndex = stages.indexOf(stage);
  for (const downstream of stages.slice(stageIndex + 1)) {
    if (state.pipeline[downstream]?.status === "COMPLETED" || state.pipeline[downstream]?.job_id) {
      state.pipeline[downstream] = { status: "NOT_STARTED", message: `Run again after ${stageNames[stage]} was rerun.` };
    }
  }
  savePipeline();
}
function papersPage() {
  const body = state.papers.length ? `<div class="table-scroll"><table><thead><tr><th>PAPER</th><th>YEAR</th><th>FULL TEXT</th><th>PARSED</th><th>CHUNKS</th></tr></thead><tbody>${state.papers.map(p => `<tr><td><b>${esc(p.title || "Untitled paper")}</b><small>${esc(p.doi || p.paper_id || "")}</small></td><td>${esc(p.publication_year || "—")}</td><td>${badge(p.full_text_status)}</td><td>${badge(p.parsing_status)}</td><td>${badge(p.chunk_status)}</td></tr>`).join("")}</tbody></table></div><div class="pager"><span>Showing ${state.paperOffset + 1}–${state.paperOffset + state.papers.length} of ${state.paperTotal || state.papers.length} papers</span><div class="pager-actions"><button class="button outline small" data-action="papers-prev" ${state.paperOffset <= 0 ? "disabled" : ""}>← Previous</button><button class="button outline small" data-action="papers-next" ${state.paperOffset + state.papers.length >= state.paperTotal ? "disabled" : ""}>Next →</button><button class="button outline small" data-action="export-bib">Export BibTeX</button></div></div>` : empty("No papers loaded", "Choose a run to browse its collected papers.");
  return `${heading("LITERATURE EXPLORER", "Collected literature.", "Search papers and review processing status.", `<button class="button outline" data-action="export-bib" ${state.papers.length ? "" : "disabled"}>Export BibTeX</button>`)}${requireRun()}${state.runId ? `<div class="toolbar"><label class="sr-only" for="paper-search">Search papers</label><input id="paper-search" aria-label="Search papers" value="${esc(state.paperSearch || "")}"><button class="button outline" data-action="search-papers">Search</button><label for="paper-sort">Sort</label><select id="paper-sort"><option value="downloaded" ${state.paperSort === "downloaded" ? "selected" : ""}>Downloaded first</option><option value="year_desc" ${state.paperSort === "year_desc" ? "selected" : ""}>Newest first</option><option value="year_asc" ${state.paperSort === "year_asc" ? "selected" : ""}>Oldest first</option><option value="title_asc" ${state.paperSort === "title_asc" ? "selected" : ""}>Title A–Z</option><option value="title_desc" ${state.paperSort === "title_desc" ? "selected" : ""}>Title Z–A</option></select><span class="toolbar-count">${state.paperTotal ?? "—"} papers</span></div>${card("", body)}` : ""}`;
}
function gapsPage() {
  const payload = state.gaps?.gaps || state.gaps?.candidates || state.gaps?.results || [];
  const list = Array.isArray(payload) ? payload : [];
  const selectedCount = state.selectedGapIds.size;
  const content = list.length ? `<div class="gap-list">${list.map((g, i) => {
    const id = g.gap_id || g.id || `gap-${i + 1}`;
    const evidence = g.evidence_passages || g.supporting_statements || [];
    const score = g.gap_score ?? g.score;
    const source = g.candidate_source || (g.detectors?.includes("llm") ? "llm" : "signal");
    const statement = g.gap_statement || g.title || g.gap || g.statement || "Candidate gap";
    const original = g.wording?.original_statement;
    const llmProposal = g.llm_proposal;
    return `<article class="gap-card"><div class="gap-top"><span class="gap-index">${String(g.rank || i + 1).padStart(2, "0")}</span><div class="gap-badges">${badge(source === "signal+llm" ? "Signal + LLM" : source === "llm" ? "LLM proposal" : "Signal based")}${badge(g.status || "CANDIDATE")}</div></div><h3>${esc(statement)}</h3><p>${esc(g.rationale || g.description || g.explanation || g.score_notes?.rationale || g.score_notes?.summary || "")}</p>${original && original !== statement ? `<details class="wording-provenance"><summary>Original signal wording</summary><p>${esc(original)}</p></details>` : ""}${llmProposal ? `<details class="wording-provenance"><summary>Matching LLM proposal</summary><p>${esc(llmProposal.gap_statement)}</p><small>LLM confidence ${esc(llmProposal.confidence)} · ${esc(llmProposal.rationale || "")}</small>${llmProposal.evidence_passages?.length ? `<blockquote>${llmProposal.evidence_passages.slice(0,2).map(p => `${esc(p.title || "Source")}: ${esc(p.snippet || "")}`).join("<br>")}</blockquote>` : ""}</details>` : ""}${g.detectors_agreeing?.length ? `<div class="tag-list">${g.detectors_agreeing.map(d => `<span>${esc(d)}</span>`).join("")}</div>` : ""}${evidence.length ? `<details><summary>${evidence.length} supporting evidence items</summary>${evidence.slice(0,4).map(e => `<blockquote><b>${esc(e.title || e.paper || "")}</b>${e.year ? ` · ${esc(e.year)}` : ""}<span>${esc(e.quote || e.sentence || e.snippet || e.text || "")}</span></blockquote>`).join("")}</details>` : ""}<div class="gap-meta">${score != null ? `<span>Pre-validation score ${esc(Number(score).toFixed(2))}</span>` : ""}${evidence.length ? `<span>${evidence.length} evidence items</span>` : ""}<code>${esc(id)}</code></div><label class="gap-select-label"><input class="gap-select" type="checkbox" data-gap-id="${esc(id)}" ${state.selectedGapIds.has(id) ? "checked" : ""}><span>Select for validation</span></label></article>`;
  }).join("")}</div>` : empty("No candidate gaps yet", "Run the gap identification stage in the pipeline. The candidate list will be loaded from the run API.", `<a class="button outline" href="#pipeline">Open pipeline →</a>`);
  const stats = state.gaps?.stats || state.gaps?.summary;
  const actions = `<div class="gap-page-actions"><button class="button outline" data-action="load-gaps" ${state.busy ? "disabled" : ""}>↻ Load candidates</button><button class="button green" id="validate-selected-gaps" data-action="go-validate-selected" ${selectedCount ? "" : "disabled"}>Validate selected (${selectedCount}) →</button></div>`;
  return `${heading("CANDIDATE GAPS", "Potential gaps, grounded in evidence.", "Review candidate statements and their rationale. Signal-based and LLM-proposed candidates are labeled separately; both remain hypotheses until validation.", actions)}${requireRun()}${state.runId ? `${stats ? `<div class="summary-strip">${Object.entries(stats).map(([k,v]) => stat(typeof v === "object" ? JSON.stringify(v) : v, k.replaceAll("_", " "))).join("")}</div>` : ""}${content}` : ""}`;
}
function validationPage() {
  const data = state.validation;
  const items = data?.results || data?.validations || data?.validation_results || data?.candidates || [];
  const list = Array.isArray(items) ? items : [];
  const stats = data?.summary || data?.validation_stats;
  const job = state.pipeline.validate || {};
  const jobPanel = job.job_id ? `<section class="job-status-panel"><div><span class="eyebrow">VALIDATION JOB</span><p>${esc(job.message || "Check the latest status returned by the API.")}</p><code>${esc(job.job_id)}</code></div>${badge(job.status)}<button class="button subtle small" data-action="check-job-status" data-stage="validate" ${state.busy ? "disabled" : ""}>Check status</button>${responseDetails("Job start response (POST)", job.startResponse, false, "validate:start")}${responseDetails("Latest job status response (GET)", job.statusResponse, job.showStatusResponse, "validate:status")}</section>` : "";
  const validateLabel = state.busy ? "Validating…" : state.selectedGapIds.size ? `Validate selected (${state.selectedGapIds.size})` : "Validate top candidates";
  const selectionNote = state.selectedGapIds.size ? `<p class="selection-note">Validating ${state.selectedGapIds.size} selected candidate gap(s).</p>` : `<p class="selection-note">No gaps selected. The API will validate the top 10 candidates.</p>`;
  return `${heading("VALIDATION STUDIO", "Test each gap against the literature.", "Validation searches external scholarly sources and records whether evidence supports or contradicts each candidate.", `<button class="button green" data-action="validate" ${state.busy || !state.runId ? "disabled" : ""}>${validateLabel}</button>`)}${requireRun()}${state.runId ? `${selectionNote}<section class="card source-card"><div class="card-body"><div><b>Validation sources</b><small>OpenAlex is the recommended default. Semantic Scholar may be rate limited.</small></div><label><input type="checkbox" checked disabled> Corpus</label><label><input type="checkbox" checked name="source" value="openalex"> OpenAlex</label><label><input type="checkbox" name="source" value="semantic_scholar"> Semantic Scholar</label><label class="field-inline">Per query <input id="per-query" type="number" min="1" max="10" value="5"></label></div></section>${jobPanel}${stats ? `<div class="summary-strip">${Object.entries(stats).map(([k,v]) => stat(typeof v === "object" ? JSON.stringify(v) : v, k.replaceAll("_", " "))).join("")}</div>` : ""}${list.length ? `<div class="validation-list">${list.map((v,i) => { const evidence = v.evidence || []; return `<article class="validation-card"><div class="section-head"><span class="eyebrow">CANDIDATE ${String(i+1).padStart(2,"0")}</span>${badge(v.classification || v.status)}</div><h3>${esc(v.gap_title || v.title || v.gap_statement || v.statement || v.gap_id || "Validated candidate")}</h3><p>${esc(v.explanation || v.rationale || v.summary || v.reason || "")}</p>${v.confidence != null ? `<div class="confidence"><span>Confidence</span><b>${esc(typeof v.confidence === "number" ? `${Math.round(v.confidence*100)}%` : v.confidence)}</b></div>` : ""}${evidence.length ? `<details><summary>${evidence.length} evidence items</summary>${evidence.map(e => `<blockquote>${esc(e.text || e.quote || e.title || e.snippet || e.summary || "")}</blockquote>`).join("")}</details>` : ""}</article>`; }).join("")}</div>` : empty("Validation results will appear here", "Run validation to assess candidate gaps against the selected literature sources.")}` : ""}`;
}
function evidencePage() {
  const vals = state.validation?.results || state.validation?.validations || state.validation?.validation_results || [];
  const nodes = Array.isArray(vals) ? vals : [];
  const graphEntities = state.graph?.top_entities || [];
  const graphEdges = state.graph?.strongest_cooccurrences || [];
  const graph = state.graph ? `<section class="graph-panel"><div class="section-head"><div><span class="eyebrow">KNOWLEDGE GRAPH</span><h2>Concept landscape</h2></div><a class="text-link" href="${API}/kg/${encodeURIComponent(state.runId)}?view=cytoscape" target="_blank" rel="noreferrer">Graph data ↗</a></div>${state.graph.stats ? `<div class="summary-strip">${Object.entries(state.graph.stats).slice(0,4).map(([k,v]) => stat(typeof v === "object" ? JSON.stringify(v) : v, k.replaceAll("_", " "))).join("")}</div>` : ""}<div class="concept-cloud">${graphEntities.map(e => `<span class="concept-pill ${esc(String(e.type || "").toLowerCase())}" title="${esc(e.papers)} papers">${esc(e.name)}<small>${esc(e.type || "")}</small></span>`).join("") || `<span class="muted">No graph entities available.</span>`}</div>${graphEdges.length ? `<div class="cooccurrences"><b>Frequent co-occurrences</b>${graphEdges.map(e => `<div><span>${esc(e.a)}</span><i>↔</i><span>${esc(e.b)}</span><small>${esc(e.papers)} papers</small></div>`).join("")}</div>` : ""}</section>` : `<section class="card graph-panel"><div class="card-body"><span class="eyebrow">KNOWLEDGE GRAPH</span><h2>Concept landscape</h2><p class="muted">Build the knowledge graph in the pipeline to map the concepts and relationships found in this run.</p><a class="button outline" href="#pipeline">Open pipeline →</a></div></section>`;
  return `${heading("EVIDENCE MAP", "Follow the claim to its sources.", "Explore the knowledge graph and connect validated claims to the evidence returned by literature search.", `<button class="button outline" data-action="load-validation">↻ Refresh evidence</button>`)}${requireRun()}${state.runId ? `${graph}${nodes.length ? `<h2 class="subheading">Validation evidence</h2><div class="evidence-flow">${nodes.map((v,i) => { const evidence = v.evidence || []; return `<section class="evidence-node"><div class="evidence-claim"><span class="eyebrow">GAP ${String(i+1).padStart(2,"0")}</span><h3>${esc(v.gap_title || v.title || v.gap_statement || v.statement || v.gap_id || "Candidate gap")}</h3>${badge(v.classification || v.status)}</div><div class="evidence-connector">${evidence.length ? "→" : "·"}</div><div class="evidence-sources">${evidence.length ? evidence.slice(0,5).map(e => `<article><small>${esc(e.source || e.provider || e.doi || e.evidence_type || "SOURCE")}</small><b>${esc(e.title || e.text || e.snippet || e.summary || "Evidence item")}</b><p>${esc(e.snippet || e.text || e.quote || e.summary || "")}</p></article>`).join("") : `<p class="muted">No linked evidence returned.</p>`}</div></section>`; }).join("")}</div>` : empty("No validation evidence yet", "Run validation to load evidence links for candidate gaps.", `<a class="button outline" href="#validation">Open validation →</a>`)}` : ""}`;
}
function reportPage() {
  const vals = state.validation?.results || state.validation?.validations || state.validation?.validation_results || [];
  const gaps = Array.isArray(state.gaps?.candidates || state.gaps?.gaps) ? (state.gaps.candidates || state.gaps.gaps) : [];
  return `${heading("RESEARCH REPORT", "A readable record of this run.", "The report is assembled from saved run data, identified gaps, and validation results. Export it as Markdown.", `<button class="button outline" data-action="export-report" ${state.runId ? "" : "disabled"}>↓ Export Markdown</button>`)}${requireRun()}${state.runId ? `<article class="report-paper"><div class="eyebrow">REFINE · RESEARCH REPORT</div><h2>${esc(state.question || "Research investigation")}</h2><p class="report-meta">Run ${esc(state.runId)} · ${new Date().toLocaleDateString()}</p><hr><h3>Investigation</h3><p>${esc(state.question || "No research question recorded.")}</p><div class="summary-strip">${stat(state.papers.length, "papers loaded")}${stat(gaps.length, "candidate gaps")}${stat(Array.isArray(vals) ? vals.length : 0, "validation results")}</div><h3>Candidate gaps</h3>${gaps.length ? `<ol>${gaps.map(g => `<li><b>${esc(g.title || g.statement || g.gap || "Candidate gap")}</b><p>${esc(g.rationale || g.description || "")}</p></li>`).join("")}</ol>` : `<p class="muted"></p>`}<h3>Validation</h3>${Array.isArray(vals) && vals.length ? `<ul>${vals.map(v => `<li>${esc(v.gap_title || v.title || v.statement || "Candidate")} — ${esc(v.classification || v.status || "unclassified")}</li>`).join("")}</ul>` : `<p class="muted"></p>`}</article>` : ""}`;
}
const views = { workspace, new: newInvestigation, pipeline, literature: papersPage, gaps: gapsPage, validation: validationPage, evidence: evidencePage, report: reportPage };
async function render() {
  document.querySelectorAll("details.response-details[data-response-key]").forEach(details => {
    state.responseDisclosureState.set(details.dataset.responseKey, details.open);
  });
  state.route = (location.hash.slice(1) || "workspace").split("?")[0]; if (!views[state.route]) state.route = "workspace";
  shell(views[state.route]()); bind();
  if (state.route === "workspace" && !state.runsLoaded) loadRuns();
  if (state.route === "literature" && state.runId && !state.papersLoaded) loadPapers();
  if ((state.route === "gaps" || state.route === "new") && state.runId && !state.gaps && (state.route !== "new" || state.fullWorkflow?.status === "COMPLETED")) loadGaps(false);
  if (["validation", "evidence", "report", "new"].includes(state.route) && state.runId && !state.validation && (state.route !== "new" || state.fullWorkflow?.status === "COMPLETED")) loadValidation(false);
  if (state.route === "evidence" && state.runId && !state.graph) loadGraph(false);
}
function bind() {
  document.querySelector("#query-form")?.addEventListener("submit", onQuery);
  document.querySelector("#load-run-form")?.addEventListener("submit", onLoadRun);
  document.querySelectorAll("[data-action]").forEach(el => el.addEventListener("click", onAction));
  document.querySelectorAll("details.response-details[data-response-key]").forEach(details => details.addEventListener("toggle", () => {
    state.responseDisclosureState.set(details.dataset.responseKey, details.open);
  }));
  document.querySelector("#llm-wording")?.addEventListener("change", event => {
    state.gapOptions.llmWording = event.currentTarget.checked;
    localStorage.setItem("refine.gapOptions", JSON.stringify(state.gapOptions));
  });
  document.querySelector("#llm-candidates")?.addEventListener("change", event => {
    state.gapOptions.llmCandidates = event.currentTarget.checked;
    localStorage.setItem("refine.gapOptions", JSON.stringify(state.gapOptions));
  });
  document.querySelectorAll(".gap-select").forEach(input => input.addEventListener("change", event => {
    const { gapId } = event.currentTarget.dataset;
    if (event.currentTarget.checked) state.selectedGapIds.add(gapId);
    else state.selectedGapIds.delete(gapId);
    const button = document.querySelector("#validate-selected-gaps");
    if (button) {
      button.disabled = state.selectedGapIds.size === 0;
      button.textContent = `Validate selected (${state.selectedGapIds.size}) →`;
    }
  }));
  document.querySelector("#paper-sort")?.addEventListener("change", event => loadPapers(state.paperSearch || "", 0, event.currentTarget.value));
}
async function onQuery(event) {
  event.preventDefault();
  const query = new FormData(event.currentTarget).get("query").trim();
  if (query) await runFullInvestigation(query);
}
function saveFullWorkflow() { localStorage.setItem("refine.fullWorkflow", JSON.stringify(state.fullWorkflow)); }
function updateFullWorkflow(changes) { state.fullWorkflow = { ...state.fullWorkflow, ...changes, updatedAt: new Date().toISOString() }; saveFullWorkflow(); }
async function runFullInvestigation(query) {
  if (state.busy) return;
  state.busy = true;
  state.question = query;
  state.queryResult = null;
  state.gaps = null;
  state.validation = null;
  state.papers = [];
  state.papersLoaded = false;
  state.pipeline = {};
  localStorage.setItem("refine.pipeline", "{}");
  activeRun("", query, "");
  state.fullWorkflow = { status: "RUNNING", current: "query", question: query, error: null, phases: { query: "RUNNING" }, expandedQueries: [] };
  saveFullWorkflow();
  render();
  try {
    const understood = await post("/query", { query });
    state.queryResult = understood;
    updateFullWorkflow({ phases: { ...state.fullWorkflow.phases, query: "COMPLETED" }, expandedQueries: understood.expanded_queries || [] });
    render();

    updateFullWorkflow({ current: "collect", phases: { ...state.fullWorkflow.phases, collect: "RUNNING" } });
    render();
    const expandedQueries = understood.expanded_queries || [];
    const collectBody = understood.saved && understood.query_id
      ? { query_id: understood.query_id, expanded_queries: expandedQueries, max_queries: Math.max(1, expandedQueries.length + 1), max_results_per_query: 10 }
      : { query, expanded_queries: expandedQueries, max_queries: Math.max(1, expandedQueries.length + 1), max_results_per_query: 10 };
    const collected = await post("/collect", collectBody);
    if (!collected.run_id) throw new Error(collected.provider_errors?.map(e => e.error).join("; ") || "Collection did not return a run ID.");
    activeRun(collected.run_id, query, understood.saved ? understood.query_id : "");
    state.papers = collected.papers || [];
    updateFullWorkflow({ current: "pipeline", phases: { ...state.fullWorkflow.phases, collect: "COMPLETED" }, runId: collected.run_id, collectionStatus: collected.status, paperCount: collected.total_results ?? state.papers.length });
    render();

    for (const stage of stages) {
      updateFullWorkflow({ current: stage, phases: { ...state.fullWorkflow.phases, [stage]: "RUNNING" } });
      render();
      await runStage(stage);
      if (state.pipeline[stage]?.status !== "COMPLETED") {
        const reason = state.pipeline[stage]?.message || `${stageNames[stage]} failed.`;
        updateFullWorkflow({ status: "FAILED", error: reason, phases: { ...state.fullWorkflow.phases, [stage]: "FAILED" } });
        throw new Error(reason);
      }
      updateFullWorkflow({ phases: { ...state.fullWorkflow.phases, [stage]: "COMPLETED" } });
    }
    updateFullWorkflow({ status: "COMPLETED", current: "done" });
    toast("Investigation completed. Validated gaps and evidence are ready below.", "success");
  } catch (error) {
    updateFullWorkflow({ status: "FAILED", error: error.message, phases: { ...state.fullWorkflow.phases, [state.fullWorkflow.current]: "FAILED" } });
    toast(`Investigation stopped: ${error.message}`, "error");
  } finally {
    state.busy = false;
    render();
  }
}
async function onLoadRun(event) {
  event.preventDefault(); const id = document.querySelector("#load-run-id")?.value.trim(); if (!id) return;
  try {
    const runs = await api("/runs?limit=100"); const match = (runs.runs || []).find(r => r.run_id === id);
    await selectRun({ runId: id, question: match?.research_question || "", queryId: match?.query_id || "", paperCount: match?.paper_count });
  } catch (e) { activeRun(id); toast(`Run selected. Could not verify it in the run list: ${e.message}`, "error"); location.hash = "#pipeline"; }
}
async function selectRun({ runId, question = "", queryId = "", paperCount = 0 }) {
  activeRun(runId, question, queryId);
  state.selectedGapIds.clear();
  state.papers = []; state.papersLoaded = false; state.gaps = null; state.validation = null; state.graph = null;
  state.pipeline = {}; localStorage.removeItem("refine.pipeline");

  // A run with stored validation is a completed investigation: restore its result view
  // instead of presenting an empty, runnable pipeline for it.
  const [gapsResult, validationResult, graphResult] = await Promise.allSettled([
    api(`/gaps/${encodeURIComponent(runId)}?view=full&top=50`),
    api(`/validate/${encodeURIComponent(runId)}`),
    api(`/kg/${encodeURIComponent(runId)}?view=summary`),
  ]);
  state.gaps = gapsResult.status === "fulfilled" ? gapsResult.value : { candidates: [] };
  state.graph = graphResult.status === "fulfilled" ? graphResult.value : null;
  const persistedStages = gapsResult.status === "fulfilled"
    ? ["chunks", "represent", "kg", "gaps"]
    : graphResult.status === "fulfilled" ? ["chunks", "represent", "kg"] : [];
  for (const stage of persistedStages) state.pipeline[stage] = { status: "COMPLETED", message: `Saved ${stageNames[stage].toLowerCase()} result is available.` };
  if (validationResult.status === "fulfilled") state.pipeline.validate = { status: "COMPLETED", message: "Saved validation result is available." };
  savePipeline();
  if (validationResult.status === "fulfilled") {
    state.validation = validationResult.value;
    const completedPhases = Object.fromEntries(["query", "collect", ...stages].map(stage => [stage, "COMPLETED"]));
    state.fullWorkflow = {
      status: "COMPLETED", current: "done", question, runId,
      paperCount: paperCount ?? 0, phases: completedPhases, expandedQueries: [],
      collectionStatus: "completed", updatedAt: new Date().toISOString(),
    };
    saveFullWorkflow();
    location.hash = "#new";
    return;
  }

  state.fullWorkflow = null; localStorage.removeItem("refine.fullWorkflow");
  location.hash = gapsResult.status === "fulfilled" ? "#gaps" : "#pipeline";
}
async function onAction(event) {
  const el = event.currentTarget; const action = el.dataset.action;
  if (action === "refresh") { await refreshHealth(); await loadRuns(true); return; }
  if (action === "select-run") return selectRun({ runId: el.dataset.run, question: el.dataset.question, queryId: el.dataset.query, paperCount: Number(el.dataset.paperCount || 0) });
  if (action === "collect") return collect();
  if (action === "run-pipeline") return runPipeline();
  if (action === "run-stage") return runStage(el.dataset.stage);
  if (action === "check-job-status") return checkJobStatus(el.dataset.stage);
  if (action === "go-validate-selected") {
    if (!state.selectedGapIds.size) return toast("Select at least one candidate gap first.", "error");
    location.hash = "#validation";
    return;
  }
  if (action === "load-gaps") return loadGaps(true);
  if (action === "validate") return validateGaps();
  if (action === "load-validation") return loadValidation(true);
  if (action === "search-papers") return loadPapers(document.querySelector("#paper-search")?.value || "");
  if (action === "papers-prev") return loadPapers(state.paperSearch || "", Math.max(0, state.paperOffset - 50));
  if (action === "papers-next") return loadPapers(state.paperSearch || "", state.paperOffset + 50);
  if (action === "export-bib") return exportBib();
  if (action === "export-report") return exportReport();
}
async function collect() {
  const selected = [...document.querySelectorAll(".query-choice input:checked")].map(i => i.value).filter(Boolean);
  if (!selected.length && !state.question) return toast("Enter a research question first.", "error");
  state.busy = true; render();
  try {
    const qr = state.queryResult || {};
    const body = qr.saved && qr.query_id ? { query_id: qr.query_id, expanded_queries: selected.length ? selected : qr.expanded_queries, max_queries: 4, max_results_per_query: 10 } : { query: state.question, expanded_queries: selected.length ? selected : undefined, max_queries: 4, max_results_per_query: 10 };
    const result = await post("/collect", body);
    activeRun(result.run_id, state.question, result.query_id || (qr.saved ? qr.query_id : ""));
    state.papers = result.papers || []; state.papersLoaded = false; state.pipeline = {}; localStorage.setItem("refine.pipeline", "{}");
    toast(`${result.total_results ?? result.papers?.length ?? 0} papers collected${result.status === "partial" ? " (partial results)" : ""}.`, result.status === "partial" ? "info" : "success");
    location.hash = "#pipeline";
  } catch (e) { toast(`Collection failed: ${e.message}`, "error"); }
  finally { state.busy = false; render(); }
}
function stageConfig(stage) {
  const configs = {
    acquire: { path: "/acquire", body: { run_id: state.runId }, status: id => `/${id}` },
    parse: { path: "/parse", body: { run_id: state.runId }, status: id => `/parse/status/${id}` },
    sections: { path: "/sections", body: { run_id: state.runId }, status: id => `/sections/status/${id}` },
    chunks: { path: "/chunks", body: { run_id: state.runId }, status: id => `/chunks/status/${id}` },
    represent: { path: "/represent", body: { run_id: state.runId }, status: id => `/represent/status/${id}` },
    kg: { path: "/kg/build", body: { run_id: state.runId, query_id: state.queryId || undefined }, status: id => `/kg/status/${id}`, result: () => `/kg/${state.runId}?view=summary` },
    gaps: { path: "/gaps", body: { run_id: state.runId, query_id: state.queryId || undefined, detectors: ["explicit", "combination", "temporal", "topics"], llm_wording: Boolean(state.gapOptions.llmWording), include_llm_candidates: Boolean(state.gapOptions.llmCandidates), include_llm_baseline: false }, status: id => `/gaps/status/${id}`, result: () => `/gaps/${state.runId}?view=full&top=50` },
    validate: { path: "/validate", body: { run_id: state.runId, gap_ids: state.selectedGapIds.size ? [...state.selectedGapIds] : null, sources: ["corpus", "openalex"], per_query: Number(document.querySelector("#per-query")?.value || 5), max_iterations: 1 }, status: id => `/validate/status/${id}`, result: () => `/validate/${state.runId}` },
  };
  return configs[stage];
}
async function runStage(stage) {
  if (!state.runId) return toast("Load or create a run first.", "error");
  const missing = (pipelinePrerequisites[stage] || []).filter(required => state.pipeline[required]?.status !== "COMPLETED");
  if (missing.length) return toast(`Run ${missing.map(required => sequenceNames[required]).join(" and ")} before ${sequenceNames[stage]}.`, "error");
  const config = stageConfig(stage); if (!config) return;
  if (stage === "gaps") state.selectedGapIds.clear();
  invalidateDownstreamStages(stage);
  state.busy = true; state.pipeline[stage] = { status: "RUNNING", message: "Starting job…" }; savePipeline(); render();
  try {
    const started = await post(config.path, config.body); const jobId = started.job_id;
    state.pipeline[stage] = { status: started.status || "RUNNING", job_id: jobId, message: started.message || "Job started.", startResponse: started }; savePipeline(); render();
    if (!jobId) throw new Error("The API did not return a job_id.");
    let latest;
    for (let attempt = 0; attempt < 720; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 2500));
      latest = await api(config.status(jobId));
      const status = String(latest.status || "RUNNING").toUpperCase();
      state.pipeline[stage] = { ...state.pipeline[stage], status, job_id: jobId, message: latest.progress || latest.message || (status === "COMPLETED" ? "Stage completed." : "Working…"), statusResponse: latest };
      state.pipeline.currentMessage = `${stageNames[stage]} · ${state.pipeline[stage].message}`; savePipeline(); render();
      if (["COMPLETED", "FAILED", "CANCELLED"].includes(status)) break;
    }
    if (!latest || !["COMPLETED", "FAILED", "CANCELLED"].includes(String(latest.status).toUpperCase())) throw new Error("Job polling timed out after 30 minutes.");
    if (String(latest.status).toUpperCase() !== "COMPLETED") throw new Error(latest.error || latest.message || `${stageNames[stage]} failed.`);
    if (stage === "chunks" && Number(latest.result_summary?.chunked || 0) < 1) {
      throw new Error("Chunking finished without creating chunks. Check section extraction and paper processing before building the knowledge graph.");
    }
    if (stage === "represent" && Number(latest.embedded_chunks || 0) < 1) {
      throw new Error("Representation finished without embedding any chunks. The knowledge graph requires represented chunks.");
    }
    state.pipeline[stage] = { ...state.pipeline[stage], status: "COMPLETED", job_id: jobId, message: latest.progress || latest.message || "Stage completed.", statusResponse: latest }; savePipeline();
    if (config.result) {
      const result = await api(config.result());
      if (stage === "gaps") state.gaps = result;
      if (stage === "validate") state.validation = result;
      state.pipeline[stage].resultResponse = result;
    }
    toast(`${stageNames[stage]} completed.`, "success");
  } catch (e) { state.pipeline[stage] = { ...(state.pipeline[stage] || {}), status: "FAILED", message: e.message }; savePipeline(); toast(`${stageNames[stage]} failed: ${e.message}`, "error"); }
  finally { state.busy = state.fullWorkflow?.status === "RUNNING"; render(); }
}
async function checkJobStatus(stage) {
  const config = stageConfig(stage);
  const existing = state.pipeline[stage] || {};
  if (!config || !existing.job_id) return toast("This stage does not have a saved job ID to check.", "error");
  if (state.busy) return;
  state.busy = true;
  existing.showStatusResponse = true;
  state.responseDisclosureState.set(`${stage}:status`, true);
  state.pipeline[stage] = existing;
  render();
  try {
    const latest = await api(config.status(existing.job_id));
    const status = String(latest.status || "UNKNOWN").toUpperCase();
    const outputMissing = (stage === "chunks" && status === "COMPLETED" && Number(latest.result_summary?.chunked || 0) < 1)
      || (stage === "represent" && status === "COMPLETED" && Number(latest.embedded_chunks || 0) < 1);
    state.pipeline[stage] = {
      ...existing,
      status: outputMissing ? "FAILED" : status,
      statusResponse: latest,
      statusCheckedAt: new Date().toISOString(),
      message: outputMissing
        ? (stage === "chunks" ? "Job completed without creating chunks." : "Job completed without embedding any chunks.")
        : (latest.progress || latest.message || latest.error || existing.message),
      showStatusResponse: true,
    };
    if (status === "COMPLETED" && !outputMissing && config.result) {
      const result = await api(config.result());
      state.pipeline[stage].resultResponse = result;
      if (stage === "gaps") state.gaps = result;
      if (stage === "validate") state.validation = result;
    }
    savePipeline();
    toast(`${stageNames[stage]} status refreshed: ${(outputMissing ? "FAILED" : status).replaceAll("_", " ")}.`, outputMissing || status === "FAILED" ? "error" : "success");
  } catch (error) {
    state.pipeline[stage] = { ...(state.pipeline[stage] || existing), statusError: error.message, showStatusResponse: true };
    toast(`Could not check ${stageNames[stage]}: ${error.message}`, "error");
  } finally {
    state.busy = false;
    savePipeline();
    render();
  }
}
function savePipeline() {
  const persisted = Object.fromEntries(Object.entries(state.pipeline).map(([stage, value]) => {
    const { resultResponse, ...rest } = value || {};
    return [stage, rest];
  }));
  localStorage.setItem("refine.pipeline", JSON.stringify(persisted));
}
async function runPipeline() {
  if (!state.runId) return toast("Load or create a run first.", "error");
  state.busy = true; render();
  try {
    for (const stage of stages) {
      if (state.pipeline[stage]?.status === "COMPLETED") continue;
      state.busy = false; await runStage(stage); if (state.pipeline[stage]?.status !== "COMPLETED") return;
      state.busy = true;
    }
    toast("All pipeline stages completed.", "success");
  } finally { state.busy = false; render(); }
}
async function loadRuns(force = false) {
  if (state.loadingRuns || (state.runsLoaded && !force)) return;
  state.loadingRuns = true;
  try { const data = await api("/runs?limit=30"); state.runs = data.runs || []; state.runsLoaded = true; if (state.route === "workspace") render(); }
  catch (e) { state.runs = []; state.runsLoaded = true; if (state.route === "workspace") { const list = document.querySelector(".run-list"); if (list) list.outerHTML = `<div class="inline-error">Could not load runs: ${esc(e.message)}</div>`; } }
  finally { state.loadingRuns = false; }
}
async function loadPapers(search = "", offset = 0, sortBy = state.paperSort) {
  if (!state.runId) return; state.paperSearch = search; state.paperOffset = offset; state.paperSort = sortBy;
  try { const q = search ? `&q=${encodeURIComponent(search)}` : ""; const data = await api(`/papers/${encodeURIComponent(state.runId)}?limit=50&offset=${offset}&sort_by=${encodeURIComponent(state.paperSort)}${q}`); state.papers = data.papers || []; state.paperTotal = data.total; state.papersLoaded = true; if (["literature", "report"].includes(state.route)) render(); }
  catch (e) { toast(`Could not load papers: ${e.message}`, "error"); }
}
async function loadGaps(notify = true) {
  if (!state.runId) return;
  try {
    state.gaps = await api(`/gaps/${encodeURIComponent(state.runId)}?view=full&top=50`);
    const availableGapIds = new Set((state.gaps.candidates || []).map(candidate => candidate.gap_id).filter(Boolean));
    state.selectedGapIds = new Set([...state.selectedGapIds].filter(gapId => availableGapIds.has(gapId)));
    for (const stage of ["chunks", "represent", "kg", "gaps"]) state.pipeline[stage] = { status: "COMPLETED", message: `Saved ${stageNames[stage].toLowerCase()} result is available.` };
    savePipeline();
    if (["gaps", "new"].includes(state.route)) render();
    if (notify) toast("Candidate gaps loaded.", "success");
  }
  catch (e) { if (notify) toast(`Could not load gaps: ${e.message}`, "error"); }
}
async function loadValidation(notify = true) {
  if (!state.runId || state.loadingValidation) return;
  state.loadingValidation = true;
  try { state.validation = await api(`/validate/${encodeURIComponent(state.runId)}`); if (["validation", "evidence", "report", "new"].includes(state.route)) render(); if (notify) toast("Validation results loaded.", "success"); }
  catch (e) { if (notify) toast(`Could not load validation results: ${e.message}`, "error"); }
  finally { state.loadingValidation = false; }
}
async function loadGraph(notify = false) {
  if (!state.runId || state.loadingGraph) return;
  state.loadingGraph = true;
  try {
    state.graph = await api(`/kg/${encodeURIComponent(state.runId)}?view=summary`);
    for (const stage of ["chunks", "represent", "kg"]) state.pipeline[stage] = { status: "COMPLETED", message: `Saved ${stageNames[stage].toLowerCase()} result is available.` };
    savePipeline();
    if (state.route === "evidence") render();
    if (notify) toast("Knowledge graph summary loaded.", "success");
  }
  catch (e) { if (notify) toast(`Could not load graph: ${e.message}`, "error"); }
  finally { state.loadingGraph = false; }
}
async function validateGaps() {
  if (!state.runId) return;
  if (state.pipeline.gaps?.status !== "COMPLETED") return toast("Identify candidate gaps before running validation.", "error");
  const sources = ["corpus", ...[...document.querySelectorAll('input[name="source"]:checked')].map(i => i.value)];
  const cfg = stageConfig("validate"); cfg.body.sources = [...new Set(sources)]; cfg.body.per_query = Math.min(10, Math.max(1, Number(document.querySelector("#per-query")?.value || 5)));
  // Keep the same job polling and persistence behavior as pipeline validation.
  state.busy = true; render();
  try {
    const started = await post(cfg.path, cfg.body); if (!started.job_id) throw new Error("The API did not return a job_id.");
    state.pipeline.validate = { status: started.status || "RUNNING", job_id: started.job_id, message: started.message || "Validation job started.", startResponse: started };
    savePipeline(); render();
    let latest;
    for (let i = 0; i < 720; i++) {
      await new Promise(r => setTimeout(r, 2500));
      latest = await api(cfg.status(started.job_id));
      const status = String(latest.status || "RUNNING").toUpperCase();
      state.pipeline.validate = { ...state.pipeline.validate, status, statusResponse: latest, message: latest.progress || latest.message || "Working…" };
      savePipeline(); render();
      if (["COMPLETED", "FAILED", "CANCELLED"].includes(status)) break;
    }
    if (String(latest?.status).toUpperCase() !== "COMPLETED") throw new Error(latest?.error || "Validation did not complete.");
    state.validation = await api(cfg.result());
    state.pipeline.validate.resultResponse = state.validation;
    savePipeline();
    toast("Gap validation completed.", "success");
  } catch (e) {
    if (state.pipeline.validate) { state.pipeline.validate = { ...state.pipeline.validate, status: "FAILED", statusError: e.message }; savePipeline(); }
    toast(`Validation failed: ${e.message}`, "error");
  }
  finally { state.busy = false; render(); }
}
function download(name, content, type = "text/plain") { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([content], { type })); a.download = name; a.click(); URL.revokeObjectURL(a.href); }
function exportBib() {
  if (!state.papers.length) return toast("No papers to export.", "error");
  const bib = state.papers.map((p, i) => { const key = (p.doi || p.paper_id || `paper${i}`).replace(/[^a-zA-Z0-9]/g, "").slice(-24); const authors = Array.isArray(p.authors) ? p.authors.map(a => typeof a === "string" ? a : a.name || "").filter(Boolean).join(" and ") : ""; return `@article{${key || `paper${i}`},\n  title = {${String(p.title || "").replace(/[{}]/g, "")}},\n  author = {${authors}},\n  year = {${p.publication_year || ""}},\n  doi = {${p.doi || ""}}\n}`; }).join("\n\n");
  download("refine-literature.bib", bib, "application/x-bibtex");
}
function exportReport() {
  const gaps = Array.isArray(state.gaps?.candidates || state.gaps?.gaps) ? (state.gaps.candidates || state.gaps.gaps) : [];
  const vals = state.validation?.results || state.validation?.validations || state.validation?.validation_results || [];
  const text = [`# ${state.question || "Research investigation"}`, "", `Run ID: ${state.runId}`, `Generated: ${new Date().toISOString()}`, "", "## Summary", `- Papers loaded: ${state.papers.length}`, `- Candidate gaps: ${gaps.length}`, `- Validation results: ${Array.isArray(vals) ? vals.length : 0}`, "", "## Candidate gaps", ...(gaps.length ? gaps.flatMap(g => [`### ${g.gap_statement || g.title || g.statement || g.gap || "Candidate gap"}`, g.rationale || g.description || g.explanation || "", ""]) : [""]), "## Validation", ...(Array.isArray(vals) && vals.length ? vals.map(v => `- **${v.gap_title || v.title || v.statement || v.gap_id || "Candidate"}** — ${v.classification || v.status || "unclassified"}`) : [""])].join("\n");
  download(`refine-report-${state.runId.slice(0,8)}.md`, text, "text/markdown");
}
async function refreshHealth() { try { const response = await fetch("/api/v1/health"); state.health = response.ok ? await response.json() : null; } catch { state.health = null; } }
window.addEventListener("hashchange", render);
await refreshHealth();
await render();

