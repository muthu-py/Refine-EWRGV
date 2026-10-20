# Refine UI design notes

The supplied reference screens are the eight PNGs inside `../banani-ui-export.zip` (Research Workspace, New Investigation, Run Pipeline, Literature Explorer, Candidate Gaps, Validation Studio, Evidence Map, Research Report). The `UI-Design` folder was otherwise empty.

## Visual language

- Warm ivory page background, white cards, thin sand-colored borders, forest-green primary actions.
- Serif display headings paired with a clean sans-serif interface font.
- Persistent left navigation; a compact run context bar on research-specific views.
- Status colors are semantic: green completed/supporting, blue running, amber partial/uncertain, red failed/contradicting.
- Dense research data stays readable through card hierarchy, visible provenance, and explicit limitations.

## Integration notes

The implementation lives in `app/ui/` and is served from FastAPI at `/`. It uses the same-origin `/api/v1` endpoints. Screens that need papers and recent runs use the `GET /research/papers/{run_id}` and `GET /research/runs` endpoints added for the interface. Reports are assembled and exported in the browser from persisted API results; no mock study data is presented as live backend data.

## Screens and workflow

The responsive interface includes the workspace, query expansion, literature collection, stage-by-stage pipeline, literature explorer, candidate gaps, validation, evidence map, and Markdown report. The Pipeline screen shows the full ordered sequence: query understanding → literature collection → acquisition → parsing → section extraction → chunking → representation → knowledge-graph construction → signal-first gap identification (with opt-in LLM proposals and wording refinement) → validation. Query and collection appear as completed context for an already-created run; the Full Investigation screen runs them before processing. Processing jobs are individually startable only after their prerequisites complete. KG construction additionally requires chunked and represented corpus data, and a rerun of an earlier stage clears downstream completion state. Collection can still be reviewed before processing from the separate Full Investigation flow.

The UI stores the selected run, question, and opt-in gap options in browser local storage. Candidate cards identify signal/LLM provenance and allow selecting candidates for validation; default validation includes up to ten candidates, balanced across sources when LLM proposals exist. Background jobs are polled from the corresponding API status routes. Job state itself lives in the backend process, so a server restart may require starting an interrupted stage again. Validation defaults to the local corpus and OpenAlex; Semantic Scholar is an optional source because it can be rate limited.

Every API request also updates a collapsible **Latest API response** panel with
the HTTP method, endpoint, status code, and indented JSON response. Pipeline
stages with a saved job ID expose a **Check status** action that calls that
stage's GET status route and shows the current response on the stage card. The
disclosure preserves its open/closed state across UI rerenders, including while
a background job is being polled. The Validation screen includes the same action
for validation jobs. These response
views use escaped, read-only JSON so request output remains inspectable without
replacing the task-focused result cards.

## Start it

From the `ewrgv` project directory, start the API with `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`, then open `http://localhost:8000/`. The API remains available through `/docs` and `/redoc`.

The app is dependency-free on the frontend and uses system fonts, so the interface does not need an external font service. It requires the API's existing database and provider configuration for live research data.
