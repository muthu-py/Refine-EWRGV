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

The responsive interface includes the workspace, query expansion, literature collection, stage-by-stage pipeline, literature explorer, candidate gaps, validation, evidence map, and Markdown report. The pipeline follows the backend stages in order: acquisition, parsing, section extraction, chunking, representation, knowledge-graph construction, gap detection, and validation. Each stage can also be started separately. Collection is deliberately separate from the remaining pipeline so users can inspect partial collection results before starting processing.

The UI stores the selected run and question in browser local storage. Background jobs are polled from the corresponding API status routes. Job state itself lives in the backend process, so a server restart may require starting an interrupted stage again. Validation defaults to the local corpus and OpenAlex; Semantic Scholar is an optional source because it can be rate limited.

## Start it

From the `ewrgv` project directory, start the API with `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`, then open `http://localhost:8000/`. The API remains available through `/docs` and `/redoc`.

The app is dependency-free on the frontend and uses system fonts, so the interface does not need an external font service. It requires the API's existing database and provider configuration for live research data.
