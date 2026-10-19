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
