"""
app/explanation/evidence_chain.py
----------------------------------
Evidence chain builder: the step-by-step trace from the candidate gap,
through each validation search, evidence classification, coverage and
confidence, to the final classification.  Every number in the chain is the
number used by the scorer, so the chain is the audit trail (RQ4).
"""

from __future__ import annotations

from app.domain.models.validation import EvidenceChainEntry


def _cite(it: dict) -> str:
    year = f", {it['year']}" if it.get("year") else ""
    where = f", {it['section_type']} p.{it['pages']}" if it.get("source") == "corpus" and it.get("pages") else ""
    doi = f", doi:{it['doi']}" if it.get("doi") else ""
    return f"\"{(it.get('title') or '')[:90]}\" ({it['source']}{year}{where}{doi})"


def build_evidence_chain(gap: dict, queries: dict[str, list[str]], items: list[dict], coverage: dict,
                         external: dict | None, scoring: dict, classification: str, reason: str,
                         iterations: int) -> list[EvidenceChainEntry]:
    chain: list[EvidenceChainEntry] = []

    def add(label: str, summary: str, ids: list[str] | None = None) -> None:
        chain.append(EvidenceChainEntry(step=len(chain) + 1, label=label, summary=summary, evidence_ids=ids or []))

    notes = gap.get("score_notes", {})
    add("Candidate", f"\"{gap['gap_statement']}\" - {gap['gap_type']} gap from {', '.join(gap['detectors'])}; "
        f"GapScore {gap['gap_score']:.2f}; stated by {notes.get('stating_papers', 0)} run paper(s).")
    for stype, qs in queries.items():
        if not qs:
            continue
        found = [it for it in items if any(f["search_type"] == stype for f in it.get("found_by", []))]
        relevant = [it for it in found if it["evidence_type"] != "Insufficient"]
        add(f"{stype.replace('_', ' ').title()} search",
            f"{len(qs)} quer{'y' if len(qs) == 1 else 'ies'} ({'; '.join(qs[:3])}) found {len(found)} items, {len(relevant)} relevant.",
            [str(it.get("id")) for it in relevant[:10]])
    for etype in ("Counter", "Contradictory", "Partial", "Supporting"):
        group = [it for it in items if it["evidence_type"] == etype]
        if group:
            top = "; ".join(f"{_cite(it)} - {it['rationale']}" for it in group[:3])
            add(f"{etype} evidence", f"{len(group)} item(s), strength {sum(i['strength'] for i in group):.2f}. {top}",
                [str(it.get("id")) for it in group])
    cov_txt = ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in coverage.items()
                        if isinstance(v, (int, float)) and v is not None)
    if external and external.get("available"):
        cov_txt += (f"; external coverage {external['coverage']:.3f} = {external['joint']} papers with all of "
                    f"{', '.join(external['terms'])} / min single-concept count {min(external['counts'].values())}"
                    + (f" (within '{external['field']}')" if external.get("field") else ""))
    add("Coverage", cov_txt or "not computed")
    comp = scoring["components"]
    caps = f" Caps: {'; '.join(comp['caps_applied'])}." if comp["caps_applied"] else ""
    add("Confidence", f"support {comp['support']:.2f} (2 x GapScore + supporting) vs counter {comp['counter']:.2f}; "
        f"base {comp['base']:.2f}; external-coverage penalty {comp['coverage_penalty']:.2f}; "
        f"confidence {scoring['confidence']:.2f} after {iterations} iteration(s).{caps}")
    add("Classification", f"{classification} ({reason}).")
    return chain
