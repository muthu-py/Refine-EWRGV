"""
tests/fixtures/sample_collection.py
-------------------------------------
Hand-created CollectionResult for offline corpus / pipeline tests.

All papers, DOIs and abstracts are FICTIONAL test data, written to exercise:
* a cross-provider duplicate matched by DOI (URL/case variants),
* a duplicate without DOI matched by title + authors + year,
* an off-topic paper,
* missing and whitespace-only abstracts.

Expected corpus: 7 records → 5 unique papers → 3 chunks
(2 skipped for missing abstracts, 2 duplicates merged).
"""

from __future__ import annotations

from app.domain.models.collection import CollectedPaper, CollectionResult, PaperProvenance
from app.domain.models.paper import Author, Paper

QUERY_ID = "run-test-001"
Q_MAIN = "graph neural network CAN bus intrusion detection"
Q_EXPANDED = "in-vehicle network anomaly detection transfer learning"

CAN_GNN_ABSTRACT = (
    "We propose a graph neural network that models CAN bus message sequences as "
    "graphs to detect injection attacks. Experiments use a single public dataset "
    "recorded from one vehicle model. Evaluation on real-time embedded hardware "
    "is left for future work."
)
TRANSFER_ABSTRACT = (
    "Deep anomaly detectors for in-vehicle networks rarely transfer across vehicle "
    "manufacturers. We study transfer learning for CAN intrusion detection but only "
    "evaluate on simulated traffic."
)
GRID_ABSTRACT = "We forecast electricity load with federated learning across utilities."

# Expected chunk IDs (representative paper_id + ":abstract")
CAN_GNN_CHUNK = "s2-can-gnn:abstract"
TRANSFER_CHUNK = "oa-transfer:abstract"
GRID_CHUNK = "s2-grid:abstract"


def _cp(paper_id, title, abstract, source, provider_id, query, doi=None, year=2023,
        authors=("A. Researcher",)) -> CollectedPaper:
    return CollectedPaper(
        paper=Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            doi=doi,
            publication_year=year,
            authors=[Author(name=a) for a in authors],
            source=source,
            paper_url=f"https://example.org/{paper_id}",
        ),
        provenance=PaperProvenance(source=source, provider_id=provider_id, source_query=query),
    )


def make_collection_result() -> CollectionResult:
    return CollectionResult(
        query_id=QUERY_ID,
        queries_searched=[Q_MAIN, Q_EXPANDED],
        providers_used=["semantic_scholar", "openalex"],
        papers=[
            _cp("s2-can-gnn", "Graph Neural Networks for CAN Bus Intrusion Detection",
                CAN_GNN_ABSTRACT, "semantic_scholar", "S2-111", Q_MAIN,
                doi="10.1000/can.gnn.2023"),
            # Same paper via OpenAlex: DOI as URL + different case, shorter abstract.
            _cp("oa-can-gnn", "Graph neural networks for CAN bus intrusion detection",
                "Graph neural networks detect CAN bus attacks.", "openalex", "W222",
                Q_EXPANDED, doi="https://doi.org/10.1000/CAN.GNN.2023"),
            _cp("oa-transfer", "Transfer Learning for In-Vehicle Network Anomaly Detection",
                TRANSFER_ABSTRACT, "openalex", "W333", Q_EXPANDED, doi=None, year=2022,
                authors=("A. Kumar",)),
            # Same paper via Semantic Scholar, no DOI, empty abstract → merged by title key.
            _cp("s2-transfer", "Transfer learning for in-vehicle network anomaly detection.",
                "", "semantic_scholar", "S2-444", Q_MAIN, doi=None, year=2022,
                authors=("A. Kumar",)),
            _cp("s2-grid", "Federated Learning for Smart Grid Forecasting", GRID_ABSTRACT,
                "semantic_scholar", "S2-555", Q_EXPANDED, doi="10.1000/grid.fl"),
            _cp("s2-survey", "A Survey of Automotive Cybersecurity", "", "semantic_scholar",
                "S2-666", Q_MAIN, doi="10.1000/auto.survey"),
            _cp("s2-blank", "Notes on Vehicle Networks", "   \n\t ", "semantic_scholar",
                "S2-777", Q_MAIN, doi="10.1000/blank"),
        ],
        status="success",
    )
