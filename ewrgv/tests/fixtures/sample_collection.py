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
from app.domain.enums import SectionType
from app.domain.models.paper import Author, Paper
from app.domain.models.paper_chunk import PaperChunk

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

# Chunk IDs of the processed run corpus (make_run_corpus)
CAN_GNN_CHUNK = "can-gnn-limitations"
CAN_GNN_RESULTS_CHUNK = "can-gnn-results"
TRANSFER_CHUNK = "transfer-future-work"
GRID_CHUNK = "grid-methods"


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


def _pc(chunk_id, paper_id, title, section_type, text, page) -> PaperChunk:
    return PaperChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section_id=f"{paper_id}-{section_type.value.lower()}",
        section_title=title,
        section_type=section_type,
        text=text,
        page_start=page,
        page_end=page,
        character_count=len(text),
    )


def make_run_chunks() -> list[PaperChunk]:
    """Full-text PaperChunks as /chunks would produce for the deduplicated papers."""
    return [
        _pc(CAN_GNN_CHUNK, "s2-can-gnn", "5 Limitations", SectionType.LIMITATIONS,
            "Our graph neural network for CAN bus intrusion detection is evaluated on a "
            "single public dataset from one vehicle model; real-time embedded hardware "
            "evaluation is left for future work.", 7),
        _pc(CAN_GNN_RESULTS_CHUNK, "s2-can-gnn", "4 Results", SectionType.RESULTS,
            "The graph neural network detects CAN bus injection attacks with 99% accuracy.", 5),
        _pc(TRANSFER_CHUNK, "oa-transfer", "6 Future Work", SectionType.FUTURE_WORK,
            "Transfer learning for in-vehicle network anomaly detection should be evaluated "
            "on real traffic from several vehicle manufacturers, not simulated traffic.", 9),
        _pc(GRID_CHUNK, "s2-grid", "3 Methods", SectionType.METHODS,
            "We forecast electricity load with federated learning across utilities.", 3),
    ]


def make_run_papers() -> list[Paper]:
    """The papers of the run (deduplicated representatives) as stored in ``papers``."""
    keep = {"s2-can-gnn", "oa-transfer", "s2-grid"}
    return [cp.paper for cp in make_collection_result().papers if cp.paper.paper_id in keep]
