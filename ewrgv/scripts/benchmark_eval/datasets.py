"""Read-only adapters for official TREC-COVID Round 5 and SciFact dev data."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkDocument:
    doc_id: str
    title: str
    text: str


@dataclass
class QueryCase:
    query_id: str
    query: str
    qrels: dict[str, int]
    claim_label: str | None = None


def read_trec_covid(data_dir: Path) -> tuple[list[BenchmarkDocument], list[QueryCase], dict]:
    """Load Round 5 valid IDs, NIST topics/qrels, and matching CORD-19 abstracts."""
    valid_ids = [line.strip() for line in (data_dir / "docids-rnd5.txt").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if len(valid_ids) != len(set(valid_ids)):
        raise ValueError("NIST Round 5 valid-ID file contains duplicate IDs")
    valid = set(valid_ids)

    topic_root = ET.parse(data_dir / "topics-rnd5.xml").getroot()
    topics: dict[str, str] = {}
    for topic in topic_root.findall("topic"):
        query = topic.findtext("query")
        if query and query.strip():
            topics[topic.attrib["number"]] = query.strip()

    qrels: dict[str, dict[str, int]] = {}
    for line in (data_dir / "qrels-covid_d5_j0.5-5.txt").read_text(encoding="utf-8").splitlines():
        cols = line.split()
        if len(cols) != 4:
            raise ValueError(f"Malformed TREC qrels line: {line[:120]!r}")
        qid, _iteration, doc_id, grade = cols
        if qid in topics:
            qrels.setdefault(qid, {})[doc_id] = int(grade)

    # CORD-19 permits repeated cord_uid rows from different sources. Choose the
    # metadata row with the most usable title/abstract text and preserve cord_uid.
    selected: dict[str, tuple[int, str, str]] = {}
    with (data_dir / "metadata-2020-07-16.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            doc_id = (row.get("cord_uid") or "").strip()
            if doc_id not in valid:
                continue
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            quality = bool(title) + bool(abstract)
            old = selected.get(doc_id)
            if old is None or quality > old[0]:
                selected[doc_id] = (quality, title, abstract)

    documents = [BenchmarkDocument(doc_id, title, "\n".join(x for x in (title, abstract) if x))
                 for doc_id, (_quality, title, abstract) in selected.items()
                 if title or abstract]
    cases = [QueryCase(qid, query, qrels.get(qid, {})) for qid, query in topics.items()]
    meta = {
        "release": "CORD-19 2020-07-16 / TREC-COVID Round 5",
        "valid_doc_ids": len(valid_ids),
        "metadata_ids_matched": len(selected),
        "documents_with_text": len(documents),
        "valid_ids_missing_metadata": len(valid - set(selected)),
        "valid_ids_without_text": len(set(selected) - {d.doc_id for d in documents}),
        "topics": len(cases),
        "qrels": sum(len(c.qrels) for c in cases),
        "qrels_docids_missing_from_valid_release": len({doc for c in cases for doc in c.qrels} - valid),
    }
    return documents, cases, meta


def read_scifact_dev(data_dir: Path) -> tuple[list[BenchmarkDocument], list[QueryCase], dict]:
    """Load the official SciFact abstract corpus and labeled dev claims only."""
    documents = []
    doc_ids: set[str] = set()
    with (data_dir / "corpus.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            if doc_id in doc_ids:
                raise ValueError(f"Duplicate SciFact corpus doc_id: {doc_id}")
            doc_ids.add(doc_id)
            sentences = row.get("abstract") or []
            abstract = " ".join(str(sentence) for sentence in sentences)
            text = "\n".join(part for part in ((row.get("title") or "").strip(), abstract.strip()) if part)
            documents.append(BenchmarkDocument(doc_id, row.get("title") or "", text))

    cases: list[QueryCase] = []
    claims_total = 0
    claims_without_evidence = 0
    with (data_dir / "claims_dev.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            claims_total += 1
            evidence = row.get("evidence") or {}
            relevant = {str(doc_id): 1 for doc_id, rationales in evidence.items() if rationales}
            if not relevant:
                claims_without_evidence += 1
                continue
            labels = {r["label"] for rationales in evidence.values() for r in rationales}
            label = next(iter(labels)) if len(labels) == 1 else "MIXED"
            cases.append(QueryCase(str(row["id"]), row["claim"], relevant, label))
    return documents, cases, {
        "release": "allenai/scifact official release; claims_dev only",
        "corpus_documents": len(documents),
        "dev_claims_total": claims_total,
        "dev_claims_with_evidence": len(cases),
        "dev_claims_without_evidence_excluded_from_evidence_retrieval": claims_without_evidence,
        "claim_label_distribution_evidence_annotated": {
            label: sum(case.claim_label == label for case in cases)
            for label in sorted({case.claim_label for case in cases})
        },
        "evidence_document_id_coverage": len({doc for case in cases for doc in case.qrels} & doc_ids),
        "evidence_document_ids_missing_from_corpus": len({doc for case in cases for doc in case.qrels} - doc_ids),
    }
