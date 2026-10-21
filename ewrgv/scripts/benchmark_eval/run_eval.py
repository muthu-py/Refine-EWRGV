"""Run read-only retrieval benchmarks with the repository's production components."""
from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import numpy as np

from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.retrieval_result import RetrievalResult
from app.providers.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddingProvider
from app.retrieval.bm25_index import BM25Index
from app.retrieval.reranking import CrossEncoderReranker
from app.retrieval.fusion import reciprocal_rank_fusion_results
from scripts.benchmark_eval.datasets import read_trec_covid, read_scifact_dev
from scripts.benchmark_eval.metrics import evaluate_ranking, aggregate

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "benchmark"
OUT = ROOT / "data" / "processed" / "benchmark_eval"
TOP_K = 100
RERANK_K = 50
EMBED_BATCH = 256


def chunks_for(docs):
    return [PaperChunk(chunk_id=d.doc_id, paper_id=d.doc_id, section_id=d.doc_id,
                       section_title=d.title or "abstract", text=d.text) for d in docs]


def result_for(chunk, score, rank, method):
    doc_id = getattr(chunk, "chunk_id", getattr(chunk, "doc_id", None))
    paper_id = getattr(chunk, "paper_id", doc_id)
    title = getattr(chunk, "section_title", getattr(chunk, "title", ""))
    return RetrievalResult(chunk_id=doc_id, paper_id=paper_id,
                           section=title, text=chunk.text, score=float(score),
                           rank=rank, retrieval_method=method,
                           metadata={"title": title})


class Store:
    def __init__(self, chunks):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
    def get_all_chunks(self): return self.chunks
    def get_chunk_by_id(self, key): return self.by_id.get(key)
    def get_paper_by_id(self, key): return None


def dense_rank(docs, cases, provider, prefix):
    # Stored production embeddings are unit-normalized. A float32 memmap lets
    # the official 191k-ID corpus run without DenseRetriever's Python-list RAM use.
    vec_path = OUT / f"{prefix}_embeddings.f32"
    dim = provider.get_dimension()
    expected_bytes = len(docs) * dim * np.dtype("float32").itemsize
    if not vec_path.exists() or vec_path.stat().st_size != expected_bytes:
        matrix = np.memmap(vec_path, mode="w+", dtype="float32", shape=(len(docs), dim))
        for start in range(0, len(docs), EMBED_BATCH):
            matrix[start:start+EMBED_BATCH] = np.asarray(
                provider.embed_batch([d.text for d in docs[start:start+EMBED_BATCH]]), dtype="float32")
        matrix.flush()
    matrix = np.memmap(vec_path, mode="r", dtype="float32", shape=(len(docs), dim))
    ids = [d.doc_id for d in docs]
    for case in cases:
        q = np.asarray(provider.embed(case.query), dtype="float32")
        scores = np.einsum("ij,j->i", matrix, q, optimize=False)
        ix = np.argpartition(scores, -min(TOP_K, len(scores)))[-min(TOP_K, len(scores)):]
        ix = sorted(ix.tolist(), key=lambda i: (-float(scores[i]), ids[i]))
        yield case.query_id, [result_for(docs[i], scores[i], rank+1, "dense") for rank, i in enumerate(ix)]
    del matrix


def sparse_rank(docs, cases):
    chunks = chunks_for(docs)
    index = BM25Index().build(chunks)
    lookup = {c.chunk_id: c for c in chunks}
    for case in cases:
        hits = index.query(case.query, TOP_K)
        hits.sort(key=lambda x: (-x[1], x[0]))
        yield case.query_id, [result_for(lookup[key], score, rank+1, "sparse")
                              for rank, (key, score) in enumerate(hits)]


def save_dataset(name, docs, cases, sparse, dense, meta, reranker, skip_rerank=False):
    by_q = {c.query_id: c for c in cases}
    all_runs = {"bm25": sparse}
    if dense:
        all_runs["dense"] = dense
    if dense:
        hybrid = {}
        for qid, case in by_q.items():
            hybrid[qid] = reciprocal_rank_fusion_results(
                [dense.get(qid, []), sparse.get(qid, [])], top_k=TOP_K, k=60)
        all_runs["hybrid_rrf"] = hybrid
    raw = {"dataset": name, "metadata": meta, "runs": {}, "metrics": {}}
    for method, run in all_runs.items():
        raw["runs"][method] = {}
        if not skip_rerank:
            raw["runs"][method + "_reranked"] = {}
        pre_rows, post_rows = [], []
        for qid, case in by_q.items():
            results = run.get(qid, [])
            raw["runs"][method][qid] = [r.chunk_id for r in results]
            pre_rows.append(evaluate_ranking([r.chunk_id for r in results], case.qrels))
            if not skip_rerank:
                reranked, _ms = reranker.rerank_results(case.query, results[:RERANK_K], RERANK_K)
                reranked += results[RERANK_K:]
                raw["runs"][method + "_reranked"][qid] = [r.chunk_id for r in reranked]
                post_rows.append(evaluate_ranking([r.chunk_id for r in reranked], case.qrels))
        raw["metrics"][method] = {"aggregate": aggregate(pre_rows), "per_query": dict(zip(by_q, pre_rows))}
        if not skip_rerank:
            raw["metrics"][method + "_reranked"] = {"aggregate": aggregate(post_rows), "per_query": dict(zip(by_q, post_rows))}
    raw["queries"] = {qid: {"query": c.query, "qrels": c.qrels, "label": c.claim_label}
                       for qid, c in by_q.items()}
    path = OUT / f"{name}_results.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"dataset": name, "metadata": meta, "metrics": {
        method: value["aggregate"] for method, value in raw["metrics"].items()}, "path": str(path)}, ensure_ascii=False))


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    provider = SentenceTransformerEmbeddingProvider(model="BAAI/bge-small-en-v1.5", dimension=384, batch_size=EMBED_BATCH)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", batch_size=16)
    datasets = [
        ("trec_covid_round5", read_trec_covid, RAW / "trec_covid"),
        ("scifact_dev", read_scifact_dev, RAW / "scifact" / "extracted" / "data"),
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["trec_covid_round5", "scifact_dev", "all"], default="all")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--sparse-only", action="store_true")
    args = parser.parse_args()
    selected = args.dataset
    if selected != "all":
        datasets = [entry for entry in datasets if entry[0] == selected]
    for name, reader, directory in datasets:
        docs, cases, meta = reader(directory)
        print(json.dumps({"loading": name, "metadata": meta}), flush=True)
        sparse = dict(sparse_rank(docs, cases))
        dense = {} if args.sparse_only else dict(dense_rank(docs, cases, provider, name))
        save_dataset(name, docs, cases, sparse, dense, meta, reranker, skip_rerank=args.skip_rerank)


if __name__ == "__main__":
    main()
