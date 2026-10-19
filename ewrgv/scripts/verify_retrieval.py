"""
End-to-end retrieval verification script.

Demonstrates the complete retrieval pipeline:
    query → dense retrieval → sparse retrieval → KG retrieval → hybrid fusion

Prints results from each retrieval method and the final fused output.
"""

import sys
import os

# Ensure the ewrgv package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retrieval.semantic import DenseRetriever
from app.retrieval.bm25 import SparseRetriever
from app.retrieval.knowledge_graph import KnowledgeGraphRetriever
from app.retrieval.fusion import HybridRetriever
from app.stubs.stub_corpus import StubCorpusStore
from app.stubs.stub_text_representation import StubEmbeddingProvider
from app.stubs.stub_knowledge_graph import StubKnowledgeGraphStore


def main():
    query = "How do Transformer models use attention mechanisms for NLP tasks?"
    top_k = 5
    
    print("=" * 80)
    print("EWRGV Phase 2.5 — End-to-End Retrieval Verification")
    print("=" * 80)
    print(f"\nQuery: {query}")
    print(f"Top-K: {top_k}")

    # --- Setup ---
    corpus = StubCorpusStore()
    embedder = StubEmbeddingProvider()
    kg = StubKnowledgeGraphStore()

    dense = DenseRetriever(embedding_provider=embedder, corpus_store=corpus)
    sparse = SparseRetriever(corpus_store=corpus)
    kg_retriever = KnowledgeGraphRetriever(kg_store=kg, corpus_store=corpus)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        kg_retriever=kg_retriever,
        top_k=top_k,
    )

    # --- Dense Retrieval ---
    print("\n" + "-" * 80)
    print("1. DENSE RETRIEVAL (Cosine Similarity)")
    print("-" * 80)
    dense_results = dense.retrieve_results(query, top_k=top_k)
    for r in dense_results:
        print(f"  Rank {r.rank}: [{r.chunk_id}] score={r.score:.4f}")
        print(f"    Paper: {r.paper_id} | Section: {r.section}")
        print(f"    Text: {r.text[:80]}...")

    # --- Sparse Retrieval ---
    print("\n" + "-" * 80)
    print("2. SPARSE RETRIEVAL (BM25)")
    print("-" * 80)
    sparse_results = sparse.retrieve_results(query, top_k=top_k)
    for r in sparse_results:
        print(f"  Rank {r.rank}: [{r.chunk_id}] score={r.score:.4f}")
        print(f"    Paper: {r.paper_id} | Section: {r.section}")
        print(f"    Text: {r.text[:80]}...")

    # --- KG Retrieval ---
    print("\n" + "-" * 80)
    print("3. KNOWLEDGE GRAPH RETRIEVAL")
    print("-" * 80)
    kg_results = kg_retriever.retrieve_results(query, top_k=top_k)
    for r in kg_results:
        entities = r.metadata.get("matched_entities", [])
        relations = r.metadata.get("relations", [])
        print(f"  Rank {r.rank}: [{r.chunk_id}] score={r.score:.4f}")
        print(f"    Paper: {r.paper_id} | Section: {r.section}")
        if entities:
            print(f"    Entities: {entities}")
        if relations:
            print(f"    Relations: {relations[:2]}...")
        print(f"    Text: {r.text[:80]}...")

    # --- Hybrid Fusion ---
    print("\n" + "-" * 80)
    print("4. HYBRID FUSION (Reciprocal Rank Fusion)")
    print("-" * 80)
    hybrid_results = hybrid.retrieve(query, top_k=top_k)
    for r in hybrid_results:
        methods = r.metadata.get("contributing_methods", [])
        method_names = [m["method"] for m in methods]
        print(f"  Rank {r.rank}: [{r.chunk_id}] RRF_score={r.score:.6f}")
        print(f"    Paper: {r.paper_id} | Section: {r.section}")
        print(f"    Contributing methods: {method_names}")
        print(f"    Text: {r.text[:80]}...")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Dense results:  {len(dense_results)}")
    print(f"  Sparse results: {len(sparse_results)}")
    print(f"  KG results:     {len(kg_results)}")
    print(f"  Hybrid results: {len(hybrid_results)}")
    
    # Verify no duplicates in hybrid
    hybrid_ids = [r.chunk_id for r in hybrid_results]
    assert len(hybrid_ids) == len(set(hybrid_ids)), "FAIL: duplicates in hybrid results!"
    print("  Duplicates in hybrid: NONE ✓")
    
    # Verify determinism
    hybrid_results2 = hybrid.retrieve(query, top_k=top_k)
    ids1 = [r.chunk_id for r in hybrid_results]
    ids2 = [r.chunk_id for r in hybrid_results2]
    assert ids1 == ids2, "FAIL: non-deterministic results!"
    print("  Deterministic: YES ✓")
    
    print("\n✓ End-to-end retrieval verification PASSED")


if __name__ == "__main__":
    main()
