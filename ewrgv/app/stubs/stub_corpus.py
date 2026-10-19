"""
app/stubs/stub_corpus.py
--------------------------
Deterministic in-memory corpus store for development and testing.

.. warning::
    This is a **development stub**.  It will be replaced by a real
    database-backed CorpusStore implementation in a future phase.

Provides a small mock corpus of 5 academic papers in the ML/NLP domain,
each with 2–4 sections and 2–4 document chunks.  All IDs and text are
fixed so that tests are fully deterministic.

Satisfies the CorpusStore protocol (app.domain.interfaces).
"""

from __future__ import annotations

from typing import Optional

from app.domain.models.paper import Author, DocumentChunk, Paper


# ======================================================================
# Fixed paper data
# ======================================================================

_PAPERS: list[Paper] = [
    Paper(
        paper_id="paper-001",
        title="Attention Is All You Need",
        authors=[
            Author(name="Vaswani, A.", affiliation="Google Brain"),
            Author(name="Shazeer, N.", affiliation="Google Brain"),
        ],
        abstract=(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. We propose a new simple "
            "network architecture, the Transformer, based solely on attention "
            "mechanisms, dispensing with recurrence and convolutions entirely."
        ),
        source="stub_corpus",
        venue="NeurIPS 2017",
        publication_year=2017,
        sections={
            "Introduction": (
                "Recurrent neural networks, long short-term memory and gated "
                "recurrent neural networks have been firmly established as state "
                "of the art approaches in sequence modeling and transduction "
                "problems such as language modeling and machine translation."
            ),
            "Model Architecture": (
                "The Transformer follows an encoder-decoder structure using "
                "stacked self-attention and point-wise, fully connected layers "
                "for both the encoder and decoder."
            ),
            "Results": (
                "On the WMT 2014 English-to-German translation task, the big "
                "transformer model outperforms the best previously reported models "
                "including ensembles by more than 2.0 BLEU."
            ),
        },
    ),
    Paper(
        paper_id="paper-002",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=[
            Author(name="Devlin, J.", affiliation="Google AI Language"),
            Author(name="Chang, M.-W.", affiliation="Google AI Language"),
        ],
        abstract=(
            "We introduce a new language representation model called BERT, which "
            "stands for Bidirectional Encoder Representations from Transformers. "
            "BERT is designed to pre-train deep bidirectional representations from "
            "unlabeled text by jointly conditioning on both left and right context."
        ),
        source="stub_corpus",
        venue="NAACL 2019",
        publication_year=2019,
        sections={
            "Introduction": (
                "Language model pre-training has been shown to be effective for "
                "improving many natural language processing tasks. These include "
                "sentence-level tasks such as natural language inference and "
                "paraphrasing, as well as token-level tasks such as named entity "
                "recognition and question answering."
            ),
            "Pre-training BERT": (
                "We pre-train BERT using two unsupervised tasks. The first task "
                "is Masked Language Model (MLM). The second task is Next Sentence "
                "Prediction (NSP)."
            ),
        },
    ),
    Paper(
        paper_id="paper-003",
        title="GPT-3: Language Models are Few-Shot Learners",
        authors=[
            Author(name="Brown, T.", affiliation="OpenAI"),
            Author(name="Mann, B.", affiliation="OpenAI"),
        ],
        abstract=(
            "We demonstrate that scaling up language models greatly improves "
            "task-agnostic, few-shot performance, sometimes even reaching "
            "competitiveness with prior state-of-the-art fine-tuning approaches."
        ),
        source="stub_corpus",
        venue="NeurIPS 2020",
        publication_year=2020,
        sections={
            "Introduction": (
                "Recent work has demonstrated substantial gains on many NLP tasks "
                "and benchmarks by pre-training on a large corpus of text followed "
                "by fine-tuning on a specific task. Our approach is based on scaling "
                "up the language model in terms of parameters and training data."
            ),
            "Approach": (
                "Our approach uses in-context learning where the model is given a "
                "natural language instruction and a few demonstrations of the task "
                "and is then expected to complete further instances of the task."
            ),
            "Limitations": (
                "GPT-3 has several notable limitations. First, text generation can "
                "sometimes be repetitive. Second, the model's performance on "
                "structured prediction and tasks requiring precise reasoning "
                "remains limited."
            ),
        },
    ),
    Paper(
        paper_id="paper-004",
        title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        authors=[
            Author(name="Dosovitskiy, A.", affiliation="Google Research"),
        ],
        abstract=(
            "While the Transformer architecture has become the de-facto standard "
            "for natural language processing tasks, its applications to computer "
            "vision remain limited. We show that a pure transformer applied "
            "directly to sequences of image patches can perform very well on "
            "image classification tasks."
        ),
        source="stub_corpus",
        venue="ICLR 2021",
        publication_year=2021,
        sections={
            "Introduction": (
                "Self-attention-based architectures, in particular Transformers, "
                "have become the model of choice in natural language processing. "
                "Inspired by the Transformer scaling successes in NLP, we apply "
                "a standard Transformer directly to images."
            ),
            "Vision Transformer": (
                "We split an image into fixed-size patches, linearly embed each "
                "of them, add position embeddings, and feed the resulting sequence "
                "of vectors to a standard Transformer encoder. The Vision "
                "Transformer (ViT) attains excellent results when pre-trained "
                "at sufficient scale."
            ),
        },
    ),
    Paper(
        paper_id="paper-005",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=[
            Author(name="Lewis, P.", affiliation="Facebook AI Research"),
            Author(name="Perez, E.", affiliation="Facebook AI Research"),
        ],
        abstract=(
            "We explore a general-purpose fine-tuning recipe for retrieval-augmented "
            "generation (RAG) models which combine pre-trained parametric and "
            "non-parametric memory for language generation. RAG models retrieve "
            "documents relevant to a question and use them to generate answers."
        ),
        source="stub_corpus",
        venue="NeurIPS 2020",
        publication_year=2020,
        sections={
            "Introduction": (
                "Pre-trained neural language models have been shown to learn a "
                "substantial amount of in-depth knowledge from data. However, "
                "their ability to access and precisely manipulate knowledge is "
                "still limited. Retrieval-augmented generation addresses this "
                "limitation by combining parametric models with non-parametric "
                "retrieval from a document index."
            ),
            "RAG Models": (
                "We combine a pre-trained sequence-to-sequence model with a "
                "dense retrieval component based on Dense Passage Retrieval (DPR). "
                "Given a query, the retriever returns the top-K relevant documents "
                "from a large corpus, which are then used as additional context "
                "for the generator."
            ),
            "Experiments": (
                "RAG achieves state-of-the-art results on open-domain question "
                "answering benchmarks including Natural Questions, TriviaQA, and "
                "WebQuestions. The retrieval component significantly improves "
                "factual accuracy compared to purely parametric models."
            ),
        },
    ),
]


# ======================================================================
# Fixed chunk data — derived from paper sections
# ======================================================================

_CHUNKS: list[DocumentChunk] = [
    # Paper 001 — Transformer
    DocumentChunk(
        chunk_id="chunk-001-intro",
        paper_id="paper-001",
        section="Introduction",
        text=(
            "Recurrent neural networks, long short-term memory and gated "
            "recurrent neural networks have been firmly established as state "
            "of the art approaches in sequence modeling and transduction "
            "problems such as language modeling and machine translation."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-001-arch",
        paper_id="paper-001",
        section="Model Architecture",
        text=(
            "The Transformer follows an encoder-decoder structure using "
            "stacked self-attention and point-wise, fully connected layers "
            "for both the encoder and decoder."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-001-results",
        paper_id="paper-001",
        section="Results",
        text=(
            "On the WMT 2014 English-to-German translation task, the big "
            "transformer model outperforms the best previously reported models "
            "including ensembles by more than 2.0 BLEU."
        ),
    ),
    # Paper 002 — BERT
    DocumentChunk(
        chunk_id="chunk-002-intro",
        paper_id="paper-002",
        section="Introduction",
        text=(
            "Language model pre-training has been shown to be effective for "
            "improving many natural language processing tasks. These include "
            "sentence-level tasks such as natural language inference and "
            "paraphrasing, as well as token-level tasks such as named entity "
            "recognition and question answering."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-002-pretrain",
        paper_id="paper-002",
        section="Pre-training BERT",
        text=(
            "We pre-train BERT using two unsupervised tasks. The first task "
            "is Masked Language Model (MLM). The second task is Next Sentence "
            "Prediction (NSP)."
        ),
    ),
    # Paper 003 — GPT-3
    DocumentChunk(
        chunk_id="chunk-003-intro",
        paper_id="paper-003",
        section="Introduction",
        text=(
            "Recent work has demonstrated substantial gains on many NLP tasks "
            "and benchmarks by pre-training on a large corpus of text followed "
            "by fine-tuning on a specific task. Our approach is based on scaling "
            "up the language model in terms of parameters and training data."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-003-approach",
        paper_id="paper-003",
        section="Approach",
        text=(
            "Our approach uses in-context learning where the model is given a "
            "natural language instruction and a few demonstrations of the task "
            "and is then expected to complete further instances of the task."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-003-limits",
        paper_id="paper-003",
        section="Limitations",
        text=(
            "GPT-3 has several notable limitations. First, text generation can "
            "sometimes be repetitive. Second, the model's performance on "
            "structured prediction and tasks requiring precise reasoning "
            "remains limited."
        ),
    ),
    # Paper 004 — ViT
    DocumentChunk(
        chunk_id="chunk-004-intro",
        paper_id="paper-004",
        section="Introduction",
        text=(
            "Self-attention-based architectures, in particular Transformers, "
            "have become the model of choice in natural language processing. "
            "Inspired by the Transformer scaling successes in NLP, we apply "
            "a standard Transformer directly to images."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-004-vit",
        paper_id="paper-004",
        section="Vision Transformer",
        text=(
            "We split an image into fixed-size patches, linearly embed each "
            "of them, add position embeddings, and feed the resulting sequence "
            "of vectors to a standard Transformer encoder. The Vision "
            "Transformer (ViT) attains excellent results when pre-trained "
            "at sufficient scale."
        ),
    ),
    # Paper 005 — RAG
    DocumentChunk(
        chunk_id="chunk-005-intro",
        paper_id="paper-005",
        section="Introduction",
        text=(
            "Pre-trained neural language models have been shown to learn a "
            "substantial amount of in-depth knowledge from data. However, "
            "their ability to access and precisely manipulate knowledge is "
            "still limited. Retrieval-augmented generation addresses this "
            "limitation by combining parametric models with non-parametric "
            "retrieval from a document index."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-005-rag",
        paper_id="paper-005",
        section="RAG Models",
        text=(
            "We combine a pre-trained sequence-to-sequence model with a "
            "dense retrieval component based on Dense Passage Retrieval (DPR). "
            "Given a query, the retriever returns the top-K relevant documents "
            "from a large corpus, which are then used as additional context "
            "for the generator."
        ),
    ),
    DocumentChunk(
        chunk_id="chunk-005-exp",
        paper_id="paper-005",
        section="Experiments",
        text=(
            "RAG achieves state-of-the-art results on open-domain question "
            "answering benchmarks including Natural Questions, TriviaQA, and "
            "WebQuestions. The retrieval component significantly improves "
            "factual accuracy compared to purely parametric models."
        ),
    ),
]

# Index for fast lookup
_PAPERS_BY_ID: dict[str, Paper] = {p.paper_id: p for p in _PAPERS}
_CHUNKS_BY_PAPER: dict[str, list[DocumentChunk]] = {}
_CHUNKS_BY_ID: dict[str, DocumentChunk] = {}
for _c in _CHUNKS:
    _CHUNKS_BY_PAPER.setdefault(_c.paper_id, []).append(_c)
    _CHUNKS_BY_ID[_c.chunk_id] = _c


class StubCorpusStore:
    """
    Deterministic in-memory corpus store for development and testing.

    Contains 5 academic papers with 13 total document chunks.
    All data is hard-coded and fully reproducible across runs.

    Satisfies the ``CorpusStore`` protocol (app.domain.interfaces).

    .. warning::
        **Development stub.**  Will be replaced by a real database-backed
        implementation when corpus storage (Phase 3+) is implemented.
    """

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all document chunks in the mock corpus."""
        return list(_CHUNKS)

    def get_chunks_by_paper_id(self, paper_id: str) -> list[DocumentChunk]:
        """Return chunks belonging to a specific paper."""
        return list(_CHUNKS_BY_PAPER.get(paper_id, []))

    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Return a paper by ID, or None if not found."""
        return _PAPERS_BY_ID.get(paper_id)

    def get_all_papers(self) -> list[Paper]:
        """Return all papers in the mock corpus."""
        return list(_PAPERS)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Return a chunk by ID, or None if not found."""
        return _CHUNKS_BY_ID.get(chunk_id)
