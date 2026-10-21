"""
tests/fixtures/sample_corpus.py
---------------------------------
Deterministic in-memory test corpus of PaperChunks.

A small corpus of 5 ML/NLP papers with 2–3 chunks each, wrapped in a
RunCorpus.  All IDs and text are fixed so retrieval tests are fully
deterministic.  (Formerly app/stubs/stub_corpus.py; the production corpus
is RunCorpus loaded from a run's chunks.json artifacts.)

Also provides FakeEmbeddingProvider: deterministic hash-based vectors so
dense retrieval can be tested without an embedding API.
"""

from __future__ import annotations

import hashlib
import math

from app.domain.enums import SectionType
from app.domain.models.paper import Author, Paper
from app.domain.models.paper_chunk import PaperChunk
from app.ingestion.corpus.run_corpus import RunCorpus


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
        source="sample_corpus",
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
        source="sample_corpus",
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
        source="sample_corpus",
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
        source="sample_corpus",
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
        source="sample_corpus",
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


_CHUNKS: list[PaperChunk] = [
    # Paper 001 — Transformer
    PaperChunk(
        chunk_id="chunk-001-intro",
        paper_id="paper-001",
        section_id="sec-chunk-001-intro",
        section_title="Introduction",
        section_type=SectionType.INTRODUCTION,
        text=(
            "Recurrent neural networks, long short-term memory and gated "
            "recurrent neural networks have been firmly established as state "
            "of the art approaches in sequence modeling and transduction "
            "problems such as language modeling and machine translation."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-001-arch",
        paper_id="paper-001",
        section_id="sec-chunk-001-arch",
        section_title="Model Architecture",
        section_type=SectionType.MODEL,
        text=(
            "The Transformer follows an encoder-decoder structure using "
            "stacked self-attention and point-wise, fully connected layers "
            "for both the encoder and decoder."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-001-results",
        paper_id="paper-001",
        section_id="sec-chunk-001-results",
        section_title="Results",
        section_type=SectionType.RESULTS,
        text=(
            "On the WMT 2014 English-to-German translation task, the big "
            "transformer model outperforms the best previously reported models "
            "including ensembles by more than 2.0 BLEU."
        ),
    ),
    # Paper 002 — BERT
    PaperChunk(
        chunk_id="chunk-002-intro",
        paper_id="paper-002",
        section_id="sec-chunk-002-intro",
        section_title="Introduction",
        section_type=SectionType.INTRODUCTION,
        text=(
            "Language model pre-training has been shown to be effective for "
            "improving many natural language processing tasks. These include "
            "sentence-level tasks such as natural language inference and "
            "paraphrasing, as well as token-level tasks such as named entity "
            "recognition and question answering."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-002-pretrain",
        paper_id="paper-002",
        section_id="sec-chunk-002-pretrain",
        section_title="Pre-training BERT",
        section_type=SectionType.METHODS,
        text=(
            "We pre-train BERT using two unsupervised tasks. The first task "
            "is Masked Language Model (MLM). The second task is Next Sentence "
            "Prediction (NSP)."
        ),
    ),
    # Paper 003 — GPT-3
    PaperChunk(
        chunk_id="chunk-003-intro",
        paper_id="paper-003",
        section_id="sec-chunk-003-intro",
        section_title="Introduction",
        section_type=SectionType.INTRODUCTION,
        text=(
            "Recent work has demonstrated substantial gains on many NLP tasks "
            "and benchmarks by pre-training on a large corpus of text followed "
            "by fine-tuning on a specific task. Our approach is based on scaling "
            "up the language model in terms of parameters and training data."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-003-approach",
        paper_id="paper-003",
        section_id="sec-chunk-003-approach",
        section_title="Approach",
        section_type=SectionType.METHODS,
        text=(
            "Our approach uses in-context learning where the model is given a "
            "natural language instruction and a few demonstrations of the task "
            "and is then expected to complete further instances of the task."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-003-limits",
        paper_id="paper-003",
        section_id="sec-chunk-003-limits",
        section_title="Limitations",
        section_type=SectionType.LIMITATIONS,
        text=(
            "GPT-3 has several notable limitations. First, text generation can "
            "sometimes be repetitive. Second, the model's performance on "
            "structured prediction and tasks requiring precise reasoning "
            "remains limited."
        ),
    ),
    # Paper 004 — ViT
    PaperChunk(
        chunk_id="chunk-004-intro",
        paper_id="paper-004",
        section_id="sec-chunk-004-intro",
        section_title="Introduction",
        section_type=SectionType.INTRODUCTION,
        text=(
            "Self-attention-based architectures, in particular Transformers, "
            "have become the model of choice in natural language processing. "
            "Inspired by the Transformer scaling successes in NLP, we apply "
            "a standard Transformer directly to images."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-004-vit",
        paper_id="paper-004",
        section_id="sec-chunk-004-vit",
        section_title="Vision Transformer",
        section_type=SectionType.MODEL,
        text=(
            "We split an image into fixed-size patches, linearly embed each "
            "of them, add position embeddings, and feed the resulting sequence "
            "of vectors to a standard Transformer encoder. The Vision "
            "Transformer (ViT) attains excellent results when pre-trained "
            "at sufficient scale."
        ),
    ),
    # Paper 005 — RAG
    PaperChunk(
        chunk_id="chunk-005-intro",
        paper_id="paper-005",
        section_id="sec-chunk-005-intro",
        section_title="Introduction",
        section_type=SectionType.INTRODUCTION,
        text=(
            "Pre-trained neural language models have been shown to learn a "
            "substantial amount of in-depth knowledge from data. However, "
            "their ability to access and precisely manipulate knowledge is "
            "still limited. Retrieval-augmented generation addresses this "
            "limitation by combining parametric models with non-parametric "
            "retrieval from a document index."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-005-rag",
        paper_id="paper-005",
        section_id="sec-chunk-005-rag",
        section_title="RAG Models",
        section_type=SectionType.MODEL,
        text=(
            "We combine a pre-trained sequence-to-sequence model with a "
            "dense retrieval component based on Dense Passage Retrieval (DPR). "
            "Given a query, the retriever returns the top-K relevant documents "
            "from a large corpus, which are then used as additional context "
            "for the generator."
        ),
    ),
    PaperChunk(
        chunk_id="chunk-005-exp",
        paper_id="paper-005",
        section_id="sec-chunk-005-exp",
        section_title="Experiments",
        section_type=SectionType.EXPERIMENTS,
        text=(
            "RAG achieves state-of-the-art results on open-domain question "
            "answering benchmarks including Natural Questions, TriviaQA, and "
            "WebQuestions. The retrieval component significantly improves "
            "factual accuracy compared to purely parametric models."
        ),
    ),
]

# Index for fast lookup


def sample_papers() -> list[Paper]:
    return [p.model_copy(deep=True) for p in _PAPERS]


def sample_chunks() -> list[PaperChunk]:
    return [c.model_copy(deep=True) for c in _CHUNKS]


def make_sample_corpus(**kwargs) -> RunCorpus:
    """RunCorpus over the sample papers and chunks (kwargs go to RunCorpus)."""
    return RunCorpus(sample_papers(), sample_chunks(), run_id=kwargs.pop("run_id", "run-sample"), **kwargs)


EMBEDDING_DIM = 128


class FakeEmbeddingProvider:
    """
    Deterministic embedding provider: SHA-256 of the text expanded to
    ``dim`` floats and L2-normalised.  Same text → same vector; no API calls.
    Satisfies the EmbeddingProvider protocol.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim
        self.embed_calls = 0
        self.batch_calls: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dim

    def get_model_name(self) -> str:
        return "fake-hash-embedding"

    def get_dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(len(texts))
        return [self._hash_to_vector(t) for t in texts]

    async def embed_text(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(digest[i % len(digest)] / 127.5) - 1.0 for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw] if norm > 0 else raw
