import pytest

from app.domain.enums import SectionType
from app.domain.models.paper_chunk import PaperChunk
from app.storage.repositories.embedding_repository import EmbeddingRecord


def test_provenance_chain_integrity():
    """
    Validates that the representation layer components preserve provenance
    all the way from embedding back to the canonical chunk and paper.
    
    This is critical for EWRGV validation.
    """
    # 1. Start with a canonical chunk
    chunk = PaperChunk(
        chunk_id="c123",
        paper_id="p456",
        section_id="s789",
        section_title="Methodology",
        section_type=SectionType.METHODOLOGY,
        chunk_index=0,
        text="The methodology uses BM25...",
        page_start=2,
        page_end=2,
    )
    
    # 2. Simulate Representation layer creating an embedding record
    record = EmbeddingRecord(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test-model",
        embedding_dimension=3,
    )
    
    # 3. Verify provenance recovery from the EmbeddingRecord
    assert record.chunk_id == chunk.chunk_id
    assert record.paper_id == chunk.paper_id
    
    # 4. In a real system, the chunk_id is used to fetch the chunk from chunks.json
    # Here we verify the canonical chunk itself retains the deeper provenance needed for explanations
    assert chunk.section_id == "s789"
    assert chunk.section_type == SectionType.METHODOLOGY
    assert chunk.page_start == 2
