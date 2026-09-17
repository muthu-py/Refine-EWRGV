from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.domain.models.paper_chunk import PaperChunk
from app.retrieval.bm25_index import BM25Index, preprocess_text

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

def create_mock_chunk(chunk_id: str, text: str) -> PaperChunk:
    return PaperChunk(
        chunk_id=chunk_id,
        paper_id="paper1",
        section_id="sec1",
        section_title="Introduction",
        text=text,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_preprocessing():
    # Basic lowercasing and tokenization
    assert preprocess_text("Hello World") == ["hello", "world"]
    
    # Stop words removed
    assert preprocess_text("the quick brown fox") == ["quick", "brown", "fox"]
    
    # Technical terminology preserved
    assert preprocess_text("GPT-4 and COVID-19 with YOLOv8") == ["gpt-4", "covid-19", "yolov8"]
    
    # Punctuation mostly stripped but hyphens in words kept
    assert preprocess_text("Wait, what? A test-case!") == ["wait", "what", "test-case"]
    
    # Unicode normalization
    text1 = "caf\u00e9" # precomposed
    text2 = "cafe\u0301" # decomposed
    assert preprocess_text(text1) == preprocess_text(text2)


def test_build_empty_index():
    index = BM25Index().build([])
    assert index.is_empty()
    assert index.size == 0
    assert index.query("test") == []


def test_build_and_query():
    chunks = [
        create_mock_chunk("c1", "RAG systems using BM25 and dense embeddings"),
        create_mock_chunk("c2", "A completely different topic about quantum physics"),
        create_mock_chunk("c3", "BM25 is a sparse retrieval algorithm"),
    ]
    
    index = BM25Index().build(chunks)
    
    assert not index.is_empty()
    assert index.size == 3
    
    # Query matching multiple
    results = index.query("BM25 retrieval", top_k=5)
    assert len(results) == 2
    # c3 has "BM25" and "retrieval", c1 only has "BM25". So c3 should rank higher.
    assert results[0][0] == "c3"
    assert results[1][0] == "c1"


def test_query_top_k():
    chunks = [
        create_mock_chunk(f"c{i}", f"test document {i}")
        for i in range(10)
    ]
    
    index = BM25Index().build(chunks)
    
    results = index.query("test", top_k=3)
    assert len(results) == 3


def test_persistence():
    chunks = [
        create_mock_chunk("c1", "RAG systems using BM25 and dense embeddings"),
        create_mock_chunk("c2", "A completely different topic about quantum physics"),
    ]
    
    original_index = BM25Index().build(chunks)
    
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "index.pkl"
        
        # Save
        original_index.save(file_path)
        assert file_path.exists()
        
        # Load
        loaded_index = BM25Index.load(file_path)
        
        assert loaded_index.size == 2
        assert not loaded_index.is_empty()
        
        # Verify query produces same results
        orig_results = original_index.query("BM25")
        loaded_results = loaded_index.query("BM25")
        
        assert orig_results == loaded_results


def test_load_nonexistent():
    with pytest.raises(FileNotFoundError):
        BM25Index.load("nonexistent_file.pkl")
