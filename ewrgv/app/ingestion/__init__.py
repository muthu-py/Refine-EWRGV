"""
app/ingestion/__init__.py
--------------------------
Literature ingestion package.

Responsibilities
----------------
* Collecting papers from external providers (collectors/)
* Parsing raw documents into Paper domain objects (parsers/)
* Splitting papers into logical sections (sectioning/)
* Chunking sections into retrievable PaperChunk objects (chunking/)
* Building a run's retrieval corpus from those chunks (corpus/)

This package does NOT perform retrieval or gap detection.
"""
