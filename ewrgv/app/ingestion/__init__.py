"""
app/ingestion/__init__.py
--------------------------
Literature ingestion package.

Responsibilities
----------------
* Collecting papers from external providers (collectors/)
* Parsing raw documents into Paper domain objects (parsers/)
* Splitting papers into logical sections (sectioning/)
* Chunking sections into retrievable DocumentChunk objects (chunking/)

This package does NOT perform retrieval or gap detection.
"""
