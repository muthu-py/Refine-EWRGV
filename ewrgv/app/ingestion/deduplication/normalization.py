"""
app/ingestion/deduplication/normalization.py
----------------------------------------------
Database-independent normalisation helpers used to build deduplication
signatures.

Shared by the database Deduplicator (service.py, which re-exports these for
backward compatibility) and the in-memory CollectionCorpusStore.  This module
must not import asyncpg or any other database driver.
"""

import re


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.lower().strip()
    doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi)
    doi = doi.strip()
    return doi if doi else None


def normalize_title_authors_year(title: str | None, authors: list[dict] | None, year: int | None) -> str | None:
    if not title or not year:
        return None

    # Normalize title
    t = title.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return None

    # Normalize authors
    author_words = []
    if authors:
        for a in authors:
            name = a.get("name", "") if isinstance(a, dict) else ""
            if name:
                words = re.findall(r'[a-z]+', name.lower())
                author_words.extend(words)
    author_words = sorted(list(set(author_words)))
    a_str = "_".join(author_words)

    return f"{t}|{a_str}|{year}"
