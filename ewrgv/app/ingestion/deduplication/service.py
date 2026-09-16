import json
import re
from typing import List, Dict, Any, Set
from asyncpg import Pool
from collections import defaultdict
from app.core.logging import get_logger

logger = get_logger(__name__)

def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.lower().strip()
    doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi)
    doi = doi.strip()
    return doi if doi else None

def normalize_title_authors_year(title: str | None, authors: List[dict] | None, year: int | None) -> str | None:
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

class Deduplicator:
    def __init__(self, db_pool: Pool):
        self._pool = db_pool
        
    async def run(self) -> dict:
        """
        Runs deduplication on the entire persisted corpus.
        Returns stats about the operation.
        """
        async with self._pool.acquire() as conn:
            # 1. Fetch all papers
            papers_rows = await conn.fetch("SELECT paper_id, title, abstract, doi, publication_year, authors FROM papers")
            
            # 2. Fetch all sources to get provider IDs
            sources_rows = await conn.fetch("SELECT source_id, paper_id, provider, provider_paper_id FROM paper_sources")
            
            # Map paper_id -> sources
            paper_to_sources = defaultdict(list)
            for s in sources_rows:
                paper_to_sources[s['paper_id']].append(s)
                
            papers = {}
            for p in papers_rows:
                p_dict = dict(p)
                authors = p_dict['authors']
                if isinstance(authors, str):
                    try:
                        authors = json.loads(authors)
                    except json.JSONDecodeError:
                        authors = []
                elif authors is None:
                    authors = []
                p_dict['authors'] = authors
                papers[p_dict['paper_id']] = p_dict

            # 3. Build graph
            # Nodes: paper_id
            # Edges: built implicitly by sharing signatures
            signatures = defaultdict(list)
            
            for pid, p in papers.items():
                # DOI signature
                doi = normalize_doi(p.get('doi'))
                if doi:
                    signatures[f"doi:{doi}"].append(pid)
                    
                # Title+Author+Year signature
                tay = normalize_title_authors_year(p.get('title'), p.get('authors'), p.get('publication_year'))
                if tay:
                    signatures[f"tay:{tay}"].append(pid)
                    
                # Provider signatures
                for src in paper_to_sources[pid]:
                    provider = src.get('provider')
                    provider_pid = src.get('provider_paper_id')
                    if provider and provider_pid:
                        signatures[f"prov:{provider}:{provider_pid}"].append(pid)
            
            # Find connected components
            adj = defaultdict(set)
            for sig, pids in signatures.items():
                if len(pids) > 1:
                    for i in range(len(pids)):
                        for j in range(i+1, len(pids)):
                            adj[pids[i]].add(pids[j])
                            adj[pids[j]].add(pids[i])
                            
            visited = set()
            components = []
            
            for pid in papers.keys():
                if pid not in visited and pid in adj:
                    comp = []
                    q = [pid]
                    visited.add(pid)
                    while q:
                        curr = q.pop(0)
                        comp.append(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    if len(comp) > 1:
                        components.append(comp)
                        
            merged_count = 0
            deleted_count = 0
            
            # 4. Merge components safely within a transaction
            async with conn.transaction():
                for comp in components:
                    # Sort to ensure deterministic canonical choice (e.g. string representation of UUID)
                    comp.sort(key=lambda x: str(x))
                    canonical_id = comp[0]
                    duplicates = comp[1:]
                    
                    c_paper = papers[canonical_id].copy()
                    
                    # Aggregate metadata (prefer non-null/non-empty values)
                    for dup_id in duplicates:
                        d_paper = papers[dup_id]
                        if not c_paper.get('abstract') and d_paper.get('abstract'):
                            c_paper['abstract'] = d_paper['abstract']
                        if not c_paper.get('doi') and d_paper.get('doi'):
                            c_paper['doi'] = d_paper['doi']
                        if not c_paper.get('publication_year') and d_paper.get('publication_year'):
                            c_paper['publication_year'] = d_paper['publication_year']
                            
                        # If canonical has no authors, inherit from duplicate
                        if not c_paper.get('authors') and d_paper.get('authors'):
                            c_paper['authors'] = d_paper['authors']
                            
                    # 1. Update canonical paper
                    await conn.execute("""
                        UPDATE papers 
                        SET title = $1, abstract = $2, doi = $3, publication_year = $4, authors = $5::jsonb, updated_at = CURRENT_TIMESTAMP
                        WHERE paper_id = $6
                    """, c_paper['title'], c_paper['abstract'], c_paper['doi'], c_paper['publication_year'], json.dumps(c_paper['authors']), canonical_id)
                    
                    # 2. Transfer provenance to the canonical paper
                    await conn.execute("""
                        UPDATE paper_sources
                        SET paper_id = $1
                        WHERE paper_id = ANY($2)
                    """, canonical_id, duplicates)
                    
                    # 3. Delete duplicate papers
                    await conn.execute("""
                        DELETE FROM papers
                        WHERE paper_id = ANY($1)
                    """, duplicates)
                    
                    merged_count += 1
                    deleted_count += len(duplicates)
                    
            logger.info("Deduplication complete.", extra={"components_merged": merged_count, "duplicates_removed": deleted_count})
            
            return {
                "components_merged": merged_count,
                "duplicates_removed": deleted_count
            }
