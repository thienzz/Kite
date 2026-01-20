"""
Advanced RAG System
Enhances VectorMemory with query transformations, hybrid search, and re-ranking.
"""

import json
from typing import List, Dict, Optional, Tuple, Any
from .vector_memory import VectorMemory
import numpy as np
from rank_bm25 import BM25Okapi
import math
import os

# Advanced Rerankers
try:
    import cohere
except ImportError:
    cohere = None
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class QueryTransformer:
    """
    Transforms user queries to improve retrieval performance.
    """
    def __init__(self, llm):
        self.llm = llm

    def hyde(self, query: str) -> str:
        """Hypothetical Document Embedding (HyDE)."""
        prompt = f"Please write a short hypothetical document that answers the following query. This document will be used to retrieve relevant information from a database.\n\nQuery: {query}\n\nHypothetical Document:"
        hypothetical_doc = self.llm.complete(prompt)
        return hypothetical_doc

    def multi_query(self, query: str, n: int = 3) -> List[str]:
        """Generate multiple variations of the query."""
        prompt = f"Generate {n} different variations of the following search query to improve retrieval. Output as a JSON list of strings.\n\nQuery: {query}\n\nVariations:"
        response = self.llm.complete(prompt)
        try:
            # Try to parse JSON
            queries = json.loads(response)
            if isinstance(queries, list):
                return queries
        except:
            # Fallback to line splitting if JSON fails
            return [line.strip() for line in response.split('\n') if line.strip() and not line.startswith('[') and not line.endswith(']')][:n]
        return [query]

    def expand(self, query: str) -> str:
        """Query expansion with relevant keywords."""
        prompt = f"Expand the following search query with relevant keywords and technical terms to improve search results. Return just the expanded query.\n\nQuery: {query}"
        return self.llm.complete(prompt).strip()

    def step_back(self, query: str) -> str:
        """Step-back prompting: generate a more general query."""
        prompt = f"Generate a broader, higher-level technical question that provides context for the following specific query. This will help retrieve foundational concepts.\n\nSpecific Query: {query}\n\nStep-back Query:"
        return self.llm.complete(prompt).strip()


class AdvancedRAG:
    """
    Wraps VectorMemory with advanced search strategies.
    """
    def __init__(self, vector_memory: VectorMemory, llm = None):
        self.memory = vector_memory
        self.llm = llm or vector_memory.embedding_provider # Fallback to provider if needed for llm
        self.transformer = QueryTransformer(self.llm) if self.llm else None
        self.bm25 = None
        self.corpus = []
        self.id_map = []
        self.cohere_client = None
        self.cross_encoder = None
        
        # Recursive Retrieval mappings
        self.child_to_parent = {} # child_id -> parent_id
        self.parents = {} # parent_id -> parent_text

    def initialize_bm25(self, documents: List[Dict]):
        """
        Initialize BM25 index with a list of documents.
        documents: list of {'id': id, 'text': text}
        """
        self.corpus = [doc['text'].lower().split() for doc in documents]
        self.id_map = [doc['id'] for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[OK] BM25 initialized with {len(self.corpus)} documents")

    def search(self, 
               query: str, 
               strategy: str = "simple", 
               k: int = 5, 
               alpha: float = 0.5) -> List[Tuple]:
        """
        Perform search using specified strategy.
        Strategies: simple, hyde, multi_query, hybrid
        """
        if strategy == "hyde" and self.transformer:
            transformed_query = self.transformer.hyde(query)
            return self.memory.search(transformed_query, k=k)
        
        elif strategy == "multi_query" and self.transformer:
            queries = self.transformer.multi_query(query)
            all_results = []
            for q in queries:
                all_results.extend(self.memory.search(q, k=k))
            
            # Simple deduplication by doc_id
            seen = set()
            unique_results = []
            for res in all_results:
                if res[0] not in seen:
                    unique_results.append(res)
                    seen.add(res[0])
            return unique_results[:k]
        
        elif strategy == "hybrid":
            return self._hybrid_search(query, k=k, alpha=alpha)
            
        elif strategy == "recursive":
            return self.search_recursive(query, k=k)
            
        return self.memory.search(query, k=k)

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> Dict[str, Any]:
        """
        Public hybrid search that returns a structured dictionary.
        """
        results = self._hybrid_search(query, k=top_k, alpha=alpha)
        
        if not results:
            return {
                "answer": "No relevant documents found.",
                "documents": [],
                "success": False
            }
            
        return {
            "answer": results[0][1], # Top result as answer
            "documents": [{"id": r[0], "content": r[1], "score": r[2]} for r in results],
            "success": True,
            "source": "hybrid_rag"
        }

    def _hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple]:
        """
        Combines BM25 and Vector Search using Weighted Fusion.
        alpha: 0 = semantic only, 1 = keyword only
        """
        # 1. Semantic Search
        vector_results = self.memory.search(query, k=k*2)
        
        # 2. Keyword Search
        if not self.bm25:
            return vector_results[:k]
            
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores to 0-1 range
        if len(bm25_scores) > 0:
            max_s = max(bm25_scores)
            min_s = min(bm25_scores)
            if max_s > min_s:
                bm25_scores = [(s - min_s) / (max_s - min_s) for s in bm25_scores]
            else:
                bm25_scores = [1.0] * len(bm25_scores)
                
        # 3. Fusion (Weighted)
        # alpha * keyword + (1 - alpha) * semantic
        # Note: Chroma distances are distance (lower better), so we use 1 - distance for similarity
        combined_scores = {}
        
        # Add vector results
        for doc_id, text, distance in vector_results:
            # Strip chunk suffix for deduplication
            base_id = doc_id.split('_chunk_')[0]
            score = 1 - distance # Similarity
            
            if base_id in combined_scores:
                # Keep highest score if already present
                combined_scores[base_id]['score'] = max(combined_scores[base_id]['score'], (1 - alpha) * score)
            else:
                combined_scores[base_id] = {
                    'text': text,
                    'score': (1 - alpha) * score
                }
            
        # Add BM25 results
        for i, score in enumerate(bm25_scores):
            doc_id = self.id_map[i]
            # Strip chunk suffix for deduplication
            base_id = doc_id.split('_chunk_')[0]
            text = " ".join(self.corpus[i]) 
            
            if base_id in combined_scores:
                combined_scores[base_id]['score'] += alpha * score
            else:
                combined_scores[base_id] = {
                    'text': text,
                    'score': alpha * score
                }
        
        # Sort and return
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        return [(doc_id, info['text'], info['score']) for doc_id, info in sorted_results[:k]]

    def mmr(self, query: str, results: List[Tuple], k: int = 3, lambda_param: float = 0.5) -> List[Tuple]:
        """
        Maximal Marginal Relevance (MMR) for diversification.
        lambda_param: 1.0 = relevance only, 0.0 = diversity only
        """
        if not results or len(results) <= k:
            return results[:k]
            
        # Get query embedding
        query_emb = self.memory._get_embedding(query)
        
        # Get document embeddings
        # We need to re-embed if not provided, which is expensive.
        # Ideally VectorMemory returns embeddings or we cache them.
        doc_embs = [self.memory._get_embedding(res[1]) for res in results]
        
        selected_indices = [0]
        remaining_indices = list(range(1, len(results)))
        
        while len(selected_indices) < k and remaining_indices:
            best_score = -float('inf')
            best_idx = -1
            
            for i in remaining_indices:
                # Similarity to query
                rel = np.dot(query_emb, doc_embs[i]) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_embs[i]))
                
                # Similarity to already selected docs (redundancy)
                max_sim = max([np.dot(doc_embs[i], doc_embs[j]) / (np.linalg.norm(doc_embs[i]) * np.linalg.norm(doc_embs[j])) for j in selected_indices])
                
                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
                
        return [results[i] for i in selected_indices]

    def rerank_cohere(self, query: str, results: List[Tuple], top_n: int = 3) -> List[Tuple]:
        """Rerank using Cohere Rerank API."""
        if cohere is None: raise ImportError("pip install cohere")
        if not self.cohere_client:
            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        
        docs = [res[1] for res in results]
        rerank_results = self.cohere_client.rerank(
            query=query, documents=docs, top_n=top_n, model="rerank-english-v3.0"
        )
        
        output = []
        for res in rerank_results.results:
            idx = res.index
            output.append((results[idx][0], results[idx][1], res.relevance_score))
        return output

    def rerank_cross_encoder(self, query: str, results: List[Tuple], top_n: int = 3) -> List[Tuple]:
        """Rerank using a Cross-Encoder model."""
        if CrossEncoder is None: raise ImportError("pip install sentence-transformers")
        if not self.cross_encoder:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
        pairs = [[query, res[1]] for res in results]
        scores = self.cross_encoder.predict(pairs)
        
        scored_results = []
        for i, score in enumerate(scores):
            scored_results.append((results[i][0], results[i][1], float(score)))
            
        scored_results.sort(key=lambda x: x[2], reverse=True)
        return scored_results[:top_n]

    def add_document_recursive(self, parent_id: str, text: str):
        """
        Store a document with parent-child chunking.
        The agent searches small chunks but retrieves the larger parent context.
        """
        # Store parent
        self.parents[parent_id] = text
        
        # Chunk into small pieces for better retrieval
        chunks = self.memory._chunk_text(text, chunk_size=200, overlap=20)
        for i, chunk in enumerate(chunks):
            child_id = f"{parent_id}_small_{i}"
            self.memory.add_document(child_id, chunk, auto_chunk=False)
            self.child_to_parent[child_id] = parent_id
        
        print(f"[OK] Added document {parent_id} recursively with {len(chunks)} small chunks")

    def search_recursive(self, query: str, k: int = 3) -> List[Tuple]:
        """
        Search small chunks but return parent contexts.
        """
        child_results = self.memory.search(query, k=k*2)
        
        parent_results = []
        seen_parents = set()
        
        for child_id, _, score in child_results:
            # Map child back to parent
            # Handle both our manual mapping and potential Naming convention
            parent_id = self.child_to_parent.get(child_id)
            if not parent_id and "_small_" in child_id:
                parent_id = child_id.split("_small_")[0]
            
            if parent_id and parent_id in self.parents:
                if parent_id not in seen_parents:
                    parent_results.append((parent_id, self.parents[parent_id], score))
                    seen_parents.add(parent_id)
            
            if len(parent_results) >= k:
                break
                
        # If recursive fails (no parents found), fallback to normal
        return parent_results if parent_results else child_results[:k]
