"""
Vector Memory System
Based on Chapter 3.3: Long-Term Memory

Semantic search using vector embeddings for agent knowledge retrieval.

Key insight from book:
Scenario A (Smart): Retrieve 1 relevant page   $0.01 per query
Scenario B (Lazy): Dump 500-page manual   $1.00 per query

For 1,000 users/day: Smart = $300/month, Lazy = $30,000/month!

Run: python vector_memory.py
"""

import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import numpy as np
import json

# import chromadb  # Moved inside _init_chroma to avoid top-level Pydantic side effects
# from chromadb.config import Settings
try:
    import faiss
except ImportError:
    faiss = None

# Production Vector DBs
# import qdrant_client  # Moved inside _init_qdrant 
# import pinecone       # Moved inside _init_pinecone
# import weaviate       # Moved inside _init_weaviate
# import pymilvus       # Moved inside _init_milvus
# import psycopg2       # Moved inside _init_pgvector
# import redis          # Moved inside _init_redis
# import elasticsearch  # Moved inside _init_elasticsearch

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Document:
    """A document to store in vector memory."""
    id: str
    text: str
    metadata: Dict = None


class VectorMemory:
    """
    Vector-based long-term memory for AI agents.
    Supports multiple backends: chroma, faiss, qdrant, memory.
    """
    def __init__(self, 
                 backend: str = "chroma",
                 collection_name: str = "agent_memory", 
                 persist_dir: str = "./vector_db",
                 embedding_provider = None):
        self.backend = backend.lower()
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_provider = embedding_provider
        self.logger = logging.getLogger(f"VectorMemory({self.backend})")
        
        if self.backend == "chroma":
            self._init_chroma()
        elif self.backend == "faiss":
            self._init_faiss()
        elif self.backend == "qdrant":
            self._init_qdrant()
        elif self.backend == "pinecone":
            self._init_pinecone()
        elif self.backend == "weaviate":
            self._init_weaviate()
        elif self.backend == "milvus":
            self._init_milvus()
        elif self.backend == "pgvector":
            self._init_pgvector()
        elif self.backend == "redis":
            self._init_redis()
        elif self.backend == "elasticsearch":
            self._init_elasticsearch()
        elif self.backend == "memory":
            self._init_memory_backend()
        else:
            self._init_chroma() # Default
            
    def _init_chroma(self):
        import chromadb
        try:
            # New Chromadb API (0.4.0+)
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        except AttributeError:
            # Old Chromadb API (0.3.x)
            from chromadb.config import Settings
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir
            ))
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        print(f"[OK] Vector memory (Chroma) initialized")

    def _init_faiss(self):
        if faiss is None:
            raise ImportError("faiss-cpu not installed. Run 'pip install faiss-cpu'")
        # Dimension depends on embedding model, 384 for all-MiniLM-L6-v2
        dim = 384 
        self.index = faiss.IndexFlatL2(dim) 
        self.doc_store = {} # int_id -> (id, text, metadata)
        self.id_to_int = {} # string_id -> int_id
        print(f"[OK] Vector memory (FAISS) initialized")

    def _init_qdrant(self):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        
        # Ensure collection exists
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        )
        print(f"[OK] Vector memory (Qdrant) initialized")

    def _init_pinecone(self):
        import pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
            
        pc = pinecone.Pinecone(api_key=api_key)
        # Check if index exists, else create
        if self.collection_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.collection_name,
                dimension=384,
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = pc.Index(self.collection_name)
        print(f"[OK] Vector memory (Pinecone) initialized")

    def _init_weaviate(self):
        import weaviate
        
        auth_config = weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")) if os.getenv("WEAVIATE_API_KEY") else None
        self.client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            auth_client_config=auth_config
        )
        print(f"[OK] Vector memory (Weaviate) initialized")

    def _init_milvus(self):
        from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
        
        connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port=os.getenv("MILVUS_PORT", "19530"))
        
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            schema = CollectionSchema(fields, "Kite vector memory")
            self.collection = Collection(self.collection_name, schema)
            
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
        else:
            self.collection = Collection(self.collection_name)
        
        self.collection.load()
        print(f"[OK] Vector memory (Milvus) initialized")

    def _init_pgvector(self):
        import psycopg2
        from psycopg2.extras import execute_values
        
        self.conn = psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"))
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"CREATE TABLE IF NOT EXISTS {self.collection_name} (id TEXT PRIMARY KEY, text TEXT, embedding vector(384), metadata JSONB)")
        self.conn.commit()
        print(f"[OK] Vector memory (PGVector) initialized")

    def _init_redis(self):
        import redis
        
        self.client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD")
        )
        # RediSearch logic would go here
        print(f"[OK] Vector memory (Redis) initialized")

    def _init_elasticsearch(self):
        from elasticsearch import Elasticsearch, helpers
        
        self.client = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
        if not self.client.indices.exists(index=self.collection_name):
            self.client.indices.create(
                index=self.collection_name,
                body={
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                            "text": {"type": "text"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )
        print(f"[OK] Vector memory (Elasticsearch) initialized")

    def _init_memory_backend(self):
        self.storage = [] # list of (id, text, vector, metadata)
        print(f"[OK] Vector memory (In-Memory) initialized")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into semantic chunks."""
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_sentences = chunks[i-1].split('. ')
                overlap_text = prev_sentences[-1] + ". " if len(prev_sentences) > 0 else ""
                overlapped_chunks.append(overlap_text + chunks[i])
            chunks = overlapped_chunks
        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding."""
        if self.embedding_provider:
            return self.embedding_provider.embed(text)
        # Fallback dummy embedding if none
        return [0.1] * 384
    
    def store(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """Alias for add_document for compatibility."""
        return self.add_document(doc_id, text, metadata)
        
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None,
        auto_chunk: bool = True
    ) -> int:
        """
        Add document to vector memory.
        
        Args:
            doc_id: Unique document ID
            text: Document text
            metadata: Optional metadata
            auto_chunk: Whether to auto-chunk large documents
            
        Returns:
            Number of chunks added
        """
        print(f"\n  Adding document: {doc_id}")
        
        # Chunk if needed
        if auto_chunk and len(text) > 500:
            chunks = self._chunk_text(text)
            print(f"  Chunked into {len(chunks)} pieces")
        else:
            chunks = [text]
        
        # Generate embeddings
        embeddings = [self._get_embedding(chunk) for chunk in chunks]
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {"source": "Kite", "id": doc_id} for _ in chunks]
        
        if self.backend == "chroma":
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
        elif self.backend == "faiss":
            for i, (cid, chunk, emb, meta) in enumerate(zip(ids, chunks, embeddings, metadatas)):
                int_id = len(self.id_to_int)
                self.id_to_int[cid] = int_id
                self.index.add(np.array([emb]).astype('float32'))
                self.doc_store[int_id] = (cid, chunk, meta)
        elif self.backend == "qdrant":
            points = [
                qmodels.PointStruct(
                    id=cid,
                    vector=emb,
                    payload={"text": chunk, **meta}
                ) for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        elif self.backend == "pinecone":
            to_upsert = [
                (cid, emb, {"text": chunk, **meta})
                for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas)
            ]
            self.index.upsert(vectors=to_upsert)
        elif self.backend == "weaviate":
            with self.client.batch as batch:
                for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas):
                    batch.add_data_object(
                        data_object={"text": chunk, **meta},
                        class_name="Document",
                        vector=emb
                    )
        elif self.backend == "milvus":
            entities = [ids, chunks, embeddings]
            self.collection.insert(entities)
        elif self.backend == "pgvector":
            with self.conn.cursor() as cur:
                execute_values(cur, 
                    f"INSERT INTO {self.collection_name} (id, text, embedding, metadata) VALUES %s ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata",
                    [(cid, chunk, emb, json.dumps(meta)) for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas)])
            self.conn.commit()
        elif self.backend == "elasticsearch":
            actions = [
                {
                    "_index": self.collection_name,
                    "_id": cid,
                    "_source": {"text": chunk, "embedding": emb, "metadata": meta}
                }
                for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas)
            ]
            helpers.bulk(self.client, actions)
        elif self.backend == "memory":
            for cid, chunk, emb, meta in zip(ids, chunks, embeddings, metadatas):
                self.storage.append((cid, chunk, emb, meta))
        # Qdrant would follow similar pattern
        
        print(f"  [OK] Added {len(chunks)} chunks with embeddings")
        return len(chunks)
    
    def search(
        self,
        query: str,
        k: int = 3,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Search for most similar documents.
        """
        k = top_k or k
        query_emb = self._get_embedding(query)
        
        if self.backend == "chroma":
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=k,
                where=filter_metadata
            )
            output = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    output.append((
                        results['ids'][0][i],
                        results['documents'][0][i],
                        results['distances'][0][i]
                    ))
            return output
        
        elif self.backend == "qdrant":
            res = self.client.search(collection_name=self.collection_name, query_vector=query_emb, limit=k)
            return [(str(p.id), p.payload["text"], p.score) for p in res]
            
        elif self.backend == "pinecone":
            res = self.index.query(vector=query_emb, top_k=k, include_metadata=True)
            return [(m.id, m.metadata["text"], m.score) for m in res.matches]

        elif self.backend == "weaviate":
            res = (self.client.query.get("Document", ["text"])
                .with_near_vector({"vector": query_emb})
                .with_limit(k)
                .with_additional(["id", "distance"]).do())
            docs = res["data"]["Get"]["Document"]
            return [(d["_additional"]["id"], d["text"], d["_additional"]["distance"]) for d in docs]

        elif self.backend == "milvus":
            res = self.collection.search(data=[query_emb], anns_field="vector", param={"metric_type": "L2"}, limit=k, output_fields=["text"])
            output = []
            for hits in res:
                for hit in hits:
                    output.append((hit.id, hit.entity.get("text"), hit.distance))
            return output

        elif self.backend == "pgvector":
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT id, text, embedding <=> %s::vector AS distance FROM {self.collection_name} ORDER BY distance LIMIT %s", (query_emb, k))
                return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]

        elif self.backend == "elasticsearch":
            res = self.client.search(index=self.collection_name, body={
                "knn": {"field": "embedding", "query_vector": query_emb, "k": k, "num_candidates": 100},
                "_source": ["text"]
            })
            return [(hit["_id"], hit["_source"]["text"], hit["_score"]) for hit in res["hits"]["hits"]]

        elif self.backend == "faiss":
            D, I = self.index.search(np.array([query_emb]).astype('float32'), k)
            output = []
            for dist, idx in zip(D[0], I[0]):
                if idx != -1 and idx in self.doc_store:
                    doc_id, text, meta = self.doc_store[idx]
                    output.append((doc_id, text, float(dist)))
            return output

        elif self.backend == "memory":
            # Simple cosine similarity (unoptimized)
            scores = []
            for cid, text, emb, meta in self.storage:
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                scores.append((cid, text, 1 - sim)) # Return distance
            
            scores.sort(key=lambda x: x[2])
            return scores[:k]
            
        return []
    
    def delete_document(self, doc_id: str):
        """Delete all chunks of a document."""
        # Find all chunks
        results = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"[OK] Deleted document: {doc_id} ({len(results['ids'])} chunks)")
        else:
            print(f"Document not found: {doc_id}")
    
    def clear_all(self):
        """Clear all documents from memory."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(self.collection_name)
        print("[OK] Cleared all documents")
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        count = self.collection.count()
        
        # Estimate cost savings
        # Assuming average query retrieves 3 chunks vs. all chunks
        avg_chunk_size = 500  # characters
        avg_chunks_retrieved = 3
        cost_per_1k_tokens = 0.0001  # rough estimate
        
        total_size = count * avg_chunk_size
        smart_retrieval_size = avg_chunks_retrieved * avg_chunk_size
        
        cost_smart = (smart_retrieval_size / 1000) * cost_per_1k_tokens
        cost_dump_all = (total_size / 1000) * cost_per_1k_tokens
        
        return {
            "total_chunks": count,
            "estimated_total_size": total_size,
            "cost_per_smart_query": cost_smart,
            "cost_per_dump_all_query": cost_dump_all,
            "cost_savings_per_query": cost_dump_all - cost_smart,
            "monthly_savings_1k_queries": (cost_dump_all - cost_smart) * 1000
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("VECTOR MEMORY SYSTEM DEMO")
    print("=" * 70)
    print("\nDemonstrating Chapter 3 concept:")
    print("Smart retrieval vs. dumping everything into context\n")
    
    # Initialize
    memory = VectorMemory(collection_name="demo_memory")
    memory.clear_all()
    
    # Add sample documents
    documents = {
        "python_intro": """
        Python is a high-level programming language known for its simplicity
        and readability. It was created by Guido van Rossum and first released
        in 1991. Python supports multiple programming paradigms including
        procedural, object-oriented, and functional programming.
        """,
        "javascript_intro": """
        JavaScript is a programming language primarily used for web development.
        It allows developers to create interactive websites and is essential for
        front-end development. JavaScript can also be used on the server-side
        with Node.js.
        """,
        "database_intro": """
        A database is an organized collection of data stored and accessed
        electronically. SQL (Structured Query Language) is used to manage
        relational databases. PostgreSQL and MySQL are popular database systems.
        """,
        "ai_intro": """
        Artificial Intelligence (AI) is the simulation of human intelligence by
        machines. Machine learning is a subset of AI that allows systems to
        learn from data. Large Language Models like GPT-4 are examples of AI.
        """
    }
    
    print("  Adding documents to memory...")
    for doc_id, text in documents.items():
        memory.add_document(doc_id, text.strip())
    
    # Example searches
    queries = [
        "programming languages for web development",
        "how to store data",
        "what is machine learning"
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        results = memory.search(query, k=2)
        
        print(f"\n[CHART] Results:")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. Score: {1 - result['distance']:.3f}")
            print(f"     Text: {result['text'][:100]}...")
            print(f"     From: {result['metadata']['doc_id']}")
    
    # Show cost analysis
    print(f"\n{'='*70}")
    print("  COST ANALYSIS (Chapter 3)")
    print('='*70)
    
    stats = memory.get_stats()
    print(f"Total chunks in memory: {stats['total_chunks']}")
    print(f"Total estimated size: {stats['estimated_total_size']:,} characters")
    print()
    print("Per Query Cost:")
    print(f"  Smart retrieval (3 chunks): ${stats['cost_per_smart_query']:.6f}")
    print(f"  Dump all ({stats['total_chunks']} chunks): ${stats['cost_per_dump_all_query']:.6f}")
    print(f"  Savings per query: ${stats['cost_savings_per_query']:.6f}")
    print()
    print(f"For 1,000 queries/day (30 days):")
    print(f"  Smart: ${stats['cost_per_smart_query'] * 30000:.2f}/month")
    print(f"  Dump all: ${stats['cost_per_dump_all_query'] * 30000:.2f}/month")
    print(f"    Total savings: ${stats['monthly_savings_1k_queries'] * 30:.2f}/month")
    
    print("\n" + "="*70)
    print("[OK] Demo complete!")


if __name__ == "__main__":
    demo()
