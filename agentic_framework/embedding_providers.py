"""
Embedding Provider Abstraction Layer
PRIORITY: Opensource, local-first solutions.

Providers:
- sentence-transformers (Local, Free) - PRIORITY
- FastEmbed (Local, Free, Fast)
- Ollama (Local, Free)
- OpenAI (Commercial)
"""

from typing import List, Optional
from abc import ABC, abstractmethod
import logging
import os


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


# ============================================================================
# LOCAL / OPENSOURCE (PRIORITY)
# ============================================================================

class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    sentence-transformers - Local, FREE, OPENSOURCE
    
    Best models:
    - all-MiniLM-L6-v2: Fast, 384d (Default)
    - all-mpnet-base-v2: Good quality, 768d
    - multi-qa-MiniLM-L6-cos-v1: Q&A optimized
    
    Installation: pip install sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.logger = logging.getLogger("SentenceTransformers")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"[OK] Loaded {model_name} ({self._dimension}d)")
        except ImportError:
            raise ImportError("pip install sentence-transformers")
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embeddings."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        ).tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def name(self) -> str:
        return f"SentenceTransformers/{self.model_name}"


class FastEmbedProvider(BaseEmbeddingProvider):
    """
    FastEmbed - Faster than sentence-transformers
    
    Installation: pip install fastembed
    Speed: 2-3x faster, same quality
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.logger = logging.getLogger("FastEmbed")
        
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name)
            self._dimension = 384  # Most models are 384d
            self.logger.info(f"[OK] FastEmbed loaded: {model_name}")
        except ImportError:
            raise ImportError("pip install fastembed")
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        return list(self.model.embed([text]))[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embeddings."""
        return [emb.tolist() for emb in self.model.embed(texts)]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def name(self) -> str:
        return f"FastEmbed/{self.model_name}"


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama Embeddings - Local, FREE
    
    Models: nomic-embed-text, mxbai-embed-large
    Same API as Ollama LLMs
    """
    
    def __init__(self, 
                 model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.logger = logging.getLogger("OllamaEmbed")
        
        try:
            import requests
            self.requests = requests
            self._dimension = self._get_dimension()
            self.logger.info(f"[OK] Ollama embed: {model}")
        except ImportError:
            raise ImportError("pip install requests")
    
    def _get_dimension(self) -> int:
        """Get embedding dimension."""
        # Test with empty string
        test_embed = self.embed("")
        return len(test_embed)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        response = self.requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )
        return response.json()["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embeddings."""
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"


# ============================================================================
# COMMERCIAL (Fallback)
# ============================================================================

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI Embeddings (PAID)
    
    Models: text-embedding-3-small (1536d), text-embedding-3-large (3072d)
    """
    
    def __init__(self, 
                 model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.logger = logging.getLogger("OpenAIEmbed")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self._dimension = 1536 if "small" in model else 3072
            self.logger.info(f"[OK] OpenAI embed: {model}")
        except ImportError:
            raise ImportError("pip install openai")
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embeddings."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [d.embedding for d in response.data]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def name(self) -> str:
        return f"OpenAI/{self.model}"


# ============================================================================
# FACTORY
# ============================================================================

class EmbeddingFactory:
    """
    Factory for creating embedding providers.
    
    Priority:
    1. sentence-transformers (Local, Free)
    2. FastEmbed (Local, Faster)
    3. Ollama (Local, Free)
    4. OpenAI (Commercial)
    """
    
    PROVIDERS = {
        'sentence-transformers': SentenceTransformersProvider,
        'fastembed': FastEmbedProvider,
        'ollama': OllamaEmbeddingProvider,
        'openai': OpenAIEmbeddingProvider,
    }
    
    @classmethod
    def create(cls, 
               provider: str = "sentence-transformers",
               model: Optional[str] = None,
               **kwargs) -> BaseEmbeddingProvider:
        """Create embedding provider."""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        provider_class = cls.PROVIDERS[provider]
        
        if model:
            if provider == "sentence-transformers":
                return provider_class(model_name=model, **kwargs)
            else:
                return provider_class(model=model, **kwargs)
        else:
            return provider_class(**kwargs)
    
    @classmethod
    def auto_detect(cls) -> BaseEmbeddingProvider:
        """
        Auto-detect best embedding provider.
        
        Priority:
        1. sentence-transformers (best for most use cases)
        2. FastEmbed (if speed is critical)
        3. Ollama (if already running)
        4. OpenAI (fallback)
        """
        logger = logging.getLogger("EmbeddingFactory")
        
        # Try sentence-transformers first
        try:
            provider = cls.create("sentence-transformers")
            logger.info("[OK] Using sentence-transformers (local, free)")
            return provider
        except:
            pass
        
        # Try FastEmbed
        try:
            provider = cls.create("fastembed")
            logger.info("[OK] Using FastEmbed (local, free, fast)")
            return provider
        except:
            pass
        
        # Try Ollama
        try:
            provider = cls.create("ollama")
            logger.info("[OK] Using Ollama embeddings (local, free)")
            return provider
        except:
            pass
        
        # Fallback to OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                provider = cls.create("openai")
                logger.warning("  Using OpenAI embeddings (paid)")
                return provider
            except:
                pass
        
        raise RuntimeError(
            "No embedding provider available. "
            "Install: pip install sentence-transformers"
        )


# ============================================================================
# RECOMMENDED MODELS
# ============================================================================

RECOMMENDED_MODELS = {
    "sentence-transformers": {
        "fast": "all-MiniLM-L6-v2",  # 384d, fastest
        "balanced": "all-mpnet-base-v2",  # 768d, good quality
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
        "qa": "multi-qa-MiniLM-L6-cos-v1",  # Q&A optimized
    },
    "fastembed": {
        "fast": "BAAI/bge-small-en-v1.5",
        "balanced": "BAAI/bge-base-en-v1.5",
    },
    "ollama": {
        "default": "nomic-embed-text",
        "large": "mxbai-embed-large",
    }
}


if __name__ == "__main__":
    print("Embedding Provider Examples:\n")
    
    # Auto-detect
    print("1. Auto-detect:")
    embed = EmbeddingFactory.auto_detect()
    print(f"   Using: {embed.name}")
    print(f"   Dimension: {embed.dimension}\n")
    
    # Test embedding
    text = "This is a test sentence"
    vector = embed.embed(text)
    print(f"   Test: '{text}'")
    print(f"   Vector: [{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-1]:.4f}]")
    print(f"   Length: {len(vector)}\n")
    
    print("2. Recommended models:")
    for provider, models in RECOMMENDED_MODELS.items():
        print(f"\n   {provider}:")
        for use_case, model in models.items():
            print(f"   - {use_case}: {model}")
