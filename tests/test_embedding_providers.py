"""
Comprehensive tests for embedding providers.
Tests embedding factory and multiple providers.
"""

import pytest
from unittest.mock import MagicMock, patch
from kite.embedding_providers import EmbeddingFactory


class TestEmbeddingFactory:
    """Test embedding provider factory."""
    
    def test_embedding_factory_auto_detect(self):
        """Test embedding provider auto-detection."""
        provider = EmbeddingFactory.auto_detect()
        assert provider is not None
    
    def test_embedding_factory_create_sentence_transformers(self):
        """Test creating sentence-transformers provider."""
        provider = EmbeddingFactory.create(
            'sentence-transformers',
            model='all-MiniLM-L6-v2'
        )
        assert provider is not None
    
    def test_embedding_factory_create_fastembed(self):
        """Test creating FastEmbed provider."""
        provider = EmbeddingFactory.create(
            'fastembed',
            model='BAAI/bge-small-en-v1.5'
        )
        assert provider is not None
    
    @patch('openai.OpenAI')
    def test_embedding_factory_create_openai(self, mock_openai):
        """Test creating OpenAI embeddings."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = EmbeddingFactory.create(
                'openai',
                model='text-embedding-3-small'
            )
            assert provider is not None


class TestSentenceTransformersProvider:
    """Test sentence-transformers provider."""
    
    def test_sentence_transformers_embed(self):
        """Test embedding generation."""
        provider = EmbeddingFactory.create(
            'sentence-transformers',
            model='all-MiniLM-L6-v2'
        )
        
        embedding = provider.embed("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_sentence_transformers_batch(self):
        """Test batch embedding."""
        provider = EmbeddingFactory.create(
            'sentence-transformers',
            model='all-MiniLM-L6-v2'
        )
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = provider.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)


class TestFastEmbedProvider:
    """Test FastEmbed provider."""
    
    def test_fastembed_embed(self):
        """Test FastEmbed embedding generation."""
        provider = EmbeddingFactory.create(
            'fastembed',
            model='BAAI/bge-small-en-v1.5'
        )
        
        embedding = provider.embed("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
    
    def test_fastembed_batch(self):
        """Test FastEmbed batch embedding."""
        provider = EmbeddingFactory.create(
            'fastembed',
            model='BAAI/bge-small-en-v1.5'
        )
        
        texts = ["Text 1", "Text 2"]
        embeddings = provider.embed_batch(texts)
        
        assert len(embeddings) == 2


class TestOpenAIEmbeddings:
    """Test OpenAI embeddings (mocked)."""
    
    @patch('openai.OpenAI')
    def test_openai_embeddings_initialization(self, mock_openai):
        """Test OpenAI embeddings initialization."""
        from kite.embedding_providers import OpenAIEmbeddingProvider
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(model='text-embedding-3-small')
            assert provider.model == 'text-embedding-3-small'
    
    @patch('openai.OpenAI')
    def test_openai_embeddings_embed(self, mock_openai):
        """Test OpenAI embedding generation."""
        from kite.embedding_providers import OpenAIEmbeddingProvider
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIEmbeddingProvider(model='text-embedding-3-small')
            provider.client = mock_client
            
            embedding = provider.embed("Test")
            assert len(embedding) == 1536


class TestEmbeddingDimensions:
    """Test embedding dimension validation."""
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        provider = EmbeddingFactory.create(
            'sentence-transformers',
            model='all-MiniLM-L6-v2'
        )
        
        embedding = provider.embed("Test")
        
        # Verify dimension
        assert hasattr(provider, 'dimension') or len(embedding) > 0
        
        # All embeddings should have same dimension
        emb1 = provider.embed("Text 1")
        emb2 = provider.embed("Text 2")
        assert len(emb1) == len(emb2)


class TestBatchEmbedding:
    """Test batch embedding generation."""
    
    def test_batch_embedding_consistency(self):
        """Test batch vs individual embedding consistency."""
        provider = EmbeddingFactory.create(
            'sentence-transformers',
            model='all-MiniLM-L6-v2'
        )
        
        texts = ["Text 1", "Text 2"]
        
        # Batch embedding
        batch_embeddings = provider.embed_batch(texts)
        
        # Individual embeddings
        individual_embeddings = [provider.embed(t) for t in texts]
        
        # Should have same number
        assert len(batch_embeddings) == len(individual_embeddings)
        
        # Dimensions should match
        assert len(batch_embeddings[0]) == len(individual_embeddings[0])


class TestEmbeddingIntegration:
    """Integration tests for embeddings."""
    
    def test_embeddings_with_kite(self, ai):
        """Test embeddings integration with Kite."""
        embeddings = ai.embeddings
        assert embeddings is not None
    
    def test_embed_single_text(self, ai):
        """Test embedding single text through Kite."""
        embedding = ai.embed("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
    
    def test_embed_batch_texts(self, ai):
        """Test batch embedding through Kite."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = ai.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
