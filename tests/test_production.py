import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import asyncio

# Ensure framework is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kite.core import Kite
from kite.memory.vector_memory import VectorMemory
from kite.memory.advanced_rag import AdvancedRAG
from kite.tools.mcp.database_mcp import DatabaseMCP
from kite.data_loaders import DocumentLoader
from kite.llm_providers import MockLLMProvider

class TestProductionFeatures(unittest.TestCase):
    @patch('kite.embedding_providers.EmbeddingFactory.auto_detect')
    def setUp(self, mock_auto_detect):
        # Mock the auto-detected embedding provider
        self.mock_embed_provider = MagicMock()
        self.mock_embed_provider.embed.side_effect = lambda doc: [0.1] * 384
        mock_auto_detect.return_value = self.mock_embed_provider
        
        self.mock_llm = MockLLMProvider()
        
        self.memory = VectorMemory(backend="memory")
        self.memory.embedding_provider = self.mock_embed_provider
        
        self.ai = Kite(config={"llm_provider": "mock", "vector_backend": "memory"})
        self.ai._llm = self.mock_llm
        self.ai.vector_memory.embedding_provider = self.mock_embed_provider

    def test_llm_streaming(self):
        """Test if streaming methods exist and return generators."""
        import asyncio
        async def run_test():
            gen = self.mock_llm.stream_complete("test")
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)
            self.assertTrue(len(chunks) > 0)
        
        asyncio.run(run_test())

    def test_vector_backends_init(self):
        """Test initialization of various vector backends (mocked)."""
        backends = ["chroma", "faiss", "memory"]
        for b in backends:
            v = VectorMemory(backend=b)
            self.assertEqual(v.backend, b)

    def test_advanced_rag_hybrid(self):
        """Test hybrid search logic."""
        rag = AdvancedRAG(self.memory, self.mock_llm)
        results = rag.search("test query", strategy="hybrid")
        self.assertIsInstance(results, list)

    def test_database_safe_execute(self):
        """Test the safety decorator in DatabaseMCP."""
        mcp = DatabaseMCP()
        # This should return an error dict instead of raising Exception if driver missing
        res = asyncio.run(mcp.query_neo4j("MATCH (n) RETURN n"))
        self.assertIn("error", res)

    def test_document_loader(self):
        """Test document loader extension handling."""
        loader = DocumentLoader()
        # Mock a file content
        with patch("builtins.open", unittest.mock.mock_open(read_data="test content")):
            text = loader.load_any("test.txt")
            self.assertEqual(text, "test content")

    def test_aggregator_router_integration(self):
        """Test router with mock agents."""
        from kite.routing.aggregator_router import AggregatorRouter
        router = AggregatorRouter(llm=self.mock_llm)
        # Test decomposition (mocked response from LLM would be needed for real test)
        # For now just verify it initializes
        self.assertTrue(len(router.agents) >= 3)

if __name__ == '__main__':
    unittest.main()
