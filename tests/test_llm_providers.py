"""
Comprehensive tests for LLM provider system.
Tests provider factory, multiple providers, and error handling.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from kite.llm_providers import MockLLMProvider, OpenAIProvider, AnthropicProvider, GroqProvider, OllamaProvider


class TestMockProvider:
    """Test mock LLM provider."""
    
    def test_mock_provider_complete(self):
        """Test mock provider completion."""
        provider = MockLLMProvider()
        result = provider.complete("Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_mock_provider_chat(self):
        """Test mock provider chat."""
        provider = MockLLMProvider()
        messages = [{"role": "user", "content": "Hello"}]
        result = provider.chat(messages)
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_mock_provider_streaming(self):
        """Test mock provider streaming."""
        provider = MockLLMProvider()
        
        chunks = []
        async for chunk in provider.stream_complete("Test"):
            chunks.append(chunk)
        
        assert len(chunks) > 0


class TestOpenAIProvider:
    """Test OpenAI provider (mocked)."""
    
    @patch('openai.OpenAI')
    def test_openai_provider_initialization(self, mock_openai):
        """Test OpenAI provider initialization."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            provider = OpenAIProvider(model='gpt-4')
            assert provider.model == 'gpt-4'
    
    @patch('openai.OpenAI')
    def test_openai_provider_complete(self, mock_openai):
        """Test OpenAI completion."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            provider = OpenAIProvider(model='gpt-4')
            result = provider.complete("Test")
            assert result == "Response"


class TestAnthropicProvider:
    """Test Anthropic provider (mocked)."""
    
    @patch('anthropic.Anthropic')
    def test_anthropic_provider_initialization(self, mock_anthropic):
        """Test Anthropic provider initialization."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = AnthropicProvider(model='claude-3-5-sonnet-20241022')
            assert provider.model == 'claude-3-5-sonnet-20241022'


class TestGroqProvider:
    """Test Groq provider (mocked)."""
    
    @patch('groq.Groq')
    def test_groq_provider_initialization(self, mock_groq):
        """Test Groq provider initialization."""
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            provider = GroqProvider(model='llama-3.3-70b-versatile')
            assert provider.model == 'llama-3.3-70b-versatile'


class TestOllamaProvider:
    """Test Ollama provider (mocked)."""
    
    @patch('httpx.Client')
    def test_ollama_provider_initialization(self, mock_client):
        """Test Ollama provider initialization."""
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_instance.get.return_value = mock_response
        mock_client.return_value = mock_instance
        
        provider = OllamaProvider(model='qwen2.5:1.5b')
        assert provider.model == 'qwen2.5:1.5b'


class TestProviderSwitching:
    """Test switching between providers."""
    
    def test_switch_providers(self, ai):
        """Test switching between different providers."""
        original_provider = ai.config.get('llm_provider')
        
        # Switch to mock provider
        ai.config['llm_provider'] = 'mock'
        ai._llm = None  # Reset cached provider
        
        provider = ai.llm
        assert provider is not None
        
        # Restore original
        ai.config['llm_provider'] = original_provider


class TestProviderErrorHandling:
    """Test API error scenarios."""
    
    @patch('openai.OpenAI')
    def test_api_error_handling(self, mock_openai):
        """Test handling API errors."""
        from kite.llm_providers import OpenAIProvider
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = OpenAIProvider(model='gpt-4')
            provider.client = mock_client
            
            with pytest.raises(Exception):
                provider.complete("Test")
    
    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(Exception):
                from kite.llm_providers import OpenAIProvider
                provider = OpenAIProvider(model='gpt-4')


class TestStreamingSupport:
    """Test streaming completions."""
    
    @pytest.mark.asyncio
    async def test_streaming_support(self):
        """Test streaming completions."""
        provider = MockLLMProvider()
        
        chunks = []
        async for chunk in provider.stream_complete("Test prompt"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
