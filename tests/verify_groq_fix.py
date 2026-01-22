import asyncio
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kite.llm_providers import GroqProvider

class MockRateLimitError(Exception):
    def __init__(self, message, response=None, body=None):
        super().__init__(message)
        self.response = response
        self.body = body

class TestGroqRetry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        os.environ["GROQ_API_KEY"] = "fake-key"
        # Mock the groq library
        self.mock_groq = MagicMock()
        self.mock_groq.RateLimitError = MockRateLimitError
        sys.modules["groq"] = self.mock_groq
        
        # Create Provider
        with patch('kite.llm_providers.RateLimiter') as mock_limiter:
            limiter_instance = mock_limiter.return_value
            limiter_instance.consume.return_value = True
            limiter_instance.get_status.return_value = {
                'remaining_requests': 10.0,
                'remaining_tokens': 10000.0,
                'rpm_limit': 30,
                'tpm_limit': 14000
            }
            # Properly mock async method
            async def mock_wait_async(*args, **kwargs):
                return
            limiter_instance.wait_async = mock_wait_async
            
            self.provider = GroqProvider(model="test-model")
            self.provider.rate_limiter = limiter_instance

    async def test_chat_sync_retry(self):
        """Verify synchronous chat retries on RateLimitError."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"
        
        # Setup mock to fail twice then succeed
        self.provider.client.chat.completions.create.side_effect = [
            MockRateLimitError("Rate limit", response=MagicMock(), body={}),
            MockRateLimitError("Rate limit", response=MagicMock(), body={}),
            mock_response
        ]
        
        with patch('time.sleep'): # Don't actually sleep
            result = self.provider.chat([{"role": "user", "content": "hello"}])
            
        self.assertEqual(result, "Success")
        self.assertEqual(self.provider.client.chat.completions.create.call_count, 3)
        print("\n[OK] Sync chat retry verified.")

    async def test_chat_async_retry(self):
        """Verify asynchronous chat retries on RateLimitError."""
        from unittest.mock import AsyncMock
        
        # Mock the parse() coroutine
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success Async"
        
        # Mock with_raw_response.create as an AsyncMock
        mock_raw_response = MagicMock()
        mock_raw_response.parse = AsyncMock(return_value=mock_response)
        
        # Setup mock to fail twice then succeed
        self.provider.async_client.chat.completions.with_raw_response.create = AsyncMock()
        self.provider.async_client.chat.completions.with_raw_response.create.side_effect = [
            MockRateLimitError("Rate limit", response=MagicMock(), body={}),
            MockRateLimitError("Rate limit", response=MagicMock(), body={}),
            mock_raw_response
        ]
        
        with patch('asyncio.sleep', return_value=None): # Don't actually sleep
            result = await self.provider.chat_async([{"role": "user", "content": "hello"}])
            
        self.assertEqual(result, "Success Async")
        self.assertEqual(self.provider.async_client.chat.completions.with_raw_response.create.call_count, 3)
        print("[OK] Async chat retry verified.")

if __name__ == "__main__":
    unittest.main()
