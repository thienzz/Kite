"""
LLM Provider Abstraction Layer
Supports multiple LLM providers with priority on open source.

Supported Providers:
- Ollama (Local, Free) - PRIORITY
- LM Studio (Local, Free)
- vLLM (Local, Free)
- Anthropic Claude
- OpenAI GPT
- Google Gemini
- Mistral AI
- Groq (Fast inference)
- Together AI
- Replicate
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import os
import logging
import json
import asyncio


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings."""
        pass
    
    @abstractmethod
    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        pass
        
    @abstractmethod
    def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    async def complete_async(self, prompt: str, **kwargs) -> str:
        """Async version of complete."""
        import asyncio
        return await asyncio.to_thread(self.complete, prompt, **kwargs)
        
    async def chat_async(self, messages: List[Dict], **kwargs) -> str:
        """Async version of chat."""
        import asyncio
        return await asyncio.to_thread(self.chat, messages, **kwargs)


# ============================================================================
# LOCAL / OPENSOURCE PROVIDERS (PRIORITY)
# ============================================================================

class OllamaProvider(BaseLLMProvider):
    """
    Ollama - Run LLMs locally (FREE, OPENSOURCE)
    
    Models: llama3, mistral, codellama, phi, gemma, etc.
    Installation: curl -fsSL https://ollama.com/install.sh | sh
    """
    
    def __init__(self, 
                 model: str = "llama3",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 600.0,
                 **kwargs):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger("Ollama")
        
        try:
            import httpx
            self._test_connection()
        except ImportError:
            raise ImportError("pip install httpx")
    
    def _test_connection(self):
        """Test Ollama connection."""
        import httpx
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    self.logger.info(f"[OK] Ollama connected: {self.model}")
                else:
                    raise ConnectionError(f"Ollama returned {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Ollama not running at {self.base_url}: {e}")
    
    def _sanitize_ollama_params(self, kwargs: Dict) -> Dict:
        """Helper to ensure only valid Ollama parameters are sent."""
        valid_top_level = {'model', 'prompt', 'messages', 'stream', 'format', 'options', 'keep_alive'}
        # Common model options that should go into 'options' dict
        valid_options = {
            'num_keep', 'seed', 'num_predict', 'top_k', 'top_p', 'tfs_z', 
            'typical_p', 'repeat_last_n', 'temperature', 'repeat_penalty', 
            'presence_penalty', 'frequency_penalty', 'mixtral_mi', 'mixtral_m', 
            'mixtral_s', 'num_ctx', 'num_batch', 'num_gqa', 'num_gpu', 
            'main_gpu', 'low_vram', 'f16_kv', 'logits_all', 'vocab_only', 
            'use_mmap', 'use_mlock', 'num_thread'
        }
        
        sanitized = {}
        options = kwargs.get('options', {})
        
        for k, v in kwargs.items():
            if k in valid_top_level:
                sanitized[k] = v
            elif k in valid_options:
                options[k] = v
                
        if options:
            sanitized['options'] = options
        return sanitized

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion (sync)."""
        import httpx
        import threading
        
        params = self._sanitize_ollama_params({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        })
        
        # Simple heartbeat thread
        stop_heartbeat = threading.Event()
        def heartbeat():
            start = time.time()
            while not stop_heartbeat.wait(30):
                self.logger.info(f"Ollama is still thinking... ({int(time.time() - start)}s elapsed)")
        
        h_thread = threading.Thread(target=heartbeat, daemon=True)
        h_thread.start()
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=params
                )
                data = response.json()
                if "response" in data:
                    return data["response"]
                elif "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                raise KeyError(f"Unexpected Ollama response format: {data}")
        finally:
            stop_heartbeat.set()

    async def complete_async(self, prompt: str, **kwargs) -> str:
        """Native async complete."""
        import httpx
        
        params = self._sanitize_ollama_params({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        })
        
        # Async heartbeat
        async def heartbeat(task):
            start = time.time()
            while not task.done():
                await asyncio.sleep(30)
                if not task.done():
                    self.logger.info(f"Ollama is still thinking... ({int(time.time() - start)}s elapsed)")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            request_task = asyncio.create_task(client.post(
                f"{self.base_url}/api/generate",
                json=params
            ))
            heartbeat_task = asyncio.create_task(heartbeat(request_task))
            
            try:
                response = await request_task
                data = response.json()
                if "response" in data:
                    return data["response"]
                elif "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                raise KeyError(f"Unexpected Ollama response format: {data}")
            finally:
                heartbeat_task.cancel()
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion (sync)."""
        import httpx
        import threading
        
        params = self._sanitize_ollama_params({
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs
        })
        
        stop_heartbeat = threading.Event()
        def heartbeat():
            start = time.time()
            while not stop_heartbeat.wait(30):
                self.logger.info(f"Ollama is still thinking... ({int(time.time() - start)}s elapsed)")
        
        h_thread = threading.Thread(target=heartbeat, daemon=True)
        h_thread.start()
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=params
                )
                if response.status_code != 200:
                    self.logger.error(f"Ollama Chat Error {response.status_code}: {response.text}")
                
                data = response.json()
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                elif "response" in data:
                    return data["response"]
                raise KeyError(f"Unexpected Ollama chat response format: {data}")
        finally:
            stop_heartbeat.set()

    async def chat_async(self, messages: List[Dict], **kwargs) -> str:
        """Native async chat."""
        import httpx
        
        params = self._sanitize_ollama_params({
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs
        })
        
        async def heartbeat(task):
            start = time.time()
            while not task.done():
                await asyncio.sleep(30)
                if not task.done():
                    self.logger.info(f"Ollama is still thinking... ({int(time.time() - start)}s elapsed)")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            request_task = asyncio.create_task(client.post(
                f"{self.base_url}/api/chat",
                json=params
            ))
            heartbeat_task = asyncio.create_task(heartbeat(request_task))
            
            try:
                response = await request_task
                if response.status_code != 200:
                    self.logger.error(f"Ollama Chat Async Error {response.status_code}: {response.text}")
                    
                data = response.json()
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                elif "response" in data:
                    return data["response"]
                raise KeyError(f"Unexpected Ollama chat response format: {data}")
            finally:
                heartbeat_task.cancel()
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings."""
        response = self.requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )
        return response.json()["embedding"]
    
    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                **kwargs
            },
            stream=True
        )
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break

    def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        response = self.requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                **kwargs
            },
            stream=True
        )
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    break
    
    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio - Local LLM with GUI (FREE, OPENSOURCE)
    
    Download: https://lmstudio.ai/
    Compatible with OpenAI API format.
    """
    
    def __init__(self, 
                 model: str = "local-model",
                 base_url: str = "http://localhost:1234/v1",
                 **kwargs):
        self.model = model
        self.base_url = base_url
        self.logger = logging.getLogger("LMStudio")
        
        try:
            import openai
            import requests
            self.requests = requests
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key="lm-studio"
            )
            self._test_connection()
            self.logger.info(f"[OK] LM Studio connected: {model}")
        except Exception as e:
            raise ConnectionError(f"LM Studio not found: {e}")
            
    def _test_connection(self):
        """Test LM Studio connection."""
        try:
            # LM Studio usually has a /v1/models endpoint
            response = self.requests.get(f"{self.base_url}/models", timeout=1)
            if response.status_code != 200:
                raise ConnectionError(f"LM Studio returned {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"LM Studio not running at {self.base_url}: {e}")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **kwargs
        )
        return response.choices[0].text
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        stream = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text

    async def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @property
    def name(self) -> str:
        return f"LMStudio/{self.model}"


class VLLMProvider(BaseLLMProvider):
    """
    vLLM - Fast inference server (FREE, OPENSOURCE)
    
    Installation: pip install vllm
    Best for: Production serving of open models
    """
    
    def __init__(self, 
                 model: str = "meta-llama/Llama-2-7b-hf",
                 base_url: str = "http://localhost:8000",
                 **kwargs):
        self.model = model
        self.base_url = base_url
        self.logger = logging.getLogger("vLLM")
        
        try:
            import requests
            self.requests = requests
            self._test_connection()
            self.logger.info(f"[OK] vLLM connected: {model}")
        except ImportError:
            raise ImportError("pip install requests")
        except Exception as e:
            raise ConnectionError(f"vLLM server not found at {base_url}: {e}")
            
    def _test_connection(self):
        """Test vLLM connection (OpenAI compatible or direct)."""
        try:
            # Check models endpoint (OpenAI compatible)
            response = self.requests.get(f"{self.base_url}/models", timeout=1)
            if response.status_code != 200:
                # Try direct generate endpoint as fallback
                response = self.requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code != 200:
                    raise ConnectionError(f"vLLM returned {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"vLLM not running at {self.base_url}")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        response = self.requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                **kwargs
            }
        )
        return response.json()["text"][0]
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        # Convert to single prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.complete(prompt, **kwargs)
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        response = self.requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "stream": True,
                "max_tokens": kwargs.get("max_tokens", 512),
                **kwargs
            },
            stream=True
        )
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("text", [""])[0]

    async def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        # Convert to single prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        async for chunk in self.stream_complete(prompt, **kwargs):
            yield chunk
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings (fallback to local)."""
        from .embedding_providers import EmbeddingFactory
        if not hasattr(self, '_embedding_fallback'):
            self._embedding_fallback = EmbeddingFactory.auto_detect()
        return self._embedding_fallback.embed(text)
    
    @property
    def name(self) -> str:
        return f"vLLM/{self.model}"


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM Provider for testing."""
    
    def __init__(self, model: str = "mock-model", **kwargs):
        self.model = model
    
    def complete(self, prompt: str, **kwargs) -> str:
        return f"Mock response to: {prompt[:50]}..."
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        return "Mock chat response: I'm here to help with your agentic tasks!"
    
    def embed(self, text: str) -> List[float]:
        import random
        return [random.random() for _ in range(1536)]
    
    async def stream_complete(self, prompt: str, **kwargs):
        yield f"Mock stream response to: {prompt[:20]}..."
        
    async def stream_chat(self, messages: List[Dict], **kwargs):
        yield "Mock chat stream: "
        yield "I'm "
        yield "helping!"
    
    @property
    def name(self) -> str:
        return "Mock/LLM"


# ============================================================================
# CLOUD OPENSOURCE PROVIDERS
# ============================================================================

class GroqProvider(BaseLLMProvider):
    """
    Groq - Ultra-fast inference (FREE tier, OPENSOURCE models)
    
    Models: llama3-70b, mixtral-8x7b, gemma-7b
    Speed: 500+ tokens/sec
    """
    
    def __init__(self, 
                 model: str = "llama3-70b-8192",
                 api_key: Optional[str] = None,
                 **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.logger = logging.getLogger("Groq")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.logger.info(f"[OK] Groq connected: {model}")
        except ImportError:
            raise ImportError("pip install groq")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings (fallback)."""
        from .embedding_providers import EmbeddingFactory
        if not hasattr(self, '_embedding_fallback'):
            self._embedding_fallback = EmbeddingFactory.auto_detect()
        return self._embedding_fallback.embed(text)
    
    @property
    def name(self) -> str:
        return f"Groq/{self.model}"


class TogetherProvider(BaseLLMProvider):
    """
    Together AI - Opensource models (PAID, but cheap)
    
    Models: llama-2, mistral, mixtral, etc.
    """
    
    def __init__(self, 
                 model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 api_key: Optional[str] = None,
                 **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.logger = logging.getLogger("Together")
        
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY required")
        
        try:
            import together
            together.api_key = self.api_key
            self.together = together
            self.logger.info(f"[OK] Together AI connected: {model}")
        except ImportError:
            raise ImportError("pip install together")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion."""
        response = self.together.Complete.create(
            prompt=prompt,
            model=self.model,
            **kwargs
        )
        return response['output']['choices'][0]['text']
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.complete(prompt, **kwargs)
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings."""
        response = self.together.Embeddings.create(
            input=text,
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        return response['data'][0]['embedding']
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        # Together uses a slightly different API for streaming depending on the model,
        # but the inference client usually supports it.
        stream = self.together.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk.choices[0].text

    async def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        stream = self.together.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content
    
    @property
    def name(self) -> str:
        return f"Together/{self.model}"


# ============================================================================
# COMMERCIAL PROVIDERS (Secondary)
# ============================================================================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude (PAID)."""
    
    def __init__(self, 
                 model: str = "claude-3-sonnet-20240229",
                 api_key: Optional[str] = None,
                 **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
    
    def complete(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024)
        )
        return response.content[0].text
    
    async def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        async for chunk in self.stream_chat([{"role": "user", "content": prompt}], **kwargs):
            yield chunk

    async def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        with self.client.messages.stream(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024)
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings (fallback)."""
        from .embedding_providers import EmbeddingFactory
        if not hasattr(self, '_embedding_fallback'):
            self._embedding_fallback = EmbeddingFactory.auto_detect()
        return self._embedding_fallback.embed(text)
    
    @property
    def name(self) -> str:
        return f"Anthropic/{self.model}"

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT (PAID)."""
    
    def __init__(self, 
                 model: str = "gpt-4-turbo-preview",
                 api_key: Optional[str] = None,
                 **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self._test_connection()
        except Exception as e:
            raise ConnectionError(f"OpenAI failed: {e}")
            
    def _test_connection(self):
        """Test OpenAI connection (auth check)."""
        if not self.api_key or self.api_key.startswith("sk-..."):
             raise ValueError("OpenAI API key is invalid or placeholder")
        try:
            # Simple list models call to check auth
            self.client.models.list()
        except Exception as e:
            raise ConnectionError(f"OpenAI auth failed: {e}")
    
    async def chat_async(self, messages: List[Dict], **kwargs) -> str:
        """True async chat for OpenAI."""
        if not hasattr(self, 'async_client'):
            import openai
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
        
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    async def complete_async(self, prompt: str, **kwargs) -> str:
        """True async complete for OpenAI."""
        return await self.chat_async([{"role": "user", "content": prompt}], **kwargs)
    
    def complete(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
        
    def stream_complete(self, prompt: str, **kwargs):
        """Stream completion."""
        stream = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].text:
                yield chunk.choices[0].text

    def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @property
    def name(self) -> str:
        return f"OpenAI/{self.model}"


# ============================================================================
# PROVIDER FACTORY
# ============================================================================

class LLMFactory:
    """
    Factory for creating LLM providers.
    
    Priority order:
    1. Local/Free (Ollama, LM Studio, vLLM)
    2. Cloud Free Tier (Groq)
    3. Cloud Opensource (Together)
    4. Commercial (Claude, GPT)
    """
    
    PROVIDERS = {
        # Local (Priority)
        'ollama': OllamaProvider,
        'lmstudio': LMStudioProvider,
        'vllm': VLLMProvider,
        
        # Cloud Opensource
        'groq': GroqProvider,
        'together': TogetherProvider,
        
        # Commercial
        'anthropic': AnthropicProvider,
        'openai': OpenAIProvider,
        
        # Testing
        'mock': MockLLMProvider,
    }
    
    @classmethod
    def create(cls, 
               provider: str = "ollama",
               model: Optional[str] = None,
               **kwargs) -> BaseLLMProvider:
        """
        Create LLM provider.
        
        Args:
            provider: Provider name
            model: Model name (optional, uses default)
            **kwargs: Provider-specific kwargs
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. "
                           f"Available: {list(cls.PROVIDERS.keys())}")
        
        provider_class = cls.PROVIDERS[provider]
        
        # Create with model if specified
        if model:
            return provider_class(model=model, **kwargs)
        else:
            return provider_class(**kwargs)
    
    @classmethod
    def auto_detect(cls) -> BaseLLMProvider:
        """
        Auto-detect best available provider.
        
        Priority:
        1. Try Ollama (local)
        2. Try LM Studio (local)
        3. Try Groq (cloud, free)
        4. Try OpenAI (fallback)
        """
        logger = logging.getLogger("LLMFactory")
        
        # Try Ollama
        try:
            provider = cls.create("ollama")
            logger.info("[OK] Using Ollama (local, free)")
            return provider
        except:
            pass
        
        # Try LM Studio
        try:
            provider = cls.create("lmstudio")
            logger.info("[OK] Using LM Studio (local, free)")
            return provider
        except:
            pass
        
        # Try Groq
        if os.getenv("GROQ_API_KEY"):
            try:
                provider = cls.create("groq")
                logger.info("[OK] Using Groq (cloud, free tier)")
                return provider
            except:
                pass
        
        # Fallback to OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                provider = cls.create("openai")
                logger.warning("  Using OpenAI (commercial, paid)")
                return provider
            except:
                pass
        
        # Try mock as ultimate fallback
        try:
            return cls.create("mock")
        except:
            pass
            
        raise RuntimeError(
            "No LLM provider available. Install Ollama or set API keys."
        )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("LLM Provider Examples:\n")
    
    # Auto-detect
    print("1. Auto-detect:")
    llm = LLMFactory.auto_detect()
    print(f"   Using: {llm.name}\n")
    
    # Specific providers
    print("2. Specific providers:")
    
    providers = [
        ("ollama", "llama3"),
        ("groq", "llama3-70b-8192"),
        ("together", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    ]
    
    for provider, model in providers:
        try:
            llm = LLMFactory.create(provider, model)
            print(f"   [OK] {llm.name}")
        except Exception as e:
            print(f"     {provider}: {e}")
