"""
Resource-Aware Optimization (Chapter 16)
Dynamically selects the optimal model (resource) based on task complexity.
"""

from typing import Dict, Any, Optional
import os

class ResourceAwareRouter:
    """
    Routes queries to the most cost-effective model.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Priority: Config > Env Var > Default Placeholder
        # Priority: Config > Env Var (Split) > Env Var (Legacy)
        
        # 1. Fast Model Resolution
        fast_provider = config.get("fast_llm_provider") or os.getenv("FAST_LLM_PROVIDER")
        fast_model_name = config.get("fast_llm_model") or os.getenv("FAST_LLM_MODEL")
        
        if fast_provider and fast_model_name:
            self.fast_model = f"{fast_provider}/{fast_model_name}"
        else:
            # Legacy fallback: check for full string in FAST_LLM_MODEL (e.g. "groq/llama...")
            self.fast_model = config.get("fast_model") or os.getenv("FAST_LLM_MODEL")
        self.smart_model = config.get("smart_model") or os.getenv("SMART_LLM_MODEL")
        
        # Fallback to defaults only if absolutely necessary, but log warning
        if not self.fast_model:
            raise ValueError("Configuration Error: 'fast_model' not found. Set FAST_LLM_MODEL env var or pass in config.")
            
        if not self.smart_model:
            # Fallback to main LLM from env if strictly necessary (User request)
            main_provider = os.getenv("LLM_PROVIDER")
            main_model = os.getenv("LLM_MODEL")
            if main_provider and main_model:
                self.smart_model = f"{main_provider}/{main_model}"
            
        if not self.smart_model:
            raise ValueError("Configuration Error: 'smart_model' not found. Set SMART_LLM_MODEL or LLM_PROVIDER/LLM_MODEL.")
            
        # Simple heuristic threshold (word count)
        self.complexity_threshold = config.get("complexity_threshold", 20)

    def select_model(self, query: str) -> str:
        """
        Selects a model based on query complexity.
        This is a simple implementation of 'Dynamic Model Switching'.
        """
        # 1. Check length
        word_count = len(query.split())
        
        if word_count < self.complexity_threshold:
            print(f"   [Optimization] Routing to FAST model ({self.fast_model}) for simple query.")
            return self.fast_model
            
        # 2. Check for complexity keywords
        complex_terms = ["analyze", "reason", "plan", "code", "compare", "evaluate"]
        if any(term in query.lower() for term in complex_terms):
            print(f"   [Optimization] Routing to SMART model ({self.smart_model}) for reasoning task.")
            return self.smart_model
            
        # Default to fast for everything else
        print(f"   [Optimization] Defaulting to FAST model.")
        return self.fast_model

    async def route(self, query: str, framework) -> Dict[str, Any]:
        """
        Executes the query using the selected model.
        """
        selected_model = self.select_model(query)
        
        # In a real system, we would instantiate a temporary agent or use the LLM directly.
        # Here we simulate the selection impacting the framework's execution.
        
        # TODO: This method signature might need adjustment to integrate deeply with Kite.
        # For this demo, it returns the model name for the agent to use.
        return {"model": selected_model}
