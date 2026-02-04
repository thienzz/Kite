import logging
from typing import Any, Dict, List, Optional
from ..agent import Agent
from ..core import Kite

class ReflectiveAgent(Agent):
    """
    An agent that implements the Reflection pattern.
    It generates an initial response, critiques it, and then refines it.
    """
    def __init__(self, 
                 name: str, 
                 system_prompt: str, 
                 tools: List = None,
                 framework: Optional[Kite] = None,
                 llm: Any = None,
                 critic_prompt: str = None,
                 max_reflections: int = 1,
                 verbose: bool = False):
        super().__init__(name, system_prompt, tools, framework, llm=llm, verbose=verbose)
        self.logger = logging.getLogger(name)
        self.critic_prompt = critic_prompt or (
            "You are a critical reviewer. Analyze the previous response for accuracy, "
            "completeness, and adherence to instructions. "
            "Identify specific flaws or missing information. "
            "If the response is perfect, simply say 'PERFECT'. "
            "Otherwise, provide a concise critique."
        )
        self.max_reflections = max_reflections

    async def run(self, input_text: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute the improved Agentic Reflection Loop:
        1. Generate initial response
        2. Critique (Reflection)
        3. Refine (Self-Correction) relative to critique
        """
        # 1. Initial Generation
        self.logger.info(f"[{self.name}] Generating initial response...")
        initial_result = await super().run(input_text, context)
        
        if not initial_result.get("success"):
            return initial_result

        current_response = initial_result.get("response")
        
        # 2. Reflection Loop
        for i in range(self.max_reflections):
            self.logger.info(f"[{self.name}] Reflection cycle {i+1}/{self.max_reflections}")
            
            # A. Critique
            critique_input = [
                {"role": "system", "content": self.critic_prompt},
                {"role": "user", "content": f"Original Request: {input_text}\n\nProposed Response: {current_response}"}
            ]
            
            # We use the same LLM for critique for now (Self-Reflection)
            # In advanced usage, this could be a stronger model.
            critique_response = self.llm.chat(critique_input)
            
            if "PERFECT" in critique_response.upper():
                self.logger.info(f"[{self.name}] Critique passed: Response is good.")
                break
                
            self.logger.info(f"[{self.name}] Critique: {critique_response}")
            
            # Emit reflection event
            if self.framework and self.framework.event_bus:
                self.framework.event_bus.emit(f"agent:{self.name}:reflection", {
                    "cycle": i+1,
                    "critique": critique_response
                })

            # B. Refine
            refinement_input = [
                {"role": "system", "content": self.system_prompt},
                {"role": "model", "content": current_response},
                {"role": "user", "content": f"Critique: {critique_response}\n\nPlease regenerate the response, addressing the critique above."}
            ]
            
            refined_response = self.llm.chat(refinement_input)
            current_response = refined_response
            
            self.logger.info(f"[{self.name}] Refined response generated.")

        return {
            "success": True,
            "response": current_response,
            "history": self.history # Returns the full thought process
        }
