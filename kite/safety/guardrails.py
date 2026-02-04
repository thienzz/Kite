"""
Guardrails / Safety Patterns (Chapter 18)
Provides mechanisms to ensure agent inputs and outputs adhere to safety and structure guidelines.
"""

from typing import Type, Optional, Any, Dict
from pydantic import BaseModel, ValidationError, Field
import re
import logging

logger = logging.getLogger("Guardrails")

class OutputGuardrail:
    """
    Enforces structured output from an LLM using Pydantic models.
    Chapter 18: Output Filtering / Post-processing.
    """
    def __init__(self, model: Type[BaseModel], fix_on_failure: bool = True):
        self.model = model
        self.fix_on_failure = fix_on_failure

    def validate(self, output: str) -> Optional[BaseModel]:
        """
        Validates and parses the LLM output. Uses regex to find the first JSON object.
        """
        try:
            # Attempt to find a JSON-like block { ... }
            # distinct from code blocks, just looking for the brace structure
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            
            if json_match:
                clean_output = json_match.group(0)
            else:
                # Fallback to original cleanup if regex fails
                clean_output = output.strip()
                if clean_output.startswith("```json"):
                    clean_output = clean_output[7:]
                if clean_output.startswith("```"):
                    clean_output = clean_output[3:]
                if clean_output.endswith("```"):
                    clean_output = clean_output[:-3]
            
            clean_output = clean_output.strip()
            
            # Parse
            parsed = self.model.model_validate_json(clean_output)
            logger.info(f"Guardrail passed: {type(parsed).__name__}")
            return parsed
            
        except ValidationError as e:
            logger.warning(f"Guardrail validation failed: {e}")
            if self.fix_on_failure:
                return None # Signal need for retry/fix
            raise e
        except Exception as e:
            logger.error(f"Guardrail parsing error: {e}")
            return None

class StandardEvaluation(BaseModel):
    """Standard schema for Agent critique/review."""
    score: int = Field(description="Score from 1-10")
    feedback: str = Field(description="Specific feedback for improvement")
    approved: bool = Field(description="Whether the output is acceptable")

class InputGuardrail:
    """
    Filters unsafe or irrelevant user inputs before they reach the agent.
    Chapter 18: Input Validation / Sanitization.
    """
    def __init__(self, forbidden_terms: list = None):
        self.forbidden_terms = forbidden_terms or ["ignore all instructions", "system prompt"]

    def check(self, user_input: str) -> bool:
        """
        Returns True if safe, False if unsafe.
        """
        content = user_input.lower()
        for term in self.forbidden_terms:
            if term in content:
                logger.warning(f"Guardrail blocked input containing: '{term}'")
                return False
        return True
