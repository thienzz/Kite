"""
Classifier SLM (Small Language Model)
Based on Chapter 1.3: Sub-Agents (The Specialists & SLMs)

A specialized small model for classification tasks.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for Classifier SLM."""
    model_name: str = "llama-3.1-8b-classify"  # Fine-tuned version
    confidence_threshold: float = 0.7
    max_tokens: int = 50
    temperature: float = 0.0

# ============================================================================
# CLASSIFIER SLM
# ============================================================================

class ClassifierSLM:
    """
    Small Language Model specialized for classification.
    
    In production, this would use a fine-tuned model (e.g., Llama 3.1 8B).
    This implementation provides a generic interface and base logic.
    """
    
    def __init__(self, config: ClassifierConfig = None, keywords: Dict[str, List[str]] = None):
        self.config = config or ClassifierConfig()
        self.keywords = keywords or {}
        
        print(f"[OK] Classifier SLM initialized")
        print(f"  Model: {self.config.model_name}")
    
    def classify(self, text: str, categories: Dict[str, List[str]] = None) -> Dict:
        """
        Classify text into categories based on keywords.
        
        Args:
            text: Text to classify
            categories: Optional override for keywords mapping (label -> [keywords])
            
        Returns:
            Classification result with confidence
        """
        start_time = time.time()
        
        text_lower = text.lower()
        target_keywords = categories or self.keywords
        
        if not target_keywords:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "latency_ms": (time.time() - start_time) * 1000
            }

        # Score each category
        scores = {}
        for label, keywords in target_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[label] = score
        
        # Determine best match
        if scores:
            best_label = max(scores.items(), key=lambda x: x[1])[0]
            total_keywords = sum(scores.values())
            # Basic confidence simulation
            confidence = min(scores[best_label] / (total_keywords + 0.1), 0.99)
        else:
            best_label = "other"
            confidence = 0.5
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "label": best_label,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "all_scores": scores
        }
    
    def get_stats(self) -> Dict:
        """Get generator statistics."""
        return {
            "model": self.config.model_name,
            "categories_count": len(self.keywords)
        }
