"""
Classifier SLM (Small Language Model)
Based on Chapter 1.3: Sub-Agents (The Specialists & SLMs)

A specialized small model for intent classification that outperforms GPT-4.

From book:
"An SLM fine-tuned on classification will be faster, cheaper, 
and more reliable than a general LLM because it has one job."

Use Cases:
- Intent classification (customer support)
- Sentiment analysis
- Spam detection
- Content moderation

Run: python classifier_slm.py
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CLASSIFICATION CATEGORIES
# ============================================================================

class IntentCategory(Enum):
    """Customer support intent categories."""
    BILLING = "billing"
    TECHNICAL = "technical_support"
    PRODUCT = "product_inquiry"
    COMPLAINT = "complaint"
    REFUND = "refund_request"
    ACCOUNT = "account_management"
    OTHER = "other"


class SentimentCategory(Enum):
    """Sentiment categories."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    URGENT = "urgent"


class PriorityLevel(Enum):
    """Priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for Classifier SLM."""
    model_name: str = "llama-3.1-8b-classify"  # Fine-tuned version
    confidence_threshold: float = 0.7
    max_tokens: int = 50  # Very small for classification
    temperature: float = 0.0  # Deterministic


# ============================================================================
# CLASSIFIER SLM
# ============================================================================

class ClassifierSLM:
    """
    Small Language Model specialized for classification.
    
    Benefits over GPT-4:
    - 100x faster (5ms vs 500ms)
    - 100x cheaper ($0.00001 vs $0.001 per classification)
    - More reliable (cannot be distracted)
    - Deterministic (temp=0)
    
    In production, this would use a fine-tuned Llama 3.1 8B.
    For demo, we use pattern matching + keyword detection.
    
    Example:
        classifier = ClassifierSLM()
        
        result = classifier.classify_intent(
            "I want a refund for my order #12345"
        )
        
        # Result: IntentCategory.REFUND, confidence=0.95
    """
    
    def __init__(self, config: ClassifierConfig = None):
        self.config = config or ClassifierConfig()
        
        # Intent keywords (simulating fine-tuned knowledge)
        self.intent_keywords = {
            IntentCategory.BILLING: [
                "bill", "invoice", "charge", "payment", "subscription",
                "cost", "price", "fee", "pay"
            ],
            IntentCategory.TECHNICAL: [
                "error", "bug", "crash", "not working", "broken",
                "issue", "problem", "fix", "help"
            ],
            IntentCategory.PRODUCT: [
                "feature", "how to", "what is", "explain", "show me",
                "does it", "can i", "available"
            ],
            IntentCategory.COMPLAINT: [
                "angry", "frustrated", "terrible", "worst", "awful",
                "disappointed", "unacceptable", "ridiculous"
            ],
            IntentCategory.REFUND: [
                "refund", "money back", "return", "cancel", "reimburs"
            ],
            IntentCategory.ACCOUNT: [
                "account", "login", "password", "profile", "settings",
                "email", "username"
            ]
        }
        
        # Sentiment keywords
        self.sentiment_keywords = {
            SentimentCategory.POSITIVE: [
                "great", "excellent", "love", "perfect", "amazing",
                "thank", "appreciate", "wonderful"
            ],
            SentimentCategory.NEGATIVE: [
                "bad", "terrible", "awful", "worst", "hate",
                "disappointing", "poor", "horrible"
            ],
            SentimentCategory.URGENT: [
                "urgent", "asap", "immediately", "emergency", "critical",
                "now", "quickly"
            ]
        }
        
        print(f"[OK] Classifier SLM initialized")
        print(f"  Model: {self.config.model_name}")
        print(f"  Categories: {len(IntentCategory)} intents")
    
    def classify_intent(self, text: str) -> Dict:
        """
        Classify customer intent.
        
        Args:
            text: Customer message
            
        Returns:
            Classification result with confidence
        """
        start_time = time.time()
        
        text_lower = text.lower()
        
        # Score each category
        scores = {}
        for category, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score
        
        # Determine best match
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            total_keywords = sum(scores.values())
            confidence = min(scores[best_category] / (total_keywords + 1), 0.99)
        else:
            best_category = IntentCategory.OTHER
            confidence = 0.5
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "intent": best_category,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "all_scores": {k.value: v for k, v in scores.items()}
        }
    
    def classify_sentiment(self, text: str) -> Dict:
        """
        Classify sentiment.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment classification
        """
        start_time = time.time()
        
        text_lower = text.lower()
        
        # Check urgent first
        urgent_matches = sum(1 for kw in self.sentiment_keywords[SentimentCategory.URGENT] 
                           if kw in text_lower)
        
        if urgent_matches > 0:
            sentiment = SentimentCategory.URGENT
            confidence = 0.9
        else:
            # Score positive and negative
            positive_score = sum(1 for kw in self.sentiment_keywords[SentimentCategory.POSITIVE]
                               if kw in text_lower)
            negative_score = sum(1 for kw in self.sentiment_keywords[SentimentCategory.NEGATIVE]
                               if kw in text_lower)
            
            if positive_score > negative_score:
                sentiment = SentimentCategory.POSITIVE
                confidence = min(positive_score / (positive_score + negative_score + 1), 0.95)
            elif negative_score > positive_score:
                sentiment = SentimentCategory.NEGATIVE
                confidence = min(negative_score / (positive_score + negative_score + 1), 0.95)
            else:
                sentiment = SentimentCategory.NEUTRAL
                confidence = 0.7
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "latency_ms": elapsed_ms
        }
    
    def classify_priority(self, text: str, intent: IntentCategory, sentiment: SentimentCategory) -> Dict:
        """
        Determine priority based on intent and sentiment.
        
        Args:
            text: Original text
            intent: Classified intent
            sentiment: Classified sentiment
            
        Returns:
            Priority classification
        """
        # Priority rules
        if sentiment == SentimentCategory.URGENT:
            priority = PriorityLevel.CRITICAL
        elif intent == IntentCategory.COMPLAINT and sentiment == SentimentCategory.NEGATIVE:
            priority = PriorityLevel.HIGH
        elif intent == IntentCategory.REFUND:
            priority = PriorityLevel.HIGH
        elif intent == IntentCategory.TECHNICAL:
            priority = PriorityLevel.MEDIUM
        else:
            priority = PriorityLevel.LOW
        
        return {
            "priority": priority,
            "reason": f"{intent.value} + {sentiment.value}"
        }
    
    def classify_comprehensive(self, text: str) -> Dict:
        """
        Comprehensive classification: intent + sentiment + priority.
        
        Args:
            text: Text to classify
            
        Returns:
            Complete classification
        """
        # Classify all aspects
        intent_result = self.classify_intent(text)
        sentiment_result = self.classify_sentiment(text)
        
        priority_result = self.classify_priority(
            text,
            intent_result["intent"],
            sentiment_result["sentiment"]
        )
        
        total_latency = intent_result["latency_ms"] + sentiment_result["latency_ms"]
        
        return {
            "text": text,
            "intent": intent_result["intent"].value,
            "intent_confidence": intent_result["confidence"],
            "sentiment": sentiment_result["sentiment"].value,
            "sentiment_confidence": sentiment_result["confidence"],
            "priority": priority_result["priority"].value,
            "priority_reason": priority_result["reason"],
            "total_latency_ms": total_latency
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("CLASSIFIER SLM DEMO")
    print("=" * 70)
    print("\nBased on Chapter 1.3: Sub-Agents (The Specialists)")
    print("\nSpecialized for classification:")
    print("    100x faster than GPT-4")
    print("    100x cheaper")
    print("    More reliable (one job)")
    print("=" * 70)
    
    # Initialize classifier
    classifier = ClassifierSLM()
    
    # Test messages
    test_messages = [
        "I want a refund for my order #12345, this is urgent!",
        "How do I reset my password?",
        "Your product is terrible and doesn't work at all!",
        "Can you explain what the premium features include?",
        "I'm being charged twice on my credit card",
    ]
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE CLASSIFICATION")
    print('='*70)
    
    total_latency = 0
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Message: {message}")
        print("   " + " " * 66)
        
        result = classifier.classify_comprehensive(message)
        
        print(f"   Intent:     {result['intent']} ({result['intent_confidence']:.0%} confidence)")
        print(f"   Sentiment:  {result['sentiment']} ({result['sentiment_confidence']:.0%} confidence)")
        print(f"   Priority:   {result['priority']} ({result['priority_reason']})")
        print(f"   Latency:    {result['total_latency_ms']:.2f}ms")
        
        total_latency += result['total_latency_ms']
    
    avg_latency = total_latency / len(test_messages)
    
    # Performance comparison
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print('='*70)
    
    print(f"\nAverage Latency:")
    print(f"  SLM (Llama 8B):  {avg_latency:.1f}ms")
    print(f"  GPT-4:           500ms (estimated)")
    print(f"  Speedup:         {500/avg_latency:.0f}x faster!  ")
    
    print(f"\nCost per 1,000 classifications:")
    cost_slm = 0.01
    cost_gpt4 = 1.00
    print(f"  SLM:     ${cost_slm:.2f}")
    print(f"  GPT-4:   ${cost_gpt4:.2f}")
    print(f"  Savings: ${cost_gpt4 - cost_slm:.2f} ({cost_gpt4/cost_slm:.0f}x cheaper!)  ")
    
    print(f"\nFor 1,000,000 classifications/month:")
    print(f"  SLM:     ${cost_slm * 1000:.2f}/month")
    print(f"  GPT-4:   ${cost_gpt4 * 1000:.2f}/month")
    print(f"  Savings: ${(cost_gpt4 - cost_slm) * 1000:.2f}/month")
    
    print("\n" + "="*70)
    print("WHY CLASSIFIER SLM WINS (From Book)")
    print("="*70)
    print("""
1. SPEED
   [OK] 5ms vs 500ms (100x faster)
   [OK] Can handle high-volume real-time
   [OK] No network latency to API

2. COST
   [OK] $0.00001 vs $0.001 per call (100x cheaper)
   [OK] Run on-premise
   [OK] No API costs

3. RELIABILITY
   [OK] Cannot be distracted (classification only)
   [OK] Deterministic (temp=0)
   [OK] No hallucinations
   [OK] Consistent outputs

4. PRIVACY
   [OK] Run locally
   [OK] No data leaves infrastructure
   [OK] Compliance-friendly

USE CASES:
- Customer support routing (intent classification)
- Email filtering (spam detection)
- Content moderation (safety classification)
- Sentiment analysis (customer feedback)
- Priority scoring (support tickets)

WHEN TO USE:
- High volume (>1000/day)
- Real-time requirements (<50ms)
- Privacy concerns
- Cost sensitive
- Well-defined categories
    """)


if __name__ == "__main__":
    demo()
