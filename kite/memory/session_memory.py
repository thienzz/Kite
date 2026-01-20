"""
Session Memory Manager
Based on Chapter 3.2: Short-Term Memory - Managing the "Now"

The Goldfish Metaphor from book:
- Sharp about 2 minutes ago
- Total amnesia about 5 minutes ago

Strategy: Sliding window with compression to avoid exponential cost growth.

Run: python session_memory.py
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(".kite.env")


@dataclass
class Message:
    """A conversation message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0


@dataclass
class SessionStats:
    """Statistics for a session."""
    total_messages: int = 0
    total_tokens: int = 0
    compressed_count: int = 0
    window_size: int = 0


class SessionMemory:
    """
    Sliding window memory manager for conversations.
    
    Features from Chapter 3.2:
    - Keep last N messages (configurable window)
    - Compress older messages for context
    - Fixed cost per conversation
    - No exponential growth
    
    Example:
        memory = SessionMemoryManager(window_size=10)
        
        # Add messages
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi! How can I help?")
        
        # Get messages for LLM
        messages = memory.get_messages()
    """
    
    def __init__(
        self,
        llm = None,
        window_size: int = 10,
        compression_enabled: bool = True,
        max_tokens_per_message: int = 500
    ):
        """
        Initialize session memory.
        
        Args:
            window_size: Number of recent messages to keep
            compression_enabled: Whether to compress old messages
            max_tokens_per_message: Soft limit for message length
        """
        self.llm = llm
        self.window_size = window_size
        self.compression_enabled = compression_enabled
        self.max_tokens_per_message = max_tokens_per_message
        
        # Message storage
        self.messages: List[Message] = []
        
        # Compressed history (for context)
        self.compressed_history: Optional[str] = None
        
        # Statistics
        self.stats = SessionStats(window_size=window_size)
        
        print(f"[OK] Session Memory initialized")
        print(f"  Window size: {window_size} messages")
        print(f"  Compression: {'enabled' if compression_enabled else 'disabled'}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Rough estimation: 1 token   4 characters in English.
        For production, use tiktoken library.
        """
        return len(text) // 4
    
    def _compress_messages(self, messages: List[Message]) -> str:
        """
        Compress messages into summary.
        
        Uses LLM to create concise summary of conversation history.
        This is much cheaper than keeping all messages.
        """
        if not messages:
            return ""
        
        # Build text from messages
        conversation = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages
        ])
        
        # Ask LLM to summarize
        prompt = f"""Summarize this conversation history in 2-3 sentences. Focus on key topics and decisions.

Conversation:
{conversation}

Summary:"""
        
        if self.llm:
            response = self.llm.complete(prompt, max_tokens=150, temperature=0.3)
            summary = response
        else:
            # Fallback for demo/testing if no LLM provided
            summary = "Summary unavailable (No LLM provider)"
        
        print(f"    Compressed {len(messages)} messages   {len(summary)} chars")
        
        return summary
    
    def add_message(self, role: str, content: str, session_id: Optional[str] = None):
        """Add a message to memory with a specific role."""
        if role.lower() == "user":
            self.add_user_message(content)
        else:
            self.add_assistant_message(content)

    def add_user_message(self, content: str):
        """Add user message to memory."""
        message = Message(
            role="user",
            content=content,
            tokens=self._estimate_tokens(content)
        )
        
        self.messages.append(message)
        self.stats.total_messages += 1
        self.stats.total_tokens += message.tokens
        
        # Apply sliding window
        self._apply_sliding_window()
    
    def add_assistant_message(self, content: str):
        """Add assistant message to memory."""
        message = Message(
            role="assistant",
            content=content,
            tokens=self._estimate_tokens(content)
        )
        
        self.messages.append(message)
        self.stats.total_messages += 1
        self.stats.total_tokens += message.tokens
        
        # Apply sliding window
        self._apply_sliding_window()
    
    def _apply_sliding_window(self):
        """
        Apply sliding window logic.
        
        From Chapter 3.2:
        - Keep last N messages (sharp memory)
        - Compress older messages (compressed context)
        - Total amnesia for very old (to save cost)
        """
        if len(self.messages) <= self.window_size:
            # Within window, no action needed
            return
        
        # Messages outside window
        old_messages = self.messages[:-self.window_size]
        
        # Keep only window
        self.messages = self.messages[-self.window_size:]
        
        # Compress old messages if enabled
        if self.compression_enabled and old_messages:
            new_compression = self._compress_messages(old_messages)
            
            # Merge with existing compression
            if self.compressed_history:
                # Combine old and new compression
                combined = f"{self.compressed_history}\n{new_compression}"
                # Compress the compression if it gets too long
                if len(combined) > 1000:
                    self.compressed_history = self._compress_messages([
                        Message(role="system", content=combined)
                    ])
                else:
                    self.compressed_history = combined
            else:
                self.compressed_history = new_compression
            
            self.stats.compressed_count += len(old_messages)
            
            print(f"      Sliding window applied: {len(old_messages)} messages compressed")
    
    def get_messages(self, include_compression: bool = True) -> List[Dict]:
        """
        Get messages in format for LLM API.
        
        Args:
            include_compression: Whether to include compressed history
            
        Returns:
            List of message dictionaries
        """
        result = []
        
        # Add compressed history as system message
        if include_compression and self.compressed_history:
            result.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.compressed_history}"
            })
        
        # Add current window
        for msg in self.messages:
            result.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return result
    
    def get_token_count(self) -> Dict[str, int]:
        """
        Get token counts for cost estimation.
        
        This shows the cost savings from sliding window approach.
        """
        # Current window tokens
        window_tokens = sum(msg.tokens for msg in self.messages)
        
        # Compressed history tokens
        compression_tokens = self._estimate_tokens(self.compressed_history or "")
        
        # What it would be WITHOUT compression
        naive_tokens = self.stats.total_tokens
        
        # Actual tokens with compression
        actual_tokens = window_tokens + compression_tokens
        
        return {
            "window_tokens": window_tokens,
            "compression_tokens": compression_tokens,
            "actual_tokens": actual_tokens,
            "naive_tokens": naive_tokens,
            "savings": naive_tokens - actual_tokens,
            "savings_percent": (
                ((naive_tokens - actual_tokens) / naive_tokens * 100)
                if naive_tokens > 0 else 0
            )
        }
    
    def clear(self):
        """Clear all memory."""
        self.messages.clear()
        self.compressed_history = None
        self.stats = SessionStats(window_size=self.window_size)
        print("[OK] Session memory cleared")
    
    def get_stats(self) -> SessionStats:
        """Get session statistics."""
        return self.stats


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("SESSION MEMORY MANAGER DEMO")
    print("=" * 70)
    print("\nBased on Chapter 3.2: The Goldfish Metaphor")
    print("- Sharp about recent messages")
    print("- Compressed for older context")
    print("- Total amnesia for very old (cost savings!)")
    print("=" * 70)
    
    # Initialize with small window for demo
    memory = SessionMemoryManager(
        window_size=6,  # Keep last 6 messages
        compression_enabled=True
    )
    
    # Simulate a long conversation
    conversation = [
        ("user", "Hi! I need help with my Python code."),
        ("assistant", "Of course! I'd be happy to help. What's the issue?"),
        ("user", "I'm getting a TypeError when I try to concatenate strings."),
        ("assistant", "That usually happens when you try to concatenate a string with a non-string type. Can you show me the code?"),
        ("user", "Sure: result = 'Count: ' + count"),
        ("assistant", "I see the issue! 'count' is probably an integer. You need to convert it: result = 'Count: ' + str(count)"),
        ("user", "That worked! Thanks!"),
        ("assistant", "Great! Is there anything else you need help with?"),
        ("user", "Actually yes, how do I read a CSV file?"),
        ("assistant", "You can use the csv module or pandas. Here's a simple example with csv module..."),
        ("user", "What about using pandas instead?"),
        ("assistant", "With pandas it's even easier: import pandas as pd; df = pd.read_csv('file.csv')"),
        ("user", "Perfect! One more thing - how do I handle errors?"),
        ("assistant", "Use try-except blocks to handle exceptions gracefully..."),
    ]
    
    print("\n  Simulating 14-message conversation...")
    print(f"   Window size: {memory.window_size} messages\n")
    
    for i, (role, content) in enumerate(conversation, 1):
        print(f"{i}. {role.upper()}: {content[:50]}...")
        
        if role == "user":
            memory.add_user_message(content)
        else:
            memory.add_assistant_message(content)
        
        # Show window status every few messages
        if i % 4 == 0:
            print(f"\n   [CHART] After {i} messages:")
            print(f"      In window: {len(memory.messages)}")
            print(f"      Compressed: {memory.stats.compressed_count}")
            if memory.compressed_history:
                print(f"      Compression: {len(memory.compressed_history)} chars")
            print()
    
    # Show final state
    print("\n" + "="*70)
    print("FINAL MEMORY STATE")
    print("="*70)
    
    messages = memory.get_messages()
    
    if memory.compressed_history:
        print(f"\n  Compressed History:")
        print(f"   {memory.compressed_history}\n")
    
    print(f"  Current Window ({len(memory.messages)} messages):")
    for msg in memory.messages:
        print(f"   {msg.role.upper()}: {msg.content[:60]}...")
    
    # Token analysis
    print("\n" + "="*70)
    print("  COST ANALYSIS")
    print("="*70)
    
    tokens = memory.get_token_count()
    
    print(f"\nToken Counts:")
    print(f"  Window tokens: {tokens['window_tokens']:,}")
    print(f"  Compression tokens: {tokens['compression_tokens']:,}")
    print(f"  Actual total: {tokens['actual_tokens']:,}")
    print()
    print(f"Without Compression:")
    print(f"  Naive total: {tokens['naive_tokens']:,}")
    print()
    print(f"  Savings:")
    print(f"  Tokens saved: {tokens['savings']:,}")
    print(f"  Percentage: {tokens['savings_percent']:.1f}%")
    
    # Cost estimate
    cost_per_1k = 0.0001  # Rough estimate
    actual_cost = (tokens['actual_tokens'] / 1000) * cost_per_1k
    naive_cost = (tokens['naive_tokens'] / 1000) * cost_per_1k
    
    print(f"\nCost Estimate (at ${cost_per_1k:.4f} per 1K tokens):")
    print(f"  With sliding window: ${actual_cost:.6f} per query")
    print(f"  Without compression: ${naive_cost:.6f} per query")
    print(f"  Savings: ${naive_cost - actual_cost:.6f} per query")
    
    # Scale to many queries
    queries_per_day = 1000
    print(f"\nFor {queries_per_day:,} queries/day (30 days):")
    print(f"  With sliding window: ${actual_cost * queries_per_day * 30:.2f}/month")
    print(f"  Without: ${naive_cost * queries_per_day * 30:.2f}/month")
    print(f"    Total savings: ${(naive_cost - actual_cost) * queries_per_day * 30:.2f}/month")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS FROM CHAPTER 3.2")
    print("="*70)
    print("""
1. GOLDFISH MEMORY STRATEGY
   - Keep recent messages (sharp memory)
   - Compress older context
   - Forget very old (cost savings)

2. COST CONTROL
   - Fixed cost per conversation
   - No exponential growth
   - Predictable budgeting

3. QUALITY MAINTAINED
   - Important context preserved via compression
   - Recent messages at full detail
   - Good balance of context vs. cost

4. SCALABILITY
   - Window size adjustable
   - Compression depth configurable
   - Works for short or long conversations
    """)


if __name__ == "__main__":
    demo()
