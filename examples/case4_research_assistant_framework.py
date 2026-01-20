"""
CASE STUDY 4: RESEARCH ASSISTANT WITH REACT LOOP (REFACTORED)
=============================================================
Demonstrates: GraphRAG, AdvancedRAG, Kill Switch, Autonomous Agent
NOW USING FRAMEWORK-LEVEL ReActAgent and KillSwitch!

Run: python case4_research_assistant_framework.py
"""

import os
import sys
import time
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_framework import AgenticAI
from agentic_framework.safety import KillSwitch

# ============================================================================
# CUSTOM RESEARCH TOOLS (Can also use ai.tools['web_search'])
# ============================================================================

def web_search_tool(query):
    """Mock web search with specific data for this case study"""
    if "scaling" in query.lower() or "chinchilla" in query.lower():
        return {
            'success': True,
            'results': [
                {'title': 'Chinchilla Scaling Laws (DeepMind 2024)', 'snippet': 'Optimal compute-to-data ratio...'},
            ]
        }
    return {'success': True, 'results': []}

def fetch_paper_tool(arxiv_id):
    """Mock paper fetching"""
    papers = {
        '2203.15556': {'title': 'Training Compute-Optimal Large Language Models', 'abstract': '...'}
    }
    return papers.get(arxiv_id, {'title': 'Paper not found', 'abstract': ''})

# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    print("="*80)
    print("CASE STUDY 4: RESEARCH ASSISTANT (FRAMEWORK REFACTORED)")
    print("="*80)
    
    # 1. Initialize Framework
    print("\n[START] Initializing framework...")
    ai = AgenticAI()
    
    # 2. Setup Kill Switch (from framework)
    print("\n[WARN]  Setting up framework KillSwitch...")
    kill_switch = KillSwitch(
        max_iterations=5,
        max_cost=0.50,
        max_same_action=2,
        max_time=60
    )
    
    # 3. Create Tools
    print("\n  Registering tools...")
    web_search = ai.create_tool("case_web_search", web_search_tool, "Search for AI research")
    fetch_paper = ai.create_tool("fetch_paper", fetch_paper_tool, "Fetch ArXiv papers")
    
    # 4. Load Knowledge Base
    print("\n  Loading knowledge base...")
    ai.vector_memory.add_document("doc_chinchilla", "Chinchilla Scaling Laws findings...")
    ai.graph_rag.add_document("doc_chinchilla", "Chinchilla Scaling Laws findings...")
    
    # 5. Create ReAct Agent (now a framework-level feature)
    print("\n  Creating framework ReActAgent...")
    researcher = ai.create_react_agent(
        name="ResearchAgent",
        system_prompt="You are a senior research assistant.",
        tools=[web_search, fetch_paper],
        kill_switch=kill_switch
    )
    
    # 6. Execute Autonomous Loop
    print("\n" + "="*80)
    print("EXECUTING AUTONOMOUS RESEARCH")
    print("="*80)
    
    goal = "What are the key findings about AI scaling laws in 2024?"
    result = await researcher.run_autonomous(goal)
    
    # 7. Print Results
    print(f"\n{'='*80}")
    print("[OK] RESEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"\nFindings:\n{result['summary']}")
    print(f"\nStats: Steps={result['steps']}, Cost=${result['total_cost']:.4f}, Stop Reason={result['reason']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
