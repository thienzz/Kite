"""
CASE 4: Multi-Agent Collaboration (Production Grade)
- Pattern: Supervisor / Iterative Refinement
- Agents: Researcher (Fast), Analyst (Adaptive), Critic (Smart)
- Tools: WebSearchTool (Real DuckDuckGo Scraper)
- Features: Guardrails, Resource Awareness, Real State Persistence
"""

import os
import sys
import asyncio
import time
import json
import re
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field

# Ensure pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite
from kite import Kite
from kite.optimization.resource_router import ResourceAwareRouter
from kite.safety.guardrails import InputGuardrail, OutputGuardrail, StandardEvaluation
from kite.tools import WebSearchTool
from kite.persistence import JSONCheckpointer

# --- 2. Define Tools ---
# WebSearchTool imported from kite.tools
# WebSearchTool imported from kite.tools

# --- 3. Define Guardrails ---
# Using StandardEvaluation from kite.safety.guardrails

# --- 4. Main Workflow ---
async def main():
    print("\n================================================================================")
    print("CASE 4: PRODUCTION-GRADE MULTI-AGENT COLLABORATION")
    print("================================================================================\n")

    # 1. Initialize
    ai = Kite() 
    router = ResourceAwareRouter(ai.config)
    search_tool = WebSearchTool()
    
    # Check for persistence
    topic = "The integration of Quantum Computing in FinTech"
    state_file = "case4_state.json"
    checkpointer = JSONCheckpointer(state_file)
    
    state = checkpointer.load()
    
    if not state:
        state = {
            "topic": topic,
            "facts": [],
            "draft_report": "",
            "feedback_history": [],
            "iteration": 0,
            "approved": False
        }
    
    # 2. Guardrails
    quality_guard = OutputGuardrail(StandardEvaluation)
    
    # 3. Agents
    # Researcher: Fast & Real Search
    researcher = ai.create_agent(
        name="Researcher",
        model=router.fast_model,
        tools=[search_tool],
        verbose=True,
        system_prompt="""You are a Senior Researcher. 
        You have access to ONE tool: 'web_search'.
        Use 'web_search' to gather REAL facts from the internet.
        Do NOT invent tools (like brave_search). 
        Do NOT make things up.
        Output a bulleted list of key findings with sources."""
    )
    
    # Analyst: Adaptive
    analyst = ai.create_agent(
        name="Analyst",
        verbose=True,
        system_prompt="""You are a lead Tech Analyst. 
        Write a comprehensive executive summary based on the provided facts.
        Structure: Title, Executive Summary, Key Risks, Strategic Recommendations.
        """
    )
    
    # Critic: Smart
    critic = ai.create_agent(
        name="Critic",
        system_prompt="""You are an implementation reviewer.
        Review the report for clarity, concrete examples, and strategic value.
        If vague, reject it.
        You MUST output JSON: {"score": int, "feedback": str, "approved": bool}."""
    )

    # 4. Supervisor Loop
    print(f"[Supervisor] Processing project: '{state['topic']}'")
    print(f"[State] Current Iteration: {state['iteration']}")
    
    max_revisions = 3
    
    while state['iteration'] < max_revisions and not state['approved']:
        state['iteration'] += 1
        print(f"\n--- Iteration {state['iteration']}/{max_revisions} ---")
        checkpointer.save(state) # Checkpoint start of iteration
        
        # Step A: Research
        # Heuristic: If we have no facts OR feedback asks for more data
        needs_research = not state['facts'] or (state['feedback_history'] and "data" in state['feedback_history'][-1].lower())
        
        if needs_research:
            print("\n[Supervisor] Tasking Researcher (Real Web Search)...")
            sub_query = f"{state['topic']} latest news"
            if state['feedback_history']:
                 # extract keywords from feedback?
                 sub_query = f"{state['topic']} {state['feedback_history'][-1][:20]}"
            
            t_res = await researcher.run(f"Find information about: {sub_query}")
            state['facts'].append(t_res['response'])
            print(f"   [Researcher] Data gathered.")
            checkpointer.save(state)

        # Step B: Drafting
        print("\n[Supervisor] Tasking Analyst to Draft...")
        # Summarize facts context to avoid context window explosion
        facts_summary = "\n".join([f[:500] + "..." for f in state['facts'][-2:]]) 
        context = f"Facts:\n{facts_summary}\n\nPrevious Feedback:\n{state['feedback_history']}"
        
        t_draft = await analyst.run(f"Write the report on {state['topic']}. Incorporate feedback.", context=context)
        state['draft_report'] = t_draft['response']
        print(f"   [Analyst] {len(state['draft_report'])} chars written.")
        checkpointer.save(state)
        
        # Step C: Critic
        print("\n[Supervisor] Tasking Critic to Review...")
        t_critique_raw = await critic.run(f"Review this draft:\n\n{state['draft_report']}")
        
        try:
            # Extract JSON
            clean_json = t_critique_raw['response']
            if "```" in clean_json:
                clean_json = re.search(r"\{.*\}", clean_json, re.DOTALL).group(0)
                
            validated = quality_guard.validate(clean_json)
            
            if validated:
                print(f"   [Critic] Score: {validated.score}/10 | Approved: {validated.approved}")
                print(f"   [Feedback] {validated.feedback}")
                
                if validated.approved:
                    state['approved'] = True
                    checkpointer.save(state)
                    break
                
                state['feedback_history'].append(validated.feedback)
                checkpointer.save(state)
            else:
                 print("   [Critic] JSON validation failed.")
                 state['feedback_history'].append("Format error - try again.")
        except Exception as e:
            print(f"   [Critic Error] {e}")
            state['feedback_history'].append("System error.")
            
        time.sleep(2)

    # 5. Final Output
    if state['approved']:
        print("\nROJECT COMPLETED SUCCESSFULLY")
        filename = "final_report.md"
        with open(filename, "w") as f:
            f.write(f"# {state['topic']}\n\n{state['draft_report']}")
        print(f"   Saved to: {filename}")
        
        # Cleanup state
        checkpointer.clear()
    else:
        print("\nPROJECT TERMINATED (Max revisions reached)")
        print("   Saving draft...")
        with open("draft_report.md", "w") as f:
            f.write(state['draft_report'])

if __name__ == "__main__":
    asyncio.run(main())
