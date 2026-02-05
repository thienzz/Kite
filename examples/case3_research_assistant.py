"""
CASE 3: PRODUCTION DEEP RESEARCH ASSISTANT
==========================================
A production-grade Deep Research system that:
1. Decomposes complex topics into sub-questions.
2. Performs real parallel/iterative web searches.
3. Synthesizes findings into a comprehensive report.
4. Uses 'Fast' models for searching and 'Smart' models for analysis.
"""

import os
import sys
import asyncio
import json
import time

# Ensure pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite
from kite.optimization.resource_router import ResourceAwareRouter
from kite.tools import WebSearchTool
from kite.persistence import JSONCheckpointer

async def main():
    print("\n" + "=" * 80)
    print("CASE 3: DEEP RESEARCH ASSISTANT (Production Mode)")
    print("=" * 80)

    # --- 1. Initialize Framework ---
    ai = Kite()
    router = ResourceAwareRouter(ai.config)
    
    # Tools
    search_tool = WebSearchTool()
    
    # Persistence
    checkpointer = JSONCheckpointer("case3_research_state.json")
    
    # --- 2. Configuration ---
    topic = "The Impact of Solid State Batteries on EV Industry by 2030"
    max_sub_questions = 4
    
    # --- 3. Build Agents ---
    
    # PLANNER: Uses Smart Model to decompose the valid research strategy
    planner = ai.create_agent(
        name="Planner",
        model=router.smart_model, # High reasoning capability
        system_prompt=f"""You are a Strategic Research Lead.
        Your goal is to break down a complex topic into {max_sub_questions} distinct, searchable sub-questions.
        Ensure coverage of: Market Size, Technology Readiness, Key Players, and Challenges.
        
        Output strictly valid JSON:
        {{
            "plan": [
                "Question 1...",
                "Question 2..."
            ]
        }}
        """,
        verbose=True
    )
    
    # RESEARCHER: Uses Fast Model + Search Tool for high-volume data gathering
    researcher = ai.create_agent(
        name="Researcher",
        model=router.fast_model, # Speed & Low Cost
        tools=[search_tool],
        system_prompt="""You are an Expert Field Researcher.
        You have access to 'web_search'.
        Your job is to find CONCRETE facts, stats, and quotes for the given question.
        
        Do NOT summarize broadly. Give detailed notes / bullet points with sources.
        Use 'web_search' aggressively. 
        """,
        verbose=True
    )
    
    # ANALYST: Uses Smart Model to synthesize
    analyst = ai.create_agent(
        name="Analyst",
        model=router.smart_model,
        system_prompt="""You are a Chief Intelligence Officer.
        Write a professional, markdown-formatted Research Report based EXCLUSIVELY on the provided notes.
        
        Structure:
        # Title
        ## Executive Summary
        ## Detailed Findings (Grouped logically)
        ## Strategic Outlook (2025-2030)
        ## References
        
        Be data-driven. Cite sources if available in notes.
        """,
        verbose=True
    )

    # --- 4. Execution Workflow ---
    
    # A. Checkpoint Recovery
    state = checkpointer.load()
    if not state:
        state = {
            "topic": topic,
            "plan": [],
            "research_notes": {}, # question -> notes
            "final_report": "",
            "status": "planning"
        }
        print(f"   [System] Starting fresh research on: '{topic}'")
    else:
        print(f"   [System] Resuming research on: '{state['topic']}' (Status: {state['status']})")
        
    # B. Planning Phase
    if state['status'] == "planning":
        print("\n[Phase 1] Planning Research Strategy...")
        res = await planner.run(f"Create a research plan for: {topic}")
        
        try:
            # Robust JSON extraction (similar to router)
            import re
            response_text = res['response']
            
            # Try to extract JSON
            cleaned = re.sub(r"```json|```", "", response_text).strip()
            
            # Try regex to find JSON array
            json_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            
            try:
                plan_data = json.loads(cleaned)
                
                # Handle both formats: {"plan": [...]} or [...]
                if isinstance(plan_data, dict):
                    state['plan'] = plan_data.get('plan', [])
                elif isinstance(plan_data, list):
                    state['plan'] = plan_data
                else:
                    raise ValueError("Invalid plan format")
                    
            except json.JSONDecodeError:
                # Fallback: extract questions from text
                print("   [WARN] JSON parse failed, extracting questions from text...")
                lines = response_text.split('\n')
                questions = []
                for line in lines:
                    # Look for numbered questions
                    match = re.match(r'\s*\d+[\.\)]\s*(.+)', line)
                    if match:
                        questions.append(match.group(1).strip())
                
                if questions:
                    state['plan'] = questions[:max_sub_questions]
                else:
                    raise ValueError("Could not extract questions from response")
            
            # Validation
            if not state['plan']:
                raise ValueError("Planner returned empty plan.")

                
            print(f"   [Planner] Generated {len(state['plan'])} sub-questions.")
            for i, q in enumerate(state['plan']):
                print(f"      {i+1}. {q}")
            
            state['status'] = "researching"
            checkpointer.save(state)
            
        except Exception as e:
            print(f"   [Error] Planning failed: {e}")
            return

    # C. Research Phase (Iterative/Parallel)
    if state['status'] == "researching":
        print("\n[Phase 2] Executing Deep Research...")
        
        # We can implement this sequentially for reliability or parallel for speed
        # For 'Deep Research', sequential often yields better 'Chain of Thought' observation in logs
        
        for i, question in enumerate(state['plan']):
            if question in state['research_notes']:
                print(f"   [Skip] Already researched: {question}")
                continue
                
            print(f"\n   👉 Researching Q{i+1}: {question}")
            
            # Using context from previous questions to avoid duplication? 
            # ideally yes, but let's keep it independent for now to avoid context overflow
            
            res = await researcher.run(f"Find detailed information for: {question}")
            state['research_notes'][question] = res['response']
            
            # Respect Rate Limits (Groq has tight limits for free tier)
            print("   [System] Cooling down for 2s...")
            time.sleep(2)
            
            # Save progress after each step
            checkpointer.save(state)
            
        state['status'] = "reporting"
        checkpointer.save(state)

    # D. Reporting Phase
    if state['status'] == "reporting":
        print("\n[Phase 3] Synthesizing Final Report...")
        
        # Compile Context
        context = f"TOPIC: {state['topic']}\n\nRESEARCH NOTES:\n"
        for q, notes in state['research_notes'].items():
            context += f"## Q: {q}\n{notes}\n\n"
            
        print(f"   [Analyst] Reading {len(context)} chars of notes...")
        
        res = await analyst.run("Write the final report now.", context=context)
        state['final_report'] = res['response']
        state['status'] = "completed"
        checkpointer.save(state)
        
    # E. Final Output
    if state['status'] == "completed":
        filename = "research_report.md"
        with open(filename, "w") as f:
            f.write(state['final_report'])
            
        print("\n" + "=" * 80)
        print(f"RESEARCH COMPLETED. Report saved to: {filename}")
        print("=" * 80)
        
        # Optional: Auto-clear checkpoint if successful
        checkpointer.clear()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Stopped] User interrupt.")
    except Exception as e:
        import traceback
        traceback.print_exc()
