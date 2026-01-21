"""
Case Study 9: Human-in-Loop (HITL)
Demonstrates checkpoints for approval and intervention points for user feedback.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite.core import Kite
from kite.pipeline.deterministic_pipeline import PipelineStatus

async def user_intervention_callback(state):
    print("\n[USER INTERVENTION]")
    print(f"Current Data: {state.data}")
    print(f"Results so far: {list(state.results.keys())}")
    
    # Simulate a human changing the plan or providing feedback
    # In a real app, this might come from a UI or CLI input
    state.data = state.data + " (Revised by Human for better accuracy)"
    print("User tweaked the data. Continuing...")

async def main():
    print("="*80)
    print("CASE STUDY 9: HUMAN-IN-LOOP PIPELINE")
    print("="*80)

    ai = Kite()
    
    # 1. Create a Workflow with human checkpoints
    workflow = ai.create_workflow("secure_research_flow")
    
    # Step 1: Automated Research
    async def automated_research(data):
        print(f"     [Research] Analyzing topic: {data}")
        await asyncio.sleep(1)
        return f"Research results for {data}"
    
    workflow.add_step("research", automated_research)
    
    # Checkpoint: Pause for approval before planning
    workflow.add_checkpoint("research", approval_required=True)
    
    # Intervention: Let user tweak the research results before planning
    workflow.add_intervention_point("create_plan", user_intervention_callback)
    
    # Step 2: Create Plan
    async def create_plan(research_data):
        print(f"     [Planning] Creating plan based on: {research_data}")
        return f"Final Plan based on {research_data}"
    
    workflow.add_step("create_plan", create_plan)

    # 2. Start Execution
    print("\nPhase 1: Starting Workflow...")
    state = await workflow.execute_async("Quantum Computing Trends 2024")
    
    print(f"\nWorkflow Status: {state.status}")
    if state.status == PipelineStatus.AWAITING_APPROVAL:
        print(f"Task ID: {state.task_id}")
        print("Workflow is waiting for your approval to proceed to the next phase.")

    # 3. Simulate Human Review and Resumption
    print("\n--- Human is reviewing the results ---")
    await asyncio.sleep(1)
    print("Human: 'Research looks good. Please proceed and I'll tweak the plan.'")
    
    print("\nPhase 2: Resuming Workflow...")
    final_state = await workflow.resume_async(state.task_id, feedback="Checked and approved")
    
    print(f"\nFinal Workflow Status: {final_state.status}")
    print(f"Final Result: {final_state.results['create_plan']}")

    print("\n" + "="*80)
    print("[OK] CASE STUDY 9 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
