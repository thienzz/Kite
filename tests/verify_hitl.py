import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite.pipeline.deterministic_pipeline import DeterministicPipeline, PipelineStatus

def step1(data):
    print(f"     [Step 1] Processing: {data}")
    return data + " -> step1"

def step2(data):
    print(f"     [Step 2] Processing: {data}")
    return data + " -> step2"

async def async_step1(data):
    await asyncio.sleep(0.1)
    print(f"     [Async Step 1] Processing: {data}")
    return data + " -> async_step1"

def my_intervention(state):
    print(f"     [Callback] Intervention triggered. Current data: {state.data}")
    # Human tweaks the data
    state.data = state.data + " (tweaked by human)"

async def test_hitl_sync():
    print("\n--- Testing HITL Sync ---")
    pipe = DeterministicPipeline("hitl_sync")
    pipe.add_step("step1", step1)
    pipe.add_checkpoint("step1", approval_required=True)
    pipe.add_step("step2", step2)
    
    # 1. Start
    state = pipe.execute("initial_data")
    print(f"Status after step1: {state.status}")
    assert state.status == PipelineStatus.AWAITING_APPROVAL
    assert state.results["step1"] == "initial_data -> step1"
    
    # 2. Resume
    state = pipe.resume(state.task_id, feedback="Proceed please")
    print(f"Status after resume: {state.status}")
    assert state.status == PipelineStatus.COMPLETED
    assert state.results["step2"] == "initial_data -> step1 -> step2"
    print("[OK] HITL Sync Verified")

async def test_hitl_async():
    print("\n--- Testing HITL Async ---")
    pipe = DeterministicPipeline("hitl_async")
    pipe.add_intervention_point("async_step1", my_intervention)
    pipe.add_step("async_step1", async_step1)
    pipe.add_checkpoint("async_step1", approval_required=True)
    
    # 1. Start
    state = await pipe.execute_async("async_data")
    print(f"Status after intervention + step1: {state.status}")
    assert state.status == PipelineStatus.AWAITING_APPROVAL
    # Note: intervention happens before step, so it should have tweaked state.data
    assert "tweaked by human" in state.results["async_step1"]
    
    # 2. Resume
    state = await pipe.resume_async(state.task_id)
    assert state.status == PipelineStatus.COMPLETED
    print("[OK] HITL Async Verified")

if __name__ == "__main__":
    asyncio.run(test_hitl_sync())
    asyncio.run(test_hitl_async())
