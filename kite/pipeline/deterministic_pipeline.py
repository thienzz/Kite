"""
Deterministic Pipeline Pattern
Level 1 Autonomy: The assembly line pattern with ZERO risk.

Flow: Input -> Step 1 -> Step 2 -> ... -> Action
- No loops
- No choices
- Precise, predictable execution
"""

import os
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class PipelineStatus(Enum):
    """Generic pipeline processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class PipelineState:
    """Current state of data in pipeline."""
    task_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    data: Any = None
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    current_step_index: int = 0
    feedback: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

class DeterministicPipeline:
    """
    A generic deterministic processing pipeline.
    
    Level 1 Autonomy:
    - Fixed sequence
    - No loops
    - Predictable
    
    Example:
        pipeline = DeterministicPipeline("data_processor")
        pipeline.add_step("load", load_func)
        pipeline.add_step("process", process_func)
        
        result = pipeline.execute(raw_data)
    """
    
    def __init__(self, name: str = "pipeline", event_bus = None):
        self.name = name
        self.event_bus = event_bus
        self.steps: List[tuple[str, Callable]] = []
        self.checkpoints: Dict[str, bool] = {}  # step_name -> approval_required
        self.intervention_points: Dict[str, Callable] = {}  # step_name -> callback
        self.history: List[PipelineState] = []
        if self.event_bus:
            self.event_bus.emit("pipeline:init", {"pipeline": self.name})
    
    def add_step(self, name: str, func: Callable):
        """Add a step to the pipeline."""
        self.steps.append((name, func))
        print(f"  [OK] Added step: {name}")

    def add_checkpoint(self, step_name: str, approval_required: bool = True):
        """Pause execution for approval after this step."""
        self.checkpoints[step_name] = approval_required
        print(f"  [OK] Added checkpoint after: {step_name}")

    def add_intervention_point(self, step_name: str, callback: Callable):
        """Call a callback for user intervention before this step."""
        self.intervention_points[step_name] = callback
        print(f"  [OK] Added intervention point before: {step_name}")
    
    def execute(self, data: Any, task_id: Optional[str] = None) -> PipelineState:
        """Execute all steps in the pipeline sequentially."""
        t_id = task_id or f"TASK-{len(self.history)+1:04d}"
        if self.event_bus:
            self.event_bus.emit("pipeline:start", {"pipeline": self.name, "task_id": t_id, "data": str(data)[:100]})
        
        state = PipelineState(task_id=t_id, data=data)
        state.status = PipelineStatus.PROCESSING
        self.history.append(state)
        
        return self._run_sync(state)

    def resume(self, task_id: str, feedback: Optional[str] = None) -> PipelineState:
        """Resume a suspended or awaiting_approval task (sync)."""
        state = next((s for s in self.history if s.task_id == task_id), None)
        if not state:
            raise ValueError(f"Task ID {task_id} not found in history")
        
        if state.status not in [PipelineStatus.SUSPENDED, PipelineStatus.AWAITING_APPROVAL]:
            print(f"[WARNING] Task {task_id} is in status {state.status}, not suspended.")
            return state

        print(f"\n[RESUME] Resuming pipeline: {self.name} (Task: {task_id})")
        state.status = PipelineStatus.PROCESSING
        state.feedback = feedback
        
        return self._run_sync(state)

    def _run_sync(self, state: PipelineState) -> PipelineState:
        """Internal runner for sync execution."""
        try:
            while state.current_step_index < len(self.steps):
                step_idx = state.current_step_index
                step_name, func = self.steps[step_idx]
                
                # 1. Intervention Point
                if step_name in self.intervention_points:
                    print(f"     [INTERVENTION] Triggering before: {step_name}...")
                    callback = self.intervention_points[step_name]
                    import inspect
                    if inspect.iscoroutinefunction(callback):
                        raise RuntimeError(f"Intervention callback for '{step_name}' is async. Use execute_async().")
                    callback(state)

                current_data = state.results[self.steps[step_idx-1][0]] if step_idx > 0 else state.data
                
                if inspect.iscoroutinefunction(func):
                    raise RuntimeError(f"Step '{step_name}' is async. Use execute_async().")
                
                result = func(current_data)
                state.results[step_name] = result

                if self.event_bus:
                    self.event_bus.emit("pipeline:step", {
                        "pipeline": self.name, 
                        "task_id": state.task_id,
                        "step": step_name,
                        "result": str(result)[:200]
                    })
                state.current_step_index += 1
                state.updated_at = datetime.now()

                # 3. Checkpoint
                if step_name in self.checkpoints:
                    approval_req = self.checkpoints[step_name]
                    if approval_req:
                        if self.event_bus:
                            self.event_bus.emit("pipeline:checkpoint", {"pipeline": self.name, "task_id": state.task_id, "step": step_name})
                        state.status = PipelineStatus.AWAITING_APPROVAL
                        return state
                    else:
                        state.status = PipelineStatus.SUSPENDED
                        return state
            
            state.status = PipelineStatus.COMPLETED
            if self.event_bus:
                self.event_bus.emit("pipeline:complete", {"pipeline": self.name, "task_id": state.task_id})
            
        except Exception as e:
            state.status = PipelineStatus.ERROR
            state.errors.append(str(e))
            state.updated_at = datetime.now()
            print(f"[ERROR] Pipeline '{self.name}' failed: {e}")
            
        return state

    async def execute_async(self, data: Any, task_id: Optional[str] = None) -> PipelineState:
        """Execute all steps in the pipeline asynchronously."""
        t_id = task_id or f"TASK-{len(self.history)+1:04d}"
        if self.event_bus:
            self.event_bus.emit("pipeline:start", {"pipeline": self.name, "task_id": t_id, "data": str(data)[:100], "mode": "async"})
        
        state = PipelineState(task_id=t_id, data=data)
        state.status = PipelineStatus.PROCESSING
        self.history.append(state)
        
        return await self._run_async(state)

    async def resume_async(self, task_id: str, feedback: Optional[str] = None) -> PipelineState:
        """Resume a suspended or awaiting_approval task."""
        state = next((s for s in self.history if s.task_id == task_id), None)
        if not state:
            raise ValueError(f"Task ID {task_id} not found in history")
        
        if state.status not in [PipelineStatus.SUSPENDED, PipelineStatus.AWAITING_APPROVAL]:
            print(f"[WARNING] Task {task_id} is in status {state.status}, not suspended.")
            return state

        print(f"\n[RESUME] Resuming pipeline async: {self.name} (Task: {task_id})")
        state.status = PipelineStatus.PROCESSING
        state.feedback = feedback
        
        return await self._run_async(state)

    async def _run_async(self, state: PipelineState) -> PipelineState:
        """Internal runner for async execution."""
        try:
            while state.current_step_index < len(self.steps):
                step_idx = state.current_step_index
                step_name, func = self.steps[step_idx]
                
                # 1. Intervention Point (Before Step)
                if step_name in self.intervention_points:
                    print(f"     [INTERVENTION] Triggering before: {step_name}...")
                    callback = self.intervention_points[step_name]
                    # We pass the state and results for human to tweak
                    await self._invoke_callback(callback, state)

                # 2. Execute Step
                current_data = state.results[self.steps[step_idx-1][0]] if step_idx > 0 else state.data
                
                import inspect
                if inspect.iscoroutinefunction(func):
                    result = await func(current_data)
                else:
                    result = func(current_data)
                
                state.results[step_name] = result

                if self.event_bus:
                    self.event_bus.emit("pipeline:step", {
                        "pipeline": self.name, 
                        "task_id": state.task_id,
                        "step": step_name,
                        "result": str(result)[:200]
                    })
                state.current_step_index += 1
                state.updated_at = datetime.now()

                # 3. Checkpoint (After Step)
                if step_name in self.checkpoints:
                    approval_req = self.checkpoints[step_name]
                    if approval_req:
                        if self.event_bus:
                            self.event_bus.emit("pipeline:checkpoint", {"pipeline": self.name, "task_id": state.task_id, "step": step_name})
                        state.status = PipelineStatus.AWAITING_APPROVAL
                        return state
                    else:
                        state.status = PipelineStatus.SUSPENDED
                        return state
            
            state.status = PipelineStatus.COMPLETED
            if self.event_bus:
                self.event_bus.emit("pipeline:complete", {"pipeline": self.name, "task_id": state.task_id})
            
        except Exception as e:
            state.status = PipelineStatus.ERROR
            state.errors.append(str(e))
            state.updated_at = datetime.now()
            print(f"[ERROR] Pipeline '{self.name}' failed: {e}")
            
        return state

    async def _invoke_callback(self, callback: Callable, state: PipelineState):
        """Invoke intervention callback."""
        import inspect
        if inspect.iscoroutinefunction(callback):
            await callback(state)
        else:
            callback(state)
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        if not self.history:
            return {"total_processed": 0}
        
        statuses = [state.status for state in self.history]
        return {
            "total_processed": len(self.history),
            "completed": statuses.count(PipelineStatus.COMPLETED),
            "errors": statuses.count(PipelineStatus.ERROR),
            "success_rate": (statuses.count(PipelineStatus.COMPLETED) / len(statuses) * 100) if statuses else 0
        }
