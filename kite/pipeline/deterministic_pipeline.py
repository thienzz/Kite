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
    
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: List[tuple[str, Callable]] = []
        self.history: List[PipelineState] = []
        print(f"[OK] Pipeline '{self.name}' initialized")
    
    def add_step(self, name: str, func: Callable):
        """Add a step to the pipeline."""
        self.steps.append((name, func))
        print(f"  [OK] Added step: {name}")
    
    def execute(self, data: Any, task_id: Optional[str] = None) -> PipelineState:
        """Execute all steps in the pipeline sequentially."""
        t_id = task_id or f"TASK-{len(self.history)+1:04d}"
        print(f"\n[START] Executing pipeline: {self.name} (Task: {t_id})")
        
        state = PipelineState(task_id=t_id, data=data)
        state.status = PipelineStatus.PROCESSING
        
        current_data = data
        try:
            for step_name, func in self.steps:
                print(f"     Executing step: {step_name}...")
                # Support both sync and async if needed, but keeping it simple for now
                import inspect
                if inspect.iscoroutinefunction(func):
                    # This requires the caller to be async, but for framework 
                    # compatibility we might need a better way. 
                    # For now, assuming sync or handled by caller.
                    raise RuntimeError(f"Step '{step_name}' is async. Use execute_async().")
                
                current_data = func(current_data)
                state.results[step_name] = current_data
                state.updated_at = datetime.now()
            
            state.status = PipelineStatus.COMPLETED
            print(f"[OK] Pipeline '{self.name}' completed successfully")
            
        except Exception as e:
            state.status = PipelineStatus.ERROR
            state.errors.append(str(e))
            state.updated_at = datetime.now()
            print(f"[ERROR] Pipeline '{self.name}' failed at step {step_name if 'step_name' in locals() else 'unknown'}: {e}")
            
        self.history.append(state)
        return state

    async def execute_async(self, data: Any, task_id: Optional[str] = None) -> PipelineState:
        """Execute all steps in the pipeline asynchronously."""
        t_id = task_id or f"TASK-{len(self.history)+1:04d}"
        print(f"\n[START] Executing pipeline async: {self.name} (Task: {t_id})")
        
        state = PipelineState(task_id=t_id, data=data)
        state.status = PipelineStatus.PROCESSING
        
        current_data = data
        try:
            for step_name, func in self.steps:
                print(f"     Executing step (async): {step_name}...")
                import inspect
                if inspect.iscoroutinefunction(func):
                    current_data = await func(current_data)
                else:
                    current_data = func(current_data)
                
                state.results[step_name] = current_data
                state.updated_at = datetime.now()
            
            state.status = PipelineStatus.COMPLETED
            print(f"[OK] Pipeline '{self.name}' completed successfully")
            
        except Exception as e:
            state.status = PipelineStatus.ERROR
            state.errors.append(str(e))
            state.updated_at = datetime.now()
            print(f"[ERROR] Pipeline '{self.name}' failed: {e}")
            
        self.history.append(state)
        return state
    
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
