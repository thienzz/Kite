"""
Reactive Pipeline Pattern
Level 2 Autonomy: Event-driven, concurrent processing with workers.

Flow: Producer -> Stage 1 (N workers) -> Stage 2 (M workers) -> ... -> Result
"""

import asyncio
import logging
import inspect
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from .deterministic_pipeline import PipelineStatus, PipelineState

@dataclass
class ReactiveStage:
    name: str
    func: Callable
    workers: int
    input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    output_queue: Optional[asyncio.Queue] = None
    tasks: List[asyncio.Task] = field(default_factory=list)

class ReactivePipeline:
    """
    A reactive, streaming data pipeline with parallel workers.
    
    Each stage runs N workers in parallel, reading from an input queue
    and passing results to the next stage's queue.
    """
    
    def __init__(self, name: str = "reactive_pipeline", event_bus = None):
        self.name = name
        self.event_bus = event_bus
        self.stages: List[ReactiveStage] = []
        self.logger = logging.getLogger(f"Pipeline:{name}")
        self.history: List[PipelineState] = []
        self._running_tasks = []
        
        if self.event_bus:
            self.event_bus.emit("pipeline:init", {"pipeline": self.name, "type": "reactive"})

    def add_stage(self, name: str, func: Callable, workers: int = 1):
        """Add a processing stage with a specific number of workers."""
        stage = ReactiveStage(name=name, func=func, workers=workers)
        
        # Connect output of previous stage to input of this stage
        if self.stages:
            self.stages[-1].output_queue = stage.input_queue
            
        self.stages.append(stage)
        self.logger.info(f"  [OK] Added stage: {name} (Workers: {workers})")

    async def execute(self, initial_data: Any, task_id: Optional[str] = None):
        """
        Start the pipeline and feed it initial data.
        Note: initial_data can be a single item or a list of items.
        """
        t_id = task_id or f"RTASK-{datetime.now().strftime('%H%M%S')}"
        
        if self.event_bus:
            self.event_bus.emit("pipeline:start", {
                "pipeline": self.name, 
                "task_id": t_id, 
                "mode": "reactive",
                "stages": [s.name for s in self.stages]
            })
            # Emit structure for dashboard
            self.event_bus.emit("pipeline:structure", {
                "pipeline": self.name,
                "task_id": t_id,
                "steps": [s.name for s in self.stages]
            })

        # Start all workers for all stages
        for i, stage in enumerate(self.stages):
            for w_idx in range(stage.workers):
                task = asyncio.create_task(
                    self._worker_loop(stage, t_id),
                    name=f"Worker-{stage.name}-{w_idx}"
                )
                stage.tasks.append(task)
                self._running_tasks.append(task)

        # Feed the first stage
        if isinstance(initial_data, list):
            for item in initial_data:
                await self.stages[0].input_queue.put(item)
        else:
            await self.stages[0].input_queue.put(initial_data)

        self.logger.info(f"Pipeline {self.name} is running...")
        return t_id

    async def _worker_loop(self, stage: ReactiveStage, task_id: str):
        """Internal loop for a single worker in a stage."""
        while True:
            item = await stage.input_queue.get()
            if item is None: # Shutdown signal
                stage.input_queue.task_done()
                break
            
            try:
                if self.event_bus:
                    self.event_bus.emit("pipeline:step_start", {
                        "pipeline": self.name,
                        "task_id": task_id,
                        "step": stage.name,
                        "data": str(item)[:100]
                    })

                # Execute function
                if inspect.iscoroutinefunction(stage.func):
                    result = await stage.func(item)
                else:
                    result = stage.func(item)

                if self.event_bus:
                    self.event_bus.emit("pipeline:step", {
                        "pipeline": self.name,
                        "task_id": task_id,
                        "step": stage.name,
                        "result": str(result)[:200]
                    })

                # Pass to next stage if it exists and result is not None
                if stage.output_queue and result is not None:
                    if isinstance(result, list):
                        for sub_item in result:
                            await stage.output_queue.put(sub_item)
                    else:
                        await stage.output_queue.put(result)

            except Exception as e:
                self.logger.error(f"Error in stage {stage.name}: {e}")
                if self.event_bus:
                    self.event_bus.emit("pipeline:error", {
                        "pipeline": self.name,
                        "task_id": task_id,
                        "step": stage.name,
                        "error": str(e)
                    })
            finally:
                stage.input_queue.task_done()

    async def wait_until_complete(self):
        """Wait for all items to flow through all queues and workers to exit."""
        for stage in self.stages:
            # 1. Wait for all items in the current queue to be PROCESSED
            await stage.input_queue.join()
            
            # 2. Tell all workers in this stage to SHUT DOWN
            for _ in range(stage.workers):
                await stage.input_queue.put(None)
            
            # 3. Wait for these specific workers to finish
            if stage.tasks:
                await asyncio.gather(*stage.tasks)
                
        self.logger.info(f"Pipeline {self.name} completed successfully.")
        
    def stop(self):
        """Force stop all workers."""
        for task in self._running_tasks:
            task.cancel()
