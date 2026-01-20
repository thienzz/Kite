"""
Batch Processor for High Throughput
Implements dynamic batching for LLM and Embedding requests.
"""

import asyncio
import time
from typing import List, Any, Callable, Dict
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    """
    Process requests in batches to improve throughput.
    """
    def __init__(self, 
                 processor_func: Callable, 
                 max_batch_size: int = 10, 
                 wait_ms: int = 50):
        self.processor_func = processor_func
        self.max_batch_size = max_batch_size
        self.wait_ms = wait_ms
        self.queue = []
        self._lock = asyncio.Lock()
        self._loop = asyncio.get_event_loop()

    async def add(self, item: Any) -> Any:
        """Add item to batch and wait for result."""
        future = self._loop.create_future()
        async with self._lock:
            self.queue.append((item, future))
            if len(self.queue) >= self.max_batch_size:
                await self._process_batch()
            elif len(self.queue) == 1:
                # First item in a new batch, start timer
                asyncio.create_task(self._timer_process())
        
        return await future

    async def _timer_process(self):
        """Process batch after timeout if not full."""
        await asyncio.sleep(self.wait_ms / 1000.0)
        async with self._lock:
            if self.queue:
                await self._process_batch()

    async def _process_batch(self):
        """Execute the batch processor function."""
        if not self.queue:
            return
            
        current_batch = self.queue
        self.queue = []
        
        items = [item for item, _ in current_batch]
        futures = [future for _, future in current_batch]
        
        try:
            # Run the actual processing (e.g., API call)
            # This is usually a blocking call wrapped in a thread or an async call
            if asyncio.iscoroutinefunction(self.processor_func):
                results = await self.processor_func(items)
            else:
                # Wrap blocking call
                with ThreadPoolExecutor() as pool:
                    results = await self._loop.run_in_executor(pool, self.processor_func, items)
            
            # Distribute results back to futures
            for i, result in enumerate(results):
                if i < len(futures):
                    futures[i].set_result(result)
                    
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)

class DynamicBatcher:
    """
    Adjusts batch size based on latency.
    """
    def __init__(self, processor_func: Callable):
        self.processor = BatchProcessor(processor_func)
        self.latencies = []

    async def process(self, item: Any) -> Any:
        start_time = time.time()
        result = await self.processor.add(item)
        latency = time.time() - start_time
        
        self.latencies.append(latency)
        if len(self.latencies) > 20:
            avg_latency = sum(self.latencies[-10:]) / 10
            # If latency is too high, reduce batch size
            if avg_latency > 2.0: # 2 seconds
                self.processor.max_batch_size = max(1, self.processor.max_batch_size - 1)
            elif avg_latency < 0.5: # 0.5 seconds
                self.processor.max_batch_size = min(50, self.processor.max_batch_size + 1)
                
        return result
