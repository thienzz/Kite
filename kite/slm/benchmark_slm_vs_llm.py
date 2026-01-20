"""
SLM vs LLM Benchmark Tool
Based on Chapter 1.3: Sub-Agents (The Specialists & SLMs)

Comprehensive benchmark comparing SLMs vs GPT-4 on specialized tasks.

From book:
"For specialized tasks, SLMs can be 50-100x faster and cheaper 
while being more accurate because they can't be distracted."

Run: python benchmark_slm_vs_llm.py
"""

import time
import statistics
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

class TaskType(Enum):
    """Types of tasks to benchmark."""
    SQL_GENERATION = "sql_generation"
    CLASSIFICATION = "classification"
    CODE_REVIEW = "code_review"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    task_type: TaskType
    model: str
    latency_ms: float
    cost_per_1k: float
    accuracy: float
    success: bool
    error: str = None


@dataclass
class BenchmarkSummary:
    """Summary comparing SLM vs LLM."""
    task_type: TaskType
    slm_avg_latency: float
    llm_avg_latency: float
    speedup: float
    slm_cost: float
    llm_cost: float
    cost_reduction: float
    slm_accuracy: float
    llm_accuracy: float


# ============================================================================
# MOCK MODELS (For demonstration)
# ============================================================================

class MockSLM:
    """Mock SLM for benchmarking."""
    
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        
        # SLM characteristics (from book)
        self.latencies = {
            TaskType.SQL_GENERATION: 40,   # 40ms
            TaskType.CLASSIFICATION: 5,     # 5ms
            TaskType.CODE_REVIEW: 100       # 100ms
        }
        
        self.costs = {
            TaskType.SQL_GENERATION: 0.0006,
            TaskType.CLASSIFICATION: 0.00001,
            TaskType.CODE_REVIEW: 0.0005
        }
        
        self.accuracies = {
            TaskType.SQL_GENERATION: 0.95,
            TaskType.CLASSIFICATION: 0.97,
            TaskType.CODE_REVIEW: 0.93
        }
    
    def execute(self) -> BenchmarkResult:
        """Execute task and return result."""
        # Simulate latency
        base_latency = self.latencies[self.task_type]
        actual_latency = base_latency * (0.9 + 0.2 * time.time() % 1)  # Add variance
        
        time.sleep(actual_latency / 1000)  # Convert to seconds
        
        return BenchmarkResult(
            task_type=self.task_type,
            model="SLM (Llama 3.1 8B)",
            latency_ms=actual_latency,
            cost_per_1k=self.costs[self.task_type],
            accuracy=self.accuracies[self.task_type],
            success=True
        )


class MockLLM:
    """Mock LLM (GPT-4) for benchmarking."""
    
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        
        # LLM characteristics
        self.latencies = {
            TaskType.SQL_GENERATION: 2000,   # 2000ms
            TaskType.CLASSIFICATION: 500,     # 500ms
            TaskType.CODE_REVIEW: 1000        # 1000ms
        }
        
        self.costs = {
            TaskType.SQL_GENERATION: 0.03,
            TaskType.CLASSIFICATION: 0.001,
            TaskType.CODE_REVIEW: 0.005
        }
        
        self.accuracies = {
            TaskType.SQL_GENERATION: 0.90,   # Lower - can be distracted
            TaskType.CLASSIFICATION: 0.92,    # Lower - overcomplicates
            TaskType.CODE_REVIEW: 0.88        # Lower - style vs substance
        }
    
    def execute(self) -> BenchmarkResult:
        """Execute task and return result."""
        # Simulate latency
        base_latency = self.latencies[self.task_type]
        actual_latency = base_latency * (0.9 + 0.2 * time.time() % 1)
        
        time.sleep(actual_latency / 1000)
        
        return BenchmarkResult(
            task_type=self.task_type,
            model="LLM (GPT-4)",
            latency_ms=actual_latency,
            cost_per_1k=self.costs[self.task_type],
            accuracy=self.accuracies[self.task_type],
            success=True
        )


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """
    Run benchmarks comparing SLM vs LLM.
    
    Measures:
    - Latency (speed)
    - Cost (efficiency)
    - Accuracy (quality)
    
    Example:
        runner = BenchmarkRunner()
        results = runner.run_all_benchmarks(iterations=10)
        runner.print_report(results)
    """
    
    def __init__(self):
        self.task_types = list(TaskType)
        print("[OK] Benchmark Runner initialized")
        print(f"  Tasks: {len(self.task_types)}")
    
    def run_benchmark(self, task_type: TaskType, iterations: int = 10) -> BenchmarkSummary:
        """
        Run benchmark for a specific task.
        
        Args:
            task_type: Type of task to benchmark
            iterations: Number of iterations per model
            
        Returns:
            Benchmark summary
        """
        print(f"\n  Benchmarking: {task_type.value}")
        print(f"   Running {iterations} iterations per model...")
        
        # Run SLM
        slm = MockSLM(task_type)
        slm_results = []
        
        for i in range(iterations):
            result = slm.execute()
            slm_results.append(result)
        
        # Run LLM
        llm = MockLLM(task_type)
        llm_results = []
        
        for i in range(iterations):
            result = llm.execute()
            llm_results.append(result)
        
        # Calculate statistics
        slm_latencies = [r.latency_ms for r in slm_results]
        llm_latencies = [r.latency_ms for r in llm_results]
        
        slm_avg_latency = statistics.mean(slm_latencies)
        llm_avg_latency = statistics.mean(llm_latencies)
        speedup = llm_avg_latency / slm_avg_latency
        
        slm_cost = slm_results[0].cost_per_1k
        llm_cost = llm_results[0].cost_per_1k
        cost_reduction = (llm_cost - slm_cost) / llm_cost * 100
        
        slm_accuracy = statistics.mean([r.accuracy for r in slm_results])
        llm_accuracy = statistics.mean([r.accuracy for r in llm_results])
        
        summary = BenchmarkSummary(
            task_type=task_type,
            slm_avg_latency=slm_avg_latency,
            llm_avg_latency=llm_avg_latency,
            speedup=speedup,
            slm_cost=slm_cost,
            llm_cost=llm_cost,
            cost_reduction=cost_reduction,
            slm_accuracy=slm_accuracy,
            llm_accuracy=llm_accuracy
        )
        
        print(f"   [OK] Complete: {speedup:.0f}x speedup, {cost_reduction:.0f}% cost reduction")
        
        return summary
    
    def run_all_benchmarks(self, iterations: int = 10) -> List[BenchmarkSummary]:
        """
        Run all benchmarks.
        
        Args:
            iterations: Number of iterations per task
            
        Returns:
            List of benchmark summaries
        """
        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE BENCHMARKS")
        print('='*70)
        
        summaries = []
        
        for task_type in self.task_types:
            summary = self.run_benchmark(task_type, iterations)
            summaries.append(summary)
        
        return summaries
    
    def print_report(self, summaries: List[BenchmarkSummary]):
        """
        Print comprehensive benchmark report.
        
        Args:
            summaries: Benchmark results
        """
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print('='*70)
        
        for summary in summaries:
            print(f"\n[CHART] {summary.task_type.value.upper()}")
            print("   " + " " * 66)
            
            print(f"\n   Latency:")
            print(f"      SLM:     {summary.slm_avg_latency:.1f}ms")
            print(f"      LLM:     {summary.llm_avg_latency:.1f}ms")
            print(f"      Speedup: {summary.speedup:.0f}x faster  ")
            
            print(f"\n   Cost (per 1,000 calls):")
            print(f"      SLM:       ${summary.slm_cost:.4f}")
            print(f"      LLM:       ${summary.llm_cost:.4f}")
            print(f"      Reduction: {summary.cost_reduction:.0f}% cheaper  ")
            
            print(f"\n   Accuracy:")
            print(f"      SLM:  {summary.slm_accuracy:.1%}")
            print(f"      LLM:  {summary.llm_accuracy:.1%}")
            
            accuracy_diff = summary.slm_accuracy - summary.llm_accuracy
            if accuracy_diff > 0:
                print(f"      SLM is {accuracy_diff:.1%} more accurate! [OK]")
            elif accuracy_diff < 0:
                print(f"      LLM is {abs(accuracy_diff):.1%} more accurate")
            else:
                print(f"      Equal accuracy")
        
        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print('='*70)
        
        avg_speedup = statistics.mean([s.speedup for s in summaries])
        avg_cost_reduction = statistics.mean([s.cost_reduction for s in summaries])
        
        print(f"\nAverage across all tasks:")
        print(f"  Speed improvement:  {avg_speedup:.0f}x faster")
        print(f"  Cost reduction:     {avg_cost_reduction:.0f}% cheaper")
        
        # ROI calculation
        print(f"\n{'='*70}")
        print("ROI CALCULATION")
        print('='*70)
        
        print(f"\nFor 100,000 calls/month across all tasks:")
        
        total_slm_cost = sum(s.slm_cost * 100 for s in summaries)
        total_llm_cost = sum(s.llm_cost * 100 for s in summaries)
        monthly_savings = total_llm_cost - total_slm_cost
        
        print(f"  SLM total:  ${total_slm_cost:.2f}/month")
        print(f"  LLM total:  ${total_llm_cost:.2f}/month")
        print(f"  Savings:    ${monthly_savings:.2f}/month")
        print(f"  Annual:     ${monthly_savings * 12:.2f}/year")


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("SLM vs LLM BENCHMARK TOOL")
    print("=" * 70)
    print("\nBased on Chapter 1.3: Sub-Agents (The Specialists & SLMs)")
    print("\nComparing specialized SLMs vs general GPT-4:")
    print("    SQL Generation")
    print("    Intent Classification")
    print("    Code Review")
    print("=" * 70)
    
    # Run benchmarks
    runner = BenchmarkRunner()
    summaries = runner.run_all_benchmarks(iterations=5)
    
    # Print report
    runner.print_report(summaries)
    
    # Key insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS FROM BENCHMARKS")
    print('='*70)
    print("""
1. SPEED WINS
   [OK] SLMs are 10-50x faster
   [OK] Suitable for real-time applications
   [OK] Can handle high volume

2. COST WINS
   [OK] 95-99% cost reduction
   [OK] ROI measured in weeks
   [OK] Scales economically

3. ACCURACY WINS (for specialized tasks)
   [OK] SLMs more focused
   [OK] Can't be distracted
   [OK] Higher precision on target task

4. WHEN SLMs WIN:
   [OK] Task is well-defined
   [OK] High volume (>1000/day)
   [OK] Cost sensitive
   [OK] Latency critical
   [OK] Need consistency

5. WHEN GPT-4 WINS:
   [OK] Open-ended tasks
   [OK] Need general knowledge
   [OK] Creative work
   [OK] Low volume
   [OK] Exploration phase

RECOMMENDATION (From Book):
"Use SLMs for production workloads where you know 
exactly what you need. Use GPT-4 for exploration 
and complex reasoning."

DEPLOYMENT STRATEGY:
1. Start with GPT-4 (exploration)
2. Identify patterns (what tasks repeat?)
3. Fine-tune SLM (specialize)
4. Deploy SLM (production)
5. Keep GPT-4 for edge cases
    """)


if __name__ == "__main__":
    demo()
