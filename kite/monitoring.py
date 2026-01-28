"""
Monitoring & Observability System
Complete production monitoring with metrics, tracing, and alerting.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from collections import defaultdict, deque
import threading

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        start_http_server, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """
    Production-grade metrics collection.
    
    Tracks:
    - Request counts
    - Latencies
    - Error rates
    - Resource usage
    - Custom metrics
    """
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # In-memory metrics
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'total_latency': 0,
            'min_latency': float('inf'),
            'max_latency': 0,
            'outcomes': defaultdict(int)
        })
        
        self.lock = threading.RLock()
        
        # History for Dashboard
        self.request_history = deque(maxlen=1000)
        self.error_logs = deque(maxlen=1000)
        self.max_history = 1000
        
        # Prometheus metrics
        if self.enable_prometheus:
            self._init_prometheus()
    
    def _init_prometheus(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            'kite_requests_total',
            'Total requests',
            ['component', 'operation']
        )
        
        self.request_latency = Histogram(
            'kite_request_latency_seconds',
            'Request latency',
            ['component', 'operation']
        )
        
        self.error_counter = Counter(
            'kite_errors_total',
            'Total errors',
            ['component', 'operation', 'error_type']
        )
        
        # Resource metrics
        self.active_requests = Gauge(
            'kite_active_requests',
            'Active requests',
            ['component']
        )
        
        # LLM metrics
        self.llm_tokens = Counter(
            'kite_llm_tokens_total',
            'Total LLM tokens',
            ['provider', 'model', 'type']
        )
        
        self.llm_cost = Counter(
            'kite_llm_cost_usd',
            'Total LLM cost in USD',
            ['provider', 'model']
        )
        
        # Memory metrics
        self.memory_operations = Counter(
            'kite_memory_operations_total',
            'Memory operations',
            ['type', 'operation']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'kite_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open)',
            ['component']
        )
        
        # Info
        self.info = Info(
            'kite',
            'Framework information'
        )
        self.info.info({
            'version': '1.0.0',
            'python_version': '3.11'
        })
    
    def record_request(self, component: str, operation: str, 
                      latency: float, success: bool = True,
                      error_type: Optional[str] = None):
        """Record a component request with timing and status."""
        key = f"{component}.{operation}"
        with self.lock:
            # Update metrics
            data = self.metrics[key]
            data['count'] += 1
            data['total_latency'] += latency
            data['min_latency'] = min(data['min_latency'], latency)
            data['max_latency'] = max(data['max_latency'], latency)
            
            if not success:
                data['errors'] += 1
            
            # Update history
            self.request_history.append({
                'timestamp': datetime.now(),
                'component': component,
                'operation': operation,
                'latency': latency,
                'success': success,
                'error_type': error_type
            })
            
            if not success and error_type:
                self.error_logs.append({
                    'timestamp': datetime.now(),
                    'component': component,
                    'error': error_type
                })
        
        # History
        with self.lock:
            self.request_history.append({
                'component': component,
                'operation': operation,
                'latency': latency,
                'success': success,
                'error_type': error_type,
                'timestamp': time.time()
            })
            if len(self.request_history) > self.max_history:
                self.request_history.pop(0)
            
            if not success:
                self.error_logs.append({
                    'component': component,
                    'operation': operation,
                    'error': error_type,
                    'timestamp': time.time()
                })
                if len(self.error_logs) > self.max_history:
                    self.error_logs.pop(0)
        
        # Prometheus
        if self.enable_prometheus:
            self.request_counter.labels(
                component=component,
                operation=operation
            ).inc()
            
            self.request_latency.labels(
                component=component,
                operation=operation
            ).observe(latency)
            
            if not success:
                self.error_counter.labels(
                    component=component,
                    operation=operation,
                    error_type=error_type or 'unknown'
                ).inc()

    def record_outcome(self, component: str, outcome_type: str):
        """Record a domain-specific outcome for a component."""
        with self.lock:
            self.metrics[component]['outcomes'][outcome_type] += 1
    
    def record_llm_usage(self, provider: str, model: str,
                        prompt_tokens: int, completion_tokens: int,
                        cost: float = 0):
        """Record LLM usage."""
        if self.enable_prometheus:
            self.llm_tokens.labels(
                provider=provider,
                model=model,
                type='prompt'
            ).inc(prompt_tokens)
            
            self.llm_tokens.labels(
                provider=provider,
                model=model,
                type='completion'
            ).inc(completion_tokens)
            
            if cost > 0:
                self.llm_cost.labels(
                    provider=provider,
                    model=model
                ).inc(cost)
    
    def record_memory_operation(self, mem_type: str, operation: str):
        """Record memory operation."""
        if self.enable_prometheus:
            self.memory_operations.labels(
                type=mem_type,
                operation=operation
            ).inc()
    
    def set_circuit_breaker_state(self, component: str, is_open: bool):
        """Set circuit breaker state."""
        if self.enable_prometheus:
            self.circuit_breaker_state.labels(
                component=component
            ).set(1 if is_open else 0)
    
    def get_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        with self.lock:
            return dict(self.metrics)
    
    def start_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        if self.enable_prometheus:
            start_http_server(port)
            logging.info(f"Metrics server started on port {port}")

    def get_history(self) -> list:
        """Get recent request history."""
        with self.lock:
            return list(self.request_history)

    def get_error_logs(self) -> list:
        """Get recent error logs."""
        with self.lock:
            return list(self.error_logs)
            
    def get_summary(self) -> Dict:
        """Get high-level system summary."""
        history = self.get_history()
        total_requests = len(self.request_history)
        total_errors = len(self.error_logs)
        
        if not history:
            return {
                "status": "idle",
                "success_rate": 1.0,
                "avg_latency": 0.0,
                "total_requests": total_requests,
                "total_errors": total_errors
            }
            
        recent = history[-100:]
        success_rate = sum(1 for r in recent if r['success']) / len(recent) if recent else 0
        avg_latency = sum(r['latency'] for r in recent) / len(recent) if recent else 0
        
        return {
            "status": "healthy" if success_rate > 0.9 else "degraded",
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_requests": total_requests,
            "total_errors": total_errors
        }

    def get_detailed_report(self) -> str:
        """Generate a human-readable detailed report of all metrics."""
        with self.lock:
            if not self.metrics:
                return "No metrics recorded."
            
            report = []
            report.append("\n" + "="*60)
            report.append("  📊 KITE SYSTEM PERFORMANCE REPORT")
            report.append("="*60)
            
            # 1. Summary Metrics
            summary = self.get_summary()
            report.append(f"Status: {summary['status'].upper()}")
            report.append(f"Total Calls: {summary['total_requests']}")
            report.append(f"Success Rate: {summary['success_rate']:.1%}")
            # report.append(f"Avg Latency: {summary['avg_latency']:.3f}s")
            report.append("-" * 60)
            
            # 2. Per-Component breakdown
            report.append(f"{'Component':<20} | {'Calls':<6} | {'Errors':<6} | {'Avg Latency':<10} | {'Outcomes'}")
            report.append("-" * 60)
            
            for key, data in sorted(self.metrics.items()):
                avg_l = data['total_latency'] / data['count'] if data['count'] > 0 else 0
                outcomes_str = ", ".join([f"{k}:{v}" for k, v in data['outcomes'].items()])
                report.append(f"{key:<20} | {data['count']:<6} | {data['errors']:<6} | {avg_l:<10.3f}s | {outcomes_str}")
            
            report.append("="*60 + "\n")
            return "\n".join(report)


class Tracer:
    """
    Distributed tracing for agent operations.
    Tracks operation flow and timing.
    """
    
    def __init__(self):
        self.traces = []
        self.current_trace = None
        self.lock = threading.Lock()
    
    def start_trace(self, trace_id: str, operation: str, metadata: Dict = None):
        """Start a new trace."""
        trace = {
            'trace_id': trace_id,
            'operation': operation,
            'start_time': time.time(),
            'metadata': metadata or {},
            'spans': []
        }
        
        with self.lock:
            self.traces.append(trace)
            self.current_trace = trace
        
        return trace
    
    def add_span(self, name: str, metadata: Dict = None):
        """Add a span to current trace."""
        if not self.current_trace:
            return
        
        span = {
            'name': name,
            'start_time': time.time(),
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.current_trace['spans'].append(span)
        
        return span
    
    def end_span(self, span: Dict):
        """End a span."""
        span['end_time'] = time.time()
        span['duration'] = span['end_time'] - span['start_time']
    
    def end_trace(self):
        """End current trace."""
        if self.current_trace:
            self.current_trace['end_time'] = time.time()
            self.current_trace['duration'] = (
                self.current_trace['end_time'] - 
                self.current_trace['start_time']
            )
            self.current_trace = None
    
    def get_traces(self, limit: int = 100) -> list:
        """Get recent traces."""
        with self.lock:
            return self.traces[-limit:]


def monitor(component: str, operation: str):
    """
    Decorator to monitor function execution.
    
    Usage:
        @monitor('llm', 'chat')
        def chat(messages):
            return llm.chat(messages)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_type = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                latency = time.time() - start_time
                
                # Get metrics collector from args if available
                if args and hasattr(args[0], 'metrics'):
                    metrics = args[0].metrics
                    metrics.record_request(
                        component, operation, latency,
                        success, error_type
                    )
        
        return wrapper
    return decorator


class HealthCheck:
    """
    Health check system for services.
    """
    
    def __init__(self):
        self.checks = {}
        self.lock = threading.Lock()
    
    def register(self, name: str, check_func: Callable):
        """Register a health check."""
        with self.lock:
            self.checks[name] = check_func
    
    def run_checks(self) -> Dict:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'success': result
                }
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Overall status
        all_healthy = all(
            r['status'] == 'healthy' 
            for r in results.values()
        )
        
        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'checks': results
        }


class AlertManager:
    """
    Alert manager for threshold-based alerts.
    """
    
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self.alerts = []
        self.thresholds = {}
    
    def set_threshold(self, metric: str, threshold: float, 
                     comparison: str = '>'):
        """Set alert threshold."""
        self.thresholds[metric] = {
            'threshold': threshold,
            'comparison': comparison
        }
    
    def check_alerts(self) -> list:
        """Check for threshold violations."""
        alerts = []
        metrics_data = self.metrics.get_metrics()
        
        for metric, config in self.thresholds.items():
            if metric in metrics_data:
                value = metrics_data[metric]['count']
                threshold = config['threshold']
                
                if config['comparison'] == '>' and value > threshold:
                    alerts.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'message': f"{metric} exceeded threshold: {value} > {threshold}"
                    })
        
        return alerts


# Global instances
_metrics = None
_tracer = None
_health = None
_alerts = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def get_tracer() -> Tracer:
    """Get global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_health_check() -> HealthCheck:
    """Get global health check."""
    global _health
    if _health is None:
        _health = HealthCheck()
    return _health


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _alerts
    if _alerts is None:
        _alerts = AlertManager(get_metrics())
    return _alerts


if __name__ == "__main__":
    # Example usage
    print("Monitoring System Example\n")
    
    metrics = get_metrics()
    tracer = get_tracer()
    health = get_health_check()
    
    # Record some metrics
    metrics.record_request('llm', 'chat', 0.5, success=True)
    metrics.record_request('llm', 'chat', 0.3, success=True)
    metrics.record_request('llm', 'chat', 1.2, success=False, error_type='Timeout')
    
    # Start a trace
    trace = tracer.start_trace('trace-1', 'agent_run')
    span = tracer.add_span('llm_call')
    time.sleep(0.1)
    tracer.end_span(span)
    tracer.end_trace()
    
    # Health check
    health.register('llm', lambda: True)
    health.register('memory', lambda: True)
    
    print("Metrics:", metrics.get_metrics())
    print("\nTraces:", len(tracer.get_traces()))
    print("\nHealth:", health.run_checks())
    
    print("\n[OK] Monitoring system working")
