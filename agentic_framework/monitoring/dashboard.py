"""
Monitoring Dashboard API
Serves metrics, traces, and system health for the UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from agentic_framework.monitoring import get_metrics, get_tracer, get_health_check

app = FastAPI(title="AgenticAI Monitoring Dashboard")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/overview")
async def get_overview():
    """System health overview."""
    metrics = get_metrics()
    health = get_health_check()
    
    return {
        "summary": metrics.get_summary(),
        "health": health.run_checks(),
        "active_components": ["llm", "embeddings", "vector_memory", "cache"]
    }

@app.get("/metrics")
async def get_detailed_metrics():
    """Detailed performance metrics."""
    metrics = get_metrics()
    return metrics.get_metrics()

@app.get("/traces")
async def get_traces(limit: int = 50):
    """Recent execution traces."""
    tracer = get_tracer()
    return tracer.get_traces(limit=limit)

@app.get("/logs")
async def get_logs():
    """Recent error logs."""
    metrics = get_metrics()
    return metrics.get_error_logs()

@app.get("/agents")
async def get_agent_stats():
    """Statistics per agent."""
    metrics = get_metrics()
    all_metrics = metrics.get_metrics()
    
    agent_stats = {}
    for key, data in all_metrics.items():
        if key.startswith("agent."):
            agent_name = key.split(".")[1]
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {"requests": 0, "errors": 0, "latency": 0}
            
            agent_stats[agent_name]["requests"] += data["count"]
            agent_stats[agent_name]["errors"] += data["errors"]
            agent_stats[agent_name]["latency"] = data["total_latency"] / data["count"] if data["count"] > 0 else 0
            
    return agent_stats

def start_dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Helper to start the dashboard server."""
    print(f"[START] Starting Monitoring Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_dashboard()
