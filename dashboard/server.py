from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import asyncio
from typing import List, Dict
from datetime import datetime

app = FastAPI(title="Kite Monitoring Dashboard")

# Storage for live events
events_log: List[Dict] = []
active_sessions: List[WebSocket] = []

@app.post("/events")
async def receive_event(request: Request):
    data = await request.json()
    # Add server arrival timestamp
    data["received_at"] = datetime.now().isoformat()
    events_log.append(data)
    
    print(f"   [Relay] Received: {data.get('event')} from {data.get('data', {}).get('agent', 'System')}")
    
    # Broadcast to all connected WebSockets
    for ws in active_sessions:
        try:
            await ws.send_json(data)
        except:
            pass
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_sessions.append(websocket)
    try:
        # Send history upon connection
        for event in events_log[-50:]:
            await websocket.send_json(event)
        
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        active_sessions.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("dashboard/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
