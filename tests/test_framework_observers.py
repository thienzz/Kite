import asyncio
import os
import json
from kite.core import Kite

async def test_observers():
    print("Testing Kite Observers...")
    ai = Kite()
    
    # Enable tracing and state tracking
    trace_file = "test_trace.json"
    session_file = "test_session.json"
    
    # Cleanup
    if os.path.exists(trace_file): os.remove(trace_file)
    if os.path.exists(session_file): os.remove(session_file)
    
    print(f"Enabling tracing: {trace_file}")
    ai.enable_tracing(trace_file)
    
    print(f"Enabling state tracking: {session_file}")
    ai.enable_state_tracking(session_file, {
        "test:result": "results"
    })
    
    # Emit some events
    print("Emitting events...")
    ai.event_bus.emit("test:info", {"msg": "hello"})
    ai.event_bus.emit("test:result", {"name": "Test Lead", "score": 100})
    ai.event_bus.emit("pipeline:complete", {})
    
    # Small sleep to ensure sync file writes complete (though they are blocking)
    await asyncio.sleep(0.5)
    
    # Check trace file
    if not os.path.exists(trace_file):
        print("❌ FAILED: Trace file not created")
        return
        
    with open(trace_file, "r") as f:
        try:
            trace = json.load(f)
            print(f"✅ Trace file valid: {len(trace)} events")
            assert len(trace) >= 3
        except Exception as e:
            print(f"❌ FAILED: Trace file JSON invalid: {e}")
            return
        
    # Check session file
    if not os.path.exists(session_file):
        print("❌ FAILED: Session file not created")
        return

    with open(session_file, "r") as f:
        try:
            session = json.load(f)
            print("✅ Session file JSON valid")
            assert session["status"] == "completed"
            assert len(session["results"]) == 1
            assert session["results"][0]["name"] == "Test Lead"
            print("✅ Session state correct")
        except Exception as e:
            print(f"❌ FAILED: Session file state invalid: {e}")
            return

    print("\n🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_observers())
