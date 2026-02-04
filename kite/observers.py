import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

class EventFileLogger:
    """Standard JSON file logger for EventBus events."""
    def __init__(self, trace_file: str):
        if not os.path.isabs(trace_file):
            trace_file = os.path.join(os.getcwd(), trace_file)
        self.trace_file = trace_file
        # Initialize file as an empty list
        with open(self.trace_file, "w") as f:
            f.write("[\n]")
    
    def on_event(self, event: str, data: Any):
        """Append event to the JSON list file."""
        try:
            # Atomic-ish append to JSON array
            with open(self.trace_file, "rb+") as f:
                f.seek(-2, os.SEEK_END)
                pos = f.tell()
                f.truncate()
                if pos > 2: f.write(b",\n")
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event": event,
                    "data": self._sanitize(data)
                }
                json_str = json.dumps(log_entry, indent=4)
                f.write(json_str.encode('utf-8'))
                f.write(b"\n]")
        except Exception:
            pass

    def _sanitize(self, data):
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize(v) for v in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)

class StateTracker:
    """Standard run state tracker that persists to JSON."""
    def __init__(self, session_file: str, event_map: Dict[str, str] = None):
        if not os.path.isabs(session_file):
            session_file = os.path.join(os.getcwd(), session_file)
        self.session_file = session_file
        # Default map for common events
        self.event_map = event_map or {
            "pipeline:lead_result": "leads",
            "agent:complete": "results"
        }
        self.data = {
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        # Initialize collections based on event_map
        for key in self.event_map.values():
            if key not in self.data:
                self.data[key] = []

    def on_event(self, event: str, data: Any):
        if event in self.event_map:
            collection_name = self.event_map[event]
            if collection_name in self.data:
                # Deduplicate if data has 'name' or 'id'
                if isinstance(data, dict) and "name" in data:
                    if any(item.get("name") == data["name"] for item in self.data[collection_name]):
                        return
                
                self.data[collection_name].append(data)
                self.save()
        elif event == "pipeline:complete":
            self.data["status"] = "completed"
            self.data["end_time"] = datetime.now().isoformat()
            self.save()

    def save(self):
        try:
            with open(self.session_file, "w") as f:
                json.dump(self.data, f, indent=4)
        except Exception:
            pass

class MarkdownReporter:
    """Standard markdown report generator for real-time updates."""
    def __init__(self, output_file: str, title: str = "Execution Report"):
        if not os.path.isabs(output_file):
            output_file = os.path.join(os.getcwd(), output_file)
        self.output_file = output_file
        with open(self.output_file, "w") as f:
            f.write(f"# {title}\n")
            f.write(f"*Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("Standardized framework report.\n\n---\n\n")

    def append(self, content: str):
        """Append a section to the report."""
        try:
            with open(self.output_file, "a") as f:
                f.write(content + "\n\n")
        except Exception:
            pass
