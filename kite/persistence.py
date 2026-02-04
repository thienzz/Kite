"""
Persistence utilities for saving and loading application state.
Supports the "Pause & Resume" (Checkpointing) pattern.
"""
import json
import os
from typing import Any, Dict, Optional

class JSONCheckpointer:
    """
    Simple file-based persistence for arbitrary state dictionaries.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def save(self, state: Dict[str, Any]) -> None:
        """Save the state dictionary to JSON file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(state, f, indent=2)
            # print(f"      [Checkpoint] Saved to {self.filepath}")
        except Exception as e:
            print(f"      [Checkpoint Error] Failed to save: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the state dictionary from JSON file. Returns None if not found."""
        if not os.path.exists(self.filepath):
            return None
            
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            print(f"      [Checkpoint] Resuming from {self.filepath}")
            return data
        except Exception as e:
            print(f"      [Checkpoint Error] Corrupt file {self.filepath}: {e}")
            return None

    def clear(self) -> None:
        """Remove the checkpoint file."""
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
