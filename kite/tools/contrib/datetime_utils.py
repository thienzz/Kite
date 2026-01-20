"""
DateTime Utilities - Get current date and time.
"""

from datetime import datetime

def get_current_datetime():
    """
    Get the current date and time.
    
    Returns:
        ISO formatted datetime string.
    """
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A")
    }
