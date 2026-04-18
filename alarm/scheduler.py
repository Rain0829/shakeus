import sys
import time
from pathlib import Path

import requests
import schedule


def _prefer_local_config() -> None:
    """Ensure direct script runs resolve this project's config.py first."""
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "config.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_prefer_local_config()
from config import ALARM_TIME

STREAM_URL = "http://localhost:8000/trigger"

def trigger():
    try:
        requests.post(STREAM_URL, timeout=5)
        print("Alarm triggered!")
    except Exception as e:
        print(f"Failed to reach stream server: {e}")

schedule.every().day.at(ALARM_TIME).do(trigger)
print(f"Alarm set for {ALARM_TIME}. Waiting...")

while True:
    schedule.run_pending()
    time.sleep(10)
