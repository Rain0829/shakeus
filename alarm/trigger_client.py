import sys
from pathlib import Path

import requests


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
from env_config import MAC_IP, MAC_PORT

def trigger_now():
    url = f"http://{MAC_IP}:{MAC_PORT}/alarm/trigger"
    r = requests.get(url)
    print(f"Response: {r.status_code} {r.text}")

if __name__ == "__main__":
    trigger_now()
