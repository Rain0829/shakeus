"""Deployment values from the environment (.env or process env).

This repo is the Pi-side alarm/stream service. A separate sleep-tracker
frontend/backend can be added later; that service would consume HTTP/WebSocket
data this process already exposes (e.g. stream state) or new small endpoints
you add here—keep secrets in env only, not in code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Required: MAC_IP, MAC_PORT, GOOGLE_HOME_IP, SONGS_BASE_URL,
# GEMINI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID
# Optional: GEMINI_MODEL (e.g. gemini-2.5-flash)

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


def _require_str(key: str) -> str:
    val = os.environ.get(key)
    if val is None or not str(val).strip():
        raise RuntimeError(
            f"Missing or empty environment variable {key}. "
            "Set it in a .env file in the project root (see env_config module docstring)."
        )
    return val.strip()


def _require_int(key: str) -> int:
    raw = _require_str(key)
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {key} must be an integer, got {raw!r}") from e


MAC_IP = _require_str("MAC_IP")
MAC_PORT = _require_int("MAC_PORT")
GOOGLE_HOME_IP = _require_str("GOOGLE_HOME_IP")

SONGS_BASE_URL = _require_str("SONGS_BASE_URL").rstrip("/")

GEMINI_API_KEY = _require_str("GEMINI_API_KEY")
# Optional: google-genai model id (e.g. gemini-2.5-flash); default is set in voice_generator
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip()

ELEVENLABS_API_KEY = _require_str("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _require_str("ELEVENLABS_VOICE_ID")
