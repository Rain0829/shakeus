import os
from pathlib import Path

import requests
from google import genai

from config import SONGS_DIR
from env_config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
)

# Default Gemini Developer API model (override with GEMINI_MODEL in .env)
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

_ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVEN_TIMEOUT_SEC = 120


def _resolve_output_path(output_filename: str) -> str:
    """Path for written audio: basename → under SONGS_DIR; already under SONGS_DIR or absolute → use as-is."""
    if os.path.isabs(output_filename):
        return output_filename
    norm = Path(output_filename).as_posix()
    songs = Path(SONGS_DIR).as_posix().rstrip("/") + "/"
    if norm.startswith(songs):
        return output_filename
    return str(Path(SONGS_DIR) / output_filename)


class VoiceGenerator:
    def __init__(self):
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_phrase(self, phase="intro"):
        if phase == "intro":
            prompt = (
                "Write a short, punchy 1-sentence intro to wake someone up and hype them up "
                "for a morning dance routine. Sound like an energetic DJ. No emojis."
            )
        else:
            prompt = (
                "Write a short, punchy 1-sentence congratulation for finishing a morning dance break. "
                "Tell them to have a great day. No emojis."
            )

        model = GEMINI_MODEL or _DEFAULT_GEMINI_MODEL
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError(
                "Gemini returned no text (empty response, safety block, or API issue). "
                "Check GEMINI_API_KEY and model availability."
            )
        return text

    def create_tts_audio(self, text, output_filename):
        url = _ELEVEN_TTS_URL.format(voice_id=ELEVENLABS_VOICE_ID)
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=_ELEVEN_TIMEOUT_SEC,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs API error {response.status_code}: {response.text[:500]}"
            )

        filepath = _resolve_output_path(output_filename)
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
