import os
from pathlib import Path
import requests

from config import SONGS_DIR
from env_config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
)

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
    # Class-level counters to ensure we iterate to the next sentence 
    # even when stream.py creates a new instance of this class!
    _intro_index = 0
    _outro_index = 0

    _INTROS = [
        "Wake up! Start dancing, or I will play Baby Shark on repeat.",
        "Good morning! The floor is lava, so get out of bed and show me your moves.",
        "Get up! Gravity is not an excuse. Let's get this morning party started.",
        "Morning! I know you are tired, but it is time to shake that bedhead.",
        "Rise and shine! Your bed misses you already, but it's time to sweat."
    ]

    _OUTROS = [
        "Alarm cleared! I have seen better dancing from a penguin, but you pass.",
        "Mission accomplished. You can stop flailing now. Go drink some coffee.",
        "Boom! The music is off. Now go conquer the world.",
        "You survived the morning dance battle. You are free to go.",
        "Okay, okay, I am turning off the music. Great job, now get out of here!"
    ]

    def __init__(self):
        # Removed Gemini initialization to bypass the 403 API blocked errors
        pass

    def generate_phrase(self, phase="intro"):
        """Grabs the next hardcoded sentence in the loop."""
        if phase == "intro":
            text = self._INTROS[VoiceGenerator._intro_index]
            # Increment and wrap back to 0 if we hit the end of the list
            VoiceGenerator._intro_index = (VoiceGenerator._intro_index + 1) % len(self._INTROS)
            return text
        else:
            text = self._OUTROS[VoiceGenerator._outro_index]
            # Increment and wrap back to 0 if we hit the end of the list
            VoiceGenerator._outro_index = (VoiceGenerator._outro_index + 1) % len(self._OUTROS)
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