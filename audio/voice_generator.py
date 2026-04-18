import requests
import os
from google import genai
from config import GEMINI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, SONGS_DIR


class VoiceGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_phrase(self, phase="intro"):
        if phase == "intro":
            prompt = "Write a short, punchy 1-sentence intro to wake someone up and hype them up for a morning dance routine. Sound like an energetic DJ. No emojis."
        else:
            prompt = "Write a short, punchy 1-sentence congratulation for finishing a morning dance break. Tell them to have a great day. No emojis."

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()

    def create_tts_audio(self, text, output_filename):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            filepath = os.path.join(SONGS_DIR, output_filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filepath
        else:
            raise Exception(f"ElevenLabs API error {response.status_code}: {response.text}")
