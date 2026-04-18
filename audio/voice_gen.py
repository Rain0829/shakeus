# audio/voice_generator.py
import google.generativeai as genai
import requests
import os
import sys

# Append parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

class VoiceGenerator:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        # Using 1.5-flash as it is exceptionally fast for simple text tasks
        self.model = genai.GenerativeModel('gemini-1.5-flash') 

    def generate_phrase(self, phase="intro"):
        """Uses Gemini to generate a random hype phrase."""
        if phase == "intro":
            prompt = "Write a short, punchy 1-sentence intro to wake someone up and hype them up for a morning dance routine. Sound like an energetic DJ. No emojis."
        else:
            prompt = "Write a short, punchy 1-sentence congratulation for finishing a morning dance break. Tell them to have a great day. No emojis."
        
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def create_tts_audio(self, text, output_filename):
        """Sends the text to ElevenLabs and saves the MP3."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Save it in the audio directory so song_server.py can host it
            filepath = os.path.join(os.path.dirname(__file__), output_filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        else:
            raise Exception(f"ElevenLabs API Error: {response.text}")