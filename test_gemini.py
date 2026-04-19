import os
from audio.voice_generator import VoiceGenerator

print("1. Initializing AI DJ...")
try:
    vg = VoiceGenerator()
    
    print("2. Generating Gemini hype text...")
    text = "Hello Rain! Have a Nice day!"
    print(f"   [Gemini says]: '{text}'")
    
    print("3. Sending to ElevenLabs for TTS...")
    output_path = vg.create_tts_audio(text, "test_voice.mp3")
    
    print(f"4. SUCCESS! Audio saved to: {output_path}")
    
except Exception as e:
    print(f"\n[ERROR] The DJ crashed: {e}")
    print("-> Double-check your .env file for API keys, or ensure elevenlabs/google-generativeai are installed!")