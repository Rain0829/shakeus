import threading
import random
import time
from flask import Flask
from config. import GOOGLE_HOME_IP, SONGS_BASE_URL, SONGS
from audio.speaker import Speaker
from audio.song_server import register_routes
from alarm.alarm_loop import run_alarm
from audio.voice_gen import VoiceGenerator

app = Flask(__name__)
register_routes(app)   # mounts /songs/* routes

def run_full_sequence(speaker, song_url, pose_label, song_name):
    """Wraps the AI Intro, the main alarm, and the AI Outro together."""
    print("Generating AI DJ Intro and Outro...")
    voice_gen = VoiceGenerator()

    # 1. Generate the audio files
    intro_text = voice_gen.generate_phrase("intro")
    voice_gen.create_tts_audio(intro_text, "current_intro.mp3")

    outro_text = voice_gen.generate_phrase("outro")
    voice_gen.create_tts_audio(outro_text, "current_outro.mp3")

    # 2. Play Intro
    # We use your existing SONGS_BASE_URL assuming the audio folder is served there
    intro_url = f"{SONGS_BASE_URL}/current_intro.mp3" 
    speaker.play(intro_url, volume=0.6)
    time.sleep(6) # Give the AI voice time to speak before dropping the beat

    # 3. Run your existing dance/alarm loop
    # This will play the song and presumably wait until the user does the correct pose
    print(f"Dropping the beat for: {song_name}...")
    run_alarm(speaker, song_url, pose_label, song_name)

    # 4. Play Outro (This runs immediately after run_alarm finishes)
    print("Dance complete! Playing outro...")
    outro_url = f"{SONGS_BASE_URL}/current_outro.mp3"
    speaker.play(outro_url, volume=0.6)
    time.sleep(6)

@app.route("/alarm/trigger", methods=["GET", "POST"])
def trigger():
    song    = random.choice(SONGS)
    url     = f"{SONGS_BASE_URL}/{song['file']}"
    speaker = Speaker(GOOGLE_HOME_IP)
    threading.Thread(
        target=run_full_sequence,
        args=(speaker, url, song["pose_label"], song["name"]),
        daemon=True,
    ).start()
    return f"ok — playing {song['name']}", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)


