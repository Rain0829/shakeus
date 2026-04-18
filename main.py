import threading
import random
from flask import Flask
from config import GOOGLE_HOME_IP, SONGS_BASE_URL, SONGS
from audio.speaker import Speaker
from audio.song_server import register_routes
from alarm.alarm_loop import run_alarm

app = Flask(__name__)
register_routes(app)   # mounts /songs/* routes


@app.route("/alarm/trigger", methods=["GET", "POST"])
def trigger():
    song    = random.choice(SONGS)
    url     = f"{SONGS_BASE_URL}/{song['file']}"
    speaker = Speaker(GOOGLE_HOME_IP)
    threading.Thread(
        target=run_alarm,
        args=(speaker, url, song["pose_label"], song["name"]),
        daemon=True,
    ).start()
    return f"ok — playing {song['name']}", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
