import os, random
from flask import send_from_directory
from config import SONGS_DIR

def register_routes(app):

    @app.route("/songs/<filename>")
    def serve_song(filename):
        return send_from_directory(
            os.path.abspath(SONGS_DIR), filename)

    @app.route("/songs/random")
    def random_song():
        songs = [f for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")]
        if not songs:
            return "No songs found", 404
        return serve_song(random.choice(songs))
