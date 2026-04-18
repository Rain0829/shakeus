# ── Network ──────────────────────────────────────
MAC_IP         = "192.168.1.x"   # run: ipconfig getifaddr en0
MAC_PORT       = 5050
GOOGLE_HOME_IP = "192.168.1.y"   # Google Home app > device > settings

# ── Alarm ────────────────────────────────────────
ALARM_TIME = "07:30"             # 24hr format HH:MM

# ── Camera ───────────────────────────────────────
WEBCAM_INDEX = 0                 # 0 = built-in Mac, 1 = USB webcam

# ── Audio ────────────────────────────────────────
SONGS_DIR      = "assets/songs"
SONGS_BASE_URL = f"http://{MAC_IP}:{MAC_PORT}/songs"

# Songs: pose_label=None → any dancing clears alarm
#        pose_label="x"  → must hold that ML-classified pose
SONGS = [
    {"name": "Scuba",         "file": "scuba.mp3",         "pose_label": "scuba"},
    {"name": "Low Cortisol",  "file": "low_cortisol.mp3",  "pose_label": "low_cortisol"},
    {"name": "Woah",          "file": "woah.mp3",          "pose_label": None},
    {"name": "Lush Life",     "file": "lush_life.mp3",     "pose_label": None},
    {"name": "Gangnam Style", "file": "gangnam_style.mp3", "pose_label": None},
    {"name": "Whip",          "file": "whip.mp3",          "pose_label": None}
]

# ── Detection model paths (Person B's trained files) ─────────────────────────
MODEL_PATH      = "detection/pose_landmarker_full.task"
CLASSIFIER_PATH = "detection/pose_classifier.pkl"

# ── Dance detection thresholds (Person B tunes these) ────────────────────────
MOVEMENT_THRESHOLD = 0.02   # min joint displacement per frame to count as movement
JOINTS_NEEDED      = 3      # how many joints must move per frame to score
SCORE_THRESHOLD    = 3.0    # seconds of movement needed to pass (movement mode)
REQUIRED_TIME      = 10     # seconds per attempt before pass/fail
POSE_CONFIDENCE    = 0.7    # classifier confidence needed to count a pose hit
POSE_HOLD_NEEDED   = 2.0    # seconds of correct pose needed to dismiss (pose mode)
COUNTDOWN_SECS     = 3      # countdown before dancing phase begins


# EARVIN's GEMINI API and ELEVEN LABS API
GEMINI_API_KEY = "your_gemini_api_key_here"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
ELEVENLABS_VOICE_ID = "your_chosen_voice_id_here"
