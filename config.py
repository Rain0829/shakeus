# Alarm / camera / chromecast tuning and song list. Secrets: see env_config.

# ── Alarm ────────────────────────────────────────────────────────────────────

ALARM_TIME = "07:30"             # 24hr format HH:MM


# ── Camera ───────────────────────────────────────────────────────────────────

WEBCAM_INDEX = 0                 # 0 = built-in Mac / first USB on Pi

# MJPEG /video to the browser: max updates per second per viewer (0 = no cap). Saves ngrok/CPU.
VIDEO_STREAM_MAX_FPS = 15


# ── Audio ────────────────────────────────────────────────────────────────────

SONGS_DIR = "assets/songs"

# Set True on Pi to cast audio to Google Home; False plays locally via pygame
USE_CHROMECAST = False


# Songs: pose_label=None → any dancing clears alarm
#        pose_label="x"  → must hold that ML-classified pose
SONGS = [
    {"name": "Scuba",         "file": "scuba.mp3",         "pose_label": "scuba"},
    {"name": "Gangnam Style", "file": "gangnam_style.mp3", "pose_label": "gangnam_style"},
    {"name": "Low Cortisol",  "file": "low_cortisol.mp3",  "pose_label": None},
    {"name": "Woah",          "file": "woah.mp3",          "pose_label": None},
    {"name": "Lush Life",     "file": "lush_life.mp3",     "pose_label": None},
    {"name": "Whip",          "file": "whip.mp3",          "pose_label": None},
]


# ── Detection model ───────────────────────────────────────────────────────────

USE_LITE_MODEL = False
MODEL_PATH = (
    "detection/pose_landmarker_lite.task" if USE_LITE_MODEL
    else "detection/pose_landmarker_full.task"
)
CLASSIFIER_PATH = "detection/pose_classifier.pkl"


# ── Dance detection thresholds ────────────────────────────────────────────────

MOVEMENT_THRESHOLD = 0.02   # min joint displacement per frame to count as movement
JOINTS_NEEDED = 3           # how many joints must move per frame to score
SCORE_THRESHOLD = 3.0       # seconds of movement needed to pass (movement mode)
REQUIRED_TIME = 10          # seconds per attempt before pass/fail
POSE_CONFIDENCE = 0.35       # classifier confidence needed to count a pose hit
POSE_HOLD_NEEDED = 3.0    # seconds of correct pose needed to dismiss (pose mode)
POSE_GRACE_SECS = 1.0     # seconds of bad frames tolerated before hold resets
COUNTDOWN_SECS = 3        # countdown before dancing phase begins


# ── Dynamic mode (wrist burst detection) ───────────────────────────────────

DYN_WINDOW = 2.0            # rolling window in seconds
DYN_BURSTS_NEEDED = 3     # wrist bursts needed inside the window
DYN_BURST_THRESHOLD = 0.04  # min wrist displacement per frame to count as a burst
