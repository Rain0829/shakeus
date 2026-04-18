# ── Network ──────────────────────────────────────────────────────────────────
# On Mac:  ipconfig getifaddr en0
# On Pi:   hostname -I
MAC_IP         = "192.168.137.143"
MAC_PORT       = 5050
GOOGLE_HOME_IP = "192.168.137.143"

# ── Alarm ────────────────────────────────────────────────────────────────────
ALARM_TIME = "07:30"             # 24hr format HH:MM

# ── Camera ───────────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0                 # 0 = built-in Mac / first USB on Pi

# ── Audio ────────────────────────────────────────────────────────────────────
SONGS_DIR      = "assets/songs"
SONGS_BASE_URL = f"http://{MAC_IP}:8000/songs"  # stream.py serves songs here

# Set True on Pi to cast audio to Google Home; False plays locally via pygame
USE_CHROMECAST = False

# Songs: pose_label=None → any dancing clears alarm
#        pose_label="x"  → must hold that ML-classified pose
SONGS = [
    {"name": "Gangnam Style", "file": "gangnam_style.mp3", "pose_label": 'gangnam_style'},
#     {"name": "Scuba",         "file": "scuba.mp3",         "pose_label": None},
#     {"name": "Low Cortisol",  "file": "low_cortisol.mp3",  "pose_label": None},
#     {"name": "Woah",          "file": "woah.mp3",          "pose_label": None},
#     {"name": "Lush Life",     "file": "lush_life.mp3",     "pose_label": None},
#     {"name": "Whip",          "file": "whip.mp3",          "pose_label": None},
 ]

# ── Detection model ───────────────────────────────────────────────────────────
# Set USE_LITE_MODEL = True on Raspberry Pi for better performance
USE_LITE_MODEL  = False
MODEL_PATH      = (
    "detection/pose_landmarker_lite.task" if USE_LITE_MODEL
    else "detection/pose_landmarker_full.task"
)
CLASSIFIER_PATH = "detection/pose_classifier.pkl"

# ── Dance detection thresholds ────────────────────────────────────────────────
MOVEMENT_THRESHOLD = 0.02   # min joint displacement per frame to count as movement
JOINTS_NEEDED      = 3      # how many joints must move per frame to score
SCORE_THRESHOLD    = 3.0    # seconds of movement needed to pass (movement mode)
REQUIRED_TIME      = 10     # seconds per attempt before pass/fail
POSE_CONFIDENCE    = 0.6    # classifier confidence needed to count a pose hit
POSE_HOLD_NEEDED   = 3.0    # seconds of correct pose needed to dismiss (pose mode)
POSE_GRACE_SECS    = 1.0    # seconds of bad frames tolerated before hold resets
COUNTDOWN_SECS     = 3      # countdown before dancing phase begins

# ── Dynamic mode (wrist burst detection) ─────────────────────────────────────
DYN_WINDOW          = 3.0   # rolling window in seconds
DYN_BURSTS_NEEDED   = 5     # wrist bursts needed inside the window
DYN_BURST_THRESHOLD = 0.08  # min wrist displacement per frame to count as a burst

# ── AI Voice ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = "AIzaSyA24bgSPtFMIQOFJidgIFTtEngYAbNuSo4"
ELEVENLABS_API_KEY  = "sk_f940f6c218c43c9bba0e003c306240ea66204e921cfbb270"
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
