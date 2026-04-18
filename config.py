# ── Network ──────────────────────────────────────
MAC_IP       = "192.168.1.x"   # run: ipconfig getifaddr en0
MAC_PORT     = 5050
GOOGLE_HOME_IP = "192.168.1.y" # Google Home app > device > settings

# ── Alarm ────────────────────────────────────────
ALARM_TIME   = "07:30"         # 24hr format HH:MM

# ── Camera ───────────────────────────────────────
WEBCAM_INDEX = 0               # 0 = built-in Mac, 1 = Logitech after H5

# ── Audio ────────────────────────────────────────
SONGS_DIR      = "assets/songs"
SONGS_BASE_URL = f"http://{MAC_IP}:{MAC_PORT}/songs"

# ── Dance detection (Person B tunes these) ───────
MOVE_SEQUENCE      = ["hands_up", "hip_shake", "arms_out"]
MOVE_HOLD_SECONDS  = 1.0    # seconds to hold each pose
JOINT_THRESHOLD    = 0.04   # mediapipe coords are 0.0–1.0, not pixels
