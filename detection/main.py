"""
detection/main.py
─────────────────
Dance alarm — runs from the project root:

    python detection/main.py

Modes are inferred from config.py SONGS list:
  pose_label = None            → general movement (any dancing clears alarm)
  pose_label = "low_cortisol" → dynamic mode (wrist burst detection)
  pose_label = anything else  → static mode (hold that ML pose for N seconds)

In ALL modes, the user must dance for the full REQUIRED_TIME before pass/fail.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import pickle
import random
import urllib.request
import pygame

import sys
sys.path.insert(0, ".")
from config import (
    MODEL_PATH, CLASSIFIER_PATH, WEBCAM_INDEX,
    MOVEMENT_THRESHOLD, JOINTS_NEEDED, SCORE_THRESHOLD,
    REQUIRED_TIME, POSE_CONFIDENCE, POSE_HOLD_NEEDED, POSE_GRACE_SECS,
    COUNTDOWN_SECS, SONGS, SONGS_DIR,
    DYN_WINDOW, DYN_BURSTS_NEEDED, DYN_BURST_THRESHOLD,
)

# ─────────────────────────────────────────
#  DISPLAY CONSTANTS
# ─────────────────────────────────────────
FAIL_DISPLAY_SECS = 2
PASS_DISPLAY_SECS = 3

# ─────────────────────────────────────────
#  MODE INFERENCE
# ─────────────────────────────────────────
DYNAMIC_LABELS = {"low_cortisol"}

def get_mode(song):
    lbl = song.get("pose_label")
    if lbl is None:
        return "movement"
    if lbl in DYNAMIC_LABELS:
        return "dynamic"
    return "static"

# ─────────────────────────────────────────
#  LANDMARK CONFIG
# ─────────────────────────────────────────
KEYPOINTS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_ankle":     27,
    "right_ankle":    28,
}

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
]

# ─────────────────────────────────────────
#  MEDIAPIPE INIT
# ─────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ─────────────────────────────────────────
#  LOAD CLASSIFIER
# ─────────────────────────────────────────
print("Loading classifier...")
with open(CLASSIFIER_PATH, "rb") as f:
    saved     = pickle.load(f)
    clf       = saved["classifier"]
    label_enc = saved["label_encoder"]
print(f"Labels: {list(label_enc.classes_)}")

def predict_pose(landmarks):
    row   = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]
    probs = clf.predict_proba([row])[0]
    idx   = int(np.argmax(probs))
    return label_enc.classes_[idx], probs[idx]

# ─────────────────────────────────────────
#  AUDIO
# ─────────────────────────────────────────
pygame.mixer.init()

def play_song(song):
    path = os.path.join(SONGS_DIR, song["file"])
    if not os.path.exists(path):
        available = [f for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")]
        if not available:
            print(f"WARNING: no MP3 files found in {SONGS_DIR}/ — music skipped")
            return
        fallback = random.choice(available)
        path = os.path.join(SONGS_DIR, fallback)
        print(f"Song '{song['file']}' not found, playing '{fallback}' instead")
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(-1)

def stop_song():
    pygame.mixer.music.stop()

# ─────────────────────────────────────────
#  PICK RANDOM SONG
# ─────────────────────────────────────────
current_song = random.choice(SONGS)
current_mode = get_mode(current_song)
print(f"\nSelected song : {current_song['name']}")
print(f"Mode          : {current_mode}\n")

# ─────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────
cap     = cv2.VideoCapture(WEBCAM_INDEX)
cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
state = {
    "phase":            "waiting",
    "phase_start":      None,
    "score":            0.0,
    "prev_positions":   {},
    "result":           None,
    "attempt":          0,
    "alarm_cleared":    False,
    # static mode
    "pose_hold_start":  None,
    "pose_hold_secs":   0.0,
    "pose_best_hold":   0.0,   # best hold achieved this attempt
    "grace_start":      None,  # when a grace period started
    "pose_complete":    False, # True once hold bar has been filled
    # dynamic mode
    "burst_times":      [],
    "prev_wrists":      None,
    "dyn_complete":     False, # True once burst target hit
}

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def shadow(img, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def score_bar(img, score, threshold, x, y, bar_w=300, bar_h=22):
    fill  = int(bar_w * min(score / threshold, 1.0))
    color = (0, 220, 80) if score >= threshold else (0, 165, 255)
    cv2.rectangle(img, (x, y), (x+bar_w, y+bar_h), (50,50,50), -1)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+bar_h), color, -1)
    cv2.rectangle(img, (x, y), (x+bar_w, y+bar_h), (180,180,180), 1)
    shadow(img, f"{int(100*score/threshold)}%",
           (x+bar_w+8, y+bar_h-4), 0.55, (255,255,255), 1)

def time_bar(img, elapsed, required, x, y, bar_w=300, bar_h=22):
    """Shows overall time progress — always counts up to REQUIRED_TIME."""
    fill  = int(bar_w * min(elapsed / required, 1.0))
    color = (0, 200, 255)
    cv2.rectangle(img, (x, y), (x+bar_w, y+bar_h), (50,50,50), -1)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+bar_h), color, -1)
    cv2.rectangle(img, (x, y), (x+bar_w, y+bar_h), (180,180,180), 1)
    time_left = max(0.0, required - elapsed)
    shadow(img, f"{time_left:.1f}s", (x+bar_w+8, y+bar_h-4), 0.55, (255,255,255), 1)

def draw_landmarks(img, landmarks, h, w):
    pts = {}
    for name, idx in KEYPOINTS.items():
        lm = landmarks[idx]
        pts[idx] = (int(lm.x * w), int(lm.y * h))
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (255,255,255), 2, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(img, pt, 5, (80,200,255), -1, cv2.LINE_AA)

def start_countdown(s, now):
    s["phase"]           = "countdown"
    s["phase_start"]     = now
    s["score"]           = 0.0
    s["prev_positions"]  = {}
    s["result"]          = None
    s["pose_hold_start"] = None
    s["pose_hold_secs"]  = 0.0
    s["pose_best_hold"]  = 0.0
    s["grace_start"]     = None
    s["pose_complete"]   = False
    s["burst_times"]     = []
    s["prev_wrists"]     = None
    s["dyn_complete"]    = False

def update_dynamic(s, landmarks, now):
    """Track wrist movement bursts over a rolling time window."""
    lw = landmarks[15]
    rw = landmarks[16]
    wrists = ((lw.x, lw.y), (rw.x, rw.y))
    if s["prev_wrists"] is not None:
        moved = (
            dist(wrists[0], s["prev_wrists"][0]) > DYN_BURST_THRESHOLD or
            dist(wrists[1], s["prev_wrists"][1]) > DYN_BURST_THRESHOLD
        )
        if moved:
            s["burst_times"].append(now)
    s["prev_wrists"] = wrists
    s["burst_times"] = [t for t in s["burst_times"] if now - t <= DYN_WINDOW]
    return len(s["burst_times"])

# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
last_ts = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now     = time.time()

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = max(int(now * 1000), last_ts + 1)
        last_ts  = ts_ms
        detection = landmarker.detect_for_video(mp_image, ts_ms)

        pose_detected     = len(detection.pose_landmarks) > 0
        current_positions = {}
        predicted_label   = None
        pred_confidence   = 0.0

        if pose_detected:
            landmarks = detection.pose_landmarks[0]
            for name, idx in KEYPOINTS.items():
                lm = landmarks[idx]
                current_positions[name] = (lm.x, lm.y)
            predicted_label, pred_confidence = predict_pose(landmarks)

        s = state

        # ── Phase machine ──────────────────────────────────────────────────
        if s["phase"] == "waiting":
            if pose_detected:
                s["attempt"] = 0
                play_song(current_song)
                start_countdown(s, now)

        elif s["phase"] == "countdown":
            if not pose_detected:
                s["phase"] = "waiting"
                stop_song()
            elif now - s["phase_start"] >= COUNTDOWN_SECS:
                s["phase"]          = "dancing"
                s["phase_start"]    = now
                s["prev_positions"] = current_positions.copy()

        elif s["phase"] == "dancing":
            elapsed = now - s["phase_start"]

            # ── STATIC: hold a specific pose ──────────────────────────────
            if current_mode == "static":
                correct = (
                    pose_detected
                    and predicted_label == current_song["pose_label"]
                    and pred_confidence >= POSE_CONFIDENCE
                )

                if correct:
                    # Good frame — clear any grace period
                    s["grace_start"] = None
                    if s["pose_hold_start"] is None:
                        s["pose_hold_start"] = now
                    s["pose_hold_secs"] = now - s["pose_hold_start"]
                    s["pose_best_hold"] = max(s["pose_best_hold"], s["pose_hold_secs"])
                    if s["pose_hold_secs"] >= POSE_HOLD_NEEDED:
                        s["pose_complete"] = True
                else:
                    # Bad frame — start or continue grace period
                    if s["pose_hold_start"] is not None:
                        if s["grace_start"] is None:
                            s["grace_start"] = now
                        elif now - s["grace_start"] > POSE_GRACE_SECS:
                            # Grace expired — reset hold
                            s["pose_hold_start"] = None
                            s["pose_hold_secs"]  = 0.0
                            s["grace_start"]     = None
                    else:
                        s["pose_hold_secs"] = 0.0

                s["score"] = s["pose_hold_secs"]

                # Only pass/fail once REQUIRED_TIME is up
                if elapsed >= REQUIRED_TIME:
                    if s["pose_complete"]:
                        s["result"] = "pass"
                    else:
                        s["attempt"] += 1
                        s["result"] = "fail"
                    s["phase"]       = "result"
                    s["phase_start"] = now

            # ── DYNAMIC: repeated arm/wrist movement ──────────────────────
            elif current_mode == "dynamic":
                burst_count = update_dynamic(s, landmarks, now) if pose_detected else len(s["burst_times"])
                s["score"] = burst_count
                if burst_count >= DYN_BURSTS_NEEDED:
                    s["dyn_complete"] = True

                # Only pass/fail once REQUIRED_TIME is up
                if elapsed >= REQUIRED_TIME:
                    if s["dyn_complete"]:
                        s["result"] = "pass"
                    else:
                        s["attempt"] += 1
                        s["result"] = "fail"
                    s["phase"]       = "result"
                    s["phase_start"] = now

            # ── MOVEMENT: any general dancing ─────────────────────────────
            else:
                if pose_detected and s["prev_positions"]:
                    moving = sum(
                        1 for key in current_positions
                        if key in s["prev_positions"]
                        and dist(current_positions[key], s["prev_positions"][key]) > MOVEMENT_THRESHOLD
                    )
                    if moving >= JOINTS_NEEDED:
                        s["score"] += 1.0 / cap_fps
                    else:
                        s["score"] = max(0.0, s["score"] - 0.3 / cap_fps)
                s["prev_positions"] = current_positions.copy()

                if elapsed >= REQUIRED_TIME:
                    if s["score"] >= SCORE_THRESHOLD:
                        s["result"] = "pass"
                    else:
                        s["attempt"] += 1
                        s["result"] = "fail"
                    s["phase"]       = "result"
                    s["phase_start"] = now

        elif s["phase"] == "result":
            if s["result"] == "pass":
                if now - s["phase_start"] >= PASS_DISPLAY_SECS:
                    s["alarm_cleared"] = True
            else:
                if now - s["phase_start"] >= FAIL_DISPLAY_SECS:
                    start_countdown(s, now)

        if s["alarm_cleared"]:
            stop_song()
            break

        # ── Draw skeleton ──────────────────────────────────────────────────
        if pose_detected:
            draw_landmarks(frame, detection.pose_landmarks[0], h, w)

        # ── Song + mode banner ─────────────────────────────────────────────
        if current_mode == "static":
            mode_tag = f"HOLD THE {current_song['pose_label'].upper().replace('_', ' ')} POSE"
        elif current_mode == "dynamic":
            mode_tag = f"DO THE {current_song['name'].upper()} MOVE"
        else:
            mode_tag = "DANCE TO CLEAR"
        shadow(frame, f"{current_song['name'].upper()}  |  {mode_tag}",
               (20, h - 20), 0.55, (255, 220, 0), 1)

        # ── Phase UI ───────────────────────────────────────────────────────
        if s["phase"] == "waiting":
            shadow(frame, "Step into frame to start",
                   (w//2 - 220, h//2), 1.0, (255,255,255))

        elif s["phase"] == "countdown":
            remaining = COUNTDOWN_SECS - int(now - s["phase_start"])
            shadow(frame, "ALARM IS ON — GET READY!",
                   (w//2 - 255, h//2 - 70), 1.0, (0,80,255), 2)
            if s["attempt"] > 0:
                shadow(frame, f"Attempt {s['attempt']+1}",
                       (w//2 - 80, h//2 - 20), 0.9, (200,200,0), 2)
            shadow(frame, str(max(remaining, 1)),
                   (w//2 - 40, h//2 + 70), 3.5, (0,220,255), 4)

        elif s["phase"] == "dancing":
            elapsed = now - s["phase_start"]

            shadow(frame, "ALARM ON", (20, 38), 0.8, (0,60,255), 2)
            if s["attempt"] > 0:
                shadow(frame, f"attempt {s['attempt']+1}", (180, 38), 0.65, (200,150,0), 1)

            # Time bar always visible across all modes
            shadow(frame, "Time", (20, 68), 0.65, (200,200,200), 1)
            time_bar(frame, elapsed, REQUIRED_TIME, 20, 76)

            if current_mode == "static":
                # Show pose hold bar
                pose_color = (0, 220, 80) if s["pose_complete"] else (255, 220, 0)
                label_text = "POSE DONE! Keep dancing!" if s["pose_complete"] else \
                             f"Hold the {current_song['pose_label'].replace('_',' ').upper()} pose!"
                shadow(frame, label_text, (20, 115), 0.75, pose_color, 2)
                score_bar(frame, s["pose_hold_secs"] if not s["pose_complete"] else POSE_HOLD_NEEDED,
                          POSE_HOLD_NEEDED, 20, 128)
                if pose_detected:
                    col = (0,220,80) if predicted_label == current_song["pose_label"] else (0,60,255)
                    shadow(frame, f"Detected: {predicted_label} ({pred_confidence:.0%})",
                           (20, 168), 0.65, col, 1)

            elif current_mode == "dynamic":
                burst_count = len(s["burst_times"])
                dyn_color = (0, 220, 80) if s["dyn_complete"] else (255, 220, 0)
                label_text = "MOVE DONE! Keep dancing!" if s["dyn_complete"] else \
                             "Move your arms to the beat!"
                shadow(frame, label_text, (20, 115), 0.75, dyn_color, 2)
                score_bar(frame, min(burst_count, DYN_BURSTS_NEEDED),
                          DYN_BURSTS_NEEDED, 20, 128)
                shadow(frame, f"Arm bursts: {burst_count}/{DYN_BURSTS_NEEDED}",
                       (20, 168), 0.65, (0,220,255), 1)
                if burst_count < 1:
                    shadow(frame, "MOVE YOUR ARMS!", (w//2 - 155, h - 70), 1.2, (0,60,255), 3)

            else:
                shadow(frame, "Score", (20, 115), 0.65, (200,200,200), 1)
                score_bar(frame, s["score"], SCORE_THRESHOLD, 20, 123)
                if s["score"] < SCORE_THRESHOLD * 0.4:
                    shadow(frame, "DANCE HARDER!", (w//2 - 155, h - 70), 1.3, (0,60,255), 3)

        elif s["phase"] == "result":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h//2-90), (w, h//2+90), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            if s["result"] == "pass":
                shadow(frame, "ALARM CLEARED!", (w//2-230, h//2+15), 2.4, (0,230,80), 5)
                shadow(frame, f"Attempts: {s['attempt']+1}  |  Song: {current_song['name']}",
                       (w//2-220, h//2+65), 0.85, (200,200,200), 2)
            else:
                shadow(frame, "NOT ENOUGH — TRY AGAIN!",
                       (w//2-295, h//2+15), 1.6, (0,60,255), 4)
                if current_mode == "static":
                    shadow(frame, "Hold the pose long enough within the 10s!",
                           (w//2-240, h//2+65), 0.75, (200,200,200), 2)
                elif current_mode == "dynamic":
                    shadow(frame, f"Need {DYN_BURSTS_NEEDED} arm bursts within the 10s!",
                           (w//2-230, h//2+65), 0.75, (200,200,200), 2)
                else:
                    shadow(frame, f"Score: {s['score']:.2f} / {SCORE_THRESHOLD:.1f} needed",
                           (w//2-200, h//2+65), 0.85, (200,200,200), 2)

        shadow(frame, "ESC to force quit", (w-200, h-15), 0.5, (120,120,120), 1)

        cv2.imshow("Dance Alarm", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_song()
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()