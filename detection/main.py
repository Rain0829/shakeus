"""
detection/main.py
─────────────────
Person B's standalone test loop. Run from the project root:

    python detection/main.py

Uses pygame for audio (no Chromecast/Flask needed).
Songs are read from assets/songs/. Model and classifier
paths come from config.py so they stay in sync with the team.
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
    REQUIRED_TIME, POSE_CONFIDENCE, POSE_HOLD_NEEDED,
    COUNTDOWN_SECS, SONGS, SONGS_DIR,
)

# ─────────────────────────────────────────
#  DISPLAY CONSTANTS
# ─────────────────────────────────────────
FAIL_DISPLAY_SECS = 2
PASS_DISPLAY_SECS = 3

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
#  AUDIO INIT (pygame — no Chromecast needed)
# ─────────────────────────────────────────
pygame.mixer.init()

def play_song(song):
    path = os.path.join(SONGS_DIR, song["file"])
    if os.path.exists(path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1)
    else:
        print(f"WARNING: audio file not found: {path}")

def stop_song():
    pygame.mixer.music.stop()

# ─────────────────────────────────────────
#  PICK RANDOM SONG
# ─────────────────────────────────────────
current_song = random.choice(SONGS)
print(f"\nSelected song: {current_song['name']}")
print(f"Mode: {'POSE — ' + current_song['pose_label'] if current_song['pose_label'] else 'ANY MOVEMENT'}\n")

# ─────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────
cap     = cv2.VideoCapture(WEBCAM_INDEX)
cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────
state = {
    "phase":           "waiting",
    "phase_start":     None,
    "score":           0.0,
    "prev_positions":  {},
    "result":          None,
    "attempt":         0,
    "alarm_cleared":   False,
    "pose_hold_start": None,
    "pose_hold_secs":  0.0,
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

        s         = state
        song_mode = current_song["pose_label"]

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

            if song_mode is not None:
                correct = (
                    pose_detected
                    and predicted_label == song_mode
                    and pred_confidence >= POSE_CONFIDENCE
                )
                if correct:
                    if s["pose_hold_start"] is None:
                        s["pose_hold_start"] = now
                    s["pose_hold_secs"] = now - s["pose_hold_start"]
                else:
                    s["pose_hold_start"] = None
                    s["pose_hold_secs"]  = 0.0

                s["score"] = s["pose_hold_secs"]

                if s["score"] >= POSE_HOLD_NEEDED:
                    s["result"]      = "pass"
                    s["phase"]       = "result"
                    s["phase_start"] = now
                elif elapsed >= REQUIRED_TIME:
                    s["attempt"] += 1
                    s["result"]   = "fail"
                    s["phase"]    = "result"
                    s["phase_start"] = now
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
                        s["score"] = max(0.0, s["score"] - 0.5 / cap_fps)

                s["prev_positions"] = current_positions.copy()

                if elapsed >= REQUIRED_TIME:
                    if s["score"] >= SCORE_THRESHOLD:
                        s["result"]      = "pass"
                        s["phase"]       = "result"
                        s["phase_start"] = now
                    else:
                        s["attempt"]    += 1
                        s["result"]      = "fail"
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
        mode_tag = (
            f"DO THE {current_song['pose_label'].upper().replace('_', ' ')} MOVE"
            if song_mode else "DANCE TO CLEAR"
        )
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
            elapsed   = now - s["phase_start"]
            time_left = max(0.0, REQUIRED_TIME - elapsed)

            shadow(frame, "ALARM ON", (20, 38), 0.8, (0,60,255), 2)
            if s["attempt"] > 0:
                shadow(frame, f"attempt {s['attempt']+1}",
                       (180, 38), 0.65, (200,150,0), 1)
            shadow(frame, f"Time left: {time_left:.1f}s", (20, 68), 0.9, (0,220,255))

            if song_mode:
                shadow(frame, f"Hold the {song_mode.replace('_',' ').upper()} pose!",
                       (20, 100), 0.75, (255, 220, 0), 2)
                score_bar(frame, s["pose_hold_secs"], POSE_HOLD_NEEDED, 20, 112)
                if pose_detected:
                    color = (0, 220, 80) if predicted_label == song_mode else (0, 60, 255)
                    shadow(frame, f"Detected: {predicted_label} ({pred_confidence:.0%})",
                           (20, 152), 0.65, color, 1)
            else:
                shadow(frame, "Score", (20, 100), 0.65, (200,200,200), 1)
                score_bar(frame, s["score"], SCORE_THRESHOLD, 20, 108)
                if s["score"] < SCORE_THRESHOLD * 0.4:
                    shadow(frame, "DANCE HARDER!",
                           (w//2 - 155, h - 70), 1.3, (0,60,255), 3)

        elif s["phase"] == "result":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h//2-90), (w, h//2+90), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            if s["result"] == "pass":
                shadow(frame, "ALARM CLEARED!",
                       (w//2-230, h//2+15), 2.4, (0,230,80), 5)
                shadow(frame, f"Attempts: {s['attempt']+1}  |  Song: {current_song['name']}",
                       (w//2-220, h//2+65), 0.85, (200,200,200), 2)
            else:
                shadow(frame, "NOT ENOUGH — TRY AGAIN!",
                       (w//2-295, h//2+15), 1.6, (0,60,255), 4)
                if song_mode:
                    shadow(frame, f"Hold the {song_mode.replace('_',' ')} pose for {POSE_HOLD_NEEDED:.0f}s",
                           (w//2-230, h//2+65), 0.75, (200,200,200), 2)
                else:
                    shadow(frame, f"Score: {s['score']:.2f} / {SCORE_THRESHOLD:.1f} needed",
                           (w//2-200, h//2+65), 0.85, (200,200,200), 2)

        shadow(frame, "ESC to force quit", (w-200, h-15), 0.5, (120,120,120), 1)

        cv2.imshow("Dance Alarm — Test Mode", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            stop_song()
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
