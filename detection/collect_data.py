"""
collect_data.py
───────────────
Records labeled pose samples to a CSV for training.

Controls:
  Press the NUMBER KEY matching the label shown on screen to save that frame.
  Press N to cycle to the next label.
  Press Q to quit and save.

Usage:
  python3 collect_data.py

Output:
  pose_data.csv  (appended each run, safe to run multiple sessions)
"""

import cv2
import mediapipe as mp
import csv
import os
import time

# ─────────────────────────────────────────
#  LABELS — add as many poses as you want
#  Key 1 = first label, key 2 = second, etc.
# ─────────────────────────────────────────
LABELS = [
    "other", #1
    "gangnam_style",  #2
]

OUTPUT_CSV  = "detection/pose_data.csv"
MODEL_PATH  = "pose_landmarker_full.task"
SAMPLES_PER_LABEL_GOAL = 200   # suggested minimum per label

# ─────────────────────────────────────────
#  MEDIAPIPE INIT
# ─────────────────────────────────────────
import urllib.request
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────
#  CSV SETUP
# ─────────────────────────────────────────
# 33 landmarks × 3 values (x, y, z) = 99 columns + 1 label
HEADER = [f"{axis}{i}" for i in range(33) for axis in ("x", "y", "z")] + ["label"]

file_exists = os.path.exists(OUTPUT_CSV)
csv_file    = open(OUTPUT_CSV, "a", newline="")
writer      = csv.writer(csv_file)
if not file_exists:
    writer.writerow(HEADER)

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def draw_text_shadow(img, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def landmarks_to_row(landmarks):
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row

# ─────────────────────────────────────────
#  COUNT EXISTING SAMPLES
# ─────────────────────────────────────────
def count_existing():
    counts = {label: 0 for label in LABELS}
    if not file_exists:
        return counts
    with open(OUTPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label")
            if lbl in counts:
                counts[lbl] += 1
    return counts

counts       = count_existing()
current_idx  = 0
cap          = cv2.VideoCapture(0)
flash_until  = 0   # timestamp to show green flash

print(f"\nExisting samples: {counts}")
print(f"Labels: {list(enumerate(LABELS, 1))}")
print("Press number key to save that label. N = next label. Q = quit.\n")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w, _   = frame.shape
        now_ms    = int(time.time() * 1000)

        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection  = landmarker.detect_for_video(mp_image, now_ms)
        pose_found = len(detection.pose_landmarks) > 0

        # ── Draw skeleton dots ────────────────────────────────────────────
        if pose_found:
            for lm in detection.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (80, 200, 255), -1, cv2.LINE_AA)

        # ── Green flash when sample saved ─────────────────────────────────
        if time.time() < flash_until:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 200, 0), 8)

        # ── UI ────────────────────────────────────────────────────────────
        current_label = LABELS[current_idx]
        color = (0, 220, 80) if pose_found else (0, 60, 255)

        draw_text_shadow(frame, f"RECORDING: {current_label.upper()}",
                         (20, 45), 1.0, color, 2)
        draw_text_shadow(frame, f"Pose {'DETECTED' if pose_found else 'NOT FOUND'}",
                         (20, 82), 0.7, color, 1)

        # Sample counts per label
        y_off = 120
        for i, lbl in enumerate(LABELS):
            bar_color = (0, 220, 80) if counts[lbl] >= SAMPLES_PER_LABEL_GOAL else (0, 165, 255)
            draw_text_shadow(frame, f"[{i+1}] {lbl}: {counts[lbl]}/{SAMPLES_PER_LABEL_GOAL}",
                             (20, y_off), 0.65, bar_color, 1)
            y_off += 28

        draw_text_shadow(frame, "N = next label   Q = quit & save",
                         (20, h - 20), 0.55, (180, 180, 180), 1)

        cv2.imshow("Pose Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        # Number keys 1–9 save the corresponding label
        for i, label in enumerate(LABELS):
            if key == ord(str(i + 1)):
                if pose_found:
                    row = landmarks_to_row(detection.pose_landmarks[0]) + [label]
                    writer.writerow(row)
                    counts[label] += 1
                    flash_until = time.time() + 0.1
                    print(f"  Saved [{label}] — total: {counts[label]}")
                else:
                    print("  No pose detected, sample skipped.")

        if key == ord("n"):
            current_idx = (current_idx + 1) % len(LABELS)
            print(f"Switched to label: {LABELS[current_idx]}")

        if key == ord("q"):
            break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Final counts: {counts}")
print(f"Saved to: {OUTPUT_CSV}")