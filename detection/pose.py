import cv2
import mediapipe as mp
import urllib.request
import os
import time
from config import MODEL_PATH, USE_LITE_MODEL

_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
MODEL_URL = (
    _BASE + "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    if USE_LITE_MODEL else
    _BASE + "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

_KEYPOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
]


class PoseDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("Downloading pose model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Done.")

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_ts = 0

    def get_landmarks(self, frame):
        """Returns list of 33 NormalizedLandmark or None."""
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = max(int(time.time() * 1000), self._last_ts + 1)
        self._last_ts = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def draw_skeleton(self, frame, landmarks):
        """Draw skeleton overlay onto frame in-place."""
        h, w = frame.shape[:2]
        pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h))
               for i in _KEYPOINTS}
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for pt in pts.values():
            cv2.circle(frame, pt, 5, (80, 200, 255), -1, cv2.LINE_AA)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
