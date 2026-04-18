import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1)
        self._drawing = mp.solutions.drawing_utils
        self._style   = mp.solutions.pose

    def get_landmarks(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if result.pose_landmarks:
            return result.pose_landmarks.landmark
        return None

    def draw_skeleton(self, frame, landmarks_obj):
        """Pass the raw result.pose_landmarks object, not .landmark list"""
        self._drawing.draw_landmarks(
            frame,
            landmarks_obj,
            self._style.POSE_CONNECTIONS)
