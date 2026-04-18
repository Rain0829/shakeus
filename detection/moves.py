import math
from config import MOVEMENT_THRESHOLD, JOINTS_NEEDED

# Named keypoints tracked for movement scoring
KEYPOINT_NAMES = {
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


def landmarks_to_positions(landmarks) -> dict:
    """Convert a 33-landmark list to {joint_name: (x, y)} for tracked joints."""
    return {name: (landmarks[idx].x, landmarks[idx].y)
            for name, idx in KEYPOINT_NAMES.items()}


def score_movement(current: dict, previous: dict) -> float:
    """
    Returns 1.0 if enough joints moved since the last frame, else 0.0.
    Caller accumulates this over time (divide by fps to get seconds of movement).
    """
    if not previous:
        return 0.0
    moved = sum(
        1 for key in current
        if key in previous
        and math.dist(current[key], previous[key]) > MOVEMENT_THRESHOLD
    )
    return 1.0 if moved >= JOINTS_NEEDED else 0.0
