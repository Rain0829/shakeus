# Run: python -m pytest tests/test_moves.py -v

import sys
sys.path.insert(0, ".")
from detection.moves import landmarks_to_positions, score_movement


class FakeLandmark:
    def __init__(self, x, y, z=0):
        self.x, self.y, self.z = x, y, z


def make_lm(**overrides):
    """Build a 33-landmark list, override specific indices with (x, y)."""
    lm = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    for idx, (x, y) in overrides.items():
        lm[idx] = FakeLandmark(x, y)
    return lm


# ── landmarks_to_positions ────────────────────────────────────────────────────

def test_landmarks_to_positions_returns_named_joints():
    lm  = make_lm()
    lm[15] = FakeLandmark(0.3, 0.2)
    lm[16] = FakeLandmark(0.7, 0.2)
    pos = landmarks_to_positions(lm)
    assert "left_wrist"  in pos
    assert "right_wrist" in pos
    assert pos["left_wrist"]  == (0.3, 0.2)
    assert pos["right_wrist"] == (0.7, 0.2)


def test_landmarks_to_positions_has_all_tracked_joints():
    pos = landmarks_to_positions(make_lm())
    expected = {
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "left_hip",      "right_hip",
        "left_ankle",    "right_ankle",
    }
    assert expected == set(pos.keys())


# ── score_movement ────────────────────────────────────────────────────────────

def test_score_movement_returns_one_when_enough_joints_moved():
    # Move 5 joints well past MOVEMENT_THRESHOLD (0.02)
    prev = {k: (0.0, 0.0) for k in [
        "left_wrist", "right_wrist", "left_elbow",
        "right_elbow", "left_shoulder",
    ]}
    curr = {k: (0.1, 0.1) for k in prev}
    assert score_movement(curr, prev) == 1.0


def test_score_movement_returns_zero_when_still():
    prev = {"left_wrist": (0.5, 0.5), "right_wrist": (0.5, 0.5)}
    curr = {"left_wrist": (0.5, 0.5), "right_wrist": (0.5, 0.5)}
    assert score_movement(curr, prev) == 0.0


def test_score_movement_returns_zero_with_empty_prev():
    curr = {"left_wrist": (0.3, 0.3)}
    assert score_movement(curr, {}) == 0.0


def test_score_movement_below_threshold_not_counted():
    # Movement of 0.005 < MOVEMENT_THRESHOLD (0.02), only 1 joint — should be 0
    prev = {"left_wrist": (0.5, 0.5)}
    curr = {"left_wrist": (0.505, 0.5)}
    assert score_movement(curr, prev) == 0.0
