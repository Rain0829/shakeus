# Run: python -m pytest tests/test_moves.py -v

import sys
sys.path.insert(0, ".")
from detection.moves import check_move

class FakeLandmark:
    def __init__(self, x, y, z=0):
        self.x, self.y, self.z = x, y, z

def make_lm(**positions):
    """Build a fake 33-landmark list, override specific indices"""
    lm = [FakeLandmark(0.5, 0.5) for _ in range(33)]
    for idx, (x, y) in positions.items():
        lm[idx] = FakeLandmark(x, y)
    return lm

def test_hands_up_true():
    lm = make_lm(**{
        15: (0.3, 0.2),   # L wrist high up
        16: (0.7, 0.2),   # R wrist high up
        11: (0.3, 0.5),   # L shoulder lower
        12: (0.7, 0.5),   # R shoulder lower
    })
    assert check_move(lm, "hands_up") is True

def test_hands_up_false():
    lm = make_lm(**{
        15: (0.3, 0.6),   # L wrist below shoulder
        16: (0.7, 0.6),
        11: (0.3, 0.5),
        12: (0.7, 0.5),
    })
    assert check_move(lm, "hands_up") is False

def test_arms_out():
    lm = make_lm(**{
        13: (0.1, 0.5),   # L elbow far left
        14: (0.9, 0.5),   # R elbow far right
        11: (0.35, 0.5),
        12: (0.65, 0.5),
    })
    assert check_move(lm, "arms_out") is True

def test_unknown_move_raises():
    import pytest
    lm = make_lm()
    with pytest.raises(ValueError):
        check_move(lm, "moonwalk")
