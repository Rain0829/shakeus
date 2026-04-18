from config import JOINT_THRESHOLD
from collections import deque

# MediaPipe landmark indices
L_WRIST, R_WRIST       = 15, 16
L_ELBOW, R_ELBOW       = 13, 14
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP           = 23, 24
L_KNEE, R_KNEE         = 25, 26

# For moves that need history (e.g. hip shake)
_hip_history = deque(maxlen=20)  # last 20 frames

def hands_up(lm) -> bool:
    """Both wrists clearly above their shoulder"""
    l = lm[L_WRIST].y < lm[L_SHOULDER].y - JOINT_THRESHOLD
    r = lm[R_WRIST].y < lm[R_SHOULDER].y - JOINT_THRESHOLD
    return l and r

def arms_out(lm) -> bool:
    """Both elbows wider than shoulders — T-pose style"""
    l = lm[L_ELBOW].x < lm[L_SHOULDER].x - JOINT_THRESHOLD
    r = lm[R_ELBOW].x > lm[R_SHOULDER].x + JOINT_THRESHOLD
    return l and r

def hip_shake(lm) -> bool:
    """Hips have been moving side-to-side recently"""
    mid_hip = (lm[L_HIP].x + lm[R_HIP].x) / 2
    _hip_history.append(mid_hip)
    if len(_hip_history) < 10:
        return False
    variance = max(_hip_history) - min(_hip_history)
    return variance > JOINT_THRESHOLD * 2

def jump(lm) -> bool:
    """Both feet (ankles) above knee height — optional 4th move"""
    l = lm[27].y < lm[L_KNEE].y  # left ankle above left knee
    r = lm[28].y < lm[R_KNEE].y
    return l and r

# ── Clean public interface — only this function is called externally ──
MOVE_REGISTRY = {
    "hands_up":  hands_up,
    "arms_out":  arms_out,
    "hip_shake": hip_shake,
    "jump":      jump,
}

def check_move(lm, move_name: str) -> bool:
    fn = MOVE_REGISTRY.get(move_name)
    if fn is None:
        raise ValueError(f"Unknown move: '{move_name}'. "
                         f"Valid: {list(MOVE_REGISTRY)}")
    return fn(lm)
