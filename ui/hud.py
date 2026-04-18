import cv2
import time
from config import POSE_HOLD_NEEDED, SCORE_THRESHOLD, REQUIRED_TIME, COUNTDOWN_SECS


def _shadow(img, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _score_bar(img, value, maximum, x, y, bar_w=300, bar_h=22):
    fill  = int(bar_w * min(value / maximum, 1.0))
    color = (0, 220, 80) if value >= maximum else (0, 165, 255)
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + bar_h), color, -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (180, 180, 180), 1)
    _shadow(img, f"{int(100 * value / maximum)}%",
            (x + bar_w + 8, y + bar_h - 4), 0.55, (255, 255, 255), 1)


def draw_hud(frame, state: dict, song_name: str, pose_label,
             pose_detected: bool, predicted_label=None,
             pred_confidence=0.0, now=None):
    if now is None:
        now = time.time()

    h, w = frame.shape[:2]
    s    = state

    # ── Song + mode banner (always visible) ──────────────────────────────────
    mode_tag = (
        f"DO THE {pose_label.upper().replace('_', ' ')} POSE"
        if pose_label else "DANCE TO CLEAR"
    )
    _shadow(frame, f"{song_name.upper()}  |  {mode_tag}",
            (20, h - 20), 0.55, (255, 220, 0), 1)

    phase = s["phase"]

    if phase == "waiting":
        _shadow(frame, "Step into frame to start",
                (w // 2 - 220, h // 2), 1.0, (255, 255, 255))

    elif phase == "countdown":
        remaining = max(1, COUNTDOWN_SECS - int(now - s["phase_start"]))
        _shadow(frame, "ALARM IS ON — GET READY!",
                (w // 2 - 255, h // 2 - 70), 1.0, (0, 80, 255), 2)
        if s["attempt"] > 0:
            _shadow(frame, f"Attempt {s['attempt'] + 1}",
                    (w // 2 - 80, h // 2 - 20), 0.9, (200, 200, 0), 2)
        _shadow(frame, str(remaining),
                (w // 2 - 40, h // 2 + 70), 3.5, (0, 220, 255), 4)

    elif phase == "dancing":
        elapsed   = now - s["phase_start"]
        time_left = max(0.0, REQUIRED_TIME - elapsed)

        _shadow(frame, "ALARM ON", (20, 38), 0.8, (0, 60, 255), 2)
        if s["attempt"] > 0:
            _shadow(frame, f"attempt {s['attempt'] + 1}",
                    (180, 38), 0.65, (200, 150, 0), 1)
        _shadow(frame, f"Time left: {time_left:.1f}s", (20, 68),
                0.9, (0, 220, 255))

        if pose_label:
            _shadow(frame, f"Hold the {pose_label.replace('_', ' ').upper()} pose!",
                    (20, 100), 0.75, (255, 220, 0), 2)
            _score_bar(frame, s.get("score", 0), POSE_HOLD_NEEDED, 20, 112)
            if pose_detected:
                color = (0, 220, 80) if predicted_label == pose_label else (0, 60, 255)
                _shadow(frame,
                        f"Detected: {predicted_label} ({pred_confidence:.0%})",
                        (20, 152), 0.65, color, 1)
        else:
            _shadow(frame, "Score", (20, 100), 0.65, (200, 200, 200), 1)
            _score_bar(frame, s.get("score", 0), SCORE_THRESHOLD, 20, 108)
            if s.get("score", 0) < SCORE_THRESHOLD * 0.4:
                _shadow(frame, "DANCE HARDER!",
                        (w // 2 - 155, h - 70), 1.3, (0, 60, 255), 3)

    _shadow(frame, "ESC to force quit", (w - 200, h - 15), 0.5, (120, 120, 120), 1)
    return frame
