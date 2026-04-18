import cv2, time
from config import MOVE_HOLD_SECONDS

def draw_hud(frame, landmarks, move_idx, sequence, hold_start, move_done=False):
    h, w = frame.shape[:2]

    # ── Move progress pills at top ──────────────────
    for i, move in enumerate(sequence):
        x = 20 + i * 160
        if i < move_idx:
            color = (60, 200, 60)    # green = done
        elif i == move_idx:
            color = (0, 180, 255)    # orange = current
        else:
            color = (60, 60, 60)     # dark = pending
        cv2.rectangle(frame, (x, 16), (x + 148, 44), color, -1)
        cv2.rectangle(frame, (x, 16), (x + 148, 44), (255,255,255), 1)
        cv2.putText(frame, move, (x + 8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ── Hold timer circle (bottom center) ───────────
    cx, cy = w // 2, h - 70
    if hold_start and move_idx < len(sequence):
        elapsed = time.time() - hold_start
        pct     = min(elapsed / MOVE_HOLD_SECONDS, 1.0)
        angle   = int(pct * 360)
        cv2.circle(frame, (cx, cy), 44, (40, 40, 40), -1)
        cv2.ellipse(frame, (cx, cy), (44, 44), -90, 0, angle,
                    (0, 220, 100), 5)
        label = sequence[move_idx] if move_idx < len(sequence) else ""
        cv2.putText(frame, label, (cx - 40, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ── "Move done!" flash ───────────────────────────
    if move_done:
        cv2.putText(frame, "MOVE DONE!", (w//2 - 80, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 120), 2)

    return frame
