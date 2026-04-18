import cv2
import time
from config import (
    WEBCAM_INDEX, REQUIRED_TIME, SCORE_THRESHOLD,
    POSE_CONFIDENCE, POSE_HOLD_NEEDED, POSE_GRACE_SECS, COUNTDOWN_SECS,
)
from detection.pose import PoseDetector
from detection.classifier import PoseClassifier
from detection.moves import landmarks_to_positions, score_movement
from ui.hud import draw_hud


def run_alarm(speaker, song_url: str, pose_label=None, song_name=""):
    """
    pose_label=None → movement mode (dance freely to clear)
    pose_label="x"  → pose mode (hold that ML-classified pose to clear)
    """
    speaker.play(song_url)

    cap     = cv2.VideoCapture(WEBCAM_INDEX)
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    with PoseDetector() as detector:
        clf = PoseClassifier() if pose_label else None

        state = {
            "phase":            "waiting",
            "phase_start":      None,
            "score":            0.0,
            "prev_positions":   {},
            "attempt":          0,
            "pose_hold_start":  None,
            "last_correct_time": None,
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            now   = time.time()

            lm               = detector.get_landmarks(frame)
            pose_detected    = lm is not None
            current_pos      = {}
            predicted_label  = None
            pred_confidence  = 0.0

            if pose_detected:
                detector.draw_skeleton(frame, lm)
                current_pos = landmarks_to_positions(lm)
                if clf:
                    predicted_label, pred_confidence = clf.predict(lm)

            s = state

            # ── Phase machine ──────────────────────────────────────────────
            if s["phase"] == "waiting":
                if pose_detected:
                    s.update(phase="countdown", phase_start=now,
                             score=0.0, prev_positions={}, pose_hold_start=None)

            elif s["phase"] == "countdown":
                if not pose_detected:
                    s["phase"] = "waiting"
                elif now - s["phase_start"] >= COUNTDOWN_SECS:
                    s["phase"]        = "dancing"
                    s["phase_start"]  = now
                    s["prev_positions"] = current_pos.copy()

            elif s["phase"] == "dancing":
                elapsed = now - s["phase_start"]

                if pose_label:
                    # Pose mode: hold the correct pose for POSE_HOLD_NEEDED seconds
                    correct = (
                        pose_detected
                        and predicted_label == pose_label
                        and pred_confidence >= POSE_CONFIDENCE
                    )
                    if correct:
                        if s["pose_hold_start"] is None:
                            s["pose_hold_start"] = now
                        s["last_correct_time"] = now
                        s["score"] = now - s["pose_hold_start"]
                    else:
                        last_ok = s["last_correct_time"]
                        if last_ok is None or (now - last_ok) > POSE_GRACE_SECS:
                            s["pose_hold_start"] = None
                            s["last_correct_time"] = None
                            s["score"] = 0.0

                    if s["score"] >= POSE_HOLD_NEEDED:
                        s["phase"] = "done"
                    elif elapsed >= REQUIRED_TIME:
                        s["attempt"] += 1
                        s.update(phase="countdown", phase_start=now,
                                 score=0.0, pose_hold_start=None,
                                 last_correct_time=None)
                else:
                    # Movement mode: accumulate dancing time
                    delta = score_movement(current_pos, s["prev_positions"])
                    if delta:
                        s["score"] += 1.0 / cap_fps
                    else:
                        s["score"] = max(0.0, s["score"] - 0.5 / cap_fps)
                    s["prev_positions"] = current_pos.copy()

                    if elapsed >= REQUIRED_TIME:
                        if s["score"] >= SCORE_THRESHOLD:
                            s["phase"] = "done"
                        else:
                            s["attempt"] += 1
                            s.update(phase="countdown", phase_start=now,
                                     score=0.0, prev_positions={})

            if s["phase"] == "done":
                break

            frame = draw_hud(
                frame, s, song_name, pose_label,
                pose_detected, predicted_label, pred_confidence, now,
            )
            cv2.imshow("Dance to dismiss!", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC force-quit
                break

    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Alarm dismissed!")
