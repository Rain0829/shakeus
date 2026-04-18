import cv2, time
from config import MOVE_SEQUENCE, MOVE_HOLD_SECONDS, WEBCAM_INDEX
from detection.pose import PoseDetector
from detection.moves import check_move
from ui.hud import draw_hud

def run_alarm(speaker, song_url: str):
    speaker.play(song_url)

    detector   = PoseDetector()
    cap        = cv2.VideoCapture(WEBCAM_INDEX)
    move_idx   = 0        # index into MOVE_SEQUENCE
    hold_start = None     # timestamp when current move started being held

    while move_idx < len(MOVE_SEQUENCE):
        ret, frame = cap.read()
        if not ret:
            continue

        lm           = detector.get_landmarks(frame)
        current_move = MOVE_SEQUENCE[move_idx]
        move_done    = False

        if lm:
            if check_move(lm, current_move):
                if hold_start is None:
                    hold_start = time.time()
                elif time.time() - hold_start >= MOVE_HOLD_SECONDS:
                    move_idx  += 1
                    hold_start = None
                    move_done  = True
            else:
                hold_start = None   # dropped pose, reset timer

        frame = draw_hud(frame, lm, move_idx, MOVE_SEQUENCE,
                         hold_start, move_done)
        cv2.imshow("Dance to dismiss!", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # All moves done — dismiss
    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Alarm dismissed!")
