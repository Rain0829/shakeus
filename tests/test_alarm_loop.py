import sys
sys.path.insert(0, ".")

from itertools import chain, repeat
from unittest.mock import patch, MagicMock


def _make_fake_landmark(x=0.5, y=0.5, z=0.0):
    lm = MagicMock()
    lm.x, lm.y, lm.z = x, y, z
    return lm


def test_alarm_dismisses_movement_mode():
    """Movement-mode alarm calls speaker.stop() after enough dancing."""
    fake_lm = [_make_fake_landmark() for _ in range(33)]

    with patch("alarm.alarm_loop.cv2") as mock_cv2, \
         patch("alarm.alarm_loop.PoseDetector") as MockDetector, \
         patch("alarm.alarm_loop.score_movement") as mock_score, \
         patch("alarm.alarm_loop.landmarks_to_positions", return_value={}), \
         patch("alarm.alarm_loop.draw_hud", return_value=MagicMock()), \
         patch("alarm.alarm_loop.time") as mock_time:

        mock_cv2.VideoCapture.return_value.read.return_value = (True, MagicMock())
        mock_cv2.VideoCapture.return_value.get.return_value = 1.0
        mock_cv2.waitKey.return_value = 0

        detector_inst = MockDetector.return_value.__enter__.return_value
        detector_inst.get_landmarks.return_value = fake_lm

        # Simulate enough 1 FPS dancing frames to reach SCORE_THRESHOLD before
        # the REQUIRED_TIME window ends.
        mock_time.time.side_effect = chain([
            0,    # waiting → countdown
            3.1,  # countdown done (>= COUNTDOWN_SECS=3)
            3.1,  # first dancing frame, score = 1.0
            4.1,  # second dancing frame, score = 2.0
            14.0, # third dancing frame, score = 3.0 and elapsed > REQUIRED_TIME
        ], repeat(14.0))
        mock_score.return_value = 1.0  # always moving → score accumulates

        speaker = MagicMock()

        from alarm.alarm_loop import run_alarm
        run_alarm(speaker, "http://fake/song.mp3", pose_label=None, song_name="Test")

        speaker.play.assert_called_once_with("http://fake/song.mp3")
        speaker.stop.assert_called_once()


def test_alarm_dismisses_pose_mode():
    """Pose-mode alarm calls speaker.stop() after holding the correct pose."""
    fake_lm = [_make_fake_landmark() for _ in range(33)]

    with patch("alarm.alarm_loop.cv2") as mock_cv2, \
         patch("alarm.alarm_loop.PoseDetector") as MockDetector, \
         patch("alarm.alarm_loop.PoseClassifier") as MockClf, \
         patch("alarm.alarm_loop.landmarks_to_positions", return_value={}), \
         patch("alarm.alarm_loop.score_movement", return_value=0.0), \
         patch("alarm.alarm_loop.draw_hud", return_value=MagicMock()), \
         patch("alarm.alarm_loop.time") as mock_time:

        mock_cv2.VideoCapture.return_value.read.return_value = (True, MagicMock())
        mock_cv2.VideoCapture.return_value.get.return_value = 30.0
        mock_cv2.waitKey.return_value = 0

        detector_inst = MockDetector.return_value.__enter__.return_value
        detector_inst.get_landmarks.return_value = fake_lm

        clf_inst = MockClf.return_value
        clf_inst.predict.return_value = ("scuba", 0.95)  # always confident

        mock_time.time.side_effect = chain([
            0,     # waiting → countdown
            3.1,   # countdown done
            3.1,   # dancing start, pose_hold_start set
            4.1,   # hold duration = 1.0
            6.2,   # hold duration = 3.1 > POSE_HOLD_NEEDED (3.0) → done
        ], repeat(6.0))

        speaker = MagicMock()

        from alarm.alarm_loop import run_alarm
        run_alarm(speaker, "http://fake/song.mp3", pose_label="scuba", song_name="Scuba")

        speaker.stop.assert_called_once()
