from unittest.mock import patch, MagicMock, call
import sys
sys.path.insert(0, ".")

def test_alarm_dismisses_after_all_moves():
    """Alarm loop exits and calls speaker.stop() once all moves complete."""
    with patch("alarm.alarm_loop.cv2") as mock_cv2, \
         patch("alarm.alarm_loop.PoseDetector") as MockDetector, \
         patch("alarm.alarm_loop.check_move") as mock_check, \
         patch("alarm.alarm_loop.draw_hud", return_value=MagicMock()), \
         patch("alarm.alarm_loop.time") as mock_time:

        mock_cv2.VideoCapture.return_value.read.return_value = (True, MagicMock())
        mock_cv2.waitKey.return_value = 0

        detector_instance = MockDetector.return_value
        detector_instance.get_landmarks.return_value = [MagicMock()] * 33

        # Each check_move call returns True immediately, hold timer fires
        mock_check.return_value = True
        mock_time.time.side_effect = [0, 0, 2, 2, 4, 4, 6]  # enough ticks

        speaker = MagicMock()

        from alarm.alarm_loop import run_alarm
        run_alarm(speaker, "http://fake/song.mp3")

        speaker.play.assert_called_once_with("http://fake/song.mp3")
        speaker.stop.assert_called_once()
