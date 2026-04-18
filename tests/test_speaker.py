from unittest.mock import patch, MagicMock

def test_speaker_play_calls_chromecast():
    with patch("audio.speaker.pychromecast") as mock_cc:
        mock_cast = MagicMock()
        mock_cc.get_listed_chromecasts.return_value = ([mock_cast], None)

        from audio.speaker import Speaker
        s = Speaker("192.168.1.1")
        s.play("http://fake/song.mp3")

        mock_cast.media_controller.play_media.assert_called_once()

def test_speaker_stop():
    with patch("audio.speaker.pychromecast") as mock_cc:
        mock_cast = MagicMock()
        mock_cc.get_listed_chromecasts.return_value = ([mock_cast], None)

        from audio.speaker import Speaker
        s = Speaker("192.168.1.1")
        s.stop()
        mock_cast.media_controller.stop.assert_called_once()
