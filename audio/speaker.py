import pychromecast, time

class Speaker:
    def __init__(self, ip: str):
        chromecasts, browser = pychromecast.get_listed_chromecasts(
            known_hosts=[ip])
        if not chromecasts:
            raise RuntimeError(f"No Chromecast found at {ip}")
        self.cast = chromecasts[0]
        self.cast.wait()
        self.mc = self.cast.media_controller

    def play(self, url: str, volume: float = 0.8):
        self.cast.set_volume(volume)
        self.mc.play_media(url, "audio/mp3")
        self.mc.block_until_active()

    def stop(self):
        self.mc.stop()
