import time
import pychromecast

from env_config import GOOGLE_HOME_IP, SONGS_BASE_URL

TEST_URL = f"{SONGS_BASE_URL}/scuba.mp3"

print(f"1. Attempting connection to {GOOGLE_HOME_IP}...")

try:
    cast = pychromecast.Chromecast(GOOGLE_HOME_IP)
    cast.wait()
    print(f"2. SUCCESS! Connected to: {cast.device.friendly_name}")

    cast.set_volume(0.7)

    mc = cast.media_controller
    print("3. Sending test audio stream...")
    mc.play_media(TEST_URL, "audio/mp3")
    mc.block_until_active()

    print("4. Playing! Listen to the Nest Mini... (Playing for 10 seconds)")
    time.sleep(10)

    print("5. Stopping audio.")
    mc.stop()

except Exception as e:
    print(f"\n[ERROR] Connection Failed: {e}")
    print("This means the IP is wrong, or the Windows Hotspot is blocking 'AP Isolation' traffic.")
