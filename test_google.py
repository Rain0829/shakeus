import time
import pychromecast

# --- CONFIG ---
# Double check your Windows Hotspot to make sure this IP is still correct!
NEST_IP = "192.168.137.220" 
TEST_URL = "https://unresponding-nettie-nonadaptive.ngrok-free.dev/songs/scuba.mp3"

print(f"1. Attempting connection to {NEST_IP}...")

try:
    # Connect directly to the speaker
    cast = pychromecast.Chromecast(NEST_IP)
    cast.wait()
    print(f"2. SUCCESS! Connected to: {cast.device.friendly_name}")

    # Ensure it isn't muted
    cast.set_volume(0.7)
    
    # Send the public audio stream
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