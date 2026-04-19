# TikTok Alarm

A dance-to-dismiss alarm: plays music through Google Home and only stops when you complete a sequence of poses detected via webcam.

---

## Prerequisites

```bash
pip install -r requirements.txt        # Mac
pip install -r pi_requirements.txt     # Raspberry Pi only
```

---

## Find your IPs

**Mac IP**
```bash
ipconfig getifaddr en0
```

**Google Home IP**
Open the Google Home app → select your device → Settings → Device info → IP address.

Update both values in `config.py` before running anything.

---

## Add songs

Download 3–5 `.mp3` files into `assets/songs/`. They are gitignored — share via Google Drive or USB.

```bash
# Example using yt-dlp
yt-dlp -x --audio-format mp3 -o "assets/songs/%(title)s.mp3" "<youtube-url>"
```

---

## Run on Mac

```bash
python main.py
```

This starts the Flask server on port 5050. It serves songs and listens for alarm triggers.

---

## Deploy to Raspberry Pi

Copy the two files the Pi needs:

```bash
scp config.py alarm/scheduler.py pi@<pi-ip>:/home/pi/tiktok-alarm/
```

On the Pi, install dependencies and set up autostart:

```bash
pip3 install -r pi_requirements.txt

crontab -e
# Add this line:
@reboot cd /home/pi/tiktok-alarm && python3 scheduler.py &
```

Run manually on the Pi:

```bash
cd /home/pi/tiktok-alarm
python3 scheduler.py
```

---

## Test without waiting for alarm time

Trigger the alarm immediately from any machine on the same network:

```bash
python alarm/trigger_client.py
```

---

## Network check (do this first)

Before writing any code, confirm the Mac, Pi, and Google Home are all on the same Wi-Fi:

```bash
python -c "import pychromecast; print(pychromecast.get_chromecasts())"
```

If your Google Home appears in the output, everything is wired up correctly.

---

## Add a new dance move

1. Write a new function in [`detection/moves.py`](detection/moves.py) following the same signature: `def my_move(lm) -> bool`
2. Register it in `MOVE_REGISTRY` at the bottom of that file
3. Add `"my_move"` to `MOVE_SEQUENCE` in [`config.py`](config.py)

## Run SonarQube locally

This repo includes a ready-to-use SonarScanner config in `sonar-project.properties`.

1. Start a local SonarQube server on `http://localhost:9000`
2. Install scanner + app dependencies if needed

```bash
python3 -m pip install -r requirements.txt
brew install sonar-scanner
```

3. Run the scan from the repo root

```bash
sonar-scanner
```

4. Open the SonarQube dashboard and inspect:

- Bugs
- Code Smells
- Duplications

### Useful debug commands

Verbose scanner output:

```bash
sonar-scanner -X
```
