"""
stream.py — headless FastAPI server for remote testing
───────────────────────────────────────────────────────
Run from project root:
    uvicorn stream:app --host 0.0.0.0 --port 8000

Then open http://<pi-or-mac-ip>:8000 from your laptop browser.
The page shows the live camera feed, alarm status, and trigger/stop buttons.
"""

import threading
import time
import random
import cv2
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

from config import (
    WEBCAM_INDEX, SONGS, SONGS_BASE_URL, GOOGLE_HOME_IP,
    REQUIRED_TIME, SCORE_THRESHOLD, POSE_CONFIDENCE,
    POSE_HOLD_NEEDED, COUNTDOWN_SECS,
)
from detection.pose import PoseDetector
from detection.classifier import PoseClassifier
from detection.moves import landmarks_to_positions, score_movement
from audio.speaker import Speaker

# ── Shared state ──────────────────────────────────────────────────────────────
_lock         = threading.Lock()
_latest_jpeg  = None          # latest encoded frame, read by MJPEG endpoint
_state = {
    "phase":           "idle",   # idle | waiting | countdown | dancing | done
    "song":            "",
    "pose_label":      None,
    "score":           0.0,
    "attempt":         0,
    "phase_start":     None,
    "pose_hold_start": None,
    "prev_positions":  {},
    "cap_fps":         30.0,
}
_speaker     = None
_speaker_lock = threading.Lock()


# ── Speaker helpers ───────────────────────────────────────────────────────────
def _play(url: str):
    global _speaker
    with _speaker_lock:
        try:
            _speaker = Speaker(GOOGLE_HOME_IP)
            _speaker.play(url)
        except Exception as e:
            print(f"[speaker] {e}")

def _stop_speaker():
    with _speaker_lock:
        global _speaker
        if _speaker:
            try:
                _speaker.stop()
            except Exception:
                pass
            _speaker = None


# ── Camera + detection loop (runs forever in background) ─────────────────────
def _camera_loop():
    global _latest_jpeg
    detector = PoseDetector()
    clf      = PoseClassifier()
    cap      = cv2.VideoCapture(WEBCAM_INDEX)

    with _lock:
        _state["cap_fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        now   = time.time()

        lm              = detector.get_landmarks(frame)
        pose_detected   = lm is not None
        current_pos     = {}
        pred_label      = None
        pred_conf       = 0.0

        if lm:
            detector.draw_skeleton(frame, lm)
            current_pos = landmarks_to_positions(lm)
            pred_label, pred_conf = clf.predict(lm)

        # ── State machine ─────────────────────────────────────────────────
        became_done = False
        with _lock:
            s         = _state
            phase     = s["phase"]
            cap_fps   = s["cap_fps"]
            pose_label = s["pose_label"]

            if phase == "waiting":
                if pose_detected:
                    s.update(phase="countdown", phase_start=now,
                             score=0.0, pose_hold_start=None, prev_positions={})

            elif phase == "countdown":
                if not pose_detected:
                    s["phase"] = "waiting"
                elif now - s["phase_start"] >= COUNTDOWN_SECS:
                    s["phase"]          = "dancing"
                    s["phase_start"]    = now
                    s["prev_positions"] = current_pos.copy()

            elif phase == "dancing":
                elapsed = now - s["phase_start"]

                if pose_label:
                    correct = (
                        pose_detected
                        and pred_label == pose_label
                        and pred_conf >= POSE_CONFIDENCE
                    )
                    if correct:
                        if s["pose_hold_start"] is None:
                            s["pose_hold_start"] = now
                        s["score"] = now - s["pose_hold_start"]
                    else:
                        s["pose_hold_start"] = None
                        s["score"] = 0.0

                    if s["score"] >= POSE_HOLD_NEEDED:
                        s["phase"] = "done"
                        became_done = True
                    elif elapsed >= REQUIRED_TIME:
                        s["attempt"] += 1
                        s.update(phase="countdown", phase_start=now,
                                 score=0.0, pose_hold_start=None)
                else:
                    delta = score_movement(current_pos, s["prev_positions"])
                    if delta:
                        s["score"] += 1.0 / cap_fps
                    else:
                        s["score"] = max(0.0, s["score"] - 0.5 / cap_fps)
                    s["prev_positions"] = current_pos.copy()

                    if elapsed >= REQUIRED_TIME:
                        if s["score"] >= SCORE_THRESHOLD:
                            s["phase"] = "done"
                            became_done = True
                        else:
                            s["attempt"] += 1
                            s.update(phase="countdown", phase_start=now,
                                     score=0.0, prev_positions={})

            snapshot = {k: v for k, v in s.items()
                        if k not in ("prev_positions",)}

        if became_done:
            threading.Thread(target=_stop_speaker, daemon=True).start()

        # ── Draw HUD onto frame ───────────────────────────────────────────
        _draw_hud(frame, snapshot, pred_label, pred_conf, now)

        # ── Encode to JPEG ────────────────────────────────────────────────
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with _lock:
            _latest_jpeg = buf.tobytes()


# ── HUD drawing ───────────────────────────────────────────────────────────────
def _shadow(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_hud(frame, s, pred_label, pred_conf, now):
    h, w  = frame.shape[:2]
    phase = s["phase"]

    if phase == "idle":
        _shadow(frame, "Idle — press Trigger to start alarm",
                (20, 40), color=(160, 160, 160))
        return

    # Song banner
    _shadow(frame, f"Song: {s['song']}", (20, h - 15), 0.55, (255, 220, 0))

    if phase == "waiting":
        _shadow(frame, "Step into frame to start", (20, 40), 0.9)

    elif phase == "countdown":
        remaining = max(1, COUNTDOWN_SECS - int(now - s["phase_start"]))
        _shadow(frame, "GET READY!", (20, 40), 1.0, (0, 80, 255), 2)
        _shadow(frame, str(remaining), (w // 2 - 25, h // 2), 3.0, (0, 220, 255), 4)
        if s["attempt"] > 0:
            _shadow(frame, f"Attempt {s['attempt'] + 1}", (20, 75), 0.7, (200, 150, 0))

    elif phase == "dancing":
        elapsed   = now - s["phase_start"]
        time_left = max(0.0, REQUIRED_TIME - elapsed)
        score     = s["score"]
        target    = POSE_HOLD_NEEDED if s["pose_label"] else SCORE_THRESHOLD

        _shadow(frame, f"Time left: {time_left:.1f}s", (20, 35), 0.8, (0, 220, 255))

        # Score bar
        pct  = min(score / target, 1.0)
        fill = int(300 * pct)
        cv2.rectangle(frame, (20, 50), (320, 74), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 50), (20 + fill, 74), (0, 220, 80), -1)
        cv2.rectangle(frame, (20, 50), (320, 74), (180, 180, 180), 1)
        _shadow(frame, f"{int(pct * 100)}%", (330, 70), 0.55)

        if s["pose_label"]:
            _shadow(frame, f"Hold: {s['pose_label'].replace('_', ' ').upper()}",
                    (20, 95), 0.75, (255, 220, 0), 2)
            if pred_label:
                color = (0, 220, 80) if pred_label == s["pose_label"] else (0, 60, 255)
                _shadow(frame, f"Detected: {pred_label} ({pred_conf:.0%})",
                        (20, 120), 0.65, color)
        else:
            if score < SCORE_THRESHOLD * 0.4:
                _shadow(frame, "DANCE HARDER!", (20, 95), 0.9, (0, 60, 255), 2)

    elif phase == "done":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        _shadow(frame, "ALARM CLEARED!", (w // 2 - 170, h // 2 + 15),
                1.6, (0, 230, 80), 3)


# ── App (lifespan starts camera thread after all functions are defined) ───────
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_camera_loop, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)


# ── MJPEG stream ──────────────────────────────────────────────────────────────
def _mjpeg_generator():
    while True:
        with _lock:
            jpeg = _latest_jpeg
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.033)   # ~30 fps


@app.get("/video")
def video():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Control endpoints ─────────────────────────────────────────────────────────
@app.post("/trigger")
def trigger():
    with _lock:
        if _state["phase"] not in ("idle", "done"):
            return JSONResponse({"ok": False, "reason": "alarm already running"})

    song = random.choice(SONGS)
    url  = f"{SONGS_BASE_URL}/{song['file']}"

    with _lock:
        _state.update(
            phase="waiting", song=song["name"], pose_label=song["pose_label"],
            score=0.0, attempt=0, phase_start=None,
            pose_hold_start=None, prev_positions={},
        )

    threading.Thread(target=_play, args=(url,), daemon=True).start()
    return {"ok": True, "song": song["name"], "pose_label": song["pose_label"]}


@app.post("/stop")
def stop():
    with _lock:
        _state.update(phase="idle", song="", score=0.0, attempt=0)
    threading.Thread(target=_stop_speaker, daemon=True).start()
    return {"ok": True}


@app.get("/status")
def status():
    with _lock:
        s = {k: v for k, v in _state.items() if k not in ("prev_positions",)}
    return JSONResponse(s)


# ── Dashboard HTML ────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>ShakeUs — Remote Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f0f; color: #eee; font-family: sans-serif;
           display: flex; flex-direction: column; align-items: center;
           padding: 24px; gap: 20px; }
    h1  { font-size: 1.4rem; letter-spacing: 2px; color: #fff; }
    img { width: 100%; max-width: 720px; border-radius: 8px;
          border: 2px solid #333; }
    .controls { display: flex; gap: 12px; }
    button {
      padding: 12px 32px; border: none; border-radius: 6px;
      font-size: 1rem; font-weight: bold; cursor: pointer;
    }
    #btn-trigger { background: #22c55e; color: #000; }
    #btn-trigger:disabled { background: #166534; color: #555; cursor: default; }
    #btn-stop    { background: #ef4444; color: #fff; }
    .status-box {
      width: 100%; max-width: 720px; background: #1a1a1a;
      border-radius: 8px; padding: 16px; display: grid;
      grid-template-columns: 1fr 1fr; gap: 8px 24px;
    }
    .row { display: contents; }
    .label { color: #888; font-size: 0.8rem; text-transform: uppercase; }
    .value { font-size: 1rem; font-weight: bold; }
    .phase-idle     { color: #888; }
    .phase-waiting  { color: #60a5fa; }
    .phase-countdown{ color: #facc15; }
    .phase-dancing  { color: #4ade80; }
    .phase-done     { color: #a3e635; }
    .bar-wrap { grid-column: 1 / -1; background: #333; border-radius: 4px;
                height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; background: #22c55e;
                transition: width 0.3s ease; }
  </style>
</head>
<body>
  <h1>ShakeUs Dashboard</h1>
  <img id="feed" src="/video" alt="Camera feed">

  <div class="controls">
    <button id="btn-trigger" onclick="trigger()">Trigger Alarm</button>
    <button id="btn-stop"    onclick="stop()">Stop</button>
  </div>

  <div class="status-box">
    <span class="label">Phase</span>
    <span class="value" id="s-phase">—</span>

    <span class="label">Song</span>
    <span class="value" id="s-song">—</span>

    <span class="label">Mode</span>
    <span class="value" id="s-mode">—</span>

    <span class="label">Score</span>
    <span class="value" id="s-score">—</span>

    <span class="label">Attempt</span>
    <span class="value" id="s-attempt">—</span>

    <span class="label">Progress</span>
    <div class="bar-wrap">
      <div class="bar-fill" id="s-bar" style="width:0%"></div>
    </div>
  </div>

  <script>
    async function trigger() {
      document.getElementById('btn-trigger').disabled = true;
      const r = await fetch('/trigger', {method:'POST'});
      const d = await r.json();
      if (!d.ok) {
        alert(d.reason);
        document.getElementById('btn-trigger').disabled = false;
      }
    }

    async function stop() {
      await fetch('/stop', {method:'POST'});
      document.getElementById('btn-trigger').disabled = false;
    }

    async function poll() {
      try {
        const r = await fetch('/status');
        const s = await r.json();

        const phase = s.phase || 'idle';
        const el = document.getElementById('s-phase');
        el.textContent = phase.toUpperCase();
        el.className   = 'value phase-' + phase;

        document.getElementById('s-song').textContent    = s.song || '—';
        document.getElementById('s-mode').textContent    = s.pose_label ? 'Pose: ' + s.pose_label : 'Movement';
        document.getElementById('s-score').textContent   = (s.score || 0).toFixed(2);
        document.getElementById('s-attempt').textContent = s.attempt ?? '—';

        const target = s.pose_label ? """ + str(POSE_HOLD_NEEDED) + """ : """ + str(SCORE_THRESHOLD) + """;
        const pct    = Math.min((s.score || 0) / target * 100, 100).toFixed(1);
        document.getElementById('s-bar').style.width = pct + '%';

        const idle = phase === 'idle' || phase === 'done';
        document.getElementById('btn-trigger').disabled = !idle;
      } catch(e) {}
    }

    setInterval(poll, 500);
    poll();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


if __name__ == "__main__":
    uvicorn.run("stream:app", host="0.0.0.0", port=8000, reload=False)
