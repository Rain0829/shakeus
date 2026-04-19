"""
stream.py — headless FastAPI server (Mac testing + Pi production)
──────────────────────────────────────────────────────────────────
Run from project root:
    python stream.py

Then open http://<ip>:8080 in a browser.
"""

import threading
import time
import random
import os

import numpy as np
import cv2
import uvicorn
import warnings
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

# Suppress the annoying MediaPipe Protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    WEBCAM_INDEX, SONGS, SONGS_DIR, USE_CHROMECAST, VIDEO_STREAM_MAX_FPS,
    VIDEO_STREAM_ONLY_WHEN_ALARM,
    REQUIRED_TIME, SCORE_THRESHOLD, POSE_CONFIDENCE,
    POSE_HOLD_NEEDED, POSE_GRACE_SECS, COUNTDOWN_SECS,
)
from env_config import GOOGLE_HOME_IP, SONGS_BASE_URL

if USE_CHROMECAST:
    import pychromecast
else:
    import pygame
    pygame.mixer.init()

from detection.pose import PoseDetector
from detection.classifier import PoseClassifier
from detection.moves import landmarks_to_positions, score_movement
from audio.voice_generator import VoiceGenerator

# ── Shared state ──────────────────────────────────────────────────────────────
_lock                   = threading.Lock()
_latest_jpeg            = None
_latest_jpeg_generation = 0  # incremented each new frame; MJPEG readers skip stale duplicates
_state = {
    "phase":            "idle",   # idle | intro | waiting | countdown | dancing | outro | done
    "song":             "",
    "pose_label":       None,
    "score":            0.0,
    "attempt":          0,
    "phase_start":      None,
    "pose_hold_start":  None,
    "last_correct_time": None,
    "prev_positions":   {},
    "cap_fps":          30.0,
}

# ── Remote schedule (alarm at HH:MM local, or one-shot timer) ────────────────
_schedule_lock = threading.Lock()
_schedule = {
    "kind": None,              # None | "alarm" | "timer"
    "song_file": None,         # optional; must match SONGS[].file
    "alarm_h": None,
    "alarm_m": None,
    "timer_until": None,       # time.monotonic() deadline for timer
    "last_alarm_fired": None,  # date — daily alarm fires once per calendar day
}
_scheduler_stop = threading.Event()


def _song_by_file(song_file: str | None):
    """Return song dict from SONGS, or None if not found."""
    if not song_file:
        return None
    for s in SONGS:
        if s["file"] == song_file:
            return s
    return None


def _pick_song(song_file: str | None):
    """Resolve optional file name to a song dict; random if missing/invalid."""
    s = _song_by_file(song_file)
    if s:
        return s
    return random.choice(SONGS)


def _fire_alarm_from_schedule():
    """Start alarm sequence if idle; used by scheduler."""
    song = None
    with _schedule_lock:
        song = _pick_song(_schedule.get("song_file"))
    with _lock:
        if _state["phase"] not in ("idle", "done"):
            return
    threading.Thread(target=_run_sequence, args=(song,), daemon=True).start()


def _scheduler_loop():
    """Wake periodically; fire timer or daily clock alarm."""
    while not _scheduler_stop.wait(timeout=0.5):
        with _schedule_lock:
            kind = _schedule["kind"]
            song_file = _schedule.get("song_file")
            timer_until = _schedule.get("timer_until")
            alarm_h = _schedule.get("alarm_h")
            alarm_m = _schedule.get("alarm_m")
            last_fired = _schedule.get("last_alarm_fired")

        if kind == "timer" and timer_until is not None:
            if time.monotonic() >= timer_until:
                sf = None
                with _schedule_lock:
                    if _schedule.get("kind") != "timer":
                        continue
                    sf = _schedule.get("song_file")
                    _schedule["kind"] = None
                    _schedule["timer_until"] = None
                    _schedule["song_file"] = None
                song = _pick_song(sf)
                with _lock:
                    if _state["phase"] not in ("idle", "done"):
                        continue
                threading.Thread(target=_run_sequence, args=(song,), daemon=True).start()
            continue

        if kind == "alarm" and alarm_h is not None and alarm_m is not None:
            now = datetime.now()
            if now.hour == alarm_h and now.minute == alarm_m and now.second <= 1:
                today = now.date()
                if last_fired != today:
                    with _schedule_lock:
                        _schedule["last_alarm_fired"] = today
                    _fire_alarm_from_schedule()


# ── Audio helpers ─────────────────────────────────────────────────────────────
_chromecast = None

def _loop_chromecast_song(url: str):
    """Background thread to loop Chromecast music when it finishes."""
    global _chromecast
    
    # Wait 5 seconds before we start checking so we don't accidentally 
    # catch the temporary 'IDLE' state while the song first buffers!
    time.sleep(5) 
    
    # Only loop if the user hasn't successfully turned off the alarm yet
    while _state["phase"] in ("waiting", "countdown", "dancing"):
        time.sleep(2)
        if _chromecast:
            try:
                mc = _chromecast.media_controller
                mc.update_status()
                
                # Check if it is IDLE specifically because the track 'FINISHED'
                if mc.status.player_state == 'IDLE' and mc.status.idle_reason == 'FINISHED':
                    print("[chromecast] Song finished! Looping back to start...")
                    mc.play_media(url, "audio/mp3")
                    mc.block_until_active()
                    time.sleep(5) # Buffer delay before we resume checking
            except Exception:
                pass
        else:
            break

def _play_song(song_file: str):
    """Start playing a song (non-blocking)."""
    global _chromecast
    if USE_CHROMECAST:
        try:
            url = f"{SONGS_BASE_URL}/{song_file}"
            print(f"[chromecast] Connecting to {GOOGLE_HOME_IP} for song...")
            _chromecast = pychromecast.Chromecast(GOOGLE_HOME_IP)
            _chromecast.wait()
            _chromecast.set_volume(0.4) # Ensure it isn't muted!
            mc = _chromecast.media_controller
            print(f"[chromecast] Sending URL: {url}")
            mc.play_media(url, "audio/mp3")
            mc.block_until_active()
            print("[chromecast] Beat dropped successfully. Looping enabled.")
            
            # Start the background watcher to loop the song if it ends
            threading.Thread(target=_loop_chromecast_song, args=(url,), daemon=True).start()
        except Exception as e:
            print(f"[chromecast] Error: {e}")
    else:
        try:
            pygame.mixer.music.load(os.path.join(SONGS_DIR, song_file))
            pygame.mixer.music.set_volume(0.8)
            pygame.mixer.music.play(-1)  # -1 loops the music indefinitely
        except Exception as e:
            print(f"[audio] {e}")

def _play_tts(filepath: str):
    """Play a TTS file and block until it finishes."""
    if USE_CHROMECAST:
        try:
            url = f"{SONGS_BASE_URL}/{os.path.basename(filepath)}"
            print(f"[chromecast tts] Connecting to {GOOGLE_HOME_IP} for voice...")
            cast = pychromecast.Chromecast(GOOGLE_HOME_IP)
            cast.wait()
            cast.set_volume(0.9) # AI Voice needs to be loud and clear
            mc = cast.media_controller
            print(f"[chromecast tts] Sending URL: {url}")
            mc.play_media(url, "audio/mp3")
            mc.block_until_active()
            print("[chromecast tts] Speaking...")
            time.sleep(8)   # Wait for AI Voice to finish speaking
            mc.stop()
        except Exception as e:
            print(f"[chromecast tts] Error: {e}")
    else:
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.set_volume(0.9)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"[audio tts] {e}")

def _stop_song():
    global _chromecast
    if USE_CHROMECAST:
        if _chromecast:
            try:
                _chromecast.media_controller.stop()
                print("[chromecast] Audio stopped.")
            except Exception:
                pass
            _chromecast = None
    else:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass


# ── Voice sequence ────────────────────────────────────────────────────────────
def _run_sequence(song: dict):
    """Full alarm sequence: AI intro → song + alarm → AI outro."""

    # 1. Generate + play intro
    with _lock:
        _state["phase"] = "intro"

    try:
        vg = VoiceGenerator()
        intro_text = vg.generate_phrase("intro")
        print(f"[voice] Intro: {intro_text}")
        intro_path = vg.create_tts_audio(intro_text, "current_intro.mp3")
        _play_tts(intro_path)
    except Exception as e:
        print(f"[voice intro skipped] {e}")

    # 2. Start song and begin alarm state machine
    _play_song(song["file"])
    with _lock:
        _state.update(
            phase="waiting", song=song["name"], pose_label=song["pose_label"],
            score=0.0, attempt=0, phase_start=None,
            pose_hold_start=None, last_correct_time=None, prev_positions={},
        )


def _play_outro():
    """Stop song, generate outro, play it, then go idle."""
    _stop_song()

    with _lock:
        _state["phase"] = "outro"

    try:
        vg = VoiceGenerator()
        outro_text = vg.generate_phrase("outro")
        print(f"[voice] Outro: {outro_text}")
        outro_path = vg.create_tts_audio(outro_text, "current_outro.mp3")
        _play_tts(outro_path)
    except Exception as e:
        print(f"[voice outro skipped] {e}")

    with _lock:
        _state["phase"] = "done"


# ── Camera + detection loop ───────────────────────────────────────────────────
def _camera_loop():
    global _latest_jpeg, _latest_jpeg_generation
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

        lm            = detector.get_landmarks(frame)
        pose_detected = lm is not None
        current_pos   = {}
        pred_label    = None
        pred_conf     = 0.0

        if lm:
            detector.draw_skeleton(frame, lm)
            current_pos = landmarks_to_positions(lm)
            pred_label, pred_conf = clf.predict(lm)

        became_done = False
        with _lock:
            s          = _state
            phase      = s["phase"]
            cap_fps    = s["cap_fps"]
            pose_label = s["pose_label"]

            if phase == "waiting":
                if pose_detected:
                    s.update(phase="countdown", phase_start=now,
                             score=0.0, pose_hold_start=None,
                             last_correct_time=None, prev_positions={})

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
                        s["last_correct_time"] = now
                        s["score"] = now - s["pose_hold_start"]
                    else:
                        last_ok = s["last_correct_time"]
                        if last_ok is None or (now - last_ok) > POSE_GRACE_SECS:
                            s["pose_hold_start"]   = None
                            s["last_correct_time"] = None
                            s["score"]             = 0.0

                    if s["score"] >= POSE_HOLD_NEEDED:
                        s["phase"] = "outro"
                        became_done = True
                    elif elapsed >= REQUIRED_TIME:
                        s["attempt"] += 1
                        s.update(phase="countdown", phase_start=now,
                                 score=0.0, pose_hold_start=None,
                                 last_correct_time=None)
                else:
                    delta = score_movement(current_pos, s["prev_positions"])
                    if delta:
                        s["score"] += 1.0 / cap_fps
                    else:
                        s["score"] = max(0.0, s["score"] - 0.5 / cap_fps)
                    s["prev_positions"] = current_pos.copy()

                    if elapsed >= REQUIRED_TIME:
                        if s["score"] >= SCORE_THRESHOLD:
                            s["phase"] = "outro"
                            became_done = True
                        else:
                            s["attempt"] += 1
                            s.update(phase="countdown", phase_start=now,
                                     score=0.0, prev_positions={})

            snapshot = {k: v for k, v in s.items() if k not in ("prev_positions",)}

        if became_done:
            threading.Thread(target=_play_outro, daemon=True).start()

        _draw_hud(frame, snapshot, pred_label, pred_conf, now)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with _lock:
            _latest_jpeg = buf.tobytes()
            _latest_jpeg_generation += 1


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

    if phase == "intro":
        _shadow(frame, "DJ is warming up...", (20, 40), 1.0, (255, 200, 0), 2)
        return

    if phase == "outro":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        _shadow(frame, "ALARM CLEARED!", (w // 2 - 170, h // 2 - 10),
                1.6, (0, 230, 80), 3)
        _shadow(frame, "DJ is signing off...", (w // 2 - 130, h // 2 + 40),
                0.8, (255, 200, 0))
        return

    if phase == "done":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        _shadow(frame, "ALARM CLEARED!", (w // 2 - 170, h // 2 + 15),
                1.6, (0, 230, 80), 3)
        return

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


# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_camera_loop, daemon=True).start()
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    yield
    _scheduler_stop.set()

app = FastAPI(lifespan=lifespan)
# NOTE: Do not mount StaticFiles at /songs before the JSON route GET /songs — the mount
# would swallow GET /songs. Mount is registered after list_songs() below.

# ── MJPEG stream ──────────────────────────────────────────────────────────────
# Browser: <img src="/video"> — one long-lived GET, not JS fetch(). Server pushes
# multipart JPEGs. We only yield when the camera produced a *new* frame (by
# generation counter) and optionally cap FPS (VIDEO_STREAM_MAX_FPS) to save
# bandwidth on ngrok/mobile vs the old fixed ~30 Hz resend of the same bytes.
#
# If VIDEO_STREAM_ONLY_WHEN_ALARM: no multipart stream while phase is idle/done — see video().

_STANDBY_JPEG: bytes | None = None


def _get_standby_jpeg() -> bytes:
    """Single static frame when live stream is off (saves ngrok bandwidth)."""
    global _STANDBY_JPEG
    if _STANDBY_JPEG is None:
        img = np.full((240, 480, 3), (42, 40, 38), dtype=np.uint8)
        cv2.putText(
            img, "Camera stream only during alarm", (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 205, 200), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, "(idle — low bandwidth mode)", (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 165, 160), 2, cv2.LINE_AA,
        )
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _STANDBY_JPEG = buf.tobytes()
    return _STANDBY_JPEG


def _mjpeg_generator():
    last_sent_gen = -1
    last_send_time = 0.0
    min_interval = (
        (1.0 / VIDEO_STREAM_MAX_FPS) if VIDEO_STREAM_MAX_FPS and VIDEO_STREAM_MAX_FPS > 0 else 0.0
    )

    while True:
        if VIDEO_STREAM_ONLY_WHEN_ALARM:
            with _lock:
                if _state["phase"] in ("idle", "done"):
                    break

        with _lock:
            gen = _latest_jpeg_generation
            jpeg = _latest_jpeg

        if not jpeg:
            time.sleep(0.05)
            continue

        now = time.time()
        if gen == last_sent_gen:
            time.sleep(0.01)
            continue

        if min_interval and (now - last_send_time) < min_interval:
            time.sleep(0.005)
            continue

        with _lock:
            jpeg = _latest_jpeg
            gen = _latest_jpeg_generation
        if not jpeg or gen == last_sent_gen:
            continue

        last_sent_gen = gen
        last_send_time = time.time()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )


@app.get("/video")
def video():
    if VIDEO_STREAM_ONLY_WHEN_ALARM:
        with _lock:
            phase = _state["phase"]
        if phase in ("idle", "done"):
            return Response(content=_get_standby_jpeg(), media_type="image/jpeg")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Control endpoints ─────────────────────────────────────────────────────────
class TriggerBody(BaseModel):
    song_file: str | None = None


def _resolve_trigger_song(song_file: str | None):
    if song_file:
        s = _song_by_file(song_file)
        if not s:
            raise HTTPException(status_code=400, detail=f"Unknown song_file: {song_file}")
        return s
    return random.choice(SONGS)


class ScheduleAlarmBody(BaseModel):
    time: str = Field(..., description="Local time HH:MM (24h)")
    song_file: str | None = None


class ScheduleTimerBody(BaseModel):
    seconds: int = Field(..., ge=1, le=86400)
    song_file: str | None = None


def _parse_hhmm(t: str) -> tuple[int, int]:
    parts = t.strip().split(":")
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="time must be HH:MM")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise HTTPException(status_code=400, detail="invalid time")
    return h, m


@app.get("/songs")
def list_songs():
    return [
        {"name": s["name"], "file": s["file"], "pose_label": s["pose_label"]}
        for s in SONGS
    ]


app.mount("/songs", StaticFiles(directory=SONGS_DIR), name="songs")


@app.post("/trigger")
def trigger(body: TriggerBody | None = Body(default=None)):
    with _lock:
        if _state["phase"] not in ("idle", "done"):
            return JSONResponse({"ok": False, "reason": "alarm already running"})

    song_file = body.song_file if body else None
    song = _resolve_trigger_song(song_file)
    threading.Thread(target=_run_sequence, args=(song,), daemon=True).start()
    return {"ok": True, "song": song["name"], "pose_label": song["pose_label"], "file": song["file"]}


@app.post("/schedule/alarm")
def schedule_alarm(body: ScheduleAlarmBody):
    if body.song_file and not _song_by_file(body.song_file):
        raise HTTPException(status_code=400, detail=f"Unknown song_file: {body.song_file}")
    h, m = _parse_hhmm(body.time)
    with _schedule_lock:
        _schedule.update(
            kind="alarm",
            alarm_h=h,
            alarm_m=m,
            song_file=body.song_file,
            timer_until=None,
            last_alarm_fired=None,
        )
    return {"ok": True, "kind": "alarm", "time": f"{h:02d}:{m:02d}", "song_file": body.song_file}


@app.post("/schedule/timer")
def schedule_timer(body: ScheduleTimerBody):
    if body.song_file and not _song_by_file(body.song_file):
        raise HTTPException(status_code=400, detail=f"Unknown song_file: {body.song_file}")
    until = time.monotonic() + float(body.seconds)
    with _schedule_lock:
        _schedule.update(
            kind="timer",
            alarm_h=None,
            alarm_m=None,
            song_file=body.song_file,
            timer_until=until,
            last_alarm_fired=None,
        )
    return {"ok": True, "kind": "timer", "seconds": body.seconds, "song_file": body.song_file}


@app.post("/schedule/cancel")
def schedule_cancel():
    with _schedule_lock:
        _schedule.update(
            kind=None,
            song_file=None,
            alarm_h=None,
            alarm_m=None,
            timer_until=None,
            last_alarm_fired=None,
        )
    return {"ok": True}


@app.get("/schedule")
def schedule_status():
    with _schedule_lock:
        k = _schedule["kind"]
        out = {"kind": k, "song_file": _schedule.get("song_file")}
        if k == "alarm" and _schedule.get("alarm_h") is not None:
            h, m = _schedule["alarm_h"], _schedule["alarm_m"]
            out["time"] = f"{h:02d}:{m:02d}"
            out["timer_remaining_sec"] = None
        elif k == "timer" and _schedule.get("timer_until") is not None:
            rem = max(0.0, _schedule["timer_until"] - time.monotonic())
            out["timer_remaining_sec"] = round(rem, 1)
            out["time"] = None
        else:
            out["time"] = None
            out["timer_remaining_sec"] = None
    return JSONResponse(out)


@app.post("/stop")
def stop():
    with _lock:
        _state.update(phase="idle", song="", score=0.0, attempt=0)
    threading.Thread(target=_stop_song, daemon=True).start()
    return {"ok": True}


@app.get("/status")
def status():
    with _lock:
        s = {k: v for k, v in _state.items() if k not in ("prev_positions",)}
    return JSONResponse(s)


# Same control endpoints under /api/* so the React app can talk to shakeus directly
# (e.g. VITE_API_BASE_URL=http://127.0.0.1:8080) without the website proxy.
@app.get("/api/songs")
def api_songs():
    return list_songs()


@app.post("/api/trigger")
def api_trigger(body: TriggerBody | None = Body(default=None)):
    return trigger(body)


@app.post("/api/schedule/alarm")
def api_schedule_alarm(body: ScheduleAlarmBody):
    return schedule_alarm(body)


@app.post("/api/schedule/timer")
def api_schedule_timer(body: ScheduleTimerBody):
    return schedule_timer(body)


@app.post("/api/schedule/cancel")
def api_schedule_cancel():
    return schedule_cancel()


@app.get("/api/schedule")
def api_schedule_status():
    return schedule_status()


@app.post("/api/stop")
def api_stop():
    return stop()


@app.get("/api/status")
def api_status():
    return status()


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
    .label { color: #888; font-size: 0.8rem; text-transform: uppercase; }
    .value { font-size: 1rem; font-weight: bold; }
    .phase-idle     { color: #888; }
    .phase-intro    { color: #f97316; }
    .phase-waiting  { color: #60a5fa; }
    .phase-countdown{ color: #facc15; }
    .phase-dancing  { color: #4ade80; }
    .phase-outro    { color: #a78bfa; }
    .phase-done     { color: #a3e635; }
    .bar-wrap { grid-column: 1 / -1; background: #333; border-radius: 4px;
                height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; background: #22c55e; transition: width 0.3s ease; }
  </style>
</head>
<body>
  <h1>ShakeUs Dashboard</h1>
  <img id="feed" src="/video" alt="Camera feed">

  <div class="controls">
    <button id="btn-trigger" onclick="triggerAlarm()">Trigger Alarm</button>
    <button id="btn-stop"    onclick="stopAlarm()">Stop</button>
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
    let pollTimer = null;
    let lastFeedStreamOn = null;

    // Poll /status so the camera img switches between placeholder JPEG and live MJPEG
    // when alarm starts/stops (VIDEO_STREAM_ONLY_WHEN_ALARM). No video bytes while idle.
    (function feedSync() {
      setInterval(async () => {
        try {
          const r = await fetch('/status');
          const s = await r.json();
          const phase = s.phase || 'idle';
          const streamOn = !(phase === 'idle' || phase === 'done');
          if (lastFeedStreamOn === null || streamOn !== lastFeedStreamOn) {
            document.getElementById('feed').src = '/video?t=' + Date.now();
            lastFeedStreamOn = streamOn;
          }
        } catch (e) {}
      }, 1500);
    })();

    function startStatusPolling() {
      if (pollTimer != null) return;
      pollTimer = setInterval(pollStatus, 1000);
      pollStatus();
    }

    function stopStatusPolling() {
      if (pollTimer != null) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    async function triggerAlarm() {
      document.getElementById('btn-trigger').disabled = true;
      const r = await fetch('/trigger', {method:'POST'});
      const d = await r.json();
      if (!d.ok) {
        alert(d.reason);
        document.getElementById('btn-trigger').disabled = false;
        return;
      }
      startStatusPolling();
    }

    async function stopAlarm() {
      stopStatusPolling();
      await fetch('/stop', {method:'POST'});
      document.getElementById('btn-trigger').disabled = false;
    }

    async function pollStatus() {
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
        if (idle) stopStatusPolling();
      } catch(e) {}
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


if __name__ == "__main__":
    uvicorn.run("stream:app", host="0.0.0.0", port=8080, reload=False)