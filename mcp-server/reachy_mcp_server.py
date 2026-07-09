#!/usr/bin/env python3
"""Reachy Mini MCP server for OpenClaw.

A stdio MCP server that exposes Reachy Mini control as real, typed tools. It
talks to the Reachy daemon's REST API over the LAN and does ElevenLabs TTS for
speech — so it needs neither the reachy-mini SDK nor the daemon's WebRTC media
path (which is what caused the camera/OOM problems). The daemon must own the
robot (i.e. no SDK client like the old clawd bridge holding/releasing media).

Environment:
  REACHY_DAEMON_URL          default http://reachy.local:8000
  REACHY_HTTP_TIMEOUT        seconds, default 15
  REACHY_CAMERA_URL          default http://reachy.local:8089 (reachy-camera service)
  REACHY_CAPTURE_DIR         default /root/.openclaw/workspace/captures
  REACHY_ELEVENLABS_API_KEY  required for reachy_say (or ELEVENLABS_API_KEY)
  REACHY_ELEVENLABS_VOICE_ID optional (default: George)
  REACHY_ELEVENLABS_MODEL_ID optional (default: eleven_multilingual_v2)

Run as MCP server:   python reachy_mcp_server.py
Self-test (no MCP):  python reachy_mcp_server.py --selftest [--say "hello"]
"""
from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Image

DAEMON_URL = os.environ.get("REACHY_DAEMON_URL", "http://reachy.local:8000").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get("REACHY_HTTP_TIMEOUT", "15"))
# The reachy-camera service on the robot box (serves one-shot JPEG frames).
CAMERA_URL = os.environ.get("REACHY_CAMERA_URL", "http://reachy.local:8089").rstrip("/")
CAPTURE_DIR = Path(os.environ.get("REACHY_CAPTURE_DIR", "/root/.openclaw/workspace/captures"))

# Safety clamps (degrees) — mirrors the clawd bridge limits.
MAX_ROLL, MAX_PITCH, MAX_YAW = 30.0, 30.0, 45.0
MIN_DURATION = 0.3

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George (premade, free-tier friendly)

# Recorded-move libraries (the desktop app's emotions/dances), pre-loaded by the daemon.
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"

# Deliberate head gestures fired CONCURRENTLY with speech (head_pose, radians/meters).
GESTURES = {
    "tilt": {"roll": math.radians(15)},
    "look_up": {"pitch": math.radians(-15)},
    "look_down": {"pitch": math.radians(12)},
    "look_left": {"yaw": math.radians(20)},
    "look_right": {"yaw": math.radians(-20)},
    "lean_in": {"x": 0.012},
}

mcp = FastMCP("reachy-mini")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _post(path: str, json: dict | None = None, **kw) -> httpx.Response:
    return httpx.post(f"{DAEMON_URL}{path}", json=json, timeout=HTTP_TIMEOUT, **kw)


def _get(path: str) -> httpx.Response:
    return httpx.get(f"{DAEMON_URL}{path}", timeout=HTTP_TIMEOUT)


@mcp.tool()
def reachy_status() -> str:
    """Get the Reachy Mini daemon/robot status (connection, readiness, version)."""
    try:
        r = _get("/api/daemon/status")
        r.raise_for_status()
        return f"OK: {r.text}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_center_head(duration: float = 1.0) -> str:
    """Return the head and body to the neutral, level, forward-facing home pose.

    Sends a full home pose (head x/y/z=0, roll/pitch/yaw=0, body_yaw=0, antennas
    centered). This truly centers — sending head angles alone leaves body_yaw
    rotated and the head looks off-center.
    """
    body = {
        "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "body_yaw": 0.0,
        "antennas": [0.0, 0.0],
        "duration": max(MIN_DURATION, duration),
        "interpolation": "minjerk",
    }
    try:
        r = _post("/api/move/goto", json=body)
        r.raise_for_status()
        return f"Centered. {r.json()}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_move_head(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    z: float | None = None,
    duration: float = 1.0,
) -> str:
    """Move the head to an orientation.

    roll/pitch/yaw are in DEGREES (clamped to +/-30, +/-30, +/-45). `z` is an
    optional vertical offset in millimeters; omit it to keep the neutral height.
    Body rotation is left unchanged (use reachy_center_head to fully re-center).
    """
    roll = _clamp(roll, -MAX_ROLL, MAX_ROLL)
    pitch = _clamp(pitch, -MAX_PITCH, MAX_PITCH)
    yaw = _clamp(yaw, -MAX_YAW, MAX_YAW)
    head = {
        "x": 0.0,
        "y": 0.0,
        "z": (z / 1000.0) if z is not None else 0.0,  # mm -> meters
        "roll": math.radians(roll),
        "pitch": math.radians(pitch),
        "yaw": math.radians(yaw),
    }
    body = {"head_pose": head, "duration": max(MIN_DURATION, duration), "interpolation": "minjerk"}
    try:
        r = _post("/api/move/goto", json=body)
        r.raise_for_status()
        return f"Moved head (roll={roll}, pitch={pitch}, yaw={yaw} deg). {r.json()}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_move_antennas(left: float = 0.0, right: float = 0.0, duration: float = 0.5) -> str:
    """Move the antennas. left/right in DEGREES (0 = neutral/upright)."""
    body = {
        "antennas": [math.radians(left), math.radians(right)],
        "duration": max(MIN_DURATION, duration),
        "interpolation": "minjerk",
    }
    try:
        r = _post("/api/move/goto", json=body)
        r.raise_for_status()
        return f"Antennas (L={left}, R={right} deg). {r.json()}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_wake_up() -> str:
    """Play the wake-up animation (raises head to the active, upright posture)."""
    try:
        r = _post("/api/move/play/wake_up")
        r.raise_for_status()
        return f"Woke up. {r.text}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_sleep() -> str:
    """Play the go-to-sleep animation (lowers the head to rest)."""
    try:
        r = _post("/api/move/play/goto_sleep")
        r.raise_for_status()
        return f"Going to sleep. {r.text}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_stop() -> str:
    """Emergency stop: cancel any in-flight motion and stop any playing sound."""
    msgs = []
    try:
        run = _get("/api/move/running")
        uuids = []
        if run.status_code == 200:
            data = run.json()
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, str):
                    uuids.append(item)
                elif isinstance(item, dict) and item.get("uuid"):
                    uuids.append(item["uuid"])
        for u in uuids:
            _post("/api/move/stop", json={"uuid": u})
        msgs.append(f"stopped {len(uuids)} move(s)")
    except Exception as e:
        msgs.append(f"move-stop error: {e}")
    try:
        _post("/api/media/stop_sound")
        msgs.append("stopped sound")
    except Exception as e:
        msgs.append(f"sound-stop error: {e}")
    return "; ".join(msgs) or "nothing to stop"


def _pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container (stdlib, no ffmpeg)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _mp3_to_wav_ffmpeg(mp3: bytes) -> bytes:
    """Transcode mp3 -> wav via ffmpeg (fallback when PCM output isn't allowed)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ElevenLabs PCM output not available on this tier and ffmpeg is not "
            "installed on the MCP host. Install ffmpeg, or enable PCM on the account."
        )
    with tempfile.NamedTemporaryFile(suffix=".mp3") as fin, tempfile.NamedTemporaryFile(suffix=".wav") as fout:
        fin.write(mp3)
        fin.flush()
        subprocess.run(
            ["ffmpeg", "-y", "-i", fin.name, "-ar", "24000", "-ac", "1", fout.name],
            capture_output=True,
            check=True,
        )
        fout.seek(0)
        return fout.read()


def _elevenlabs_wav(text: str, voice_id: str) -> bytes:
    """Synthesize speech and return WAV bytes (the daemon's play_sound needs WAV, not mp3)."""
    api_key = os.environ.get("REACHY_ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing REACHY_ELEVENLABS_API_KEY (or ELEVENLABS_API_KEY)")
    model_id = (
        os.environ.get("REACHY_ELEVENLABS_MODEL_ID")
        or os.environ.get("ELEVENLABS_MODEL_ID")
        or "eleven_multilingual_v2"
    )
    url = f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id}

    # Prefer raw PCM (wrap to WAV with stdlib — no ffmpeg). Fall back across rates,
    # then to mp3 + ffmpeg if the account tier rejects PCM.
    for fmt, rate in (("pcm_24000", 24000), ("pcm_16000", 16000)):
        r = httpx.post(url, params={"output_format": fmt},
                       headers={**headers, "Accept": "audio/basic"},
                       json=payload, timeout=30.0)
        if r.status_code == 200:
            return _pcm_to_wav(r.content, rate)
    # Fallback: mp3 (works on free tier) -> wav via ffmpeg
    r = httpx.post(url, params={"output_format": "mp3_44100_128"},
                   headers={**headers, "Accept": "audio/mpeg"},
                   json=payload, timeout=30.0)
    r.raise_for_status()
    return _mp3_to_wav_ffmpeg(r.content)


@mcp.tool()
def reachy_say(text: str, gesture: str = "none") -> str:
    """Speak text aloud through Reachy's speaker using ElevenLabs TTS.

    The head bobs along with the audio automatically (daemon wobble). Optionally
    pass a `gesture` that is fired AT THE SAME TIME as the speech (the head moves
    while talking, not before): one of tilt, look_up, look_down, look_left,
    look_right, lean_in, or "none". Keep text short and on-topic (max 800 chars).
    """
    text = (text or "").strip()
    if not text:
        return "ERROR: empty text"
    if len(text) > 800:
        return "ERROR: text exceeds 800 chars"
    voice_id = (
        os.environ.get("REACHY_ELEVENLABS_VOICE_ID")
        or os.environ.get("ELEVENLABS_VOICE_ID")
        or DEFAULT_VOICE_ID
    )
    try:
        audio = _elevenlabs_wav(text, voice_id)
    except Exception as e:
        return f"ERROR (tts): {e}"
    try:
        files = {"file": ("reachy_say.wav", audio, "audio/wav")}
        up = _post("/api/media/sounds/upload", files=files)
        up.raise_for_status()
        fname = "reachy_say.wav"
        try:
            j = up.json()
            if isinstance(j, dict):
                fname = j.get("filename") or j.get("file") or j.get("name") or fname
            elif isinstance(j, str):
                fname = j
        except Exception:
            pass
    except Exception as e:
        return f"ERROR (upload): {e}"
    # Fire a deliberate head gesture concurrently with the audio. Both the goto and
    # play_sound are async on the daemon, so the head moves AS Reachy speaks rather
    # than before. The gesture is best-effort — never let it block speech.
    gesture = (gesture or "none").lower()
    if gesture in GESTURES:
        try:
            _post("/api/move/goto", json={
                "head_pose": GESTURES[gesture], "duration": 0.6, "interpolation": "minjerk",
            })
        except Exception:
            pass
    try:
        pl = _post("/api/media/play_sound", json={"file": fname})
        pl.raise_for_status()
        return f"Spoke: {text[:80]} (file={fname}, gesture={gesture})"
    except Exception as e:
        return f"ERROR (play, file={fname}): {e}"


@mcp.tool()
def reachy_capture_image(width: int = 1920, height: int = 1080) -> list:
    """Take a photo with Reachy's head camera and return it.

    Returns the image inline plus the path of a saved JPEG copy (under the
    workspace captures/ dir) for re-reading later. 1920x1080 is the camera's
    native default; it also supports up to 3840x2592. Point the head first
    (reachy_move_head / reachy_center_head) — the camera looks where the head
    points.
    """
    try:
        r = httpx.get(
            f"{CAMERA_URL}/capture",
            params={"width": width, "height": height},
            timeout=30.0,
        )
        r.raise_for_status()
    except Exception as e:
        return [f"ERROR (capture): {e} — is reachy-camera.service running on the robot box?"]
    jpeg = r.content
    note = f"{width}x{height}, {len(jpeg)} bytes"
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = CAPTURE_DIR / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path.write_bytes(jpeg)
        note = f"Saved to {path} ({note})"
    except Exception as e:
        note = f"Captured ({note}) but could not save a copy: {e}"
    return [Image(data=jpeg, format="jpeg"), note]


@mcp.tool()
def reachy_track_faces(enable: bool) -> str:
    """Turn face tracking on or off (Reachy's head follows the nearest face).

    While tracking is on, deliberate moves (gestures, emotions, dances) still
    play — the tracker pauses for them and resumes afterwards. Turn tracking
    off before precise manual head positioning with reachy_move_head.
    """
    try:
        if enable:
            r = httpx.post(f"{CAMERA_URL}/track/start", params={"target": "face"}, timeout=10.0)
        else:
            r = httpx.post(f"{CAMERA_URL}/track/stop", timeout=10.0)
        r.raise_for_status()
        return f"Face tracking {'enabled' if enable else 'disabled'}. {r.json()}"
    except Exception as e:
        return f"ERROR: {e} — is reachy-camera.service running on the robot box?"


@mcp.tool()
def reachy_track_object(label: str) -> str:
    """Make Reachy's head follow an object instead of faces.

    `label` is a COCO class, e.g. person, cat, dog, bird, cup, bottle, laptop,
    cell phone, book, teddy bear, sports ball. Tracks the LARGEST instance in
    view. Use reachy_track_faces(true) to go back to faces, or
    reachy_track_faces(false) to stop tracking entirely. Detection runs at
    ~5fps locally, so following is a bit slower than face mode.
    """
    try:
        r = httpx.post(f"{CAMERA_URL}/track/start", params={"target": label}, timeout=10.0)
        if r.status_code == 400:
            return f"ERROR: {r.json().get('detail', 'unknown label')}"
        r.raise_for_status()
        return f"Now tracking '{label}'. {r.json()}"
    except Exception as e:
        return f"ERROR: {e} — is reachy-camera.service running on the robot box?"


@mcp.tool()
def reachy_detect_objects() -> str:
    """Fast local object detection on the current camera view (no photo).

    Returns labels, confidences, and pixel positions of COCO-class objects.
    Cheap and structured, but limited to 80 classes and modest accuracy — for
    a rich scene description, use reachy_capture_image and look at the photo
    instead.
    """
    try:
        r = httpx.get(f"{CAMERA_URL}/detect", timeout=15.0)
        r.raise_for_status()
        return str(r.json())
    except Exception as e:
        return f"ERROR: {e} — is reachy-camera.service running on the robot box?"


@mcp.tool()
def reachy_track_status() -> str:
    """Report face-tracking state: enabled?, face currently visible?, head angles."""
    try:
        r = httpx.get(f"{CAMERA_URL}/track/status", timeout=10.0)
        r.raise_for_status()
        return str(r.json())
    except Exception as e:
        return f"ERROR: {e} — is reachy-camera.service running on the robot box?"


@mcp.tool()
def reachy_list_emotions() -> str:
    """List the emotion animations Reachy can play (e.g. curious1, laughing1, welcoming1, surprised1, proud1)."""
    try:
        r = _get(f"/api/move/recorded-move-datasets/list/{quote(EMOTIONS_DATASET, safe='')}")
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_play_emotion(name: str) -> str:
    """Play an emotion animation by name (the desktop-app expressions).

    Call reachy_list_emotions first to get valid names — do not invent names.
    Example names: curious1, laughing1, welcoming1, surprised1, proud1, sad1, shy1.
    """
    try:
        r = _post(f"/api/move/play/recorded-move-dataset/{quote(EMOTIONS_DATASET, safe='')}/{quote(name, safe='')}")
        r.raise_for_status()
        return f"Playing emotion '{name}'. {r.text}"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_list_dances() -> str:
    """List the dance animations Reachy can play."""
    try:
        r = _get(f"/api/move/recorded-move-datasets/list/{quote(DANCES_DATASET, safe='')}")
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def reachy_play_dance(name: str) -> str:
    """Play a dance animation by name. Call reachy_list_dances first — do not invent names."""
    try:
        r = _post(f"/api/move/play/recorded-move-dataset/{quote(DANCES_DATASET, safe='')}/{quote(name, safe='')}")
        r.raise_for_status()
        return f"Playing dance '{name}'. {r.text}"
    except Exception as e:
        return f"ERROR: {e}"


def _selftest() -> int:
    print("DAEMON_URL =", DAEMON_URL)
    print("status     ->", reachy_status())
    print("center     ->", reachy_center_head(duration=1.0))
    if "--say" in sys.argv:
        i = sys.argv.index("--say")
        text = sys.argv[i + 1] if i + 1 < len(sys.argv) else "Hello, I am Reachy."
        print("say        ->", reachy_say(text))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    mcp.run()
