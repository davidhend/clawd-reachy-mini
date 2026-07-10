"""Standalone voice loop for Reachy Mini — no SDK, no WebRTC.

Implements: wake word -> mic capture -> STT -> OpenClaw gateway -> spoken reply.

Unlike the original bridge (interface.py), this service never imports the
reachy_mini SDK. It takes the microphone by asking the daemon to release its
media devices (POST /api/media/release — the daemon holds the mic open but
never reads PCM from it), then captures directly from ALSA via an `arecord`
subprocess. Speech replies go out through the daemon's REST API (upload +
play_sound), which keeps working while media is released because playback
pipelines are created on demand.
"""

from __future__ import annotations

import asyncio
import difflib
import io
import logging
import os
import re
import signal
import sys
import time
import wave
from collections import deque

import httpx
import numpy as np

from clawd_reachy_mini.config import Config, load_config
from clawd_reachy_mini.gateway import GatewayClient, ReplyPending
from clawd_reachy_mini.stt import create_stt_backend

logger = logging.getLogger(__name__)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George (premade, free tier)

CHUNK_FRAMES = 1024  # 64ms at 16kHz
SAMPLE_WIDTH = 2  # S16_LE
PREROLL_CHUNKS = 6  # ~0.4s kept from before speech onset

# Prepended to every utterance sent to the gateway. Must match the tag that
# IDENTITY.md on the OpenClaw box keys the Gizmo voice persona on.
VOICE_TAG = os.environ.get("VOICE_MESSAGE_TAG", "[Gizmo voice]")

# Thinking motion while waiting on the agent's reply (THINKING_MOTION=false to
# disable): alternate slow up-left/up-right head glances until the reply
# arrives, then recenter. Set THINKING_EMOTION to a clip name (e.g.
# "thoughtful1") to also play that emotion once before the glances — the user
# tried it and found it too busy, so the default is glances only.
THINKING_MOTION = os.environ.get("THINKING_MOTION", "true").strip().lower() not in {"false", "0", "no", "off"}
THINKING_EMOTION = os.environ.get("THINKING_EMOTION", "").strip()
EMOTIONS_DATASET = "pollen-robotics%2Freachy-mini-emotions-library"

# Spoken when the agent is still working after the gateway reply timeout
# (OPENCLAW_REPLY_TIMEOUT, default 120s). The real answer is announced by
# _late_reply_announcer whenever the run eventually finishes.
STILL_WORKING_LINE = os.environ.get(
    "STILL_WORKING_LINE", "Still working on that. I'll let you know when it's done."
)

# IDENTITY.md tells the agent to answer exactly NO_REPLY to in-session speech
# that wasn't addressed to Gizmo (ambient meeting talk forwarded during the
# follow-up window). Lenient on separators/trailing period, nothing else.
_NO_REPLY_RE = re.compile(r"no[_\s-]?reply\.?", re.IGNORECASE)


def is_no_reply(text: str) -> bool:
    """True if the agent declined to answer ambient speech."""
    return bool(_NO_REPLY_RE.fullmatch(text.strip()))

# Per-unit calibration: some units' heads lean at commanded roll 0 (e.g. one
# test unit needed -0.12 rad to read as level). Set REACHY_ROLL_TRIM to suit.
ROLL_TRIM = float(os.environ.get("REACHY_ROLL_TRIM", "0.0"))
GLANCE_PITCH = -0.22  # radians; negative = up
GLANCE_YAW = 0.35  # radians; alternates +/- (left/right)

# Whisper transcribes the robot's name inconsistently, so wake matching is
# punctuation-insensitive and phonetic on the name, not an exact substring test.
# "Gizmo" is a real word, so Whisper usually spells it correctly — the alias set
# only needs the few plausible variants ("gismo", a trailing "s", etc.).
# Greetings come in two tiers: strong ones accept any name-like candidate;
# weak ones (Whisper's renderings of a mumbled/clipped "hey") only accept an
# exact curated alias, so ambient speech cannot wake.
WAKE_GREETINGS = {"hey", "hay", "hei", "hi", "hiya", "okay", "ok", "yo"}
WAKE_WEAK_GREETINGS = {"a", "ay", "aye", "eh", "uh", "um", "oh", "he", "hee"}
WAKE_NAME_ALIASES = {
    "gizmo", "gizmos", "gismo", "gismos", "gizmoe", "gizma", "gizzmo",
    "gizmoh", "gizemo",
}
WAKE_NAME_FUZZ_THRESHOLD = 0.7

_SOUNDEX_CODES = str.maketrans(
    "bfpvcgjkqsxzdtlmnr",
    "111122222222334556",
)


def _soundex(word: str) -> str:
    """Plain Soundex. reggie/ricci/richie/reachy all code to R200."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    first = word[0]
    digits = word.translate(_SOUNDEX_CODES)
    code = first.upper()
    prev = digits[0] if digits[0].isdigit() else ""
    for ch, d in zip(word[1:], digits[1:]):
        if d.isdigit():
            if d != prev:
                code += d
            prev = d
        elif ch not in "hw":
            prev = ""
    return (code + "000")[:4]


def match_wake_phrase(text: str, name: str = "gizmo") -> str | None:
    """Find a wake phrase ("hey <name>"-ish) in `text`, tolerating Whisper's
    punctuation and creative spellings of the name.

    Returns the remainder of the utterance after the wake phrase (possibly
    empty), or None if no wake phrase is present.
    """
    lowered = text.lower()
    # Name-first address ("Gizmo, install the updates" / bare "Gizmo") —
    # exact alias at utterance start ONLY, no soundex/fuzz, so ambient
    # mid-sentence mentions of the name don't wake.
    first = re.match(r"([a-z']+)[\s,.!?-]*", lowered)
    if first and (first.group(1) in WAKE_NAME_ALIASES or first.group(1) == name):
        return lowered[first.end():].strip(" ,.!?")
    # Lookahead keeps matches single-token so every adjacent word pair is
    # tested (a plain two-token pattern would consume "um, hey" and never
    # examine the "hey reachy" pair).
    for m in re.finditer(r"\b([a-z']+)[\s,.!?-]+(?=([a-z']+))", lowered):
        greeting, candidate = m.group(1), m.group(2)
        exact_alias = candidate in WAKE_NAME_ALIASES or candidate == name
        if greeting in WAKE_GREETINGS:
            matched = (
                exact_alias
                or _soundex(candidate) == _soundex(name)
                or difflib.SequenceMatcher(None, candidate, name).ratio() >= WAKE_NAME_FUZZ_THRESHOLD
            )
        elif greeting in WAKE_WEAK_GREETINGS and m.start(1) == 0:
            # Only at utterance start — "A-Reach, ..." is a clipped "hey",
            # but "that is a reach" mid-sentence is ambient speech.
            matched = exact_alias
        else:
            matched = False
        if matched:
            return lowered[m.end(2):].strip(" ,.!?")
    return None


class DaemonClient:
    """Thin async client for the reachy-mini-daemon REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=15.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def release_media(self) -> None:
        r = await self._http.post("/api/media/release")
        r.raise_for_status()
        logger.info("📤 Daemon released media devices (mic is ours)")

    async def acquire_media(self) -> None:
        r = await self._http.post("/api/media/acquire")
        r.raise_for_status()
        logger.info("📥 Daemon re-acquired media devices")

    async def enable_wobbling(self) -> None:
        """Turn on the daemon's audio-reactive head bob. Best-effort.

        Off by default in the daemon and resets on daemon restart — the media
        watchdog re-enables it alongside re-releasing media. The bob offsets
        compose with the current target pose before IK, so it works at idle,
        during face tracking, and on top of moves alike.
        """
        try:
            r = await self._http.post("/api/media/wobbling/enable")
            r.raise_for_status()
            logger.info("🎵 Daemon head wobbling enabled (bob while audio plays)")
        except Exception as e:
            logger.warning(f"enable_wobbling failed: {e}")

    async def media_released(self) -> bool | None:
        """True/False from /api/media/status, None if the daemon is unreachable."""
        try:
            r = await self._http.get("/api/media/status")
            r.raise_for_status()
            return bool(r.json().get("released"))
        except Exception as e:
            logger.warning(f"media_status failed: {e}")
            return None

    async def play_wav(self, wav_bytes: bytes, filename: str = "voice_reply.wav") -> float:
        """Upload WAV bytes and play them. Returns the clip duration in seconds."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            duration = w.getnframes() / w.getframerate()

        files = {"file": (filename, wav_bytes, "audio/wav")}
        up = await self._http.post("/api/media/sounds/upload", files=files)
        up.raise_for_status()
        try:
            j = up.json()
            if isinstance(j, dict):
                filename = j.get("filename") or j.get("file") or j.get("name") or filename
        except Exception:
            pass

        pl = await self._http.post("/api/media/play_sound", json={"file": filename})
        pl.raise_for_status()
        return duration

    async def antenna_ack(self) -> None:
        """Quick antenna snap acknowledging the wake word. Best-effort."""
        try:
            for left, right in ((0.7, -0.7), (0.0, 0.0)):
                await self._http.post(
                    "/api/move/goto",
                    json={"antennas": [left, right], "duration": 0.25, "interpolation": "minjerk"},
                )
                await asyncio.sleep(0.3)
        except Exception as e:
            logger.debug(f"antenna ack failed: {e}")

    async def _moves_running(self) -> bool:
        try:
            r = await self._http.get("/api/move/running")
            data = r.json()
            items = data if isinstance(data, list) else [data]
            return any(items)
        except Exception:
            return False

    async def play_emotion(self, name: str) -> bool:
        """Fire-and-forget a recorded emotion clip. Returns False on failure."""
        try:
            r = await self._http.post(
                f"/api/move/play/recorded-move-dataset/{EMOTIONS_DATASET}/{name}"
            )
            r.raise_for_status()
            return True
        except Exception as e:
            logger.debug(f"play_emotion {name} failed: {e}")
            return False

    async def _wait_moves_done(self, stop: asyncio.Event, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not stop.is_set() and time.monotonic() < deadline:
            if not await self._moves_running():
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

    async def head_thinking(self, stop: asyncio.Event) -> None:
        """Thinking motion while the agent is working. Best-effort.

        Plays the THINKING_EMOTION clip once, waits for it to finish, then
        alternates slow up-left/up-right head glances. Defers to any other
        daemon move (agent gestures, emotions, sleep/wake) by skipping beats
        while /api/move/running reports activity; each glance finishes well
        inside the beat period so our own move never blocks the next check.
        Always tries to end recentered. The caller must pause the face tracker
        first — it would grab the head back between glances.
        """
        try:
            if THINKING_EMOTION and await self.play_emotion(THINKING_EMOTION):
                await self._wait_moves_done(stop, timeout_s=15.0)
            beat = 0
            while not stop.is_set():
                if not await self._moves_running():
                    yaw = GLANCE_YAW if beat % 2 == 0 else -GLANCE_YAW
                    await self._http.post(
                        "/api/move/goto",
                        json={
                            "head_pose": {"roll": ROLL_TRIM, "pitch": GLANCE_PITCH, "yaw": yaw},
                            "duration": 1.2,
                            "interpolation": "minjerk",
                        },
                    )
                    beat += 1
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2.5)
                except asyncio.TimeoutError:
                    pass
        except Exception as e:
            logger.debug(f"thinking motion failed: {e}")
        finally:
            try:
                await self._http.post(
                    "/api/move/goto",
                    json={
                        "head_pose": {"roll": ROLL_TRIM, "pitch": 0.0, "yaw": 0.0},
                        "body_yaw": 0.0,
                        "duration": 0.8,
                        "interpolation": "minjerk",
                    },
                )
            except Exception:
                pass


class TrackerClient:
    """Pause/resume the reachy-camera head tracker (port 8089). Best-effort."""

    def __init__(self, base_url: str):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=5.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def pause(self) -> str | None:
        """Stop tracking if it was on; returns the prior target for resume()."""
        try:
            r = await self._http.get("/track/status")
            st = r.json()
            if not st.get("tracking"):
                return None
            await self._http.post("/track/stop")
            return st.get("target") or "face"
        except Exception as e:
            logger.debug(f"tracker pause failed: {e}")
            return None

    async def resume(self, target: str | None) -> None:
        if not target:
            return
        try:
            await self._http.post("/track/start", params={"target": target})
        except Exception as e:
            logger.debug(f"tracker resume failed: {e}")


async def elevenlabs_wav(text: str, timeout_s: float = 30.0) -> bytes:
    """Synthesize speech, returning WAV bytes (PCM output wrapped via stdlib, no ffmpeg)."""
    api_key = os.environ.get("REACHY_ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing REACHY_ELEVENLABS_API_KEY (or ELEVENLABS_API_KEY)")
    voice_id = (
        os.environ.get("REACHY_ELEVENLABS_VOICE_ID")
        or os.environ.get("ELEVENLABS_VOICE_ID")
        or DEFAULT_VOICE_ID
    )
    model_id = (
        os.environ.get("REACHY_ELEVENLABS_MODEL_ID")
        or os.environ.get("ELEVENLABS_MODEL_ID")
        or "eleven_multilingual_v2"
    )
    url = f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id}

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for fmt, rate in (("pcm_24000", 24000), ("pcm_16000", 16000)):
            r = await client.post(
                url,
                params={"output_format": fmt},
                headers={**headers, "Accept": "audio/basic"},
                json=payload,
            )
            if r.status_code == 200:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(r.content)
                return buf.getvalue()
        raise RuntimeError(
            f"ElevenLabs PCM output rejected (last status {r.status_code}): {r.text[:200]}"
        )


class ArecordCapture:
    """Continuous mic capture through an `arecord` subprocess.

    Keeping one long-lived arecord avoids per-utterance open latency and pops.
    While muted (TTS playing) the stream keeps draining but samples are
    discarded, so the robot does not transcribe its own voice.
    """

    def __init__(self, device: str, sample_rate: int):
        self.device = device
        self.sample_rate = sample_rate
        self._proc: asyncio.subprocess.Process | None = None
        self.muted = False

    async def start(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            "arecord",
            "-q",
            "-D", self.device,
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", "1",
            "-t", "raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info(f"🎙️ arecord started on {self.device} ({self.sample_rate}Hz mono)")

    async def stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._proc.kill()
        self._proc = None

    async def read_chunk(self) -> np.ndarray | None:
        """Read one chunk as float32 in [-1, 1]. None if the stream died."""
        if not self._proc or self._proc.returncode is not None:
            stderr = b""
            if self._proc and self._proc.stderr:
                stderr = await self._proc.stderr.read()
            logger.warning(f"arecord not running ({stderr.decode(errors='replace').strip()}); restarting")
            await self.stop()
            await asyncio.sleep(1.0)
            await self.start()
            if not self._proc or self._proc.returncode is not None:
                return None
        try:
            raw = await self._proc.stdout.readexactly(CHUNK_FRAMES * SAMPLE_WIDTH)
        except (asyncio.IncompleteReadError, AttributeError):
            return None
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class VoiceService:
    """The conversation loop: wake word -> STT -> OpenClaw -> spoken reply."""

    def __init__(self, config: Config):
        self.config = config
        self.daemon = DaemonClient(os.environ.get("REACHY_DAEMON_URL", "http://127.0.0.1:8000"))
        self.tracker = TrackerClient(os.environ.get("REACHY_CAMERA_URL", "http://127.0.0.1:8089"))
        self.capture = ArecordCapture(
            device=os.environ.get("REACHY_MIC_DEVICE", "plughw:Audio,0"),
            sample_rate=config.sample_rate,
        )
        self.stt = create_stt_backend(config)
        self.gateway = GatewayClient(config)
        self.wake_word = (config.wake_word or "hey gizmo").strip().lower()
        self.wake_session_s = float(os.environ.get("WAKE_SESSION_SECONDS", "45"))
        self._last_exchange = 0.0
        self._running = False
        # Replies from runs that outlived the gateway wait (ReplyPending);
        # drained by _late_reply_announcer.
        self._late_replies: asyncio.Queue[str] = asyncio.Queue()
        self.gateway.register_late_result_callback(self._late_replies.put)
        # Serializes turn replies vs late announcements (both TTS + play).
        self._speak_lock = asyncio.Lock()

    async def start(self) -> None:
        logger.info("🧠 Loading speech recognition model...")
        await asyncio.to_thread(self.stt.preload)
        logger.info("✅ Speech recognition ready")

        await self.daemon.release_media()
        await self.daemon.enable_wobbling()
        await self.capture.start()
        await self.gateway.connect()

        self._running = True
        logger.info("=" * 50)
        logger.info(f'Say "{self.wake_word}" to talk to Gizmo')
        logger.info("=" * 50)

    async def stop(self) -> None:
        self._running = False
        await self.capture.stop()
        try:
            await self.daemon.acquire_media()
        except Exception as e:
            logger.warning(f"Could not hand media back to the daemon: {e}")
        await self.gateway.disconnect()
        await self.daemon.close()
        await self.tracker.close()

    async def run(self) -> None:
        await self.start()
        watchdog = asyncio.create_task(self._media_watchdog())
        announcer = asyncio.create_task(self._late_reply_announcer())
        try:
            while self._running:
                try:
                    await self._turn()
                except Exception:
                    logger.exception("Error in conversation turn")
                    await asyncio.sleep(1.0)
        finally:
            for task in (watchdog, announcer):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _late_reply_announcer(self) -> None:
        """Speak agent replies that finished after their turn stopped waiting."""
        while True:
            text = await self._late_replies.get()
            if is_no_reply(text):
                logger.info("🤐 Late reply was NO_REPLY — staying quiet")
                continue
            logger.info(f'💬 Late reply: "{text[:200]}"')
            asyncio.create_task(self.daemon.antenna_ack())
            await self._speak(text)
            # Open the follow-up window so the user can respond without the
            # wake word, same as after an ordinary exchange.
            self._last_exchange = time.monotonic()

    async def _media_watchdog(self) -> None:
        """Re-release media if the daemon restarts and takes the mic back."""
        while True:
            await asyncio.sleep(10.0)
            released = await self.daemon.media_released()
            if released is False:
                logger.warning("Daemon re-acquired media (restart?) — releasing again")
                try:
                    await self.daemon.release_media()
                except Exception as e:
                    logger.error(f"Re-release failed: {e}")
                # A restarted daemon also forgets the wobbling toggle.
                await self.daemon.enable_wobbling()

    async def _turn(self) -> None:
        audio = await self._capture_utterance()
        if audio is None:
            return

        text = await asyncio.to_thread(self.stt.transcribe, audio, self.config.sample_rate)
        text = (text or "").strip()
        if not text:
            return
        logger.info(f'📝 Heard: "{text}"')

        in_session = (time.monotonic() - self._last_exchange) < self.wake_session_s
        if not in_session:
            remainder = match_wake_phrase(text, name=self.wake_word.split()[-1])
            if remainder is None:
                logger.info(f'⏳ No wake word ("{self.wake_word}") — ignoring')
                return
            logger.info("✅ Wake word detected!")
            asyncio.create_task(self.daemon.antenna_ack())
            text = remainder
            if not text:
                await self._speak("Yes?")
                self._last_exchange = time.monotonic()
                return

        logger.info("🤖 Sending to OpenClaw...")
        if not self.gateway.is_connected:
            logger.info("Gateway disconnected — reconnecting")
            await self.gateway.connect()
        # The gateway session looks like plain webchat to the agent; this tag is
        # what lets it know the message was spoken to the robot (IDENTITY.md on
        # the OpenClaw box tells it to answer as Gizmo, briefly, speech-only).
        stop_thinking = asyncio.Event()
        thinking: asyncio.Task | None = None
        prior_target: str | None = None
        if THINKING_MOTION:
            # Face tracking would grab the head back between glances.
            prior_target = await self.tracker.pause()
            thinking = asyncio.create_task(self.daemon.head_thinking(stop_thinking))
        try:
            reply = await self.gateway.send_message(f"{VOICE_TAG} {text}")
        except ReplyPending:
            # Agent is still working; _late_reply_announcer speaks the result
            # when the run finishes.
            reply = STILL_WORKING_LINE
        finally:
            stop_thinking.set()
            if thinking is not None:
                try:
                    await asyncio.wait_for(thinking, timeout=5.0)
                except Exception:
                    pass
            await self.tracker.resume(prior_target)
        if is_no_reply(reply):
            # Ambient room talk the agent declined — stay quiet and do NOT
            # extend the follow-up window, so meeting chatter dies out
            # instead of chaining exchanges forever.
            logger.info("🤐 Agent declined ambient speech (NO_REPLY) — staying quiet")
            return
        self._last_exchange = time.monotonic()
        logger.info(f'💬 Reply: "{reply[:200]}"')

        if reply.strip():
            await self._speak(reply)
            self._last_exchange = time.monotonic()

    async def _capture_utterance(self) -> np.ndarray | None:
        """Energy-VAD utterance capture with a pre-roll ring buffer."""
        preroll: deque[np.ndarray] = deque(maxlen=PREROLL_CHUNKS)
        frames: list[np.ndarray] = []
        silence_chunks = 0
        chunks_per_s = self.config.sample_rate / CHUNK_FRAMES
        max_silence = int(self.config.silence_duration * chunks_per_s)
        max_chunks = int(self.config.max_recording_duration * chunks_per_s)
        speaking = False

        while self._running:
            chunk = await self.capture.read_chunk()
            if chunk is None:
                await asyncio.sleep(0.1)
                return None
            if self.capture.muted:
                preroll.clear()
                frames.clear()
                speaking = False
                continue

            energy = float(np.abs(chunk).mean())
            if not speaking:
                preroll.append(chunk)
                if energy > self.config.silence_threshold:
                    logger.info("🗣️ Speech detected")
                    speaking = True
                    frames.extend(preroll)
                    preroll.clear()
            else:
                frames.append(chunk)
                silence_chunks = silence_chunks + 1 if energy <= self.config.silence_threshold else 0
                if silence_chunks >= max_silence or len(frames) >= max_chunks:
                    break

        if not frames:
            return None
        audio = np.concatenate(frames)
        logger.info(f"📼 Captured {len(audio) / self.config.sample_rate:.2f}s of audio")
        return audio

    async def _speak(self, text: str) -> None:
        """TTS the reply and play it via the daemon, half-duplex (mic discarded)."""
        clean = text.replace("**", "").replace("*", "").replace("`", "")
        # Drop emoji and other unspeakable symbols; an emoji-only reply (e.g.
        # "👍") would otherwise synthesize a 0.1s clip of silence.
        clean = "".join(
            ch for ch in clean if ch.isalnum() or ch.isspace() or ch in ".,!?;:()'\"-%$@/&"
        )
        if not clean.strip():
            logger.info("Reply has no speakable text (emoji only?) — skipping TTS")
            return
        async with self._speak_lock:
            self.capture.muted = True
            try:
                logger.info("☁️ Generating speech with ElevenLabs...")
                wav = await elevenlabs_wav(clean)
                # No tracker pause here: the daemon's speech bob (see
                # enable_wobbling) composes its offsets on top of the target pose,
                # so it animates on top of face tracking rather than fighting it.
                duration = await self.daemon.play_wav(wav)
                logger.info(f"🔊 Playing reply ({duration:.1f}s)")
                await asyncio.sleep(duration + self.config.post_speech_guard)
            except Exception as e:
                logger.error(f"TTS/playback failed: {e}")
            finally:
                self.capture.muted = False


async def async_main(config: Config) -> int:
    service = VoiceService(config)
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    run_task = asyncio.create_task(service.run())
    shutdown_task = asyncio.create_task(shutdown.wait())
    done, pending = await asyncio.wait(
        [run_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await service.stop()

    if run_task in done and run_task.exception():
        logger.error(f"Voice service crashed: {run_task.exception()}")
        return 1
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG if "-v" in sys.argv else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()
    # The systemd unit shares bridge.env with the old bridge, which names the
    # tunnel's local end LOCAL_PORT rather than OPENCLAW_PORT.
    if "OPENCLAW_PORT" not in os.environ and os.environ.get("LOCAL_PORT"):
        config.gateway_port = int(os.environ["LOCAL_PORT"])
    if not config.wake_word:
        config.wake_word = "hey gizmo"

    logger.info(f"Gateway: {config.gateway_url}")
    logger.info(f"STT: {config.stt_backend} ({config.whisper_model})")
    sys.exit(asyncio.run(async_main(config)))


if __name__ == "__main__":
    main()
