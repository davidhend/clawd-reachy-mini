"""Main Reachy Mini interface for OpenClaw."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import subprocess
import tempfile
from enum import Enum, auto

import numpy as np

from clawd_reachy_mini.audio import AudioCapture, WakeWordDetector
from clawd_reachy_mini.config import Config
from clawd_reachy_mini.elevenlabs import (
    elevenlabs_tts_to_temp_audio_file,
    load_elevenlabs_config,
)
from clawd_reachy_mini.gateway import GatewayClient
from clawd_reachy_mini.stt import STTBackend, create_stt_backend

logger = logging.getLogger(__name__)


class InterfaceState(Enum):
    """Current state of the interface."""

    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    ERROR = auto()


class ReachyInterface:
    """
    Main interface connecting Reachy Mini to OpenClaw.

    Handles the voice conversation loop:
    1. Capture audio from Reachy Mini's microphone
    2. Transcribe speech to text
    3. Send to OpenClaw Gateway
    4. Receive response
    5. Speak response through Reachy Mini
    6. Animate robot during conversation
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = InterfaceState.IDLE

        # Components
        self._reachy = None
        self._gateway: GatewayClient | None = None
        self._stt: STTBackend | None = None
        self._audio: AudioCapture | None = None
        self._wake_detector: WakeWordDetector | None = None

        # State
        self._running = False
        self._conversation_active = False

    async def start(self) -> None:
        """Start the interface."""
        logger.info("Starting Reachy Mini interface...")

        # Connect to Reachy Mini
        await self._connect_reachy()

        # In no_media mode there is no mic/speaker stream, so skip STT + audio
        # capture entirely and run motion/tool-only (driven by gateway tool calls).
        self._media_enabled = self.config.reachy_media_backend != "no_media"

        if self._media_enabled:
            # Initialize components
            logger.info("🧠 Loading speech recognition model...")
            self._stt = create_stt_backend(self.config)
            await asyncio.to_thread(self._stt.preload)
            logger.info("✅ Speech recognition ready")

            self._audio = AudioCapture(self.config, self._reachy)

            if self.config.wake_word:
                self._wake_detector = WakeWordDetector(self.config.wake_word)
        else:
            logger.warning(
                "REACHY_MEDIA_BACKEND=no_media — voice loop disabled; "
                "running motion/tool-only (no mic, no speaker)."
            )

        # Connect to OpenClaw Gateway (unless in standalone mode)
        if not self.config.standalone_mode:
            self._gateway = GatewayClient(self.config)
            await self._gateway.connect()
            self._register_tool_dispatcher()
        else:
            logger.info("Running in standalone mode - no gateway connection")

        # Start audio capture
        if self._audio:
            await self._audio.start()

        self._running = True
        self.state = InterfaceState.IDLE

        # Wake up the robot so it can move
        if self._reachy:
            logger.info("🦞 Waking up Reachy...")
            await asyncio.to_thread(self._reachy.wake_up)
            await asyncio.sleep(0.5)

        logger.info("✨ Reachy Mini interface started")
        logger.info("=" * 50)
        if self.config.wake_word:
            logger.info(f"Say \"{self.config.wake_word}\" to activate")
        else:
            logger.info("Speak anytime - I'm always listening!")
        logger.info("=" * 50)

        # Play startup animation - snap claws!
        # Note: antenna positions are in RADIANS (0.7 rad ≈ 40 degrees)
        if self._reachy:
            try:
                self._reachy.set_target_antenna_joint_positions([0.7, -0.7])
                await asyncio.sleep(0.2)
                self._reachy.set_target_antenna_joint_positions([-0.7, 0.7])
                await asyncio.sleep(0.2)
                self._reachy.set_target_antenna_joint_positions([0.0, 0.0])
            except Exception as e:
                logger.debug(f"Startup animation failed: {e}")

    async def stop(self) -> None:
        """Stop the interface."""
        logger.info("Stopping Reachy Mini interface...")

        self._running = False

        if self._audio:
            await self._audio.stop()

        if self._gateway:
            await self._gateway.disconnect()

        if self._reachy:
            self._reachy.__exit__(None, None, None)
            self._reachy = None

        self.state = InterfaceState.IDLE
        logger.info("Reachy Mini interface stopped")

    async def run(self) -> None:
        """Main conversation loop."""
        if not self._running:
            await self.start()

        logger.info("Entering conversation loop...")

        # Start idle animation task
        idle_task = None
        if self.config.idle_animations:
            idle_task = asyncio.create_task(self._idle_animation_loop())

        try:
            if self._audio is None:
                # Motion/tool-only mode (no_media): no voice loop. Stay alive so
                # the gateway tool dispatcher keeps handling motion commands.
                logger.info("🛠️ Tool-only mode — waiting for gateway commands (no voice).")
                while self._running:
                    await asyncio.sleep(1.0)
            else:
                while self._running:
                    await self._conversation_turn()

        except asyncio.CancelledError:
            logger.info("Conversation loop cancelled")
        except Exception as e:
            logger.error(f"Error in conversation loop: {e}")
            self.state = InterfaceState.ERROR
        finally:
            if idle_task:
                idle_task.cancel()
                try:
                    await idle_task
                except asyncio.CancelledError:
                    pass

    async def _conversation_turn(self) -> None:
        """Handle one turn of conversation."""
        # Listen for speech
        self.state = InterfaceState.LISTENING
        logger.info("🎤 Listening... (speak now)")
        audio = await self._audio.capture_utterance()

        if audio is None:
            await asyncio.sleep(0.1)
            return

        # Transcribe
        self.state = InterfaceState.PROCESSING
        logger.info("🔄 Processing speech...")
        try:
            text = await asyncio.to_thread(
                self._stt.transcribe, audio, self.config.sample_rate
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return

        if not text or not text.strip():
            logger.info("(no speech detected)")
            return

        logger.info(f"📝 You said: \"{text}\"")

        # Check wake word if configured
        if self._wake_detector and not self._conversation_active:
            if not self._wake_detector.detect(text):
                logger.info(f"⏳ Waiting for wake word \"{self.config.wake_word}\"...")
                return
            logger.info("✅ Wake word detected!")

            # Quick lobster claw snap animation on activation (radians: 0.7 ≈ 40°)
            if self._reachy:
                logger.info("🦞 Playing wake-up claw snap!")
                try:
                    self._reachy.set_target_antenna_joint_positions([0.7, -0.7])
                    await asyncio.sleep(0.2)
                    self._reachy.set_target_antenna_joint_positions([-0.7, 0.7])
                    await asyncio.sleep(0.2)
                    self._reachy.set_target_antenna_joint_positions([0.0, 0.0])
                except Exception as e:
                    logger.error(f"Antenna animation failed: {e}")

            # Remove wake word from text
            text = text.lower().replace(self.config.wake_word.lower(), "").strip()
            self._conversation_active = True

        if not text:
            return

        # Get response - either from gateway or standalone echo
        if self.config.standalone_mode:
            # In standalone mode, just echo back what was heard
            response = f"I heard you say: {text}"
        else:
            # Send to OpenClaw and get response
            logger.info("🤖 Sending to AI...")

            # Start lobster claw animation while waiting
            animation_task = None
            if self._reachy:
                animation_task = asyncio.create_task(self._lobster_claw_animation())
                await asyncio.sleep(0.1)  # Let animation start

            try:
                response = await self._gateway.send_message(text)
            except Exception as e:
                logger.error(f"Gateway error: {e}")
                if self.config.play_emotions:
                    await self._play_emotion("sad")
                return
            finally:
                # Stop animation
                if animation_task:
                    animation_task.cancel()
                    try:
                        await animation_task
                    except asyncio.CancelledError:
                        pass
                    # Reset antennas to neutral (0 radians = center)
                    if self._reachy:
                        try:
                            self._reachy.set_target_antenna_joint_positions([0.0, 0.0])
                        except Exception:
                            pass

        logger.info(f"💬 Response: \"{response}\"")

        # Speak response
        self.state = InterfaceState.SPEAKING
        logger.info("🔊 Speaking response...")
        await self._speak(response)

        # Return to idle
        self.state = InterfaceState.IDLE
        logger.info("✅ Ready for next turn")

    def _register_tool_dispatcher(self) -> None:
        """Wire gateway-originated tool calls to the shared bridge."""
        if not self._gateway or not self._reachy:
            return
        try:
            from clawd_reachy_mini.actions.bridge import get_bridge
            from clawd_reachy_mini.actions.tools import dispatch

            bridge = get_bridge()
            bridge.attach_existing(self._reachy)
            bridge.set_speak_handler(self._make_tool_speak_handler())
            self._gateway.register_tool_dispatcher(dispatch)
            logger.info("Tool dispatcher registered (gateway → bridge)")
        except Exception as e:
            logger.warning(f"Could not register tool dispatcher: {e}")

    def _make_tool_speak_handler(self):
        """Build a blocking, thread-safe speak callable for the bridge.

        Tool dispatch runs in a worker thread (gateway uses asyncio.to_thread),
        so `reachy_say` must hop back onto this event loop to drive playback. We
        block on the result and let exceptions propagate so the tool reports a
        real status instead of a silent success.
        """
        loop = asyncio.get_running_loop()

        def handler(text: str) -> None:
            future = asyncio.run_coroutine_threadsafe(self._speak_for_tool(text), loop)
            future.result()

        return handler

    async def _connect_reachy(self) -> None:
        """Connect to Reachy Mini robot, retrying while the daemon comes up.

        systemd only guarantees the daemon *process* has launched before the
        bridge starts, not that it's *ready* (motors configured + uvicorn
        listening takes a few seconds). A single attempt loses that race and
        leaves the bridge running blind (no media, mic loop spinning). So retry
        with backoff for ~a minute before giving up.
        """
        try:
            from reachy_mini import ReachyMini
        except ImportError:
            logger.warning("reachy-mini not installed, running in simulation mode")
            self._reachy = None
            return

        kwargs = {}
        if self.config.reachy_connection_mode != "auto":
            kwargs["connection_mode"] = self.config.reachy_connection_mode
        if self.config.reachy_media_backend != "default":
            kwargs["media_backend"] = self.config.reachy_media_backend

        max_attempts = 20
        retry_delay = 3.0
        for attempt in range(1, max_attempts + 1):
            try:
                reachy = ReachyMini(**kwargs)
                reachy.__enter__()
                self._reachy = reachy
                logger.info(f"Connected to Reachy Mini (attempt {attempt})")
                return
            except Exception as e:
                self._reachy = None
                if attempt < max_attempts:
                    logger.warning(
                        f"Reachy Mini not ready (attempt {attempt}/{max_attempts}): {e} "
                        f"— retrying in {retry_delay:.0f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to Reachy Mini after {max_attempts} attempts: {e}. "
                        "Running without robot (no motion/audio)."
                    )

    async def _speak(self, text: str) -> None:
        """Speak text through Reachy Mini for conversational replies.

        Never raises: TTS/playback failures are logged so the conversation loop
        keeps running. Tool-triggered speech uses `_speak_for_tool`, which does
        raise so the `reachy_say` tool can report a real error status.
        """
        if self._audio:
            self._audio.mute()
        try:
            await self._tts_speak(text)
        except ValueError as e:
            logger.error(f"ElevenLabs TTS configuration error: {e}")
            logger.info(
                "Set REACHY_ELEVENLABS_API_KEY or ELEVENLABS_API_KEY to enable speech."
            )
            logger.info(f"[TTS] {text}")
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            logger.info(f"[TTS] {text}")
        finally:
            # Unmute after a guard window so room echo doesn't get captured.
            if self._audio:
                await self._audio.unmute()

    async def _speak_for_tool(self, text: str) -> None:
        """Speak on behalf of the `reachy_say` tool. Raises on failure.

        Mutes the mic for the duration like the conversational path, but lets
        exceptions propagate so the bridge can surface them to the agent instead
        of silently reporting success.
        """
        if self._audio:
            self._audio.mute()
        try:
            await self._tts_speak(text)
        finally:
            if self._audio:
                await self._audio.unmute()

    async def _tts_speak(self, text: str) -> None:
        """Synthesize `text` with ElevenLabs and play it on Reachy. Raises on failure."""

        # Clean up markdown formatting for speech
        clean_text = text.replace("**", "").replace("*", "").replace("`", "")
        temp_audio_path: str | None = None
        temp_wav_path: str | None = None

        try:
            tts_cfg = load_elevenlabs_config()
            logger.info("☁️ Generating speech with ElevenLabs...")
            temp_audio_path = await elevenlabs_tts_to_temp_audio_file(
                text=clean_text,
                config=tts_cfg,
                voice_settings={"use_speaker_boost": True},
            )

            # Play through Reachy Mini if available
            if self._reachy and hasattr(self._reachy, "media"):
                # Convert generated audio to 16k mono wav for Reachy
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                    temp_wav_path = wf.name
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_audio_path,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        temp_wav_path,
                    ],
                    capture_output=True,
                    check=True,
                )

                # Play the audio
                import wave
                with wave.open(temp_wav_path, "rb") as wf:
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    audio_float = audio_data.astype(np.float32) / 32768.0

                self._reachy.media.start_playing()
                try:
                    # Push audio in chunks with proper timing
                    sample_rate = 16000
                    chunk_size = 1600  # 100ms chunks
                    chunk_duration = chunk_size / sample_rate

                    # Start speaking animation in background
                    speak_animation_task = asyncio.create_task(
                        self._speak_animation(len(audio_float) / sample_rate)
                    )

                    # Push audio chunks
                    total_chunks = len(audio_float) // chunk_size
                    logger.info(f"🦞 Speaking with {total_chunks} chunks...")

                    for i in range(0, len(audio_float), chunk_size):
                        chunk = audio_float[i:i+chunk_size]
                        self._reachy.media.push_audio_sample(chunk)
                        await asyncio.sleep(chunk_duration * 0.9)

                    # Stop animation and reset
                    speak_animation_task.cancel()
                    try:
                        await speak_animation_task
                    except asyncio.CancelledError:
                        pass
                    logger.info("🗣️ Speech done, resetting head")
                    self._reachy.set_target_antenna_joint_positions([0.0, 0.0])
                    await asyncio.sleep(0.5)
                finally:
                    self._reachy.media.stop_playing()
            else:
                # Fallback: play locally on Mac
                subprocess.run(["afplay", temp_audio_path], capture_output=True)
        finally:
            for path in (temp_wav_path, temp_audio_path):
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass

    async def _play_emotion(self, emotion: str) -> None:
        """Play emotion animation on Reachy Mini."""
        if self._reachy and hasattr(self._reachy, "play_emotion"):
            try:
                await asyncio.to_thread(self._reachy.play_emotion, emotion)
            except Exception as e:
                logger.debug(f"Emotion playback failed: {e}")

    async def _speak_animation(self, duration: float) -> None:
        """Animate head bobbing while speaking to simulate talking."""
        if not self._reachy:
            return

        from reachy_mini.utils import create_head_pose

        logger.info(f"🗣️ Starting head bob animation for {duration:.1f}s")
        bob_state = 0
        try:
            while True:
                # Small head nods to simulate talking
                if bob_state == 0:
                    pose = create_head_pose(pitch=3, degrees=True)
                    bob_state = 1
                else:
                    pose = create_head_pose(pitch=-3, degrees=True)
                    bob_state = 0
                self._reachy.set_target_head_pose(pose)
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            # Reset head to neutral
            try:
                self._reachy.set_target_head_pose(create_head_pose(pitch=0, degrees=True))
            except Exception:
                pass
            logger.info("🗣️ Head bob animation stopped")
            raise

    async def _lobster_claw_animation(self) -> None:
        """Animate antennas like lobster claws while thinking (positions in radians: 0.7 ≈ 40°)."""
        if not self._reachy:
            logger.warning("No Reachy for animation")
            return

        logger.info("🦞 Starting thinking claw animation...")
        try:
            while True:
                # Claws open
                self._reachy.set_target_antenna_joint_positions([0.7, -0.7])
                await asyncio.sleep(0.35)
                # Claws close
                self._reachy.set_target_antenna_joint_positions([-0.7, 0.7])
                await asyncio.sleep(0.35)
        except asyncio.CancelledError:
            logger.info("🦞 Claw animation stopped")
            raise

    async def _idle_animation_loop(self) -> None:
        """Play subtle idle animations when not in conversation."""
        idle_movements = [
            {"roll": 5, "pitch": 0},
            {"roll": -5, "pitch": 0},
            {"roll": 0, "pitch": 5},
            {"roll": 0, "pitch": -5},
        ]

        while self._running:
            try:
                if self.state == InterfaceState.IDLE:
                    # Small random head movement
                    if self._reachy and random.random() < 0.3:
                        movement = random.choice(idle_movements)
                        from reachy_mini.utils import create_head_pose

                        await asyncio.to_thread(
                            self._reachy.goto_target,
                            head=create_head_pose(**movement, degrees=True),
                            duration=2.0,
                        )

                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Idle animation error: {e}")
                await asyncio.sleep(5.0)
