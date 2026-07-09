"""Bridge between OpenClaw tool calls and the Reachy Mini SDK."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from clawd_reachy_mini.actions.config import get_config, ReachyConfig

if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini.motion.recorded_move import RecordedMoves

logger = logging.getLogger(__name__)

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"


class ReachyBridge:
    """Manages connection and communication with Reachy Mini robot."""

    def __init__(self, config: ReachyConfig | None = None):
        self.config = config or get_config()
        self._mini: ReachyMini | None = None
        self._connected = False
        self._emotions: RecordedMoves | None = None
        self._dances: RecordedMoves | None = None
        # Serialize motion commands so two tool calls cannot collide on the wire.
        # The daemon also locks, but failing fast in-process gives a cleaner error.
        self._motion_lock = threading.Lock()
        self._last_command_ts: dict[str, float] = {}
        # Speech is delegated to the interface's ElevenLabs path (which also owns
        # the mic so it can half-duplex). Set via set_speak_handler(); the callable
        # blocks until playback finishes and raises if synthesis/playback fails.
        self._speak_handler: Callable[[str], None] | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._mini is not None

    def attach_existing(self, mini: "ReachyMini") -> None:
        """Attach to an externally-owned ReachyMini instance.

        Used when the bridge process already holds the robot connection (e.g.
        the OpenClaw bridge service) and we want tool calls to share it instead
        of opening a second connection.
        """
        self._mini = mini
        self._connected = True

    def detach(self) -> None:
        """Release an externally-owned ReachyMini instance without disconnecting."""
        self._mini = None
        self._connected = False
        self._emotions = None
        self._dances = None
        self._speak_handler = None

    def set_speak_handler(self, handler: Callable[[str], None] | None) -> None:
        """Inject the speech callback used by `say()`.

        The handler is supplied by the interface and routes text through the
        ElevenLabs TTS + Reachy media playback path. It must block until playback
        completes and raise on failure so `say()` can report a real status.
        """
        self._speak_handler = handler

    def _rate_limit(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_command_ts.get(key, 0.0)
        if now - last < self.config.min_command_interval:
            return False
        self._last_command_ts[key] = now
        return True

    def connect(self, connection_mode: str | None = None) -> dict:
        if self.is_connected:
            return {"status": "already_connected"}

        try:
            from reachy_mini import ReachyMini

            mode = connection_mode or self.config.connection_mode
            kwargs = {}
            if mode != "auto":
                kwargs["connection_mode"] = mode

            self._mini = ReachyMini(**kwargs)
            self._mini.__enter__()
            self._connected = True

            logger.info("Connected to Reachy Mini")
            return {"status": "connected", "mode": mode}

        except ImportError:
            return {"status": "error", "message": "reachy-mini package not installed"}
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return {"status": "error", "message": str(e)}

    def disconnect(self) -> dict:
        if not self.is_connected:
            return {"status": "not_connected"}

        try:
            self._mini.__exit__(None, None, None)
            self._mini = None
            self._connected = False
            self._emotions = None
            self._dances = None

            logger.info("Disconnected from Reachy Mini")
            return {"status": "disconnected"}

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            return {"status": "error", "message": str(e)}

    def stop(self) -> dict:
        """Emergency stop: cancel any in-flight motion."""
        if not self.is_connected:
            return {"status": "not_connected"}
        try:
            self._mini.cancel_move()
            logger.warning("Motion cancelled (emergency stop)")
            return {"status": "stopped"}
        except Exception as e:
            logger.error(f"Failed to cancel motion: {e}")
            return {"status": "error", "message": str(e)}

    def move_head(
        self,
        z: float | None = None,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        duration: float | None = None,
    ) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        if not self._rate_limit("move_head"):
            return {"status": "rate_limited"}

        try:
            from reachy_mini.utils import create_head_pose

            roll = max(-self.config.max_roll, min(self.config.max_roll, roll))
            pitch = max(-self.config.max_pitch, min(self.config.max_pitch, pitch))
            yaw = max(-self.config.max_yaw, min(self.config.max_yaw, yaw))
            dur = max(self.config.min_duration, duration or self.config.default_duration)

            # Only override the head's neutral resting height when z is explicitly
            # given. Forcing z=0 makes "center" an unreachable pose the IK clamps,
            # so the head silently fails to move (the bug we're fixing).
            pose_kwargs = {"roll": roll, "pitch": pitch, "yaw": yaw, "degrees": True}
            if z is not None:
                pose_kwargs["z"] = z
                pose_kwargs["mm"] = True

            if not self._motion_lock.acquire(blocking=False):
                return {"status": "busy"}
            try:
                self._mini.goto_target(
                    head=create_head_pose(**pose_kwargs),
                    duration=dur,
                )
            finally:
                self._motion_lock.release()

            return {
                "status": "success",
                "position": {"z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
                "duration": dur,
            }

        except Exception as e:
            logger.error(f"Failed to move head: {e}")
            return {"status": "error", "message": str(e)}

    def move_antennas(
        self,
        left: float = 0,
        right: float = 0,
        duration: float | None = None,
    ) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        if not self._rate_limit("move_antennas"):
            return {"status": "rate_limited"}

        try:
            dur = max(self.config.min_duration, duration or self.config.antenna_duration)

            if not self._motion_lock.acquire(blocking=False):
                return {"status": "busy"}
            try:
                self._mini.goto_target(
                    left_antenna=left,
                    right_antenna=right,
                    duration=dur,
                )
            finally:
                self._motion_lock.release()

            return {
                "status": "success",
                "antennas": {"left": left, "right": right},
                "duration": dur,
            }

        except Exception as e:
            logger.error(f"Failed to move antennas: {e}")
            return {"status": "error", "message": str(e)}

    def _get_emotions(self):
        if self._emotions is None:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self._emotions = RecordedMoves(EMOTIONS_DATASET)
        return self._emotions

    def _get_dances(self):
        if self._dances is None:
            from reachy_mini.motion.recorded_move import RecordedMoves
            self._dances = RecordedMoves(DANCES_DATASET)
        return self._dances

    def play_emotion(self, emotion: str) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        if not self._rate_limit("play_emotion"):
            return {"status": "rate_limited"}

        try:
            library = self._get_emotions()
            try:
                move = library.get(emotion)
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Unknown emotion '{emotion}'",
                    "available": library.list_moves(),
                }

            if not self._motion_lock.acquire(blocking=False):
                return {"status": "busy"}
            try:
                self._mini.play_move(move, initial_goto_duration=1.0)
            finally:
                self._motion_lock.release()

            return {"status": "success", "emotion": emotion}

        except Exception as e:
            logger.error(f"Failed to play emotion: {e}")
            return {"status": "error", "message": str(e)}

    def dance(self, dance_name: str) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        if not self._rate_limit("dance"):
            return {"status": "rate_limited"}

        try:
            library = self._get_dances()
            try:
                move = library.get(dance_name)
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Unknown dance '{dance_name}'",
                    "available": library.list_moves(),
                }

            if not self._motion_lock.acquire(blocking=False):
                return {"status": "busy"}
            try:
                self._mini.play_move(move, initial_goto_duration=1.0)
            finally:
                self._motion_lock.release()

            return {"status": "success", "dance": dance_name}

        except Exception as e:
            logger.error(f"Failed to dance: {e}")
            return {"status": "error", "message": str(e)}

    def list_emotions(self) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        try:
            return {"status": "success", "emotions": self._get_emotions().list_moves()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_dances(self) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        try:
            return {"status": "success", "dances": self._get_dances().list_moves()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def capture_image(self) -> dict:
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        if not self._rate_limit("capture_image"):
            return {"status": "rate_limited"}

        try:
            frame = self._mini.media.get_frame()
            if frame is None:
                return {"status": "error", "message": "No frame available"}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.config.capture_dir / f"capture_{timestamp}.jpg"

            try:
                import cv2
                cv2.imwrite(str(filepath), frame)
            except ImportError:
                if hasattr(frame, "save"):
                    frame.save(filepath)
                else:
                    return {"status": "error", "message": "cv2 not available for saving images"}

            return {"status": "success", "filepath": str(filepath)}

        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return {"status": "error", "message": str(e)}

    def say(self, text: str, voice: str | None = None) -> dict:
        """Speak text via the interface's ElevenLabs TTS path.

        `voice` is accepted for backward compatibility but ignored; the voice is
        configured through REACHY_ELEVENLABS_VOICE_ID on the interface side.
        """
        if not self.is_connected:
            return {"status": "error", "message": "Not connected to robot"}
        if not self._rate_limit("say"):
            return {"status": "rate_limited"}
        if not text or not text.strip():
            return {"status": "error", "message": "Empty text"}
        if len(text) > self.config.max_say_chars:
            return {"status": "error", "message": f"Text exceeds {self.config.max_say_chars} chars"}
        if self._speak_handler is None:
            return {"status": "error", "message": "Speech is not available (no TTS handler registered)"}

        try:
            self._speak_handler(text)
            return {"status": "success", "chars": len(text)}
        except Exception as e:
            logger.error(f"Failed to speak: {e}")
            return {"status": "error", "message": str(e)}

    def get_status(self) -> dict:
        return {
            "connected": self.is_connected,
            "config": {
                "connection_mode": self.config.connection_mode,
                "media_backend": self.config.media_backend,
            },
        }


_bridge: ReachyBridge | None = None


def get_bridge() -> ReachyBridge:
    global _bridge
    if _bridge is None:
        _bridge = ReachyBridge()
    return _bridge


def set_bridge(bridge: ReachyBridge) -> None:
    """Inject a shared bridge (used when the bridge process owns the robot)."""
    global _bridge
    _bridge = bridge
