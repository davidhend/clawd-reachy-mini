"""OpenClaw tool definitions for Reachy Mini control."""

import logging

from clawd_reachy_mini.actions.bridge import get_bridge

logger = logging.getLogger(__name__)


def reachy_connect(connection_mode: str = "auto") -> dict:
    """Connect to the Reachy Mini robot."""
    return get_bridge().connect(connection_mode)


def reachy_disconnect() -> dict:
    """Disconnect from the Reachy Mini robot."""
    return get_bridge().disconnect()


def reachy_stop() -> dict:
    """Emergency stop: cancel any in-flight robot motion immediately."""
    return get_bridge().stop()


def reachy_move_head(
    z: float | None = None,
    roll: float = 0,
    pitch: float = 0,
    yaw: float = 0,
    duration: float = 1.0,
) -> dict:
    """Move the robot's head to a target position.

    Leave `z` unset to keep the head's neutral resting height (passing roll/pitch/
    yaw of 0 then re-centers the head). Only set `z` to deliberately raise/lower it.
    """
    return get_bridge().move_head(z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration)


def reachy_move_antennas(
    left: float = 0,
    right: float = 0,
    duration: float = 0.5,
) -> dict:
    """Move the robot's antennas."""
    return get_bridge().move_antennas(left=left, right=right, duration=duration)


def reachy_play_emotion(emotion: str) -> dict:
    """Play a predefined emotion animation."""
    return get_bridge().play_emotion(emotion)


def reachy_dance(dance_name: str) -> dict:
    """Trigger a dance routine."""
    return get_bridge().dance(dance_name)


def reachy_list_emotions() -> dict:
    """List available emotion names."""
    return get_bridge().list_emotions()


def reachy_list_dances() -> dict:
    """List available dance names."""
    return get_bridge().list_dances()


def reachy_capture_image() -> dict:
    """Capture an image from the robot's camera."""
    return get_bridge().capture_image()


def reachy_say(text: str, voice: str | None = None) -> dict:
    """Make the robot speak using text-to-speech."""
    return get_bridge().say(text=text, voice=voice)


def reachy_status() -> dict:
    """Get the current status of the robot."""
    return get_bridge().get_status()


_DISPATCH = {
    "connect": reachy_connect,
    "disconnect": reachy_disconnect,
    "stop": reachy_stop,
    "move_head": reachy_move_head,
    "move_antennas": reachy_move_antennas,
    "play_emotion": reachy_play_emotion,
    "dance": reachy_dance,
    "list_emotions": reachy_list_emotions,
    "list_dances": reachy_list_dances,
    "capture_image": reachy_capture_image,
    "say": reachy_say,
    "status": reachy_status,
}


def dispatch(tool_name: str, arguments: dict) -> dict:
    """Route a gateway tool.request to the matching bridge action.

    Accepts bare names (`move_head`) or prefixed names (`reachy_move_head`).
    """
    key = tool_name[len("reachy_"):] if tool_name.startswith("reachy_") else tool_name
    handler = _DISPATCH.get(key)
    if handler is None:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    try:
        return handler(**(arguments or {}))
    except TypeError as e:
        return {"status": "error", "message": f"Bad arguments for {tool_name}: {e}"}
