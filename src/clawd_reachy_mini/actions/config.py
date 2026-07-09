"""Configuration for the Reachy action surface (separate from bridge runtime config)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReachyConfig:
    """Configuration for the action-skill side of the bridge."""

    connection_mode: str = "auto"  # "auto", "localhost_only", "network"
    media_backend: str = "default"  # "default" or "gstreamer"
    capture_dir: Path = field(default_factory=lambda: Path.home() / ".clawd-reachy-mini" / "captures")

    # Movement defaults
    default_duration: float = 1.0
    antenna_duration: float = 0.5
    min_duration: float = 0.3

    # Safety limits (degrees) — tighter than SDK clamps
    max_roll: float = 30.0
    max_pitch: float = 30.0
    max_yaw: float = 45.0

    # Rate limiting (seconds between commands of the same kind)
    min_command_interval: float = 0.25

    # Speech
    piper_model_path: str | None = None  # Set via env REACHY_PIPER_MODEL
    max_say_chars: int = 800

    def __post_init__(self):
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        if self.piper_model_path is None:
            self.piper_model_path = os.environ.get("REACHY_PIPER_MODEL")


_config: ReachyConfig | None = None


def get_config() -> ReachyConfig:
    global _config
    if _config is None:
        _config = ReachyConfig()
    return _config


def set_config(config: ReachyConfig) -> None:
    global _config
    _config = config
