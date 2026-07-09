"""Tests for the actions sub-package bridge surface."""

from unittest.mock import MagicMock, patch

import pytest

from clawd_reachy_mini.actions.bridge import ReachyBridge
from clawd_reachy_mini.actions.config import ReachyConfig
from clawd_reachy_mini.actions.tools import dispatch


@pytest.fixture
def config():
    return ReachyConfig(min_command_interval=0.0)


@pytest.fixture
def bridge(config):
    return ReachyBridge(config=config)


class TestReachyBridge:
    def test_initial_state(self, bridge):
        assert not bridge.is_connected
        assert bridge._mini is None

    def test_disconnect_when_not_connected(self, bridge):
        assert bridge.disconnect()["status"] == "not_connected"

    def test_stop_when_not_connected(self, bridge):
        assert bridge.stop()["status"] == "not_connected"

    def test_move_head_when_not_connected(self, bridge):
        result = bridge.move_head(z=10, roll=5)
        assert result["status"] == "error"
        assert "Not connected" in result["message"]

    def test_get_status_when_not_connected(self, bridge):
        assert bridge.get_status()["connected"] is False

    def test_attach_existing_marks_connected(self, bridge):
        mini = MagicMock()
        bridge.attach_existing(mini)
        assert bridge.is_connected
        assert bridge._mini is mini

    def test_detach_releases_without_calling_exit(self, bridge):
        mini = MagicMock()
        bridge.attach_existing(mini)
        bridge.detach()
        assert not bridge.is_connected
        mini.__exit__.assert_not_called()

    @patch("reachy_mini.utils.create_head_pose", return_value={"pose": "data"})
    def test_move_head_clamps_to_safety_limits(self, _pose, bridge):
        bridge.attach_existing(MagicMock())
        result = bridge.move_head(roll=999, pitch=-999, yaw=999)
        assert result["status"] == "success"
        assert result["position"]["roll"] == bridge.config.max_roll
        assert result["position"]["pitch"] == -bridge.config.max_pitch
        assert result["position"]["yaw"] == bridge.config.max_yaw

    @patch("reachy_mini.utils.create_head_pose", return_value={"pose": "data"})
    def test_move_head_respects_min_duration(self, _pose, bridge):
        bridge.attach_existing(MagicMock())
        result = bridge.move_head(duration=0.01)
        assert result["duration"] == bridge.config.min_duration

    def test_rate_limit_blocks_rapid_calls(self):
        config = ReachyConfig(min_command_interval=10.0)
        bridge = ReachyBridge(config=config)
        bridge.attach_existing(MagicMock())
        with patch("reachy_mini.utils.create_head_pose", return_value={}):
            first = bridge.move_head(roll=1)
            second = bridge.move_head(roll=2)
        assert first["status"] == "success"
        assert second["status"] == "rate_limited"

    def test_stop_calls_cancel_move(self, bridge):
        mini = MagicMock()
        bridge.attach_existing(mini)
        assert bridge.stop()["status"] == "stopped"
        mini.cancel_move.assert_called_once()

    def test_say_rejects_empty(self, bridge):
        bridge.attach_existing(MagicMock())
        assert bridge.say("")["status"] == "error"

    def test_say_rejects_oversize(self, bridge):
        bridge.attach_existing(MagicMock())
        result = bridge.say("x" * (bridge.config.max_say_chars + 1))
        assert result["status"] == "error"
        assert "exceeds" in result["message"]


class TestDispatch:
    def test_unknown_tool(self):
        result = dispatch("not_a_tool", {})
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    def test_strips_reachy_prefix(self):
        # Routes to reachy_status, which is safe on a disconnected bridge.
        result = dispatch("reachy_status", {})
        assert "connected" in result

    def test_bare_name(self):
        result = dispatch("status", {})
        assert "connected" in result

    def test_bad_arguments_returns_error(self):
        result = dispatch("move_head", {"nonexistent_arg": 5})
        assert result["status"] == "error"
        assert "Bad arguments" in result["message"]


class TestConfig:
    def test_default_config(self):
        config = ReachyConfig()
        assert config.connection_mode == "auto"
        assert config.default_duration == 1.0
        assert config.max_roll == 30.0

    def test_custom_config(self):
        config = ReachyConfig(connection_mode="localhost_only", default_duration=2.0)
        assert config.connection_mode == "localhost_only"
        assert config.default_duration == 2.0
