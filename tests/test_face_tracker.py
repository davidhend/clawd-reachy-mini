"""Tests for the face-tracking control math in the camera service."""

import math

from clawd_reachy_mini.camera import (
    DEADBAND_RAD,
    MAX_STEP_RAD,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    compute_step,
    pixel_to_angles,
)


def test_face_right_of_center_turns_right():
    # Daemon convention: positive yaw = left, so right of frame -> negative yaw.
    yaw_err, _ = pixel_to_angles(STREAM_WIDTH * 0.9, STREAM_HEIGHT / 2)
    assert yaw_err < 0


def test_face_left_of_center_turns_left():
    yaw_err, _ = pixel_to_angles(STREAM_WIDTH * 0.1, STREAM_HEIGHT / 2)
    assert yaw_err > 0


def test_face_above_center_pitches_up():
    # Daemon convention: positive pitch = down, so above frame center -> negative.
    _, pitch_err = pixel_to_angles(STREAM_WIDTH / 2, STREAM_HEIGHT * 0.1)
    assert pitch_err < 0


def test_real_capture_face_position():
    # Face center from the first real Reachy frame (2026-06-12): (978, 235).
    yaw_err, pitch_err = pixel_to_angles(978, 235)
    assert abs(math.degrees(yaw_err) - (-1.03)) < 0.1
    assert abs(math.degrees(pitch_err) - (-16.95)) < 0.1


def test_deadband_suppresses_small_errors():
    assert compute_step(DEADBAND_RAD * 0.9) == 0.0
    assert compute_step(-DEADBAND_RAD * 0.9) == 0.0


def test_step_is_slew_limited():
    assert compute_step(math.radians(60)) == MAX_STEP_RAD
    assert compute_step(math.radians(-60)) == -MAX_STEP_RAD


def test_step_is_proportional_in_between():
    err = math.radians(5)
    step = compute_step(err)
    assert 0 < step < MAX_STEP_RAD
    assert abs(step - 0.35 * err) < 1e-9
