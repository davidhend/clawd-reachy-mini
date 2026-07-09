---
name: reachy-mini
description: Control the Reachy Mini robot through natural language. Use the reachy_* MCP tools — move/center the head and antennas, speak via ElevenLabs TTS, play emotions/dances, and check status.
---

# Reachy Mini

Control the Reachy Mini robot via the **`reachy-mini` MCP server**, which exposes
typed `reachy_*` tools.

## IMPORTANT: use the MCP tools, do not curl the daemon

Robot control is provided by the `reachy_*` MCP tools listed below. **Always use
those tools.** Do **not** call the Reachy daemon's HTTP API directly (e.g.
`curl http://<robot>:8000/api/move/goto` or `/api/media/...`). The MCP tools add
safety clamps, correct centering, and ElevenLabs speech; raw daemon calls bypass
all of that and (for centering) leave the body rotated so the head looks off-center.

## Tools

- **`reachy_status`** — daemon/robot status (connection, readiness, version).
- **`reachy_center_head(duration=1.0)`** — return head + body to the neutral,
  level, forward-facing home pose. This is the correct way to "center" the head.
- **`reachy_move_head(roll=0, pitch=0, yaw=0, z=None, duration=1.0)`** — move the
  head. `roll/pitch/yaw` in **degrees** (clamped to ±30, ±30, ±45). `z` optional,
  in **mm** (omit to keep neutral height). Body rotation is left unchanged.
- **`reachy_move_antennas(left=0, right=0, duration=0.5)`** — move antennas, in
  degrees (0 = neutral).
- **`reachy_wake_up`** — raise head to active posture. **`reachy_sleep`** — lower
  head to rest.
- **`reachy_say(text, gesture="none")`** — speak via ElevenLabs TTS (≤ 800 chars).
  The head bobs with the audio automatically. Pass a `gesture` (`tilt`, `look_up`,
  `look_down`, `look_left`, `look_right`, `lean_in`) to move the head **while**
  speaking (fired in sync with the audio, not before). Prefer this over a separate
  `reachy_move_head` before `reachy_say` (that reads as move-then-talk).
- **`reachy_stop`** — emergency stop: cancel motion and stop any sound.
- **`reachy_play_emotion(name)`** / **`reachy_list_emotions`** — play the desktop-app
  emotion animations (e.g. `curious1`, `laughing1`, `welcoming1`, `surprised1`,
  `proud1`, `shy1`). Call `reachy_list_emotions` first; do not invent names.
- **`reachy_play_dance(name)`** / **`reachy_list_dances`** — play dance animations.
  Call `reachy_list_dances` first; do not invent names.

## Operating policy

Reachy is a **physical robot** — every tool call is a real-world action.

- **Movement is gentle and brief.** Use `duration` ≥ 0.5s for meaningful motion.
  Don't chain motion tools in tight loops or call the same one more than ~3 times
  in a row without a user instruction.
- **Speech respects the room.** Keep `reachy_say` short and on-topic. Don't speak
  unprompted after a quiet period unless asked.
- **Stop on command.** If the user says "stop", "wait", "pause", "quiet", "shut
  up", or "emergency stop", call `reachy_stop` first, then acknowledge.
- **Fail closed.** If a tool returns a string starting with `ERROR`, do not retry
  immediately — report the problem.

## Examples

- "Center Reachy's head" → `reachy_center_head`
- "Look up and to the left" → `reachy_move_head(pitch=-15, yaw=20)`
- "Introduce yourself" → `reachy_say("Hi, I'm Reachy!")`, then a small
  `reachy_move_head`/`reachy_move_antennas` wiggle
- "Do a happy dance" → `reachy_list_moves(...)` then `reachy_play_recorded_move(...)`
- "What's your status?" → `reachy_status`

## Notes

- The robot daemon must be running and own the hardware (no other SDK client
  holding it). The MCP server reaches the daemon over the LAN.
- `reachy_say` requires `REACHY_ELEVENLABS_API_KEY` configured on the MCP server.
