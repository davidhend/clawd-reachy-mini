"""Tests for the fuzzy wake-phrase matcher in the voice service."""

import pytest

from clawd_reachy_mini.voice import match_wake_phrase


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Punctuation-insensitive, tolerant of Whisper spellings of "gizmo":
        ("Hey, Gizmo, what time is it?", "what time is it"),
        ("Hey Gismo, what time is it?", "what time is it"),
        ("Hey gizmos, what time is it?", "what time is it"),
        # Weak greetings need an exact alias — must only count at utterance start:
        ("uh, gizmo, are you there", "are you there"),
        ("that is a gizmo to be honest", None),
        # Strong greeting variants:
        ("Hey Gizmo what time is it", "what time is it"),
        ("hi gizmo, how are you", "how are you"),
        ("Okay Gizmo. Tell me a joke.", "tell me a joke"),
        # Wake phrase mid-utterance, preceded by filler:
        ("um, hey gizmo, hello", "hello"),
        # Bare wake phrase -> empty remainder (service answers "Yes?"):
        ("Hey, Gizmo!", ""),
        # Name-first address (no greeting) at utterance start:
        ("Gizmo, go ahead and install the updates on Solbox", "go ahead and install the updates on solbox"),
        ("Gizmo, was the dashboard refreshed?", "was the dashboard refreshed"),
        ("Gismo what time is it", "what time is it"),
        # Bare name -> empty remainder (service answers "Yes?"):
        ("Gizmo", ""),
        ("Gizmo?", ""),
        # Name mid-sentence must NOT wake (name-first is start-anchored):
        ("the gizmo broke again", None),
        # Near-name at start must NOT wake (exact alias only, no fuzz):
        ("Gizmondo, hello", None),
        # Ambient speech must NOT wake:
        ("It might be a thing...", None),
        ("There is not really a lot of them there.", None),
        ("Yeah, I wonder", None),
        # Greeting followed by a non-name word must NOT wake:
        ("hey there, that works", None),
        # The old name must NOT wake any more (renamed to Gizmo):
        ("Hey Reachy what time is it", None),
        ("Hey, Ricci, what time is it?", None),
    ],
)
def test_match_wake_phrase(text, expected):
    assert match_wake_phrase(text) == expected
