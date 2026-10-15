"""Labels for individual RMBench button presses.

The ordinal is observable from V72's completed-subtask history at deployment.
The target total is stored in annotation metadata, not fed to this current-only
policy, because the deployed condition does not expose that total.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


RAW_BUTTON_TEXT = {
    "Press the left button one time.": "left",
    "Press the middle button one time.": "middle",
    "Press the confirm button.": "confirm",
}
ORDINALS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth",
)


def numbered_press_label(current: str, completed: Sequence[str]) -> str:
    current = str(current).strip()
    if current not in RAW_BUTTON_TEXT:
        raise ValueError(f"Unexpected press_button subtask: {current!r}")
    if any(str(item).strip() not in RAW_BUTTON_TEXT for item in completed):
        raise ValueError("Unexpected completed press_button subtask")
    button = RAW_BUTTON_TEXT[current]
    if button == "confirm":
        return current
    ordinal = Counter(RAW_BUTTON_TEXT[str(item).strip()] for item in completed)[button]
    if ordinal >= len(ORDINALS):
        raise ValueError(f"Press ordinal exceeds supported range: {ordinal + 1}")
    return f"Press the {button} button for the {ORDINALS[ordinal]} time."


def numbered_press_labels(raw_texts: Sequence[str]) -> tuple[str, ...]:
    return tuple(numbered_press_label(text, raw_texts[:index])
                 for index, text in enumerate(raw_texts))


def validate_numbered_press_texts(texts: Sequence[str]) -> tuple[int, int]:
    """Check the full left*, middle*, confirm order and each exact ordinal."""
    counts = {"left": 0, "middle": 0}
    phase = "left"
    for index, text in enumerate(texts):
        if text == "Press the confirm button.":
            if index != len(texts) - 1 or not counts["left"] or not counts["middle"]:
                raise ValueError("Confirm must be the final press after both buttons")
            return counts["left"], counts["middle"]
        button = "left" if str(text).startswith("Press the left button for the ") else "middle"
        if button == "middle":
            phase = "middle"
        elif phase != "left":
            raise ValueError("Left presses must precede middle presses")
        if counts[button] >= 9:
            raise ValueError("RMBench card targets are limited to nine presses per button")
        expected = f"Press the {button} button for the {ORDINALS[counts[button]]} time."
        if text != expected:
            raise ValueError(f"Expected numbered press {expected!r}, got {text!r}")
        counts[button] += 1
    raise ValueError("Missing confirm press")
