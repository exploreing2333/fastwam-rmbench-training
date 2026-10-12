from __future__ import annotations

from typing import Any, Optional, Sequence


def select_interval_history(
    current: Any,
    history: Optional[Sequence[Any]],
    num_frames: int,
    stride: int,
) -> tuple[list[Any], list[bool]]:
    """Select oldest-to-current observations while only padding missing history slots."""
    num_frames = int(num_frames)
    stride = int(stride)
    if num_frames <= 0 or stride <= 0:
        raise ValueError(f"Invalid history config: num_frames={num_frames}, stride={stride}.")

    history_span = (num_frames - 1) * stride
    available = list(history) if history is not None else [current]
    if len(available) == 0:
        raise ValueError("`history` must contain the current observation.")
    available = available[-(history_span + 1) :]
    current_index = len(available) - 1

    selected = []
    is_pad = []
    for frame_index in range(num_frames):
        index = current_index - (num_frames - 1 - frame_index) * stride
        padded = index < 0
        selected.append(available[max(index, 0)])
        is_pad.append(padded)
    return selected, is_pad


def select_uniform_history(
    current: Any,
    history: Optional[Sequence[Any]],
    num_frames: int,
) -> list[Any]:
    """Uniformly sample oldest-to-current observations over all available history."""
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    available = list(history) if history is not None else [current]
    if len(available) == 0:
        raise ValueError("`history` must contain the current observation.")
    if num_frames == 1:
        return [available[-1]]

    span = len(available) - 1
    denominator = num_frames - 1
    indices = [
        (frame_idx * span + denominator // 2) // denominator
        for frame_idx in range(num_frames)
    ]
    return [available[index] for index in indices]
