"""Build frame spans using V72's adjacent-instruction merge rule."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence


@dataclass(frozen=True)
class GroundTruthSubtaskSegment:
    text: str
    start: int
    stop: int  # Exclusive frame boundary.
    raw_first: int
    raw_last: int  # Inclusive raw annotation index.


def build_gt_subtask_segments(
    annotations: Sequence,
    task_name: str,
    mode: str = "v72_semantic",
    event_instance_tasks: Sequence[str] = ("press_button",),
) -> list[GroundTruthSubtaskSegment]:
    if mode not in {"raw_annotation", "v72_semantic"}:
        raise ValueError(f"Unknown GT subtask segment mode: {mode!r}")
    if not annotations:
        raise ValueError("GT subtask annotations must be non-empty.")
    merge = mode == "v72_semantic" and task_name not in event_instance_tasks
    segments = []
    cursor = 0
    for raw_index, row in enumerate(annotations):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"Malformed GT subtask annotation: {row!r}")
        text, duration = str(row[0]).strip(), int(row[1])
        if not text or duration <= 0:
            raise ValueError(f"Invalid GT subtask annotation: {row!r}")
        stop = cursor + duration
        normalized = " ".join(text.casefold().split())
        if merge and segments and " ".join(segments[-1].text.casefold().split()) == normalized:
            segments[-1] = replace(segments[-1], stop=stop, raw_last=raw_index)
        else:
            segments.append(GroundTruthSubtaskSegment(text, cursor, stop, raw_index, raw_index))
        cursor = stop
    return segments
