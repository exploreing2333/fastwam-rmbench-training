#!/usr/bin/env python3
"""Build reproducible per-press GT labels and target-count metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastwam.datasets.lerobot.press_button_labels import (  # noqa: E402
    RAW_BUTTON_TEXT, numbered_press_labels, validate_numbered_press_texts,
)

DEST = ROOT / "data/RMBench/data/data/press_button/demo_clean"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-original", type=Path,
                        help="Original RMBench annotation, required only on first build")
    parser.add_argument("--output-dir", type=Path, default=DEST)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    backup = directory / "language_annotation.before_count_labels.json"
    if not backup.is_file():
        if args.source_original is None:
            parser.error("Pass --source-original on first build")
        shutil.copyfile(args.source_original, backup)
    elif args.source_original and backup.read_bytes() != args.source_original.read_bytes():
        raise ValueError("Original annotation differs from the preserved backup")

    original_bytes = backup.read_bytes()
    original = json.loads(original_bytes)
    if len(original) != 50:
        raise ValueError("Expected 50 original press_button demonstrations")
    revised: dict[str, list[list[str | int]]] = {}
    episodes: dict[str, dict] = {}
    total_segments = total_frames = 0
    for episode, segments in original.items():
        texts = tuple(str(item[0]).strip() for item in segments)
        if len(segments) < 3 or texts[-1] != "Press the confirm button.":
            raise ValueError(f"Invalid button sequence in {episode}")
        if any(text not in RAW_BUTTON_TEXT for text in texts):
            raise ValueError(f"Unexpected original label in {episode}")
        buttons = tuple(RAW_BUTTON_TEXT[text] for text in texts)
        left = buttons.count("left")
        middle = buttons.count("middle")
        if buttons != ("left",) * left + ("middle",) * middle + ("confirm",):
            raise ValueError(f"Nonmonotonic button order in {episode}")
        if not (1 <= left <= 9 and 1 <= middle <= 9 and left + middle <= 10):
            raise ValueError(f"Invalid card count in {episode}")
        labels = numbered_press_labels(texts)
        if validate_numbered_press_texts(labels) != (left, middle):
            raise AssertionError(episode)
        events = []
        revised[episode] = []
        for index, (label, raw_row, button) in enumerate(zip(labels, segments, buttons)):
            duration = int(raw_row[1])
            if duration <= 0:
                raise ValueError(f"Invalid duration in {episode}")
            revised[episode].append([label, duration])
            events.append({
                "raw_index": index,
                "button": button,
                "ordinal": buttons[:index + 1].count(button),
                "target_total": 1 if button == "confirm" else (left if button == "left" else middle),
                "label": label,
                "duration_frames": duration,
            })
            total_frames += duration
            total_segments += 1
        episodes[episode] = {
            "left_target_total": left,
            "middle_target_total": middle,
            "presses": events,
        }
    if total_segments != 448:
        raise ValueError(f"Expected 448 press segments, got {total_segments}")
    annotation_bytes = (json.dumps(revised, ensure_ascii=False, indent=2) + "\n").encode()
    metadata = {
        "format": "rmbench_press_button_ordinal_per_press_v1",
        "original_annotation_sha256": sha256(original_bytes),
        "numbered_annotation_sha256": sha256(annotation_bytes),
        "episode_count": len(episodes),
        "press_segment_count": total_segments,
        "raw_frame_count": total_frames,
        "model_label_contract": "one numbered label per physical press; target total is metadata only",
        "episodes": episodes,
    }
    (directory / "language_annotation.json").write_bytes(annotation_bytes)
    (directory / "press_count_labels.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episodes": len(episodes), "segments": total_segments,
                      "raw_frames": total_frames, "labels": len({r[0] for v in revised.values() for r in v}),
                      "annotation_sha256": metadata["numbered_annotation_sha256"]}))


if __name__ == "__main__":
    main()
