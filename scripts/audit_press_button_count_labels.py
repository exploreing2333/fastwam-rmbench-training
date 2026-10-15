#!/usr/bin/env python3
"""Check all numbered GT labels, unchanged spans, T5 files, and 32-step hold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from fastwam.datasets.lerobot.gt_subtask_video_dataset import GroundTruthSubtaskRobotVideoDataset
from fastwam.datasets.lerobot.press_button_labels import (
    numbered_press_labels, validate_numbered_press_texts,
)

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/RMBench/data/data/press_button/demo_clean"


class MetadataEpisodes:
    def __init__(self, lengths):
        stops = torch.tensor(lengths).cumsum(0)
        self.episode_data_index = {
            "from": torch.cat([torch.tensor([0]), stops[:-1]]), "to": stops,
        }

    def __len__(self):
        return int(self.episode_data_index["to"][-1])


def metadata_dataset(episodes, rows, source_map, *, numbered):
    dataset = GroundTruthSubtaskRobotVideoDataset.__new__(GroundTruthSubtaskRobotVideoDataset)
    dataset.episode_indices = tuple(episodes)
    dataset.lerobot_dataset = MetadataEpisodes([rows[episode]["length"] for episode in episodes])
    dataset.subtask_source_map = source_map
    dataset.gt_subtask_text_embedding_cache_dir = ROOT / "data/gt_subtask_text_embeds"
    dataset.context_len = 128
    dataset.training_shuffle_unit = "trajectory"
    dataset.trajectory_anchor_stride = 1
    dataset.gt_subtask_segment_mode = "v72_semantic"
    dataset.gt_subtask_event_instance_tasks = ("press_button",)
    dataset.press_button_label_mode = "ordinal_per_press_v1" if numbered else None
    dataset._gt_stage_index = {}
    dataset._subtask_segment_index = {}
    dataset._subtask_segment_end = {}
    dataset._gt_texts_by_episode = {}
    dataset._gt_raw_ranges_by_episode = {}
    dataset._load_gt_subtask_trajectories()
    return dataset


def main() -> None:
    torch.set_num_threads(1)
    annotation_bytes = (DIRECTORY / "language_annotation.json").read_bytes()
    original_bytes = (DIRECTORY / "language_annotation.before_count_labels.json").read_bytes()
    annotation = json.loads(annotation_bytes)
    original = json.loads(original_bytes)
    labels = json.loads((DIRECTORY / "press_count_labels.json").read_text())
    assert labels["original_annotation_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert labels["numbered_annotation_sha256"] == hashlib.sha256(annotation_bytes).hexdigest()
    assert len(annotation) == len(original) == labels["episode_count"] == 50
    source = json.loads((ROOT / "data/source_map_template.json").read_text())
    rows = {int(row["episode_index"]): row for row in source["entries"]}
    episodes = tuple(range(50))
    assert all(rows[index]["source_task_name"] == "press_button" for index in episodes)
    cache = {}
    total_segments = total_frames = held = cross = 0
    with tempfile.TemporaryDirectory(prefix="press-button-count-audit-") as temporary:
        temporary = Path(temporary)
        current_map = temporary / "current_map.json"
        current_map.write_text(json.dumps({**source, "raw_root": str(ROOT / "data/RMBench/data/data")}))
        old_root = temporary / "original" / "press_button/demo_clean"
        old_root.mkdir(parents=True)
        (old_root / "language_annotation.json").write_bytes(original_bytes)
        old_map = temporary / "old_map.json"
        old_map.write_text(json.dumps({**source, "raw_root": str(temporary / "original")}))
        current = metadata_dataset(episodes, rows, current_map, numbered=True)
        old = metadata_dataset(episodes, rows, old_map, numbered=False)
        for episode in episodes:
            row = rows[episode]
            key = f"episode_{row['source_episode_id']}"
            original_rows, current_rows = original[key], annotation[key]
            expected = numbered_press_labels(tuple(item[0] for item in original_rows))
            target_left, target_middle = validate_numbered_press_texts(expected)
            assert current._gt_texts_by_episode[episode] == expected
            assert tuple(item[0] for item in current_rows) == expected
            assert tuple(item[1] for item in current_rows) == tuple(item[1] for item in original_rows)
            assert labels["episodes"][key]["left_target_total"] == target_left
            assert labels["episodes"][key]["middle_target_total"] == target_middle
            assert len(labels["episodes"][key]["presses"]) == len(expected)
            for i, text in enumerate(expected):
                event = labels["episodes"][key]["presses"][i]
                assert event["label"] == text and event["duration_frames"] == current_rows[i][1]
                if text not in cache:
                    context = current._load_gt_context(text)
                    assert context.dtype == torch.bfloat16
                    cache[text] = context
            stage = current._gt_stage_index[episode]
            end = current._subtask_segment_end[episode]
            np.testing.assert_array_equal(stage, old._gt_stage_index[episode])
            np.testing.assert_array_equal(end, old._subtask_segment_end[episode])
            assert len(stage) == row["length"]
            assert len(expected) == len(np.unique(stage))
            offsets = np.arange(len(stage))[:, None] + np.arange(32)
            targets = np.minimum(offsets, end[:, None])
            cross += int(np.count_nonzero(stage[targets] != stage[:, None]))
            held += int(np.count_nonzero(offsets > end[:, None]))
            video_offsets = np.arange(0, 33, 4)
            video_targets = np.minimum(np.arange(len(stage))[:, None] + video_offsets, end[:, None])
            assert np.all(stage[video_targets] == stage[:, None])
            total_frames += len(stage)
            total_segments += len(expected)
    assert total_frames == 26302 and total_segments == 448 and len(cache) == 19
    assert held == 222208 and cross == 0
    report = {
        "passed": True, "episodes": 50, "frames": total_frames,
        "press_segments": total_segments, "unique_numbered_labels": len(cache),
        "original_spans_unchanged": True, "cross_press_action_targets": cross,
        "cross_press_video_targets": 0, "held_action_targets": held,
        "sampling_contract": current.sampling_contract,
    }
    output = ROOT / "reports/press_button_count_labels_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PRESS_BUTTON_COUNT_LABELS_AUDIT_PASSED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
