"""Current-only FastWAM with authoritative, frame-aligned GT subtasks.

No ProgressModel output, history image, event anchor, or init cache is loaded.
Only the original FastWAM camera layout and subtask terminal-hold operations
are reused from the deployed-subtask dataset; the labels come from RMBench GT.
Default spans follow V72: adjacent equal instructions merge, while counted
press_button events retain their individual boundaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fastwam_qwen.datasets.robot_video_dataset import (
    DEFAULT_PROMPT,
    RobotVideoDatasetNoT5,
)
from .progress_subtask_video_dataset import ProgressSubtaskRobotVideoDataset
from .gt_subtask_segments import build_gt_subtask_segments
from .press_button_labels import validate_numbered_press_texts


class GroundTruthSubtaskRobotVideoDataset(RobotVideoDatasetNoT5):
    def __init__(
        self,
        *args,
        subtask_source_map: str,
        gt_subtask_text_embedding_cache_dir: str,
        exclude_terminal_hold_loss_subtasks=(),
        gt_subtask_segment_mode: str = "v72_semantic",
        gt_subtask_event_instance_tasks=("press_button",),
        gt_subtask_annotation_files=None,
        press_button_label_mode: str | None = None,
        **kwargs,
    ) -> None:
        if isinstance(exclude_terminal_hold_loss_subtasks, str):
            raise ValueError("exclude_terminal_hold_loss_subtasks must be a sequence of subtask texts.")
        if any(not isinstance(text, str) or not text.strip()
               for text in exclude_terminal_hold_loss_subtasks):
            raise ValueError("Terminal-hold loss exclusions must be non-empty subtask texts.")
        self.exclude_terminal_hold_loss_subtasks = frozenset(
            " ".join(text.casefold().split())
            for text in exclude_terminal_hold_loss_subtasks
        )
        for name in (
            "progress_cache_root",
            "progress_subtask_text_embedding_cache_dir",
            "text_embedding_cache_dir",
            "tmd3_latent_cache_root",
        ):
            if kwargs.get(name) not in (None, ""):
                raise ValueError(f"GT current-only training cannot load {name}.")
        expected = {
            "history_num_frames": 1,
            "robometer_history_num_frames": 0,
            "require_qwen_history": False,
            "progress_require_gt_subtask_exact": False,
            "long_image_history_latents": 0,
            "long_image_future_latents": 0,
            "skip_rgb_when_precomputed_latents": False,
            "skip_padding_as_possible": False,
            "training_shuffle_unit": "trajectory",
            "concat_multi_camera": "robotwin",
        }
        for name, value in expected.items():
            if kwargs.get(name, value) != value:
                raise ValueError(f"GT current-only training requires {name}={value!r}.")
            kwargs[name] = value
        self.subtask_source_map = Path(subtask_source_map).expanduser().resolve()
        if gt_subtask_annotation_files is not None and not hasattr(gt_subtask_annotation_files, "items"):
            raise ValueError("gt_subtask_annotation_files must map task names to annotation files.")
        self.gt_subtask_annotation_files = {
            str(task): Path(path).expanduser().resolve()
            for task, path in (gt_subtask_annotation_files or {}).items()
        }
        self.gt_subtask_text_embedding_cache_dir = Path(
            gt_subtask_text_embedding_cache_dir
        ).expanduser().resolve()
        if gt_subtask_segment_mode not in {"raw_annotation", "v72_semantic"}:
            raise ValueError(f"Unknown GT subtask segment mode: {gt_subtask_segment_mode!r}")
        if isinstance(gt_subtask_event_instance_tasks, str):
            raise ValueError("gt_subtask_event_instance_tasks must be a sequence of task names.")
        self.gt_subtask_segment_mode = gt_subtask_segment_mode
        self.gt_subtask_event_instance_tasks = tuple(gt_subtask_event_instance_tasks)
        if press_button_label_mode not in (None, "ordinal_per_press_v1"):
            raise ValueError(f"Unknown press_button label mode: {press_button_label_mode!r}")
        if press_button_label_mode and "press_button" not in self.gt_subtask_event_instance_tasks:
            raise ValueError("Numbered press labels require individual press boundaries")
        self.press_button_label_mode = press_button_label_mode
        self.subtask_trajectory_mode = True
        self.subtask_terminal_hold = True
        self.subtask_segment_source = "gt_annotation"
        self._gt_stage_index = {}
        self._subtask_segment_index = {}
        self._subtask_segment_end = {}
        self._gt_texts_by_episode = {}
        self._gt_contexts = {}
        self._gt_raw_ranges_by_episode = {}
        super().__init__(*args, **kwargs)
        if self.current_obs_index != 0 or self.action_horizon != 32:
            raise ValueError("Expected one current condition and a 32-control action chunk.")
        if len(self.video_sample_indices) != 9:
            raise ValueError("Expected current RGB plus eight stride-4 future video targets.")
        processor = self.lerobot_dataset.processor
        if processor.action_state_transforms or processor.use_stepwise_action_norm:
            raise ValueError("Terminal hold requires absolute-qpos, non-stepwise normalization.")
        self._load_gt_subtask_trajectories()
        for texts in self._gt_texts_by_episode.values():
            for text in texts:
                if text not in self._gt_contexts:
                    self._gt_contexts[text] = self._load_gt_context(text)

        self._validate_terminal_hold_loss_subtasks()

    def _gt_annotation_path(self, raw_root, task: str) -> Path:
        return getattr(self, "gt_subtask_annotation_files", {}).get(
            task, Path(raw_root) / task / "demo_clean/language_annotation.json"
        )

    def _load_gt_subtask_trajectories(self) -> None:
        # Preserve the native annotation/episode/frame validation, then build a
        # semantic view without changing raw files or the population of anchors.
        ProgressSubtaskRobotVideoDataset._load_gt_subtask_trajectories(self)
        source = json.loads(self.subtask_source_map.read_text(encoding="utf-8"))
        rows = {int(row["episode_index"]): row for row in source["entries"]}
        annotations = {}
        for episode, raw_stage_index in self._gt_stage_index.items():
            row = rows[episode]
            task = row["source_task_name"]
            if task not in annotations:
                path = self._gt_annotation_path(source["raw_root"], task)
                annotations[task] = json.loads(path.read_text(encoding="utf-8"))
            raw_segments = annotations[task][f"episode_{row['source_episode_id']}"]
            if task == "press_button" and getattr(self, "press_button_label_mode", None):
                validate_numbered_press_texts(tuple(str(item[0]).strip() for item in raw_segments))
            segments = build_gt_subtask_segments(
                raw_segments, task, self.gt_subtask_segment_mode,
                self.gt_subtask_event_instance_tasks,
            )
            raw_to_semantic = np.empty(len(raw_segments), dtype=np.int32)
            semantic_ends = np.empty(len(segments), dtype=np.int32)
            for index, segment in enumerate(segments):
                raw_to_semantic[segment.raw_first:segment.raw_last + 1] = index
                semantic_ends[index] = min(segment.stop, len(raw_stage_index)) - 1
            stage_index = raw_to_semantic[raw_stage_index]
            self._gt_stage_index[episode] = stage_index
            self._subtask_segment_index[episode] = stage_index
            self._subtask_segment_end[episode] = semantic_ends[stage_index]
            self._gt_texts_by_episode[episode] = tuple(segment.text for segment in segments)
            self._gt_raw_ranges_by_episode[episode] = tuple(
                (segment.raw_first, segment.raw_last) for segment in segments
            )

    def _load_gt_context(self, text: str) -> torch.Tensor:
        prompt = DEFAULT_PROMPT.format(task=text)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.gt_subtask_text_embedding_cache_dir / (
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["context"], payload["mask"].bool()
        if tuple(context.shape) != (self.context_len, 4096):
            raise ValueError(f"Invalid GT T5 context shape in {path}.")
        if tuple(mask.shape) != (self.context_len,) or not bool(mask.any()):
            raise ValueError(f"Invalid GT T5 mask in {path}.")
        if not bool(torch.isfinite(context).all()):
            raise ValueError(f"Nonfinite GT T5 context in {path}.")
        context = context.clone()
        context[~mask] = 0
        return context

    def _validate_terminal_hold_loss_subtasks(self) -> None:
        available = {
            " ".join(text.casefold().split())
            for texts in self._gt_texts_by_episode.values() for text in texts
        }
        missing = self.exclude_terminal_hold_loss_subtasks - available
        if missing:
            raise ValueError(f"Terminal-hold loss exclusions have no matching GT subtask: {sorted(missing)}")

    def _apply_subtask_action_loss_mask(self, result: dict, text: str) -> None:
        excluded = getattr(self, "exclude_terminal_hold_loss_subtasks", ())
        if excluded:
            # Keep the fixed-length carrier, real pauses, and segment boundary.
            # A separate mask removes only artificial hold supervision; the
            # action_is_pad denominator remains 32 even for a one-step suffix.
            result["action_loss_mask"] = (
                result["action_real_mask"].clone()
                if " ".join(text.casefold().split()) in excluded
                else torch.ones_like(result["action_real_mask"])
            )

    def _aligned_anchor_indices(self) -> list[int]:
        # All GT frames are admitted, including short boundary suffixes. Do not
        # filter them through V72's predicted-condition/GT agreement mask.
        indices = self.lerobot_dataset.episode_data_index
        return [
            anchor
            for start, stop in zip(indices["from"].tolist(), indices["to"].tolist())
            for anchor in range(int(start), int(stop), self.trajectory_anchor_stride)
        ]

    @property
    def sampling_contract(self) -> str:
        digest = hashlib.sha256()
        for episode in sorted(self._gt_stage_index):
            digest.update(str(episode).encode("utf-8"))
            digest.update(np.asarray(self._gt_stage_index[episode], dtype=np.int32).tobytes())
            digest.update(json.dumps(self._gt_texts_by_episode[episode]).encode("utf-8"))
        return (
            f"gt_subtask_currentonly_stride{self.trajectory_anchor_stride}_"
            f"terminalhold_frame_shuffle_without_replacement_{digest.hexdigest()[:16]}_v1"
        )

    def _merge_cameras(self, video):
        return ProgressSubtaskRobotVideoDataset._merge_cameras(self, video)

    def _get(self, idx):
        result = super()._get(idx)
        episode = int(result["episode_index"])
        frame = int(result["frame_index"])
        stage = int(self._gt_stage_index[episode][frame])
        text = self._gt_texts_by_episode[episode][stage]
        ProgressSubtaskRobotVideoDataset._apply_subtask_terminal_hold(self, result)
        self._apply_subtask_action_loss_mask(result, text)
        result["context"] = self._gt_contexts[text].clone()
        # Original FastWAM leaves zeroed T5 padding visible to cross-attention.
        result["context_mask"] = torch.ones(self.context_len, dtype=torch.bool)
        result["prompt"] = DEFAULT_PROMPT.format(task=text)
        result["current_subtask"] = text
        result["gt_subtask_index"] = torch.tensor(stage, dtype=torch.long)
        result["gt_subtask_end_frame"] = torch.tensor(
            int(self._subtask_segment_end[episode][frame]), dtype=torch.long
        )
        return result
