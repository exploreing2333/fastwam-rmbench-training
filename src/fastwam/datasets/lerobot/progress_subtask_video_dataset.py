from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from fastwam_qwen.datasets.robot_video_dataset import RobotVideoDatasetNoT5


class ProgressSubtaskRobotVideoDataset(RobotVideoDatasetNoT5):
    """FastWAM RGB/action path with cached V72 subtask/progress conditions."""

    def __init__(
        self,
        *args,
        subtask_trajectory_mode: bool = False,
        subtask_source_map: str | None = None,
        subtask_terminal_hold: bool = False,
        subtask_segment_source: str = "gt_annotation",
        progress_conditioning_enabled: bool = False,
        progress_clean_probability: float = 0.5,
        progress_shift_probability: float = 0.2,
        progress_dropout_probability: float = 0.3,
        progress_shift_min_bins: float = 0.25,
        progress_shift_max_bins: float = 0.75,
        task_sampling_exponent: float = 1.0,
        event_anchor_latent_cache_root: str | None = None,
        event_anchor_max_frames: int = 4,
        event_anchor_cache_max_open_episodes: int = 8,
        event_anchor_batch_grouping: bool = False,
        event_anchor_candidates_enabled: bool = False,
        event_anchor_microbatch_shuffle: bool = False,
        full_clip_latent_cache_root: str | None = None,
        **kwargs,
    ) -> None:
        """Optionally treat every deployed/annotated subtask as a trajectory.

        One FastWAM text context conditions the complete 32-step target.  If
        that target crosses a subtask boundary, the next-stage action is
        otherwise learned under the previous subtask.  Subtask-trajectory mode
        retains legal anchors and replaces only the crossed suffix with a
        terminal hold.
        """
        self.subtask_trajectory_mode = bool(subtask_trajectory_mode)
        self.subtask_terminal_hold = bool(subtask_terminal_hold)
        self.subtask_segment_source = str(subtask_segment_source).lower()
        self.progress_conditioning_enabled = bool(progress_conditioning_enabled)
        self.progress_clean_probability = float(progress_clean_probability)
        self.progress_shift_probability = float(progress_shift_probability)
        self.progress_dropout_probability = float(progress_dropout_probability)
        self.progress_shift_min_bins = float(progress_shift_min_bins)
        self.progress_shift_max_bins = float(progress_shift_max_bins)
        self.task_sampling_exponent = float(task_sampling_exponent)
        self.event_anchor_latent_cache_root = event_anchor_latent_cache_root
        self.event_anchor_max_frames = int(event_anchor_max_frames)
        self.event_anchor_cache_max_open_episodes = int(
            event_anchor_cache_max_open_episodes
        )
        self.event_anchor_batch_grouping = bool(event_anchor_batch_grouping)
        self.event_anchor_candidates_enabled = bool(event_anchor_candidates_enabled)
        self.event_anchor_microbatch_shuffle = bool(event_anchor_microbatch_shuffle)
        if self.event_anchor_microbatch_shuffle and not (
            self.event_anchor_batch_grouping and self.subtask_trajectory_mode
            and self.task_sampling_exponent != 1.0
        ):
            raise ValueError("Microbatch shuffling requires task-balanced event/subtask grouping.")
        if self.event_anchor_max_frames <= 0:
            raise ValueError(
                "event_anchor_max_frames must be positive, got "
                f"{event_anchor_max_frames}."
            )
        if self.task_sampling_exponent < 0.0:
            raise ValueError(
                "task_sampling_exponent must be non-negative, got "
                f"{task_sampling_exponent}."
            )
        probabilities = (
            self.progress_clean_probability,
            self.progress_shift_probability,
            self.progress_dropout_probability,
        )
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError(
                "Progress augmentation probabilities must be non-negative."
            )
        if not np.isclose(sum(probabilities), 1.0, atol=1e-8):
            raise ValueError(
                "Progress augmentation probabilities must sum to one, got "
                f"{probabilities}."
            )
        if not 0.0 <= self.progress_shift_min_bins <= self.progress_shift_max_bins:
            raise ValueError(
                "Progress shift range must satisfy 0 <= min <= max, got "
                f"[{self.progress_shift_min_bins}, {self.progress_shift_max_bins}]."
            )
        if self.subtask_segment_source not in {
            "gt_annotation",
            "progress_condition_index",
        }:
            raise ValueError(
                "subtask_segment_source must be 'gt_annotation' or "
                f"'progress_condition_index', got {subtask_segment_source!r}."
            )
        self.subtask_source_map = (
            None
            if subtask_source_map in (None, "")
            else Path(str(subtask_source_map)).expanduser().resolve()
        )
        if self.subtask_terminal_hold and not self.subtask_trajectory_mode:
            raise ValueError(
                "subtask_terminal_hold requires subtask_trajectory_mode=true."
            )
        if (
            self.subtask_trajectory_mode
            and self.subtask_segment_source == "gt_annotation"
            and self.subtask_source_map is None
        ):
            raise ValueError(
                "GT-annotation subtask trajectories require subtask_source_map."
            )
        if self.subtask_trajectory_mode:
            if str(kwargs.get("training_shuffle_unit", "frame")) != "trajectory":
                raise ValueError(
                    "subtask_trajectory_mode requires "
                    "training_shuffle_unit='trajectory'."
                )
            require_gt_exact = bool(
                kwargs.get("progress_require_gt_subtask_exact", False)
            )
            if self.subtask_segment_source == "gt_annotation" and not require_gt_exact:
                raise ValueError(
                    "GT-annotation subtask trajectories require "
                    "progress_require_gt_subtask_exact=true."
                )
            if (
                self.subtask_segment_source == "progress_condition_index"
                and require_gt_exact
            ):
                raise ValueError(
                    "Progress-condition subtask trajectories retain the exact "
                    "deployed V72 stream and therefore require "
                    "progress_require_gt_subtask_exact=false."
                )

        super().__init__(*args, **kwargs)
        self.event_anchor_latent_store = None
        if self.event_anchor_latent_cache_root:
            from fastwam.datasets.lerobot.event_anchor_cache import (
                EventAnchorLatentStore,
            )

            self.event_anchor_latent_store = EventAnchorLatentStore(
                self.event_anchor_latent_cache_root,
                max_anchors=self.event_anchor_max_frames,
                max_cached_episodes=self.event_anchor_cache_max_open_episodes,
                include_candidates=self.event_anchor_candidates_enabled,
            )
        self._gt_stage_index: dict[int, np.ndarray] = {}
        self._subtask_segment_index: dict[int, np.ndarray] = {}
        self._subtask_segment_end: dict[int, np.ndarray] = {}
        self._task_sampling_anchor_groups: dict[str, np.ndarray] | None = None
        self._task_sampling_segment_groups: dict[
            str, dict[tuple[int, int], np.ndarray]
        ] | None = None
        self._task_sampling_source_by_episode: dict[int, str] = {}
        if self.task_sampling_exponent != 1.0:
            if self.subtask_source_map is None:
                raise ValueError(
                    "Task-balanced sampling requires subtask_source_map so "
                    "each anchor can be assigned to its RMBench task."
                )
            source_map = json.loads(
                self.subtask_source_map.read_text(encoding="utf-8")
            )
            self._task_sampling_source_by_episode = {
                int(row["episode_index"]): str(row["source_task_name"])
                for row in source_map["entries"]
            }
        if self.subtask_trajectory_mode:
            if self.subtask_segment_source == "gt_annotation":
                self._load_gt_subtask_trajectories()
            else:
                self._load_progress_condition_trajectories()

        self.full_clip_latent_store = None
        if full_clip_latent_cache_root:
            from .full_clip_latent_cache import FullClipLatentStore, clip_contract
            self.full_clip_latent_store = FullClipLatentStore(
                full_clip_latent_cache_root, expected_contract=clip_contract(self))
            # Reuse the existing metadata-only RGB transport after validating
            # our own full-clip cache; do not pretend it is a TMD latent cache.
            self.skip_rgb_when_precomputed_latents = True
            self.lerobot_dataset._set_return_images(False)
            self.lerobot_dataset.processor.allow_missing_images = True
            self.lerobot_dataset.processor.image_output_layout = "robotwin"

    @staticmethod
    def _shift_progress_probabilities(
        probabilities: torch.Tensor,
        offset_bins: float,
    ) -> torch.Tensor:
        """Translate a categorical distribution without circular wraparound."""
        if probabilities.ndim != 1 or probabilities.numel() <= 1:
            raise ValueError(
                "Progress probabilities must be a one-dimensional multi-bin tensor."
            )
        output = torch.zeros_like(probabilities)
        positions = (
            torch.arange(
                probabilities.numel(),
                dtype=torch.float32,
                device=probabilities.device,
            )
            + float(offset_bins)
        ).clamp(0.0, float(probabilities.numel() - 1))
        lower = positions.floor().long()
        upper = positions.ceil().long()
        upper_weight = positions - lower.to(dtype=positions.dtype)
        output.scatter_add_(0, lower, probabilities * (1.0 - upper_weight))
        output.scatter_add_(0, upper, probabilities * upper_weight)
        return output / output.sum().clamp_min(torch.finfo(output.dtype).eps)

    def _augment_progress_logits(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.ndim != 1:
            raise ValueError(
                f"Progress logits must be one-dimensional, got {tuple(logits.shape)}."
            )
        probabilities = torch.softmax(logits.float(), dim=0)
        draw = float(torch.rand(()).item())
        clean_boundary = self.progress_clean_probability
        shift_boundary = clean_boundary + self.progress_shift_probability
        present = torch.tensor(True, dtype=torch.bool)
        shift_bins = torch.tensor(0.0, dtype=torch.float32)
        if draw < clean_boundary:
            mode = 0  # original V72 distribution
        elif draw < shift_boundary:
            mode = 1  # symmetric sub-bin translation
            magnitude = self.progress_shift_min_bins + float(
                torch.rand(()).item()
            ) * (self.progress_shift_max_bins - self.progress_shift_min_bins)
            direction = -1.0 if float(torch.rand(()).item()) < 0.5 else 1.0
            shift_bins = torch.tensor(direction * magnitude, dtype=torch.float32)
            probabilities = self._shift_progress_probabilities(
                probabilities,
                float(shift_bins.item()),
            )
        else:
            mode = 2  # exact dropout: restore the subtask-only baseline token
            present = torch.tensor(False, dtype=torch.bool)
            probabilities = torch.zeros_like(probabilities)
        return (
            probabilities,
            present,
            torch.tensor(mode, dtype=torch.long),
            shift_bins,
        )

    def _load_progress_condition_trajectories(self) -> None:
        """Use the deployed V72 condition stream as the action-policy truth."""
        if self.progress_feature_store is None:
            raise ValueError(
                "progress_condition_index segmentation requires a Progress cache."
            )
        episode_from = self.lerobot_dataset.episode_data_index["from"].tolist()
        episode_to = self.lerobot_dataset.episode_data_index["to"].tolist()
        selected_episodes = (
            tuple(range(len(episode_from)))
            if self.episode_indices is None
            else self.episode_indices
        )
        if len(selected_episodes) != len(episode_from):
            raise RuntimeError(
                "Selected episode metadata does not match LeRobot trajectories: "
                f"episodes={len(selected_episodes)}, trajectories={len(episode_from)}."
            )

        for trajectory_id, episode_index in enumerate(selected_episodes):
            length = int(episode_to[trajectory_id]) - int(episode_from[trajectory_id])
            # StateProgressFeatureStore owns the validated SWMR H5 handles.
            handle = self.progress_feature_store._open(int(episode_index))
            condition_index = np.asarray(
                handle["condition_index"][...], dtype=np.int64
            )
            if tuple(condition_index.shape) != (length,):
                raise RuntimeError(
                    f"Episode {episode_index} condition_index has shape "
                    f"{condition_index.shape}, expected {(length,)}."
                )
            if np.any(condition_index < 0):
                missing = np.flatnonzero(condition_index < 0)[:10].tolist()
                raise RuntimeError(
                    f"Episode {episode_index} has missing deployed V72 conditions: "
                    f"{missing}."
                )

            starts = np.r_[
                0,
                np.flatnonzero(condition_index[1:] != condition_index[:-1]) + 1,
            ]
            stops = np.r_[starts[1:], length]
            segment_end = np.empty(length, dtype=np.int32)
            segment_index = np.empty(length, dtype=np.int32)
            previous_text = None
            for segment_id, (start, stop) in enumerate(zip(starts.tolist(), stops.tolist())):
                condition_id = int(condition_index[start])
                condition_name = f"{condition_id:06d}"
                if condition_name not in handle["conditions"]:
                    raise KeyError(
                        f"Episode {episode_index} lacks V72 condition "
                        f"{condition_name}."
                    )
                text = str(
                    handle["conditions"][condition_name].attrs.get(
                        "deployed_subtask", ""
                    )
                ).strip()
                if not text:
                    raise RuntimeError(
                        f"Episode {episode_index} condition {condition_name} "
                        "has no deployed_subtask."
                    )
                if text == previous_text:
                    raise RuntimeError(
                        "V72 condition_index split one unchanged deployed "
                        f"subtask in episode {episode_index} at frame {start}."
                    )
                previous_text = text
                segment_end[start:stop] = stop - 1
                segment_index[start:stop] = int(segment_id)
            self._subtask_segment_end[int(episode_index)] = segment_end
            self._subtask_segment_index[int(episode_index)] = segment_index

    def _load_gt_subtask_trajectories(self) -> None:
        """Load authoritative RMBench language segments for selected episodes."""
        assert self.subtask_source_map is not None
        if not self.subtask_source_map.is_file():
            raise FileNotFoundError(
                f"RMBench source map is missing: {self.subtask_source_map}"
            )
        source_map = json.loads(self.subtask_source_map.read_text(encoding="utf-8"))
        source_by_episode = {
            int(row["episode_index"]): row for row in source_map["entries"]
        }
        annotation_root = Path(str(source_map["raw_root"])).expanduser().resolve()
        annotations: dict[str, dict] = {}

        episode_from = self.lerobot_dataset.episode_data_index["from"].tolist()
        episode_to = self.lerobot_dataset.episode_data_index["to"].tolist()
        selected_episodes = (
            tuple(range(len(episode_from)))
            if self.episode_indices is None
            else self.episode_indices
        )
        if len(selected_episodes) != len(episode_from):
            raise RuntimeError(
                "Selected episode metadata does not match LeRobot trajectories: "
                f"episodes={len(selected_episodes)}, trajectories={len(episode_from)}."
            )

        for trajectory_id, episode_index in enumerate(selected_episodes):
            if int(episode_index) not in source_by_episode:
                raise KeyError(
                    f"Episode {episode_index} is absent from {self.subtask_source_map}."
                )
            source = source_by_episode[int(episode_index)]
            task_name = str(source["source_task_name"])
            source_episode_id = int(source["source_episode_id"])
            length = int(episode_to[trajectory_id]) - int(episode_from[trajectory_id])
            if length != int(source["length"]):
                raise RuntimeError(
                    f"Episode {episode_index} length mismatch: "
                    f"LeRobot={length}, source_map={source['length']}."
                )
            if task_name not in annotations:
                annotation_path = (
                    annotation_root
                    / task_name
                    / "demo_clean"
                    / "language_annotation.json"
                )
                if not annotation_path.is_file():
                    raise FileNotFoundError(
                        f"RMBench GT annotation is missing: {annotation_path}"
                    )
                annotations[task_name] = json.loads(
                    annotation_path.read_text(encoding="utf-8")
                )
            raw_segments = annotations[task_name][f"episode_{source_episode_id}"]
            annotated_length = sum(int(row[1]) for row in raw_segments)
            if annotated_length != int(source["raw_length"]):
                raise RuntimeError(
                    f"Episode {episode_index} annotation length mismatch: "
                    f"annotation={annotated_length}, raw={source['raw_length']}."
                )

            stage_index = np.full(length, -1, dtype=np.int32)
            segment_end = np.full(length, -1, dtype=np.int32)
            cursor = 0
            for gt_index, raw_segment in enumerate(raw_segments):
                if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) < 2:
                    raise ValueError(
                        f"Malformed GT subtask segment for episode {episode_index}: "
                        f"{raw_segment!r}"
                    )
                subtask, duration = str(raw_segment[0]).strip(), int(raw_segment[1])
                if not subtask or duration <= 0:
                    raise ValueError(
                        f"Invalid GT subtask segment for episode {episode_index}: "
                        f"{raw_segment!r}"
                    )
                start, stop = cursor, cursor + duration
                cursor = stop
                clipped_start, clipped_stop = max(0, start), min(length, stop)
                if clipped_start < clipped_stop:
                    stage_index[clipped_start:clipped_stop] = gt_index
                    segment_end[clipped_start:clipped_stop] = clipped_stop - 1
            if np.any(stage_index < 0) or np.any(segment_end < 0):
                missing = np.flatnonzero(stage_index < 0)[:10].tolist()
                raise RuntimeError(
                    f"GT subtask segments do not cover episode {episode_index}: "
                    f"missing={missing}."
                )
            self._gt_stage_index[int(episode_index)] = stage_index
            self._subtask_segment_end[int(episode_index)] = segment_end
            self._subtask_segment_index[int(episode_index)] = stage_index

    def _aligned_anchor_indices(self) -> list[int]:
        if not self.subtask_trajectory_mode:
            return super()._aligned_anchor_indices()
        episode_from = self.lerobot_dataset.episode_data_index["from"].tolist()
        episode_to = self.lerobot_dataset.episode_data_index["to"].tolist()
        selected_episodes = (
            tuple(range(len(episode_from)))
            if self.episode_indices is None
            else self.episode_indices
        )
        anchors: list[int] = []
        for trajectory_id, episode_index in enumerate(selected_episodes):
            start = int(episode_from[trajectory_id])
            end = int(episode_to[trajectory_id])
            if self.subtask_segment_source == "progress_condition_index":
                # The deployed V72 stream is the train/inference contract, so
                # keep every frame including its slightly early/late switches.
                local = np.arange(
                    0,
                    end - start,
                    self.trajectory_anchor_stride,
                    dtype=np.int64,
                )
            else:
                valid = self.progress_feature_store.load_training_valid_mask(
                    int(episode_index)
                )
                if tuple(valid.shape) != (end - start,):
                    raise RuntimeError(
                        f"Episode {episode_index} GT-exact mask has shape "
                        f"{valid.shape}, expected {(end - start,)}."
                    )
                local = np.flatnonzero(valid)[:: self.trajectory_anchor_stride]
            anchors.extend((local + start).tolist())
        if not anchors:
            raise RuntimeError("No subtask-trajectory anchors were found.")
        return anchors

    def _get_task_sampling_anchor_groups(
        self,
        anchors: list[int],
    ) -> dict[str, np.ndarray]:
        if self._task_sampling_anchor_groups is not None:
            return self._task_sampling_anchor_groups
        episode_from = np.asarray(
            self.lerobot_dataset.episode_data_index["from"].tolist(),
            dtype=np.int64,
        )
        episode_to = np.asarray(
            self.lerobot_dataset.episode_data_index["to"].tolist(),
            dtype=np.int64,
        )
        selected_episodes = (
            tuple(range(len(episode_from)))
            if self.episode_indices is None
            else self.episode_indices
        )
        anchor_array = np.asarray(anchors, dtype=np.int64)
        trajectory_ids = np.searchsorted(episode_to, anchor_array, side="right")
        groups: dict[str, list[int]] = {}
        for anchor, trajectory_id in zip(anchor_array.tolist(), trajectory_ids.tolist()):
            if not 0 <= int(trajectory_id) < len(selected_episodes):
                raise RuntimeError(
                    "Unable to map aligned anchor to an episode: "
                    f"anchor={anchor}, trajectory_id={trajectory_id}."
                )
            episode_index = int(selected_episodes[int(trajectory_id)])
            task_name = self._task_sampling_source_by_episode.get(episode_index)
            if task_name is None:
                raise KeyError(
                    f"Episode {episode_index} has no source task in "
                    f"{self.subtask_source_map}."
                )
            groups.setdefault(task_name, []).append(int(anchor))
        self._task_sampling_anchor_groups = {
            task: np.asarray(values, dtype=np.int64)
            for task, values in groups.items()
        }
        return self._task_sampling_anchor_groups

    def shuffled_epoch_indices(self, generator: torch.Generator) -> list[int]:
        if self.event_anchor_batch_grouping and self.subtask_trajectory_mode:
            anchors = self._aligned_anchor_indices()
            if self.task_sampling_exponent != 1.0:
                return self._shuffled_task_balanced_segment_indices(
                    anchors=anchors,
                    generator=generator,
                )
            episode_from = self.lerobot_dataset.episode_data_index["from"].tolist()
            episode_to = self.lerobot_dataset.episode_data_index["to"].tolist()
            selected_episodes = (
                tuple(range(len(episode_from)))
                if self.episode_indices is None
                else self.episode_indices
            )
            anchor_array = np.asarray(anchors, dtype=np.int64)
            trajectory_ids = np.searchsorted(
                np.asarray(episode_to, dtype=np.int64), anchor_array, side="right"
            )
            groups: dict[tuple[int, int], list[int]] = {}
            for anchor, trajectory_id in zip(anchor_array.tolist(), trajectory_ids.tolist()):
                episode_index = int(selected_episodes[int(trajectory_id)])
                local_frame = int(anchor - int(episode_from[int(trajectory_id)]))
                segment_id = int(
                    self._subtask_segment_index[episode_index][local_frame]
                )
                groups.setdefault((episode_index, segment_id), []).append(int(anchor))
            group_keys = list(groups)
            if group_keys:
                order = torch.randperm(len(group_keys), generator=generator).tolist()
                output: list[int] = []
                for position in order:
                    values = groups[group_keys[position]]
                    value_order = torch.randperm(len(values), generator=generator).tolist()
                    output.extend(values[index] for index in value_order)
                return output
        if self.task_sampling_exponent == 1.0:
            return super().shuffled_epoch_indices(generator)
        anchors = self._aligned_anchor_indices()
        groups = self._get_task_sampling_anchor_groups(anchors)
        # Give every anchor in task t weight N_t^(alpha-1), so the total
        # probability mass of task t is proportional to N_t^alpha.
        weighted = torch.zeros(len(anchors), dtype=torch.float64)
        anchor_position = {int(anchor): i for i, anchor in enumerate(anchors)}
        for values in groups.values():
            count = float(len(values))
            per_anchor_weight = count ** (self.task_sampling_exponent - 1.0)
            positions = [anchor_position[int(anchor)] for anchor in values]
            weighted[positions] = per_anchor_weight
        sampled_positions = torch.multinomial(
            weighted,
            num_samples=len(anchors),
            replacement=True,
            generator=generator,
        )
        return [anchors[int(position)] for position in sampled_positions.tolist()]

    def _get_task_sampling_segment_groups(
        self,
        anchors: list[int],
    ) -> dict[str, dict[tuple[int, int], np.ndarray]]:
        """Index anchors by task and V72 subtask segment.

        The task-balancing path must preserve the segment ordering contract:
        anchors from different subtasks may not be interleaved merely because
        they belong to the same task.  The returned nested mapping is cached
        because it is invariant across epochs; only the sampling order changes.
        """
        if self._task_sampling_segment_groups is not None:
            return self._task_sampling_segment_groups
        if not self._task_sampling_source_by_episode:
            raise RuntimeError(
                "Task-balanced segment sampling requires an episode-to-task map."
            )
        episode_from = np.asarray(
            self.lerobot_dataset.episode_data_index["from"].tolist(),
            dtype=np.int64,
        )
        episode_to = np.asarray(
            self.lerobot_dataset.episode_data_index["to"].tolist(),
            dtype=np.int64,
        )
        selected_episodes = (
            tuple(range(len(episode_from)))
            if self.episode_indices is None
            else self.episode_indices
        )
        anchor_array = np.asarray(anchors, dtype=np.int64)
        trajectory_ids = np.searchsorted(episode_to, anchor_array, side="right")
        grouped: dict[str, dict[tuple[int, int], list[int]]] = {}
        for anchor, trajectory_id in zip(
            anchor_array.tolist(), trajectory_ids.tolist()
        ):
            trajectory_id = int(trajectory_id)
            episode_index = int(selected_episodes[trajectory_id])
            local_frame = int(anchor - episode_from[trajectory_id])
            segment_id = int(
                self._subtask_segment_index[episode_index][local_frame]
            )
            task_name = self._task_sampling_source_by_episode[episode_index]
            grouped.setdefault(task_name, {}).setdefault(
                (episode_index, segment_id), []
            ).append(int(anchor))
        self._task_sampling_segment_groups = {
            task_name: {
                key: np.asarray(values, dtype=np.int64)
                for key, values in segment_groups.items()
            }
            for task_name, segment_groups in grouped.items()
        }
        return self._task_sampling_segment_groups

    @staticmethod
    def _balanced_task_counts(
        task_counts: dict[str, int],
        total_samples: int,
        exponent: float,
    ) -> dict[str, int]:
        """Allocate an epoch budget with mass proportional to N_task**exponent."""
        if not task_counts:
            raise RuntimeError("Cannot balance an empty task population.")
        names = sorted(task_counts)
        counts = np.asarray([float(task_counts[name]) for name in names])
        if np.any(counts <= 0):
            raise RuntimeError(f"Every task must have positive anchors, got {task_counts}.")
        weights = np.power(counts, float(exponent))
        raw = float(total_samples) * weights / weights.sum()
        targets = np.floor(raw).astype(np.int64)
        remainder = int(total_samples - int(targets.sum()))
        if remainder > 0:
            fractional_order = np.argsort(-(raw - targets))
            for position in fractional_order[:remainder]:
                targets[int(position)] += 1
        return {name: int(target) for name, target in zip(names, targets.tolist())}

    def _shuffled_task_balanced_segment_indices(
        self,
        *,
        anchors: list[int],
        generator: torch.Generator,
    ) -> list[int]:
        """Sample a task-balanced epoch while retaining contiguous segments.

        Sampling is without replacement until a task's allocated quota exceeds
        its population; only then are complete task permutations repeated.  The
        result is still grouped by subtask segment, so event KV prefill is never
        shared across a segment boundary.
        """
        segment_groups = self._get_task_sampling_segment_groups(anchors)
        task_counts = {
            task_name: int(sum(values.size for values in groups.values()))
            for task_name, groups in segment_groups.items()
        }
        target_counts = self._balanced_task_counts(
            task_counts=task_counts,
            total_samples=len(anchors),
            exponent=self.task_sampling_exponent,
        )

        task_samples: dict[str, list[int]] = {}
        for task_name in sorted(segment_groups):
            values = np.concatenate(
                list(segment_groups[task_name].values()), axis=0
            )
            target = int(target_counts[task_name])
            selected: list[int] = []
            remaining = target
            while remaining > 0:
                order = torch.randperm(len(values), generator=generator).tolist()
                take = min(remaining, len(order))
                selected.extend(int(values[position]) for position in order[:take])
                remaining -= take
            task_samples[task_name] = selected

        # Restore segment locality after task-level sampling.  Duplicated
        # anchors (needed when oversampling a short task) stay in their segment.
        anchor_to_segment: dict[int, tuple[int, int]] = {}
        for task_name, groups in segment_groups.items():
            for segment_key, values in groups.items():
                for anchor in values.tolist():
                    anchor_to_segment[int(anchor)] = segment_key

        output_groups: list[list[int]] = []
        for task_name, selected in task_samples.items():
            grouped_selected: dict[tuple[int, int], list[int]] = {}
            for anchor in selected:
                grouped_selected.setdefault(anchor_to_segment[anchor], []).append(anchor)
            group_keys = list(grouped_selected)
            group_order = torch.randperm(len(group_keys), generator=generator).tolist()
            for position in group_order:
                values = grouped_selected[group_keys[position]]
                value_order = torch.randperm(len(values), generator=generator).tolist()
                output_groups.append([values[index] for index in value_order])

        group_order = torch.randperm(len(output_groups), generator=generator).tolist()
        output: list[int] = []
        for position in group_order:
            output.extend(output_groups[position])
        if len(output) != len(anchors):
            raise RuntimeError(
                "Task-balanced segment sampler produced the wrong epoch size: "
                f"{len(output)} != {len(anchors)}."
            )
        return output

    def reorder_event_microbatches(self, indices, *, generator, batch_size, num_processes):
        from fastwam.utils.event_microbatch_sampler import shuffle_event_microbatches
        return shuffle_event_microbatches(
            indices, segment_groups=self._get_task_sampling_segment_groups(
                self._aligned_anchor_indices()
            ), batch_size=batch_size, num_processes=num_processes, generator=generator,
        )

    @property
    def sampling_contract(self) -> str:
        if not self.subtask_trajectory_mode:
            return super().sampling_contract
        epoch_sampling = (
            "without_replacement"
            if self.task_sampling_exponent == 1.0
            else f"taskpower{self.task_sampling_exponent:g}_replacement"
        )
        contract = (
            "subtask_trajectory_"
            f"stride{self.trajectory_anchor_stride}_"
            f"segments{self.subtask_segment_source}_"
            f"terminalhold{int(self.subtask_terminal_hold)}_"
            f"{epoch_sampling}_segmentgroup_v3"
        )
        if self.event_anchor_microbatch_shuffle:
            contract += "_microbatchshuffle_v4"
        return contract

    @property
    def requires_unique_epoch_indices(self) -> bool:
        return self.task_sampling_exponent == 1.0

    def _apply_subtask_terminal_hold(self, result: dict) -> None:
        episode_index = int(torch.as_tensor(result["episode_index"]).item())
        frame_index = int(torch.as_tensor(result["frame_index"]).item())
        if episode_index not in self._subtask_segment_end:
            raise KeyError(f"No subtask trajectory for episode {episode_index}.")
        end = int(self._subtask_segment_end[episode_index][frame_index])
        remaining = end - frame_index + 1
        if remaining <= 0:
            raise RuntimeError(
                f"Invalid subtask remainder for episode={episode_index}, "
                f"frame={frame_index}, end={end}."
            )

        action = result["action"]
        action_is_pad = result["action_is_pad"]
        horizon = int(action.shape[0])
        valid_action_steps = min(remaining, horizon)
        result["action_real_mask"] = torch.arange(horizon) < valid_action_steps
        if valid_action_steps < horizon:
            # Absolute-qpos FastWAM: repeating the last same-subtask command is
            # a terminal hold.  Keep it loss-visible so the generated suffix
            # cannot learn an arbitrary next-subtask action.
            hold_action = action[valid_action_steps - 1].clone()
            action[valid_action_steps:] = hold_action
            action_is_pad[valid_action_steps:] = False

        video = result["video"]
        image_is_pad = result["image_is_pad"]
        video_offsets = [
            index - self.current_obs_index for index in self.video_sample_indices
        ]
        if len(video_offsets) != int(video.shape[1]):
            raise RuntimeError(
                "FastWAM video offsets do not match the processed video: "
                f"offsets={video_offsets}, shape={tuple(video.shape)}."
            )
        last_same_subtask = 0
        for video_index, offset in enumerate(video_offsets):
            if frame_index + int(offset) <= end:
                last_same_subtask = video_index
                continue
            # Repeat only a frame already sampled from this subtask segment. This
            # avoids next-subtask pixels without adding an extra image decode.
            video[:, video_index] = video[:, last_same_subtask]
            image_is_pad[video_index] = False

    def _merge_cameras(self, video: torch.Tensor) -> torch.Tensor:
        """Reproduce FastWAM's released RoboTwin three-camera mosaic.

        Layout: the 256x320 high camera is on top, with the two 128x160 wrist
        cameras side by side underneath it.  This deliberately does not use
        the later TMD layout, which places the wrist block above the high view.
        """
        num_cameras, _, _, _, _ = video.shape
        if self.concat_multi_camera != "robotwin":
            return super()._merge_cameras(video)
        if num_cameras != 3:
            raise ValueError(
                "FastWAM RoboTwin layout requires exactly three cameras, got "
                f"{num_cameras}."
            )
        cam_high = transforms_F.resize(
            video[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_left = transforms_F.resize(
            video[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        cam_right = transforms_F.resize(
            video[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        wrist_block = torch.cat([cam_left, cam_right], dim=-1)
        return torch.cat([cam_high, wrist_block], dim=-2).contiguous()

    def _get(self, idx):
        result = super()._get(idx)
        if self.event_anchor_latent_store is not None:
            episode_index = int(torch.as_tensor(result["episode_index"]).item())
            frame_index = int(torch.as_tensor(result["frame_index"]).item())
            result.update(
                self.event_anchor_latent_store.load(
                    episode_index,
                    frame_index,
                )
            )
            if episode_index in self._subtask_segment_index:
                segment_id = int(self._subtask_segment_index[episode_index][frame_index])
                # A globally unique integer key lets the model share an event
                # cache only inside one episode/subtask segment.
                result["event_anchor_group_id"] = torch.tensor(
                    episode_index * 1_000_000 + segment_id,
                    dtype=torch.long,
                )
        if self.subtask_terminal_hold:
            self._apply_subtask_terminal_hold(result)
        if "progress_subtask_tokens" not in result:
            raise KeyError(
                "Progress-subtask FastWAM training requires a progress cache "
                "with decoded subtask T5 embeddings."
            )

        subtask = result["progress_subtask_tokens"]
        subtask_mask = result["progress_subtask_mask"].bool()
        if subtask.ndim != 2 or int(subtask.shape[-1]) != 4096:
            raise ValueError(
                "Progress subtask context must be [L,4096], got "
                f"{tuple(subtask.shape)}."
            )
        if tuple(subtask_mask.shape) != (int(subtask.shape[0]),):
            raise ValueError(
                "Progress subtask mask must match its token length, got "
                f"context={tuple(subtask.shape)}, mask={tuple(subtask_mask.shape)}."
            )
        if int(subtask.shape[0]) > self.context_len:
            raise ValueError(
                f"Subtask has {subtask.shape[0]} tokens, exceeding "
                f"context_len={self.context_len}."
            )

        # Match original FastWAM/Wan text conditioning exactly: right-pad the
        # cached T5 context to 128 tokens, zero its padding, then leave those
        # zero tokens visible to cross-attention.
        context = torch.zeros(
            self.context_len,
            4096,
            dtype=subtask.dtype,
        )
        context[: subtask.shape[0]] = subtask
        context[: subtask.shape[0]][~subtask_mask] = 0
        result["context"] = context
        result["context_mask"] = torch.ones(
            self.context_len,
            dtype=torch.bool,
        )

        if self.progress_conditioning_enabled:
            if "progress_logits" not in result:
                raise KeyError(
                    "Progress conditioning requires cached `progress_logits`."
                )
            (
                result["wam_progress_probabilities"],
                result["wam_progress_present"],
                result["wam_progress_augmentation"],
                result["wam_progress_shift_bins"],
            ) = self._augment_progress_logits(result["progress_logits"])

        # Remove the heavyweight TMD side channels after retaining only the
        # explicitly audited WAM progress distribution above.
        for key in tuple(result):
            if key.startswith("progress_"):
                result.pop(key)
        if getattr(self, "full_clip_latent_store", None) is not None:
            result["fastwam_video_latents"] = self.full_clip_latent_store.load(
                int(result["episode_index"]), int(result["frame_index"]))
            result["fastwam_video_shape"] = torch.tensor(
                [3, len(self.video_sample_indices), *self.video_size], dtype=torch.long)
        return result
