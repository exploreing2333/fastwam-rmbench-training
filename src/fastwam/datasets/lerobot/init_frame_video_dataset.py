"""Original GT subtask clips with the complete episode-frame-zero visual anchor."""
from __future__ import annotations

from pathlib import Path

import torch

from fastwam.datasets.lerobot.event_anchor_cache import EventAnchorLatentStore
from fastwam.datasets.lerobot.gt_subtask_video_dataset import (
    GroundTruthSubtaskRobotVideoDataset,
)


class EpisodeInitFrameRobotVideoDataset(GroundTruthSubtaskRobotVideoDataset):
    def __init__(
        self,
        *args,
        event_anchor_latent_cache_root: str,
        event_anchor_max_frames: int = 1,
        event_anchor_batch_grouping: bool = True,
        event_anchor_cache_max_open_episodes: int = 8,
        init_frame_anchor_only: bool = True,
        **kwargs,
    ) -> None:
        if not init_frame_anchor_only or event_anchor_max_frames != 1:
            raise ValueError("Episode init-frame memory requires exactly frame zero.")
        if not event_anchor_batch_grouping:
            raise ValueError("Init-frame memory requires grouped cache training.")
        dataset_dirs = kwargs.get("dataset_dirs")
        if not dataset_dirs or len(dataset_dirs) != 1:
            raise ValueError("Init-frame cache requires exactly one dataset root.")
        dataset_root = Path(dataset_dirs[0]).resolve()
        self.event_anchor_latent_cache_root = str(Path(event_anchor_latent_cache_root).resolve())
        self.event_anchor_max_frames = 1
        self.event_anchor_batch_grouping = True
        self.event_anchor_microbatch_shuffle = False
        self.init_frame_anchor_only = True
        super().__init__(*args, **kwargs)
        self.init_anchor_store = EventAnchorLatentStore(
            self.event_anchor_latent_cache_root,
            max_anchors=1,
            max_cached_episodes=event_anchor_cache_max_open_episodes,
        )
        expected_episodes = set(map(int, self._gt_stage_index))
        cached_episodes = set(map(int, self.init_anchor_store.manifest["episode_indices"]))
        if not expected_episodes.issubset(cached_episodes):
            raise ValueError(f"Missing init anchors: {sorted(expected_episodes - cached_episodes)}")
        if self.init_anchor_store.manifest.get("anchor_policy") != "episode_frame_zero_only":
            raise ValueError("The configured cache does not contain fixed init anchors.")
        cached_root = Path(self.init_anchor_store.manifest["dataset_root"]).resolve()
        if dataset_root != cached_root:
            raise ValueError("Init-frame cache and training dataset must have the same root.")

    @property
    def sampling_contract(self) -> str:
        return super().sampling_contract + "_init_frame_anchor_v1"

    def _get(self, idx):
        result = super()._get(idx)
        episode = int(result["episode_index"])
        frame = int(result["frame_index"])
        anchor = self.init_anchor_store.load(episode, frame)
        if int(anchor["event_anchor_frame_indices"][0]) != 0:
            raise RuntimeError(f"Nonzero init anchor in episode {episode}.")
        result.update(anchor)
        # The video prefix is text conditioned, so it can be shared only
        # within the same episode and current GT subtask segment.
        stage = int(self._gt_stage_index[episode][frame])
        result["event_anchor_group_id"] = torch.tensor(
            episode * 1_000_000 + stage, dtype=torch.long
        )
        return result
