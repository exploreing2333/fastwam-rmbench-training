"""Original full task text with continuous episode clips and shuffled frame anchors."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from fastwam_qwen.datasets.robot_video_dataset import RobotVideoDatasetNoT5
from .event_anchor_cache import EventAnchorLatentStore
from .progress_subtask_video_dataset import ProgressSubtaskRobotVideoDataset


class FullInstructionRobotVideoDataset(RobotVideoDatasetNoT5):
    def __init__(self, *args, **kwargs):
        if not kwargs.get('text_embedding_cache_dir'):
            raise ValueError('Full instruction training requires the original Wan T5 instruction cache.')
        for key, expected in {
            'training_shuffle_unit': 'frame', 'history_num_frames': 1,
            'robometer_history_num_frames': 0, 'require_qwen_history': False,
            'skip_padding_as_possible': False, 'text_embedding_use_true_mask': False,
        }.items():
            if kwargs.get(key, expected) != expected:
                raise ValueError(f'Full instruction training requires {key}={expected!r}.')
            kwargs[key] = expected
        for key in ('progress_cache_root', 'progress_subtask_text_embedding_cache_dir'):
            if kwargs.get(key) not in (None, ''):
                raise ValueError(f'Full instruction training cannot read {key}.')
        super().__init__(*args, **kwargs)
        self._full_instruction_contexts = {}
        if self.current_obs_index != 0 or self.action_horizon != 32:
            raise ValueError('Expected the original current observation and 32-control action target.')

    def _get_cached_text_context(self, prompt):
        if prompt not in self._full_instruction_contexts:
            self._full_instruction_contexts[prompt] = super()._get_cached_text_context(prompt)
        return self._full_instruction_contexts[prompt]

    def _merge_cameras(self, video):
        return ProgressSubtaskRobotVideoDataset._merge_cameras(self, video)

    @property
    def sampling_contract(self):
        selection = repr((self.episode_indices, self.context_len)).encode()
        digest = hashlib.sha256(selection).hexdigest()[:16]
        return f'full_instruction_episode_clips_frame_without_replacement_{digest}_v1'

    @property
    def pad_to_global_batch(self):
        return True

    def __getitem__(self, idx):
        # Preserve exact epoch coverage: never replace a failed frame randomly.
        result = self._get(0 if int(idx) < 0 else int(idx))
        result['sample_weight'] = torch.tensor(0.0 if int(idx) < 0 else 1.0)
        return result


class EpisodeInitFrameFullInstructionRobotVideoDataset(FullInstructionRobotVideoDataset):
    """Keep the task's existing frame-zero visual KV with one full text per episode."""
    def __init__(self, *args, event_anchor_latent_cache_root,
                 event_anchor_max_frames=1, event_anchor_batch_grouping=True,
                 event_anchor_cache_max_open_episodes=8, init_frame_anchor_only=True,
                 **kwargs):
        if not init_frame_anchor_only or event_anchor_max_frames != 1 or not event_anchor_batch_grouping:
            raise ValueError('Init KV requires one trajectory-frame-zero anchor with grouped KV computation.')
        dataset_dirs = kwargs.get('dataset_dirs')
        if not dataset_dirs or len(dataset_dirs) != 1:
            raise ValueError('Init KV requires exactly one dataset root.')
        self.event_anchor_batch_grouping = True
        self.event_anchor_microbatch_shuffle = False
        self.event_anchor_max_frames = 1
        super().__init__(*args, **kwargs)
        self.init_anchor_store = EventAnchorLatentStore(
            event_anchor_latent_cache_root, max_anchors=1,
            max_cached_episodes=event_anchor_cache_max_open_episodes)
        manifest = self.init_anchor_store.manifest
        if manifest.get('anchor_policy') != 'episode_frame_zero_only':
            raise ValueError('Init KV requires a frame-zero cache.')
        if Path(manifest['dataset_root']).resolve() != Path(dataset_dirs[0]).resolve():
            raise ValueError('Init cache and training dataset roots differ.')
        if self.episode_indices is None or not set(self.episode_indices).issubset(manifest['episode_indices']):
            raise ValueError('Init cache does not cover the selected episodes.')

    @property
    def sampling_contract(self):
        return super().sampling_contract + '_init_frame_anchor_v1'

    def _get(self, idx):
        result = super()._get(idx)
        episode = int(result['episode_index'])
        anchor = self.init_anchor_store.load(episode, int(result['frame_index']))
        if anchor['event_anchor_frame_indices'].tolist() != [0]:
            raise ValueError('Expected only frame zero.')
        result.update(anchor)
        # Each episode has one instruction, so its init branch can be shared.
        result['event_anchor_group_id'] = torch.tensor(episode, dtype=torch.long)
        return result
