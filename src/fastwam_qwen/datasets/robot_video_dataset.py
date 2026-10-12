import hashlib
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class RobotVideoDatasetNoT5(torch.utils.data.Dataset):
    """LIBERO/RoboTwin video dataset variant that returns raw prompts only.

    The original ``RobotVideoDataset`` always reads precomputed T5 embeddings.
    This side-path dataset keeps the same visual/action preprocessing but leaves
    language handling to Qwen3-VL inside the model.
    """

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=(384, 640),
        camera_key=None,
        processor=None,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        history_num_frames: int = 1,
        history_stride: Optional[int] = None,
        robometer_history_num_frames: int = 0,
        robometer_camera_key: Optional[str] = None,
        robometer_history_sampling: str = "causal_uniform",
        progress_cache_root: Optional[str] = None,
        progress_cache_contract: str = "legacy_state_instruction",
        progress_memory_mode: str = "all",
        progress_subtask_mode: str = "final",
        state_history_matches_condition_images: bool = False,
        state_history_control_offsets: Optional[list[int]] = None,
        state_history_dense_control_steps: bool = False,
        require_qwen_history: bool = True,
        sparse_image_loading: bool = True,
        video_backend: Optional[str] = None,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        episode_indices: Optional[list[int]] = None,
        training_shuffle_unit: str = "frame",
        samples_per_trajectory: int = 1,
        trajectory_anchor_stride: int = 1,
        tmd3_latent_cache_root: Optional[str] = None,
        tmd3_latent_max_mapped_episodes: int = 8,
        long_image_history_latents: int = 0,
        long_image_future_latents: int = 0,
        require_precomputed_video_latents: bool = False,
        skip_rgb_when_precomputed_latents: bool = False,
        endpose_action_source_map: Optional[str] = None,
        endpose_action_stats: Optional[str] = None,
        endpose_action_max_open_episodes: int = 8,
        endpose_action_max_cached_episodes: int = 8,
        progress_cache_max_open_episodes: int = 8,
        progress_cache_max_cached_instruction_episodes: int = 8,
        progress_subtask_text_embedding_cache_dir: Optional[str] = None,
        progress_require_ordinal_context: bool = False,
        progress_require_gt_subtask_exact: bool = False,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        text_embedding_use_true_mask: bool = False,
    ):
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.history_num_frames = int(history_num_frames)
        self.history_stride = int(history_stride if history_stride is not None else action_video_freq_ratio)
        self.robometer_history_num_frames = int(robometer_history_num_frames)
        self.robometer_camera_key = robometer_camera_key
        self.robometer_history_sampling = str(robometer_history_sampling).lower()
        self.progress_cache_root = progress_cache_root
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir in (None, "")
            else str(Path(text_embedding_cache_dir).expanduser().resolve())
        )
        self.context_len = int(context_len)
        self.text_embedding_use_true_mask = bool(
            text_embedding_use_true_mask
        )
        if self.context_len <= 0:
            raise ValueError("`context_len` must be positive.")
        self.progress_cache_contract = str(progress_cache_contract).lower()
        self.progress_require_gt_subtask_exact = bool(
            progress_require_gt_subtask_exact
        )
        self.state_history_matches_condition_images = bool(
            state_history_matches_condition_images
        )
        self.state_history_dense_control_steps = bool(
            state_history_dense_control_steps
        )
        self.state_history_control_offsets = (
            None
            if state_history_control_offsets is None
            else tuple(int(value) for value in state_history_control_offsets)
        )
        self.require_qwen_history = bool(require_qwen_history)
        self.training_shuffle_unit = str(training_shuffle_unit).lower()
        self.samples_per_trajectory = int(samples_per_trajectory)
        self.trajectory_anchor_stride = int(trajectory_anchor_stride)
        self.long_image_history_latents = int(long_image_history_latents)
        self.long_image_future_latents = int(long_image_future_latents)
        if (
            self.long_image_history_latents < 0
            or self.long_image_future_latents < 0
        ):
            raise ValueError(
                "Long-image latent counts cannot be negative: "
                f"history={self.long_image_history_latents}, "
                f"future={self.long_image_future_latents}."
            )
        if (self.long_image_history_latents == 0) != (
            self.long_image_future_latents == 0
        ):
            raise ValueError(
                "Long-image history and future counts must be enabled together."
            )
        self.episode_indices = (
            None
            if episode_indices is None
            else tuple(int(index) for index in episode_indices)
        )
        if self.training_shuffle_unit not in {"frame", "trajectory"}:
            raise ValueError(
                "`training_shuffle_unit` must be `frame` or `trajectory`, got "
                f"{training_shuffle_unit!r}."
            )
        if self.samples_per_trajectory <= 0:
            raise ValueError(
                "`samples_per_trajectory` must be positive, got "
                f"{self.samples_per_trajectory}."
            )
        if (
            self.training_shuffle_unit == "trajectory"
            and self.samples_per_trajectory != 1
        ):
            raise ValueError(
                "Full aligned-anchor sampling is without replacement, so "
                "`samples_per_trajectory` must be 1."
            )
        if self.trajectory_anchor_stride <= 0:
            raise ValueError(
                "`trajectory_anchor_stride` must be positive, got "
                f"{self.trajectory_anchor_stride}."
            )
        if self.training_shuffle_unit == "trajectory" and not is_training_set:
            raise ValueError(
                "Trajectory shuffle is a training-only sampling policy."
            )
        self.tmd3_latent_store = None
        if tmd3_latent_cache_root:
            from TMD3.latent_cache import TMD3LatentStore

            camera_order = [str(item["key"]) for item in shape_meta["images"]]
            self.tmd3_latent_store = TMD3LatentStore(
                tmd3_latent_cache_root,
                camera_order=camera_order,
                history_stride=self.history_stride,
                action_horizon=self.num_frames - 1,
                anchor_stride=self.trajectory_anchor_stride,
                max_mapped_episodes=tmd3_latent_max_mapped_episodes,
            )
        elif require_precomputed_video_latents:
            raise ValueError(
                "Official LingBot-style TMD3 training requires "
                "`tmd3_latent_cache_root`. Run the episode latent precompute "
                "before training."
            )
        if self.long_image_history_latents and self.tmd3_latent_store is None:
            raise ValueError(
                "Long-image training requires `tmd3_latent_cache_root`."
            )
        self.skip_rgb_when_precomputed_latents = bool(
            skip_rgb_when_precomputed_latents
        )
        if (
            self.skip_rgb_when_precomputed_latents
            and self.tmd3_latent_store is None
        ):
            raise ValueError(
                "`skip_rgb_when_precomputed_latents` requires a validated "
                "TMD3 latent cache."
            )
        self.endpose_action_store = None
        if endpose_action_source_map or endpose_action_stats:
            if not endpose_action_source_map or not endpose_action_stats:
                raise ValueError(
                    "Both `endpose_action_source_map` and "
                    "`endpose_action_stats` are required."
                )
            from TMD3.endpose_action import RMBenchEndposeActionStore

            self.endpose_action_store = RMBenchEndposeActionStore(
                endpose_action_source_map,
                endpose_action_stats,
                max_open_episodes=endpose_action_max_open_episodes,
                max_cached_episodes=endpose_action_max_cached_episodes,
            )
        self.endpose_state_matches_action = self.endpose_action_store is not None
        if self.endpose_state_matches_action:
            if not self.state_history_matches_condition_images:
                raise ValueError(
                    "RobotWin EEF action requires an aligned EEF state-history "
                    "contract; qpos state with EEF action is forbidden."
                )
        self.progress_feature_store = None
        if self.progress_cache_root:
            if self.progress_cache_contract == "legacy_state_instruction":
                from TMD2.progress_cache import ProgressFeatureStore

                self.progress_feature_store = ProgressFeatureStore(
                    self.progress_cache_root,
                    memory_mode=progress_memory_mode,
                    subtask_mode=progress_subtask_mode,
                )
            elif self.progress_cache_contract in {
                "current_progress_state_stream",
                "progress_logits_subtask_state4",
            }:
                from TMD3.progress_cache import StateProgressFeatureStore

                self.progress_feature_store = StateProgressFeatureStore(
                    self.progress_cache_root,
                    max_open_episodes=progress_cache_max_open_episodes,
                    max_cached_instruction_episodes=(
                        progress_cache_max_cached_instruction_episodes
                    ),
                    subtask_text_embedding_cache_dir=(
                        progress_subtask_text_embedding_cache_dir
                    ),
                    require_ordinal_context=progress_require_ordinal_context,
                    require_gt_subtask_exact=(
                        self.progress_require_gt_subtask_exact
                    ),
                )
            else:
                raise ValueError(
                    "Unknown progress_cache_contract: "
                    f"{self.progress_cache_contract!r}."
                )
        elif self.progress_require_gt_subtask_exact:
            raise ValueError(
                "progress_require_gt_subtask_exact requires progress_cache_root."
            )
        if self.robometer_history_num_frames < 0 or self.robometer_history_num_frames > 8:
            raise ValueError(
                "`robometer_history_num_frames` must be in [0, 8], got "
                f"{self.robometer_history_num_frames}."
            )
        if self.robometer_history_num_frames and not self.robometer_camera_key:
            raise ValueError(
                "`robometer_camera_key` is required when Robometer history is enabled."
            )
        if self.history_num_frames <= 0:
            raise ValueError(f"`history_num_frames` must be positive, got {self.history_num_frames}")
        if self.history_stride <= 0:
            raise ValueError(f"`history_stride` must be positive, got {self.history_stride}")
        if (self.history_num_frames - 1) % 4 != 0:
            raise ValueError(
                "`history_num_frames` must follow the Wan VAE 1+4n temporal layout so every "
                f"condition image is encoded, got K={self.history_num_frames}."
            )
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, "
                f"got {self.num_frames - 1} and {self.action_video_freq_ratio}"
            )

        self.history_obs_span = (self.history_num_frames - 1) * self.history_stride
        self.current_obs_index = self.history_obs_span
        self.base_num_frames = self.history_obs_span + self.num_frames
        self.action_horizon = self.num_frames - 1
        self.condition_video_indices = [
            self.current_obs_index - (self.history_num_frames - 1 - i) * self.history_stride
            for i in range(self.history_num_frames)
        ]
        self.future_video_indices = list(
            range(
                self.current_obs_index + self.action_video_freq_ratio,
                self.current_obs_index + self.num_frames,
                self.action_video_freq_ratio,
            )
        )
        self.video_sample_indices = self.condition_video_indices + self.future_video_indices
        if self.state_history_dense_control_steps:
            if not self.state_history_matches_condition_images:
                raise ValueError(
                    "`state_history_dense_control_steps` requires "
                    "`state_history_matches_condition_images=true`."
                )
            if self.state_history_control_offsets is not None:
                raise ValueError(
                    "Dense control-step state history computes its offsets "
                    "from the 64-step visual window; do not also provide "
                    "`state_history_control_offsets`."
                )
            # The first sampled image is the anchor at t-64.  The 64 actual
            # control states following that anchor are t-63,...,t and split
            # into four 16-step groups aligned with H1,...,H4.
            self.state_obs_indices = list(
                range(
                    self.current_obs_index - self.history_obs_span + 1,
                    self.current_obs_index + 1,
                )
            )
            if len(self.state_obs_indices) != self.history_obs_span:
                raise RuntimeError(
                    "Dense state history must contain one state per control "
                    f"step, got {len(self.state_obs_indices)} for span "
                    f"{self.history_obs_span}."
                )
        elif self.state_history_control_offsets is not None:
            if not self.state_history_matches_condition_images:
                raise ValueError(
                    "`state_history_control_offsets` requires "
                    "`state_history_matches_condition_images=true`."
                )
            allowed_state_offsets = {
                (-48, -32, -16, 0),
                (0,),
            }
            if self.state_history_control_offsets not in allowed_state_offsets:
                raise ValueError(
                    "TMD3 state offsets must be latent-aligned "
                    "[-48, -32, -16, 0] or current-only [0], got "
                    f"{list(self.state_history_control_offsets)}."
                )
            self.state_obs_indices = [
                self.current_obs_index + offset
                for offset in self.state_history_control_offsets
            ]
        elif self.state_history_matches_condition_images:
            self.state_obs_indices = list(self.condition_video_indices)
        else:
            self.state_obs_indices = [self.current_obs_index]
        if any(
            index < 0 or index >= self.base_num_frames
            for index in self.state_obs_indices
        ):
            raise ValueError(
                "Computed state indices are outside the LeRobot read window: "
                f"{self.state_obs_indices}."
            )
        self.state_history_num_frames = len(self.state_obs_indices)
        self.sparse_image_loading = bool(sparse_image_loading)
        if len(set(self.condition_video_indices)) != len(self.condition_video_indices):
            raise ValueError(
                "`history_stride` produced duplicate history indices: "
                f"{self.condition_video_indices}"
            )
        if any(i < 0 or i >= self.base_num_frames for i in self.video_sample_indices):
            raise ValueError(
                "Computed video indices are outside the LeRobot read window: "
                f"indices={self.video_sample_indices}, base_num_frames={self.base_num_frames}"
            )
        if len(self.future_video_indices) % 4 != 0:
            raise ValueError(
                "The number of future video frames must be divisible by the Wan VAE temporal factor (4), "
                f"got future_T={len(self.future_video_indices)} from num_frames={self.num_frames} "
                f"and action_video_freq_ratio={self.action_video_freq_ratio}."
            )

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=self.base_num_frames,
            action_size=self.base_num_frames - 1,
            past_obs_size=self.history_obs_span,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_obs_indices=(self.video_sample_indices if self.sparse_image_loading else None),
            # TMD3 may select four endpoint states or every control-step state
            # represented by H1..H4. Legacy paths keep one current token.
            state_obs_indices=self.state_obs_indices,
            # Qwen images use a separate, per-sample episode-start-to-current time grid.
            episode_uniform_image_history_frames=(
                self.history_num_frames if self.require_qwen_history else 0
            ),
            episode_causal_history_frames=self.robometer_history_num_frames,
            episode_causal_camera_key=self.robometer_camera_key,
            episode_causal_history_sampling=self.robometer_history_sampling,
            video_backend=video_backend,
            selected_episode_indices=episode_indices,
        )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(
            not self.skip_rgb_when_precomputed_latents
        )

        self.video_size = list(video_size)
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = OmegaConf.create(OmegaConf.to_container(processor, resolve=True))
                if "num_obs_steps" in processor and int(processor.num_obs_steps) != self.base_num_frames:
                    logger.info(
                        "Overriding processor.num_obs_steps from %s to %s for history window.",
                        processor.num_obs_steps,
                        self.base_num_frames,
                    )
                    processor.num_obs_steps = self.base_num_frames
                processor = instantiate(processor)
            elif hasattr(processor, "num_obs_steps") and int(processor.num_obs_steps) != self.base_num_frames:
                logger.info(
                    "Overriding processor.num_obs_steps from %s to %s for history window.",
                    processor.num_obs_steps,
                    self.base_num_frames,
                )
                processor.num_obs_steps = self.base_num_frames
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets "
                        "since stats must not be recalculated on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(
                        processor,
                        action_horizon=self.action_horizon,
                    )
                    save_dataset_stats_to_json(
                        dataset_stats,
                        os.path.join(misc.get_work_dir(), "dataset_stats.json"),
                    )
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info("Using dataset stats: %s", pretrained_norm_stats)
                if PartialState().is_main_process:
                    save_dataset_stats_to_json(
                        dataset_stats,
                        os.path.join(misc.get_work_dir(), "dataset_stats.json"),
                    )

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def shuffled_epoch_indices(
        self,
        generator: torch.Generator,
    ) -> list[int]:
        """Return every legal aligned anchor once, in a random order."""
        if self.training_shuffle_unit != "trajectory":
            return torch.randperm(len(self), generator=generator).tolist()

        anchors = self._aligned_anchor_indices()
        order = torch.randperm(len(anchors), generator=generator).tolist()
        return [anchors[index] for index in order]

    def _aligned_anchor_indices(self) -> list[int]:
        """Enumerate episode-local ``t=0,stride,2*stride,...`` anchors."""
        episode_from = self.lerobot_dataset.episode_data_index["from"]
        episode_to = self.lerobot_dataset.episode_data_index["to"]
        anchors = []
        for trajectory_id, (start_value, end_value) in enumerate(
            zip(episode_from.tolist(), episode_to.tolist())
        ):
            start = int(start_value)
            end = int(end_value)
            # The ordinary TMD3 target extends through t+16.  Long-image
            # training may request up to four consecutive future latents, so
            # reserve the complete real future rather than padding it.
            future_span = int(self.action_horizon)
            if self.long_image_future_latents:
                future_span = max(
                    future_span,
                    int(self.long_image_future_latents)
                    * int(self.action_horizon),
                )
            high = end - future_span
            if high <= start:
                raise ValueError(
                    "Trajectory is too short for one future latent and a "
                    f"{self.action_horizon}-step action chunk: "
                    f"trajectory={trajectory_id}, range=[{start}, {end})."
                )
            episode_anchors = np.arange(
                start,
                high,
                self.trajectory_anchor_stride,
                dtype=np.int64,
            )
            if self.progress_require_gt_subtask_exact:
                episode_index = (
                    trajectory_id
                    if self.episode_indices is None
                    else self.episode_indices[trajectory_id]
                )
                valid = self.progress_feature_store.load_training_valid_mask(
                    episode_index
                )
                local_anchors = episode_anchors - start
                episode_anchors = episode_anchors[valid[local_anchors]]
            anchors.extend(episode_anchors.tolist())
        return anchors

    @property
    def shuffled_epoch_size(self) -> int:
        if self.training_shuffle_unit != "trajectory":
            return len(self)
        return len(self._aligned_anchor_indices())

    @property
    def anchor_population_size(self) -> int:
        """Number of distinct real anchors in one no-replacement epoch."""
        if self.training_shuffle_unit != "trajectory":
            return len(self)
        return self.shuffled_epoch_size

    @property
    def sampling_contract(self) -> str:
        if self.training_shuffle_unit == "trajectory":
            if self.episode_indices is None:
                episode_selection = "all"
            else:
                selection_text = ",".join(
                    str(index) for index in self.episode_indices
                )
                episode_selection = hashlib.sha1(
                    selection_text.encode("utf-8")
                ).hexdigest()[:12]
            return (
                "control_anchor_without_replacement_"
                f"stride{self.trajectory_anchor_stride}_"
                f"episodes{episode_selection}_"
                f"longh{self.long_image_history_latents}_"
                f"longf{self.long_image_future_latents}_"
                f"gtsubtaskexact{int(self.progress_require_gt_subtask_exact)}_v4"
            )
        return "frame_without_replacement_v1"

    @property
    def pad_to_global_batch(self) -> bool:
        """Use zero-loss sentinels instead of repeating real anchors in DDP."""
        return self.training_shuffle_unit == "trajectory"

    def _sample_lerobot_item(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if not self.skip_padding_as_possible:
                break

            has_pad = bool(sample["action_is_pad"].any().item())
            has_pad = has_pad or bool(sample["image_is_pad"].any().item())
            has_pad = has_pad or bool(sample["proprio_is_pad"].any().item())
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))
        return sample

    def _merge_cameras(self, video: torch.Tensor) -> torch.Tensor:
        num_cameras, t_video, c, h, w = video.shape
        if self.concat_multi_camera == "robotwin":
            from TMD3.multiview import RGB_TRANSPORT_SIZE, compose_robotwin_rgb

            if num_cameras == 1:
                if (h, w) != RGB_TRANSPORT_SIZE:
                    raise ValueError(
                        "Precomposed RobotWin RGB input must have size "
                        f"{RGB_TRANSPORT_SIZE}, got {(h, w)}."
                    )
                return video[0]
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
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
            return compose_robotwin_rgb([cam_high, cam_left, cam_right])

        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                return torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            if self.concat_multi_camera == "vertical":
                return torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            raise ValueError(
                f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                "Expected one of: horizontal, vertical, robotwin."
            )
        return video.squeeze(0)

    def _get(self, idx):
        sample = self._sample_lerobot_item(idx)
        if self.sparse_image_loading:
            if sample["image_is_pad"].shape[0] != len(self.video_sample_indices):
                raise ValueError(
                    "Sparse image padding mask length mismatch: "
                    f"got {sample['image_is_pad'].shape[0]}, expected {len(self.video_sample_indices)}."
                )
            condition_image_is_pad = sample["image_is_pad"][: self.history_num_frames]
            future_image_is_pad = sample["image_is_pad"][self.history_num_frames :]
        else:
            condition_image_is_pad = sample["image_is_pad"][self.condition_video_indices]
            future_image_is_pad = sample["image_is_pad"][self.future_video_indices]
        image_is_pad = torch.cat([condition_image_is_pad, future_image_is_pad], dim=0)

        video = sample["pixel_values"]
        if self.skip_rgb_when_precomputed_latents:
            if video.ndim != 5 or tuple(video.shape[2:]) != (3, 1, 1):
                raise ValueError(
                    "Precomputed-latent training expects the lightweight "
                    f"[N,T,3,1,1] RGB carrier, got {tuple(video.shape)}."
                )
            if not self.sparse_image_loading:
                video = video[:, self.video_sample_indices]
            t_video = int(video.shape[1])
            if t_video != len(self.video_sample_indices):
                raise ValueError(
                    f"Selected video must contain {len(self.video_sample_indices)} frames, got {t_video}."
                )
            # Preserve the public [C,T,H,W] shape contract without allocating,
            # resizing, normalizing, or later transferring unused RGB pixels.
            video = video.new_empty((3, t_video, 1, 1))
        else:
            if video.ndim == 5:
                if not self.sparse_image_loading:
                    video = video[:, self.video_sample_indices, :, :, :]
                num_cameras, t_video, c, h, w = video.shape
            else:
                if video.ndim != 4:
                    raise ValueError(f"Expected video [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}")
                if not self.sparse_image_loading:
                    video = video[self.video_sample_indices, :, :, :]
                t_video, c, h, w = video.shape
                num_cameras = 1
            if t_video != len(self.video_sample_indices):
                raise ValueError(
                    f"Selected video must contain {len(self.video_sample_indices)} frames, got {t_video}."
                )
            video = video.view(num_cameras, t_video, c, h, w)
            video = self._merge_cameras(video)

            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)
            video = video.permute(1, 0, 2, 3)

        qwen_video = sample.get("qwen_pixel_values")
        if self.require_qwen_history:
            if qwen_video is None:
                raise KeyError("Dynamically sampled `qwen_pixel_values` are required.")
            if qwen_video.ndim == 5:
                qwen_cameras, qwen_t, qwen_c, qwen_h, qwen_w = qwen_video.shape
            elif qwen_video.ndim == 4:
                qwen_t, qwen_c, qwen_h, qwen_w = qwen_video.shape
                qwen_cameras = 1
            else:
                raise ValueError(
                    "Expected Qwen video [K,C,H,W] or [N,K,C,H,W], "
                    f"got {tuple(qwen_video.shape)}."
                )
            if qwen_t != self.history_num_frames:
                raise ValueError(
                    f"Qwen history must contain K={self.history_num_frames} frames, got {qwen_t}."
                )
            qwen_video = qwen_video.view(qwen_cameras, qwen_t, qwen_c, qwen_h, qwen_w)
            qwen_video = self._merge_cameras(qwen_video)
            qwen_video = self.resize_transform(qwen_video)
            qwen_video = self.crop_transform(qwen_video)
            qwen_video = self.normalize_transform(qwen_video)

        action = sample["action"][: self.action_horizon]
        action_is_pad = sample["action_is_pad"][: self.action_horizon]
        proprio = sample["proprio"]
        proprio_is_pad = sample["proprio_is_pad"]
        episode_index = int(torch.as_tensor(sample["episode_index"]).item())
        frame_index = int(torch.as_tensor(sample["frame_index"]).item())
        if self.endpose_state_matches_action:
            state_frame_indices = [
                frame_index + index - self.current_obs_index
                for index in self.state_obs_indices
            ]
            proprio = self.endpose_action_store.normalize_states(
                episode_index,
                state_frame_indices,
            )
        if self.state_history_matches_condition_images:
            expected_state_shape = (
                self.state_history_num_frames,
                (
                    self.endpose_action_store.model_dim
                    if self.endpose_state_matches_action
                    else 14
                ),
            )
            if (
                tuple(proprio.shape) != expected_state_shape
                or tuple(proprio_is_pad.shape) != (self.state_history_num_frames,)
            ):
                raise ValueError(
                    "State history must align with the configured latent endpoints: "
                    f"proprio={tuple(proprio.shape)}, "
                    f"pad={tuple(proprio_is_pad.shape)}, "
                    f"expected={expected_state_shape}."
                )
            state_history = proprio
            state_history_mask = ~proprio_is_pad.bool()
            qwen_proprio = proprio[-1:]
            qwen_proprio_is_pad = proprio_is_pad[-1:]
        else:
            if proprio.shape[0] != 1 or proprio_is_pad.shape != (1,):
                raise ValueError(
                    "State loading must return exactly the current frame, "
                    f"got proprio={tuple(proprio.shape)}, pad={tuple(proprio_is_pad.shape)}."
                )
            state_history = None
            state_history_mask = None
            qwen_proprio = proprio
            qwen_proprio_is_pad = proprio_is_pad
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        future_video_steps = video.shape[1] - self.history_num_frames
        if future_video_steps <= 0:
            raise ValueError(
                "`video` must include at least one future frame after the condition prefix, "
                f"got video_shape={tuple(video.shape)} and history_num_frames={self.history_num_frames}"
            )
        if action.shape[0] % future_video_steps != 0:
            raise ValueError(
                "`action` horizon must be divisible by future video steps, "
                f"got action_horizon={action.shape[0]} and future_video_steps={future_video_steps}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        result = {
            "video": video,
            "action": action,
            "proprio": qwen_proprio,
            "qwen_proprio": qwen_proprio,
            "episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"],
            "prompt": instruction,
            "num_condition_video_frames": self.history_num_frames,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": qwen_proprio_is_pad,
            "qwen_proprio_is_pad": qwen_proprio_is_pad,
            "task": task,
            **(
                {
                    "state_history": state_history,
                    "state_history_mask": state_history_mask,
                }
                if state_history is not None
                else {}
            ),
            **(
                {
                    "robometer_frames": sample["robometer_frames"],
                    "robometer_states": sample["robometer_states"],
                    "robometer_history_indices": sample["robometer_history_indices"],
                }
                if self.robometer_history_num_frames
                else {}
            ),
        }
        if self.text_embedding_cache_dir is not None:
            context, context_mask = self._get_cached_text_context(instruction)
            # Wan2.2 zeroes padded embeddings.  The original FastWAM path also
            # marked those zero tokens visible; the instruction ablation can
            # retain the tokenizer's real mask so it can skip that
            # content-free suffix without dropping instruction tokens.
            context = context.clone()
            context[~context_mask] = 0.0
            result["context"] = context
            result["context_mask"] = (
                context_mask
                if self.text_embedding_use_true_mask
                else torch.ones_like(context_mask, dtype=torch.bool)
            )
        if self.endpose_action_store is not None:
            result["action"] = self.endpose_action_store.normalize_window(
                episode_index,
                frame_index,
                self.action_horizon,
                action_is_pad=action_is_pad,
            )
        if self.require_qwen_history:
            result["qwen_images"] = qwen_video
            result["qwen_image_is_pad"] = sample["qwen_image_is_pad"]
            result["qwen_history_indices"] = sample["qwen_history_indices"]
        if self.progress_feature_store is not None:
            episode_index = int(torch.as_tensor(sample["episode_index"]).item())
            frame_index = int(torch.as_tensor(sample["frame_index"]).item())
            result.update(
                self.progress_feature_store.load(episode_index, frame_index)
            )
        if self.tmd3_latent_store is not None:
            episode_index = int(torch.as_tensor(sample["episode_index"]).item())
            frame_index = int(torch.as_tensor(sample["frame_index"]).item())
            result.update(
                self.tmd3_latent_store.load(episode_index, frame_index)
            )
            if self.long_image_history_latents:
                result.update(
                    self.tmd3_latent_store.load_long_image_clip(
                        episode_index,
                        frame_index,
                        max_history_latents=self.long_image_history_latents,
                        max_future_latents=self.long_image_future_latents,
                    )
                )
        return result

    def _get_cached_text_context(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("`text_embedding_cache_dir` is not configured.")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = Path(self.text_embedding_cache_dir) / (
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Missing T5 instruction cache: {cache_path}. Run "
                "scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2 or tuple(context.shape) != (
            self.context_len,
            4096,
        ):
            raise ValueError(
                "Cached T5 context must be "
                f"[{self.context_len},4096], got {tuple(context.shape)} "
                f"from {cache_path}."
            )
        if tuple(context_mask.shape) != (self.context_len,):
            raise ValueError(
                "Cached T5 mask must be "
                f"[{self.context_len}], got {tuple(context_mask.shape)} "
                f"from {cache_path}."
            )
        return context, context_mask

    @staticmethod
    def collate_fn(samples):
        """Collate FastWAM tensors while preserving variable real histories."""
        if not samples:
            return {}
        variable_keys = {
            "robometer_frames",
            "robometer_states",
            "robometer_history_indices",
            "progress_memory_layers",
            "progress_memory_mask",
            "progress_subtask_tokens",
            "progress_subtask_layers",
            "progress_subtask_mask",
            "progress_condition_index",
            "progress_decode_source_frame",
            "progress_parse_valid",
            "progress_active_idx",
            "progress_plan_length",
            "progress_current_layers",
            "progress_current_mask",
            "progress_instruction_layers",
            "progress_instruction_mask",
            "progress_logits",
        }
        fixed = [
            {key: value for key, value in sample.items() if key not in variable_keys}
            for sample in samples
        ]
        batch = default_collate(fixed)
        for key in variable_keys:
            if key in samples[0]:
                if key.startswith("progress_"):
                    continue
                batch[key] = [sample[key] for sample in samples]
        if "progress_logits" in samples[0] or "progress_current_layers" in samples[0]:
            from TMD3.progress_cache import pad_state_progress_features

            batch.update(pad_state_progress_features(samples))
        elif "progress_memory_layers" in samples[0]:
            from TMD2.progress_cache import pad_progress_features

            batch.update(pad_progress_features(samples))
        return batch

    def __getitem__(self, idx):
        is_padding = int(idx) < 0
        source_idx = 0 if is_padding else int(idx)
        try:
            result = self._get(source_idx)
            result["sample_weight"] = torch.tensor(
                0.0 if is_padding else 1.0,
                dtype=torch.float32,
            )
            return result
        except Exception as exc:
            if self.training_shuffle_unit == "trajectory":
                raise RuntimeError(
                    "Aligned-anchor epochs cannot replace a failed sample "
                    f"without violating no-replacement coverage (idx={idx})."
                ) from exc
            print(f"Error processing sample idx {idx}: {exc}. Returning a random sample instead.")
            print(traceback.format_exc())
            return self._get(np.random.randint(len(self)))
