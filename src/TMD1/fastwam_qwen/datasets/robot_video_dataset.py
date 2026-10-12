import os
import traceback
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from TMD1.fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from TMD1.fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from TMD1.fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)
from TMD1.fastwam.utils import misc
from TMD1.fastwam.utils.logging_config import get_logger

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
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )

        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, "
                f"got {self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "video frames must be divisible by 4 for tokenization, "
                f"got {(self.num_frames - 1) // self.action_video_freq_ratio}"
            )
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

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
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets "
                        "since stats must not be recalculated on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
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
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
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
            return torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-1)], dim=-2)

        if self.concat_multi_camera == "robotwin_libero":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin_libero'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[224, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[112, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[112, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            return torch.cat([cam_top, torch.cat([cam_left, cam_right], dim=-2)], dim=-1)

        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                return torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            if self.concat_multi_camera == "vertical":
                return torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            raise ValueError(
                f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                "Expected one of: horizontal, vertical, robotwin, robotwin_libero."
            )
        return video.squeeze(0)

    def _get(self, idx):
        sample = self._sample_lerobot_item(idx)
        image_is_pad = sample["image_is_pad"][self.video_sample_indices]

        video = sample["pixel_values"]
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, c, h, w = video.shape
        else:
            if video.ndim != 4:
                raise ValueError(f"Expected video [T,C,H,W] or [N,T,C,H,W], got {tuple(video.shape)}")
            video = video[self.video_sample_indices, :, :, :]
            t_video, c, h, w = video.shape
            num_cameras = 1
        video = video.view(num_cameras, t_video, c, h, w)
        video = self._merge_cameras(video)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                "`action` horizon must be divisible by video transitions, "
                f"got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as exc:
            print(f"Error processing sample idx {idx}: {exc}. Returning a random sample instead.")
            print(traceback.format_exc())
            return self._get(np.random.randint(len(self)))
