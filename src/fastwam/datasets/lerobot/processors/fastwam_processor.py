from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Dict, Any, Optional, List, Literal

import torch
import numpy as np
from copy import deepcopy
from ..utils.normalizer import LinearNormalizer, NormMode
from fastwam.utils.pytorch_utils import dict_apply
from fastwam.utils.logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)

class FastWAMProcessor(BaseProcessor):
    def __init__(
        self,
        # keys
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int, 
        action_output_dim: int,
        proprio_output_dim: int,

        action_state_transforms: Optional[List[Any]], 

        # action & state normalization
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Dict[str, Dict[str, NormMode]], 

        action_state_merger, 

        # image transform
        train_transforms: Dict[str, List[Any]] | None,
        val_transforms: Dict[str, List[Any]] | None, 

        # instruction transform
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,

        tokenizer: Optional[Any] = None,
        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,
        image_output_layout: Optional[str] = None,
        allow_missing_images: bool = False,
        normalization_dim_mask: Optional[
            Dict[str, Dict[str, List[bool]]]
        ] = None,
        norm_output_clip_range: float = 5.0,
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = num_obs_steps
        self.num_output_cameras = num_output_cameras
        self.action_output_dim = action_output_dim
        self.proprio_output_dim = proprio_output_dim

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        # image
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.image_output_layout = (
            None
            if image_output_layout is None
            else str(image_output_layout).strip().lower()
        )
        if self.image_output_layout not in {None, "robotwin"}:
            raise ValueError(
                "`image_output_layout` must be null or `robotwin`, got "
                f"{image_output_layout!r}."
            )
        self.allow_missing_images = bool(allow_missing_images)

        self._is_train = None

        self.action_state_transforms = action_state_transforms
        self.action_state_merger = action_state_merger
        self.action_state_merger.set_shape_meta(self.shape_meta)

        self.use_stepwise_action_norm = use_stepwise_action_norm
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self.normalization_dim_mask = normalization_dim_mask
        self.norm_output_clip_range = float(norm_output_clip_range)
        self._normalizer = None

        self.tokenizer = tokenizer
        if delta_action_dim_mask is None:
            self.delta_action_dim_mask = None
        else:
            action_meta = self.shape_meta["action"]
            expected_keys = [m["key"] for m in action_meta]
            provided_keys = list(delta_action_dim_mask.keys())
            if set(provided_keys) != set(expected_keys):
                raise ValueError(
                    f"`delta_action_dim_mask` keys mismatch. Expected {expected_keys}, got {provided_keys}."
                )

            self.delta_action_dim_mask = {}
            for meta in action_meta:
                key = meta["key"]
                expected_dim = meta["shape"]
                mask = delta_action_dim_mask[key]
                if len(mask) != expected_dim:
                    raise ValueError(
                        f"`delta_action_dim_mask[{key}]` length must be {expected_dim}, got {len(mask)}."
                    )
                self.delta_action_dim_mask[key] = torch.as_tensor(mask, dtype=torch.bool)

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any] = None):
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
            dim_mask=self.normalization_dim_mask,
            output_clip_range=self.norm_output_clip_range,
        )

    def augment_instruction(self, data: Dict[str, str] | List[str]) -> List[str]:
        """
        Args:
            data: Dict[str, str] | List[str], lerobot sample in raw mcap

        Returns:
            List[str], processed instructions
        """
        # if single instruction, convert to list
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"{low_level_instruction}"
        else: 
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"
        
        return instruction

    def action_state_transform(self, batch):
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["raw_shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
                    
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual raw shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        if self.action_state_transforms is not None: 
            for trans in self.action_state_transforms:
                batch = trans.forward(batch)
        
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                k, meta_shape = meta["key"], meta["shape"]
                actual_shape = batch["action"][k].shape[-1]
                assert actual_shape == meta_shape, \
                    f"Action key {k} actual transformed shape {actual_shape} mismatch with meta shape {meta_shape}."
        
        for meta in self.shape_meta["state"]:
            k, meta_shape = meta["key"], meta["shape"]
            actual_shape = batch["state"][k].shape[-1]
            assert actual_shape == meta_shape, \
                f"State key {k} actual transformed shape {actual_shape} mismatch with meta raw shape {meta_shape}."
        
        return batch

    def zero_padded_delta_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in data or self.delta_action_dim_mask is None:
            return data
        action_is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
        if not bool(action_is_pad.any().item()):
            return data
        for key, dim_mask in self.delta_action_dim_mask.items():
            cur_action = data["action"][key]
            cur_action_is_pad = action_is_pad.to(device=cur_action.device)
            cur_dim_mask = dim_mask.to(device=cur_action.device)
            pad_delta_mask = cur_action_is_pad.unsqueeze(1) & cur_dim_mask.unsqueeze(0)
            cur_action[pad_delta_mask] = 0.0
        return data

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for the policy model.
        
        Args:
            Data: Dict[str, Any], lerobot sample in raw mcap obtained from dataset __getitem__:
                - "action": Optional, Dict[str, torch.Tensor] -> [action_horizon, action_dim]
                - "state": Dict[str, torch.Tensor] -> [num_obs_steps, state_dim]
                - "images": Dict[str, torch.Tensor] -> [num_obs_steps, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "idx": int, sample index
                
        Returns:
            Sample: Dict[str, Any], which can collated:
                - "input_ids": torch.Tensor -> [max_image_text_tokens,]
                - "attention_mask": torch.Tensor -> [max_image_text_tokens,]
                - "pixel_values": torch.Tensor -> [num_input_cameras, C, H, W]
                - "image_is_pad": torch.Tensor -> [num_obs_steps,]
                - "proprio": torch.Tensor -> [num_obs_steps, proprio_dim]
                - "state_is_pad": torch.Tensor -> [num_obs_steps,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action: Optional, deepcopy of input action for open loop eval, which is left untouched
                - "idx": int, sample index
        """
        sample = {}
        # 1. instruction
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]
        image_num_obs_steps = int(sample["image_is_pad"].shape[0])

        # 2. image
        processed_images = []
        source_images = data.get("images", {})
        if not source_images:
            if not self.allow_missing_images:
                raise KeyError("Processor input contains no camera images.")
            if self.image_output_layout != "robotwin":
                raise ValueError(
                    "Missing-image synthesis is only defined for RobotWin layout."
                )
            # When images are intentionally absent, downstream training uses
            # precomputed VAE latents. Keep only T/C metadata here instead of
            # allocating a full 384x320 zero video that is never consumed.
            pixel_values = torch.zeros(
                (1, image_num_obs_steps, 3, 1, 1),
                dtype=torch.float32,
            )
        else:
            for meta in self.shape_meta["images"]:
                key, shape = meta["key"], meta["shape"]
                image = source_images[key]  # [num_obs_steps, C, H, W]
                assert image.ndim == 4, f"Expected 4 dimensions (num_obs_steps, C, H, W), got shape {image.shape}"
                assert image.shape[0] == image_num_obs_steps, (
                    f"Image time dimension {image.shape[0]} must match image padding mask "
                    f"{image_num_obs_steps} for key {key}."
                )

                # Apply transforms efficiently on the merged batch
                transforms = self.train_transforms if self.is_train else self.val_transforms
                current_transforms = (
                    transforms[key] if isinstance(transforms, Mapping) else transforms
                )
                for trans in current_transforms:
                    image = trans(image)

                meta_shape = [image_num_obs_steps] + shape
                assert image.shape == meta_shape, \
                    f"Expected shape {meta_shape}, got {image.shape} after transforms for key {key}"

                processed_images.append(image)
            if self.image_output_layout == "robotwin":
                from TMD3.multiview import CAMERA_ORDER, compose_robotwin_rgb

                image_keys = tuple(str(meta["key"]) for meta in self.shape_meta["images"])
                if image_keys != CAMERA_ORDER:
                    raise ValueError(
                        "RobotWin image output requires camera order "
                        f"{CAMERA_ORDER}, got {image_keys}."
                    )
                pixel_values = compose_robotwin_rgb(processed_images).unsqueeze(0)
            else:
                pixel_values = torch.stack(processed_images, dim=0) # [num_input_cameras, T, C, H, W]
        
        if self.num_output_cameras > pixel_values.shape[0]:
            out = torch.zeros((self.num_output_cameras,) + pixel_values.shape[1:], device=pixel_values.device, dtype=pixel_values.dtype)
            out[0: pixel_values.shape[0]] = pixel_values
            sample["pixel_values"] = out
        elif self.num_output_cameras < pixel_values.shape[0]:
            logger.warning(f"num_output_cameras {self.num_output_cameras} is less than the number of cameras in data {pixel_values.shape[0]}, "
                           f"truncating the input to the first {self.num_output_cameras} cameras.")
            sample["pixel_values"] = pixel_values[:self.num_output_cameras]
        else:
            sample["pixel_values"] = pixel_values

        if "qwen_images" in data:
            qwen_image_is_pad = torch.as_tensor(data["qwen_image_is_pad"], dtype=torch.bool)
            qwen_num_steps = int(qwen_image_is_pad.shape[0])
            processed_qwen_images = []
            for meta in self.shape_meta["images"]:
                key, shape = meta["key"], meta["shape"]
                image = data["qwen_images"][key]
                assert image.ndim == 4 and image.shape[0] == qwen_num_steps, (
                    f"Qwen image for {key} must be [K,C,H,W] with K={qwen_num_steps}, "
                    f"got {tuple(image.shape)}."
                )
                transforms = self.train_transforms if self.is_train else self.val_transforms
                current_transforms = (
                    transforms[key]
                    if isinstance(transforms, Mapping)
                    else transforms
                )
                for trans in current_transforms:
                    image = trans(image)
                expected_shape = [qwen_num_steps] + shape
                assert list(image.shape) == expected_shape, (
                    f"Expected Qwen image shape {expected_shape}, got {list(image.shape)} for key {key}."
                )
                processed_qwen_images.append(image)
            if self.image_output_layout == "robotwin":
                from TMD3.multiview import compose_robotwin_rgb

                qwen_pixel_values = compose_robotwin_rgb(
                    processed_qwen_images
                ).unsqueeze(0)
            else:
                qwen_pixel_values = torch.stack(processed_qwen_images, dim=0)
            if self.num_output_cameras > qwen_pixel_values.shape[0]:
                out = torch.zeros(
                    (self.num_output_cameras,) + qwen_pixel_values.shape[1:],
                    device=qwen_pixel_values.device,
                    dtype=qwen_pixel_values.dtype,
                )
                out[: qwen_pixel_values.shape[0]] = qwen_pixel_values
                qwen_pixel_values = out
            elif self.num_output_cameras < qwen_pixel_values.shape[0]:
                qwen_pixel_values = qwen_pixel_values[: self.num_output_cameras]
            sample["qwen_pixel_values"] = qwen_pixel_values
            sample["qwen_image_is_pad"] = qwen_image_is_pad
            sample["qwen_history_indices"] = data["qwen_history_indices"]

        # Copy action before transform for open-loop evaluation, 
        # disabled for training dataset as it may cause collating key problem.
        if not self.is_train and "action" in data:
            sample["gt_action"] = deepcopy(data["action"])

        # 3. action & state
        data = self.action_state_transform(data)
        data = self.zero_padded_delta_actions(data)
        data = self.normalizer.forward(data)
        data = self.action_state_merger.forward(data)

        if "action" in data:
            sample["action"] = data["action"] # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"] # [action_horizon,]
            sample["action_dim_is_pad"] = data["action_dim_is_pad"] # [action_dim,]
            assert sample["action"].shape[-1] == self.action_output_dim
            # sample["action"][sample["action_is_pad"], :-1] = 0.0 # NOTE: we assume use delta_eef_pose + gripper， so pad action is 0

        
        # TODO: rename all "state" into "proprio"
        sample["proprio"] = data["state"] # [num_obs_steps, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"] # [num_obs_steps,]
        sample["proprio_dim_is_pad"] = data["state_dim_is_pad"] # [proprio_dim,]
        assert sample["proprio"].shape[-1] == self.proprio_output_dim

        sample["idx"] = data["idx"]

        # Stable source coordinates are required by offline feature stores.
        for key in ("episode_index", "frame_index", "timestamp"):
            if key in data:
                value = data[key]
                sample[key] = value.clone() if torch.is_tensor(value) else value

        # Robometer owns its image/state preprocessing contract. Keep these
        # causal-history tensors byte-for-byte outside FastWAM transforms and
        # normalization.
        for key in (
            "robometer_frames",
            "robometer_states",
            "robometer_history_indices",
        ):
            if key in data:
                sample[key] = data[key]

        # sample = self.tokenizer(sample)
        
        return sample

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess the data for the policy model.
        
        Args:
            data: Dict[str, Any], lerobot sample in raw mcap

        Returns:
            data: Dict[str, Any], processed data including unnormalized action
        """
        assert "action" in data, "Action is required in postprocess"
        data["state"] = data.pop("proprio")
        data = self.action_state_merger.backward(data)
        data = self.normalizer.backward(data)
        data = self.inverse_action_state_transform(data)

        start_obs_step = self.num_obs_steps - 1
        data["action"] = dict_apply(data["action"], lambda x: x[:, start_obs_step:, :])
        return data

    def inverse_action_state_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.action_state_transforms is not None:
            for trans in reversed(self.action_state_transforms):
                data = trans.backward(data)
        return data
