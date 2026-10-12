from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from TMD3.multiview import (
    compose_robotwin_latents,
    split_robotwin_rgb,
)
from fastwam.models.wan22.helpers.loader import (
    _load_registered_model,
    _resolve_configs,
)
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)


def _install_diffusers_036_context_parallel_compat() -> None:
    """Supply the metadata-only Diffusers 0.36 classes on FastWAM's 0.35.

    Giga's native MoT does not enable context parallelism in this experiment,
    but importing the official module still constructs its ``_cp_plan`` class
    attribute.  Diffusers 0.35 has the same attention/model APIs used by the
    MoT, except for these two plan descriptors.
    """

    module_name = "diffusers.models._modeling_parallel"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    compat = types.ModuleType(module_name)

    class _ContextParallelDescriptor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    compat.ContextParallelInput = _ContextParallelDescriptor
    compat.ContextParallelOutput = _ContextParallelDescriptor
    sys.modules[module_name] = compat


def _load_native_giga_class(repo_path: str):
    repo = str(Path(repo_path).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _install_diffusers_036_context_parallel_compat()
    from world_action_model.models import CasualWorldActionTransformer_MoT

    return CasualWorldActionTransformer_MoT


def _load_fastwam_vae(
    *,
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool,
    device: str,
    torch_dtype: torch.dtype,
):
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    )
    vae.eval().requires_grad_(False)
    return vae, str(vae_config.path)


class NativeGigaWorldPolicy(nn.Module):
    """Official GWP-0.5 MoT with the RMBench data/action preprocessing.

    The transformer, its parameter shapes, attention mask, token order and
    forward methods come directly from the pinned official Giga repository.
    This wrapper only adapts our already-normalized 14-D qpos samples to the
    released checkpoint's 16-D action boundary and exposes FastWAM's trainer
    and RMBench streaming-policy interfaces.
    """

    contract_name = "GigaWorld-Policy-0.5-Native-RMBench"
    action_horizon = 16
    condition_latents = 1
    future_latents = 1

    def __init__(
        self,
        *,
        dit: nn.Module,
        vae: nn.Module,
        model_dtype: torch.dtype,
        action_dim: int = 14,
        model_action_dim: int = 16,
        visual_flow_shift: float = 5.0,
        action_flow_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_contract: str = "absolute_qpos_14_padded_to_giga16",
        supervise_padded_action_dims: bool = False,
        condition_encoder_config: Optional[dict[str, Any]] = None,
        model_paths: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.torch_dtype = model_dtype
        self.action_dim = int(action_dim)
        self.model_action_dim = int(model_action_dim)
        self.visual_flow_shift = float(visual_flow_shift)
        self.action_flow_shift = float(action_flow_shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.action_contract = str(action_contract)
        self.supervise_padded_action_dims = bool(supervise_padded_action_dims)
        self.condition_encoder_config = dict(condition_encoder_config or {})
        self.model_paths = dict(model_paths or {})
        if self.action_dim != 14 or self.model_action_dim != 16:
            raise ValueError(
                "The controlled RMBench Giga run requires 14-D qpos padded "
                f"to the released 16-D boundary, got {self.action_dim}->{self.model_action_dim}."
            )
        supported_action_contracts = {
            "absolute_qpos_14_padded_to_giga16",
            "giga_official_current_relative_joint_absolute_gripper_14_padded_to_giga16",
        }
        if self.action_contract not in supported_action_contracts:
            raise ValueError(
                f"Unsupported native-Giga action contract {self.action_contract!r}; "
                f"expected one of {sorted(supported_action_contracts)}."
            )
        if int(self.dit.config.in_action_channels) != self.model_action_dim:
            raise ValueError(
                "Giga input width does not match its checkpoint config: "
                f"model={self.dit.config.in_action_channels}, adapter={self.model_action_dim}."
            )
        if int(self.dit.config.out_action_channels) != self.model_action_dim:
            raise ValueError(
                "Giga output width does not match its checkpoint config: "
                f"model={self.dit.config.out_action_channels}, adapter={self.model_action_dim}."
            )
        self._streaming_condition_latents_by_id: dict[str, torch.Tensor] = {}
        self._streaming_condition_valid_masks_by_id: dict[str, torch.Tensor] = {}
        self._streaming_condition_frame_indices_by_id: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return next(self.dit.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.dit.parameters()).dtype

    @staticmethod
    def _shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _pad_action_boundary(self, tensor: torch.Tensor) -> torch.Tensor:
        if int(tensor.shape[-1]) != self.action_dim:
            raise ValueError(
                f"Expected RMBench {self.action_dim}-D tensor, got {tuple(tensor.shape)}."
            )
        return F.pad(tensor, (0, self.model_action_dim - self.action_dim))

    def _prepare_text(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tokens = tokens.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if tokens.ndim != 3 or int(tokens.shape[-1]) != int(self.dit.config.text_dim):
            raise ValueError(
                "Giga text input must be [B,L,4096], got "
                f"{tuple(tokens.shape)}."
            )
        if mask is not None:
            mask = mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if tuple(mask.shape) != tuple(tokens.shape[:2]):
                raise ValueError(
                    f"Text mask mismatch: tokens={tuple(tokens.shape)}, mask={tuple(mask.shape)}."
                )
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        return tokens

    def _prepare_training_inputs(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        required = (
            "long_image_history_latents",
            "long_image_history_valid_mask",
            "long_image_future_latents",
            "action",
            "action_is_pad",
            "state_history",
            "state_history_mask",
            "progress_subtask_tokens",
            "progress_subtask_mask",
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"Native Giga sample is missing fields: {missing}")

        history = sample["long_image_history_latents"].to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        history_valid = sample["long_image_history_valid_mask"].to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        future = sample["long_image_future_latents"].to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        if history.ndim != 5 or future.ndim != 5:
            raise ValueError("Native Giga cached latents must be [B,C,T,H,W].")
        if not bool(history_valid[:, -1].all().item()):
            raise ValueError("The current/reference latent must be valid for every sample.")
        ref_latents = history[:, :, -1:].contiguous()
        future_latents = future[:, :, :1].contiguous()

        action = sample["action"].to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        action_is_pad = sample["action_is_pad"].to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if tuple(action.shape[1:]) != (self.action_horizon, self.action_dim):
            raise ValueError(
                "Native Giga action target must be [B,16,14], got "
                f"{tuple(action.shape)}."
            )
        action = self._pad_action_boundary(action)

        state = sample["state_history"].to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        state_mask = sample["state_history_mask"].to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if state.ndim != 3 or int(state.shape[-1]) != self.action_dim:
            raise ValueError(f"Native Giga state must be [B,S,14], got {tuple(state.shape)}.")
        if tuple(state_mask.shape) != tuple(state.shape[:2]):
            raise ValueError("Native Giga state mask does not match state tokens.")
        if not bool(state_mask[:, -1].all().item()):
            raise ValueError("The current state token must be valid for every sample.")
        state = self._pad_action_boundary(state[:, -1:].contiguous())

        text = self._prepare_text(
            sample["progress_subtask_tokens"],
            sample.get("progress_subtask_mask"),
        )
        return {
            "ref_latents": ref_latents,
            "future_latents": future_latents,
            "action": action,
            "action_is_pad": action_is_pad,
            "state": state,
            "text": text,
        }

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        del tiled
        inputs = self._prepare_training_inputs(sample)
        ref_latents = inputs["ref_latents"]
        future = inputs["future_latents"]
        action = inputs["action"]
        state = inputs["state"]
        text = inputs["text"]
        action_is_pad = inputs["action_is_pad"]
        batch_size = int(action.shape[0])

        visual_u = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        action_u = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        visual_sigma = self._shift_sigma(visual_u, self.visual_flow_shift)
        action_sigma = self._shift_sigma(action_u, self.action_flow_shift)
        visual_timestep = torch.round(
            visual_sigma * self.num_train_timesteps
        ).clamp(0, self.num_train_timesteps).to(torch.long)
        action_timestep = torch.round(
            action_sigma * self.num_train_timesteps
        ).clamp(0, self.num_train_timesteps).to(torch.long)

        visual_noise = torch.randn_like(future)
        action_noise = torch.randn_like(action)
        visual_sigma_view = visual_sigma.to(future).view(batch_size, 1, 1, 1, 1)
        action_sigma_view = action_sigma.to(action).view(batch_size, 1, 1)
        noisy_future = visual_noise * visual_sigma_view + future * (1 - visual_sigma_view)
        noisy_action = action_noise * action_sigma_view + action * (1 - action_sigma_view)
        visual_target = visual_noise - future
        action_target = action_noise - action

        patch_h, patch_w = (int(x) for x in self.dit.config.patch_size[1:])
        ref_tokens = (int(ref_latents.shape[-2]) // patch_h) * (
            int(ref_latents.shape[-1]) // patch_w
        )
        future_tokens = (int(noisy_future.shape[-2]) // patch_h) * (
            int(noisy_future.shape[-1]) // patch_w
        )
        num_state_tokens = int(state.shape[1])
        num_action_tokens = int(noisy_action.shape[1])
        timestep = torch.zeros(
            batch_size,
            num_state_tokens + ref_tokens + num_action_tokens + future_tokens,
            device=self.device,
            dtype=self.dtype,
        )
        action_start = num_state_tokens + ref_tokens
        timestep[:, action_start : action_start + num_action_tokens] = (
            action_timestep[:, None].to(timestep)
        )
        timestep[:, action_start + num_action_tokens :] = (
            visual_timestep[:, None].to(timestep)
        )
        embodiment_id = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        visual_pred, action_pred = self.dit(
            ref_latents=ref_latents,
            noisy_latents=noisy_future,
            timestep=timestep,
            encoder_hidden_states=text,
            return_dict=False,
            state=state,
            action=noisy_action,
            embodiment_id=embodiment_id,
        )
        video_per_sample = (visual_pred.float() - visual_target.float()).square().mean(
            dim=(1, 2, 3, 4)
        )
        supervised_dim = (
            self.model_action_dim
            if self.supervise_padded_action_dims
            else self.action_dim
        )
        action_element = (
            action_pred[..., :supervised_dim].float()
            - action_target[..., :supervised_dim].float()
        ).square().mean(dim=-1)
        action_valid = (~action_is_pad).to(action_element)
        action_per_sample = (action_element * action_valid).sum(dim=1) / action_valid.sum(
            dim=1
        ).clamp(min=1.0)

        sample_weight = sample.get("sample_weight")
        if sample_weight is None:
            sample_weight = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        else:
            sample_weight = sample_weight.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            ).reshape(batch_size)
        loss_video = (video_per_sample * sample_weight).sum() / batch_size
        loss_action = (action_per_sample * sample_weight).sum() / batch_size
        total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return total, {
            "loss_video": self.loss_lambda_video * loss_video.detach(),
            "loss_action": self.loss_lambda_action * loss_action.detach(),
        }

    @staticmethod
    def _normalize_inference_images(images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected inference image tensor, got {type(images)}.")
        if images.ndim != 5:
            raise ValueError(f"Expected inference images [B,T,C,H,W], got {tuple(images.shape)}.")
        if int(images.shape[2]) != 3:
            raise ValueError(f"Inference image channel dimension must be 3, got {tuple(images.shape)}.")
        return images

    @torch.no_grad()
    def _encode_streaming_camera_chunk(
        self,
        input_images: torch.Tensor,
        *,
        reset: bool,
        stream_id: str,
    ) -> torch.Tensor:
        images = self._normalize_inference_images(input_images)
        video = images.to(device=self.device, dtype=self.torch_dtype).permute(
            0, 2, 1, 3, 4
        ).contiguous()
        if int(video.shape[0]) != 1:
            raise ValueError("Native Giga streaming VAE requires batch size one.")
        high_video, left_video, right_video = split_robotwin_rgb(video)
        wrist_video = torch.stack([left_video, right_video], dim=1).flatten(0, 1)
        prefix = "" if stream_id == "default" else f"{stream_id}:"
        high_latent = self.vae.streaming_encode(
            high_video,
            device=self.device,
            reset=reset,
            stream_id=f"{prefix}high",
        )
        wrist_latents = self.vae.streaming_encode(
            wrist_video,
            device=self.device,
            reset=reset,
            stream_id=f"{prefix}wrists",
        )
        if int(high_latent.shape[0]) != 1:
            raise RuntimeError(
                "Native Giga high-camera latent batch must be one, got "
                f"{tuple(high_latent.shape)}."
            )
        if int(wrist_latents.shape[0]) != 2:
            raise RuntimeError(
                "Native Giga wrist latent batch must contain left and right, got "
                f"{tuple(wrist_latents.shape)}."
            )
        # Wrist cameras are half the height and width of the high camera, so
        # their latent spatial shape is also half-sized.  Preserve that shape
        # when restoring the explicit two-camera axis; using high_latent here
        # incorrectly requested four times as many elements.
        wrist_latents = wrist_latents.reshape(
            1, 2, *wrist_latents.shape[1:]
        )
        return compose_robotwin_latents(
            high_latent,
            wrist_latents[:, 0],
            wrist_latents[:, 1],
        )

    def reset_streaming_history(self) -> None:
        self.vae.reset_streaming_encode()
        self._streaming_condition_latents_by_id.clear()
        self._streaming_condition_valid_masks_by_id.clear()
        self._streaming_condition_frame_indices_by_id.clear()

    @torch.no_grad()
    def initialize_streaming_history(
        self,
        input_images: torch.Tensor,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        images = self._normalize_inference_images(input_images)
        if int(images.shape[1]) != 1:
            raise ValueError("Native Giga initialization requires one RGB frame.")
        stream_id = str(stream_id)
        latent = self._encode_streaming_camera_chunk(
            images, reset=True, stream_id=stream_id
        )
        if int(latent.shape[2]) != 1:
            raise RuntimeError(f"Native Giga reference must be one latent, got {tuple(latent.shape)}.")
        valid = torch.ones((1, 1), device=latent.device, dtype=torch.bool)
        indices = torch.zeros((1, 1), device=latent.device, dtype=torch.long)
        self._streaming_condition_latents_by_id[stream_id] = latent
        self._streaming_condition_valid_masks_by_id[stream_id] = valid
        self._streaming_condition_frame_indices_by_id[stream_id] = indices
        return latent

    @torch.no_grad()
    def update_streaming_history(
        self,
        input_images: torch.Tensor,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        stream_id = str(stream_id)
        if stream_id not in self._streaming_condition_latents_by_id:
            raise RuntimeError(f"Native Giga stream {stream_id!r} is not initialized.")
        images = self._normalize_inference_images(input_images)
        if int(images.shape[1]) != int(self.vae.temporal_downsample_factor):
            raise ValueError("Native Giga streaming update requires four RGB frames.")
        latent = self._encode_streaming_camera_chunk(
            images, reset=False, stream_id=stream_id
        )
        previous_index = self._streaming_condition_frame_indices_by_id[stream_id]
        self._streaming_condition_latents_by_id[stream_id] = latent
        self._streaming_condition_frame_indices_by_id[stream_id] = previous_index + 1
        return latent

    def get_streaming_condition_latents(self, *, stream_id: str = "default") -> torch.Tensor:
        return self._streaming_condition_latents_by_id[str(stream_id)]

    def get_streaming_condition_valid_mask(self, *, stream_id: str = "default") -> torch.Tensor:
        return self._streaming_condition_valid_masks_by_id[str(stream_id)]

    def get_streaming_condition_frame_indices(self, *, stream_id: str = "default") -> torch.Tensor:
        return self._streaming_condition_frame_indices_by_id[str(stream_id)]

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del prompt, input_image
        if int(action_horizon) != self.action_horizon:
            raise ValueError(f"Native Giga requires action_horizon=16, got {action_horizon}.")
        ref_latents = kwargs.pop("precomputed_condition_latents").to(
            device=self.device, dtype=self.dtype
        )
        valid = kwargs.pop("precomputed_condition_valid_mask").to(
            device=self.device, dtype=torch.bool
        )
        kwargs.pop("precomputed_condition_frame_indices")
        if int(ref_latents.shape[2]) != 1 or not bool(valid.all().item()):
            raise ValueError(
                "Native Giga inference requires exactly one valid clean reference latent."
            )
        state = kwargs.pop("proprio")
        state_mask = kwargs.pop("proprio_mask")
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if state_mask.ndim == 1:
            state_mask = state_mask.unsqueeze(0)
        if not bool(state_mask[:, -1].all().item()):
            raise ValueError("Native Giga inference requires a valid current state.")
        state = self._pad_action_boundary(
            state[:, -1:].to(device=self.device, dtype=self.dtype)
        )
        text = kwargs.pop("progress_subtask_tokens")
        text_mask = kwargs.pop("progress_subtask_mask")
        text = self._prepare_text(
            text.unsqueeze(0) if text.ndim == 2 else text,
            text_mask.unsqueeze(0) if text_mask.ndim == 1 else text_mask,
        )
        kwargs.pop("progress_memory_layers", None)
        kwargs.pop("progress_memory_mask", None)
        kwargs.pop("progress_subtask_layers", None)
        kwargs.pop("progress_instruction_layers", None)
        kwargs.pop("progress_instruction_mask", None)
        kwargs.pop("progress_active_idx", None)
        kwargs.pop("progress_plan_length", None)
        kwargs.pop("context", None)
        kwargs.pop("context_mask", None)
        num_inference_steps = int(kwargs.pop("num_inference_steps", 10))
        sigma_shift = kwargs.pop("sigma_shift", None)
        seed = int(kwargs.pop("seed", 0))
        rand_device = str(kwargs.pop("rand_device", "cpu"))
        kwargs.pop("tiled", None)
        kwargs.pop("use_history_kv_cache", None)
        if kwargs:
            raise TypeError(f"Unsupported Native Giga inference args: {sorted(kwargs)}")

        generator = torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (1, self.action_horizon, self.model_action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=(self.action_flow_shift if sigma_shift is None else float(sigma_shift)),
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        ref_tokens = (int(ref_latents.shape[-2]) // 2) * (
            int(ref_latents.shape[-1]) // 2
        )
        num_state_tokens = int(state.shape[1])
        num_action_tokens = int(action.shape[1])
        embodiment_id = torch.zeros(1, device=self.device, dtype=torch.long)
        self.dit._enable_action_only_prefix_cache = True
        self.dit.reset_action_only_prefix_cache()
        try:
            for timestep_value in scheduler.timesteps:
                timestep = torch.zeros(
                    (1, num_state_tokens + ref_tokens + num_action_tokens),
                    device=self.device,
                    dtype=self.dtype,
                )
                action_start = num_state_tokens + ref_tokens
                timestep[:, action_start:] = timestep_value.to(timestep)
                action_pred = self.dit(
                    ref_latents=ref_latents,
                    timestep=timestep,
                    encoder_hidden_states=text,
                    state=state,
                    action=action,
                    embodiment_id=embodiment_id,
                    action_only=True,
                )
                action = scheduler.step(
                    action_pred, timestep_value, action, return_dict=False
                )[0]
        finally:
            self.dit.reset_action_only_prefix_cache()
        return {"action": action[0, :, : self.action_dim].detach().float().cpu()}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "condition_encoder_config": self.condition_encoder_config,
            "native_giga": True,
            "native_giga_input_contract": "state_ref_action_future_v1",
            "action_contract": self.action_contract,
            "supervise_padded_action_dims": self.supervise_padded_action_dims,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        # Native Giga checkpoints are produced locally and contain OmegaConf
        # action/condition metadata, so PyTorch's weights-only loader is not
        # sufficient here.
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = payload.get("dit")
        if state is None:
            raise ValueError("Native Giga checkpoint is missing its complete `dit` state dict.")
        self.dit.load_state_dict(state, strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


def create_native_giga_policy(
    *,
    model_id: str,
    tokenizer_model_id: str,
    giga_repo_path: str,
    giga_world_policy_pretrained_path: str,
    redirect_common_files: bool = True,
    action_dim: int = 14,
    model_action_dim: int = 16,
    visual_flow_shift: float = 5.0,
    action_flow_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    loss_lambda_video: float = 1.0,
    loss_lambda_action: float = 1.0,
    action_contract: str = "absolute_qpos_14_padded_to_giga16",
    supervise_padded_action_dims: bool = False,
    condition_encoder_config=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **unused,
) -> NativeGigaWorldPolicy:
    del unused
    model_cls = _load_native_giga_class(giga_repo_path)
    pretrained = str(Path(giga_world_policy_pretrained_path).expanduser().resolve())
    dit = model_cls.from_pretrained(
        pretrained,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=model_dtype)
    vae, vae_path = _load_fastwam_vae(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
        device=device,
        torch_dtype=model_dtype,
    )
    parameter_count = sum(parameter.numel() for parameter in dit.parameters())
    logger.info(
        "Loaded official native GigaWorld-Policy-0.5: class=%s params=%d "
        "action=%d->%d repo=%s weights=%s",
        type(dit).__name__,
        parameter_count,
        action_dim,
        model_action_dim,
        str(Path(giga_repo_path).resolve()),
        pretrained,
    )
    return NativeGigaWorldPolicy(
        dit=dit,
        vae=vae,
        model_dtype=model_dtype,
        action_dim=action_dim,
        model_action_dim=model_action_dim,
        visual_flow_shift=visual_flow_shift,
        action_flow_shift=action_flow_shift,
        num_train_timesteps=num_train_timesteps,
        loss_lambda_video=loss_lambda_video,
        loss_lambda_action=loss_lambda_action,
        action_contract=action_contract,
        supervise_padded_action_dims=supervise_padded_action_dims,
        condition_encoder_config=(
            {} if condition_encoder_config is None else dict(condition_encoder_config)
        ),
        model_paths={
            "giga_repo": str(Path(giga_repo_path).resolve()),
            "giga_world_policy": pretrained,
            "vae": vae_path,
        },
    )


__all__ = ["NativeGigaWorldPolicy", "create_native_giga_policy"]
