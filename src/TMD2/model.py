from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.helpers.loader import (
    load_wan22_ti2v_5b_components,
)
from fastwam_qwen.models.qwen3vl_mot.dit_experts import (
    ContextFreeWanVideoExpert,
    load_action_dit_for_mot_layers,
)
from fastwam_qwen.models.qwen3vl_mot.fastwam_qwen3vl import FastWAMQwen3VL
from fastwam.utils.logging_config import get_logger

from .robometer_mot import ProgressMoT

logger = get_logger(__name__)


TMD2_ACTION_HORIZON = 16
TMD2_CONDITION_LATENTS = 1
TMD2_FUTURE_LATENTS = 1


class ProgressTMD(FastWAMQwen3VL):
    """TMD2 trained from frozen, offline Progress Model layer features."""

    condition_latents = TMD2_CONDITION_LATENTS
    future_latents = TMD2_FUTURE_LATENTS
    action_horizon = TMD2_ACTION_HORIZON
    contract_name = "TMD2"

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: ProgressMoT,
        vae,
        text_dim: int,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        condition_encoder_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            qwen_expert=None,
            mot=mot,
            vae=vae,
            text_dim=text_dim,
            proprio_dim=None,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            condition_encoder_config=condition_encoder_config,
        )
        self.mot.to(device=self.device, dtype=self.torch_dtype)
        action_dim = int(self.action_expert.action_dim)
        valid_channels = self.condition_encoder_config.get(
            "action_valid_channels"
        )
        if valid_channels is None and action_dim == 30:
            valid_channels = tuple(range(14)) + (28, 29)
        action_channel_mask = torch.ones(action_dim, dtype=torch.bool)
        if valid_channels is not None:
            action_channel_mask.zero_()
            for channel in valid_channels:
                channel = int(channel)
                if channel < 0 or channel >= action_dim:
                    raise ValueError(
                        "Action valid channel is outside the configured "
                        f"dimension: channel={channel}, dim={action_dim}."
                    )
                action_channel_mask[channel] = True
            if not bool(action_channel_mask.any()):
                raise ValueError("At least one action channel must be valid.")
        self.register_buffer(
            "action_channel_mask",
            action_channel_mask,
            persistent=False,
        )
        if int(self.num_condition_latents) != int(self.condition_latents):
            raise ValueError(
                f"{self.contract_name} requires "
                f"condition_encoder_config.num_condition_latents="
                f"{self.condition_latents}, got {self.num_condition_latents}."
            )
        subtask_mode = self.condition_encoder_config.get(
            "cross_attention_subtask_mode",
            "final",
        )
        if subtask_mode not in {"final", "layerwise"}:
            raise ValueError(
                "cross_attention_subtask_mode must be `final` or `layerwise`, "
                f"got {subtask_mode!r}."
            )
        logger.info(
            "TMD2 MoT Progress contract: full_attention=%s cross_attention=%s",
            self.condition_encoder_config.get(
                "full_attention_memory_mode",
                "all",
            ),
            subtask_mode,
        )

    @property
    def qwen_expert(self):
        raise AttributeError("TMD2 replaces Qwen with frozen Progress Model features.")

    def _mask_action_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        if int(tensor.shape[-1]) != int(self.action_channel_mask.numel()):
            raise ValueError(
                "Action tensor width does not match its channel mask: "
                f"tensor={tensor.shape[-1]}, "
                f"mask={self.action_channel_mask.numel()}."
            )
        shape = [1] * tensor.ndim
        shape[-1] = int(self.action_channel_mask.numel())
        return tensor * self.action_channel_mask.to(
            device=tensor.device,
            dtype=tensor.dtype,
        ).view(*shape)

    def _action_loss_per_token(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        element = F.mse_loss(
            prediction.float(),
            target.float(),
            reduction="none",
        )
        shape = [1] * element.ndim
        shape[-1] = int(self.action_channel_mask.numel())
        mask = self.action_channel_mask.to(
            device=element.device,
            dtype=element.dtype,
        ).view(*shape)
        return (element * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(
            min=1.0
        )

    def _select_full_attention_memory(
        self,
        memory_layers: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.condition_encoder_config.get(
            "full_attention_memory_mode",
            "all",
        )
        if mode == "all":
            return memory_layers, memory_mask
        if mode == "state_only":
            if memory_layers.ndim != 4 or memory_mask.ndim != 2:
                raise ValueError(
                    "TMD2 full-attention memory must be [B,L,M,H] with a "
                    f"[B,M] mask, got {tuple(memory_layers.shape)} and "
                    f"{tuple(memory_mask.shape)}."
                )
            state_tokens = int(
                self.condition_encoder_config.get(
                    "full_attention_state_tokens",
                    1,
                )
            )
            if state_tokens <= 0:
                raise ValueError(
                    "full_attention_state_tokens must be positive."
                )
            if (
                memory_layers.shape[2] < state_tokens
                or memory_mask.shape[1] < state_tokens
            ):
                raise ValueError(
                    "TMD2 state-only memory does not contain the configured "
                    f"{state_tokens} state tokens."
                )
            return (
                memory_layers[:, :, :state_tokens],
                memory_mask[:, :state_tokens],
            )
        raise RuntimeError(f"Unhandled full-attention memory mode: {mode!r}.")

    def _progress_subtask_key(self) -> str:
        mode = self.condition_encoder_config.get(
            "cross_attention_subtask_mode",
            "final",
        )
        if mode == "final":
            return "progress_subtask_tokens"
        if mode == "layerwise":
            return "progress_subtask_layers"
        raise RuntimeError(f"Unhandled cross-attention subtask mode: {mode!r}.")

    @classmethod
    def _create_mot(
        cls,
        *,
        video_expert,
        action_expert,
        progress_hidden_dim: int,
        condition_dim: int,
        checkpoint_attention: bool,
        condition_encoder_config: Optional[dict[str, Any]] = None,
    ):
        del condition_encoder_config
        return ProgressMoT(
            video_expert=video_expert,
            action_expert=action_expert,
            progress_hidden_dim=int(progress_hidden_dim),
            condition_dim=int(condition_dim),
            checkpoint_attention=bool(checkpoint_attention),
        )

    def _progress_training_fields(self) -> tuple[str, ...]:
        return (
            "progress_memory_layers",
            "progress_memory_mask",
            self._progress_subtask_key(),
            "progress_subtask_mask",
        )

    def _mot_training_kwargs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        memory_layers, memory_mask = self._select_full_attention_memory(
            inputs["progress_memory_layers"],
            inputs["progress_memory_mask"],
        )
        return {
            "memory_tokens": memory_layers,
            "memory_mask": memory_mask,
            "subtask_tokens": inputs[self._progress_subtask_key()],
            "subtask_mask": inputs["progress_subtask_mask"],
        }

    @classmethod
    def _validate_training_temporal_contract(
        cls,
        inputs: dict[str, Any],
        temporal_factor: int,
    ) -> None:
        expected_condition_video_frames = 1 + (
            int(temporal_factor) * int(cls.condition_latents)
        )
        if int(inputs["num_condition_video_frames"]) != expected_condition_video_frames:
            raise ValueError(
                f"{cls.contract_name} training requires one anchor plus "
                f"{cls.condition_latents}x{temporal_factor} condition frames: "
                f"expected={expected_condition_video_frames}, "
                f"actual={inputs['num_condition_video_frames']}."
            )
        if int(inputs["num_condition_latent_frames"]) != int(cls.condition_latents):
            raise ValueError(
                f"{cls.contract_name} training requires exactly "
                f"{cls.condition_latents} endpoint condition latents, "
                f"got {inputs['num_condition_latent_frames']}."
            )
        future_latents = (
            int(inputs["input_latents"].shape[2])
            - int(inputs["num_condition_latent_frames"])
        )
        if future_latents != int(cls.future_latents):
            raise ValueError(
                f"{cls.contract_name} training predicts exactly one future "
                "latent z2, "
                f"got {future_latents}."
            )
        if int(inputs["action"].shape[1]) != int(cls.action_horizon):
            raise ValueError(
                f"{cls.contract_name} training predicts exactly one "
                f"{cls.action_horizon}-step action chunk, "
                f"got {inputs['action'].shape[1]}."
            )

    def _prepare_training_condition_latents(
        self,
        condition_latents: torch.Tensor,
        condition_latent_valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return clean TMD2 conditions and their zero-timestep default."""
        del condition_latent_valid_mask
        return condition_latents, None

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str,
        torch_dtype: torch.dtype,
        model_id: str,
        tokenizer_model_id: str,
        tokenizer_max_len: int,
        video_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None,
        progress_hidden_dim: int = 2560,
        progress_checkpoint_path: str | None = None,
        redirect_common_files: bool = True,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        condition_encoder_config: Optional[dict[str, Any]] = None,
    ):
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config.text_dim` is required.")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=False,
        )
        video_expert = ContextFreeWanVideoExpert(components.dit)
        action_expert = load_action_dit_for_mot_layers(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = cls._create_mot(
            video_expert=video_expert,
            action_expert=action_expert,
            progress_hidden_dim=int(progress_hidden_dim),
            condition_dim=int(video_dit_config["text_dim"]),
            checkpoint_attention=mot_checkpoint_mixed_attn,
            condition_encoder_config=condition_encoder_config,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_dim=int(video_dit_config["text_dim"]),
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            condition_encoder_config=condition_encoder_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": "DISABLED_PROGRESS_CACHE_PATH",
            "tokenizer": "DISABLED_PROGRESS_CACHE_PATH",
            "progress_model": str(progress_checkpoint_path or "OFFLINE_CACHE"),
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else action_dit_pretrained_path
            ),
        }
        return model

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        self._validate_training_temporal_contract(inputs, temporal_factor)
        required = self._progress_training_fields()
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"TMD2 sample is missing Progress cache fields: {missing}")
        inputs.update(
            {
                key: sample[key].to(
                    device=self.device,
                    dtype=(
                        torch.bool
                        if key.endswith("_mask")
                        else (
                            torch.float32
                            if key == "progress_logits"
                            else self.torch_dtype
                        )
                    ),
                    non_blocking=True,
                )
                for key in required
            }
        )
        return inputs

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        sample_weight = sample.get("sample_weight")
        if sample_weight is None:
            sample_weight = torch.ones(
                batch_size,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            sample_weight = sample_weight.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ).reshape(batch_size)
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        num_condition_video_frames = int(inputs["num_condition_video_frames"])
        num_condition_latent_frames = int(inputs["num_condition_latent_frames"])
        video_latent_is_pad = self._build_video_latent_is_pad(
            image_is_pad,
            num_condition_video_frames=num_condition_video_frames,
            num_condition_latent_frames=num_condition_latent_frames,
            condition_latent_valid_mask=inputs.get(
                "condition_latent_valid_mask"
            ),
        )

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        condition_latents, condition_timestep = (
            self._prepare_training_condition_latents(
                inputs["condition_latents"],
                inputs.get("condition_latent_valid_mask"),
            )
        )
        latents[:, :, :num_condition_latent_frames] = condition_latents

        noise_action = torch.randn_like(action)
        action = self._mask_action_channels(action)
        noise_action = self._mask_action_channels(noise_action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        noisy_action = self._mask_action_channels(noisy_action)
        target_action = self._mask_action_channels(target_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            action=action,
            fuse_vae_embedding_in_latents=inputs[
                "fuse_vae_embedding_in_latents"
            ],
            num_condition_latent_frames=num_condition_latent_frames,
            condition_timestep=condition_timestep,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        attention_mask = self._build_dit_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_condition_latent_frames=num_condition_latent_frames,
            video_latent_is_pad=video_latent_is_pad,
            action_is_pad=action_is_pad,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            dit_attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            **self._mot_training_kwargs(inputs),
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_video = pred_video[:, :, num_condition_latent_frames:]
        target_video = target_video[:, :, num_condition_latent_frames:]

        loss_video_per_sample = self._compute_video_loss_after_condition_prefix(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            num_condition_video_frames=num_condition_video_frames,
        )
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video
        ).to(loss_video_per_sample)
        loss_video = (
            loss_video_per_sample * video_weight * sample_weight
        ).sum() / batch_size

        action_loss_token = self._action_loss_per_token(
            pred_action,
            target_action,
        )
        if action_is_pad is not None:
            valid = (~action_is_pad).to(action_loss_token)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / (
                valid.sum(dim=1).clamp(min=1.0)
            )
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(action_loss_per_sample)
        loss_action = (
            action_loss_per_sample * action_weight * sample_weight
        ).sum() / batch_size

        total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        return total, {
            "loss_video": self.loss_lambda_video * loss_video.detach(),
            "loss_action": self.loss_lambda_action * loss_action.detach(),
        }

    def _prepare_inference_condition_cache(
        self,
        *,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        proprio_mask: Optional[torch.Tensor],
        progress_memory_layers: Optional[torch.Tensor],
        progress_memory_mask: Optional[torch.Tensor],
        progress_instruction_layers: Optional[torch.Tensor],
        progress_instruction_mask: Optional[torch.Tensor],
        progress_subtask_tokens: Optional[torch.Tensor],
        progress_subtask_layers: Optional[torch.Tensor],
        progress_subtask_mask: Optional[torch.Tensor],
        progress_active_idx: Optional[torch.Tensor],
        progress_plan_length: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        del (
            context,
            context_mask,
            proprio,
            proprio_mask,
            progress_instruction_layers,
            progress_instruction_mask,
            progress_active_idx,
            progress_plan_length,
        )
        subtask_key = self._progress_subtask_key()
        selected_subtask = (
            progress_subtask_layers
            if subtask_key == "progress_subtask_layers"
            else progress_subtask_tokens
        )
        required_progress = {
            "progress_memory_layers": progress_memory_layers,
            "progress_memory_mask": progress_memory_mask,
            subtask_key: selected_subtask,
            "progress_subtask_mask": progress_subtask_mask,
        }
        missing = [name for name, value in required_progress.items() if value is None]
        if missing:
            raise ValueError(
                "TMD2 inference requires live Progress Model features: "
                f"{missing}."
            )
        if progress_memory_layers.ndim == 3:
            progress_memory_layers = progress_memory_layers.unsqueeze(0)
        if progress_memory_mask.ndim == 1:
            progress_memory_mask = progress_memory_mask.unsqueeze(0)
        if subtask_key == "progress_subtask_layers":
            if selected_subtask.ndim == 3:
                selected_subtask = selected_subtask.unsqueeze(0)
        elif selected_subtask.ndim == 2:
            selected_subtask = selected_subtask.unsqueeze(0)
        if progress_subtask_mask.ndim == 1:
            progress_subtask_mask = progress_subtask_mask.unsqueeze(0)
        progress_memory_layers, progress_memory_mask = (
            self._select_full_attention_memory(
                progress_memory_layers,
                progress_memory_mask,
            )
        )
        return self.mot.prepare_condition_cache(
            progress_memory_layers.to(self.device, dtype=self.torch_dtype),
            progress_memory_mask.to(self.device, dtype=torch.bool),
            selected_subtask.to(self.device, dtype=self.torch_dtype),
            progress_subtask_mask.to(self.device, dtype=torch.bool),
        )

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        condition_cache: dict[str, Any],
        fuse_vae_embedding_in_latents: bool,
        num_condition_latent_frames: int,
        gt_action: Optional[torch.Tensor] = None,
        video_latent_is_pad: Optional[torch.Tensor] = None,
    ):
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_condition_latent_frames=num_condition_latent_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
        )
        attention_mask = self._build_dit_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_condition_latent_frames=num_condition_latent_frames,
            video_latent_is_pad=video_latent_is_pad,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            dit_attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            condition_cache=condition_cache,
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[Union[str, Sequence[str]]],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        precomputed_condition_latents: Optional[torch.Tensor] = None,
        precomputed_condition_valid_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_mask: Optional[torch.Tensor] = None,
        progress_memory_layers: Optional[torch.Tensor] = None,
        progress_memory_mask: Optional[torch.Tensor] = None,
        progress_instruction_layers: Optional[torch.Tensor] = None,
        progress_instruction_mask: Optional[torch.Tensor] = None,
        progress_subtask_tokens: Optional[torch.Tensor] = None,
        progress_subtask_layers: Optional[torch.Tensor] = None,
        progress_subtask_mask: Optional[torch.Tensor] = None,
        progress_active_idx: Optional[torch.Tensor] = None,
        progress_plan_length: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
        use_history_kv_cache: bool = True,
    ) -> dict[str, Any]:
        del (
            negative_prompt,
            text_cfg_scale,
            test_action_with_infer_action,
        )
        self.eval()
        if prompt is None and (context is None or context_mask is None):
            raise ValueError(
                "TMD2 inference requires either a task instruction or "
                "precomputed context/context_mask."
            )
        input_images = self._normalize_inference_images(input_image)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        expected_condition_frames = 1 + (
            temporal_factor * int(self.condition_latents)
        )
        supplied_condition_latents = precomputed_condition_latents is not None
        input_condition_video_frames = int(input_images.shape[1])
        num_condition_video_frames = (
            expected_condition_frames
            if supplied_condition_latents
            else input_condition_video_frames
        )
        if not self._uses_local_endpoint_condition_encoder(self):
            raise RuntimeError(
                f"{self.contract_name} inference requires "
                "a LingBot temporal history encoder."
            )
        if (
            not supplied_condition_latents
            and num_condition_video_frames != expected_condition_frames
        ):
            raise ValueError(
                f"{self.contract_name} requires one anchor plus "
                f"{self.condition_latents}x{temporal_factor} short-term "
                f"history frames, expected K={expected_condition_frames}, "
                f"got K={num_condition_video_frames}."
            )
        num_future_video_frames = int(num_video_frames) - num_condition_video_frames
        allowed_future_video_frames = {
            0,
            temporal_factor * int(self.future_latents),
        }
        if num_future_video_frames not in allowed_future_video_frames:
            raise ValueError(
                f"{self.contract_name} inference accepts action-only generation "
                "or exactly one "
                f"future Wan latent ({temporal_factor} raw frames), got "
                f"total={num_video_frames}, condition={num_condition_video_frames}."
            )
        if supplied_condition_latents and num_future_video_frames != 0:
            raise ValueError(
                "Precomputed online condition latents currently support "
                "action-only inference."
            )
        if int(action_horizon) != int(self.action_horizon):
            raise ValueError(
                f"{self.contract_name} inference requires exactly one "
                f"{self.action_horizon}-step action chunk, "
                f"got {action_horizon}."
            )
        condition_cache = self._prepare_inference_condition_cache(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            proprio_mask=proprio_mask,
            progress_memory_layers=progress_memory_layers,
            progress_memory_mask=progress_memory_mask,
            progress_instruction_layers=progress_instruction_layers,
            progress_instruction_mask=progress_instruction_mask,
            progress_subtask_tokens=progress_subtask_tokens,
            progress_subtask_layers=progress_subtask_layers,
            progress_subtask_mask=progress_subtask_mask,
            progress_active_idx=progress_active_idx,
            progress_plan_length=progress_plan_length,
        )

        _, _, _, height, width = input_images.shape
        decode_frames = 1 + num_future_video_frames
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, decode_frames
        )
        if (checked_h, checked_w, checked_t) != (
            height,
            width,
            decode_frames,
        ):
            raise ValueError("Input H/W/T do not satisfy Wan latent layout.")
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            action = action.to(device=self.device, dtype=self.torch_dtype)

        num_condition_latent_frames = int(self.condition_latents)
        num_future_latent_frames = num_future_video_frames // temporal_factor
        latent_t = num_condition_latent_frames + num_future_latent_frames
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        latents_video, latents_action = self._sample_joint_initial_noise(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            (1, action_horizon, self.action_expert.action_dim),
            seed=seed,
            rand_device=rand_device,
        )
        latents_video = latents_video.to(device=self.device, dtype=self.torch_dtype)
        latents_action = latents_action.to(device=self.device, dtype=self.torch_dtype)
        latents_action = self._mask_action_channels(latents_action)
        condition_video = input_images.to(
            device=self.device,
            dtype=self.torch_dtype,
        ).permute(0, 2, 1, 3, 4).contiguous()
        if supplied_condition_latents:
            condition_latents = precomputed_condition_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            expected_latent_shape = (
                1,
                int(self.vae.model.z_dim),
                num_condition_latent_frames,
                latent_h,
                latent_w,
            )
            if tuple(condition_latents.shape) != expected_latent_shape:
                raise ValueError(
                    "Precomputed TMD3 history latent shape mismatch: "
                    f"got={tuple(condition_latents.shape)}, "
                    f"expected={expected_latent_shape}."
                )
            if precomputed_condition_valid_mask is None:
                raise ValueError(
                    "Precomputed TMD3 history requires a condition valid mask."
                )
            condition_valid_mask = precomputed_condition_valid_mask.to(
                device=self.device,
                dtype=torch.bool,
            )
            if tuple(condition_valid_mask.shape) == (
                num_condition_latent_frames,
            ):
                condition_valid_mask = condition_valid_mask.unsqueeze(0)
            expected_valid_shape = (1, num_condition_latent_frames)
            if tuple(condition_valid_mask.shape) != expected_valid_shape:
                raise ValueError(
                    "Precomputed TMD3 valid-mask shape mismatch: "
                    f"got={tuple(condition_valid_mask.shape)}, "
                    f"expected={expected_valid_shape}."
                )
            if not bool(condition_valid_mask.any().item()):
                raise ValueError("At least the episode anchor must be valid.")
        else:
            condition_latents = self._encode_local_endpoint_condition_latent(
                self,
                condition_video=condition_video,
                tiled=tiled,
            )
            condition_valid_mask = torch.ones(
                (1, num_condition_latent_frames),
                device=self.device,
                dtype=torch.bool,
            )
        latents_video[:, :, :num_condition_latent_frames] = condition_latents
        video_latent_is_pad = torch.cat(
            [
                ~condition_valid_mask,
                torch.zeros(
                    (
                        int(condition_valid_mask.shape[0]),
                        num_future_latent_frames,
                    ),
                    device=condition_valid_mask.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        # Exact inference-only acceleration: when all Video tokens are fixed
        # history conditions, compute their per-layer K/V once and reuse them
        # throughout the Action diffusion loop.  The cache is intentionally
        # discarded at the end of this replan.
        history_kv_cache = None
        if bool(use_history_kv_cache) and num_future_latent_frames == 0:
            history_pre = self.video_expert.pre_dit(
                x=condition_latents,
                timestep=torch.zeros(
                    (condition_latents.shape[0],),
                    device=condition_latents.device,
                    dtype=condition_latents.dtype,
                ),
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
                num_condition_latent_frames=num_condition_latent_frames,
            )
            history_attention_mask = self._build_dit_attention_mask(
                video_seq_len=history_pre["tokens"].shape[1],
                action_seq_len=0,
                video_tokens_per_frame=int(
                    history_pre["meta"]["tokens_per_frame"]
                ),
                device=history_pre["tokens"].device,
                num_condition_latent_frames=num_condition_latent_frames,
                video_latent_is_pad=video_latent_is_pad,
            )
            history_kv_cache = self.mot.prepare_history_cache(
                video_tokens=history_pre["tokens"],
                video_freqs=history_pre["freqs"],
                video_t_mod=history_pre["t_mod"],
                video_attention_mask=history_attention_mask,
                condition_cache=condition_cache,
            )

        video_schedule = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        action_schedule = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, delta_video, step_t_action, delta_action in zip(
            *video_schedule,
            *action_schedule,
        ):
            if history_kv_cache is None:
                pred_video, pred_action = self._predict_joint_noise(
                    latents_video,
                    latents_action,
                    step_t_video.unsqueeze(0).to(latents_video),
                    step_t_action.unsqueeze(0).to(latents_action),
                    condition_cache,
                    fuse_flag,
                    num_condition_latent_frames,
                    action,
                    video_latent_is_pad,
                )
                latents_video = self.infer_video_scheduler.step(
                    pred_video, delta_video, latents_video
                )
            else:
                action_pre = self.action_expert.pre_dit(
                    action_tokens=latents_action,
                    timestep=step_t_action.unsqueeze(0).to(latents_action),
                )
                attention_mask = self._build_dit_attention_mask(
                    video_seq_len=int(history_kv_cache["video_length"]),
                    action_seq_len=action_pre["tokens"].shape[1],
                    video_tokens_per_frame=int(
                        history_pre["meta"]["tokens_per_frame"]
                    ),
                    device=action_pre["tokens"].device,
                    num_condition_latent_frames=num_condition_latent_frames,
                    video_latent_is_pad=video_latent_is_pad,
                )
                action_tokens = self.mot.forward_action_with_history_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=action_pre["freqs"],
                    action_t_mod=action_pre["t_mod"],
                    dit_attention_mask=attention_mask,
                    condition_cache=condition_cache,
                    history_cache=history_kv_cache,
                )
                pred_action = self.action_expert.post_dit(
                    action_tokens,
                    action_pre,
                )
            latents_action = self.infer_action_scheduler.step(
                pred_action, delta_action, latents_action
            )
            latents_action = self._mask_action_channels(latents_action)
            latents_video[:, :, :num_condition_latent_frames] = condition_latents

        decoded_video = []
        if not supplied_condition_latents:
            decoded_video = self._condition_images_to_pil(input_images)
        if num_future_video_frames > 0 and not supplied_condition_latents:
            current_anchor = self._encode_video_latents(
                condition_video[:, :, -1:], tiled=tiled
            )
            future_latents = latents_video[:, :, num_condition_latent_frames:]
            decoded_anchor_and_future = self._decode_latents(
                torch.cat([current_anchor, future_latents], dim=2),
                tiled=tiled,
            )
            decoded_video.extend(decoded_anchor_and_future[1:])
        if (
            not supplied_condition_latents
            and len(decoded_video) != int(num_video_frames)
        ):
            raise RuntimeError(
                "Decoded video length does not match condition+future contract: "
                f"got={len(decoded_video)}, expected={num_video_frames}."
            )
        return {
            "video": decoded_video,
            "action": latents_action[0].detach().float().cpu(),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        **kwargs,
    ):
        precomputed_condition_latents = kwargs.get(
            "precomputed_condition_latents"
        )
        if precomputed_condition_latents is None:
            num_video_frames = self._infer_condition_video_frames(input_image)
        else:
            num_video_frames = 1 + (
                int(self.vae.temporal_downsample_factor)
                * int(self.condition_latents)
            )
        output = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            **kwargs,
        )
        return {"action": output["action"]}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        **kwargs,
    ):
        if action_horizon is None:
            if action is None:
                raise ValueError("`action_horizon` is required when action is absent.")
            action_horizon = int(action.shape[-2])
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            **kwargs,
        )


RobometerTMD = ProgressTMD
