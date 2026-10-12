from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.utils.logging_config import get_logger

from .dit_experts import ContextFreeWanVideoExpert, load_action_dit_for_mot_layers
from .history_aggregator import SpatialTemporalHistoryAggregator
from .qwen_expert import Qwen3VLExpert
from .trimot import TriMoT

logger = get_logger(__name__)


class FastWAMQwen3VL(FastWAM):
    """FastWAM side path that replaces T5 conditioning with Qwen3-VL MoT tokens."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        qwen_expert: Qwen3VLExpert,
        mot: TriMoT,
        vae,
        text_dim: int,
        proprio_dim: Optional[int] = None,
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
            mot=mot,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            text_dim=int(text_dim),
            proprio_dim=proprio_dim,
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
        )
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(device=self.device, dtype=self.torch_dtype)

        self.condition_encoder_config = self._normalize_condition_encoder_config(
            condition_encoder_config
        )
        self.condition_encoder_mode = self.condition_encoder_config["mode"]
        self.num_history_frames = int(self.condition_encoder_config["num_history_frames"])
        self.num_condition_latents = int(
            self.condition_encoder_config.get("num_condition_latents", 1)
        )
        self.history_condition_noise_prob = float(
            self.condition_encoder_config.get("history_condition_noise_prob", 0.0)
        )
        self.history_condition_noise_u_min = float(
            self.condition_encoder_config.get("history_condition_noise_u_min", 0.5)
        )
        self.history_condition_noise_u_max = float(
            self.condition_encoder_config.get("history_condition_noise_u_max", 1.0)
        )
        self.history_condition_noise_sampling = str(
            self.condition_encoder_config.get(
                "history_condition_noise_sampling",
                "lingbot_timestep",
            )
        )
        self.history_encode_batch_size = int(
            self.condition_encoder_config.get("history_encode_batch_size", 4)
        )
        self.vae_encode_batch_size = int(
            self.condition_encoder_config.get("vae_encode_batch_size", 4)
        )
        self.history_aggregator = None
        if self._uses_independent_condition_encoder(self):
            latent_dim = getattr(getattr(self.vae, "model", None), "z_dim", None)
            if latent_dim is None:
                raise ValueError(
                    "`vae.model.z_dim` is required to construct the independent history aggregator."
                )
            self.history_aggregator = SpatialTemporalHistoryAggregator(
                latent_dim=int(latent_dim),
                num_history_frames=self.num_history_frames,
                hidden_dim=int(self.condition_encoder_config["aggregator"]["hidden_dim"]),
            ).to(device=self.device, dtype=self.torch_dtype)

    @property
    def qwen_expert(self) -> Qwen3VLExpert:
        return self.mot.mixtures["qwen"]

    @staticmethod
    def _normalize_condition_encoder_config(
        condition_encoder_config: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        config = {} if condition_encoder_config is None else dict(condition_encoder_config)
        allowed_keys = {
            "mode",
            "num_history_frames",
            "num_condition_latents",
            "history_condition_noise_prob",
            "history_condition_noise_u_min",
            "history_condition_noise_u_max",
            "history_condition_noise_sampling",
            "history_encode_batch_size",
            "vae_encode_batch_size",
            "aggregator",
            "full_attention_memory_mode",
            "full_attention_state_tokens",
            "progress_bins",
            "progress_bottleneck_dim",
            "progress_gate_init",
            "progress_injection_mode",
            "state_dim",
            "state_history_tokens",
            "state_history_latents",
            "state_steps_per_latent",
            "state_control_offsets",
            "state_time_start",
            "state_time_end",
            "state_time_gate_init",
            "state_temporal_encoding",
            "state_cross_attention_scope",
            "video_time_offset",
            "video_tokens_per_frame",
            "action_time_start",
            "action_time_end",
            "action_valid_channels",
            "cross_attention_instruction_mode",
            "cross_attention_subtask_mode",
            "subtask_condition_mode",
        }
        unexpected_keys = sorted(set(config) - allowed_keys)
        if unexpected_keys:
            raise ValueError(
                "Unsupported `condition_encoder_config` keys: "
                f"{unexpected_keys}."
            )

        mode = str(config.get("mode", "legacy_wan_k5")).strip().lower()
        supported_modes = {
            "legacy_wan_k5",
            "independent_history_current",
            "lingbot_local_endpoint",
            "lingbot_streaming_history",
        }
        if mode not in supported_modes:
            raise ValueError(
                "Unsupported condition encoder mode: "
                f"{mode!r}. Expected one of {sorted(supported_modes)}."
            )
        num_history_frames = int(config.get("num_history_frames", 4))
        if num_history_frames <= 0:
            raise ValueError(
                "`condition_encoder_config.num_history_frames` must be positive, "
                f"got {num_history_frames}."
            )
        num_condition_latents = int(config.get("num_condition_latents", 1))
        if num_condition_latents <= 0:
            raise ValueError(
                "`condition_encoder_config.num_condition_latents` must be positive, "
                f"got {num_condition_latents}."
            )
        history_condition_noise_prob = float(
            config.get("history_condition_noise_prob", 0.0)
        )
        if not 0.0 <= history_condition_noise_prob <= 1.0:
            raise ValueError(
                "`history_condition_noise_prob` must be in [0,1], got "
                f"{history_condition_noise_prob}."
            )
        history_condition_noise_u_min = float(
            config.get("history_condition_noise_u_min", 0.5)
        )
        history_condition_noise_u_max = float(
            config.get("history_condition_noise_u_max", 1.0)
        )
        if not (
            0.0 <= history_condition_noise_u_min
            <= history_condition_noise_u_max
            <= 1.0
        ):
            raise ValueError(
                "History condition-noise u bounds must satisfy "
                "0 <= min <= max <= 1, got "
                f"({history_condition_noise_u_min}, "
                f"{history_condition_noise_u_max})."
            )
        history_condition_noise_sampling = str(
            config.get(
                "history_condition_noise_sampling",
                "lingbot_timestep",
            )
        ).strip().lower()
        supported_noise_sampling = {
            "lingbot_timestep",
            "uniform_mix_per_sample",
        }
        if history_condition_noise_sampling not in supported_noise_sampling:
            raise ValueError(
                "Unsupported history condition-noise sampling mode: "
                f"{history_condition_noise_sampling!r}. Expected one of "
                f"{sorted(supported_noise_sampling)}."
            )
        history_encode_batch_size = int(config.get("history_encode_batch_size", 4))
        if history_encode_batch_size <= 0:
            raise ValueError(
                "`condition_encoder_config.history_encode_batch_size` must be positive, "
                f"got {history_encode_batch_size}."
            )
        vae_encode_batch_size = int(config.get("vae_encode_batch_size", 4))
        if vae_encode_batch_size <= 0:
            raise ValueError(
                "`condition_encoder_config.vae_encode_batch_size` must be positive, "
                f"got {vae_encode_batch_size}."
            )
        aggregator = config.get("aggregator", {})
        if aggregator is None:
            aggregator = {}
        if not isinstance(aggregator, dict):
            raise TypeError(
                "`condition_encoder_config.aggregator` must be a mapping, "
                f"got {type(aggregator)}."
            )
        unexpected_aggregator_keys = sorted(set(aggregator) - {"hidden_dim"})
        if unexpected_aggregator_keys:
            raise ValueError(
                "Unsupported `condition_encoder_config.aggregator` keys: "
                f"{unexpected_aggregator_keys}."
            )
        hidden_dim = int(aggregator.get("hidden_dim", 128))
        if hidden_dim <= 0:
            raise ValueError(
                "`condition_encoder_config.aggregator.hidden_dim` must be positive, "
                f"got {hidden_dim}."
            )
        normalized = {
            "mode": mode,
            "num_history_frames": num_history_frames,
        }
        if "num_condition_latents" in config:
            normalized["num_condition_latents"] = num_condition_latents
        if "history_condition_noise_prob" in config:
            normalized["history_condition_noise_prob"] = (
                history_condition_noise_prob
            )
        if "history_condition_noise_u_min" in config:
            normalized["history_condition_noise_u_min"] = (
                history_condition_noise_u_min
            )
        if "history_condition_noise_u_max" in config:
            normalized["history_condition_noise_u_max"] = (
                history_condition_noise_u_max
            )
        if "history_condition_noise_sampling" in config:
            normalized["history_condition_noise_sampling"] = (
                history_condition_noise_sampling
            )
        if "full_attention_memory_mode" in config:
            full_attention_memory_mode = str(
                config["full_attention_memory_mode"]
            ).strip().lower()
            supported_memory_modes = {
                "all",
                "state_only",
                "current_progress_state_stream",
                "compressed_progress",
                "disabled",
            }
            if full_attention_memory_mode not in supported_memory_modes:
                raise ValueError(
                    "Unsupported full-attention memory mode: "
                    f"{full_attention_memory_mode!r}. Expected one of "
                    f"{sorted(supported_memory_modes)}."
                )
            normalized["full_attention_memory_mode"] = (
                full_attention_memory_mode
            )
        if "full_attention_state_tokens" in config:
            full_attention_state_tokens = int(
                config["full_attention_state_tokens"]
            )
            if full_attention_state_tokens <= 0:
                raise ValueError(
                    "full_attention_state_tokens must be positive."
                )
            normalized["full_attention_state_tokens"] = (
                full_attention_state_tokens
            )
        if "state_dim" in config:
            state_dim = int(config["state_dim"])
            if state_dim <= 0:
                raise ValueError("state_dim must be positive.")
            normalized["state_dim"] = state_dim
        if "state_history_tokens" in config:
            state_history_tokens = int(config["state_history_tokens"])
            if state_history_tokens <= 0:
                raise ValueError("state_history_tokens must be positive.")
            normalized["state_history_tokens"] = state_history_tokens
        for key in (
            "state_history_latents",
            "state_steps_per_latent",
            "video_tokens_per_frame",
        ):
            if key in config:
                value = int(config[key])
                if value <= 0:
                    raise ValueError(f"{key} must be positive.")
                normalized[key] = value
        for key in ("progress_bins", "progress_bottleneck_dim"):
            if key in config:
                value = int(config[key])
                if value <= 0:
                    raise ValueError(f"{key} must be positive.")
                normalized[key] = value
        for key in ("progress_gate_init", "state_time_gate_init"):
            if key in config:
                normalized[key] = float(config[key])
        if "progress_injection_mode" in config:
            progress_injection_mode = str(
                config["progress_injection_mode"]
            ).strip().lower()
            if progress_injection_mode not in {
                "full_attention_kv",
                "cross_attention_video_only",
                "disabled",
            }:
                raise ValueError(
                    "progress_injection_mode must be `full_attention_kv`, "
                    "`cross_attention_video_only`, or `disabled`, got "
                    f"{progress_injection_mode!r}."
                )
            normalized["progress_injection_mode"] = progress_injection_mode
        if "state_temporal_encoding" in config:
            normalized["state_temporal_encoding"] = bool(
                config["state_temporal_encoding"]
            )
        if "state_cross_attention_scope" in config:
            state_cross_attention_scope = str(
                config["state_cross_attention_scope"]
            ).strip().lower()
            if state_cross_attention_scope not in {
                "video_action",
                "action_only",
            }:
                raise ValueError(
                    "state_cross_attention_scope must be `video_action` or "
                    f"`action_only`, got {state_cross_attention_scope!r}."
                )
            normalized["state_cross_attention_scope"] = (
                state_cross_attention_scope
            )
        if "state_control_offsets" in config:
            state_control_offsets = [
                int(value) for value in config["state_control_offsets"]
            ]
            if len(state_control_offsets) != int(
                config.get("state_history_tokens", len(state_control_offsets))
            ):
                raise ValueError(
                    "state_control_offsets must match state_history_tokens."
                )
            normalized["state_control_offsets"] = state_control_offsets
        if "action_valid_channels" in config:
            action_valid_channels = [
                int(value) for value in config["action_valid_channels"]
            ]
            if (
                not action_valid_channels
                or any(value < 0 for value in action_valid_channels)
                or len(set(action_valid_channels)) != len(action_valid_channels)
            ):
                raise ValueError(
                    "action_valid_channels must be a non-empty list of "
                    "unique non-negative channel indices."
                )
            normalized["action_valid_channels"] = action_valid_channels
        for key, default in (
            ("state_time_start", 0.0),
            ("state_time_end", 4.0),
            ("video_time_offset", 1.0),
            ("action_time_start", 4.0),
            ("action_time_end", 5.0),
        ):
            if key in config:
                normalized[key] = float(config.get(key, default))
        if "cross_attention_instruction_mode" in config:
            instruction_mode = str(
                config["cross_attention_instruction_mode"]
            ).strip().lower()
            if instruction_mode not in {"layerwise", "disabled"}:
                raise ValueError(
                    "cross_attention_instruction_mode must be `layerwise` or `disabled`."
                )
            normalized["cross_attention_instruction_mode"] = instruction_mode
        if "cross_attention_subtask_mode" in config:
            cross_attention_subtask_mode = str(
                config["cross_attention_subtask_mode"]
            ).strip().lower()
            if cross_attention_subtask_mode not in {"final", "layerwise"}:
                raise ValueError(
                    "cross_attention_subtask_mode must be `final` or "
                    f"`layerwise`, got {cross_attention_subtask_mode!r}."
                )
            normalized["cross_attention_subtask_mode"] = (
                cross_attention_subtask_mode
            )
        if "subtask_condition_mode" in config:
            subtask_condition_mode = str(
                config["subtask_condition_mode"]
            ).strip().lower()
            if subtask_condition_mode not in {
                "progress_hidden",
                "t5_text_context",
            }:
                raise ValueError(
                    "subtask_condition_mode must be `progress_hidden` or "
                    f"`t5_text_context`, got {subtask_condition_mode!r}."
                )
            normalized["subtask_condition_mode"] = subtask_condition_mode
        if mode == "independent_history_current":
            normalized["history_encode_batch_size"] = history_encode_batch_size
            normalized["aggregator"] = {"hidden_dim": hidden_dim}
        elif mode in {
            "lingbot_local_endpoint",
            "lingbot_streaming_history",
        }:
            normalized["vae_encode_batch_size"] = vae_encode_batch_size
        return normalized

    @staticmethod
    def _uses_independent_condition_encoder(model) -> bool:
        return (
            getattr(model, "condition_encoder_mode", "legacy_wan_k5")
            == "independent_history_current"
        )

    @staticmethod
    def _uses_local_endpoint_condition_encoder(model) -> bool:
        return (
            getattr(model, "condition_encoder_mode", "legacy_wan_k5")
            in {
                "lingbot_local_endpoint",
                "lingbot_streaming_history",
            }
        )

    @staticmethod
    def _encode_video_latents_batched(
        model,
        video: torch.Tensor,
        tiled: bool,
    ) -> torch.Tensor:
        """Encode independent batch items together without changing temporal layout."""
        if bool(getattr(model, "camera_latent_concat_enabled", False)):
            return model._encode_video_latents(video, tiled=tiled)

        single_encode = getattr(getattr(model, "vae", None), "single_encode", None)
        if tiled or not callable(single_encode):
            return model._encode_video_latents(video, tiled=tiled)

        encode_batch_size = int(getattr(model, "vae_encode_batch_size", 4))
        if encode_batch_size <= 0:
            raise ValueError(
                "`vae_encode_batch_size` must be positive, "
                f"got {encode_batch_size}."
            )
        with torch.no_grad():
            chunks = [
                single_encode(
                    video[start : start + encode_batch_size],
                    device=model.device,
                )
                for start in range(0, video.shape[0], encode_batch_size)
            ]
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    @staticmethod
    def _encode_local_endpoint_condition_latent(
        model,
        condition_video: torch.Tensor,
        tiled: bool,
    ) -> torch.Tensor:
        """Encode Wan history continuously or as explicit local-reset windows."""
        if condition_video.ndim != 5:
            raise ValueError(
                f"`condition_video` must be [B,3,K,H,W], got {tuple(condition_video.shape)}."
            )
        if condition_video.shape[1] != 3:
            raise ValueError(
                f"`condition_video` must have 3 channels, got {condition_video.shape[1]}."
            )
        temporal_factor = int(model.vae.temporal_downsample_factor)
        num_condition_latents = int(getattr(model, "num_condition_latents", 1))
        configured_history_frames = int(getattr(model, "num_history_frames", temporal_factor))
        if configured_history_frames != temporal_factor:
            raise ValueError(
                "The LingBot temporal encoder requires one Wan temporal chunk "
                f"of history: num_history_frames={temporal_factor}, "
                f"got {configured_history_frames}."
            )
        expected_condition_frames = 1 + temporal_factor * num_condition_latents
        if condition_video.shape[2] != expected_condition_frames:
            raise ValueError(
                "The LingBot temporal encoder requires one anchor plus chronological "
                f"{num_condition_latents}x{temporal_factor}-frame transitions "
                "ending at current, "
                f"got condition_T={condition_video.shape[2]}, "
                f"expected={expected_condition_frames}."
            )

        if not bool(
            getattr(model, "independent_local_reset_endpoints", False)
        ):
            all_condition_latents = (
                FastWAMQwen3VL._encode_video_latents_batched(
                    model,
                    condition_video,
                    tiled=tiled,
                )
            )
            expected_latents = 1 + num_condition_latents
            if (
                all_condition_latents.ndim != 5
                or all_condition_latents.shape[2] != expected_latents
            ):
                raise RuntimeError(
                    "Continuous Wan history encoding must emit one anchor "
                    f"followed by {num_condition_latents} history latents, "
                    f"got {tuple(all_condition_latents.shape)}."
                )
            return all_condition_latents[:, :, 1:].contiguous()

        batch_size, channels, _, height, width = condition_video.shape
        endpoint_windows = torch.stack(
            [
                condition_video[
                    :,
                    :,
                    endpoint_idx * temporal_factor : (
                        endpoint_idx + 1
                    )
                    * temporal_factor
                    + 1,
                ]
                for endpoint_idx in range(num_condition_latents)
            ],
            dim=1,
        ).reshape(
            batch_size * num_condition_latents,
            channels,
            temporal_factor + 1,
            height,
            width,
        )
        endpoint_latents = FastWAMQwen3VL._encode_video_latents_batched(
            model,
            endpoint_windows,
            tiled=tiled,
        )
        if (
            endpoint_latents.ndim != 5
            or endpoint_latents.shape[0]
            != batch_size * num_condition_latents
            or endpoint_latents.shape[2] != 2
        ):
            raise RuntimeError(
                "Each Wan local-reset endpoint window must emit one anchor "
                "and one transition latent, got "
                f"{tuple(endpoint_latents.shape)}."
            )
        _, latent_channels, _, latent_height, latent_width = (
            endpoint_latents.shape
        )
        return (
            endpoint_latents[:, :, 1]
            .reshape(
                batch_size,
                num_condition_latents,
                latent_channels,
                latent_height,
                latent_width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @staticmethod
    def _encode_history_frames_independently(
        model,
        history_video: torch.Tensor,
        tiled: bool,
    ) -> torch.Tensor:
        """Encode each history image as its own one-frame Wan VAE input."""
        if history_video.ndim != 5:
            raise ValueError(
                f"`history_video` must be [B,3,K,H,W], got {tuple(history_video.shape)}."
            )
        batch_size, channels, num_history_frames, height, width = history_video.shape
        if channels != 3:
            raise ValueError(f"`history_video` must have 3 channels, got {channels}.")
        expected_num_history_frames = int(getattr(model, "num_history_frames", 4))
        if num_history_frames != expected_num_history_frames:
            raise ValueError(
                "Independent condition encoder history length mismatch: "
                f"got {num_history_frames}, expected {expected_num_history_frames}."
            )
        flattened_history_video = history_video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_history_frames,
            channels,
            1,
            height,
            width,
        ).contiguous()
        # WanVideoVAE.encode loops over batch elements.  Its single_encode path
        # accepts a real batch, so use small batches of one-frame histories
        # instead of launching one VAE call per history image.  The bounded
        # chunk size avoids turning B*4 history images into a large VAE peak.
        batched_single_encode = getattr(getattr(model, "vae", None), "single_encode", None)
        if callable(batched_single_encode) and not tiled:
            history_encode_batch_size = int(
                getattr(model, "history_encode_batch_size", 4)
            )
            if history_encode_batch_size <= 0:
                raise ValueError(
                    "`history_encode_batch_size` must be positive, "
                    f"got {history_encode_batch_size}."
                )
            with torch.no_grad():
                flattened_history_latents = torch.cat(
                    [
                        batched_single_encode(
                            flattened_history_video[start : start + history_encode_batch_size],
                            device=model.device,
                        )
                        for start in range(
                            0,
                            flattened_history_video.shape[0],
                            history_encode_batch_size,
                        )
                    ],
                    dim=0,
                )
        else:
            flattened_history_latents = model._encode_video_latents(
                flattened_history_video,
                tiled=tiled,
            )
        if (
            flattened_history_latents.ndim != 5
            or flattened_history_latents.shape[0] != batch_size * num_history_frames
            or flattened_history_latents.shape[2] != 1
        ):
            raise RuntimeError(
                "Independent history frames must each encode to one Wan latent frame, "
                f"got {tuple(flattened_history_latents.shape)}."
            )
        _, latent_dim, _, latent_height, latent_width = flattened_history_latents.shape
        return flattened_history_latents[:, :, 0].reshape(
            batch_size,
            num_history_frames,
            latent_dim,
            latent_height,
            latent_width,
        )

    @staticmethod
    def _build_independent_condition_latents(
        model,
        condition_video: torch.Tensor,
        current_latent: torch.Tensor,
        history_is_pad: Optional[torch.Tensor],
        tiled: bool,
    ) -> torch.Tensor:
        """Build the Video DiT condition order ``[z_history, z_current]``."""
        if condition_video.ndim != 5:
            raise ValueError(
                f"`condition_video` must be [B,3,K,H,W], got {tuple(condition_video.shape)}."
            )
        expected_condition_frames = int(getattr(model, "num_history_frames", 4)) + 1
        if condition_video.shape[2] != expected_condition_frames:
            raise ValueError(
                "Independent condition encoder requires four history images plus one current image: "
                f"got K={condition_video.shape[2]}, expected {expected_condition_frames}."
            )
        if current_latent.ndim != 5 or current_latent.shape[2] != 1:
            raise ValueError(
                "`current_latent` must be [B,C,1,H,W], "
                f"got {tuple(current_latent.shape)}."
            )
        if current_latent.shape[0] != condition_video.shape[0]:
            raise ValueError(
                "Current/history batch mismatch: "
                f"{current_latent.shape[0]} vs {condition_video.shape[0]}."
            )
        history_aggregator = getattr(model, "history_aggregator", None)
        if history_aggregator is None:
            raise RuntimeError(
                "Independent condition encoder is enabled but `history_aggregator` is not initialized."
            )

        history_latents = FastWAMQwen3VL._encode_history_frames_independently(
            model,
            condition_video[:, :, :-1],
            tiled=tiled,
        )
        history_latent = history_aggregator(
            history_latents,
            history_is_pad=history_is_pad,
            fallback_latent=current_latent,
        )
        if history_latent.ndim != 4:
            raise RuntimeError(
                "History aggregator must return [B,C,H,W], "
                f"got {tuple(history_latent.shape)}."
            )
        if history_latent.shape != current_latent[:, :, 0].shape:
            raise RuntimeError(
                "History/current latent shapes must match, "
                f"got {tuple(history_latent.shape)} vs {tuple(current_latent[:, :, 0].shape)}."
            )
        return torch.cat([history_latent.unsqueeze(2), current_latent], dim=2)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = False,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        mot_checkpoint_qwen_post: Optional[bool] = None,
        qwen_model_path: str = "/home/cenzl/VLA/base_model/Qwen3-VL-2B-Instruct",
        qwen_num_mot_layers: Optional[int] = None,
        qwen_attn_implementation: str = "sdpa",
        qwen_condition_mode: str = "cross_attn",
        qwen_condition_dim: int = 4096,
        qwen_context_target_rms: float = 0.07,
        qwen_use_gradient_checkpointing: bool = False,
        freeze_qwen: bool = False,
        freeze_qwen_vision: bool = False,
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
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAMQwen3VL.")
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for FastWAMQwen3VL.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required by the underlying Wan DiT constructor.")
        if load_text_encoder:
            logger.warning("FastWAMQwen3VL ignores `load_text_encoder=True`; T5 is not loaded in this path.")

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
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for TriMoT.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for TriMoT.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        qwen_expert = Qwen3VLExpert(
            model_path=qwen_model_path,
            shared_num_heads=int(video_expert.num_heads),
            shared_attn_head_dim=int(video_expert.attn_head_dim),
            num_mot_layers=qwen_num_mot_layers,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=qwen_attn_implementation,
            freeze_qwen=freeze_qwen,
            freeze_vision=freeze_qwen_vision,
            condition_dim=qwen_condition_dim,
            condition_target_rms=qwen_context_target_rms,
            use_gradient_checkpointing=qwen_use_gradient_checkpointing,
        )
        if int(qwen_expert.num_layers) > int(len(video_expert.blocks)):
            raise ValueError(
                "Qwen MoT layer count cannot exceed video/action DiT layers in this path: "
                f"qwen={qwen_expert.num_layers}, video={len(video_expert.blocks)}. "
                "Use fewer Qwen MoT layers or a deeper DiT."
            )

        mot = TriMoT(
            qwen_expert=qwen_expert,
            video_expert=video_expert,
            action_expert=action_expert,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            mot_checkpoint_qwen_post=mot_checkpoint_qwen_post,
            condition_mode=qwen_condition_mode,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            qwen_expert=qwen_expert,
            mot=mot,
            vae=components.vae,
            text_dim=int(qwen_expert.hidden_dim),
            proprio_dim=proprio_dim,
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
            "text_encoder": "DISABLED_QWEN3VL_PATH",
            "tokenizer": "DISABLED_QWEN3VL_PATH",
            "qwen3vl": qwen_model_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Persist the condition encoder metadata and trainable MoT weights."""
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "condition_encoder_config": self.condition_encoder_config,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.history_aggregator is not None:
            payload["history_aggregator"] = self.history_aggregator.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        saved_condition_encoder_config = payload.get("condition_encoder_config")
        if (
            saved_condition_encoder_config is not None
            and dict(saved_condition_encoder_config) != self.condition_encoder_config
        ):
            logger.warning(
                "Checkpoint condition encoder config differs from the active config: "
                "checkpoint=%s active=%s",
                saved_condition_encoder_config,
                self.condition_encoder_config,
            )
        if self.history_aggregator is not None:
            history_aggregator_state = payload.get("history_aggregator")
            if history_aggregator_state is None:
                logger.warning(
                    "Checkpoint has no `history_aggregator` weights. Keeping the newly initialized "
                    "aggregator; this is expected when starting from Wan or a legacy K5 checkpoint."
                )
            else:
                self.history_aggregator.load_state_dict(history_aggregator_state, strict=True)
        elif "history_aggregator" in payload:
            logger.warning(
                "Checkpoint contains `history_aggregator` weights but the active "
                "condition encoder mode is %s; ignoring them.",
                self.condition_encoder_mode,
            )
        return payload

    def _normalize_prompts(self, prompt: Union[str, Sequence[str]], batch_size: int) -> list[str]:
        if isinstance(prompt, str):
            return [prompt] * batch_size
        prompts = list(prompt)
        if len(prompts) != batch_size:
            raise ValueError(f"Prompt batch mismatch: got {len(prompts)} prompts for batch={batch_size}")
        return [str(p) for p in prompts]

    @staticmethod
    def _append_text_position_ids(
        position_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        append_count: int,
    ) -> torch.Tensor:
        append_count = int(append_count)
        if append_count <= 0:
            return position_ids
        if valid_mask is None:
            valid_mask = torch.ones(
                position_ids.shape[-2:],
                dtype=torch.bool,
                device=position_ids.device,
            )
        valid_mask = valid_mask.to(device=position_ids.device, dtype=torch.bool)

        if position_ids.ndim == 3:
            mask = valid_mask.unsqueeze(0)
            masked_position_ids = torch.where(
                mask,
                position_ids,
                torch.full_like(position_ids, -1),
            )
            next_position = masked_position_ids.amax(dim=(0, 2)).clamp_min(-1) + 1
            offsets = torch.arange(append_count, device=position_ids.device, dtype=position_ids.dtype)
            append_position = next_position.to(dtype=position_ids.dtype).view(1, position_ids.shape[1], 1)
            append_position = append_position + offsets.view(1, 1, append_count)
            append_position = append_position.expand(position_ids.shape[0], -1, -1)
            return torch.cat([position_ids, append_position], dim=2)

        if position_ids.ndim == 2:
            masked_position_ids = torch.where(
                valid_mask,
                position_ids,
                torch.full_like(position_ids, -1),
            )
            next_position = masked_position_ids.amax(dim=1).clamp_min(-1) + 1
            offsets = torch.arange(append_count, device=position_ids.device, dtype=position_ids.dtype)
            append_position = next_position.to(dtype=position_ids.dtype).view(position_ids.shape[0], 1)
            append_position = append_position + offsets.view(1, append_count)
            return torch.cat([position_ids, append_position], dim=1)

        raise ValueError(f"`position_ids` must be 2D or 3D, got shape {tuple(position_ids.shape)}")

    @staticmethod
    def _append_text_position_id(
        position_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return FastWAMQwen3VL._append_text_position_ids(position_ids, valid_mask, append_count=1)

    def _prepare_training_proprio(self, sample, batch_size: int) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            return None
        proprio = sample.get("qwen_proprio")
        if proprio is None:
            proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("`sample['qwen_proprio']` or `sample['proprio']` is required when `proprio_dim` is enabled.")
        if proprio.ndim != 3:
            raise ValueError(f"`proprio` must be 3D [B,T,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[0] != batch_size:
            raise ValueError(f"`proprio` batch must be {batch_size}, got {proprio.shape[0]}")
        if self.proprio_dim is None or proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
        if "qwen_proprio" not in sample:
            proprio = proprio[:, 0:1, :]
        return proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

    def _prepare_training_proprio_is_pad(self, sample, batch_size: int) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            return None
        proprio_is_pad = sample.get("qwen_proprio_is_pad")
        if proprio_is_pad is None:
            proprio_is_pad = sample.get("proprio_is_pad")
            if proprio_is_pad is not None:
                proprio_is_pad = proprio_is_pad[:, 0:1]
        if proprio_is_pad is None:
            return None
        if proprio_is_pad.ndim != 2:
            raise ValueError(f"`proprio_is_pad` must be 2D [B,T], got shape {tuple(proprio_is_pad.shape)}")
        if proprio_is_pad.shape[0] != batch_size:
            raise ValueError(f"`proprio_is_pad` batch must be {batch_size}, got {proprio_is_pad.shape[0]}")
        return proprio_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

    def _prepare_inference_proprio(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            return None
        if proprio is None:
            raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
        if proprio.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"1D `proprio` only supports batch_size=1, got batch_size={batch_size}")
            proprio = proprio.view(1, 1, -1)
        elif proprio.ndim == 2:
            if proprio.shape[0] == batch_size:
                proprio = proprio.unsqueeze(1)
            elif batch_size == 1:
                proprio = proprio.unsqueeze(0)
            else:
                raise ValueError(f"2D `proprio` must be [B,D] or [T,D], got shape {tuple(proprio.shape)}")
        elif proprio.ndim == 3:
            pass
        else:
            raise ValueError(f"`proprio` must be [D], [B,D], or [B,T,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[0] != batch_size:
            raise ValueError(f"`proprio` batch must be {batch_size}, got {proprio.shape[0]}")
        if self.proprio_dim is None or proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
        return proprio.to(device=self.device, dtype=self.torch_dtype)

    def _prepare_inference_proprio_is_pad(
        self,
        proprio_is_pad: Optional[torch.Tensor],
        batch_size: int,
        num_proprio_frames: int,
    ) -> Optional[torch.Tensor]:
        if proprio_is_pad is None:
            return None
        if proprio_is_pad.ndim == 1:
            if batch_size == 1:
                proprio_is_pad = proprio_is_pad.unsqueeze(0)
            else:
                raise ValueError(f"1D `proprio_is_pad` only supports batch_size=1, got {batch_size}")
        if proprio_is_pad.ndim != 2:
            raise ValueError(f"`proprio_is_pad` must be [T] or [B,T], got shape {tuple(proprio_is_pad.shape)}")
        if proprio_is_pad.shape != (batch_size, num_proprio_frames):
            raise ValueError(
                "`proprio_is_pad` shape mismatch: "
                f"got {tuple(proprio_is_pad.shape)}, expected {(batch_size, num_proprio_frames)}"
            )
        return proprio_is_pad.to(device=self.device, dtype=torch.bool)

    def _encode_proprio_tokens(
        self,
        proprio: Optional[torch.Tensor],
        proprio_is_pad: Optional[torch.Tensor] = None,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.proprio_encoder is None or proprio is None:
            return None, None

        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        if proprio.ndim != 3:
            raise ValueError(f"`proprio` must be [B,D] or [B,T,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[0] != batch_size:
            raise ValueError(f"`proprio` batch must be {batch_size}, got {proprio.shape[0]}")
        if self.proprio_dim is None or proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
        num_proprio_tokens = int(proprio.shape[1])

        if proprio_is_pad is None:
            proprio_mask = torch.ones(
                (batch_size, num_proprio_tokens),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            if proprio_is_pad.ndim != 2 or proprio_is_pad.shape != proprio.shape[:2]:
                raise ValueError(
                    "`proprio_is_pad` must match proprio [B,T], "
                    f"got {tuple(proprio_is_pad.shape)} vs {tuple(proprio.shape[:2])}"
                )
            proprio_mask = ~proprio_is_pad.to(device=self.device, dtype=torch.bool)

        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=dtype)
        ).to(device=self.device, dtype=dtype)
        proprio_token = proprio_token * proprio_mask.to(dtype=proprio_token.dtype).unsqueeze(-1)
        if proprio_token.shape[-1] != self.qwen_expert.hidden_dim:
            raise ValueError(
                "Qwen proprio token dim mismatch: "
                f"got {proprio_token.shape[-1]}, expected {self.qwen_expert.hidden_dim}"
            )
        return proprio_token, proprio_mask

    @staticmethod
    def _condition_images_to_pil(images: torch.Tensor) -> list[Image.Image]:
        if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3:
            raise ValueError(f"`images` must be [1,K,3,H,W], got {tuple(images.shape)}")
        x = images[0].detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
        x = ((x + 1.0) * 127.5).round().to(torch.uint8)
        return [Image.fromarray(frame.permute(1, 2, 0).contiguous().numpy()) for frame in x]

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got {tuple(video.shape)}")
        if "prompt" not in sample:
            raise ValueError("FastWAMQwen3VL training requires `sample['prompt']`.")

        precomputed_video_latents = sample.get("precomputed_video_latents")
        batch_size, _, num_frames, carrier_height, carrier_width = video.shape
        if precomputed_video_latents is None:
            height, width = carrier_height, carrier_width
        else:
            if precomputed_video_latents.ndim != 5:
                raise ValueError(
                    "`precomputed_video_latents` must be [B,C,T,H,W], got "
                    f"{tuple(precomputed_video_latents.shape)}."
                )
            upsampling_factor = int(self.vae.upsampling_factor)
            height = int(precomputed_video_latents.shape[-2]) * upsampling_factor
            width = int(precomputed_video_latents.shape[-1]) * upsampling_factor
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAMQwen3VL training.")

        supplied_qwen_images = sample.get("qwen_images")
        if supplied_qwen_images is not None:
            if (
                supplied_qwen_images.ndim != 5
                or supplied_qwen_images.shape[0] != batch_size
                or supplied_qwen_images.shape[2] != 3
            ):
                raise ValueError(
                    "`sample['qwen_images']` must be [B,K,3,H,W], "
                    f"got {tuple(supplied_qwen_images.shape)}"
                )
            if (
                precomputed_video_latents is None
                and supplied_qwen_images.shape[-2:] != video.shape[-2:]
            ):
                raise ValueError(
                    "`sample['qwen_images']` spatial shape must match `sample['video']`, "
                    f"got {tuple(supplied_qwen_images.shape[-2:])} vs {tuple(video.shape[-2:])}"
                )
            supplied_qwen_image_frames = int(supplied_qwen_images.shape[1])
        else:
            supplied_qwen_image_frames = None

        num_condition_video_frames_value = sample.get("num_condition_video_frames")
        if num_condition_video_frames_value is None:
            num_condition_video_frames = supplied_qwen_image_frames or 1
        elif torch.is_tensor(num_condition_video_frames_value):
            if num_condition_video_frames_value.numel() == 0:
                raise ValueError("`num_condition_video_frames` tensor must not be empty.")
            flat_condition_frames = num_condition_video_frames_value.reshape(-1).to(
                device="cpu",
                dtype=torch.long,
            )
            if not torch.all(flat_condition_frames == flat_condition_frames[0]):
                raise ValueError(
                    "`num_condition_video_frames` must be identical across the batch, "
                    f"got {flat_condition_frames.tolist()}"
                )
            num_condition_video_frames = int(flat_condition_frames[0].item())
        else:
            num_condition_video_frames = int(num_condition_video_frames_value)
        if (
            supplied_qwen_image_frames is not None
            and num_condition_video_frames != supplied_qwen_image_frames
        ):
            raise ValueError(
                "Qwen and VAE must receive the same number of real condition images, "
                f"got condition_T={num_condition_video_frames} and qwen_T={supplied_qwen_image_frames}"
            )
        if num_condition_video_frames <= 0 or num_condition_video_frames > num_frames:
            raise ValueError(
                "`num_condition_video_frames` must be in [1, video_T], "
                f"got {num_condition_video_frames} for video_T={num_frames}"
            )
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        use_independent_condition_encoder = FastWAMQwen3VL._uses_independent_condition_encoder(self)
        use_local_endpoint_condition_encoder = (
            FastWAMQwen3VL._uses_local_endpoint_condition_encoder(self)
        )
        if use_independent_condition_encoder:
            expected_condition_video_frames = int(getattr(self, "num_history_frames", 4)) + 1
            if num_condition_video_frames != expected_condition_video_frames:
                raise ValueError(
                    "`independent_history_current` requires one current image after the configured "
                    "history images, "
                    f"got condition_T={num_condition_video_frames}, "
                    f"expected={expected_condition_video_frames}."
                )
            num_condition_latent_frames = 2
        elif use_local_endpoint_condition_encoder:
            configured_history_frames = int(getattr(self, "num_history_frames", temporal_factor))
            if configured_history_frames != temporal_factor:
                raise ValueError(
                    "The LingBot temporal encoder requires num_history_frames to match "
                    f"the Wan temporal factor {temporal_factor}, got "
                    f"{configured_history_frames}."
                )
            num_condition_latent_frames = int(
                getattr(self, "num_condition_latents", 1)
            )
            expected_condition_video_frames = (
                1 + temporal_factor * num_condition_latent_frames
            )
            if num_condition_video_frames != expected_condition_video_frames:
                condition_latent_label = (
                    "one"
                    if num_condition_latent_frames == 1
                    else str(num_condition_latent_frames)
                )
                raise ValueError(
                    "The LingBot temporal encoder requires one anchor plus "
                    f"{condition_latent_label} chronological "
                    f"{temporal_factor}-frame transitions ending at current, "
                    f"got condition_T={num_condition_video_frames}, "
                    f"expected={expected_condition_video_frames}."
                )
        else:
            if (num_condition_video_frames - 1) % temporal_factor != 0:
                raise ValueError(
                    "Condition image count must follow the Wan VAE 1+4n temporal layout so every image is encoded, "
                    f"got condition_T={num_condition_video_frames}, temporal_factor={temporal_factor}."
                )
            num_condition_latent_frames = 1 + (num_condition_video_frames - 1) // temporal_factor

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B,T,D], got {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        future_video_steps = num_frames - num_condition_video_frames
        if future_video_steps <= 0:
            raise ValueError(
                "`sample['video']` must contain future frames after the condition prefix during training, "
                f"got video_T={num_frames}, condition_T={num_condition_video_frames}"
            )
        if action_horizon % future_video_steps != 0:
            raise ValueError(
                "`sample['action']` temporal dimension must be divisible by future video steps "
                f"({future_video_steps}), got {action_horizon}"
            )
        if future_video_steps % temporal_factor != 0:
            raise ValueError(
                "Future video frame count must be divisible by the Wan VAE temporal factor, "
                f"got future_T={future_video_steps}, temporal_factor={temporal_factor}."
            )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or action_is_pad.shape != action.shape[:2]:
                raise ValueError(
                    f"`action_is_pad` must be [B,T]={tuple(action.shape[:2])}, got {tuple(action_is_pad.shape)}"
                )
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            if image_is_pad.ndim != 2 or image_is_pad.shape != (batch_size, num_frames):
                raise ValueError(
                    f"`image_is_pad` must be [B,T]={batch_size, num_frames}, got {tuple(image_is_pad.shape)}"
                )

        input_video = (
            video.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            if precomputed_video_latents is None
            else None
        )
        if supplied_qwen_images is None:
            qwen_image = (
                input_video[:, :, :num_condition_video_frames].permute(
                    0, 2, 1, 3, 4
                ).contiguous()
                if input_video is not None
                else None
            )
        else:
            qwen_image = supplied_qwen_images.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        condition_video = (
            input_video[:, :, :num_condition_video_frames]
            if input_video is not None
            else None
        )
        future_video = (
            input_video[:, :, num_condition_video_frames:]
            if input_video is not None
            else None
        )
        condition_latents = None
        condition_latent_valid_mask = sample.get(
            "condition_latent_valid_mask"
        )
        if precomputed_video_latents is not None:
            if not use_local_endpoint_condition_encoder:
                raise ValueError(
                    "Precomputed episode-streaming latents require a LingBot "
                    "temporal condition encoder."
                )
            precomputed_video_latents = precomputed_video_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            expected_precomputed_shape = (
                batch_size,
                int(self.vae.model.z_dim),
                num_condition_latent_frames
                + (future_video_steps // temporal_factor),
                height // int(self.vae.upsampling_factor),
                width // int(self.vae.upsampling_factor),
            )
            if tuple(precomputed_video_latents.shape) != expected_precomputed_shape:
                raise ValueError(
                    "Precomputed episode-streaming latent shape mismatch: "
                    f"got={tuple(precomputed_video_latents.shape)}, "
                    f"expected={expected_precomputed_shape}."
                )
            condition_latents = precomputed_video_latents[
                :, :, :num_condition_latent_frames
            ].contiguous()
            future_latents = precomputed_video_latents[
                :, :, num_condition_latent_frames:
            ].contiguous()
            if condition_latent_valid_mask is None:
                raise ValueError(
                    "Episode-streaming precomputed latents require "
                    "`condition_latent_valid_mask`."
                )
            condition_latent_valid_mask = condition_latent_valid_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            expected_valid_shape = (
                batch_size,
                num_condition_latent_frames,
            )
            if tuple(condition_latent_valid_mask.shape) != expected_valid_shape:
                raise ValueError(
                    "Condition-latent valid-mask shape mismatch: "
                    f"got={tuple(condition_latent_valid_mask.shape)}, "
                    f"expected={expected_valid_shape}."
                )
            if not torch.all(condition_latent_valid_mask.any(dim=1)):
                raise ValueError(
                    "Every sample must contain at least the episode anchor latent."
                )
        elif use_local_endpoint_condition_encoder:
            if bool(
                getattr(self, "independent_local_reset_endpoints", False)
            ):
                condition_latents = (
                    FastWAMQwen3VL._encode_local_endpoint_condition_latent(
                        self,
                        condition_video=condition_video,
                        tiled=tiled,
                    )
                )
                future_with_anchor = torch.cat(
                    [condition_video[:, :, -1:], future_video],
                    dim=2,
                )
                future_with_anchor_latents = (
                    FastWAMQwen3VL._encode_video_latents_batched(
                        self,
                        future_with_anchor,
                        tiled=tiled,
                    )
                )
                expected_future_with_anchor_latents = (
                    1 + future_video_steps // temporal_factor
                )
                if (
                    future_with_anchor_latents.ndim != 5
                    or future_with_anchor_latents.shape[2]
                    != expected_future_with_anchor_latents
                ):
                    raise RuntimeError(
                        "Local-reset future VAE latent length mismatch: "
                        f"got={tuple(future_with_anchor_latents.shape)}, "
                        f"expected_T={expected_future_with_anchor_latents}."
                    )
                future_latents = future_with_anchor_latents[:, :, 1:]
            else:
                # Preserve the original TMD2 training path: reset once, encode
                # the complete 1+4+4... sequence, then split condition/target.
                all_latents = FastWAMQwen3VL._encode_video_latents_batched(
                    self,
                    input_video,
                    tiled=tiled,
                )
                expected_all_latents = 1 + (num_frames - 1) // temporal_factor
                if (
                    all_latents.ndim != 5
                    or all_latents.shape[2] != expected_all_latents
                ):
                    raise RuntimeError(
                        "Continuous history/future VAE latent length mismatch: "
                        f"got={tuple(all_latents.shape)}, "
                        f"expected_T={expected_all_latents}."
                    )
                condition_start = 1
                condition_end = condition_start + num_condition_latent_frames
                condition_latents = all_latents[
                    :, :, condition_start:condition_end
                ]
                future_latents = all_latents[:, :, condition_end:]
        else:
            if not use_independent_condition_encoder:
                condition_latents = self._encode_video_latents(
                    condition_video,
                    tiled=tiled,
                )
            future_with_anchor = torch.cat(
                [condition_video[:, :, -1:], future_video],
                dim=2,
            )
            future_with_anchor_latents = self._encode_video_latents(
                future_with_anchor,
                tiled=tiled,
            )
            expected_future_with_anchor_latents = (
                1 + future_video_steps // temporal_factor
            )
            if future_with_anchor_latents.shape[2] != expected_future_with_anchor_latents:
                raise RuntimeError(
                    "Future VAE latent length mismatch: "
                    f"got={future_with_anchor_latents.shape[2]}, "
                    f"expected={expected_future_with_anchor_latents}."
                )
            future_latents = future_with_anchor_latents[:, :, 1:]
            if use_independent_condition_encoder:
                # Wan's causal first latent of [current, future...] is exactly E(current),
                # so reuse it rather than performing a third VAE pass during training.
                current_latent = future_with_anchor_latents[:, :, :1]
                history_is_pad = (
                    None
                    if image_is_pad is None
                    else image_is_pad[
                        :, : int(getattr(self, "num_history_frames", 4))
                    ]
                )
                condition_latents = FastWAMQwen3VL._build_independent_condition_latents(
                    self,
                    condition_video=condition_video,
                    current_latent=current_latent,
                    history_is_pad=history_is_pad,
                    tiled=tiled,
                )
        if condition_latents.shape[2] != num_condition_latent_frames:
            raise RuntimeError(
                "Condition VAE latent length mismatch: "
                f"got={condition_latents.shape[2]}, expected={num_condition_latent_frames}, "
                f"condition_T={num_condition_video_frames}."
            )
        input_latents = torch.cat([condition_latents, future_latents], dim=2)

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        if not fuse_flag:
            raise NotImplementedError("FastWAMQwen3VL requires fused VAE condition latents.")

        prompt = self._normalize_prompts(sample["prompt"], batch_size=batch_size)
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        proprio = self._prepare_training_proprio(sample, batch_size=batch_size)
        proprio_is_pad = self._prepare_training_proprio_is_pad(sample, batch_size=batch_size)

        return {
            "prompt": prompt,
            "qwen_image": qwen_image,
            "input_latents": input_latents,
            "condition_latents": condition_latents,
            "condition_latent_valid_mask": condition_latent_valid_mask,
            "num_condition_video_frames": num_condition_video_frames,
            "num_condition_latent_frames": num_condition_latent_frames,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "proprio": proprio,
            "proprio_is_pad": proprio_is_pad,
        }

    @torch.no_grad()
    def _build_video_latent_is_pad(
        self,
        image_is_pad: Optional[torch.Tensor],
        num_condition_video_frames: int,
        num_condition_latent_frames: int,
        condition_latent_valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if image_is_pad is None and condition_latent_valid_mask is None:
            return None
        if image_is_pad is not None and image_is_pad.ndim != 2:
            raise ValueError(f"`image_is_pad` must be [B,T], got {tuple(image_is_pad.shape)}")

        if condition_latent_valid_mask is not None:
            if condition_latent_valid_mask.ndim != 2:
                raise ValueError(
                    "`condition_latent_valid_mask` must be [B,C], got "
                    f"{tuple(condition_latent_valid_mask.shape)}."
                )
            if int(condition_latent_valid_mask.shape[1]) != int(
                num_condition_latent_frames
            ):
                raise ValueError(
                    "Condition valid-mask length does not match latent prefix."
                )
            batch_size = int(condition_latent_valid_mask.shape[0])
            condition_latent_is_pad = ~condition_latent_valid_mask.to(
                dtype=torch.bool
            )
        else:
            batch_size = int(image_is_pad.shape[0])
            condition_latent_is_pad = torch.zeros(
                (batch_size, int(num_condition_latent_frames)),
                dtype=torch.bool,
                device=image_is_pad.device,
            )

        if image_is_pad is None:
            return condition_latent_is_pad

        temporal_factor = int(self.vae.temporal_downsample_factor)
        future_is_pad = image_is_pad[:, int(num_condition_video_frames) :]
        if future_is_pad.shape[1] % temporal_factor != 0:
            raise ValueError(
                "Cannot align future `image_is_pad` with video latent steps: "
                f"future_frames={future_is_pad.shape[1]}, temporal_factor={temporal_factor}."
            )
        future_latent_is_pad = future_is_pad.reshape(
            image_is_pad.shape[0],
            -1,
            temporal_factor,
        ).all(dim=2)
        if int(image_is_pad.shape[0]) != batch_size:
            raise ValueError(
                "Image and condition-latent masks have different batch sizes."
            )
        condition_latent_is_pad = condition_latent_is_pad.to(
            device=image_is_pad.device
        )
        return torch.cat([condition_latent_is_pad, future_latent_is_pad], dim=1)

    @staticmethod
    def _resolve_attention_batch_size(*masks: Optional[torch.Tensor]) -> int:
        batch_sizes = []
        for mask in masks:
            if mask is None:
                continue
            if mask.ndim == 1:
                batch_sizes.append(1)
            elif mask.ndim == 2:
                batch_sizes.append(int(mask.shape[0]))
            else:
                raise ValueError(f"Sequence masks must be 1D or 2D, got {tuple(mask.shape)}")
        non_singleton = {size for size in batch_sizes if size != 1}
        if len(non_singleton) > 1:
            raise ValueError(f"Attention mask batch sizes disagree: {batch_sizes}")
        return next(iter(non_singleton), 1)

    @staticmethod
    def _as_batched_sequence_mask(
        mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        name: str,
        default: bool,
    ) -> torch.Tensor:
        if mask is None:
            return torch.full((batch_size, seq_len), default, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or mask.shape[1] != seq_len:
            raise ValueError(f"`{name}` must be [B,{seq_len}], got {tuple(mask.shape)}")
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(f"`{name}` batch must be {batch_size}, got {mask.shape[0]}")
        return mask

    @torch.no_grad()
    def _build_trimot_attention_mask(
        self,
        qwen_seq_len: int,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        qwen_visual_pos_mask: Optional[torch.Tensor] = None,
        qwen_downstream_visible_mask: Optional[torch.Tensor] = None,
        num_condition_latent_frames: int = 1,
        qwen_attention_mask: Optional[torch.Tensor] = None,
        video_latent_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`video_seq_len` must be divisible by `video_tokens_per_frame`, "
                f"got {video_seq_len} and {video_tokens_per_frame}."
            )
        num_video_latent_frames = video_seq_len // video_tokens_per_frame
        batch_size = self._resolve_attention_batch_size(
            qwen_attention_mask,
            qwen_visual_pos_mask,
            qwen_downstream_visible_mask,
            video_latent_is_pad,
            action_is_pad,
        )
        total_seq_len = qwen_seq_len + video_seq_len + action_seq_len
        mask = torch.zeros((batch_size, total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        q0, q1 = 0, qwen_seq_len
        v0, v1 = q1, q1 + video_seq_len
        a0, a1 = v1, total_seq_len

        qwen_valid = self._as_batched_sequence_mask(
            qwen_attention_mask,
            batch_size=batch_size,
            seq_len=qwen_seq_len,
            device=device,
            name="qwen_attention_mask",
            default=True,
        )
        qwen_visual = self._as_batched_sequence_mask(
            qwen_visual_pos_mask,
            batch_size=batch_size,
            seq_len=qwen_seq_len,
            device=device,
            name="qwen_visual_pos_mask",
            default=False,
        )
        video_frame_is_pad = self._as_batched_sequence_mask(
            video_latent_is_pad,
            batch_size=batch_size,
            seq_len=num_video_latent_frames,
            device=device,
            name="video_latent_is_pad",
            default=False,
        )
        video_valid = (~video_frame_is_pad).repeat_interleave(video_tokens_per_frame, dim=1)
        action_valid = ~self._as_batched_sequence_mask(
            action_is_pad,
            batch_size=batch_size,
            seq_len=action_seq_len,
            device=device,
            name="action_is_pad",
            default=False,
        )
        if qwen_downstream_visible_mask is None:
            qwen_downstream_visible = qwen_valid & ~qwen_visual
        else:
            qwen_downstream_visible = qwen_valid & self._as_batched_sequence_mask(
                qwen_downstream_visible_mask,
                batch_size=batch_size,
                seq_len=qwen_seq_len,
                device=device,
                name="qwen_downstream_visible_mask",
                default=False,
            )

        # Qwen pre_dit lays out text, then proprio, then the oldest-to-newest
        # image blocks. The global decoder-causal mask keeps both text and
        # proprio image-independent while letting every image read the task,
        # robot state, and earlier frames. FastWAM cross-modal visibility is an
        # explicit text+proprio mask, so post-image wrapper tokens cannot leak
        # visual information back into the policy stream.
        qwen_causal = torch.ones(
            (qwen_seq_len, qwen_seq_len),
            dtype=torch.bool,
            device=device,
        ).tril()
        mask[:, q0:q1, q0:q1] = (
            qwen_valid.unsqueeze(2)
            & qwen_valid.unsqueeze(1)
            & qwen_causal.unsqueeze(0)
        )
        mask[:, v0:v1, q0:q1] = video_valid.unsqueeze(2) & qwen_downstream_visible.unsqueeze(1)
        video_to_video = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            num_condition_frames=num_condition_latent_frames,
        )
        mask[:, v0:v1, v0:v1] = (
            video_to_video.unsqueeze(0) & video_valid.unsqueeze(2) & video_valid.unsqueeze(1)
        )
        mask[:, a0:a1, q0:q1] = action_valid.unsqueeze(2) & qwen_downstream_visible.unsqueeze(1)
        mask[:, a0:a1, a0:a1] = action_valid.unsqueeze(2) & action_valid.unsqueeze(1)
        condition_tokens = min(int(num_condition_latent_frames) * video_tokens_per_frame, video_seq_len)
        mask[:, a0:a1, v0 : v0 + condition_tokens] = (
            action_valid.unsqueeze(2) & video_valid[:, :condition_tokens].unsqueeze(1)
        )
        return mask.unsqueeze(1)

    @torch.no_grad()
    def _build_dit_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        num_condition_latent_frames: int = 1,
        video_latent_is_pad: Optional[torch.Tensor] = None,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`video_seq_len` must be divisible by `video_tokens_per_frame`, "
                f"got {video_seq_len} and {video_tokens_per_frame}."
            )
        num_video_latent_frames = video_seq_len // video_tokens_per_frame
        batch_size = self._resolve_attention_batch_size(video_latent_is_pad, action_is_pad)
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((batch_size, total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        v0, v1 = 0, video_seq_len
        a0, a1 = v1, total_seq_len

        video_frame_is_pad = self._as_batched_sequence_mask(
            video_latent_is_pad,
            batch_size=batch_size,
            seq_len=num_video_latent_frames,
            device=device,
            name="video_latent_is_pad",
            default=False,
        )
        video_valid = (~video_frame_is_pad).repeat_interleave(video_tokens_per_frame, dim=1)
        action_valid = ~self._as_batched_sequence_mask(
            action_is_pad,
            batch_size=batch_size,
            seq_len=action_seq_len,
            device=device,
            name="action_is_pad",
            default=False,
        )
        video_to_video = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            num_condition_frames=num_condition_latent_frames,
        )
        mask[:, v0:v1, v0:v1] = (
            video_to_video.unsqueeze(0) & video_valid.unsqueeze(2) & video_valid.unsqueeze(1)
        )
        mask[:, a0:a1, a0:a1] = action_valid.unsqueeze(2) & action_valid.unsqueeze(1)
        condition_tokens = min(int(num_condition_latent_frames) * video_tokens_per_frame, video_seq_len)
        mask[:, a0:a1, v0 : v0 + condition_tokens] = (
            action_valid.unsqueeze(2) & video_valid[:, :condition_tokens].unsqueeze(1)
        )
        return mask.unsqueeze(1)

    def _compute_video_loss_after_condition_prefix(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        num_condition_video_frames: int,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        num_condition_video_frames = int(num_condition_video_frames)
        if num_condition_video_frames <= 0 or num_condition_video_frames > image_is_pad.shape[1]:
            raise ValueError(
                "`num_condition_video_frames` must be in [1, image_is_pad_T], "
                f"got {num_condition_video_frames} for T={image_is_pad.shape[1]}."
            )
        future_is_pad = image_is_pad[:, num_condition_video_frames:]
        if future_is_pad.shape[1] % temporal_factor != 0:
            raise ValueError(
                "Cannot align future `image_is_pad` with future video latent steps: "
                f"future_frames={future_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        video_is_pad = future_is_pad.reshape(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        return (video_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
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
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["condition_latents"] is not None:
            latents[:, :, :num_condition_latent_frames] = inputs["condition_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        proprio_tokens, proprio_attention_mask = self._encode_proprio_tokens(
            inputs["proprio"],
            proprio_is_pad=inputs["proprio_is_pad"],
            batch_size=batch_size,
            dtype=input_latents.dtype,
        )
        qwen_pre = self.qwen_expert.pre_dit(
            images=inputs["qwen_image"],
            prompts=inputs["prompt"],
            proprio_tokens=proprio_tokens,
            proprio_attention_mask=proprio_attention_mask,
        )
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            num_condition_latent_frames=num_condition_latent_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        attention_mask = None
        if self.mot.condition_mode == "joint_mot":
            attention_mask = self._build_trimot_attention_mask(
                qwen_seq_len=qwen_pre["tokens"].shape[1],
                video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
                qwen_visual_pos_mask=qwen_pre.get("visual_pos_masks"),
                qwen_downstream_visible_mask=qwen_pre.get("downstream_visible_masks"),
                num_condition_latent_frames=num_condition_latent_frames,
                qwen_attention_mask=qwen_pre.get("attention_mask"),
                video_latent_is_pad=video_latent_is_pad,
                action_is_pad=action_is_pad,
            )
        dit_attention_mask = self._build_dit_attention_mask(
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
                "qwen": qwen_pre["tokens"],
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            dit_attention_mask=dit_attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            qwen_pre_state=qwen_pre,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if inputs["condition_latents"] is not None:
            pred_video = pred_video[:, :, num_condition_latent_frames:]
            target_video = target_video[:, :, num_condition_latent_frames:]

        loss_video_per_sample = self._compute_video_loss_after_condition_prefix(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            num_condition_video_frames=num_condition_video_frames,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        qwen_image: torch.Tensor,
        prompts: Sequence[str],
        proprio: Optional[torch.Tensor],
        proprio_is_pad: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        num_condition_latent_frames: int,
        gt_action: Optional[torch.Tensor] = None,
        qwen_pre_state: Optional[dict[str, Any]] = None,
        qwen_condition_cache: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if qwen_pre_state is None:
            proprio_tokens, proprio_attention_mask = self._encode_proprio_tokens(
                proprio,
                proprio_is_pad=proprio_is_pad,
                batch_size=int(qwen_image.shape[0]),
                dtype=latents_video.dtype,
            )
            qwen_pre_state = self.qwen_expert.pre_dit(
                images=qwen_image,
                prompts=prompts,
                proprio_tokens=proprio_tokens,
                proprio_attention_mask=proprio_attention_mask,
            )
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
        attention_mask = None
        if self.mot.condition_mode == "joint_mot":
            attention_mask = self._build_trimot_attention_mask(
                qwen_seq_len=qwen_pre_state["tokens"].shape[1],
                video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
                qwen_visual_pos_mask=qwen_pre_state.get("visual_pos_masks"),
                qwen_downstream_visible_mask=qwen_pre_state.get("downstream_visible_masks"),
                num_condition_latent_frames=num_condition_latent_frames,
                qwen_attention_mask=qwen_pre_state.get("attention_mask"),
            )
        dit_attention_mask = self._build_dit_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_condition_latent_frames=num_condition_latent_frames,
        )
        tokens_out = self.mot(
            embeds_all={
                "qwen": qwen_pre_state["tokens"],
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            dit_attention_mask=dit_attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            qwen_pre_state=qwen_pre_state,
            qwen_condition_cache=qwen_condition_cache,
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @staticmethod
    def _normalize_inference_images(input_image: torch.Tensor) -> torch.Tensor:
        if input_image.ndim == 3:
            if input_image.shape[0] != 3:
                raise ValueError(f"`input_image` must be [3,H,W], got {tuple(input_image.shape)}")
            return input_image.unsqueeze(0).unsqueeze(0)
        if input_image.ndim == 4:
            if input_image.shape[0] == 1 and input_image.shape[1] == 3:
                return input_image.unsqueeze(1)
            if input_image.shape[1] == 3:
                return input_image.unsqueeze(0)
            raise ValueError(
                "`input_image` must be [1,3,H,W], [K,3,H,W], or [1,K,3,H,W], "
                f"got {tuple(input_image.shape)}"
            )
        if input_image.ndim == 5:
            if input_image.shape[0] != 1 or input_image.shape[2] != 3:
                raise ValueError(f"`input_image` must be [1,K,3,H,W], got {tuple(input_image.shape)}")
            return input_image
        raise ValueError(
            "`input_image` must be [3,H,W], [1,3,H,W], [K,3,H,W], or [1,K,3,H,W], "
            f"got {tuple(input_image.shape)}"
        )

    @classmethod
    def _infer_condition_video_frames(cls, input_image: torch.Tensor) -> int:
        return int(cls._normalize_inference_images(input_image).shape[1])

    @staticmethod
    def _sample_joint_initial_noise(
        video_shape: Sequence[int],
        action_shape: Sequence[int],
        *,
        seed: Optional[int],
        rand_device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        video_noise = torch.randn(
            tuple(video_shape),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        )
        action_noise = torch.randn(
            tuple(action_shape),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        )
        return video_noise, action_noise

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[Union[str, Sequence[str]]],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        qwen_images: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_is_pad: Optional[torch.Tensor] = None,
        test_action_with_infer_action: bool = False,
    ) -> dict[str, Any]:
        del context, context_mask, negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()
        if prompt is None:
            raise ValueError("FastWAMQwen3VL inference requires raw `prompt` text.")
        input_images = self._normalize_inference_images(input_image)
        qwen_input_images = (
            input_images
            if qwen_images is None
            else self._normalize_inference_images(qwen_images)
        )
        num_condition_video_frames = int(input_images.shape[1])
        if num_video_frames < num_condition_video_frames:
            raise ValueError(
                "`num_video_frames` must be at least the number of condition images, "
                f"got num_video_frames={num_video_frames}, condition_images={num_condition_video_frames}"
            )

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        use_independent_condition_encoder = FastWAMQwen3VL._uses_independent_condition_encoder(self)
        use_local_endpoint_condition_encoder = (
            FastWAMQwen3VL._uses_local_endpoint_condition_encoder(self)
        )
        if use_independent_condition_encoder:
            expected_condition_video_frames = int(getattr(self, "num_history_frames", 4)) + 1
            if num_condition_video_frames != expected_condition_video_frames:
                raise ValueError(
                    "`independent_history_current` requires one current image after the configured "
                    "history images, "
                    f"got condition_T={num_condition_video_frames}, "
                    f"expected={expected_condition_video_frames}."
                )
            if history_is_pad is not None:
                if not torch.is_tensor(history_is_pad):
                    history_is_pad = torch.as_tensor(history_is_pad)
                if history_is_pad.ndim == 1:
                    history_is_pad = history_is_pad.unsqueeze(0)
                expected_history_is_pad_shape = (1, int(getattr(self, "num_history_frames", 4)))
                if history_is_pad.shape != expected_history_is_pad_shape:
                    raise ValueError(
                        "`history_is_pad` must mark the non-current condition images as "
                        f"{expected_history_is_pad_shape}, got {tuple(history_is_pad.shape)}."
                    )
                history_is_pad = history_is_pad.to(device=self.device, dtype=torch.bool)
        elif use_local_endpoint_condition_encoder:
            configured_history_frames = int(getattr(self, "num_history_frames", temporal_factor))
            if configured_history_frames != temporal_factor:
                raise ValueError(
                    "The LingBot temporal encoder requires num_history_frames to match "
                    f"the Wan temporal factor {temporal_factor}, got "
                    f"{configured_history_frames}."
                )
            num_condition_latent_frames = int(
                getattr(self, "num_condition_latents", 1)
            )
            expected_condition_video_frames = (
                1 + temporal_factor * num_condition_latent_frames
            )
            if num_condition_video_frames != expected_condition_video_frames:
                raise ValueError(
                    "The LingBot temporal encoder requires one anchor plus "
                    f"{num_condition_latent_frames} chronological "
                    f"{temporal_factor}-frame transitions ending at current, "
                    f"got condition_T={num_condition_video_frames}, "
                    f"expected={expected_condition_video_frames}."
                )
        elif (num_condition_video_frames - 1) % temporal_factor != 0:
            raise ValueError(
                "Condition image count must follow the Wan VAE 1+4n temporal layout so every image is encoded, "
                f"got condition_T={num_condition_video_frames}, temporal_factor={temporal_factor}."
            )
        num_future_video_frames = int(num_video_frames) - num_condition_video_frames
        if num_future_video_frames % temporal_factor != 0:
            raise ValueError(
                "The number of future video frames must be divisible by the Wan VAE temporal factor: "
                f"future_T={num_future_video_frames}, temporal_factor={temporal_factor}, "
                f"num_video_frames={num_video_frames}, condition_T={num_condition_video_frames}."
            )

        _, _, _, height, width = input_images.shape
        decode_frames = 1 + num_future_video_frames
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height,
            width,
            decode_frames,
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f"`input_image` must have H/W multiples of 16, got HxW=({height},{width})")
        if checked_t != decode_frames:
            raise ValueError(
                "The current-frame anchor plus future frames must satisfy the Wan VAE temporal layout, "
                f"got decode_T={decode_frames}."
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(f"`action` must be [1,T,D] or [T,D], got {tuple(action.shape)}")
            action = action.to(device=self.device, dtype=self.torch_dtype)

        if use_independent_condition_encoder:
            num_condition_latent_frames = 2
        elif use_local_endpoint_condition_encoder:
            num_condition_latent_frames = int(
                getattr(self, "num_condition_latents", 1)
            )
        else:
            num_condition_latent_frames = 1 + (
                num_condition_video_frames - 1
            ) // temporal_factor
        num_future_latent_frames = num_future_video_frames // temporal_factor
        latent_t = num_condition_latent_frames + num_future_latent_frames
        if num_condition_latent_frames > latent_t:
            raise ValueError(
                "`num_condition_latent_frames` exceeds latent video length: "
                f"{num_condition_latent_frames} > {latent_t}"
            )
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

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        if not fuse_flag:
            raise NotImplementedError("FastWAMQwen3VL requires fused VAE condition latents.")
        qwen_image = qwen_input_images.to(device=self.device, dtype=self.torch_dtype)
        condition_video = input_images.to(
            device=self.device,
            dtype=self.torch_dtype,
        ).permute(0, 2, 1, 3, 4).contiguous()
        current_anchor_latent = None
        if use_independent_condition_encoder:
            current_anchor_latent = self._encode_video_latents(
                condition_video[:, :, -1:],
                tiled=tiled,
            )
            condition_latents = FastWAMQwen3VL._build_independent_condition_latents(
                self,
                condition_video=condition_video,
                current_latent=current_anchor_latent,
                history_is_pad=history_is_pad,
                tiled=tiled,
            )
        elif use_local_endpoint_condition_encoder:
            condition_latents = FastWAMQwen3VL._encode_local_endpoint_condition_latent(
                self,
                condition_video=condition_video,
                tiled=tiled,
            )
        else:
            condition_latents = self._encode_video_latents(condition_video, tiled=tiled)
        if condition_latents.shape[2] != num_condition_latent_frames:
            raise ValueError(
                "Encoded condition latent length mismatch: "
                f"got {condition_latents.shape[2]}, expected {num_condition_latent_frames} "
                f"from condition_T={condition_video.shape[2]}."
            )
        latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()
        prompts = self._normalize_prompts(prompt, batch_size=1)
        proprio = self._prepare_inference_proprio(proprio, batch_size=1)
        proprio_is_pad = self._prepare_inference_proprio_is_pad(
            proprio_is_pad,
            batch_size=1,
            num_proprio_frames=0 if proprio is None else int(proprio.shape[1]),
        )
        proprio_tokens, proprio_attention_mask = self._encode_proprio_tokens(
            proprio,
            proprio_is_pad=proprio_is_pad,
            batch_size=1,
            dtype=latents_video.dtype,
        )
        qwen_pre_state = self.qwen_expert.pre_dit(
            images=qwen_image,
            prompts=prompts,
            proprio_tokens=proprio_tokens,
            proprio_attention_mask=proprio_attention_mask,
        )
        qwen_condition_cache = None
        if self.mot.condition_mode == "cross_attn":
            qwen_condition_cache = self.mot.prepare_condition_cache(qwen_pre_state)

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                qwen_image=qwen_image,
                prompts=prompts,
                proprio=proprio,
                proprio_is_pad=proprio_is_pad,
                fuse_vae_embedding_in_latents=fuse_flag,
                num_condition_latent_frames=num_condition_latent_frames,
                gt_action=action,
                qwen_pre_state=qwen_pre_state,
                qwen_condition_cache=qwen_condition_cache,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, :num_condition_latent_frames] = condition_latents.clone()

        decoded_video = self._condition_images_to_pil(input_images)
        if num_future_video_frames > 0:
            future_latents = latents_video[:, :, num_condition_latent_frames:]
            if future_latents.shape[2] != num_future_latent_frames:
                raise RuntimeError(
                    "Generated future latent length mismatch: "
                    f"got={future_latents.shape[2]}, expected={num_future_latent_frames}."
                )
            if current_anchor_latent is None:
                current_anchor_latent = self._encode_video_latents(
                    condition_video[:, :, -1:],
                    tiled=tiled,
                )
            if current_anchor_latent.shape[2] != 1:
                raise RuntimeError(
                    "Current image must encode to one anchor latent frame, "
                    f"got latent_T={current_anchor_latent.shape[2]}."
                )
            anchor_and_future_latents = torch.cat([current_anchor_latent, future_latents], dim=2)
            decoded_anchor_and_future = self._decode_latents(anchor_and_future_latents, tiled=tiled)
            expected_decoded_frames = decode_frames
            if len(decoded_anchor_and_future) != expected_decoded_frames:
                raise RuntimeError(
                    "Decoded anchor/future frame count mismatch: "
                    f"got={len(decoded_anchor_and_future)}, expected={expected_decoded_frames}."
                )
            decoded_future_frames = decoded_anchor_and_future[1:]
            if len(decoded_future_frames) != num_future_video_frames:
                raise RuntimeError(
                    "Decoded interval-sampled future frame count mismatch: "
                    f"got={len(decoded_future_frames)}, expected={num_future_video_frames}."
                )
            decoded_video.extend(decoded_future_frames)
        if len(decoded_video) != int(num_video_frames):
            raise RuntimeError(
                "Decoded real-frame count mismatch: "
                f"got={len(decoded_video)}, expected={num_video_frames}."
            )
        return {
            "video": decoded_video,
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        qwen_images: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del context, context_mask, negative_prompt, text_cfg_scale
        num_video_frames = self._infer_condition_video_frames(input_image)
        out = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            qwen_images=qwen_images,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            proprio_is_pad=proprio_is_pad,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            history_is_pad=history_is_pad,
        )
        return {"action": out["action"]}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        qwen_images: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_is_pad: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_is_pad: Optional[torch.Tensor] = None,
    ):
        del action_cfg_scale
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            qwen_images=qwen_images,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            proprio_is_pad=proprio_is_pad,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            history_is_pad=history_is_pad,
        )
