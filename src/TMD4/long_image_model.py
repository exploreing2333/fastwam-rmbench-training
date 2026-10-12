from __future__ import annotations

from typing import Any, Optional

import torch

from TMD3.state_model import StateConditionedProgressTMD
from TMD4.long_image_mot import LongImageStateProgressMoT


class LongImageProgressTMD(StateConditionedProgressTMD):
    """TMD4: LingBot-style long Image memory on the existing MoT contract.

    Image uses a causal global phase-stream axis. Action keeps the TMD3 token
    and visibility contract but shares Video's 3D RoPE basis so mixed
    attention is temporally aligned. Progress, decoded subtask and current
    state keep the existing TMD3 routing contract.
    """

    contract_name = "TMD4-LongImage-GlobalRoPE"

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        giga_world_policy_pretrained_path: str | None = None,
        **kwargs,
    ):
        requested_skip = bool(
            kwargs.pop("skip_dit_load_from_pretrain", False)
        )
        use_gigaworld = bool(giga_world_policy_pretrained_path)
        model = super().from_wan22_pretrained(
            *args,
            # The public checkpoint replaces both experts. Avoid loading the
            # old Wan visual and interpolated Action backbones first.
            skip_dit_load_from_pretrain=(requested_skip or use_gigaworld),
            **kwargs,
        )
        if use_gigaworld:
            report = model.mot.load_gigaworld_pretrained(
                str(giga_world_policy_pretrained_path)
            )
            model.gigaworld_load_report = report
            model.model_paths["video_dit"] = str(
                giga_world_policy_pretrained_path
            )
            model.model_paths["action_dit_backbone"] = str(
                giga_world_policy_pretrained_path
            )
            model.model_paths["giga_world_policy"] = str(
                giga_world_policy_pretrained_path
            )
        return model

    def __init__(self, *args, condition_encoder_config=None, **kwargs):
        config = dict(condition_encoder_config or {})
        self.history_condition_keep_current_clean = bool(
            config.pop("history_condition_keep_current_clean", False)
        )
        self.long_image_history_max = int(
            config.pop("long_image_history_max", 64)
        )
        self.long_image_window_min = int(
            config.pop("long_image_window_min", 4)
        )
        self.long_image_window_max = int(
            config.pop("long_image_window_max", 64)
        )
        self.long_image_inference_window = int(
            config.pop(
                "long_image_inference_window",
                self.long_image_history_max,
            )
        )
        self.long_image_future_chunk_min = int(
            config.pop("long_image_future_chunk_min", 1)
        )
        self.long_image_future_chunk_max = int(
            config.pop("long_image_future_chunk_max", 4)
        )
        self.long_image_checkpoint_min_frames = int(
            config.pop("long_image_checkpoint_min_frames", 17)
        )
        self.long_image_position_contract = str(
            config.pop(
                "long_image_position_contract",
                "phase_global_video_action_3d_v1",
            )
        )
        if self.long_image_position_contract != "phase_global_video_action_3d_v1":
            raise ValueError(
                "Unsupported TMD4 position contract: "
                f"{self.long_image_position_contract!r}."
            )
        if not (
            1
            <= self.long_image_window_min
            <= self.long_image_window_max
            <= self.long_image_history_max
        ):
            raise ValueError(
                "TMD4 Image window must satisfy 1 <= min <= max <= history, "
                f"got min={self.long_image_window_min}, "
                f"max={self.long_image_window_max}, "
                f"history={self.long_image_history_max}."
            )
        if not (
            1
            <= self.long_image_inference_window
            <= self.long_image_history_max
        ):
            raise ValueError(
                "TMD4 inference window must satisfy 1 <= inference <= "
                "history, got "
                f"inference={self.long_image_inference_window}, "
                f"history={self.long_image_history_max}."
            )
        if not (
            1
            <= self.long_image_future_chunk_min
            <= self.long_image_future_chunk_max
        ):
            raise ValueError(
                "TMD4 future chunk must satisfy 1 <= min <= max, got "
                f"{self.long_image_future_chunk_min}.."
                f"{self.long_image_future_chunk_max}."
            )
        if self.long_image_checkpoint_min_frames <= 0:
            raise ValueError("TMD4 checkpoint threshold must be positive.")
        super().__init__(
            *args,
            condition_encoder_config=config,
            **kwargs,
        )
        # TMD2 constructs the model with the pre-adoption expert wrappers.
        # TMD4 executes through the unregistered views owned by the unified
        # GigaWorld-named MoT, including its new Action MLP I/O.
        del self.video_expert
        del self.action_expert
        self.video_expert = self.mot.video_expert
        self.action_expert = self.mot.action_expert
        # Keep the complete training contract in checkpoint metadata while
        # leaving the shared TMD3 config normalizer strict and unchanged.
        self.condition_encoder_config.update(
            {
                "long_image_history_max": self.long_image_history_max,
                "long_image_window_min": self.long_image_window_min,
                "long_image_window_max": self.long_image_window_max,
                "long_image_inference_window": (
                    self.long_image_inference_window
                ),
                "long_image_future_chunk_min": self.long_image_future_chunk_min,
                "long_image_future_chunk_max": self.long_image_future_chunk_max,
                "long_image_checkpoint_min_frames": (
                    self.long_image_checkpoint_min_frames
                ),
                "long_image_position_contract": self.long_image_position_contract,
                "history_condition_keep_current_clean": (
                    self.history_condition_keep_current_clean
                ),
            }
        )

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
        config = dict(condition_encoder_config or {})
        return LongImageStateProgressMoT(
            video_expert=video_expert,
            action_expert=action_expert,
            progress_hidden_dim=int(progress_hidden_dim),
            condition_dim=int(condition_dim),
            progress_bins=int(config.get("progress_bins", 10)),
            progress_bottleneck_dim=int(
                config.get("progress_bottleneck_dim", 256)
            ),
            progress_gate_init=float(config.get("progress_gate_init", -3.0)),
            state_dim=int(config.get("state_dim", 14)),
            state_history_tokens=int(config.get("state_history_tokens", 4)),
            state_time_start=float(config.get("state_time_start", 1.0)),
            state_time_end=float(config.get("state_time_end", 4.0)),
            state_time_gate_init=float(
                config.get("state_time_gate_init", -3.0)
            ),
            state_temporal_encoding=bool(
                config.get("state_temporal_encoding", True)
            ),
            video_time_offset=float(config.get("video_time_offset", 1.0)),
            action_time_start=float(config.get("action_time_start", 4.0)),
            action_time_end=float(config.get("action_time_end", 5.0)),
            progress_injection_mode=str(
                config.get("progress_injection_mode", "full_attention_kv")
            ),
            state_cross_attention_scope=str(
                config.get("state_cross_attention_scope", "video_action")
            ),
            subtask_condition_mode=str(
                config.get("subtask_condition_mode", "progress_hidden")
            ),
            progress_ordinal_context_mode=str(
                config.get("progress_ordinal_context_mode", "none")
            ),
            progress_max_plan_length=int(
                config.get("progress_max_plan_length", 32)
            ),
            checkpoint_attention=bool(checkpoint_attention),
        )

    def reset_streaming_history(self) -> None:
        """Clear real-latent positions together with the inherited VAE banks."""
        super().reset_streaming_history()
        self._streaming_condition_frame_indices = None
        self._streaming_condition_frame_indices_by_id = {}

    @torch.no_grad()
    def initialize_streaming_history(
        self,
        input_images: torch.Tensor,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        """Initialize LingBot-style real history with one valid anchor."""
        input_images = self._normalize_inference_images(input_images)
        if int(input_images.shape[1]) != 1:
            raise ValueError(
                "TMD4 streaming initialization requires one real image, "
                f"got T={input_images.shape[1]}."
            )
        stream_id = str(stream_id)
        newest = self._encode_streaming_camera_chunk(
            input_images,
            reset=True,
            stream_id=stream_id,
        )
        if int(newest.shape[2]) != 1:
            raise RuntimeError(
                "TMD4 streaming initialization must emit one latent, "
                f"got {tuple(newest.shape)}."
            )
        if not hasattr(self, "_streaming_condition_latents_by_id"):
            self._streaming_condition_latents_by_id = {}
        if not hasattr(self, "_streaming_condition_valid_masks_by_id"):
            self._streaming_condition_valid_masks_by_id = {}
        if not hasattr(self, "_streaming_condition_frame_indices_by_id"):
            self._streaming_condition_frame_indices_by_id = {}
        history = torch.zeros(
            (
                int(newest.shape[0]),
                int(newest.shape[1]),
                self.long_image_history_max,
                int(newest.shape[3]),
                int(newest.shape[4]),
            ),
            device=newest.device,
            dtype=newest.dtype,
        )
        history[:, :, -1:] = newest
        valid = torch.zeros(
            (int(newest.shape[0]), self.long_image_history_max),
            device=newest.device,
            dtype=torch.bool,
        )
        valid[:, -1] = True
        frame_indices = torch.full(
            (int(newest.shape[0]), self.long_image_history_max),
            -1,
            device=newest.device,
            dtype=torch.long,
        )
        frame_indices[:, -1] = 0
        self._streaming_condition_latents_by_id[stream_id] = history
        self._streaming_condition_valid_masks_by_id[stream_id] = valid
        self._streaming_condition_frame_indices_by_id[stream_id] = frame_indices
        if stream_id == "default":
            self._streaming_condition_latents = history
            self._streaming_condition_valid_mask = valid
            self._streaming_condition_frame_indices = frame_indices
        return history

    @torch.no_grad()
    def update_streaming_history(
        self,
        input_images: torch.Tensor,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        """Append one latent produced from four new stride-4 real images."""
        stream_id = str(stream_id)
        previous = self._streaming_condition_latents_by_id.get(stream_id)
        previous_valid = self._streaming_condition_valid_masks_by_id.get(
            stream_id
        )
        previous_indices = self._streaming_condition_frame_indices_by_id.get(
            stream_id
        )
        if previous is None or previous_valid is None or previous_indices is None:
            raise RuntimeError(
                "TMD4 streaming history must be initialized before update: "
                f"stream_id={stream_id!r}."
            )
        input_images = self._normalize_inference_images(input_images)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if int(input_images.shape[1]) != temporal_factor:
            raise ValueError(
                "TMD4 streaming update requires four new real images, "
                f"got T={input_images.shape[1]}."
            )
        newest = self._encode_streaming_camera_chunk(
            input_images,
            reset=False,
            stream_id=stream_id,
        )
        if int(newest.shape[2]) != 1:
            raise RuntimeError(
                "TMD4 streaming update must emit one latent, "
                f"got {tuple(newest.shape)}."
            )
        history = torch.cat([previous[:, :, 1:], newest], dim=2).contiguous()
        valid = torch.cat(
            [
                previous_valid[:, 1:],
                torch.ones(
                    (int(previous_valid.shape[0]), 1),
                    device=previous_valid.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        latest_index = previous_indices.masked_fill(~previous_valid, -1).max(
            dim=1,
            keepdim=True,
        ).values + 1
        frame_indices = torch.cat(
            [previous_indices[:, 1:], latest_index],
            dim=1,
        )
        self._streaming_condition_latents_by_id[stream_id] = history
        self._streaming_condition_valid_masks_by_id[stream_id] = valid
        self._streaming_condition_frame_indices_by_id[stream_id] = frame_indices
        if stream_id == "default":
            self._streaming_condition_latents = history
            self._streaming_condition_valid_mask = valid
            self._streaming_condition_frame_indices = frame_indices
        return history

    def get_streaming_condition_frame_indices(
        self,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        stream_id = str(stream_id)
        by_id = getattr(
            self,
            "_streaming_condition_frame_indices_by_id",
            None,
        )
        frame_indices = None if by_id is None else by_id.get(stream_id)
        if stream_id == "default" and frame_indices is None:
            frame_indices = getattr(
                self,
                "_streaming_condition_frame_indices",
                None,
            )
        if frame_indices is None:
            raise RuntimeError(
                "TMD4 streaming positions are not initialized: "
                f"stream_id={stream_id!r}."
            )
        return frame_indices

    def _select_inference_history(
        self,
        condition_latents: torch.Tensor,
        condition_valid: torch.BoolTensor,
        condition_frame_indices: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """Crop the persistent history bank to the configured recent window."""
        window = int(self.long_image_inference_window)
        return (
            condition_latents[:, :, -window:].contiguous(),
            condition_valid[:, -window:].contiguous(),
            condition_frame_indices[:, -window:].contiguous(),
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Infer Action from clean real H<=8 using the TMD4 training mask."""
        del prompt
        self.eval()
        if int(action_horizon) != int(self.action_horizon):
            raise ValueError(
                f"TMD4 requires action_horizon={self.action_horizon}, "
                f"got {action_horizon}."
            )
        images = self._normalize_inference_images(input_image)
        if int(images.shape[0]) != 1:
            raise ValueError("TMD4 closed-loop inference requires batch size one.")
        condition_latents = kwargs.pop("precomputed_condition_latents", None)
        condition_valid = kwargs.pop(
            "precomputed_condition_valid_mask",
            None,
        )
        condition_frame_indices = kwargs.pop(
            "precomputed_condition_frame_indices",
            None,
        )
        if (
            condition_latents is None
            or condition_valid is None
            or condition_frame_indices is None
        ):
            raise ValueError(
                "TMD4 inference requires its episode-persistent streaming "
                "VAE history, valid mask and real frame indices."
            )
        condition_latents = condition_latents.to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        condition_valid = condition_valid.to(
            device=self.device,
            dtype=torch.bool,
        )
        condition_frame_indices = condition_frame_indices.to(
            device=self.device,
            dtype=torch.long,
        )
        if condition_valid.ndim == 1:
            condition_valid = condition_valid.unsqueeze(0)
        if condition_frame_indices.ndim == 1:
            condition_frame_indices = condition_frame_indices.unsqueeze(0)
        stored_history = self.long_image_history_max
        if int(condition_latents.shape[2]) != stored_history or tuple(
            condition_valid.shape
        ) != (1, stored_history) or tuple(condition_frame_indices.shape) != (
            1,
            stored_history,
        ):
            raise ValueError(
                "TMD4 streaming history mismatch: "
                f"latents={tuple(condition_latents.shape)}, "
                f"valid={tuple(condition_valid.shape)}, "
                f"positions={tuple(condition_frame_indices.shape)}, "
                f"expected_H={stored_history}."
            )
        expected_history = self.long_image_inference_window
        (
            condition_latents,
            condition_valid,
            condition_frame_indices,
        ) = self._select_inference_history(
            condition_latents,
            condition_valid,
            condition_frame_indices,
        )
        if not bool(condition_valid.any().item()):
            raise ValueError("TMD4 requires at least one valid real history latent.")

        condition_cache = self._prepare_inference_condition_cache(
            context=kwargs.pop("context", None),
            context_mask=kwargs.pop("context_mask", None),
            proprio=kwargs.pop("proprio", None),
            proprio_mask=kwargs.pop("proprio_mask", None),
            progress_memory_layers=kwargs.pop("progress_memory_layers", None),
            progress_memory_mask=kwargs.pop("progress_memory_mask", None),
            progress_instruction_layers=kwargs.pop(
                "progress_instruction_layers",
                None,
            ),
            progress_instruction_mask=kwargs.pop(
                "progress_instruction_mask",
                None,
            ),
            progress_subtask_tokens=kwargs.pop(
                "progress_subtask_tokens",
                None,
            ),
            progress_subtask_layers=kwargs.pop(
                "progress_subtask_layers",
                None,
            ),
            progress_subtask_mask=kwargs.pop(
                "progress_subtask_mask",
                None,
            ),
            progress_active_idx=kwargs.pop("progress_active_idx", None),
            progress_plan_length=kwargs.pop("progress_plan_length", None),
        )
        num_inference_steps = int(kwargs.pop("num_inference_steps", 20))
        sigma_shift = kwargs.pop("sigma_shift", None)
        seed = kwargs.pop("seed", None)
        rand_device = str(kwargs.pop("rand_device", "cpu"))
        kwargs.pop("tiled", None)
        kwargs.pop("use_history_kv_cache", None)
        if kwargs:
            raise TypeError(f"Unsupported TMD4 inference arguments: {sorted(kwargs)}")

        fuse_flag = bool(
            getattr(
                self.video_expert,
                "fuse_vae_embedding_in_latents",
                False,
            )
        )
        history_pre = self.video_expert.pre_dit(
            x=condition_latents,
            timestep=torch.zeros(
                (1,),
                device=self.device,
                dtype=self.torch_dtype,
            ),
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            num_condition_latent_frames=expected_history,
            condition_timestep=torch.zeros(
                (1, expected_history),
                device=self.device,
                dtype=self.torch_dtype,
            ),
        )
        history_cache = self.mot.prepare_long_image_history_cache(
            video_tokens=history_pre["tokens"],
            video_freqs=history_pre["freqs"],
            video_t_mod=history_pre["t_mod"],
            video_frame_valid=condition_valid,
            video_frame_positions=condition_frame_indices,
            tokens_per_frame=int(history_pre["meta"]["tokens_per_frame"]),
            condition_cache=condition_cache,
        )

        _, latents_action = self._sample_joint_initial_noise(
            tuple(condition_latents.shape),
            (1, int(action_horizon), int(self.action_expert.action_dim)),
            seed=seed,
            rand_device=rand_device,
        )
        latents_action = self._mask_action_channels(
            latents_action.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        )
        action_schedule = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, delta_action in zip(*action_schedule):
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=step_t_action.unsqueeze(0).to(latents_action),
            )
            action_tokens = (
                self.mot.forward_action_with_long_image_history_cache(
                    action_tokens=action_pre["tokens"],
                    action_t_mod=action_pre["t_mod"],
                    action_is_pad=None,
                    condition_cache=condition_cache,
                    history_cache=history_cache,
                )
            )
            pred_action = self.action_expert.post_dit(
                action_tokens,
                action_pre,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action,
                delta_action,
                latents_action,
            )
            latents_action = self._mask_action_channels(latents_action)
        return {"action": latents_action[0].detach().float().cpu()}

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        required = (
            "long_image_history_latents",
            "long_image_history_valid_mask",
            "long_image_history_frame_indices",
            "long_image_future_latents",
            "long_image_future_frame_indices",
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"TMD4 sample is missing long Image fields: {missing}")
        history = sample["long_image_history_latents"].to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        history_valid = sample["long_image_history_valid_mask"].to(
            device=self.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        history_frame_indices = sample[
            "long_image_history_frame_indices"
        ].to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        future = sample["long_image_future_latents"].to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        future_frame_indices = sample["long_image_future_frame_indices"].to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        if history.ndim != 5 or future.ndim != 5:
            raise ValueError("TMD4 long Image latents must be [B,C,T,H,W].")
        batch_size = int(history.shape[0])
        if tuple(history_valid.shape) != (
            batch_size,
            int(history.shape[2]),
        ):
            raise ValueError("TMD4 history valid mask does not match history.")
        if tuple(history_frame_indices.shape) != tuple(history_valid.shape):
            raise ValueError("TMD4 history positions do not match history.")
        if tuple(future_frame_indices.shape) != (
            batch_size,
            int(future.shape[2]),
        ):
            raise ValueError("TMD4 future positions do not match future.")
        if int(history.shape[2]) != self.long_image_history_max:
            raise ValueError(
                "TMD4 cached history length differs from the model contract: "
                f"{history.shape[2]} vs {self.long_image_history_max}."
            )
        if int(future.shape[2]) < self.long_image_future_chunk_max:
            raise ValueError(
                "TMD4 cache has too few future targets: "
                f"{future.shape[2]} vs {self.long_image_future_chunk_max}."
            )
        if history.shape[:2] != future.shape[:2] or history.shape[3:] != future.shape[3:]:
            raise ValueError("TMD4 history/future latent shapes are incompatible.")
        inputs.update(
            {
                "long_image_history_latents": history,
                "long_image_history_valid_mask": history_valid,
                "long_image_history_frame_indices": history_frame_indices,
                "long_image_future_latents": future,
                "long_image_future_frame_indices": future_frame_indices,
            }
        )
        return inputs

    def _prepare_long_history(
        self,
        history: torch.Tensor,
        valid: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply condition corruption, optionally keeping current clean."""
        batch_size, _, num_frames, _, _ = history.shape
        if (
            self.history_condition_noise_prob <= 0.0
            or torch.rand((), device=history.device).item()
            >= self.history_condition_noise_prob
        ):
            return history, torch.zeros(
                (batch_size, num_frames),
                device=history.device,
                dtype=history.dtype,
            )

        u = torch.rand(
            (num_frames,),
            device=history.device,
            dtype=torch.float32,
        )
        u = self.history_condition_noise_u_min + u * (
            self.history_condition_noise_u_max
            - self.history_condition_noise_u_min
        )
        steps = int(self.train_video_scheduler.num_train_timesteps)
        timestep_ids = torch.floor(u * steps).clamp(min=0, max=steps - 1)
        base_sigma = 1.0 - timestep_ids / float(steps)
        sigma = self.train_video_scheduler._phi(
            base_sigma,
            self.train_video_scheduler.shift,
        )
        timesteps = (sigma * float(steps)).to(history.dtype)
        sigma_view = sigma.to(history.dtype).view(1, 1, num_frames, 1, 1)
        noisy = (1.0 - sigma_view) * history + sigma_view * torch.randn_like(history)
        corruption_valid = valid
        if getattr(self, "history_condition_keep_current_clean", False):
            # Long-image clips are right-aligned and the last valid slot is the
            # action anchor. Preserve its exact spatial evidence while keeping
            # the original LingBot corruption distribution for older history.
            corruption_valid = valid.clone()
            corruption_valid[:, -1] = False
        noisy = torch.where(
            corruption_valid[:, None, :, None, None],
            noisy,
            history,
        )
        timesteps = timesteps.unsqueeze(0).expand(batch_size, -1)
        timesteps = torch.where(
            corruption_valid,
            timesteps,
            torch.zeros_like(timesteps),
        )
        return noisy, timesteps

    def _sample_long_image_shape(self, device: torch.device) -> tuple[int, int]:
        """Sample one global microbatch shape shared by every DDP rank.

        The previous per-rank sampling preserved each sample's marginal
        distribution but made every collective wait for the longest of eight
        independent windows.  Sharing the two augmentation draws matches the
        ordinary meaning of one batch-level LingBot augmentation while keeping
        exactly the same uniform window/chunk distributions.
        """
        sampled = torch.stack(
            [
                torch.randint(
                    self.long_image_window_min,
                    self.long_image_window_max + 1,
                    (),
                    device=device,
                ),
                torch.randint(
                    self.long_image_future_chunk_min,
                    self.long_image_future_chunk_max + 1,
                    (),
                    device=device,
                ),
            ]
        ).to(dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(sampled, src=0)
        return int(sampled[0].item()), int(sampled[1].item())

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        history_all = inputs["long_image_history_latents"]
        history_valid_all = inputs["long_image_history_valid_mask"]
        history_positions_all = inputs["long_image_history_frame_indices"]
        future_all = inputs["long_image_future_latents"]
        future_positions_all = inputs["long_image_future_frame_indices"]
        batch_size = int(history_all.shape[0])

        window, future_chunk = self._sample_long_image_shape(
            history_all.device
        )
        history = history_all[:, :, -window:].contiguous()
        history_valid = history_valid_all[:, -window:].contiguous()
        history_positions = history_positions_all[:, -window:].contiguous()
        future = future_all[:, :, :future_chunk].contiguous()
        future_positions = future_positions_all[:, :future_chunk].contiguous()
        noisy_history, history_timesteps = self._prepare_long_history(
            history,
            history_valid,
        )

        noise_video = torch.randn_like(future)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=future_chunk,
            device=self.device,
            dtype=future.dtype,
        )
        sigma_video = (
            timestep_video / float(self.train_video_scheduler.num_train_timesteps)
        ).to(future).view(1, 1, future_chunk, 1, 1)
        noisy_future = (1.0 - sigma_video) * future + sigma_video * noise_video
        target_video = noise_video - future
        noisy_video = torch.cat([noisy_history, noisy_future], dim=2)
        video_frame_valid = torch.cat(
            [
                history_valid,
                torch.ones(
                    (batch_size, future_chunk),
                    device=self.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )

        action = self._mask_action_channels(inputs["action"])
        action_is_pad = inputs["action_is_pad"]
        noise_action = self._mask_action_channels(torch.randn_like(action))
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self._mask_action_channels(
            self.train_action_scheduler.add_noise(
                action,
                noise_action,
                timestep_action,
            )
        )
        target_action = self._mask_action_channels(
            self.train_action_scheduler.training_target(
                action,
                noise_action,
                timestep_action,
            )
        )

        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=timestep_video.unsqueeze(0).expand(batch_size, -1),
            action=action,
            fuse_vae_embedding_in_latents=inputs[
                "fuse_vae_embedding_in_latents"
            ],
            num_condition_latent_frames=window,
            condition_timestep=history_timesteps,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        checkpoint_long = (
            window + future_chunk >= self.long_image_checkpoint_min_frames
        )
        previous_video_checkpoint = bool(
            self.video_expert.base.use_gradient_checkpointing
        )
        previous_attention_checkpoint = bool(self.mot.checkpoint_attention)
        self.video_expert.base.use_gradient_checkpointing = checkpoint_long
        self.mot.checkpoint_attention = checkpoint_long
        try:
            tokens_out = self.mot.forward_long_image(
                video_tokens=video_pre["tokens"],
                action_tokens=action_pre["tokens"],
                video_freqs=video_pre["freqs"],
                action_freqs=action_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                action_t_mod=action_pre["t_mod"],
                video_frame_valid=video_frame_valid,
                video_frame_positions=torch.cat(
                    [history_positions, future_positions],
                    dim=1,
                ),
                action_is_pad=action_is_pad,
                condition_frames=window,
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                **self._mot_training_kwargs(inputs),
            )
        finally:
            self.video_expert.base.use_gradient_checkpointing = (
                previous_video_checkpoint
            )
            self.mot.checkpoint_attention = previous_attention_checkpoint
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_video = pred_video[:, :, window:]
        pred_action = self.action_expert.post_dit(
            tokens_out["action"],
            action_pre,
        )

        video_element = (pred_video.float() - target_video.float()).square()
        video_per_frame = video_element.mean(dim=(1, 3, 4))
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video
        ).to(video_per_frame).view(1, future_chunk)
        video_per_sample = (video_per_frame * video_weight).mean(dim=1)

        action_loss_token = self._action_loss_per_token(
            pred_action,
            target_action,
        )
        if action_is_pad is None:
            action_per_sample = action_loss_token.mean(dim=1)
        else:
            action_valid = (~action_is_pad).to(action_loss_token)
            action_per_sample = (
                (action_loss_token * action_valid).sum(dim=1)
                / action_valid.sum(dim=1).clamp(min=1.0)
            )
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(action_per_sample)
        action_per_sample = action_per_sample * action_weight

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
        loss_video = (video_per_sample * sample_weight).sum() / batch_size
        loss_action = (action_per_sample * sample_weight).sum() / batch_size
        total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        return total, {
            "loss_video": self.loss_lambda_video * loss_video.detach(),
            "loss_action": self.loss_lambda_action * loss_action.detach(),
            "long_image_window": torch.tensor(
                float(window),
                device=total.device,
            ),
            "long_image_future_chunk": torch.tensor(
                float(future_chunk),
                device=total.device,
            ),
        }


ProgressTMD = LongImageProgressTMD


__all__ = ["LongImageProgressTMD", "ProgressTMD"]
