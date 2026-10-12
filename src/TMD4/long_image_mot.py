from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from TMD3.state_progress_mot import StateProgressMoT
from TMD4.gigaworld_backbone import (
    adopt_gigaworld_layout,
    convert_legacy_tmd4_state_dict,
    load_gigaworld_checkpoint,
)
from fastwam.models.wan22.wan_video_dit import flash_attention


# Eager FlexAttention uses the dense SDPA backward fallback for this sequence
# length, which defeats the block mask and requires several extra GiB per
# layer.  LingBot compiles the higher-order op so both forward and backward
# use its sparse Triton kernels.  Dynamic sequence length is required because
# window/chunk augmentation changes the number of Image frames per batch.
_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
_compiled_create_block_mask = torch.compile(create_block_mask)
_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}


class LongImageStateProgressMoT(StateProgressMoT):
    """MoT with block-sparse, action-centered long-Image attention.

    Video and Action remain different experts.  Condition Image tokens are
    causal and never read Action or Future.  Future Image tokens read the
    sampled condition window, their own chunk, and the current Action K/V.
    Action reads every valid frame in the sampled condition window and its own
    current chunk, but never Future.  Progress/Subtask/State retain the existing
    expert-specific cross-attention scopes implemented by
    :class:`StateProgressMoT`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        legacy_video = self.mixtures["video"]
        legacy_action = self.mixtures["action"]
        adopt_gigaworld_layout(
            self,
            legacy_video,
            legacy_action,
        )
        # ``adopt_gigaworld_layout`` leaves only unregistered execution views
        # here, so serialized backbone keys exactly match the public model.

    def load_gigaworld_pretrained(self, path: str) -> dict[str, Any]:
        return load_gigaworld_checkpoint(self, path)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = convert_legacy_tmd4_state_dict(self, state_dict)
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    @staticmethod
    def build_long_image_block_mask(
        *,
        frame_valid: torch.BoolTensor,
        condition_frames: int,
        tokens_per_frame: int,
        num_heads: int,
        device: torch.device,
        action_valid: torch.BoolTensor,
        padded_tokens: int = 0,
        padded_kv_tokens: int = 0,
    ):
        if frame_valid.ndim != 2:
            raise ValueError(
                f"Long-image frame mask must be [B,F], got {tuple(frame_valid.shape)}."
            )
        batch_size, total_frames = map(int, frame_valid.shape)
        condition_frames = int(condition_frames)
        tokens_per_frame = int(tokens_per_frame)
        padded_tokens = int(padded_tokens)
        padded_kv_tokens = int(padded_kv_tokens)
        if not 0 < condition_frames <= total_frames:
            raise ValueError(
                "Long-image sequence must contain condition frames and may "
                "optionally contain prediction frames; "
                f"got condition={condition_frames}, total={total_frames}."
            )
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive.")
        if padded_tokens < 0:
            raise ValueError("padded_tokens cannot be negative.")
        if action_valid.ndim != 2 or int(action_valid.shape[0]) != batch_size:
            raise ValueError(
                "Action validity mask must be [B,A], got "
                f"{tuple(action_valid.shape)} for batch={batch_size}."
            )
        action_length = int(action_valid.shape[1])
        if action_length <= 0:
            raise ValueError("Action sequence must contain at least one token.")
        if padded_kv_tokens < 0:
            raise ValueError("padded_kv_tokens cannot be negative.")
        valid = frame_valid.to(device=device, dtype=torch.bool)
        action_valid = action_valid.to(device=device, dtype=torch.bool)
        real_sequence_length = total_frames * tokens_per_frame
        video_sequence_length = real_sequence_length + padded_tokens
        condition_tokens = condition_frames * tokens_per_frame
        future_tokens = real_sequence_length - condition_tokens
        action_start = condition_tokens
        action_stop = action_start + action_length
        future_start = action_stop
        future_stop = future_start + future_tokens
        kv_sequence_length = future_stop + padded_kv_tokens

        # Condition frames are strictly causal by frame.  All noisy Future
        # frames belong to the one sampled prediction chunk and may interact
        # within that chunk.  Future additionally reads current Action K/V.
        # Condition never reads either Future or Action.  Queries keep the
        # original [Condition, Future, Padding] order, while the compact K/V
        # layout is [Condition, Action, Future, Padding].  Putting Action in
        # the middle lets its attention reuse a contiguous [Condition, Action]
        # prefix without rebuilding another K/V tensor per layer.
        def mask_mod(b, h, q_idx, kv_idx):
            del h
            q_real = q_idx < real_sequence_length
            q_frame = torch.clamp(
                q_idx // tokens_per_frame,
                max=total_frames - 1,
            )
            kv_condition = kv_idx < condition_tokens
            kv_action = (kv_idx >= action_start) & (kv_idx < action_stop)
            kv_future = (kv_idx >= future_start) & (kv_idx < future_stop)
            condition_kv_frame = torch.clamp(
                kv_idx // tokens_per_frame,
                max=condition_frames - 1,
            )
            future_kv_frame = torch.clamp(
                condition_frames
                + torch.clamp(
                    (kv_idx - future_start) // tokens_per_frame,
                    min=0,
                    max=max(total_frames - condition_frames - 1, 0),
                ),
                max=total_frames - 1,
            )
            kv_frame = torch.where(
                kv_condition,
                condition_kv_frame,
                future_kv_frame,
            )
            q_condition = q_frame < condition_frames
            condition_causal = q_condition & kv_condition & (
                kv_frame <= q_frame
            )
            future_to_condition = (~q_condition) & kv_condition
            future_same_chunk = (~q_condition) & kv_future
            video_allowed = (
                (kv_condition | kv_future)
                & valid[b, kv_frame]
                & (
                    condition_causal
                    | future_to_condition
                    | future_same_chunk
                )
            )
            action_idx = torch.clamp(
                kv_idx - action_start,
                min=0,
                max=action_length - 1,
            )
            action_allowed = (
                (~q_condition)
                & kv_action
                & action_valid[b, action_idx]
            )
            return (
                q_real
                & valid[b, q_frame]
                & (video_allowed | action_allowed)
            )

        return _compiled_create_block_mask(
            mask_mod,
            batch_size,
            1,
            video_sequence_length,
            kv_sequence_length,
            device=device,
            BLOCK_SIZE=128,
            _compile=True,
        )

    @torch.no_grad()
    def prepare_long_image_history_cache(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_frame_valid: torch.BoolTensor,
        video_frame_positions: torch.LongTensor,
        tokens_per_frame: int,
        condition_cache: dict[str, Any],
    ) -> dict[str, Any]:
        """Encode clean real Image history once for one Action replan.

        The causal visibility and physical time positions are identical to
        ``forward_long_image``.  Every slot is retained, while invalid leading
        padding remains hidden by ``history_key_valid``.  This exactly matches
        training, where Action reads the complete sampled condition window.
        The cache is rebuilt for every replan because current Progress and
        decoded Subtask condition the Video representation.
        """
        if self.training:
            raise RuntimeError("Long-image history caching is inference-only.")
        if video_frame_valid.ndim != 2:
            raise ValueError("video_frame_valid must be [B,H].")
        batch_size, history_frames = map(int, video_frame_valid.shape)
        tokens_per_frame = int(tokens_per_frame)
        real_video_length = history_frames * tokens_per_frame
        if tuple(video_tokens.shape[:2]) != (batch_size, real_video_length):
            raise ValueError(
                "Long-image inference token layout mismatch: "
                f"tokens={tuple(video_tokens.shape[:2])}, "
                f"expected={(batch_size, real_video_length)}."
            )
        condition_layers = condition_cache.get("layers")
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")

        frame_valid = video_frame_valid.to(
            device=video_tokens.device,
            dtype=torch.bool,
        )
        frame_positions = video_frame_positions.to(
            device=video_tokens.device,
            dtype=torch.long,
        )
        self._validate_frame_positions(frame_positions, frame_valid)
        token_valid = frame_valid.repeat_interleave(tokens_per_frame, dim=1)
        frame_ids = torch.arange(
            history_frames,
            device=video_tokens.device,
        ).repeat_interleave(tokens_per_frame)
        causal = frame_ids[:, None] >= frame_ids[None, :]
        video_mask = (
            token_valid[:, None, :, None]
            & token_valid[:, None, None, :]
            & causal[None, None]
        )
        physical_video_freqs = self._video_freqs_at_frame_positions(
            video_freqs,
            frame_positions,
            tokens_per_frame,
        )
        tokens = video_tokens
        layers = []
        for layer_idx in range(self.num_layers):
            query, key, value, post = self._build_dit_attention_io(
                self.video_expert,
                self.video_expert.blocks[layer_idx],
                tokens,
                physical_video_freqs,
                video_t_mod,
            )
            mixed = flash_attention(
                q=query,
                k=key,
                v=value,
                num_heads=self.num_heads,
                ctx_mask=video_mask,
            )
            layers.append(
                {
                    "video_key": key.detach(),
                    "video_value": value.detach(),
                }
            )
            tokens = self._post_block(
                post,
                mixed,
                condition_layers[layer_idx]["cross"]["video"],
            )
        return {
            "layers": layers,
            "history_key_valid": token_valid.detach(),
            "latest_frame_position": frame_positions.masked_fill(
                ~frame_valid,
                -1,
            ).max(dim=1).values.detach(),
            "history_frames": history_frames,
            "tokens_per_frame": tokens_per_frame,
        }

    @torch.no_grad()
    def forward_action_with_long_image_history_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_is_pad: Optional[torch.BoolTensor],
        condition_cache: dict[str, Any],
        history_cache: dict[str, Any],
    ) -> torch.Tensor:
        """Denoise Action against every valid cached Image latent."""
        if self.training:
            raise RuntimeError("Long-image Action caching is inference-only.")
        condition_layers = condition_cache.get("layers")
        history_layers = history_cache.get("layers")
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        if not isinstance(history_layers, list) or len(history_layers) != self.num_layers:
            raise ValueError("History-cache depth does not match MoT depth.")
        batch_size, action_length = map(int, action_tokens.shape[:2])
        if action_is_pad is None:
            action_valid = torch.ones(
                (batch_size, action_length),
                device=action_tokens.device,
                dtype=torch.bool,
            )
        else:
            action_valid = (~action_is_pad).to(
                device=action_tokens.device,
                dtype=torch.bool,
            )
        history_valid = history_cache["history_key_valid"].to(
            device=action_tokens.device,
            dtype=torch.bool,
        )
        key_valid = torch.cat([history_valid, action_valid], dim=1)
        action_mask = (
            action_valid[:, None, :, None]
            & key_valid[:, None, None, :]
        )
        latest_frame_position = history_cache["latest_frame_position"].to(
            device=action_tokens.device,
            dtype=torch.long,
        )
        physical_action_freqs = self._action_freqs_after_latest_frame(
            latest_frame_position,
            action_length,
            action_tokens.device,
        )
        tokens = action_tokens
        for layer_idx in range(self.num_layers):
            query, key, value, post = self._build_dit_attention_io(
                self.action_expert,
                self.action_expert.blocks[layer_idx],
                tokens,
                physical_action_freqs,
                action_t_mod,
            )
            layer_history = history_layers[layer_idx]
            mixed = flash_attention(
                q=query,
                k=torch.cat([layer_history["video_key"], key], dim=1),
                v=torch.cat([layer_history["video_value"], value], dim=1),
                num_heads=self.num_heads,
                ctx_mask=action_mask,
            )
            tokens = self._post_block(
                post,
                mixed,
                condition_layers[layer_idx]["cross"]["action"],
            )
        return tokens

    @staticmethod
    def _validate_frame_positions(
        positions: torch.LongTensor,
        valid: torch.BoolTensor,
    ) -> None:
        if positions.ndim != 2 or tuple(positions.shape) != tuple(valid.shape):
            raise ValueError(
                "Long-image positions and valid mask must both be [B,F], got "
                f"positions={tuple(positions.shape)}, valid={tuple(valid.shape)}."
            )
        # Keep these checks asynchronous on CUDA; host-side ``.item()`` here
        # would introduce several global synchronizations on every train step.
        torch._assert_async(
            (positions[~valid] == -1).all(),
            "Invalid long-image frames must use position -1.",
        )
        torch._assert_async(
            (positions[valid] >= 0).all(),
            "Real long-image frames require non-negative positions.",
        )
        torch._assert_async(
            valid.any(dim=1).all(),
            "Every long-image sample needs one real frame.",
        )
        # Fixed-shape padding is a leading prefix only.  Consecutive real VAE
        # latents retain their actual phase-stream coordinates.
        torch._assert_async(
            ~(valid[:, :-1] & ~valid[:, 1:]).any(),
            "Long-image padding must precede all real frames.",
        )
        adjacent = valid[:, :-1] & valid[:, 1:]
        deltas = positions[:, 1:] - positions[:, :-1]
        torch._assert_async(
            ~(adjacent & (deltas != 1)).any(),
            "Real long-image frame positions must be consecutive.",
        )

    def _video_freqs_at_frame_positions(
        self,
        freqs: torch.Tensor,
        frame_positions: torch.LongTensor,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        """Replace local frame IDs with each sample's real stream positions."""
        if freqs.ndim != 3:
            raise ValueError(
                f"Video RoPE must be [S,1,D], got {tuple(freqs.shape)}."
            )
        batch_size, num_frames = map(int, frame_positions.shape)
        tokens_per_frame = int(tokens_per_frame)
        sequence_length = num_frames * tokens_per_frame
        if int(freqs.shape[0]) != sequence_length:
            raise ValueError(
                "Video RoPE/frame layout mismatch: "
                f"freqs={freqs.shape[0]}, expected={sequence_length}."
            )
        base_freqs = self.video_expert.base.freqs[0]
        temporal_width = int(base_freqs.shape[-1])
        if int(freqs.shape[-1]) < temporal_width:
            raise ValueError("Video RoPE has fewer entries than its time axis.")
        one_step = base_freqs[1].to(device=freqs.device)
        local_positions = torch.arange(
            num_frames,
            device=freqs.device,
            dtype=torch.long,
        ).view(1, num_frames).expand(batch_size, -1)
        real_positions = frame_positions.to(device=freqs.device).clamp(min=0)
        offsets = (real_positions - local_positions).repeat_interleave(
            tokens_per_frame,
            dim=1,
        )
        phase = offsets.to(torch.float64)[..., None] * torch.angle(one_step)[
            None,
            None,
        ]
        factor = torch.polar(torch.ones_like(phase), phase).to(freqs.dtype)
        shifted = freqs.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        shifted[..., :temporal_width] = (
            shifted[..., :temporal_width]
            * factor.unsqueeze(2)
        )
        return shifted

    def _action_freqs_after_latest_frame(
        self,
        latest_frame_position: torch.LongTensor,
        action_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if latest_frame_position.ndim != 1:
            raise ValueError("Latest long-image position must be [B].")
        torch._assert_async(
            (latest_frame_position >= 0).all(),
            "Latest long-image position must be non-negative.",
        )
        action_length = int(action_length)
        fractions = torch.arange(
            1,
            action_length + 1,
            device=device,
            dtype=torch.float32,
        ) / float(action_length + 1)
        temporal_positions = latest_frame_position.to(
            device=device,
            dtype=torch.float32,
        )[:, None] + fractions[None]
        base_freqs = self.video_expert.base.freqs
        axis_freqs = []
        for axis, positions in enumerate(
            (
                temporal_positions,
                torch.full_like(temporal_positions, -1.0),
                torch.full_like(temporal_positions, -1.0),
            )
        ):
            one_step = base_freqs[axis][1].to(device=device)
            phase = positions.to(torch.float64)[..., None] * torch.angle(
                one_step
            )[None, None]
            axis_freqs.append(torch.polar(torch.ones_like(phase), phase))
        # Match LingBot's Action grid: (fractional global time, -1, -1).
        # This keeps Action self-attention one-dimensional in time while making
        # its Q/K basis compatible with the Video expert's 3D RoPE layout.
        return torch.cat(axis_freqs, dim=-1).unsqueeze(2)

    def forward_long_image(
        self,
        *,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_frame_valid: torch.BoolTensor,
        video_frame_positions: torch.LongTensor,
        action_is_pad: Optional[torch.BoolTensor],
        condition_frames: int,
        tokens_per_frame: int,
        progress_logits: Optional[torch.Tensor] = None,
        subtask_tokens: Optional[torch.Tensor] = None,
        subtask_mask: Optional[torch.BoolTensor] = None,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
        progress_active_idx: Optional[torch.Tensor] = None,
        progress_plan_length: Optional[torch.Tensor] = None,
        instruction: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.BoolTensor] = None,
    ) -> dict[str, torch.Tensor]:
        del action_freqs
        if instruction is not None or instruction_mask is not None:
            if instruction is None or instruction_mask is None:
                raise ValueError(
                    "TMD4 instruction conditioning requires both instruction "
                    "and instruction_mask."
                )
            if any(
                value is not None
                for value in (
                    progress_logits,
                    subtask_tokens,
                    subtask_mask,
                    progress_active_idx,
                    progress_plan_length,
                )
            ):
                raise ValueError(
                    "TMD4 instruction-only conditioning cannot be combined "
                    "with Progress or decoded-subtask inputs."
                )
            self._validate_instruction_conditions(
                instruction,
                instruction_mask,
                state_history,
                state_mask,
            )
            precomputed = self._precompute_instruction_inputs(
                instruction,
                instruction_mask,
                state_history,
                state_mask,
            )
            cross_layers = [
                self._instruction_cross_payload(
                    precomputed,
                    layer_idx,
                    cache_kv=False,
                )
                for layer_idx in range(self.num_layers)
            ]
        else:
            if self.progress_injection_mode not in {
                "cross_attention_video_only",
                "disabled",
            }:
                raise ValueError(
                    "TMD4 long-image training supports Video-only numeric Progress "
                    "or decoded-subtask/state conditioning with numeric Progress disabled."
                )
            if progress_logits is None or subtask_tokens is None or subtask_mask is None:
                raise ValueError(
                    "TMD4 Progress conditioning requires logits, decoded "
                    "subtask tokens, and subtask mask."
                )
            self._validate_conditions(
                progress_logits,
                subtask_tokens,
                subtask_mask,
                state_history,
                state_mask,
                progress_active_idx,
                progress_plan_length,
            )
            precomputed = self._precompute_condition_inputs(
                progress_logits,
                subtask_tokens,
                subtask_mask,
                state_history,
                state_mask,
                progress_active_idx,
                progress_plan_length,
            )
            shared_cross = self._layerwise_cross_payload(
                precomputed,
                0,
                cache_kv=False,
            )
            cross_layers = [shared_cross] * self.num_layers

        batch_size = int(video_tokens.shape[0])
        action_length = int(action_tokens.shape[1])
        condition_frames = int(condition_frames)
        tokens_per_frame = int(tokens_per_frame)
        real_video_length = int(video_frame_valid.shape[1]) * tokens_per_frame
        if int(video_tokens.shape[1]) != real_video_length:
            raise ValueError(
                "Long-image token/frame layout mismatch: "
                f"tokens={video_tokens.shape[1]}, expected={real_video_length}."
            )
        if int(action_tokens.shape[0]) != batch_size:
            raise ValueError("Video/Action batch sizes differ.")
        frame_positions = video_frame_positions.to(
            device=video_tokens.device,
            dtype=torch.long,
        )
        frame_valid = video_frame_valid.to(
            device=video_tokens.device,
            dtype=torch.bool,
        )
        self._validate_frame_positions(frame_positions, frame_valid)
        if action_is_pad is None:
            action_valid = torch.ones(
                (batch_size, action_length),
                device=action_tokens.device,
                dtype=torch.bool,
            )
        else:
            action_valid = (~action_is_pad).to(
                device=action_tokens.device,
                dtype=torch.bool,
            )
            if tuple(action_valid.shape) != (batch_size, action_length):
                raise ValueError(
                    "Action padding mask must be [B,A], got "
                    f"{tuple(action_valid.shape)}."
                )

        # LingBot pads its fused long sequence to a 128-token boundary before
        # FlexAttention.  Do the same for the Image domain; invalid padding is
        # excluded by the block mask and stripped before Video post_dit.
        physical_video_freqs = self._video_freqs_at_frame_positions(
            video_freqs,
            frame_positions,
            tokens_per_frame,
        )
        padded_tokens = (-real_video_length) % 128
        if padded_tokens:
            video_tokens = F.pad(video_tokens, (0, 0, 0, padded_tokens))
            physical_video_freqs = torch.cat(
                [
                    physical_video_freqs,
                    physical_video_freqs.new_zeros(
                        (
                            batch_size,
                            padded_tokens,
                            *physical_video_freqs.shape[2:],
                        )
                    ),
                ],
                dim=1,
            )
            video_t_mod = F.pad(video_t_mod, (0, 0, 0, 0, 0, padded_tokens))

        padded_kv_tokens = (-(real_video_length + action_length)) % 128
        video_block_mask = self.build_long_image_block_mask(
            frame_valid=video_frame_valid,
            condition_frames=condition_frames,
            tokens_per_frame=tokens_per_frame,
            num_heads=self.num_heads,
            device=video_tokens.device,
            action_valid=action_valid,
            padded_tokens=padded_tokens,
            padded_kv_tokens=padded_kv_tokens,
        )
        latest_condition_position = frame_positions[
            :, :condition_frames
        ].masked_fill(
            ~frame_valid[:, :condition_frames],
            -1,
        ).max(dim=1).values
        physical_action_freqs = self._action_freqs_after_latest_frame(
            latest_condition_position,
            action_length,
            action_tokens.device,
        )

        # Action reads the complete sampled condition window (H=4..8 in the
        # current recipe) plus its own chunk, but never noisy Future.
        history_stop_token = condition_frames * tokens_per_frame
        history_key_valid = video_frame_valid[:, :condition_frames].to(
            device=video_tokens.device,
            dtype=torch.bool,
        ).repeat_interleave(tokens_per_frame, dim=1)
        action_key_valid = torch.cat(
            [history_key_valid, action_valid],
            dim=1,
        )
        action_attention_mask = (
            action_valid[:, None, :, None]
            & action_key_valid[:, None, None, :]
        )
        action_context_length = history_stop_token + action_length

        tokens = {"video": video_tokens, "action": action_tokens}
        for layer_idx in range(self.num_layers):
            video_query, video_key, video_value, video_post = (
                self._build_dit_attention_io(
                    self.video_expert,
                    self.video_expert.blocks[layer_idx],
                    tokens["video"],
                    physical_video_freqs,
                    video_t_mod,
                )
            )
            action_query, action_key, action_value, action_post = (
                self._build_dit_attention_io(
                    self.action_expert,
                    self.action_expert.blocks[layer_idx],
                    tokens["action"],
                    physical_action_freqs,
                    action_t_mod,
                )
            )

            q_video = rearrange(
                video_query,
                "b s (h d) -> b h s d",
                h=self.num_heads,
            )
            # One compact K/V allocation serves both attention paths.  The
            # Video/Future FlexAttention consumes the complete packed tensor;
            # Action consumes the contiguous [Condition, Action] prefix below.
            # Video padding K/V is omitted because it was masked in the old
            # layout and can never affect a real query.
            joint_context_key = torch.cat(
                [
                    video_key[:, :history_stop_token],
                    action_key,
                    video_key[:, history_stop_token:real_video_length],
                ],
                dim=1,
            )
            joint_context_value = torch.cat(
                [
                    video_value[:, :history_stop_token],
                    action_value,
                    video_value[:, history_stop_token:real_video_length],
                ],
                dim=1,
            )
            if padded_kv_tokens:
                joint_context_key = F.pad(
                    joint_context_key,
                    (0, 0, 0, padded_kv_tokens),
                )
                joint_context_value = F.pad(
                    joint_context_value,
                    (0, 0, 0, padded_kv_tokens),
                )
            video_context_key = rearrange(
                joint_context_key,
                "b s (h d) -> b h s d",
                h=self.num_heads,
            )
            video_context_value = rearrange(
                joint_context_value,
                "b s (h d) -> b h s d",
                h=self.num_heads,
            )

            def attend_video(q, k, v):
                return _compiled_flex_attention(
                    q,
                    k,
                    v,
                    block_mask=video_block_mask,
                    kernel_options=_FLEX_KERNEL_OPTIONS,
                )

            if self.checkpoint_attention and self.training:
                mixed_video = torch.utils.checkpoint.checkpoint(
                    attend_video,
                    q_video,
                    video_context_key,
                    video_context_value,
                    use_reentrant=False,
                )
            else:
                mixed_video = attend_video(
                    q_video,
                    video_context_key,
                    video_context_value,
                )
            mixed_video = rearrange(
                mixed_video,
                "b h s d -> b s (h d)",
            )

            action_context_key = joint_context_key[:, :action_context_length]
            action_context_value = joint_context_value[:, :action_context_length]

            def attend_action(q, k, v):
                return flash_attention(
                    q=q,
                    k=k,
                    v=v,
                    num_heads=self.num_heads,
                    ctx_mask=action_attention_mask,
                )

            if self.checkpoint_attention and self.training:
                mixed_action = torch.utils.checkpoint.checkpoint(
                    attend_action,
                    action_query,
                    action_context_key,
                    action_context_value,
                    use_reentrant=False,
                )
            else:
                mixed_action = attend_action(
                    action_query,
                    action_context_key,
                    action_context_value,
                )

            cross = cross_layers[layer_idx]
            tokens["video"] = self._post_block(
                video_post,
                mixed_video,
                cross["video"],
            )
            tokens["action"] = self._post_block(
                action_post,
                mixed_action,
                cross["action"],
            )
        tokens["video"] = tokens["video"][:, :real_video_length]
        return tokens


__all__ = ["LongImageStateProgressMoT"]
