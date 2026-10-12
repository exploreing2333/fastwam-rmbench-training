from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from fastwam.models.wan22.wan_video_dit import (
    flash_attention,
    sinusoidal_embedding_1d,
)
from fastwam.utils.logging_config import get_logger
from TMD2.robometer_mot import ProgressMoT
from TMD3.state_progress_mot import continuous_rope_1d


logger = get_logger(__name__)


class InstructionStateStreamMoT(ProgressMoT):
    """Experimental dense-qpos state stream for TMD3.

    Sixty-four real qpos observations are embedded as an independent temporal
    modality.  They share the Action expert's transformer blocks, use fixed
    fractional physical-time RoPE, and participate in full attention.  Text is
    the only cross-attention context.  State has no prediction head or loss.
    This is a local TMD3 design, not the state-conditioning method used by
    LingBot-VA or MemoryWAM.
    """

    requires_video_tokens_per_frame = True

    def __init__(
        self,
        video_expert: nn.Module,
        action_expert: nn.Module,
        *,
        condition_dim: int = 4096,
        state_dim: int = 14,
        state_history_tokens: int = 64,
        state_history_latents: int = 4,
        state_steps_per_latent: int = 16,
        state_time_start: float = 0.0,
        state_time_end: float = 4.0,
        video_time_offset: float = 1.0,
        action_time_start: float = 4.0,
        action_time_end: float = 5.0,
        video_tokens_per_frame: int = 120,
        checkpoint_attention: bool = False,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            progress_hidden_dim=1,
            condition_dim=int(condition_dim),
            checkpoint_attention=bool(checkpoint_attention),
        )
        # This variant has no Progress/subtask parameters.
        del self.progress_k
        del self.progress_v
        del self.subtask_adapter

        self.state_dim = int(state_dim)
        self.state_history_tokens = int(state_history_tokens)
        self.state_history_latents = int(state_history_latents)
        self.state_steps_per_latent = int(state_steps_per_latent)
        self.state_time_start = float(state_time_start)
        self.state_time_end = float(state_time_end)
        self.video_time_offset = float(video_time_offset)
        self.action_time_start = float(action_time_start)
        self.action_time_end = float(action_time_end)
        self.video_tokens_per_frame = int(video_tokens_per_frame)

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.state_history_latents != 4:
            raise ValueError("TMD3 requires exactly four history latents.")
        if self.state_steps_per_latent != 16:
            raise ValueError("Dense-state TMD3 requires 16 states per latent.")
        if self.state_history_tokens != (
            self.state_history_latents * self.state_steps_per_latent
        ):
            raise ValueError(
                "state_history_tokens must equal history_latents * "
                "steps_per_latent."
            )
        if self.state_time_end <= self.state_time_start:
            raise ValueError("state_time_end must be greater than state_time_start.")
        if self.action_time_end <= self.action_time_start:
            raise ValueError("action_time_end must be greater than action_time_start.")
        if self.video_tokens_per_frame <= 0:
            raise ValueError("video_tokens_per_frame must be positive.")

        # The experimental state stream shares Action blocks but has its own
        # modality-specific input projection.
        self.state_encoder = nn.Linear(
            self.state_dim,
            int(self.action_expert.hidden_dim),
        )
        logger.info(
            "Initialized InstructionStateStreamMoT state=%dx%d groups=%dx%d "
            "full_attention=state+video+action cross_attention=instruction_only "
            "state_rope=[%.1f,%.1f] action_rope=[%.1f,%.1f]",
            self.state_history_tokens,
            self.state_dim,
            self.state_history_latents,
            self.state_steps_per_latent,
            self.state_time_start,
            self.state_time_end,
            self.action_time_start,
            self.action_time_end,
        )

    @staticmethod
    def _validate_mask(mask: torch.Tensor, tokens: torch.Tensor, name: str) -> None:
        if mask.ndim != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError(
                f"{name} mask must match [B,S]={tuple(tokens.shape[:2])}, "
                f"got {tuple(mask.shape)}."
            )

    def _validate_conditions(
        self,
        instruction: torch.Tensor,
        instruction_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> None:
        if instruction.ndim != 3 or int(instruction.shape[-1]) != self.condition_dim:
            raise ValueError(
                "Instruction must be [B,L,condition_dim], got "
                f"{tuple(instruction.shape)}."
            )
        self._validate_mask(instruction_mask, instruction, "instruction")
        if state_history.ndim != 3 or tuple(state_history.shape[1:]) != (
            self.state_history_tokens,
            self.state_dim,
        ):
            raise ValueError(
                "Dense state history must be "
                f"[B,{self.state_history_tokens},{self.state_dim}], got "
                f"{tuple(state_history.shape)}."
            )
        self._validate_mask(state_mask, state_history, "state")
        if int(state_history.shape[0]) != int(instruction.shape[0]):
            raise ValueError("Instruction/state batch mismatch.")
        if not bool(instruction_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain instruction context.")
        if not bool(state_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least its current state.")

    def _state_positions(self, device: torch.device) -> torch.Tensor:
        group_width = (
            self.state_time_end - self.state_time_start
        ) / float(self.state_history_latents)
        fractions = torch.arange(
            1,
            self.state_steps_per_latent + 1,
            device=device,
            dtype=torch.float32,
        ) / float(self.state_steps_per_latent + 1)
        group_starts = self.state_time_start + torch.arange(
            self.state_history_latents,
            device=device,
            dtype=torch.float32,
        ) * group_width
        return (
            group_starts[:, None] + fractions[None] * group_width
        ).reshape(-1)

    def _state_inputs(
        self,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        clean = state_history.masked_fill(~state_mask.unsqueeze(-1), 0.0)
        tokens = self.state_encoder(clean)
        positions = self._state_positions(tokens.device)
        freqs = continuous_rope_1d(positions, self.attn_head_dim).to(tokens.device)

        # State is a clean condition stream: diffusion timestep is always zero.
        zeros = torch.zeros(
            int(tokens.shape[0]),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        t = self.action_expert.time_embedding(
            sinusoidal_embedding_1d(
                int(self.action_expert.freq_dim),
                zeros,
            )
        )
        t_mod = self.action_expert.time_projection(t).unflatten(
            1,
            (6, int(self.action_expert.hidden_dim)),
        )
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "mask": state_mask.bool(),
        }

    def video_freqs_on_physical_axis(self, freqs: torch.Tensor) -> torch.Tensor:
        if self.video_time_offset == 0.0:
            return freqs
        base_freqs = self.video_expert.base.freqs[0]
        temporal_width = int(base_freqs.shape[-1])
        one_step = base_freqs[1].to(device=freqs.device)
        phase = torch.angle(one_step) * self.video_time_offset
        factor = torch.polar(torch.ones_like(phase), phase).to(freqs.dtype)
        shifted = freqs.clone()
        shifted[..., :temporal_width] = (
            shifted[..., :temporal_width] * factor.view(1, 1, -1)
        )
        return shifted

    def action_freqs_on_physical_axis(
        self,
        action_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        fractions = torch.arange(
            1,
            int(action_length) + 1,
            device=device,
            dtype=torch.float32,
        ) / float(int(action_length) + 1)
        positions = self.action_time_start + fractions * (
            self.action_time_end - self.action_time_start
        )
        return continuous_rope_1d(positions, self.attn_head_dim)

    @staticmethod
    def _right_dit_mask(
        dit_mask: torch.Tensor,
        batch_size: int,
        dit_length: int,
    ) -> torch.BoolTensor:
        if dit_mask.ndim == 2:
            right = dit_mask[None, None].expand(batch_size, 1, -1, -1)
        elif dit_mask.ndim == 4:
            if int(dit_mask.shape[0]) not in (1, batch_size):
                raise ValueError("DiT attention-mask batch does not match tokens.")
            right = dit_mask.expand(batch_size, -1, -1, -1)
        else:
            raise ValueError("DiT attention mask must be [Q,K] or [B,1,Q,K].")
        if tuple(right.shape[-2:]) != (dit_length, dit_length):
            raise ValueError(
                "DiT attention mask must cover Video+Action exactly: "
                f"mask={tuple(right.shape)}, tokens={dit_length}."
            )
        return right.bool()

    def _state_full_attention_mask(
        self,
        dit_mask: torch.Tensor,
        state_mask: torch.BoolTensor,
        video_length: int,
        action_length: int,
    ) -> torch.BoolTensor:
        """Build [State,Video,Action] mask without changing the TMD2 block."""
        batch_size, state_length = state_mask.shape
        if state_length != self.state_history_tokens:
            raise ValueError("State mask length does not match state stream.")
        right = self._right_dit_mask(
            dit_mask,
            batch_size,
            int(video_length) + int(action_length),
        )
        if int(video_length) % self.video_tokens_per_frame:
            raise ValueError("Video length is not divisible by tokens_per_frame.")
        video_frames = int(video_length) // self.video_tokens_per_frame
        if video_frames not in {
            self.state_history_latents,
            self.state_history_latents + 1,
        }:
            raise ValueError(
                "State64 expects four history video latents and optionally one "
                f"future latent, got {video_frames}."
            )

        total = state_length + int(video_length) + int(action_length)
        full = torch.zeros(
            batch_size,
            int(right.shape[1]),
            total,
            total,
            device=right.device,
            dtype=torch.bool,
        )
        full[:, :, state_length:, state_length:] = right

        state_valid = state_mask.to(device=right.device, dtype=torch.bool)
        # State-stream causality is defined at latent/action-frame granularity:
        # all 16 control tokens inside one history latent are mutually visible,
        # while a group cannot read a later group.
        state_groups = torch.arange(
            state_length,
            device=right.device,
        ) // self.state_steps_per_latent
        state_causal = state_groups[:, None] >= state_groups[None, :]
        full[:, :, :state_length, :state_length] = (
            state_causal[None, None]
            & state_valid[:, None, :, None]
            & state_valid[:, None, None, :]
        )
        # Keep padded query rows numerically defined; padded state keys remain
        # invisible to every real Video/Action/state query.
        invalid = ~state_valid
        diagonal = torch.arange(state_length, device=right.device)
        full[:, :, diagonal, diagonal] |= invalid[:, None, :]

        video_start = state_length
        action_start = state_length + int(video_length)
        video_right = right[:, :, :video_length, :video_length]
        video_valid = torch.diagonal(video_right, dim1=-2, dim2=-1)
        if int(video_valid.shape[1]) != 1:
            video_valid = video_valid.any(dim=1, keepdim=True)
        video_valid = video_valid[:, 0]

        # State group g is the 16-step interval ending at history latent Hg.
        # It can read only clean visual history through Hg, never Future/Action.
        for group in range(self.state_history_latents):
            s0 = group * self.state_steps_per_latent
            s1 = s0 + self.state_steps_per_latent
            v1 = (group + 1) * self.video_tokens_per_frame
            full[:, :, s0:s1, video_start : video_start + v1] = (
                state_valid[:, None, s0:s1, None]
                & video_valid[:, None, None, :v1]
            )

        # Each video latent reads states no later than its physical endpoint;
        # the Future latent reads the complete observed state history.
        for frame in range(video_frames):
            v0 = frame * self.video_tokens_per_frame
            v1 = v0 + self.video_tokens_per_frame
            visible_groups = min(frame + 1, self.state_history_latents)
            s1 = visible_groups * self.state_steps_per_latent
            full[
                :,
                :,
                video_start + v0 : video_start + v1,
                :s1,
            ] = (
                video_valid[:, None, v0:v1, None]
                & state_valid[:, None, None, :s1]
            )

        if action_length:
            action_right = right[
                :,
                :,
                video_length : video_length + action_length,
                video_length : video_length + action_length,
            ]
            action_valid = torch.diagonal(action_right, dim1=-2, dim2=-1)
            if int(action_valid.shape[1]) != 1:
                action_valid = action_valid.any(dim=1, keepdim=True)
            action_valid = action_valid[:, 0]
            full[:, :, action_start:, :state_length] = (
                action_valid[:, None, :, None]
                & state_valid[:, None, None, :]
            )
        return full

    def _precompute_inputs(
        self,
        instruction: torch.Tensor,
        instruction_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        return {
            "contexts": {
                name: self.mixtures[name].text_embedding(instruction)
                for name in ("video", "action")
            },
            "context_mask": instruction_mask[:, None, None, :],
            "state": self._state_inputs(state_history, state_mask),
        }

    def _cross_payload(
        self,
        precomputed: dict[str, Any],
        layer_idx: int,
        *,
        cache_kv: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        payloads = {}
        for name in ("video", "action"):
            context = precomputed["contexts"][name]
            if cache_kv:
                cross = self.mixtures[name].blocks[layer_idx].cross_attn
                payloads[name] = {
                    "key": cross.norm_k(cross.k(context)).detach(),
                    "value": cross.v(context).detach(),
                    "mask": precomputed["context_mask"].detach(),
                }
            else:
                payloads[name] = {
                    "context": context,
                    "mask": precomputed["context_mask"],
                }
        return payloads

    @torch.no_grad()
    def prepare_condition_cache(
        self,
        instruction: torch.Tensor,
        instruction_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("Condition K/V caching is inference-only.")
        self._validate_conditions(
            instruction,
            instruction_mask,
            state_history,
            state_mask,
        )
        precomputed = self._precompute_inputs(
            instruction,
            instruction_mask,
            state_history,
            state_mask,
        )
        return {
            "layers": [
                {
                    "cross": self._cross_payload(
                        precomputed,
                        layer_idx,
                        cache_kv=True,
                    )
                }
                for layer_idx in range(self.num_layers)
            ],
            "state": {
                key: value.detach() for key, value in precomputed["state"].items()
            },
        }

    @torch.no_grad()
    def prepare_history_cache(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_attention_mask: torch.Tensor,
        condition_cache: dict[str, Any],
    ) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("History K/V caching is inference-only.")
        layers = condition_cache.get("layers")
        state = condition_cache.get("state")
        if not isinstance(layers, list) or len(layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        if not isinstance(state, dict):
            raise ValueError("Condition cache is missing the state stream.")

        video_length = int(video_tokens.shape[1])
        state_length = int(state["tokens"].shape[1])
        full_mask = self._state_full_attention_mask(
            video_attention_mask,
            state["mask"],
            video_length,
            0,
        ).to(video_tokens.device)
        video_freqs = self.video_freqs_on_physical_axis(video_freqs)
        state_tokens = state["tokens"]
        current_video = video_tokens
        history_layers = []

        for layer_idx in range(self.num_layers):
            state_io = self._build_dit_attention_io(
                self.action_expert,
                self.action_expert.blocks[layer_idx],
                state_tokens,
                state["freqs"],
                state["t_mod"],
            )
            video_io = self._build_dit_attention_io(
                self.video_expert,
                self.video_expert.blocks[layer_idx],
                current_video,
                video_freqs,
                video_t_mod,
            )
            query = torch.cat([state_io[0], video_io[0]], dim=1)
            key = torch.cat([state_io[1], video_io[1]], dim=1)
            value = torch.cat([state_io[2], video_io[2]], dim=1)
            mixed = flash_attention(
                q=query,
                k=key,
                v=value,
                num_heads=self.num_heads,
                ctx_mask=full_mask,
            )
            history_layers.append(
                {
                    "state_key": state_io[1].detach(),
                    "state_value": state_io[2].detach(),
                    "video_key": video_io[1].detach(),
                    "video_value": video_io[2].detach(),
                }
            )
            cross = layers[layer_idx]["cross"]
            state_tokens = self._post_block(
                state_io[3],
                mixed[:, :state_length],
                cross["action"],
            )
            current_video = self._post_block(
                video_io[3],
                mixed[:, state_length:],
                cross["video"],
            )
        return {
            "layers": history_layers,
            "video_tokens": current_video.detach(),
            "state_tokens": state_tokens.detach(),
            "video_length": video_length,
            "state_length": state_length,
        }

    def forward_action_with_history_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        dit_attention_mask: torch.Tensor,
        condition_cache: dict[str, Any],
        history_cache: dict[str, Any],
    ) -> torch.Tensor:
        del action_freqs
        if self.training:
            raise RuntimeError("History K/V caching is inference-only.")
        condition_layers = condition_cache.get("layers")
        history_layers = history_cache.get("layers")
        state = condition_cache.get("state")
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        if not isinstance(history_layers, list) or len(history_layers) != self.num_layers:
            raise ValueError("History-cache depth does not match MoT depth.")

        video_length = int(history_cache["video_length"])
        state_length = int(history_cache["state_length"])
        action_length = int(action_tokens.shape[1])
        full_mask = self._state_full_attention_mask(
            dit_attention_mask,
            state["mask"],
            video_length,
            action_length,
        ).to(action_tokens.device)
        action_mask = full_mask[:, :, state_length + video_length :, :]
        action_freqs = self.action_freqs_on_physical_axis(
            action_length,
            action_tokens.device,
        )
        tokens = action_tokens
        for layer_idx in range(self.num_layers):
            action_io = self._build_dit_attention_io(
                self.action_expert,
                self.action_expert.blocks[layer_idx],
                tokens,
                action_freqs,
                action_t_mod,
            )
            cached = history_layers[layer_idx]
            mixed = flash_attention(
                q=action_io[0],
                k=torch.cat(
                    [cached["state_key"], cached["video_key"], action_io[1]],
                    dim=1,
                ),
                v=torch.cat(
                    [cached["state_value"], cached["video_value"], action_io[2]],
                    dim=1,
                ),
                num_heads=self.num_heads,
                ctx_mask=action_mask,
            )
            tokens = self._post_block(
                action_io[3],
                mixed,
                condition_layers[layer_idx]["cross"]["action"],
            )
        return tokens

    def forward(
        self,
        embeds_all: dict[str, torch.Tensor],
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        instruction: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.BoolTensor] = None,
        state_history: Optional[torch.Tensor] = None,
        state_mask: Optional[torch.BoolTensor] = None,
        condition_cache: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        tokens = {name: embeds_all[name] for name in ("video", "action")}
        if condition_cache is None:
            required = {
                "instruction": instruction,
                "instruction_mask": instruction_mask,
                "state_history": state_history,
                "state_mask": state_mask,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"InstructionStateStreamMoT conditions are incomplete: {missing}."
                )
            self._validate_conditions(
                instruction,
                instruction_mask,
                state_history,
                state_mask,
            )
            precomputed = self._precompute_inputs(
                instruction,
                instruction_mask,
                state_history,
                state_mask,
            )
            condition_layers = None
            state = precomputed["state"]
        else:
            precomputed = None
            condition_layers = condition_cache.get("layers")
            state = condition_cache.get("state")
            if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
                raise ValueError("Condition-cache depth does not match MoT depth.")
            if not isinstance(state, dict):
                raise ValueError("Condition cache is missing the state stream.")

        video_length = int(tokens["video"].shape[1])
        action_length = int(tokens["action"].shape[1])
        state_length = int(state["tokens"].shape[1])
        full_mask = self._state_full_attention_mask(
            dit_attention_mask,
            state["mask"],
            video_length,
            action_length,
        ).to(tokens["video"].device)
        video_freqs = self.video_freqs_on_physical_axis(freqs_all["video"])
        action_freqs = self.action_freqs_on_physical_axis(
            action_length,
            tokens["action"].device,
        )
        state_tokens = state["tokens"]

        for layer_idx in range(self.num_layers):
            state_io = self._build_dit_attention_io(
                self.action_expert,
                self.action_expert.blocks[layer_idx],
                state_tokens,
                state["freqs"],
                state["t_mod"],
            )
            video_io = self._build_dit_attention_io(
                self.video_expert,
                self.video_expert.blocks[layer_idx],
                tokens["video"],
                video_freqs,
                t_mod_all["video"],
            )
            action_io = self._build_dit_attention_io(
                self.action_expert,
                self.action_expert.blocks[layer_idx],
                tokens["action"],
                action_freqs,
                t_mod_all["action"],
            )
            query = torch.cat([state_io[0], video_io[0], action_io[0]], dim=1)
            key = torch.cat([state_io[1], video_io[1], action_io[1]], dim=1)
            value = torch.cat([state_io[2], video_io[2], action_io[2]], dim=1)

            def attend(q, k, v, mask):
                return flash_attention(
                    q=q,
                    k=k,
                    v=v,
                    num_heads=self.num_heads,
                    ctx_mask=mask,
                )

            if self.checkpoint_attention and self.training:
                mixed = torch.utils.checkpoint.checkpoint(
                    attend,
                    query,
                    key,
                    value,
                    full_mask,
                    use_reentrant=False,
                )
            else:
                mixed = attend(query, key, value, full_mask)

            cross = (
                self._cross_payload(precomputed, layer_idx, cache_kv=False)
                if condition_layers is None
                else condition_layers[layer_idx]["cross"]
            )
            video_start = state_length
            action_start = state_length + video_length
            state_tokens = self._post_block(
                state_io[3],
                mixed[:, :video_start],
                cross["action"],
            )
            tokens["video"] = self._post_block(
                video_io[3],
                mixed[:, video_start:action_start],
                cross["video"],
            )
            tokens["action"] = self._post_block(
                action_io[3],
                mixed[:, action_start:],
                cross["action"],
            )
        return tokens


__all__ = ["InstructionStateStreamMoT"]
