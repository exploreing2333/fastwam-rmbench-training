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


logger = get_logger(__name__)


def continuous_rope_1d(
    positions: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Return LingBot-style continuous one-dimensional RoPE frequencies."""
    if positions.ndim != 1:
        raise ValueError(
            f"Continuous RoPE positions must be one-dimensional, got {tuple(positions.shape)}."
        )
    if int(head_dim) <= 0 or int(head_dim) % 2:
        raise ValueError(f"RoPE head_dim must be positive and even, got {head_dim}.")
    exponent = torch.arange(
        0,
        int(head_dim),
        2,
        device=positions.device,
        dtype=torch.float64,
    ) / float(head_dim)
    inverse_frequency = 1.0 / (float(theta) ** exponent)
    angles = positions.to(torch.float64)[:, None] * inverse_frequency[None]
    return torch.polar(torch.ones_like(angles), angles).unsqueeze(1)


class StateProgressMoT(ProgressMoT):
    """TMD3 MoT with explicit expert-specific Progress/State routing.

    The legacy mode retains compact Progress K/V in full attention.  The
    cross-attention mode routes one compressed progress-logit token only to
    Video, routes decoded subtask tokens to both experts, and can route a
    direct projected current state only to Action.  Neither mode constructs an
    autoregressive state stream.
    """

    def __init__(
        self,
        video_expert: nn.Module,
        action_expert: nn.Module,
        progress_hidden_dim: int = 2560,
        condition_dim: int = 4096,
        progress_bins: int = 10,
        progress_bottleneck_dim: int = 256,
        progress_gate_init: float = -3.0,
        state_dim: int = 14,
        state_history_tokens: int = 4,
        state_time_start: float = 1.0,
        state_time_end: float = 4.0,
        state_time_gate_init: float = -3.0,
        state_temporal_encoding: bool = True,
        video_time_offset: float = 1.0,
        action_time_start: float = 4.0,
        action_time_end: float = 5.0,
        progress_injection_mode: str = "full_attention_kv",
        state_cross_attention_scope: str = "video_action",
        subtask_condition_mode: str = "progress_hidden",
        progress_ordinal_context_mode: str = "none",
        progress_max_plan_length: int = 32,
        checkpoint_attention: bool = False,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            progress_hidden_dim=progress_hidden_dim,
            condition_dim=condition_dim,
            checkpoint_attention=checkpoint_attention,
        )
        if int(progress_bins) <= 1:
            raise ValueError("progress_bins must be greater than one.")
        if int(progress_bottleneck_dim) <= 0:
            raise ValueError("progress_bottleneck_dim must be positive.")
        if int(state_dim) <= 0:
            raise ValueError("state_dim must be positive.")
        if int(state_history_tokens) <= 0:
            raise ValueError("state_history_tokens must be positive.")
        if float(state_time_end) <= float(state_time_start):
            raise ValueError("state_time_end must be greater than state_time_start.")
        if float(action_time_end) <= float(action_time_start):
            raise ValueError("action_time_end must be greater than action_time_start.")

        self.progress_bins = int(progress_bins)
        self.progress_feature_dim = self.progress_bins + 2
        self.state_dim = int(state_dim)
        self.state_history_tokens = int(state_history_tokens)
        self.state_time_start = float(state_time_start)
        self.state_time_end = float(state_time_end)
        self.state_temporal_encoding = bool(state_temporal_encoding)
        self.video_time_offset = float(video_time_offset)
        self.action_time_start = float(action_time_start)
        self.action_time_end = float(action_time_end)
        self.progress_injection_mode = str(progress_injection_mode)
        self.state_cross_attention_scope = str(state_cross_attention_scope)
        self.subtask_condition_mode = str(subtask_condition_mode)
        self.progress_ordinal_context_mode = str(
            progress_ordinal_context_mode
        ).lower()
        self.progress_max_plan_length = int(progress_max_plan_length)
        if self.progress_injection_mode not in {
            "full_attention_kv",
            "cross_attention_video_only",
            "disabled",
        }:
            raise ValueError(
                "progress_injection_mode must be 'full_attention_kv', "
                "'cross_attention_video_only', or 'disabled', "
                f"got {self.progress_injection_mode!r}."
            )
        if self.state_cross_attention_scope not in {
            "video_action",
            "action_only",
        }:
            raise ValueError(
                "state_cross_attention_scope must be 'video_action' or "
                f"'action_only', got {self.state_cross_attention_scope!r}."
            )
        if self.subtask_condition_mode not in {
            "progress_hidden",
            "t5_text_context",
        }:
            raise ValueError(
                "subtask_condition_mode must be 'progress_hidden' or "
                f"'t5_text_context', got {self.subtask_condition_mode!r}."
            )
        if self.progress_ordinal_context_mode not in {
            "none",
            "active_plan_embedding",
        }:
            raise ValueError(
                "progress_ordinal_context_mode must be 'none' or "
                f"'active_plan_embedding', got {self.progress_ordinal_context_mode!r}."
            )
        if self.progress_max_plan_length <= 0:
            raise ValueError("progress_max_plan_length must be positive.")
        if self.progress_ordinal_context_mode == "active_plan_embedding":
            # Index zero of active_embedding represents bootstrap active_idx=-1;
            # plan terminal K is represented without clipping. All parameter
            # extents are 4096-aligned, avoiding the old scalar-BF16 ZeRO
            # flattening misalignment.
            self.progress_active_embedding = nn.Embedding(
                self.progress_max_plan_length + 2,
                self.condition_dim,
            )
            self.progress_plan_length_embedding = nn.Embedding(
                self.progress_max_plan_length + 1,
                self.condition_dim,
            )
            self.progress_ordinal_norm = nn.LayerNorm(self.condition_dim)
        if self.subtask_condition_mode == "t5_text_context":
            # The decoded subtask has already been converted by the same Wan
            # T5 encoder used by the H5 instruction path.  It is therefore
            # already 4096-wide and must not pass through the unrelated
            # Progress-hidden (2560 -> 4096) adapter.
            del self.subtask_adapter

        if self.progress_injection_mode == "cross_attention_video_only":
            # Match the direct state-conditioning contract: the bounded C51
            # probability vector is projected once into one 4096-wide context
            # token.  Do not register a scalar gate here: besides being
            # unnecessary for this direct route, a one-element BF16 parameter
            # shifts every following ZeRO-1 flattened parameter by two bytes
            # and de-aligns the large Video/Action GEMM weights.
            del self.progress_k
            del self.progress_v
            self.progress_projection = nn.Linear(
                self.progress_bins,
                self.condition_dim,
            )
        elif self.progress_injection_mode == "disabled":
            # The Progress expert is still allowed to supply decoded subtask
            # text, but its numeric C51 logits have no registered projection or
            # attention route into either expert.
            del self.progress_k
            del self.progress_v
        else:
            # Retain the original full-attention compatibility path.  New
            # cross-attention experiments do not instantiate these modules.
            self.progress_encoder = nn.Sequential(
                nn.Linear(self.progress_feature_dim, int(progress_bottleneck_dim)),
                nn.SiLU(),
                nn.Linear(int(progress_bottleneck_dim), self.progress_hidden_dim),
                nn.LayerNorm(self.progress_hidden_dim),
            )
            self.progress_gate = nn.Parameter(
                torch.tensor(float(progress_gate_init), dtype=torch.float32)
            )

        # FastWAM-style direct proprio projection into cross-attention context.
        # Multi-state legacy configurations add fixed slot identity; the
        # current-state path disables temporal encoding and uses Linear(qpos[t]).
        self.state_encoder = nn.Linear(self.state_dim, self.condition_dim)
        self.state_time_gate = nn.Parameter(
            torch.tensor(float(state_time_gate_init), dtype=torch.float32)
        )
        if not self.state_temporal_encoding:
            del self.state_time_gate

        logger.info(
            "Initialized StateProgressMoT progress_mode=%s "
            "progress_features=%d projection=%s cross=decoded_subtask "
            "state_scope=%s state=%dx%d state_temporal=%s state_time=[%.2f,%.2f] "
            "video_offset=%.2f action_time=[%.2f,%.2f] "
            "subtask_condition=%s ordinal_context=%s max_plan=%d instruction=disabled",
            self.progress_injection_mode,
            self.progress_feature_dim,
            (
                "softmax_linear_10_to_condition"
                if self.progress_injection_mode == "cross_attention_video_only"
                else "disabled"
                if self.progress_injection_mode == "disabled"
                else f"legacy_gated_hidden_init_{float(progress_gate_init):.2f}"
            ),
            self.state_cross_attention_scope,
            self.state_history_tokens,
            self.state_dim,
            self.state_temporal_encoding,
            self.state_time_start,
            self.state_time_end,
            self.video_time_offset,
            self.action_time_start,
            self.action_time_end,
            self.subtask_condition_mode,
            self.progress_ordinal_context_mode,
            self.progress_max_plan_length,
        )

    @staticmethod
    def _validate_mask(mask: torch.Tensor, tokens: torch.Tensor, name: str) -> None:
        if mask.ndim != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError(
                f"{name} mask must match [B,S]={tuple(tokens.shape[:2])}, "
                f"got {tuple(mask.shape)}."
            )

    def _progress_tokens(self, progress_logits: torch.Tensor) -> torch.Tensor:
        if progress_logits.ndim == 3 and int(progress_logits.shape[1]) == 1:
            progress_logits = progress_logits[:, 0]
        if progress_logits.ndim != 2 or int(progress_logits.shape[-1]) != self.progress_bins:
            raise ValueError(
                "Progress logits must be [B,10] (or [B,1,10]), got "
                f"{tuple(progress_logits.shape)}."
            )
        probabilities = torch.softmax(progress_logits.float(), dim=-1)
        if self.progress_injection_mode == "cross_attention_video_only":
            projection_dtype = self.progress_projection.weight.dtype
            return self.progress_projection(
                probabilities.to(dtype=projection_dtype)
            ).unsqueeze(1)

        centers = torch.linspace(
            0.0,
            1.0,
            self.progress_bins,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        mean = (probabilities * centers).sum(dim=-1, keepdim=True)
        entropy = -(
            probabilities * probabilities.clamp_min(1.0e-8).log()
        ).sum(dim=-1, keepdim=True)
        features = torch.cat([probabilities, mean, entropy], dim=-1)
        encoder_dtype = self.progress_encoder[0].weight.dtype
        encoded = self.progress_encoder(features.to(dtype=encoder_dtype))
        gate = torch.sigmoid(self.progress_gate).to(encoded)
        return (encoded * gate).unsqueeze(1)

    def _state_context(self, state_history: torch.Tensor) -> torch.Tensor:
        if state_history.ndim != 3:
            raise ValueError(
                "State history must be [B,S,D], got "
                f"{tuple(state_history.shape)}."
            )
        if tuple(state_history.shape[1:]) != (
            self.state_history_tokens,
            self.state_dim,
        ):
            raise ValueError(
                "State history shape does not match the configured contract: "
                f"expected=(*,{self.state_history_tokens},{self.state_dim}), "
                f"got={tuple(state_history.shape)}."
            )
        encoded = self.state_encoder(state_history)
        if not self.state_temporal_encoding:
            return encoded
        positions = torch.linspace(
            self.state_time_start,
            self.state_time_end,
            self.state_history_tokens,
            device=encoded.device,
            dtype=torch.float32,
        )
        temporal = sinusoidal_embedding_1d(
            self.condition_dim,
            positions,
        ).to(encoded)
        temporal_gate = torch.sigmoid(self.state_time_gate).to(encoded)
        return encoded + temporal_gate * temporal.unsqueeze(0)

    def _ordinal_context(
        self,
        active_idx: Optional[torch.Tensor],
        plan_length: Optional[torch.Tensor],
        *,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if self.progress_ordinal_context_mode == "none":
            return None
        if active_idx is None or plan_length is None:
            raise ValueError(
                "Active-plan ordinal context requires progress_active_idx and "
                "progress_plan_length."
            )
        active_idx = active_idx.reshape(-1).to(dtype=torch.long)
        plan_length = plan_length.reshape(-1).to(dtype=torch.long)
        if int(active_idx.numel()) != int(batch_size) or int(
            plan_length.numel()
        ) != int(batch_size):
            raise ValueError(
                "Ordinal context must contain one active_idx/plan_length pair "
                f"per sample, got {tuple(active_idx.shape)} and "
                f"{tuple(plan_length.shape)} for B={batch_size}."
            )
        invalid = (
            (plan_length < 0)
            | (plan_length > self.progress_max_plan_length)
            | (active_idx < -1)
            | (active_idx > plan_length)
        )
        if bool(invalid.any().item()):
            bad = torch.nonzero(invalid, as_tuple=False)[0].item()
            raise ValueError(
                "Invalid planner ordinal context: "
                f"active_idx={int(active_idx[bad])}, "
                f"plan_length={int(plan_length[bad])}, "
                f"max={self.progress_max_plan_length}."
            )
        active_idx = active_idx.to(self.progress_active_embedding.weight.device)
        plan_length = plan_length.to(
            self.progress_plan_length_embedding.weight.device
        )
        token = self.progress_active_embedding(active_idx + 1)
        token = token + self.progress_plan_length_embedding(plan_length)
        return self.progress_ordinal_norm(token).unsqueeze(1)

    def video_freqs_on_physical_axis(self, freqs: torch.Tensor) -> torch.Tensor:
        """Shift H1..H4 from model indices 0..3 to physical endpoints 1..4."""
        if self.video_time_offset == 0.0:
            return freqs
        base_freqs = self.video_expert.base.freqs[0]
        temporal_width = int(base_freqs.shape[-1])
        if freqs.shape[-1] < temporal_width:
            raise ValueError("Video RoPE has fewer entries than its temporal component.")
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
            raise ValueError(
                f"DiT attention mask must be [Q,K] or [B,1,Q,K], got {tuple(dit_mask.shape)}."
            )
        if tuple(right.shape[-2:]) != (int(dit_length), int(dit_length)):
            raise ValueError(
                "DiT attention mask must cover Video+Action exactly: "
                f"mask={tuple(right.shape)}, tokens={dit_length}."
            )
        return right.bool()

    @classmethod
    def _kv_only_attention_mask(
        cls,
        dit_mask: torch.Tensor,
        progress_mask: torch.BoolTensor,
        dit_length: int,
    ) -> torch.BoolTensor:
        """Build rectangular Q=DiT, K=[Progress,DiT] without progress queries."""
        if progress_mask.ndim != 2 or int(progress_mask.shape[1]) != 1:
            raise ValueError("Compressed progress mask must be [B,1].")
        batch_size = int(progress_mask.shape[0])
        right = cls._right_dit_mask(dit_mask, batch_size, dit_length)
        heads = int(right.shape[1])
        full = torch.zeros(
            batch_size,
            heads,
            dit_length,
            1 + dit_length,
            device=right.device,
            dtype=torch.bool,
        )
        full[:, :, :, :1] = progress_mask[:, None, None, :]
        full[:, :, :, 1:] = right
        return full

    def _conditioned_dit_mask(
        self,
        dit_mask: torch.Tensor,
        progress_mask: Optional[torch.BoolTensor],
        batch_size: int,
        dit_length: int,
    ) -> torch.BoolTensor:
        if self.progress_injection_mode == "full_attention_kv":
            if progress_mask is None:
                raise ValueError("Full-attention progress mode requires progress_mask.")
            return self._kv_only_attention_mask(
                dit_mask,
                progress_mask,
                dit_length,
            )
        return self._right_dit_mask(dit_mask, batch_size, dit_length)

    def _conditioned_kv(
        self,
        progress_key: Optional[torch.Tensor],
        progress_value: Optional[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.progress_injection_mode == "full_attention_kv":
            if progress_key is None or progress_value is None:
                raise ValueError("Full-attention progress mode requires progress K/V.")
            return (
                torch.cat([progress_key, *keys], dim=1),
                torch.cat([progress_value, *values], dim=1),
            )
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def _validate_conditions(
        self,
        progress_logits: torch.Tensor,
        subtask_tokens: torch.Tensor,
        subtask_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
        progress_active_idx: Optional[torch.Tensor] = None,
        progress_plan_length: Optional[torch.Tensor] = None,
    ) -> None:
        if self.progress_injection_mode != "disabled":
            if progress_logits.ndim not in (2, 3):
                raise ValueError("Progress logits must be [B,10] or [B,1,10].")
            batch_size = int(progress_logits.shape[0])
        else:
            batch_size = int(subtask_tokens.shape[0])
        if subtask_tokens.ndim != 3:
            raise ValueError(
                "subtask_tokens must be decoded final hidden [B,S,H], "
                f"got {tuple(subtask_tokens.shape)}."
            )
        expected_subtask_width = (
            self.condition_dim
            if self.subtask_condition_mode == "t5_text_context"
            else self.progress_hidden_dim
        )
        if int(subtask_tokens.shape[-1]) != expected_subtask_width:
            raise ValueError(
                f"Subtask condition width must be {expected_subtask_width} "
                f"for mode={self.subtask_condition_mode!r}."
            )
        if int(subtask_tokens.shape[0]) != batch_size:
            raise ValueError("Progress/subtask batch mismatch.")
        self._validate_mask(subtask_mask, subtask_tokens, "subtask")
        self._validate_mask(state_mask, state_history, "state")
        if not bool(subtask_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain generated-subtask context.")
        if not bool(state_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least its current state.")
        # Also validates the exact bin count when numeric Progress is active.
        if self.progress_injection_mode != "disabled":
            self._progress_tokens(progress_logits)
        self._state_context(state_history)
        self._ordinal_context(
            progress_active_idx,
            progress_plan_length,
            batch_size=batch_size,
        )

    def _precompute_condition_inputs(
        self,
        progress_logits: torch.Tensor,
        subtask_tokens: torch.Tensor,
        subtask_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
        progress_active_idx: Optional[torch.Tensor] = None,
        progress_plan_length: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        batch_size = int(subtask_tokens.shape[0])
        if self.progress_injection_mode == "disabled":
            progress_tokens = None
            progress_key = None
            progress_value = None
            progress_context = None
            progress_mask = None
        else:
            progress_tokens = self._progress_tokens(progress_logits)

        if self.progress_injection_mode == "full_attention_kv":
            progress_key, progress_value = self._project_memory(progress_tokens)
            progress_context = None
            progress_mask = torch.ones(
                (int(progress_logits.shape[0]), 1),
                device=progress_logits.device,
                dtype=torch.bool,
            )
        elif self.progress_injection_mode == "cross_attention_video_only":
            progress_key = None
            progress_value = None
            # `_progress_tokens` already returns the direct 4096-wide
            # probability projection in cross-attention mode.
            progress_context = progress_tokens
            progress_mask = None

        subtask = (
            subtask_tokens
            if self.subtask_condition_mode == "t5_text_context"
            else self.subtask_adapter(subtask_tokens)
        )
        states = self._state_context(state_history)
        ordinal = self._ordinal_context(
            progress_active_idx,
            progress_plan_length,
            batch_size=batch_size,
        )
        contexts = {}
        context_masks = {}
        for name in ("video", "action"):
            pieces = [subtask]
            masks = [subtask_mask]
            if ordinal is not None:
                pieces.append(ordinal)
                masks.append(
                    torch.ones(
                        (batch_size, 1),
                        device=subtask_tokens.device,
                        dtype=torch.bool,
                    )
                )
            if (
                self.progress_injection_mode == "cross_attention_video_only"
                and name == "video"
            ):
                pieces.append(progress_context)
                masks.append(
                    torch.ones(
                        (batch_size, 1),
                        device=subtask_tokens.device,
                        dtype=torch.bool,
                    )
                )
            if self.state_cross_attention_scope == "video_action" or name == "action":
                pieces.append(states)
                masks.append(state_mask)
            # The decoded-subtask/progress/state condition is one shared final
            # context, not a different tensor for every transformer layer.
            # Project it through each expert's shared text embedding once and
            # reuse the result in all layer-specific cross-attention blocks.
            adapted = torch.cat(pieces, dim=1)
            adapted_mask = torch.cat(masks, dim=1)
            contexts[name] = self.mixtures[name].text_embedding(adapted)
            context_masks[name] = adapted_mask[:, None, None, :]
        return {
            "progress_key": progress_key,
            "progress_value": progress_value,
            "progress_mask": progress_mask,
            "contexts": contexts,
            "context_masks": context_masks,
        }

    def _layerwise_cross_payload(
        self,
        precomputed: dict[str, Any],
        layer_idx: int,
        *,
        cache_kv: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        payloads: dict[str, dict[str, torch.Tensor]] = {}
        for name in ("video", "action"):
            # `context` is shared across depth.  Only the cross-attention K/V
            # projections below are layer-specific.
            context = precomputed["contexts"][name]
            mask = precomputed["context_masks"][name]
            if cache_kv:
                cross = self.mixtures[name].blocks[layer_idx].cross_attn
                payloads[name] = {
                    "key": cross.norm_k(cross.k(context)).detach(),
                    "value": cross.v(context).detach(),
                    "mask": mask.detach(),
                }
            else:
                payloads[name] = {"context": context, "mask": mask}
        return payloads

    @torch.no_grad()
    def prepare_condition_cache(
        self,
        progress_logits: torch.Tensor,
        subtask_tokens: torch.Tensor,
        subtask_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
        progress_active_idx: Optional[torch.Tensor] = None,
        progress_plan_length: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("Condition K/V caching is inference-only.")
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
        layers = [
            {
                "cross": self._layerwise_cross_payload(
                    precomputed,
                    layer_idx,
                    cache_kv=True,
                )
            }
            for layer_idx in range(self.num_layers)
        ]
        cache = {"layers": layers}
        for key in ("progress_key", "progress_value", "progress_mask"):
            value = precomputed[key]
            if value is not None:
                cache[key] = value.detach()
        return cache

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
        condition_layers = condition_cache.get("layers")
        progress_mask = condition_cache.get("progress_mask")
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        video_length = int(video_tokens.shape[1])
        mask = self._conditioned_dit_mask(
            video_attention_mask,
            progress_mask,
            int(video_tokens.shape[0]),
            video_length,
        ).to(video_tokens.device)
        tokens = video_tokens
        shifted_freqs = self.video_freqs_on_physical_axis(video_freqs)
        layers = []
        for layer_idx in range(self.num_layers):
            query, key, value, post = self._build_dit_attention_io(
                self.video_expert,
                self.video_expert.blocks[layer_idx],
                tokens,
                shifted_freqs,
                video_t_mod,
            )
            conditioned_key, conditioned_value = self._conditioned_kv(
                condition_cache.get("progress_key"),
                condition_cache.get("progress_value"),
                [key],
                [value],
            )
            mixed = flash_attention(
                q=query,
                k=conditioned_key,
                v=conditioned_value,
                num_heads=self.num_heads,
                ctx_mask=mask,
            )
            layers.append({"video_key": key.detach(), "video_value": value.detach()})
            tokens = self._post_block(
                post,
                mixed,
                condition_layers[layer_idx]["cross"]["video"],
            )
        return {
            "layers": layers,
            "video_tokens": tokens.detach(),
            "video_length": video_length,
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
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        if not isinstance(history_layers, list) or len(history_layers) != self.num_layers:
            raise ValueError("History-cache depth does not match MoT depth.")
        video_length = int(history_cache["video_length"])
        action_length = int(action_tokens.shape[1])
        dit_length = video_length + action_length
        full_mask = self._conditioned_dit_mask(
            dit_attention_mask,
            condition_cache.get("progress_mask"),
            int(action_tokens.shape[0]),
            dit_length,
        ).to(action_tokens.device)
        action_mask = full_mask[:, :, video_length:, :]
        physical_action_freqs = self.action_freqs_on_physical_axis(
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
            conditioned_key, conditioned_value = self._conditioned_kv(
                condition_cache.get("progress_key"),
                condition_cache.get("progress_value"),
                [history_layers[layer_idx]["video_key"], key],
                [history_layers[layer_idx]["video_value"], value],
            )
            mixed = flash_attention(
                q=query,
                k=conditioned_key,
                v=conditioned_value,
                num_heads=self.num_heads,
                ctx_mask=action_mask,
            )
            tokens = self._post_block(
                post,
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
        progress_logits: Optional[torch.Tensor] = None,
        subtask_tokens: Optional[torch.Tensor] = None,
        subtask_mask: Optional[torch.BoolTensor] = None,
        state_history: Optional[torch.Tensor] = None,
        state_mask: Optional[torch.BoolTensor] = None,
        progress_active_idx: Optional[torch.Tensor] = None,
        progress_plan_length: Optional[torch.Tensor] = None,
        condition_cache: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        tokens = {name: embeds_all[name] for name in ("video", "action")}
        if condition_cache is None:
            required = {
                "progress_logits": progress_logits,
                "subtask_tokens": subtask_tokens,
                "subtask_mask": subtask_mask,
                "state_history": state_history,
                "state_mask": state_mask,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"StateProgressMoT conditions are incomplete: {missing}.")
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
            cached_layers = None
            progress_key = precomputed["progress_key"]
            progress_value = precomputed["progress_value"]
            progress_mask = precomputed["progress_mask"]
        else:
            precomputed = None
            cached_layers = condition_cache.get("layers")
            progress_key = condition_cache.get("progress_key")
            progress_value = condition_cache.get("progress_value")
            progress_mask = condition_cache.get("progress_mask")
            if not isinstance(cached_layers, list) or len(cached_layers) != self.num_layers:
                raise ValueError("Condition-cache depth does not match MoT depth.")

        if self.progress_injection_mode in {
            "cross_attention_video_only",
            "disabled",
        }:
            if cached_layers is None:
                # Final decoded-subtask/progress/state contexts are shared over
                # depth exactly like the H5 instruction context.  Per-layer
                # cross-attention modules still project their own K/V.
                shared_cross = self._layerwise_cross_payload(
                    precomputed,
                    0,
                    cache_kv=False,
                )
                cross_layers = [shared_cross] * self.num_layers
            else:
                cross_layers = [layer["cross"] for layer in cached_layers]
            return self._forward_h5_cross_core(
                tokens=tokens,
                dit_attention_mask=dit_attention_mask,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                cross_layers=cross_layers,
            )

        video_length = int(tokens["video"].shape[1])
        action_length = int(tokens["action"].shape[1])
        dit_length = video_length + action_length
        full_mask = self._conditioned_dit_mask(
            dit_attention_mask,
            progress_mask,
            int(tokens["video"].shape[0]),
            dit_length,
        ).to(tokens["video"].device)
        video_freqs = self.video_freqs_on_physical_axis(freqs_all["video"])
        action_freqs = self.action_freqs_on_physical_axis(
            action_length,
            tokens["action"].device,
        )

        for layer_idx in range(self.num_layers):
            queries = []
            keys = []
            values = []
            post = {}
            lengths = {}
            for name, freqs in (("video", video_freqs), ("action", action_freqs)):
                expert = self.mixtures[name]
                query, key, value, cache = self._build_dit_attention_io(
                    expert,
                    expert.blocks[layer_idx],
                    tokens[name],
                    freqs,
                    t_mod_all[name],
                )
                queries.append(query)
                keys.append(key)
                values.append(value)
                post[name] = cache
                lengths[name] = int(query.shape[1])

            cross = (
                self._layerwise_cross_payload(
                    precomputed,
                    layer_idx,
                    cache_kv=False,
                )
                if cached_layers is None
                else cached_layers[layer_idx]["cross"]
            )
            query_cat = torch.cat(queries, dim=1)
            key_cat, value_cat = self._conditioned_kv(
                progress_key,
                progress_value,
                keys,
                values,
            )

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
                    query_cat,
                    key_cat,
                    value_cat,
                    full_mask,
                    use_reentrant=False,
                )
            else:
                mixed = attend(query_cat, key_cat, value_cat, full_mask)

            video_end = lengths["video"]
            tokens["video"] = self._post_block(
                post["video"],
                mixed[:, :video_end],
                cross["video"],
            )
            tokens["action"] = self._post_block(
                post["action"],
                mixed[:, video_end:],
                cross["action"],
            )
        return tokens

    def _forward_h5_cross_core(
        self,
        *,
        tokens: dict[str, torch.Tensor],
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        cross_layers: list[dict[str, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """Run the exact H5 Video/Action MoT core with supplied contexts."""
        if len(cross_layers) != self.num_layers:
            raise ValueError("Cross-attention context depth does not match MoT depth.")
        video_length = int(tokens["video"].shape[1])
        action_length = int(tokens["action"].shape[1])
        dit_length = video_length + action_length
        full_mask = self._right_dit_mask(
            dit_attention_mask,
            int(tokens["video"].shape[0]),
            dit_length,
        ).to(tokens["video"].device)
        video_freqs = self.video_freqs_on_physical_axis(freqs_all["video"])
        action_freqs = self.action_freqs_on_physical_axis(
            action_length,
            tokens["action"].device,
        )

        for layer_idx in range(self.num_layers):
            queries = []
            keys = []
            values = []
            post = {}
            lengths = {}
            for name, freqs in (("video", video_freqs), ("action", action_freqs)):
                expert = self.mixtures[name]
                query, key, value, cache = self._build_dit_attention_io(
                    expert,
                    expert.blocks[layer_idx],
                    tokens[name],
                    freqs,
                    t_mod_all[name],
                )
                queries.append(query)
                keys.append(key)
                values.append(value)
                post[name] = cache
                lengths[name] = int(query.shape[1])

            query_cat = torch.cat(queries, dim=1)
            key_cat = torch.cat(keys, dim=1)
            value_cat = torch.cat(values, dim=1)

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
                    query_cat,
                    key_cat,
                    value_cat,
                    full_mask,
                    use_reentrant=False,
                )
            else:
                mixed = attend(query_cat, key_cat, value_cat, full_mask)

            video_end = lengths["video"]
            cross = cross_layers[layer_idx]
            tokens["video"] = self._post_block(
                post["video"],
                mixed[:, :video_end],
                cross["video"],
            )
            tokens["action"] = self._post_block(
                post["action"],
                mixed[:, video_end:],
                cross["action"],
            )
        return tokens


__all__ = ["StateProgressMoT", "continuous_rope_1d"]
