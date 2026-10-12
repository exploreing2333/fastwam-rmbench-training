from __future__ import annotations

from typing import Any, Optional

from TMD3.instruction_model import InstructionTMD
from TMD3.state_stream_mot import InstructionStateStreamMoT


class InstructionStateStreamTMD(InstructionTMD):
    """TMD3 with T5 cross-attention and a dense causal qpos stream."""

    contract_name = "TMD3-T5Instruction-State64FullAttention"

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
        del progress_hidden_dim
        config = dict(condition_encoder_config or {})
        return InstructionStateStreamMoT(
            video_expert=video_expert,
            action_expert=action_expert,
            condition_dim=int(condition_dim),
            state_dim=int(config.get("state_dim", 14)),
            state_history_tokens=int(config.get("state_history_tokens", 64)),
            state_history_latents=int(config.get("state_history_latents", 4)),
            state_steps_per_latent=int(config.get("state_steps_per_latent", 16)),
            state_time_start=float(config.get("state_time_start", 0.0)),
            state_time_end=float(config.get("state_time_end", 4.0)),
            video_time_offset=float(config.get("video_time_offset", 1.0)),
            action_time_start=float(config.get("action_time_start", 4.0)),
            action_time_end=float(config.get("action_time_end", 5.0)),
            video_tokens_per_frame=int(config.get("video_tokens_per_frame", 120)),
            checkpoint_attention=bool(checkpoint_attention),
        )


__all__ = ["InstructionStateStreamTMD"]
