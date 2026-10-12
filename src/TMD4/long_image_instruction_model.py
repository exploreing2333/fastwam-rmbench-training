from __future__ import annotations

from typing import Any, Optional

import torch

from TMD3.instruction_model import InstructionTMD
from TMD4.long_image_instruction_mot import LongImageInstructionStateMoT
from TMD4.long_image_model import LongImageProgressTMD


class LongImageInstructionTMD(LongImageProgressTMD, InstructionTMD):
    """Progress-free TMD4 conditioned by one complete static instruction."""

    contract_name = "TMD4-LongImage-T5Instruction-NoProgress"

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
        return LongImageInstructionStateMoT(
            video_expert=video_expert,
            action_expert=action_expert,
            condition_dim=int(condition_dim),
            state_dim=int(config.get("state_dim", 14)),
            state_history_tokens=int(config.get("state_history_tokens", 1)),
            state_time_start=float(config.get("state_time_start", 1.0)),
            state_time_end=float(config.get("state_time_end", 1.0)),
            state_time_gate_init=float(
                config.get("state_time_gate_init", -3.0)
            ),
            state_temporal_encoding=bool(
                config.get("state_temporal_encoding", False)
            ),
            state_cross_attention_scope=str(
                config.get("state_cross_attention_scope", "action_only")
            ),
            video_time_offset=float(config.get("video_time_offset", 1.0)),
            action_time_start=float(config.get("action_time_start", 4.0)),
            action_time_end=float(config.get("action_time_end", 5.0)),
            checkpoint_attention=bool(checkpoint_attention),
        )

    def _progress_training_fields(self) -> tuple[str, ...]:
        return (
            "context",
            "context_mask",
            "state_history",
            "state_history_mask",
        )

    def _mot_training_kwargs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "instruction": inputs["context"],
            "instruction_mask": inputs["context_mask"],
            "state_history": inputs["state_history"],
            "state_mask": inputs["state_history_mask"],
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
        return InstructionTMD._prepare_inference_condition_cache(
            self,
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


__all__ = ["LongImageInstructionTMD"]
