from __future__ import annotations

from typing import Any, Optional

import torch

from TMD3.instruction_mot import InstructionStateMoT
from TMD3.model import ProgressTMD as StreamingProgressTMD


class InstructionTMD(StreamingProgressTMD):
    """TMD3 conditioned by cached full T5 instruction and configured state."""

    contract_name = "TMD3-T5Instruction-NoProgress"

    @staticmethod
    def _trim_instruction_padding(
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop only the right-padding suffix shared by a batch."""
        if context.ndim not in (2, 3) or context_mask.ndim != context.ndim - 1:
            raise ValueError(
                "Instruction context/mask must be [L,D]/[L] or "
                f"[B,L,D]/[B,L], got {tuple(context.shape)} and "
                f"{tuple(context_mask.shape)}."
            )
        active_columns = (
            context_mask.bool()
            if context_mask.ndim == 1
            else context_mask.bool().any(dim=0)
        )
        active_indices = torch.nonzero(active_columns, as_tuple=False)
        if int(active_indices.numel()) == 0:
            raise ValueError("T5 instruction cannot be entirely padding.")
        keep_length = int(active_indices[-1].item()) + 1
        return context[..., :keep_length, :], context_mask[..., :keep_length]

    def build_inputs(self, sample, tiled: bool = False):
        # Trim on CPU before the parent transfers the 4096-wide context to GPU.
        # This preserves every tokenizer-valid token and only removes the
        # shared all-padding suffix.
        if "context" not in sample or "context_mask" not in sample:
            return super().build_inputs(sample, tiled=tiled)
        trimmed_sample = dict(sample)
        (
            trimmed_sample["context"],
            trimmed_sample["context_mask"],
        ) = self._trim_instruction_padding(
            sample["context"],
            sample["context_mask"],
        )
        return super().build_inputs(trimmed_sample, tiled=tiled)

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
        return InstructionStateMoT(
            video_expert=video_expert,
            action_expert=action_expert,
            condition_dim=int(condition_dim),
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
            state_cross_attention_scope=str(
                config.get("state_cross_attention_scope", "video_action")
            ),
            video_time_offset=float(config.get("video_time_offset", 1.0)),
            action_time_start=float(config.get("action_time_start", 4.0)),
            action_time_end=float(config.get("action_time_end", 5.0)),
            checkpoint_attention=bool(checkpoint_attention),
        )

    def _progress_training_fields(self) -> tuple[str, ...]:
        # The parent hook name is retained for checkpoint compatibility.  None
        # of these tensors comes from the Progress Expert.
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

    @staticmethod
    def _with_batch(tensor: torch.Tensor, unbatched_ndim: int) -> torch.Tensor:
        return tensor.unsqueeze(0) if tensor.ndim == unbatched_ndim else tensor

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
            progress_memory_layers,
            progress_memory_mask,
            progress_instruction_layers,
            progress_instruction_mask,
            progress_subtask_tokens,
            progress_subtask_layers,
            progress_subtask_mask,
            progress_active_idx,
            progress_plan_length,
        )
        required = {
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "proprio_mask": proprio_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "TMD3 T5-instruction inference conditions are incomplete: "
                f"{missing}."
            )
        context, context_mask = self._trim_instruction_padding(
            context,
            context_mask,
        )
        instruction = self._with_batch(context, 2)
        instruction_mask = self._with_batch(context_mask, 1)
        state_history = self._with_batch(proprio, 2)
        state_mask = self._with_batch(proprio_mask, 1)
        return self.mot.prepare_condition_cache(
            instruction.to(self.device, dtype=self.torch_dtype),
            instruction_mask.to(self.device, dtype=torch.bool),
            state_history.to(self.device, dtype=self.torch_dtype),
            state_mask.to(self.device, dtype=torch.bool),
        )

    @classmethod
    def from_wan22_pretrained(cls, *args, **kwargs):
        model = super().from_wan22_pretrained(*args, **kwargs)
        model.model_paths["text_encoder"] = "PRECOMPUTED_T5_CACHE"
        model.model_paths["tokenizer"] = "PRECOMPUTED_T5_CACHE"
        model.model_paths["progress_model"] = "DISABLED_ABLATION"
        return model


__all__ = ["InstructionTMD"]
