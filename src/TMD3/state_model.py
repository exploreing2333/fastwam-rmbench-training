from __future__ import annotations

from typing import Any, Optional

import torch

from TMD3.model import ProgressTMD as StreamingProgressTMD
from TMD3.state_progress_mot import StateProgressMoT


class StateConditionedProgressTMD(StreamingProgressTMD):
    """TMD3 with configurable compact Progress and direct state routing."""

    contract_name = "TMD3-StateProgress"

    @staticmethod
    def _normalize_condition_encoder_config(
        condition_encoder_config: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Normalize the two ordinal fields owned by StateProgressMoT.

        The shared Qwen condition normalizer is intentionally strict, so
        subclass-owned fields must be removed before delegating and restored
        in the checkpoint metadata afterwards.
        """
        config = dict(condition_encoder_config or {})
        ordinal_mode = str(
            config.pop("progress_ordinal_context_mode", "none")
        ).strip().lower()
        if ordinal_mode not in {"none", "active_plan_embedding"}:
            raise ValueError(
                "progress_ordinal_context_mode must be 'none' or "
                f"'active_plan_embedding', got {ordinal_mode!r}."
            )
        max_plan_length = int(config.pop("progress_max_plan_length", 32))
        if max_plan_length <= 0:
            raise ValueError("progress_max_plan_length must be positive.")
        normalized = StreamingProgressTMD._normalize_condition_encoder_config(
            config
        )
        normalized.update(
            {
                "progress_ordinal_context_mode": ordinal_mode,
                "progress_max_plan_length": max_plan_length,
            }
        )
        return normalized

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
        return StateProgressMoT(
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

    def _progress_training_fields(self) -> tuple[str, ...]:
        fields = (
            "progress_logits",
            "progress_subtask_tokens",
            "progress_subtask_mask",
            "state_history",
            "state_history_mask",
        )
        if self.mot.progress_ordinal_context_mode != "none":
            fields += (
                "progress_active_idx",
                "progress_plan_length",
            )
        return fields

    def _mot_training_kwargs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        kwargs = {
            "progress_logits": inputs["progress_logits"],
            "subtask_tokens": inputs["progress_subtask_tokens"],
            "subtask_mask": inputs["progress_subtask_mask"],
            "state_history": inputs["state_history"],
            "state_mask": inputs["state_history_mask"],
        }
        if self.mot.progress_ordinal_context_mode != "none":
            kwargs.update(
                {
                    "progress_active_idx": inputs["progress_active_idx"],
                    "progress_plan_length": inputs["progress_plan_length"],
                }
            )
        return kwargs

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
        del context, context_mask, progress_subtask_layers
        required = {
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "progress_logits": progress_memory_layers,
            "progress_subtask_tokens": progress_subtask_tokens,
            "progress_subtask_mask": progress_subtask_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "TMD3-StateProgress inference conditions are incomplete: "
                f"{missing}."
            )
        del (
            progress_memory_mask,
            progress_instruction_layers,
            progress_instruction_mask,
        )
        progress_logits = self._with_batch(progress_memory_layers, 1)
        subtask = self._with_batch(progress_subtask_tokens, 2)
        subtask_mask = self._with_batch(progress_subtask_mask, 1)
        state_history = self._with_batch(proprio, 2)
        state_mask = self._with_batch(proprio_mask, 1)
        active_idx = (
            None
            if progress_active_idx is None
            else self._with_batch(progress_active_idx, 0)
        )
        plan_length = (
            None
            if progress_plan_length is None
            else self._with_batch(progress_plan_length, 0)
        )
        return self.mot.prepare_condition_cache(
            progress_logits.to(self.device, dtype=torch.float32),
            subtask.to(self.device, dtype=self.torch_dtype),
            subtask_mask.to(self.device, dtype=torch.bool),
            state_history.to(self.device, dtype=self.torch_dtype),
            state_mask.to(self.device, dtype=torch.bool),
            (
                None
                if active_idx is None
                else active_idx.to(self.device, dtype=torch.long)
            ),
            (
                None
                if plan_length is None
                else plan_length.to(self.device, dtype=torch.long)
            ),
        )


ProgressTMD = StateConditionedProgressTMD


__all__ = ["ProgressTMD", "StateConditionedProgressTMD"]
