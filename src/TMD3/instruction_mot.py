from __future__ import annotations

from typing import Any, Optional

import torch

from fastwam.models.wan22.wan_video_dit import flash_attention
from fastwam.utils.logging_config import get_logger
from TMD3.state_progress_mot import StateProgressMoT


logger = get_logger(__name__)


class InstructionStateMoT(StateProgressMoT):
    """TMD3 ablation with T5 instruction and optional state cross-attention.

    Video/Action full attention retains the exact TMD2 visibility mask.  No
    Progress Expert key/value, progress-logit, or generated-subtask tensor is
    constructed or consumed.  The legacy contract lets both experts read the
    complete static T5 instruction followed by state4.  The current-state
    contract lets only Action read one projected qpos[t] token; Video reads the
    instruction alone.
    """

    def __init__(
        self,
        video_expert,
        action_expert,
        *,
        condition_dim: int = 4096,
        state_dim: int = 14,
        state_history_tokens: int = 4,
        state_time_start: float = 1.0,
        state_time_end: float = 4.0,
        state_time_gate_init: float = -3.0,
        state_temporal_encoding: bool = True,
        state_cross_attention_scope: str = "video_action",
        video_time_offset: float = 1.0,
        action_time_start: float = 4.0,
        action_time_end: float = 5.0,
        checkpoint_attention: bool = False,
    ):
        # Reuse the already validated temporal/state and block helpers.  The
        # Progress-specific modules are deleted immediately and therefore do
        # not enter parameters, checkpoints, optimizer state, or execution.
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            progress_hidden_dim=2560,
            condition_dim=int(condition_dim),
            progress_bins=10,
            progress_bottleneck_dim=1,
            progress_gate_init=-3.0,
            state_dim=int(state_dim),
            state_history_tokens=int(state_history_tokens),
            state_time_start=float(state_time_start),
            state_time_end=float(state_time_end),
            state_time_gate_init=float(state_time_gate_init),
            video_time_offset=float(video_time_offset),
            action_time_start=float(action_time_start),
            action_time_end=float(action_time_end),
            checkpoint_attention=bool(checkpoint_attention),
        )
        del self.progress_encoder
        del self.progress_gate
        del self.progress_k
        del self.progress_v
        del self.subtask_adapter
        self.state_temporal_encoding = bool(state_temporal_encoding)
        self.state_cross_attention_scope = str(state_cross_attention_scope)
        if self.state_cross_attention_scope not in {"video_action", "action_only"}:
            raise ValueError(
                "state_cross_attention_scope must be 'video_action' or "
                f"'action_only', got {self.state_cross_attention_scope!r}."
            )
        if not self.state_temporal_encoding:
            del self.state_time_gate
        logger.info(
            "Initialized InstructionStateMoT full=video+action_only "
            "cross_scope=%s state_tokens=%d state_temporal=%s progress=disabled",
            self.state_cross_attention_scope,
            self.state_history_tokens,
            self.state_temporal_encoding,
        )

    def _state_context(self, state_history: torch.Tensor) -> torch.Tensor:
        if self.state_temporal_encoding:
            return super()._state_context(state_history)
        if state_history.ndim != 3 or tuple(state_history.shape[1:]) != (
            self.state_history_tokens,
            self.state_dim,
        ):
            raise ValueError(
                "Current-state input must match [B,S,D]: "
                f"expected=(*,{self.state_history_tokens},{self.state_dim}), "
                f"got={tuple(state_history.shape)}."
            )
        return self.state_encoder(state_history)

    def _validate_instruction_conditions(
        self,
        instruction: torch.Tensor,
        instruction_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> None:
        if instruction.ndim != 3:
            raise ValueError(
                "T5 instruction must be [B,L,D], got "
                f"{tuple(instruction.shape)}."
            )
        if int(instruction.shape[-1]) != self.condition_dim:
            raise ValueError(
                f"T5 instruction width must be {self.condition_dim}, got "
                f"{instruction.shape[-1]}."
            )
        self._validate_mask(instruction_mask, instruction, "instruction")
        self._validate_mask(state_mask, state_history, "state")
        if int(state_history.shape[0]) != int(instruction.shape[0]):
            raise ValueError("Instruction/state batch mismatch.")
        if not bool(instruction_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain instruction context.")
        if not bool(state_mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least its current state.")
        self._state_context(state_history)

    def _precompute_instruction_inputs(
        self,
        instruction: torch.Tensor,
        instruction_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        states = self._state_context(state_history)
        adapted = {
            "video": (
                instruction
                if self.state_cross_attention_scope == "action_only"
                else torch.cat([instruction, states], dim=1)
            ),
            "action": torch.cat([instruction, states], dim=1),
        }
        contexts = {
            name: self.mixtures[name].text_embedding(adapted[name])
            for name in ("video", "action")
        }
        context_masks = {
            "video": (
                instruction_mask
                if self.state_cross_attention_scope == "action_only"
                else torch.cat([instruction_mask, state_mask], dim=1)
            ),
            "action": torch.cat([instruction_mask, state_mask], dim=1),
        }
        return {
            "contexts": contexts,
            "context_masks": {
                name: mask[:, None, None, :]
                for name, mask in context_masks.items()
            },
        }

    def _instruction_cross_payload(
        self,
        precomputed: dict[str, Any],
        layer_idx: int,
        *,
        cache_kv: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        payloads = {}
        for name in ("video", "action"):
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
        instruction: torch.Tensor,
        instruction_mask: torch.BoolTensor,
        state_history: torch.Tensor,
        state_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("Condition K/V caching is inference-only.")
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
        return {
            "layers": [
                {
                    "cross": self._instruction_cross_payload(
                        precomputed,
                        layer_idx,
                        cache_kv=True,
                    )
                }
                for layer_idx in range(self.num_layers)
            ]
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
        condition_layers = condition_cache.get("layers")
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        video_length = int(video_tokens.shape[1])
        mask = self._right_dit_mask(
            video_attention_mask,
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
            mixed = flash_attention(
                q=query,
                k=key,
                v=value,
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
        full_mask = self._right_dit_mask(
            dit_attention_mask,
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
            mixed = flash_attention(
                q=query,
                k=torch.cat(
                    [history_layers[layer_idx]["video_key"], key], dim=1
                ),
                v=torch.cat(
                    [history_layers[layer_idx]["video_value"], value], dim=1
                ),
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
                    f"InstructionStateMoT conditions are incomplete: {missing}."
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
            cached_layers = None
        else:
            precomputed = None
            cached_layers = condition_cache.get("layers")
            if not isinstance(cached_layers, list) or len(cached_layers) != self.num_layers:
                raise ValueError("Condition-cache depth does not match MoT depth.")

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

            cross = (
                self._instruction_cross_payload(
                    precomputed,
                    layer_idx,
                    cache_kv=False,
                )
                if cached_layers is None
                else cached_layers[layer_idx]["cross"]
            )
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


__all__ = ["InstructionStateMoT"]
