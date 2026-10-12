from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.wan_video_dit import (
    flash_attention,
    modulate,
    rope_apply,
)
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProgressMoT(nn.Module):
    """Full-attention MoT over cached Progress, Video and Action tokens."""

    def __init__(
        self,
        video_expert: nn.Module,
        action_expert: nn.Module,
        progress_hidden_dim: int = 2560,
        condition_dim: int = 4096,
        checkpoint_attention: bool = False,
    ):
        super().__init__()
        self.mixtures = nn.ModuleDict(
            {"video": video_expert, "action": action_expert}
        )
        self.num_layers = int(len(video_expert.blocks))
        self.num_heads = int(video_expert.num_heads)
        self.attn_head_dim = int(video_expert.attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.progress_hidden_dim = int(progress_hidden_dim)
        self.condition_dim = int(condition_dim)
        self.checkpoint_attention = bool(checkpoint_attention)

        if len(action_expert.blocks) != self.num_layers:
            raise ValueError("Video/action experts must have the same depth.")
        if int(action_expert.num_heads) != self.num_heads:
            raise ValueError("Video/action experts must have the same head count.")
        if int(action_expert.attn_head_dim) != self.attn_head_dim:
            raise ValueError("Video/action experts must have the same head width.")

        # Match the original Qwen MoT contract: the expert-to-shared-space
        # adapters are shared across depth, while their inputs come from the
        # corresponding Progress Model layer.
        self.progress_k = nn.Linear(self.progress_hidden_dim, self.attn_dim)
        self.progress_v = nn.Linear(self.progress_hidden_dim, self.attn_dim)
        self.subtask_adapter = nn.Linear(
            self.progress_hidden_dim,
            self.condition_dim,
        )
        logger.info(
            "Initialized ProgressMoT full_attention=true layers=%d heads=%d head_dim=%d "
            "memory_dim=%d condition_dim=%d gate=disabled logit_bias=disabled",
            self.num_layers,
            self.num_heads,
            self.attn_head_dim,
            self.progress_hidden_dim,
            self.condition_dim,
        )

    @property
    def video_expert(self):
        return self.mixtures["video"]

    @property
    def action_expert(self):
        return self.mixtures["action"]

    @staticmethod
    def _norm(norm_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm_module, nn.LayerNorm):
            weight = None if norm_module.weight is None else norm_module.weight.to(x)
            bias = None if norm_module.bias is None else norm_module.bias.to(x)
            return F.layer_norm(
                x,
                norm_module.normalized_shape,
                weight=weight,
                bias=bias,
                eps=float(norm_module.eps),
            )
        return F.rms_norm(
            x,
            (x.shape[-1],),
            weight=norm_module.weight.to(x),
            eps=float(norm_module.eps),
        )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        values = (
            block.modulation.to(device=t_mod.device, dtype=t_mod.dtype) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            values = tuple(value.squeeze(2) for value in values)
        return values

    def _build_dit_attention_io(
        self,
        expert,
        block,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._split_modulation(block, t_mod)
        attn_input = modulate(
            self._norm(block.norm1, tokens),
            shift_msa,
            scale_msa,
        )
        query = self._norm(block.self_attn.norm_q, block.self_attn.q(attn_input))
        key = self._norm(block.self_attn.norm_k, block.self_attn.k(attn_input))
        value = block.self_attn.v(attn_input)
        query = rope_apply(query, freqs, block.num_heads)
        key = rope_apply(key, freqs, block.num_heads)
        return query, key, value, {
            "block": block,
            "tokens": tokens,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "checkpoint": bool(getattr(expert, "use_gradient_checkpointing", False)),
        }

    @staticmethod
    def _cross_attention(block, x, payload):
        mask = payload["mask"]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        query = block.cross_attn.norm_q(block.cross_attn.q(x))
        if "key" in payload:
            key = payload["key"]
            value = payload["value"]
        else:
            context = payload["context"]
            key = block.cross_attn.norm_k(block.cross_attn.k(context))
            value = block.cross_attn.v(context)
        attended = flash_attention(
            q=query,
            k=key,
            v=value,
            num_heads=block.cross_attn.num_heads,
            ctx_mask=mask,
        )
        return block.cross_attn.o(attended)

    def _post_block(self, cache, mixed: torch.Tensor, payload: dict) -> torch.Tensor:
        block = cache["block"]
        gigaworld_layout = bool(
            getattr(block, "_gigaworld_layout", False)
        )
        cross_attention_norm = (
            block.norm2 if gigaworld_layout else block.norm3
        )
        feed_forward_norm = (
            block.norm3 if gigaworld_layout else block.norm2
        )

        def post(
            mixed_value,
            residual,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            context=None,
            context_mask=None,
        ):
            local_payload = payload
            if context is not None:
                local_payload = {"context": context, "mask": context_mask}
            x = block.gate(
                residual,
                gate_msa,
                block.self_attn.o(mixed_value),
            )
            x = x + self._cross_attention(
                block,
                cross_attention_norm(x),
                local_payload,
            )
            mlp_input = modulate(
                self._norm(feed_forward_norm, x),
                shift_mlp,
                scale_mlp,
            )
            return block.gate(x, gate_mlp, block.ffn(mlp_input))

        args = (
            mixed,
            cache["tokens"],
            cache["gate_msa"],
            cache["shift_mlp"],
            cache["scale_mlp"],
            cache["gate_mlp"],
        )
        if cache["checkpoint"] and self.training and "context" in payload:
            return torch.utils.checkpoint.checkpoint(
                post,
                *args,
                payload["context"],
                payload["mask"],
                use_reentrant=False,
            )
        if cache["checkpoint"] and self.training:
            return torch.utils.checkpoint.checkpoint(
                post,
                *args,
                use_reentrant=False,
            )
        return post(*args)

    @staticmethod
    def _full_attention_mask(
        dit_mask: torch.Tensor,
        memory_mask: torch.BoolTensor,
        dit_length: int,
    ) -> torch.BoolTensor:
        batch_size, memory_length = memory_mask.shape
        if dit_mask.ndim == 2:
            right = dit_mask[None, None].expand(batch_size, 1, -1, -1)
        elif dit_mask.ndim == 4:
            if dit_mask.shape[0] not in (1, batch_size):
                raise ValueError("DiT attention-mask batch does not match tokens.")
            right = dit_mask.expand(batch_size, -1, -1, -1)
        else:
            raise ValueError(
                f"DiT attention mask must be [Q,K] or [B,1,Q,K], got {tuple(dit_mask.shape)}"
            )
        if right.shape[-2:] != (dit_length, dit_length):
            raise ValueError(
                "DiT attention mask must cover Video+Action tokens exactly: "
                f"mask={tuple(right.shape)}, tokens={dit_length}."
            )
        full = torch.zeros(
            batch_size,
            right.shape[1],
            memory_length + dit_length,
            memory_length + dit_length,
            device=right.device,
            dtype=torch.bool,
        )
        progress_self = (
            memory_mask[:, None, :, None]
            & memory_mask[:, None, None, :]
        )
        full[:, :, :memory_length, :memory_length] = progress_self
        full[:, :, memory_length:, :memory_length] = memory_mask[
            :, None, None, :
        ].expand(
            batch_size,
            right.shape[1],
            dit_length,
            memory_length,
        )
        full[:, :, memory_length:, memory_length:] = right.bool()
        return full

    def _project_memory(self, memory_tokens: torch.Tensor):
        key = self.progress_k(memory_tokens)
        key = F.rms_norm(key, (key.shape[-1],), eps=1e-6)
        return key, self.progress_v(memory_tokens)

    def _subtask_for_layer(
        self,
        subtask_tokens: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if subtask_tokens.ndim == 3:
            return subtask_tokens
        if subtask_tokens.ndim != 4:
            raise ValueError(
                "Progress subtask conditioning must be final [B,S,H] or "
                f"layerwise [B,L,S,H], got {tuple(subtask_tokens.shape)}."
            )
        if int(subtask_tokens.shape[1]) != self.num_layers:
            raise ValueError(
                "Layerwise Progress subtask depth does not match MoT depth: "
                f"{subtask_tokens.shape[1]} vs {self.num_layers}."
            )
        return subtask_tokens[:, layer_idx]

    def _build_cross_payloads(
        self,
        subtask_tokens: torch.Tensor,
        subtask_mask: torch.BoolTensor,
        layer_idx: int,
        cache_kv: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        adapted = self.subtask_adapter(subtask_tokens)
        mask = subtask_mask[:, None, None, :]
        payloads = {}
        for name in ("video", "action"):
            expert = self.mixtures[name]
            context = expert.text_embedding(adapted)
            if cache_kv:
                cross = expert.blocks[layer_idx].cross_attn
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
        memory_tokens: torch.Tensor,
        memory_mask: torch.BoolTensor,
        subtask_tokens: torch.Tensor,
        subtask_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("Condition K/V caching is inference-only.")
        layers = []
        if memory_tokens.ndim != 4 or memory_tokens.shape[1] != self.num_layers:
            raise ValueError("Inference memory must be [B,30,M,H].")
        for layer_idx in range(self.num_layers):
            memory_key, memory_value = self._project_memory(
                memory_tokens[:, layer_idx],
            )
            layers.append(
                {
                    "memory_key": memory_key.detach(),
                    "memory_value": memory_value.detach(),
                    "cross": self._build_cross_payloads(
                        self._subtask_for_layer(
                            subtask_tokens,
                            layer_idx,
                        ),
                        subtask_mask,
                        layer_idx,
                        cache_kv=True,
                    ),
                }
            )
        return {"layers": layers, "memory_mask": memory_mask.detach()}

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
        """Cache the static history branch for one diffusion replan.

        This cache is deliberately local to a single replan.  It is exact for
        the TMD2 mask because Video history queries cannot see Action tokens;
        Progress and history are fixed throughout all denoising iterations.
        """
        if self.training:
            raise RuntimeError("History K/V caching is inference-only.")
        cached_layers = condition_cache.get("layers")
        memory_mask = condition_cache.get("memory_mask")
        if not isinstance(cached_layers, list) or len(cached_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        if not torch.is_tensor(memory_mask) or memory_mask.ndim != 2:
            raise ValueError("Condition cache must provide memory_mask [B,M].")

        video_length = int(video_tokens.shape[1])
        full_attention_mask = self._full_attention_mask(
            video_attention_mask,
            memory_mask,
            video_length,
        ).to(device=video_tokens.device)
        memory_length = int(memory_mask.shape[1])
        tokens = video_tokens
        layers = []

        for layer_idx in range(self.num_layers):
            expert = self.video_expert
            query, key, value, post_cache = self._build_dit_attention_io(
                expert,
                expert.blocks[layer_idx],
                tokens,
                video_freqs,
                video_t_mod,
            )
            layer_condition = cached_layers[layer_idx]
            memory_key = layer_condition["memory_key"]
            memory_value = layer_condition["memory_value"]
            layers.append(
                {
                    "video_key": key.detach(),
                    "video_value": value.detach(),
                }
            )
            mixed = flash_attention(
                q=torch.cat([memory_key, query], dim=1),
                k=torch.cat([memory_key, key], dim=1),
                v=torch.cat([memory_value, value], dim=1),
                num_heads=self.num_heads,
                ctx_mask=full_attention_mask,
            )
            tokens = self._post_block(
                post_cache,
                mixed[:, memory_length:],
                layer_condition["cross"]["video"],
            )

        return {
            "layers": layers,
            "video_tokens": tokens.detach(),
            "video_length": video_length,
            "memory_length": memory_length,
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
        """Run only the noisy Action branch against cached static history."""
        if self.training:
            raise RuntimeError("History K/V caching is inference-only.")
        condition_layers = condition_cache.get("layers")
        memory_mask = condition_cache.get("memory_mask")
        history_layers = history_cache.get("layers")
        if not isinstance(condition_layers, list) or len(condition_layers) != self.num_layers:
            raise ValueError("Condition-cache depth does not match MoT depth.")
        if not isinstance(history_layers, list) or len(history_layers) != self.num_layers:
            raise ValueError("History-cache depth does not match MoT depth.")

        video_length = int(history_cache["video_length"])
        action_length = int(action_tokens.shape[1])
        dit_length = video_length + action_length
        full_attention_mask = self._full_attention_mask(
            dit_attention_mask,
            memory_mask,
            dit_length,
        ).to(device=action_tokens.device)
        memory_length = int(memory_mask.shape[1])
        # Query only Action rows while retaining Progress+Video+Action keys.
        action_attention_mask = full_attention_mask[
            :,
            :,
            memory_length + video_length :,
            :,
        ]
        tokens = action_tokens

        for layer_idx in range(self.num_layers):
            expert = self.action_expert
            query, key, value, post_cache = self._build_dit_attention_io(
                expert,
                expert.blocks[layer_idx],
                tokens,
                action_freqs,
                action_t_mod,
            )
            layer_condition = condition_layers[layer_idx]
            layer_history = history_layers[layer_idx]
            mixed = flash_attention(
                q=query,
                k=torch.cat(
                    [
                        layer_condition["memory_key"],
                        layer_history["video_key"],
                        key,
                    ],
                    dim=1,
                ),
                v=torch.cat(
                    [
                        layer_condition["memory_value"],
                        layer_history["video_value"],
                        value,
                    ],
                    dim=1,
                ),
                num_heads=self.num_heads,
                ctx_mask=action_attention_mask,
            )
            tokens = self._post_block(
                post_cache,
                mixed,
                layer_condition["cross"]["action"],
            )
        return tokens

    def forward(
        self,
        embeds_all: dict[str, torch.Tensor],
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.BoolTensor] = None,
        subtask_tokens: Optional[torch.Tensor] = None,
        subtask_mask: Optional[torch.BoolTensor] = None,
        condition_cache: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        tokens = {name: embeds_all[name] for name in ("video", "action")}
        if condition_cache is None:
            required = (memory_tokens, memory_mask, subtask_tokens, subtask_mask)
            if any(value is None for value in required):
                raise ValueError("Progress Model condition tensors are incomplete.")
            if memory_tokens.ndim != 4:
                raise ValueError(
                    "Progress layer memory must be [B,30,M,H], got "
                    f"{tuple(memory_tokens.shape)}."
                )
            if memory_tokens.shape[1] != self.num_layers:
                raise ValueError(
                    "Progress cache depth does not match MoT depth: "
                    f"{memory_tokens.shape[1]} vs {self.num_layers}."
                )
            cached_layers = None
        else:
            cached_layers = condition_cache.get("layers")
            memory_mask = condition_cache.get("memory_mask")
            if not isinstance(cached_layers, list) or len(cached_layers) != self.num_layers:
                raise ValueError("Condition-cache depth does not match MoT depth.")

        dit_length = tokens["video"].shape[1] + tokens["action"].shape[1]
        full_attention_mask = self._full_attention_mask(
            dit_attention_mask,
            memory_mask,
            dit_length,
        ).to(device=tokens["video"].device)
        memory_length = int(memory_mask.shape[1])

        for layer_idx in range(self.num_layers):
            queries = []
            keys = []
            values = []
            caches = {}
            lengths = {}
            for name in ("video", "action"):
                expert = self.mixtures[name]
                query, key, value, cache = self._build_dit_attention_io(
                    expert,
                    expert.blocks[layer_idx],
                    tokens[name],
                    freqs_all[name],
                    t_mod_all[name],
                )
                queries.append(query)
                keys.append(key)
                values.append(value)
                caches[name] = cache
                lengths[name] = query.shape[1]

            if cached_layers is None:
                memory_key, memory_value = self._project_memory(
                    memory_tokens[:, layer_idx],
                )
                cross_payloads = self._build_cross_payloads(
                    self._subtask_for_layer(
                        subtask_tokens,
                        layer_idx,
                    ),
                    subtask_mask,
                    layer_idx,
                    cache_kv=False,
                )
            else:
                memory_key = cached_layers[layer_idx]["memory_key"]
                memory_value = cached_layers[layer_idx]["memory_value"]
                cross_payloads = cached_layers[layer_idx]["cross"]

            # Progress rows are evaluated in the same full attention call.
            # Their visibility is restricted to Progress keys and their
            # outputs are discarded because the frozen standalone branch has
            # already produced the next cached layer state.
            query_cat = torch.cat([memory_key, *queries], dim=1)
            key_cat = torch.cat([memory_key, *keys], dim=1)
            value_cat = torch.cat([memory_value, *values], dim=1)

            def attend(q, k, v):
                return flash_attention(
                    q=q,
                    k=k,
                    v=v,
                    num_heads=self.num_heads,
                    ctx_mask=full_attention_mask,
                )

            if self.checkpoint_attention and self.training:
                mixed = torch.utils.checkpoint.checkpoint(
                    attend,
                    query_cat,
                    key_cat,
                    value_cat,
                    use_reentrant=False,
                )
            else:
                mixed = attend(query_cat, key_cat, value_cat)

            video_start = memory_length
            video_end = video_start + lengths["video"]
            tokens["video"] = self._post_block(
                caches["video"],
                mixed[:, video_start:video_end],
                cross_payloads["video"],
            )
            tokens["action"] = self._post_block(
                caches["action"],
                mixed[:, video_end:],
                cross_payloads["action"],
            )
        return tokens


# Keep old checkpoints/imports readable while public naming uses Progress Model.
RobometerMoT = ProgressMoT
