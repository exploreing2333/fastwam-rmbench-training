from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class TriMoT(nn.Module):
    """Qwen-conditioned Video/Action MoT.

    ``condition_mode='cross_attn'`` keeps Qwen's native causal self-attention,
    runs only Video/Action in shared MoT attention, and conditions both DiTs at
    every layer through their original cross-attention blocks.  ``joint_mot``
    preserves the previous three-way shared-attention implementation.
    """

    def __init__(
        self,
        qwen_expert: nn.Module,
        video_expert: nn.Module,
        action_expert: nn.Module,
        mot_checkpoint_mixed_attn: bool = True,
        mot_checkpoint_qwen_post: Optional[bool] = None,
        condition_mode: str = "cross_attn",
    ):
        super().__init__()
        self.mixtures = nn.ModuleDict(
            {
                "qwen": qwen_expert,
                "video": video_expert,
                "action": action_expert,
            }
        )
        self.expert_order = ["qwen", "video", "action"]
        self.condition_mode = str(condition_mode).lower()
        if self.condition_mode not in {"cross_attn", "joint_mot"}:
            raise ValueError(
                "`condition_mode` must be 'cross_attn' or 'joint_mot', got "
                f"{condition_mode!r}"
            )
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.mot_checkpoint_qwen_post = (
            self.mot_checkpoint_mixed_attn
            if mot_checkpoint_qwen_post is None
            else bool(mot_checkpoint_qwen_post)
        )
        self.qwen_num_layers = int(qwen_expert.num_layers)
        self.num_layers = int(len(video_expert.blocks))
        self.num_heads = int(video_expert.num_heads)
        self.attn_head_dim = int(video_expert.attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim

        if int(action_expert.num_heads) != self.num_heads:
            raise ValueError("ActionDiT `num_heads` must match video expert for TriMoT.")
        if int(action_expert.attn_head_dim) != self.attn_head_dim:
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for TriMoT.")
        if self.condition_mode == "joint_mot":
            if int(qwen_expert.shared_num_heads) != self.num_heads:
                raise ValueError("Qwen shared_num_heads must match video expert for TriMoT.")
            if int(qwen_expert.shared_attn_head_dim) != self.attn_head_dim:
                raise ValueError("Qwen shared_attn_head_dim must match video expert for TriMoT.")
        else:
            for name, expert in (("video", video_expert), ("action", action_expert)):
                text_embedding = expert.text_embedding
                input_dim = int(text_embedding[0].in_features)
                if input_dim != int(qwen_expert.condition_dim):
                    raise ValueError(
                        f"{name} text embedding expects {input_dim} condition dims, but Qwen "
                        f"adapter outputs {qwen_expert.condition_dim}."
                    )
        if len(video_expert.blocks) != len(action_expert.blocks):
            raise ValueError("Video/action experts must have the same number of layers.")
        if self.qwen_num_layers <= 0:
            raise ValueError(f"Qwen MoT layer count must be positive, got {self.qwen_num_layers}")
        if self.qwen_num_layers > self.num_layers:
            raise ValueError(
                "Qwen MoT layers cannot exceed video/action DiT layers: "
                f"qwen={self.qwen_num_layers}, video/action={self.num_layers}."
            )

        logger.info(
            "Initialized TriMoT experts=%s qwen_layers=%d dit_layers=%d heads=%d head_dim=%d "
            "condition_mode=%s checkpoint_mixed_attn=%s checkpoint_qwen_post=%s",
            self.expert_order,
            self.qwen_num_layers,
            self.num_layers,
            self.num_heads,
            self.attn_head_dim,
            self.condition_mode,
            self.mot_checkpoint_mixed_attn,
            self.mot_checkpoint_qwen_post,
        )

    def configure_trainable_parameters(self) -> None:
        """Disable parameters that are structurally unused by the selected path."""
        qwen = self.mixtures["qwen"]
        configure = getattr(qwen, "apply_training_freeze_policy", None)
        if callable(configure):
            configure(disable_shared_adapters=self.condition_mode == "cross_attn")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(
            6, dim=chunk_dim
        )
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _norm(norm_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm_module, nn.LayerNorm):
            weight = None if norm_module.weight is None else norm_module.weight.to(device=x.device, dtype=x.dtype)
            bias = None if norm_module.bias is None else norm_module.bias.to(device=x.device, dtype=x.dtype)
            return F.layer_norm(
                x,
                norm_module.normalized_shape,
                weight=weight,
                bias=bias,
                eps=float(norm_module.eps),
            )
        weight = norm_module.weight.to(device=x.device, dtype=x.dtype)
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=float(norm_module.eps))

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get("context")
            context_key = context_payload.get("key")
            context_value = context_payload.get("value")
            if context is not None or context_key is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                cross_query = block.norm3(x)
                if context_key is not None:
                    if context_value is None:
                        raise ValueError("Cached cross-attention key requires a cached value.")
                    cross_out = block.cross_attn.forward_with_kv(
                        cross_query,
                        context_key,
                        context_value,
                        ctx_mask=context_mask,
                    )
                else:
                    cross_out = block.cross_attn(cross_query, context, ctx_mask=context_mask)
                x = x + cross_out
        mlp_input = modulate(TriMoT._norm(block.norm2, x), shift_mlp, scale_mlp)
        return block.gate(x, gate_mlp, block.ffn(mlp_input))

    def _build_dit_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(self._norm(block.norm1, x), shift_msa, scale_msa)
        q = self._norm(block.self_attn.norm_q, block.self_attn.q(attn_input))
        k = self._norm(block.self_attn.norm_k, block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        return q, k, v, {
            "block": block,
            "residual_x": x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "use_gradient_checkpointing": bool(getattr(expert, "use_gradient_checkpointing", False)),
        }

    def _apply_dit_post(
        self,
        cache: dict[str, Any],
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _context: Optional[torch.Tensor] = None,
            _context_mask: Optional[torch.Tensor] = None,
            _block=cache["block"],
        ) -> torch.Tensor:
            local_payload = context_payload
            if _context is not None:
                local_payload = {"context": _context, "mask": _context_mask}
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=local_payload,
            )

        if cache["use_gradient_checkpointing"] and self.training:
            if context_payload is not None and context_payload.get("context") is not None:
                return torch.utils.checkpoint.checkpoint(
                    _post_fn,
                    mixed_slice,
                    cache["residual_x"],
                    cache["gate_msa"],
                    cache["shift_mlp"],
                    cache["scale_mlp"],
                    cache["gate_mlp"],
                    context_payload["context"],
                    context_payload.get("mask"),
                    use_reentrant=False,
                )
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                cache["residual_x"],
                cache["gate_msa"],
                cache["shift_mlp"],
                cache["scale_mlp"],
                cache["gate_mlp"],
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            cache["residual_x"],
            cache["gate_msa"],
            cache["shift_mlp"],
            cache["scale_mlp"],
            cache["gate_mlp"],
        )

    def _apply_qwen_post(
        self,
        qwen,
        layer_idx: int,
        mixed_slice: torch.Tensor,
        cache: dict[str, Any],
        *,
        from_shared: bool = True,
    ) -> torch.Tensor:
        def _post_fn(_mixed_slice: torch.Tensor, _residual: torch.Tensor) -> torch.Tensor:
            local_cache = dict(cache)
            local_cache["residual"] = _residual
            if from_shared:
                return qwen.apply_post_attention(layer_idx, _mixed_slice, local_cache)
            return qwen.apply_post_attention(
                layer_idx,
                _mixed_slice,
                local_cache,
                from_shared=False,
            )

        if self.mot_checkpoint_qwen_post and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                cache["residual"],
                use_reentrant=False,
            )
        if from_shared:
            return qwen.apply_post_attention(layer_idx, mixed_slice, cache)
        return qwen.apply_post_attention(
            layer_idx,
            mixed_slice,
            cache,
            from_shared=False,
        )

    def _run_qwen_native_layer(
        self,
        qwen,
        layer_idx: int,
        tokens: torch.Tensor,
        qwen_pre_state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, cache = qwen.build_attention_io(
            layer_idx=layer_idx,
            tokens=tokens,
            pre_state=qwen_pre_state,
            project_to_shared=False,
        )

        def _native_attention(_q: torch.Tensor, _k: torch.Tensor, _v: torch.Tensor) -> torch.Tensor:
            return qwen.run_native_self_attention(
                _q,
                _k,
                _v,
                qwen_pre_state["attention_mask"],
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            attention_out = torch.utils.checkpoint.checkpoint(
                _native_attention,
                q,
                k,
                v,
                use_reentrant=False,
            )
        else:
            attention_out = _native_attention(q, k, v)
        next_tokens = self._apply_qwen_post(
            qwen,
            layer_idx,
            attention_out,
            cache,
            from_shared=False,
        )
        return next_tokens, cache["normalized_tokens"]

    def _build_condition_payloads(
        self,
        normalized_qwen_tokens: torch.Tensor,
        qwen_pre_state: dict[str, Any],
        layer_idx: int,
        *,
        cache_kv: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        qwen = self.mixtures["qwen"]
        condition, condition_valid = qwen.build_condition_tokens(
            normalized_qwen_tokens,
            qwen_pre_state,
        )
        mask = condition_valid[:, None, None, :]
        payloads = {}
        for name in ("video", "action"):
            expert = self.mixtures[name]
            context = expert.text_embedding(condition)
            if cache_kv:
                key, value = expert.blocks[layer_idx].cross_attn.project_context(context)
                payloads[name] = {
                    "key": key.detach(),
                    "value": value.detach(),
                    "mask": mask.detach(),
                }
            else:
                payloads[name] = {"context": context, "mask": mask}
        return payloads

    @torch.no_grad()
    def prepare_condition_cache(
        self,
        qwen_pre_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Qwen once and cache per-layer Video/Action cross-attention K/V."""
        if self.condition_mode != "cross_attn":
            raise RuntimeError("Condition K/V caching is only available in cross-attention mode.")
        if self.training:
            raise RuntimeError("Condition K/V caching is inference-only; call model.eval() first.")
        qwen = self.mixtures["qwen"]
        tokens = qwen_pre_state["tokens"]
        layers = []
        for layer_idx in range(self.qwen_num_layers):
            tokens, normalized = self._run_qwen_native_layer(
                qwen,
                layer_idx,
                tokens,
                qwen_pre_state,
            )
            layers.append(
                self._build_condition_payloads(
                    normalized,
                    qwen_pre_state,
                    layer_idx,
                    cache_kv=True,
                )
            )
        final_normalized = qwen.final_condition_input(tokens)
        for layer_idx in range(self.qwen_num_layers, self.num_layers):
            layers.append(
                self._build_condition_payloads(
                    final_normalized,
                    qwen_pre_state,
                    layer_idx,
                    cache_kv=True,
                )
            )
        return {"layers": layers, "qwen_tokens": tokens.detach()}

    @staticmethod
    def _validate_square_mask(mask: torch.Tensor, name: str) -> None:
        if mask.ndim not in (2, 4) or mask.shape[-2] != mask.shape[-1]:
            raise ValueError(
                f"`{name}` must be square [S,S] or [B,1,S,S], got {tuple(mask.shape)}"
            )

    def _validate_mixed_mask(
        self,
        mask: torch.Tensor,
        tokens: torch.Tensor,
        name: str,
    ) -> None:
        self._validate_square_mask(mask, name)
        if mask.shape[-1] != tokens.shape[1]:
            raise ValueError(
                f"{name} seq length mismatch: mask={tuple(mask.shape)} vs tokens={tokens.shape[1]}"
            )
        if mask.ndim == 4 and mask.shape[0] not in (1, tokens.shape[0]):
            raise ValueError(
                f"{name} batch mismatch: mask_batch={mask.shape[0]} vs token_batch={tokens.shape[0]}"
            )

    def _forward_cross_attn(
        self,
        embeds_all: dict[str, torch.Tensor],
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        qwen_pre_state: dict[str, Any],
        qwen_condition_cache: Optional[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        tokens_all = {name: embeds_all[name] for name in self.expert_order}
        qwen = self.mixtures["qwen"]
        cached_layers = None
        if qwen_condition_cache is not None:
            cached_layers = qwen_condition_cache.get("layers")
            if not isinstance(cached_layers, list) or len(cached_layers) != self.num_layers:
                raise ValueError("Qwen condition cache depth does not match DiT depth.")
            tokens_all["qwen"] = qwen_condition_cache.get("qwen_tokens", tokens_all["qwen"])

        final_normalized = None
        for layer_idx in range(self.num_layers):
            if cached_layers is not None:
                context_payloads = cached_layers[layer_idx]
            elif layer_idx < self.qwen_num_layers:
                tokens_all["qwen"], normalized = self._run_qwen_native_layer(
                    qwen,
                    layer_idx,
                    tokens_all["qwen"],
                    qwen_pre_state,
                )
                context_payloads = self._build_condition_payloads(
                    normalized,
                    qwen_pre_state,
                    layer_idx,
                    cache_kv=False,
                )
            else:
                if final_normalized is None:
                    final_normalized = qwen.final_condition_input(tokens_all["qwen"])
                context_payloads = self._build_condition_payloads(
                    final_normalized,
                    qwen_pre_state,
                    layer_idx,
                    cache_kv=False,
                )

            q_chunks = []
            k_chunks = []
            v_chunks = []
            caches = {}
            for name in ("video", "action"):
                expert = self.mixtures[name]
                q, k, v, cache = self._build_dit_attention_io(
                    expert=expert,
                    block=expert.blocks[layer_idx],
                    x=tokens_all[name],
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                caches[name] = cache

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            self._validate_mixed_mask(dit_attention_mask, q_cat, "dit_attention_mask")
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, dit_attention_mask)

            start = 0
            for name, q_chunk in zip(("video", "action"), q_chunks):
                end = start + q_chunk.shape[1]
                tokens_all[name] = self._apply_dit_post(
                    cache=caches[name],
                    mixed_slice=mixed[:, start:end],
                    context_payload=context_payloads[name],
                )
                start = end
        return tokens_all

    def _forward_joint_mot(
        self,
        embeds_all: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        qwen_pre_state: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        self._validate_square_mask(attention_mask, "attention_mask")
        self._validate_square_mask(dit_attention_mask, "dit_attention_mask")
        tokens_all = {name: embeds_all[name] for name in self.expert_order}
        qwen = self.mixtures["qwen"]

        for layer_idx in range(self.num_layers):
            use_qwen = layer_idx < self.qwen_num_layers
            layer_order = self.expert_order if use_qwen else ["video", "action"]
            q_chunks = []
            k_chunks = []
            v_chunks = []
            caches = {}
            seq_lens = []

            if use_qwen:
                q, k, v, cache = qwen.build_attention_io(
                    layer_idx=layer_idx,
                    tokens=tokens_all["qwen"],
                    pre_state=qwen_pre_state,
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(q.shape[1])
                caches["qwen"] = cache

            for name in ("video", "action"):
                expert = self.mixtures[name]
                q, k, v, cache = self._build_dit_attention_io(
                    expert=expert,
                    block=expert.blocks[layer_idx],
                    x=tokens_all[name],
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(q.shape[1])
                caches[name] = cache

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            layer_mask = attention_mask if use_qwen else dit_attention_mask
            self._validate_mixed_mask(layer_mask, q_cat, "attention_mask")
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, layer_mask)

            start = 0
            for name, seq_len in zip(layer_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end]
                if name == "qwen":
                    tokens_all[name] = self._apply_qwen_post(qwen, layer_idx, mixed_slice, caches[name])
                else:
                    tokens_all[name] = self._apply_dit_post(
                        cache=caches[name],
                        mixed_slice=mixed_slice,
                        context_payload=None,
                    )
                start = end
        return tokens_all

    def forward(
        self,
        embeds_all: dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        qwen_pre_state: dict[str, Any],
        qwen_condition_cache: Optional[dict[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        if self.condition_mode == "cross_attn":
            return self._forward_cross_attn(
                embeds_all=embeds_all,
                dit_attention_mask=dit_attention_mask,
                freqs_all=freqs_all,
                t_mod_all=t_mod_all,
                qwen_pre_state=qwen_pre_state,
                qwen_condition_cache=qwen_condition_cache,
            )
        if qwen_condition_cache is not None:
            raise ValueError("Qwen condition cache cannot be used with joint_mot mode.")
        if attention_mask is None:
            raise ValueError("`attention_mask` is required by joint_mot mode.")
        return self._forward_joint_mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            dit_attention_mask=dit_attention_mask,
            freqs_all=freqs_all,
            t_mod_all=t_mod_all,
            qwen_pre_state=qwen_pre_state,
        )
