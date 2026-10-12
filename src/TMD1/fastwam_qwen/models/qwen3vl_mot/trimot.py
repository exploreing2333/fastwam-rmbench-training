from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from TMD1.fastwam.models.wan22.wan_video_dit import flash_attention, modulate, rope_apply
from TMD1.fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class TriMoT(nn.Module):
    """MoT for Qwen3-VL, video DiT and action DiT.

    Qwen participates for ``qwen_expert.num_layers`` layers.  If the DiT experts
    have more layers, the remaining layers continue with the original
    video/action MoT visibility and no Qwen tokens.
    """

    def __init__(
        self,
        qwen_expert: nn.Module,
        video_expert: nn.Module,
        action_expert: nn.Module,
        mot_checkpoint_mixed_attn: bool = True,
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
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.qwen_num_layers = int(qwen_expert.num_layers)
        self.num_layers = int(len(video_expert.blocks))
        self.num_heads = int(video_expert.num_heads)
        self.attn_head_dim = int(video_expert.attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim

        if int(action_expert.num_heads) != self.num_heads:
            raise ValueError("ActionDiT `num_heads` must match video expert for TriMoT.")
        if int(action_expert.attn_head_dim) != self.attn_head_dim:
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for TriMoT.")
        if int(qwen_expert.shared_num_heads) != self.num_heads:
            raise ValueError("Qwen shared_num_heads must match video expert for TriMoT.")
        if int(qwen_expert.shared_attn_head_dim) != self.attn_head_dim:
            raise ValueError("Qwen shared_attn_head_dim must match video expert for TriMoT.")
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
            "Initialized TriMoT experts=%s qwen_layers=%d dit_layers=%d heads=%d head_dim=%d",
            self.expert_order,
            self.qwen_num_layers,
            self.num_layers,
            self.num_heads,
            self.attn_head_dim,
        )

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
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
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
            _block=cache["block"],
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if cache["use_gradient_checkpointing"] and self.training:
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
    ) -> torch.Tensor:
        def _post_fn(_mixed_slice: torch.Tensor, _residual: torch.Tensor) -> torch.Tensor:
            local_cache = dict(cache)
            local_cache["residual"] = _residual
            return qwen.apply_post_attention(layer_idx, _mixed_slice, local_cache)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                cache["residual"],
                use_reentrant=False,
            )
        return qwen.apply_post_attention(layer_idx, mixed_slice, cache)

    def forward(
        self,
        embeds_all: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        dit_attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        t_mod_all: dict[str, torch.Tensor],
        qwen_pre_state: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square [S,S], got {tuple(attention_mask.shape)}")
        if dit_attention_mask.ndim != 2 or dit_attention_mask.shape[0] != dit_attention_mask.shape[1]:
            raise ValueError(
                f"`dit_attention_mask` must be square [S,S], got {tuple(dit_attention_mask.shape)}"
            )
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
                block = expert.blocks[layer_idx]
                q, k, v, cache = self._build_dit_attention_io(
                    expert=expert,
                    block=block,
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
            mask_seq_len = layer_mask.shape[-1]
            if mask_seq_len != q_cat.shape[1] or layer_mask.shape[-2] != q_cat.shape[1]:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={tuple(layer_mask.shape)} vs tokens={q_cat.shape[1]}"
                )
            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=layer_mask)

            start = 0
            for name, seq_len in zip(layer_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
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
