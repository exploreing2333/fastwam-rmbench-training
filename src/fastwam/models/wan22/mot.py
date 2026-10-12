from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        progress_conditioning_bins: Optional[int] = None,
        progress_conditioning_dim: Optional[int] = None,
        progress_conditioning_mode: str = "context_token",
        progress_modulation_hidden_dim: int = 128,
        progress_action_modulation_scale: float = 1.0,
        progress_video_modulation_scale: float = 0.1,
        train_progress_adapter_only: bool = False,
    ):
        super().__init__()
        # KV-only prefills never consume the final block output. This flag is
        # intentionally non-persistent so old checkpoint state_dicts stay exact.
        self.skip_unused_final_kv_output = True
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.train_progress_adapter_only = bool(
            train_progress_adapter_only
        )
        self.progress_conditioning_mode = str(
            progress_conditioning_mode
        ).strip().lower()
        if self.progress_conditioning_mode not in {
            "context_token",
            "append_context_token",
            "modulation",
        }:
            raise ValueError(
                "`progress_conditioning_mode` must be `context_token`, "
                "`append_context_token`, or "
                f"`modulation`, got {progress_conditioning_mode!r}."
            )
        self.progress_feature_encoder = None
        self.progress_modulation_projections = None
        self.progress_modulation_scales: dict[str, float] = {}
        if progress_conditioning_bins is None:
            self.progress_projection = None
        else:
            if int(progress_conditioning_bins) <= 1:
                raise ValueError(
                    "`progress_conditioning_bins` must be greater than one."
                )
            if (
                progress_conditioning_dim is None
                or int(progress_conditioning_dim) <= 0
            ):
                raise ValueError(
                    "`progress_conditioning_dim` must be positive when progress conditioning is enabled."
                )
            if self.progress_conditioning_mode in {
                "context_token",
                "append_context_token",
            }:
                self.progress_projection = nn.Linear(
                    int(progress_conditioning_bins),
                    int(progress_conditioning_dim),
                )
                # Start the new condition at a literal zero vector.  The legacy
                # fixed-slot mode is then exactly the subtask-only policy; the
                # append mode additionally controls token visibility with its
                # explicit present mask.
                nn.init.zeros_(self.progress_projection.weight)
                nn.init.zeros_(self.progress_projection.bias)
            else:
                self.progress_projection = None
                hidden_dim = int(progress_modulation_hidden_dim)
                if hidden_dim <= 0:
                    raise ValueError(
                        "`progress_modulation_hidden_dim` must be positive."
                    )
                # Preserve the whole soft distribution and expose its normalized
                # entropy as an explicit reliability feature.  Dropout is still
                # represented by `present_mask`, which zeros the residual exactly.
                self.progress_feature_encoder = nn.Sequential(
                    nn.Linear(int(progress_conditioning_bins) + 1, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )
                self.progress_modulation_projections = nn.ModuleDict()
                requested_scales = {
                    "video": float(progress_video_modulation_scale),
                    "action": float(progress_action_modulation_scale),
                }
                for name, expert in self.mixtures.items():
                    if name not in requested_scales:
                        continue
                    scale = requested_scales[name]
                    if scale < 0.0:
                        raise ValueError(
                            f"Progress modulation scale for {name!r} must be non-negative."
                        )
                    projection = nn.Linear(
                        hidden_dim,
                        6 * int(expert.hidden_dim),
                    )
                    # A zero final projection makes a legacy checkpoint exactly
                    # subtask-only at initialization, while retaining gradients
                    # for the new residual path from the first optimizer step.
                    nn.init.zeros_(projection.weight)
                    nn.init.zeros_(projection.bias)
                    self.progress_modulation_projections[name] = projection
                    self.progress_modulation_scales[name] = scale
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    def configure_trainable_parameters(self) -> None:
        """Optionally isolate training to the new progress adapter."""
        if not self.train_progress_adapter_only:
            return
        if (
            self.progress_projection is None
            and self.progress_modulation_projections is None
        ):
            raise RuntimeError(
                "Adapter-only training requested without a progress adapter."
            )
        self.requires_grad_(False)
        self.mixtures.eval()
        if self.progress_projection is not None:
            self.progress_projection.requires_grad_(True)
            self.progress_projection.train()
        if self.progress_feature_encoder is not None:
            self.progress_feature_encoder.requires_grad_(True)
            self.progress_feature_encoder.train()
        if self.progress_modulation_projections is not None:
            self.progress_modulation_projections.requires_grad_(True)
            self.progress_modulation_projections.train()

    def encode_progress(
        self,
        probabilities: torch.Tensor,
        present_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.progress_projection is None:
            raise RuntimeError("Progress conditioning is disabled.")
        if probabilities.ndim != 2:
            raise ValueError(
                "`probabilities` must be [B,num_bins], got "
                f"{tuple(probabilities.shape)}."
            )
        if present_mask.ndim != 1 or present_mask.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "`present_mask` must be [B], got "
                f"{tuple(present_mask.shape)} for batch {probabilities.shape[0]}."
            )
        projection = self.progress_projection
        token = projection(probabilities.to(dtype=projection.weight.dtype))
        return token * present_mask.to(device=token.device, dtype=token.dtype).unsqueeze(1)

    def encode_progress_modulations(
        self,
        probabilities: torch.Tensor,
        present_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode local progress as independent zero-init DiT modulations."""
        if (
            self.progress_feature_encoder is None
            or self.progress_modulation_projections is None
        ):
            raise RuntimeError("Progress modulation conditioning is disabled.")
        if probabilities.ndim != 2:
            raise ValueError(
                "`probabilities` must be [B,num_bins], got "
                f"{tuple(probabilities.shape)}."
            )
        if present_mask.ndim != 1 or present_mask.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "`present_mask` must be [B], got "
                f"{tuple(present_mask.shape)} for batch {probabilities.shape[0]}."
            )
        encoder = self.progress_feature_encoder
        probabilities_f32 = probabilities.to(device=present_mask.device, dtype=torch.float32)
        entropy = -(
            probabilities_f32.clamp_min(1e-8)
            * probabilities_f32.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        entropy = entropy / torch.log(
            probabilities_f32.new_tensor(float(probabilities_f32.shape[1]))
        )
        features = torch.cat([probabilities_f32, entropy], dim=1).to(
            device=next(encoder.parameters()).device,
            dtype=next(encoder.parameters()).dtype,
        )
        encoded = encoder(features)
        active = present_mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(1)
        encoded = encoded * active

        outputs: Dict[str, torch.Tensor] = {}
        for name, projection in self.progress_modulation_projections.items():
            expert = self.mixtures[name]
            modulation = projection(encoded).unflatten(1, (6, int(expert.hidden_dim)))
            outputs[name] = (
                modulation
                * active.unsqueeze(2)
                * self.progress_modulation_scales[name]
            )
        return outputs

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
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

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        kv_only: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = None if kv_only else block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        if q is not None:
            q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
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

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            kv_only = self.skip_unused_final_kv_output and layer_idx == self.num_layers - 1
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
                kv_only=kv_only,
            )
            if kv_only:
                kv_cache.append({"k": k, "v": v})
                continue
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def append_video_with_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Append new video tokens to a causal persistent video KV cache.

        The cached prefix is treated as immutable history: new video queries
        attend to the prefix and to the new frame, while prefix K/V are never
        recomputed or allowed to attend to the new frame.  This is the exact
        streaming-cache operation needed for event anchors.
        """
        _, cache = self._video_with_cache(
            video_tokens=video_tokens, video_freqs=video_freqs,
            video_t_mod=video_t_mod, video_context_payload=video_context_payload,
            video_kv_cache=video_kv_cache, attention_mask=attention_mask,
            video_seq_len=video_seq_len, collect_kv=True,
        )
        return cache

    def forward_video_with_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Denoise future video queries against an immutable clean prefix.

        Unlike append_video_with_cache, noisy future K/V are not returned or
        persisted. Both video and action losses may backpropagate into the
        shared prefix; future targets cannot alter the action-conditioning KV.
        """
        tokens, _ = self._video_with_cache(
            video_tokens=video_tokens, video_freqs=video_freqs,
            video_t_mod=video_t_mod, video_context_payload=video_context_payload,
            video_kv_cache=video_kv_cache, attention_mask=attention_mask,
            video_seq_len=video_seq_len, collect_kv=False,
        )
        return tokens

    def _video_with_cache(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        collect_kv: bool,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `append_video_with_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, "
                f"got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be [S_new,S_prefix+S_new], got {tuple(attention_mask.shape)}"
            )
        new_seq_len = int(video_tokens.shape[1])
        total_seq_len = int(video_seq_len) + new_seq_len
        if tuple(attention_mask.shape) != (new_seq_len, total_seq_len):
            raise ValueError(
                "Append-video attention mask mismatch: "
                f"mask={tuple(attention_mask.shape)}, expected={(new_seq_len, total_seq_len)}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        output_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            kv_only = collect_kv and self.skip_unused_final_kv_output and layer_idx == self.num_layers - 1
            (
                q_new,
                k_new,
                v_new,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
                kv_only=kv_only,
            )
            layer_cache = video_kv_cache[layer_idx]
            k_prefix = layer_cache["k"]
            v_prefix = layer_cache["v"]
            if k_prefix.shape[1] != video_seq_len or v_prefix.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` prefix length mismatch: "
                    f"expected={video_seq_len}, got={k_prefix.shape[1]}/{v_prefix.shape[1]}"
                )
            k_cat = torch.cat([k_prefix, k_new], dim=1)
            v_cat = torch.cat([v_prefix, v_new], dim=1)
            if kv_only:
                # append_video_with_cache discards the token output. Future
                # prediction (collect_kv=False) always executes the full block.
                output_cache.append({"k": k_cat, "v": v_cat})
                continue
            mixed = self._mixed_attention(
                q_cat=q_new,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            if collect_kv:
                output_cache.append({"k": k_cat, "v": v_cat})
        return x, output_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
