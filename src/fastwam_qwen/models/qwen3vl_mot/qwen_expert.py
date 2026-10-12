from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _as_prompt_list(prompts: str | Sequence[str], batch_size: int) -> list[str]:
    if isinstance(prompts, str):
        return [prompts] * batch_size
    out = list(prompts)
    if len(out) != batch_size:
        raise ValueError(f"Prompt batch mismatch: got {len(out)} prompts for batch={batch_size}")
    return [str(p) for p in out]


def _identity_or_linear(in_dim: int, out_dim: int) -> nn.Module:
    if int(in_dim) == int(out_dim):
        return nn.Identity()
    return nn.Linear(int(in_dim), int(out_dim), bias=False)


class Qwen3VLExpert(nn.Module):
    """Qwen3-VL expert with native attention and a FastWAM condition adapter.

    Qwen3-VL has a different native attention width/head count than Wan DiT.
    The legacy joint-MoT path remains available through shared Q/K/V projection
    adapters.  The default path keeps Qwen's pretrained causal self-attention
    intact and exposes only compact language/state tokens through a trainable
    ``hidden_dim -> condition_dim`` adapter.  Passing ``num_mot_layers=None``
    uses all language decoder layers in the local Qwen checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        shared_num_heads: int,
        shared_attn_head_dim: int,
        num_mot_layers: Optional[int] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        freeze_qwen: bool = False,
        freeze_vision: bool = False,
        condition_dim: int = 4096,
        condition_target_rms: float = 0.07,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.shared_num_heads = int(shared_num_heads)
        self.shared_attn_head_dim = int(shared_attn_head_dim)
        self.shared_attn_dim = self.shared_num_heads * self.shared_attn_head_dim
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - depends on external env
            raise ImportError(
                "Qwen3VLExpert requires a transformers build that provides "
                "`Qwen3VLForConditionalGeneration`. Install the source/latest "
                "transformers package in the FastWAM environment."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "attn_implementation": attn_implementation,
            "low_cpu_mem_usage": True,
        }
        try:
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch_dtype,
                **load_kwargs,
            )
        except TypeError:
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )
        self.qwen.to(device=self.device, dtype=torch_dtype)

        text_cfg = self.qwen.config.text_config
        self.hidden_dim = int(text_cfg.hidden_size)
        self.native_num_heads = int(text_cfg.num_attention_heads)
        self.native_num_key_value_heads = int(text_cfg.num_key_value_heads)
        self.native_attn_head_dim = int(getattr(text_cfg, "head_dim", self.hidden_dim // self.native_num_heads))
        self.native_attn_dim = self.native_num_heads * self.native_attn_head_dim
        self.condition_dim = int(condition_dim)
        self.condition_target_rms = float(condition_target_rms)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.freeze_qwen = bool(freeze_qwen)
        self.freeze_vision = bool(freeze_vision)
        if self.condition_dim <= 0:
            raise ValueError(f"`condition_dim` must be positive, got {condition_dim}")
        if self.condition_target_rms <= 0.0:
            raise ValueError(
                f"`condition_target_rms` must be positive, got {condition_target_rms}"
            )

        self.layers = self.qwen.model.language_model.layers
        if num_mot_layers is None:
            self.num_layers = len(self.layers)
        else:
            self.num_layers = int(num_mot_layers)
        if self.num_layers <= 0:
            raise ValueError(f"`num_mot_layers` must be positive, got {num_mot_layers}")
        if self.num_layers > len(self.layers):
            raise ValueError(
                f"`num_mot_layers` exceeds Qwen decoder depth: requested={self.num_layers}, "
                f"available={len(self.layers)}"
            )

        self.q_to_shared = _identity_or_linear(self.native_attn_dim, self.shared_attn_dim)
        self.k_to_shared = _identity_or_linear(self.native_attn_dim, self.shared_attn_dim)
        self.v_to_shared = _identity_or_linear(self.native_attn_dim, self.shared_attn_dim)
        self.attn_from_shared = _identity_or_linear(self.shared_attn_dim, self.native_attn_dim)
        self.condition_adapter = nn.Linear(self.hidden_dim, self.condition_dim, bias=False)
        nn.init.xavier_uniform_(self.condition_adapter.weight)

        if self.use_gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable()
            if hasattr(self.qwen.config, "use_cache"):
                self.qwen.config.use_cache = False

        if freeze_qwen:
            self.qwen.requires_grad_(False)
        elif freeze_vision:
            self.qwen.model.visual.requires_grad_(False)

        self.to(device=self.device, dtype=torch_dtype)

        logger.info(
            "Initialized Qwen3VLExpert path=%s text_layers=%d mot_layers=%d "
            "native_heads=%d native_dim=%d shared_heads=%d shared_dim=%d condition_dim=%d "
            "condition_target_rms=%.4f gradient_checkpointing=%s freeze_qwen=%s freeze_vision=%s",
            self.model_path,
            len(self.layers),
            self.num_layers,
            self.native_num_heads,
            self.native_attn_dim,
            self.shared_num_heads,
            self.shared_attn_dim,
            self.condition_dim,
            self.condition_target_rms,
            self.use_gradient_checkpointing,
            freeze_qwen,
            freeze_vision,
        )

    def apply_training_freeze_policy(self, *, disable_shared_adapters: bool) -> None:
        """Reapply configured freezes after a trainer enables the whole MoT."""
        if self.freeze_qwen:
            self.qwen.requires_grad_(False)
        elif self.freeze_vision:
            self.qwen.model.visual.requires_grad_(False)
        if disable_shared_adapters:
            for module in (
                self.q_to_shared,
                self.k_to_shared,
                self.v_to_shared,
                self.attn_from_shared,
            ):
                module.requires_grad_(False)
        self.condition_adapter.requires_grad_(True)

    @staticmethod
    def _tensor_to_processor_groups(images: torch.Tensor) -> list[list[torch.Tensor]]:
        if images.ndim == 4:
            if images.shape[1] != 3:
                raise ValueError(f"`images` must be [B,3,H,W] or [B,K,3,H,W], got {tuple(images.shape)}")
            images = images.unsqueeze(1)
        elif images.ndim == 5:
            if images.shape[2] != 3:
                raise ValueError(f"`images` must be [B,K,3,H,W], got {tuple(images.shape)}")
        else:
            raise ValueError(f"`images` must be [B,3,H,W] or [B,K,3,H,W], got {tuple(images.shape)}")

        batch_size, num_images = int(images.shape[0]), int(images.shape[1])
        x = images.detach().to(dtype=torch.float32).clamp(-1.0, 1.0)
        x = ((x + 1.0) * 127.5).round().to(torch.uint8)
        return [
            [x[batch_idx, image_idx] for image_idx in range(num_images)]
            for batch_idx in range(batch_size)
        ]

    def _processor_to_device(self, batch: Any) -> dict[str, torch.Tensor]:
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        out = {}
        for key, value in dict(batch).items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device)
            else:
                out[key] = value
        return out

    @staticmethod
    def _build_image_first_conversations(
        processor_image_groups: Sequence[Sequence[torch.Tensor]],
        prompt_list: Sequence[str],
    ) -> list[list[dict[str, Any]]]:
        conversations = []
        for image_group, prompt in zip(processor_image_groups, prompt_list):
            if len(image_group) > 1:
                prompt = (
                    "The images are ordered from oldest to newest, and the last image is the current "
                    f"observation. {prompt}"
                )
            content = [{"type": "image", "image": image} for image in image_group]
            content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])
        return conversations

    @staticmethod
    def _insert_proprio_placeholders_after_language(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        proprio_attention_mask: torch.Tensor,
        *,
        vision_end_token_id: int,
        im_end_token_id: int,
        placeholder_token_id: int,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "`input_ids` and `attention_mask` must both be [B,S], "
                f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
            )
        if proprio_attention_mask.ndim != 2 or proprio_attention_mask.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "`proprio_attention_mask` must be [B,K], "
                f"got {tuple(proprio_attention_mask.shape)} for batch={input_ids.shape[0]}"
            )

        batch_size, old_seq_len = input_ids.shape
        num_proprio_tokens = int(proprio_attention_mask.shape[1])
        new_seq_len = old_seq_len + num_proprio_tokens
        new_input_ids = torch.full(
            (batch_size, new_seq_len),
            int(placeholder_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        new_attention_mask = torch.zeros(
            (batch_size, new_seq_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        text_pos_masks = torch.zeros(
            (batch_size, new_seq_len),
            dtype=torch.bool,
            device=input_ids.device,
        )
        proprio_pos_masks = torch.zeros_like(text_pos_masks)
        insert_positions = []
        language_start_positions = []
        valid = attention_mask.to(dtype=torch.bool)
        proprio_valid = proprio_attention_mask.to(device=input_ids.device, dtype=torch.bool)

        for batch_idx in range(batch_size):
            vision_end_positions = torch.nonzero(
                (input_ids[batch_idx] == int(vision_end_token_id)) & valid[batch_idx],
                as_tuple=False,
            ).flatten()
            if vision_end_positions.numel() == 0:
                raise ValueError(
                    "Image-first Qwen input must contain a valid vision-end token."
                )
            language_start = int(vision_end_positions[-1].item()) + 1
            im_end_positions = torch.nonzero(
                (input_ids[batch_idx] == int(im_end_token_id)) & valid[batch_idx],
                as_tuple=False,
            ).flatten()
            im_end_positions = im_end_positions[im_end_positions >= language_start]
            if im_end_positions.numel() == 0:
                raise ValueError(
                    "Image-first Qwen input must end its language span with a valid im-end token."
                )
            insert_at = int(im_end_positions[-1].item())
            if insert_at <= language_start:
                raise ValueError("Image-first Qwen input contains no language tokens after the images.")
            insert_positions.append(insert_at)
            language_start_positions.append(language_start)

            new_input_ids[batch_idx, :insert_at] = input_ids[batch_idx, :insert_at]
            new_input_ids[batch_idx, insert_at + num_proprio_tokens :] = input_ids[batch_idx, insert_at:]
            new_attention_mask[batch_idx, :insert_at] = attention_mask[batch_idx, :insert_at]
            new_attention_mask[batch_idx, insert_at : insert_at + num_proprio_tokens] = (
                proprio_valid[batch_idx].to(dtype=attention_mask.dtype)
            )
            new_attention_mask[batch_idx, insert_at + num_proprio_tokens :] = attention_mask[batch_idx, insert_at:]

            text_pos_masks[batch_idx, language_start:insert_at] = valid[
                batch_idx, language_start:insert_at
            ]
            proprio_pos_masks[batch_idx, insert_at : insert_at + num_proprio_tokens] = proprio_valid[batch_idx]

        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "text_pos_masks": text_pos_masks,
            "proprio_pos_masks": proprio_pos_masks,
            "insert_positions": insert_positions,
            "language_start_positions": language_start_positions,
        }

    def pre_dit(
        self,
        images: torch.Tensor,
        prompts: str | Sequence[str],
        proprio_tokens: Optional[torch.Tensor] = None,
        proprio_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        batch_size = int(images.shape[0])
        prompt_list = _as_prompt_list(prompts, batch_size=batch_size)
        processor_image_groups = self._tensor_to_processor_groups(images)
        conversations = self._build_image_first_conversations(
            processor_image_groups,
            prompt_list,
        )
        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        encoded = self._processor_to_device(encoded)

        model = self.qwen.model
        language_model = model.language_model
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if proprio_tokens is None:
            if proprio_attention_mask is not None:
                raise ValueError("`proprio_attention_mask` was provided without `proprio_tokens`.")
            proprio_tokens = torch.empty(
                (batch_size, 0, self.hidden_dim),
                device=input_ids.device,
                dtype=self.torch_dtype,
            )
            proprio_attention_mask = torch.empty(
                (batch_size, 0),
                device=input_ids.device,
                dtype=torch.bool,
            )
        else:
            if proprio_tokens.ndim != 3 or proprio_tokens.shape[0] != batch_size:
                raise ValueError(
                    "`proprio_tokens` must be [B,K,D], "
                    f"got {tuple(proprio_tokens.shape)} for batch={batch_size}"
                )
            if proprio_tokens.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"`proprio_tokens` hidden dim must be {self.hidden_dim}, got {proprio_tokens.shape[2]}"
                )
            if proprio_attention_mask is None:
                proprio_attention_mask = torch.ones(
                    proprio_tokens.shape[:2],
                    device=input_ids.device,
                    dtype=torch.bool,
                )
            elif proprio_attention_mask.shape != proprio_tokens.shape[:2]:
                raise ValueError(
                    "`proprio_attention_mask` must match proprio [B,K], "
                    f"got {tuple(proprio_attention_mask.shape)} vs {tuple(proprio_tokens.shape[:2])}"
                )
            proprio_tokens = proprio_tokens.to(device=input_ids.device)
            proprio_attention_mask = proprio_attention_mask.to(device=input_ids.device, dtype=torch.bool)

        vision_end_token_id = int(self.qwen.config.vision_end_token_id)
        im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id is None or int(im_end_token_id) < 0:
            raise ValueError("Qwen tokenizer must define the <|im_end|> token.")
        placeholder_token_id = self.processor.tokenizer.pad_token_id
        if placeholder_token_id is None:
            placeholder_token_id = self.processor.tokenizer.eos_token_id
        if placeholder_token_id is None:
            raise ValueError("Qwen tokenizer must define a pad or eos token for proprio placeholders.")
        layout = self._insert_proprio_placeholders_after_language(
            input_ids,
            attention_mask,
            proprio_attention_mask,
            vision_end_token_id=vision_end_token_id,
            im_end_token_id=int(im_end_token_id),
            placeholder_token_id=int(placeholder_token_id),
        )
        input_ids = layout["input_ids"]
        attention_mask = layout["attention_mask"]
        text_pos_masks = layout["text_pos_masks"]
        proprio_pos_masks = layout["proprio_pos_masks"]
        insert_positions = layout["insert_positions"]

        mm_token_type_ids = encoded.get("mm_token_type_ids")
        if mm_token_type_ids is not None and proprio_tokens.shape[1] > 0:
            old_mm_token_type_ids = mm_token_type_ids
            mm_token_type_ids = torch.zeros(
                input_ids.shape,
                dtype=old_mm_token_type_ids.dtype,
                device=old_mm_token_type_ids.device,
            )
            num_proprio_tokens = int(proprio_tokens.shape[1])
            for batch_idx, insert_at in enumerate(insert_positions):
                mm_token_type_ids[batch_idx, :insert_at] = old_mm_token_type_ids[batch_idx, :insert_at]
                mm_token_type_ids[batch_idx, insert_at + num_proprio_tokens :] = old_mm_token_type_ids[
                    batch_idx, insert_at:
                ]
        inputs_embeds = language_model.embed_tokens(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")

        if pixel_values is not None:
            image_outputs = model.get_image_features(
                pixel_values,
                image_grid_thw,
            )
            image_pooler = getattr(image_outputs, "pooler_output", None)
            if image_pooler is None:
                image_pooler = image_outputs[0]
            if isinstance(image_pooler, torch.Tensor):
                image_embeds_raw = image_pooler
            else:
                image_embeds_raw = torch.cat(image_pooler, dim=0)
            image_embeds = image_embeds_raw.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            image_mask, _ = model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = getattr(image_outputs, "deepstack_features", None)
            if deepstack_visual_embeds is None and len(image_outputs) > 1:
                deepstack_visual_embeds = image_outputs[1]

        if proprio_tokens.shape[1] > 0:
            inputs_embeds = inputs_embeds.clone()
            num_proprio_tokens = int(proprio_tokens.shape[1])
            proprio_tokens = proprio_tokens.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            for batch_idx, insert_at in enumerate(insert_positions):
                inputs_embeds[batch_idx, insert_at : insert_at + num_proprio_tokens] = proprio_tokens[batch_idx]

        if hasattr(model, "compute_3d_position_ids"):
            position_ids = model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=mm_token_type_ids,
            )
        elif hasattr(model, "get_rope_index"):
            position_ids, rope_deltas = model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            model.rope_deltas = rope_deltas
        else:
            position_ids = None
        if position_ids is None:
            seq = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = seq.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            ).unsqueeze(-1)

        position_embeddings = language_model.rotary_emb(inputs_embeds, rope_position_ids)
        return {
            "tokens": inputs_embeds,
            "position_embeddings": position_embeddings,
            "text_position_ids": text_position_ids,
            "rope_position_ids": rope_position_ids,
            "attention_mask": attention_mask,
            "visual_pos_masks": visual_pos_masks,
            "text_pos_masks": text_pos_masks,
            "proprio_pos_masks": proprio_pos_masks,
            "downstream_visible_masks": text_pos_masks | proprio_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "meta": {
                "batch_size": batch_size,
                "seq_len": int(inputs_embeds.shape[1]),
                "num_proprio_tokens": int(proprio_tokens.shape[1]),
                "proprio_insert_positions": insert_positions,
            },
        }

    def _repeat_kv_to_native_heads(self, tensor: torch.Tensor, layer) -> torch.Tensor:
        num_groups = int(layer.self_attn.num_key_value_groups)
        if num_groups == 1:
            return tensor
        batch, num_kv_heads, seq_len, head_dim = tensor.shape
        tensor = tensor[:, :, None, :, :].expand(batch, num_kv_heads, num_groups, seq_len, head_dim)
        return tensor.reshape(batch, num_kv_heads * num_groups, seq_len, head_dim)

    def build_condition_tokens(
        self,
        normalized_tokens: torch.Tensor,
        pre_state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact language/state tokens and adapt them from 2048 to 4096.

        Image tokens stay private to Qwen.  The returned boolean mask is
        sample-aligned and can be broadcast over any DiT query length.
        """
        if normalized_tokens.ndim != 3 or normalized_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "`normalized_tokens` must be [B,S,QwenD], got "
                f"{tuple(normalized_tokens.shape)} with QwenD={self.hidden_dim}"
            )
        visible = pre_state["downstream_visible_masks"].to(
            device=normalized_tokens.device,
            dtype=torch.bool,
        )
        attention_mask = pre_state.get("attention_mask")
        if attention_mask is not None:
            visible = visible & attention_mask.to(
                device=normalized_tokens.device,
                dtype=torch.bool,
            )
        if visible.shape != normalized_tokens.shape[:2]:
            raise ValueError(
                "Qwen language/state mask does not align with tokens: "
                f"mask={tuple(visible.shape)}, tokens={tuple(normalized_tokens.shape[:2])}"
            )

        counts = visible.sum(dim=1)
        max_context = int(counts.max().item())
        if max_context <= 0 or bool((counts <= 0).any()):
            raise ValueError("Every sample requires at least one visible language/state token.")
        batch_size = int(normalized_tokens.shape[0])
        compact = normalized_tokens.new_zeros((batch_size, max_context, self.hidden_dim))
        compact_valid = torch.zeros(
            (batch_size, max_context),
            device=normalized_tokens.device,
            dtype=torch.bool,
        )
        for batch_idx in range(batch_size):
            selected = normalized_tokens[batch_idx, visible[batch_idx]]
            count = int(selected.shape[0])
            compact[batch_idx, :count] = selected
            compact_valid[batch_idx, :count] = True

        adapted = self.condition_adapter(compact)
        # The pretrained Wan/Action cross-attention path expects the small T5
        # encoder-state scale.  Calibrate each sample/layer without changing
        # token directions; gradients still flow through the trainable adapter.
        adapted_float = adapted.float()
        valid_float = compact_valid[:, :, None].to(dtype=adapted_float.dtype)
        element_count = (
            compact_valid.sum(dim=1).to(dtype=adapted_float.dtype) * self.condition_dim
        ).clamp_min(1.0)
        rms = (
            (adapted_float.square() * valid_float).sum(dim=(1, 2)) / element_count
        ).clamp_min(1e-12).sqrt()
        scale = (self.condition_target_rms / rms)[:, None, None]
        adapted = (adapted_float * scale * valid_float).to(dtype=normalized_tokens.dtype)
        return adapted, compact_valid

    def run_native_self_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run Qwen causal self-attention without Video/Action MoT tokens."""
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
            raise ValueError("Qwen native Q/K/V must be matching [B,S,D] tensors.")
        batch_size, seq_len, _ = q.shape
        valid = attention_mask.to(device=q.device, dtype=torch.bool)
        if valid.shape != (batch_size, seq_len):
            raise ValueError(
                f"Qwen attention mask mismatch: {tuple(valid.shape)} vs {(batch_size, seq_len)}"
            )
        q = q.view(batch_size, seq_len, self.native_num_heads, self.native_attn_head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.native_num_heads, self.native_attn_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.native_num_heads, self.native_attn_head_dim).transpose(1, 2)
        causal = torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool).tril()
        mask = valid[:, None, :, None] & valid[:, None, None, :] & causal[None, None]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(batch_size, seq_len, self.native_attn_dim)
        return output * valid[:, :, None].to(dtype=output.dtype)

    def final_condition_input(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return Qwen's final normalized residual stream for DiT tail layers."""
        return self.qwen.model.language_model.norm(tokens)

    def build_attention_io(
        self,
        layer_idx: int,
        tokens: torch.Tensor,
        pre_state: dict[str, Any],
        *,
        project_to_shared: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if layer_idx >= self.num_layers:
            raise ValueError(f"Qwen layer_idx={layer_idx} exceeds num_layers={self.num_layers}")

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        layer = self.layers[layer_idx]
        attn = layer.self_attn
        residual = tokens
        hidden_states = layer.input_layernorm(tokens)
        batch_size, seq_len, _ = hidden_states.shape

        q = attn.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.native_num_heads,
            self.native_attn_head_dim,
        ).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.native_num_key_value_heads,
            self.native_attn_head_dim,
        ).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.native_num_key_value_heads,
            self.native_attn_head_dim,
        ).transpose(1, 2)

        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, *pre_state["position_embeddings"])
        k = self._repeat_kv_to_native_heads(k, layer)
        v = self._repeat_kv_to_native_heads(v, layer)

        q = q.transpose(1, 2).reshape(batch_size, seq_len, self.native_attn_dim)
        k = k.transpose(1, 2).reshape(batch_size, seq_len, self.native_attn_dim)
        v = v.transpose(1, 2).reshape(batch_size, seq_len, self.native_attn_dim)
        cache = {
            "layer": layer,
            "residual": residual,
            "normalized_tokens": hidden_states,
            "pre_state": pre_state,
        }
        if not project_to_shared:
            return q, k, v, cache
        return self.q_to_shared(q), self.k_to_shared(k), self.v_to_shared(v), cache

    def apply_post_attention(
        self,
        layer_idx: int,
        mixed_slice: torch.Tensor,
        cache: dict[str, Any],
        *,
        from_shared: bool = True,
    ) -> torch.Tensor:
        layer = cache["layer"]
        residual = cache["residual"]
        pre_state = cache["pre_state"]

        native_attn = self.attn_from_shared(mixed_slice) if from_shared else mixed_slice
        hidden_states = residual + layer.self_attn.o_proj(native_attn)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)

        deepstack_visual_embeds = pre_state.get("deepstack_visual_embeds")
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden_states = self.qwen.model.language_model._deepstack_process(
                hidden_states,
                pre_state["visual_pos_masks"],
                deepstack_visual_embeds[layer_idx],
            )
        return hidden_states
