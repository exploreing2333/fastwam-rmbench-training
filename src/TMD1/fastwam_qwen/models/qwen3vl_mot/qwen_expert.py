from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from TMD1.fastwam.utils.logging_config import get_logger

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
    """Qwen3-VL text/vision expert adapted to FastWAM mixed attention.

    Qwen3-VL has a different native attention width/head count than Wan DiT.
    The selected decoder layers are exposed through projection
    adapters so their Q/K/V can participate in the shared Wan MoT attention
    space while Qwen's residual, output projection and MLP stay native.  Passing
    ``num_mot_layers=None`` uses all language decoder layers in the local Qwen
    checkpoint.
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

        if freeze_qwen:
            self.qwen.requires_grad_(False)
        elif freeze_vision:
            self.qwen.model.visual.requires_grad_(False)

        self.to(device=self.device, dtype=torch_dtype)

        logger.info(
            "Initialized Qwen3VLExpert path=%s text_layers=%d mot_layers=%d "
            "native_heads=%d native_dim=%d shared_heads=%d shared_dim=%d freeze_qwen=%s freeze_vision=%s",
            self.model_path,
            len(self.layers),
            self.num_layers,
            self.native_num_heads,
            self.native_attn_dim,
            self.shared_num_heads,
            self.shared_attn_dim,
            freeze_qwen,
            freeze_vision,
        )

    @staticmethod
    def _tensor_to_pil_batch(images: torch.Tensor) -> list[Image.Image]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"`images` must be [B,3,H,W], got {tuple(images.shape)}")
        x = images.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
        x = ((x + 1.0) * 127.5).round().to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous().numpy()
        return [Image.fromarray(arr, mode="RGB") for arr in x]

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

    def pre_dit(self, images: torch.Tensor, prompts: str | Sequence[str]) -> dict[str, Any]:
        batch_size = int(images.shape[0])
        prompt_list = _as_prompt_list(prompts, batch_size=batch_size)
        pil_images = self._tensor_to_pil_batch(images)
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image, prompt in zip(pil_images, prompt_list)
        ]
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
        inputs_embeds = language_model.embed_tokens(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")
        mm_token_type_ids = encoded.get("mm_token_type_ids")

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
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "meta": {
                "batch_size": batch_size,
                "seq_len": int(inputs_embeds.shape[1]),
            },
        }

    def _repeat_kv_to_native_heads(self, tensor: torch.Tensor, layer) -> torch.Tensor:
        num_groups = int(layer.self_attn.num_key_value_groups)
        if num_groups == 1:
            return tensor
        batch, num_kv_heads, seq_len, head_dim = tensor.shape
        tensor = tensor[:, :, None, :, :].expand(batch, num_kv_heads, num_groups, seq_len, head_dim)
        return tensor.reshape(batch, num_kv_heads * num_groups, seq_len, head_dim)

    def build_attention_io(
        self,
        layer_idx: int,
        tokens: torch.Tensor,
        pre_state: dict[str, Any],
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
            "pre_state": pre_state,
        }
        return self.q_to_shared(q), self.k_to_shared(k), self.v_to_shared(v), cache

    def apply_post_attention(
        self,
        layer_idx: int,
        mixed_slice: torch.Tensor,
        cache: dict[str, Any],
    ) -> torch.Tensor:
        layer = cache["layer"]
        residual = cache["residual"]
        pre_state = cache["pre_state"]

        native_attn = self.attn_from_shared(mixed_slice)
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
