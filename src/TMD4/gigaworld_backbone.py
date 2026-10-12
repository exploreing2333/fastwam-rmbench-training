from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)


class _Projection(nn.Module):
    """Two-layer projection with the parameter names used by Diffusers."""

    def __init__(
        self,
        linear_1: nn.Linear,
        linear_2: nn.Linear,
        activation: nn.Module,
    ):
        super().__init__()
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(inputs)))


class GigaWorldConditionEmbedder(nn.Module):
    """Parameter-compatible view of the Wan text/timestep embeddings."""

    def __init__(self, expert: nn.Module):
        super().__init__()
        self.text_embedder = _Projection(
            expert.text_embedding[0],
            expert.text_embedding[2],
            expert.text_embedding[1],
        )
        self.time_embedder = _Projection(
            expert.time_embedding[0],
            expert.time_embedding[2],
            expert.time_embedding[1],
        )
        self.time_proj = expert.time_projection[1]

    def project_time(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.time_proj(F.silu(embedding))


class _FeedForwardInput(nn.Module):
    def __init__(self, projection: nn.Linear, activation: nn.Module):
        super().__init__()
        self.proj = projection
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(inputs))


class GigaWorldFeedForward(nn.Module):
    """Wan FFN with ``net.0.proj`` / ``net.2`` checkpoint names."""

    def __init__(self, ffn: nn.Sequential):
        super().__init__()
        self.net = nn.ModuleList(
            [
                _FeedForwardInput(ffn[0], ffn[1]),
                nn.Identity(),
                ffn[2],
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.net[0](inputs)
        hidden = self.net[1](hidden)
        return self.net[2](hidden)


class GigaWorldAttention(nn.Module):
    """Wan attention with the public GigaWorld/Diffusers key layout."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.to_q = attention.q
        self.to_k = attention.k
        self.to_v = attention.v
        self.to_out = nn.ModuleList([attention.o, nn.Dropout(0.0)])
        self.norm_q = attention.norm_q
        self.norm_k = attention.norm_k
        self.heads = int(attention.num_heads)
        self.num_heads = self.heads
        self.inner_dim = int(attention.attn_hidden_dim)

    # Non-registering aliases keep the existing TMD attention implementation
    # usable while the serialized parameter names remain GigaWorld-native.
    @property
    def q(self) -> nn.Linear:
        return self.to_q

    @property
    def k(self) -> nn.Linear:
        return self.to_k

    @property
    def v(self) -> nn.Linear:
        return self.to_v

    @property
    def o(self) -> nn.Linear:
        return self.to_out[0]


class GigaWorldExpertBlock(nn.Module):
    """One visual/action expert block using public checkpoint names."""

    _gigaworld_layout = True

    def __init__(self, block: nn.Module):
        super().__init__()
        self.norm1 = block.norm1
        self.attn1 = GigaWorldAttention(block.self_attn)
        self.attn2 = GigaWorldAttention(block.cross_attn)
        # GigaWorld/Diffusers calls the affine cross-attention norm ``norm2``
        # and the non-affine MLP norm ``norm3``. FastWAM historically used the
        # opposite serialized names.
        self.norm2 = block.norm3
        self.ffn = GigaWorldFeedForward(block.ffn)
        self.norm3 = block.norm2
        self.scale_shift_table = block.modulation
        self.gate = block.gate
        self.num_heads = int(block.num_heads)
        self.attn_head_dim = int(block.attn_head_dim)

    @property
    def self_attn(self) -> GigaWorldAttention:
        return self.attn1

    @property
    def cross_attn(self) -> GigaWorldAttention:
        return self.attn2

    @property
    def modulation(self) -> nn.Parameter:
        return self.scale_shift_table


class GigaWorldMoTBlock(nn.Module):
    def __init__(self, visual_block: nn.Module, action_block: nn.Module):
        super().__init__()
        self.action_expert = GigaWorldExpertBlock(action_block)
        self.visual_expert = GigaWorldExpertBlock(visual_block)


class GigaWorldActionEncoder(nn.Module):
    """Official three-layer Action encoder without embodiment routing."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(int(action_dim), 128)
        self.mid_proj = nn.Linear(128, 256)
        self.out_proj = nn.Linear(256, int(hidden_dim))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.in_proj(action))
        hidden = F.gelu(self.mid_proj(hidden))
        return self.out_proj(hidden)


class GigaWorldActionDecoder(nn.Module):
    """Official three-layer Action decoder without embodiment routing."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(int(hidden_dim), 256)
        self.mid_proj = nn.Linear(256, 128)
        self.out_proj = nn.Linear(128, int(action_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.in_proj(hidden_states))
        hidden = F.gelu(self.mid_proj(hidden))
        return self.out_proj(hidden)


class _ExpertProxy:
    """Unregistered execution view over the unified GigaWorld backbone."""

    def __init__(self, owner: nn.Module, legacy_wrapper: nn.Module, kind: str):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_legacy_wrapper", legacy_wrapper)
        object.__setattr__(self, "_legacy_base", legacy_wrapper.base)
        object.__setattr__(self, "_kind", str(kind))

    @property
    def base(self) -> "_ExpertProxy":
        return self

    @property
    def blocks(self) -> list[nn.Module]:
        name = "visual_expert" if self._kind == "video" else "action_expert"
        return [getattr(block, name) for block in self._owner.blocks]

    @property
    def training(self) -> bool:
        return bool(self._owner.training)

    @property
    def use_gradient_checkpointing(self) -> bool:
        return bool(self._legacy_base.use_gradient_checkpointing)

    @use_gradient_checkpointing.setter
    def use_gradient_checkpointing(self, enabled: bool) -> None:
        self._legacy_base.use_gradient_checkpointing = bool(enabled)

    def text_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        embedder = (
            self._owner.condition_embedder
            if self._kind == "video"
            else self._owner.action_condition_embedder
        )
        return embedder.text_embedder(inputs)

    def time_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        embedder = (
            self._owner.condition_embedder
            if self._kind == "video"
            else self._owner.action_condition_embedder
        )
        return embedder.time_embedder(inputs)

    def time_projection(self, inputs: torch.Tensor) -> torch.Tensor:
        embedder = (
            self._owner.condition_embedder
            if self._kind == "video"
            else self._owner.action_condition_embedder
        )
        return embedder.project_time(inputs)

    def pre_dit(self, *args, **kwargs):
        if self._kind == "video":
            # The legacy wrapper contains only execution logic. Its parameter
            # modules are the same objects now registered under official keys.
            self._legacy_base.train(self._owner.training)
            return self._legacy_wrapper.pre_dit(*args, **kwargs)

        action_tokens = kwargs.pop("action_tokens", args[0] if args else None)
        timestep = kwargs.pop(
            "timestep",
            args[1] if len(args) > 1 else None,
        )
        if kwargs:
            raise TypeError(f"Unexpected Action pre_dit arguments: {sorted(kwargs)}")
        if action_tokens is None or timestep is None:
            raise TypeError("Action pre_dit requires action_tokens and timestep.")
        if action_tokens.ndim != 3:
            raise ValueError(
                f"action_tokens must be [B,T,D], got {tuple(action_tokens.shape)}"
            )
        if int(action_tokens.shape[-1]) != int(self.action_dim):
            raise ValueError(
                f"Action width must be {self.action_dim}, got {action_tokens.shape[-1]}."
            )
        batch_size = int(action_tokens.shape[0])
        if timestep.ndim != 1 or int(timestep.shape[0]) not in (1, batch_size):
            raise ValueError("Action timestep must be [1] or [B].")
        if int(timestep.shape[0]) == 1 and batch_size > 1:
            if self.training:
                raise ValueError("Training requires one Action timestep per sample.")
            timestep = timestep.expand(batch_size)
        sequence_length = int(action_tokens.shape[1])
        if sequence_length > int(self.freqs.shape[0]):
            raise ValueError("Action sequence exceeds the RoPE cache.")

        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        time_modulation = self.time_projection(time_embedding).unflatten(
            1,
            (6, self.hidden_dim),
        )
        tokens = self._owner.action_encoder(action_tokens)
        frequencies = self.freqs[:sequence_length].view(
            sequence_length,
            1,
            -1,
        ).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": frequencies,
            "t": time_embedding,
            "t_mod": time_modulation,
            "context": None,
            "context_mask": None,
            "meta": {
                "batch_size": batch_size,
                "seq_len": sequence_length,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]):
        if self._kind == "video":
            return self._legacy_wrapper.post_dit(tokens, pre_state)
        timestep_embedding = pre_state["t"]
        shift, scale = (
            self._owner.action_scale_shift_table.to(
                device=timestep_embedding.device,
                dtype=timestep_embedding.dtype,
            )
            + timestep_embedding.unsqueeze(1)
        ).chunk(2, dim=1)
        hidden = self._owner.action_norm_out(tokens.float())
        hidden = hidden * (1 + scale.squeeze(1))[:, None]
        hidden = hidden + shift.squeeze(1)[:, None]
        return self._owner.action_decoder(hidden.to(tokens.dtype))

    def __getattr__(self, name: str):
        return getattr(self._legacy_base, name)


def _legacy_block_key(key: str, expert_name: str) -> Optional[str]:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "blocks":
        return None
    layer, suffix = parts[1], parts[2]
    suffix = suffix.replace("modulation", "scale_shift_table")
    suffix = suffix.replace("self_attn.", "attn1.")
    suffix = suffix.replace("cross_attn.", "attn2.")
    suffix = suffix.replace(".q.", ".to_q.")
    suffix = suffix.replace(".k.", ".to_k.")
    suffix = suffix.replace(".v.", ".to_v.")
    suffix = suffix.replace(".o.", ".to_out.0.")
    suffix = suffix.replace("ffn.0.", "ffn.net.0.proj.")
    suffix = suffix.replace("ffn.2.", "ffn.net.2.")
    if suffix.startswith("norm3."):
        suffix = "norm2." + suffix[len("norm3.") :]
    return f"blocks.{layer}.{expert_name}.{suffix}"


def legacy_tmd4_key_to_gigaworld(key: str) -> Optional[str]:
    video_prefix = "mixtures.video.base."
    action_prefix = "mixtures.action.base."
    if key.startswith(video_prefix):
        suffix = key[len(video_prefix) :]
        if suffix.startswith("blocks."):
            return _legacy_block_key(suffix, "visual_expert")
        replacements = (
            ("patch_embedding.", "patch_embedding."),
            ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
            ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
            ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
            ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
            ("time_projection.1.", "condition_embedder.time_proj."),
            ("head.modulation", "scale_shift_table"),
            ("head.head.", "proj_out."),
        )
    elif key.startswith(action_prefix):
        suffix = key[len(action_prefix) :]
        if suffix.startswith("blocks."):
            return _legacy_block_key(suffix, "action_expert")
        # The old single Linear Action I/O cannot be represented by the new
        # three-layer MLP and is intentionally reinitialized.
        if suffix.startswith("action_encoder.") or suffix.startswith("head."):
            return None
        replacements = (
            ("text_embedding.0.", "action_condition_embedder.text_embedder.linear_1."),
            ("text_embedding.2.", "action_condition_embedder.text_embedder.linear_2."),
            ("time_embedding.0.", "action_condition_embedder.time_embedder.linear_1."),
            ("time_embedding.2.", "action_condition_embedder.time_embedder.linear_2."),
            ("time_projection.1.", "action_condition_embedder.time_proj."),
        )
    else:
        return key

    for old, new in replacements:
        if suffix == old or suffix.startswith(old):
            return new + suffix[len(old) :]
    return None


def adopt_gigaworld_layout(
    mot: nn.Module,
    video_expert: nn.Module,
    action_expert: nn.Module,
) -> tuple[_ExpertProxy, _ExpertProxy]:
    """Re-register TMD4's backbone under the public GigaWorld key layout."""

    video_base = video_expert.base
    action_base = action_expert.base
    if len(video_base.blocks) != len(action_base.blocks):
        raise ValueError("GigaWorld layout requires equal visual/action depth.")

    # Remove the legacy registered ownership before attaching the same
    # parameters below their public checkpoint names.
    del mot.mixtures
    mot.patch_embedding = video_base.patch_embedding
    mot.condition_embedder = GigaWorldConditionEmbedder(video_base)
    mot.action_condition_embedder = GigaWorldConditionEmbedder(action_base)
    mot.blocks = nn.ModuleList(
        [
            GigaWorldMoTBlock(visual_block, action_block)
            for visual_block, action_block in zip(
                video_base.blocks,
                action_base.blocks,
            )
        ]
    )
    mot.norm_out = video_base.head.norm
    mot.scale_shift_table = video_base.head.modulation
    mot.proj_out = video_base.head.head

    action_dim = int(action_base.action_dim)
    action_hidden_dim = int(action_base.hidden_dim)
    mot.action_encoder = GigaWorldActionEncoder(action_dim, action_hidden_dim)
    mot.action_norm_out = nn.LayerNorm(
        action_hidden_dim,
        eps=float(action_base.blocks[0].norm1.eps),
        elementwise_affine=False,
    )
    mot.action_scale_shift_table = nn.Parameter(
        torch.randn(1, 2, action_hidden_dim) / action_hidden_dim**0.5
    )
    mot.action_decoder = GigaWorldActionDecoder(action_hidden_dim, action_dim)

    video_proxy = _ExpertProxy(mot, video_expert, "video")
    action_proxy = _ExpertProxy(mot, action_expert, "action")
    mot.mixtures = {"video": video_proxy, "action": action_proxy}
    return video_proxy, action_proxy


def _checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    index_path = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        names = sorted(set(index["weight_map"].values()))
        files = [path / name for name in names]
    else:
        files = sorted(path.glob("*.safetensors"))
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GigaWorld checkpoint shards: {missing}")
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {path}")
    return files


def _adapt_action_io_tensor(
    key: str,
    source: torch.Tensor,
    target_shape: tuple[int, ...],
) -> Optional[torch.Tensor]:
    """Collapse GigaWorld's first embodiment into a regular Linear layer.

    The public boundary layers serialize their weights as
    ``[embodiment, in, out]``. TMD4 deliberately keeps one embodiment and a
    task-native action width, so select embodiment zero (the same leading
    slice convention as GigaWorld's own preload helper), transpose to the
    PyTorch Linear layout, and retain the leading action channels.
    """

    weight_keys = {
        "action_encoder.in_proj.weight",
        "action_decoder.out_proj.weight",
    }
    bias_keys = {
        "action_encoder.in_proj.bias",
        "action_decoder.out_proj.bias",
    }
    if key in weight_keys and source.ndim == 3:
        candidate = source[0].transpose(0, 1)
    elif key in bias_keys and source.ndim == 2:
        candidate = source[0]
    else:
        return None
    if candidate.ndim != len(target_shape) or any(
        source_dim < target_dim
        for source_dim, target_dim in zip(candidate.shape, target_shape)
    ):
        return None
    slices = tuple(slice(0, size) for size in target_shape)
    return candidate[slices].contiguous()


@torch.no_grad()
def load_gigaworld_checkpoint(
    module: nn.Module,
    path: str | Path,
) -> dict[str, Any]:
    """Load same-name, same-shape public weights without a key conversion."""

    from safetensors import safe_open

    checkpoint_path = Path(path).expanduser().resolve()
    files = _checkpoint_files(checkpoint_path)
    parameters = dict(module.named_parameters())
    loaded: list[str] = []
    adapted: dict[str, dict[str, Any]] = {}
    shape_mismatch: dict[str, dict[str, tuple[int, ...]]] = {}
    unexpected: list[str] = []

    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                target = parameters.get(key)
                if target is None:
                    unexpected.append(key)
                    continue
                source_shape = tuple(shard.get_slice(key).get_shape())
                target_shape = tuple(target.shape)
                if source_shape != target_shape:
                    value = shard.get_tensor(key)
                    value = _adapt_action_io_tensor(key, value, target_shape)
                    if value is None:
                        shape_mismatch[key] = {
                            "checkpoint": source_shape,
                            "model": target_shape,
                        }
                        continue
                    adapted[key] = {
                        "checkpoint": source_shape,
                        "model": target_shape,
                        "embodiment_index": 0,
                    }
                else:
                    value = shard.get_tensor(key)
                target.copy_(value.to(device=target.device, dtype=target.dtype))
                loaded.append(key)

    loaded_set = set(loaded)
    essential_prefixes = (
        "patch_embedding.",
        "condition_embedder.",
        "blocks.",
        "scale_shift_table",
        "proj_out.",
        "action_condition_embedder.",
        "action_scale_shift_table",
    )
    essential = {
        key
        for key in parameters
        if any(key == prefix or key.startswith(prefix) for prefix in essential_prefixes)
    }
    missing_essential = sorted(essential - loaded_set)
    if missing_essential:
        raise RuntimeError(
            "GigaWorld checkpoint did not cover the complete TMD4 backbone: "
            f"{missing_essential[:20]}"
        )

    report = {
        "path": str(checkpoint_path),
        "files": [str(file) for file in files],
        "loaded_tensors": len(loaded),
        "loaded_parameters": sum(parameters[key].numel() for key in loaded),
        "adapted_tensors": adapted,
        "shape_mismatch": shape_mismatch,
        "unexpected_tensors": sorted(unexpected),
    }
    logger.info(
        "Loaded GigaWorld-Policy-0.5 directly by native keys: tensors=%d "
        "parameters=%d adapted=%d shape_mismatch=%d unused=%d path=%s",
        report["loaded_tensors"],
        report["loaded_parameters"],
        len(adapted),
        len(shape_mismatch),
        len(unexpected),
        checkpoint_path,
    )
    if shape_mismatch:
        logger.warning(
            "Skipped GigaWorld tensors with incompatible shapes: %s",
            shape_mismatch,
        )
    return report


def convert_legacy_tmd4_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Migrate old TMD4 checkpoints after the one-time key-layout change."""

    converted: dict[str, torch.Tensor] = {}
    legacy = any(key.startswith("mixtures.") for key in state_dict)
    if not legacy:
        return dict(state_dict)
    for key, value in state_dict.items():
        mapped = legacy_tmd4_key_to_gigaworld(key)
        if mapped is not None:
            converted[mapped] = value

    # The legacy single-linear Action I/O has no exact representation in the
    # requested three-layer architecture. Preserve the freshly initialized
    # values so strict legacy checkpoint loading remains operational.
    current = module.state_dict()
    for prefix in (
        "action_encoder.",
        "action_decoder.",
        "action_scale_shift_table",
    ):
        for key, value in current.items():
            if key == prefix or key.startswith(prefix):
                converted.setdefault(key, value)
    logger.warning(
        "Migrated a legacy TMD4 checkpoint to GigaWorld-native keys; the new "
        "three-layer Action I/O remains freshly initialized."
    )
    return converted


__all__ = [
    "adopt_gigaworld_layout",
    "convert_legacy_tmd4_state_dict",
    "legacy_tmd4_key_to_gigaworld",
    "load_gigaworld_checkpoint",
]
