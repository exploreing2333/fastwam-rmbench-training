from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import (
    sinusoidal_embedding_1d,
)
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class _ContextFreeExpertWrapper(nn.Module):
    """Expose a DiT expert without the legacy text-context interface."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)


class ContextFreeWanVideoExpert(_ContextFreeExpertWrapper):
    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        num_condition_latent_frames: int = 1,
        condition_timestep: Optional[torch.Tensor] = None,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        base = self.base
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
        if timestep.ndim not in (1, 2):
            raise ValueError(
                "`timestep` must be [B], [1], or per noisy frame "
                f"[B,F], got {tuple(timestep.shape)}"
            )

        batch_size = x.shape[0]
        num_latent_frames = x.shape[2]
        num_condition_latent_frames = int(num_condition_latent_frames)
        if num_condition_latent_frames <= 0:
            raise ValueError(
                f"`num_condition_latent_frames` must be positive, got {num_condition_latent_frames}"
            )
        if num_condition_latent_frames > num_latent_frames:
            raise ValueError(
                "`num_condition_latent_frames` cannot exceed latent video length, "
                f"got {num_condition_latent_frames} > {num_latent_frames}"
            )
        num_noisy_latent_frames = num_latent_frames - num_condition_latent_frames
        if base.action_conditioned:
            allow_actionless_condition_only = num_noisy_latent_frames == 0 and action is None
            if not allow_actionless_condition_only:
                if action is None:
                    raise ValueError("Action input is required for action-conditioned video expert.")
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B,T,D], got {tuple(action.shape)}")
                if action.shape[0] != batch_size:
                    raise ValueError(f"Batch mismatch between video and action: {batch_size} vs {action.shape[0]}")
                if action.shape[2] != base.action_dim:
                    raise ValueError(f"`action` last dim must be {base.action_dim}, got {action.shape[2]}")
                if num_noisy_latent_frames <= 0:
                    raise ValueError(
                        "video length must include at least one non-condition latent frame when action is provided, "
                        f"got num_latent_frames={num_latent_frames}, "
                        f"num_condition_latent_frames={num_condition_latent_frames}"
                    )
                if action.shape[1] % num_noisy_latent_frames != 0:
                    raise ValueError(
                        "action horizon must be divisible by non-condition latent frames, "
                        f"got action_horizon={action.shape[1]} and num_noisy_latent_frames={num_noisy_latent_frames}"
                    )

        if timestep.ndim == 1:
            if timestep.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"`timestep` length must be 1 or batch_size({batch_size}), "
                    f"got {timestep.shape[0]}"
                )
            if timestep.shape[0] == 1 and batch_size > 1:
                if base.training:
                    raise ValueError(
                        "During training, timestep length must match batch_size."
                    )
                timestep = timestep.expand(batch_size)
            noisy_frame_timesteps = timestep.view(batch_size, 1).expand(
                batch_size,
                num_noisy_latent_frames,
            )
        else:
            expected_timestep_shape = (batch_size, num_noisy_latent_frames)
            if tuple(timestep.shape) != expected_timestep_shape:
                raise ValueError(
                    "Per-frame video timestep must match the noisy latent "
                    f"layout {expected_timestep_shape}, got "
                    f"{tuple(timestep.shape)}."
                )
            noisy_frame_timesteps = timestep

        patch_h = int(base.patch_size[1])
        patch_w = int(base.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if base.seperated_timestep and fuse_vae_embedding_in_latents:
            token_timesteps = torch.zeros(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            )
            if condition_timestep is None:
                condition_timestep = torch.zeros(
                    (batch_size, num_condition_latent_frames),
                    device=timestep.device,
                    dtype=timestep.dtype,
                )
            else:
                condition_timestep = condition_timestep.to(
                    device=timestep.device,
                    dtype=timestep.dtype,
                )
                if condition_timestep.ndim == 1 and batch_size == 1:
                    condition_timestep = condition_timestep.unsqueeze(0)
                expected_shape = (batch_size, num_condition_latent_frames)
                if tuple(condition_timestep.shape) != expected_shape:
                    raise ValueError(
                        "`condition_timestep` must be per condition latent with "
                        f"shape {expected_shape}, got "
                        f"{tuple(condition_timestep.shape)}."
                    )
            token_timesteps[:, :num_condition_latent_frames, :] = (
                condition_timestep.unsqueeze(-1)
            )
            if num_noisy_latent_frames:
                token_timesteps[:, num_condition_latent_frames:, :] = (
                    noisy_frame_timesteps.unsqueeze(-1)
                )
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(base.freq_dim, token_timesteps.reshape(-1))
            t = base.time_embedding(token_t_emb).reshape(batch_size, -1, base.hidden_dim)
            t_mod = base.time_projection(t).unflatten(2, (6, base.hidden_dim))
        else:
            raise NotImplementedError("Only separated timestep with fused VAE condition latents is supported.")

        x = base.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]
        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = torch.cat(
            [
                base.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                base.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                base.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "num_condition_latent_frames": num_condition_latent_frames,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        return self.base.post_dit(x_tokens, pre_state)


class ContextFreeActionExpert(_ContextFreeExpertWrapper):
    def pre_dit(self, action_tokens: torch.Tensor, timestep: torch.Tensor) -> dict[str, Any]:
        base = self.base
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be 3D [B,T,D], got {tuple(action_tokens.shape)}")
        if action_tokens.shape[2] != base.action_dim:
            raise ValueError(f"`action_tokens` last dim must be {base.action_dim}, got {action_tokens.shape[2]}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got {tuple(timestep.shape)}")

        batch_size = action_tokens.shape[0]
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if base.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        seq_len = action_tokens.shape[1]
        if seq_len > base.freqs.shape[0]:
            raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {base.freqs.shape[0]}.")

        t = base.time_embedding(sinusoidal_embedding_1d(base.freq_dim, timestep))
        t_mod = base.time_projection(t).unflatten(1, (6, base.hidden_dim))
        tokens = base.action_encoder(action_tokens)
        freqs = base.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        return self.base.post_dit(tokens, pre_state)


def load_action_dit_for_mot_layers(
    action_dit_config: dict[str, Any],
    action_dit_pretrained_path: str | None,
    skip_dit_load_from_pretrain: bool,
    device: str,
    torch_dtype: torch.dtype,
) -> ContextFreeActionExpert:
    """Load ActionDiT for this side path without writing layer-sliced payloads.

    The official ActionDiT payload may contain more blocks than this MoT path
    uses.  This loader initializes the requested ActionDiT depth and copies the
    matching backbone keys from the payload directly in memory.
    """

    if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
        return ContextFreeActionExpert(
            ActionDiT.from_pretrained(
                action_dit_config=action_dit_config,
                action_dit_pretrained_path=action_dit_pretrained_path,
                skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
        )

    p = Path(action_dit_pretrained_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[4] / p
    if not p.is_file():
        raise FileNotFoundError(f"`action_dit_pretrained_path` does not exist: {p}")

    action_cfg = dict(action_dit_config)
    action_expert = ActionDiT(**action_cfg).to(device=device, dtype=torch_dtype)
    action_state = action_expert.state_dict()
    expected_backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    payload = torch.load(str(p), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid action backbone payload type from {p}: {type(payload)}")
    policy = payload.get("policy", {})
    if policy:
        logger.info("ActionDiT backbone payload policy: %s", policy)

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"`meta` must be a dict in {p}")
    for key in ActionDiT.ACTION_BACKBONE_META_KEYS:
        if key not in meta:
            raise ValueError(f"`meta.{key}` missing in {p}")
    for key in ("hidden_dim", "ffn_dim", "num_heads", "attn_head_dim", "text_dim", "freq_dim"):
        if int(meta[key]) != int(action_cfg[key]):
            raise ValueError(f"`meta.{key}` mismatch in {p}: expected {action_cfg[key]}, got {meta[key]}")
    if abs(float(meta["eps"]) - float(action_cfg["eps"])) > 1e-12:
        raise ValueError(f"`meta.eps` mismatch in {p}: expected {action_cfg['eps']}, got {meta['eps']}")
    if int(meta["num_layers"]) < int(action_cfg["num_layers"]):
        raise ValueError(
            f"Action payload has fewer layers than requested: payload={meta['num_layers']} "
            f"requested={action_cfg['num_layers']}"
        )

    backbone_state_dict = payload.get("backbone_state_dict")
    if not isinstance(backbone_state_dict, dict):
        raise ValueError(f"`backbone_state_dict` must be a dict in {p}, got {type(backbone_state_dict)}")

    missing_keys = sorted(key for key in expected_backbone_keys if key not in backbone_state_dict)
    if missing_keys:
        raise ValueError(
            "Action backbone payload is missing keys needed by requested MoT depth: "
            f"{missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}"
        )

    merged_state = dict(action_state)
    for key in expected_backbone_keys:
        value = backbone_state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"`backbone_state_dict[{key}]` must be torch.Tensor in {p}, got {type(value)}")
        target = merged_state[key]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Shape mismatch for `{key}` in {p}: expected {tuple(target.shape)}, got {tuple(value.shape)}")
        merged_state[key] = value.to(device=target.device, dtype=target.dtype)

    action_expert.load_state_dict(merged_state, strict=True)
    logger.info(
        "Loaded ActionDiT backbone from %s into requested MoT depth %d/%d layers (keys=%d; random_kept_prefixes=%s).",
        p,
        int(action_cfg["num_layers"]),
        int(meta["num_layers"]),
        len(expected_backbone_keys),
        list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
    )
    return ContextFreeActionExpert(action_expert.to(device=device, dtype=torch_dtype))


__all__ = [
    "ContextFreeActionExpert",
    "ContextFreeWanVideoExpert",
    "load_action_dit_for_mot_layers",
]
