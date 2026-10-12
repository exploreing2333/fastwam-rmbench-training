from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


CAMERA_ORDER = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)
CAMERA_IMAGE_SIZES = {
    "cam_high": (256, 320),
    "cam_left_wrist": (128, 160),
    "cam_right_wrist": (128, 160),
}
RGB_LAYOUT = "wrists_top_high_bottom"
RGB_TRANSPORT_SIZE = (384, 320)
LATENT_LAYOUT = "left_right_top_high_bottom"
LATENT_SIZE = (24, 20)
VAE_STREAM_IDS = ("high", "wrists")


def _ordered_views(
    views: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(views, Mapping):
        missing = [key for key in CAMERA_ORDER if key not in views]
        if missing:
            raise KeyError(f"RobotWin views are missing cameras: {missing}.")
        return tuple(views[key] for key in CAMERA_ORDER)  # type: ignore[return-value]
    if len(views) != len(CAMERA_ORDER):
        raise ValueError(
            f"RobotWin layout requires three views, got {len(views)}."
        )
    return tuple(views)  # type: ignore[return-value]


def compose_robotwin_rgb(
    views: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compose already-resized views as ``[left|right; high]``."""
    high, left, right = _ordered_views(views)
    leading_shape = tuple(high.shape[:-2])
    for key, view in zip(CAMERA_ORDER, (high, left, right), strict=True):
        expected_size = CAMERA_IMAGE_SIZES[key]
        if tuple(view.shape[:-2]) != leading_shape:
            raise ValueError(
                "RobotWin views must have identical leading dimensions: "
                f"high={tuple(high.shape)}, {key}={tuple(view.shape)}."
            )
        if tuple(view.shape[-2:]) != expected_size:
            raise ValueError(
                f"RobotWin {key} must be {expected_size}, got "
                f"{tuple(view.shape[-2:])}."
            )
    wrist_block = torch.cat([left, right], dim=-1)
    mosaic = torch.cat([wrist_block, high], dim=-2)
    if tuple(mosaic.shape[-2:]) != RGB_TRANSPORT_SIZE:
        raise RuntimeError(f"Unexpected RobotWin RGB mosaic: {tuple(mosaic.shape)}.")
    return mosaic.contiguous()


def split_robotwin_rgb(
    mosaic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[left|right; high]`` without resizing or mixing camera pixels."""
    if tuple(mosaic.shape[-2:]) != RGB_TRANSPORT_SIZE:
        raise ValueError(
            f"RobotWin RGB transport must end in {RGB_TRANSPORT_SIZE}, got "
            f"{tuple(mosaic.shape)}."
        )
    wrist_height, wrist_width = CAMERA_IMAGE_SIZES["cam_left_wrist"]
    wrist_block = mosaic[..., :wrist_height, :]
    high = mosaic[..., wrist_height:, :]
    left = wrist_block[..., :wrist_width]
    right = wrist_block[..., wrist_width:]
    return high.contiguous(), left.contiguous(), right.contiguous()


def compose_robotwin_latents(
    high: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compose VAE latents as ``[left|right; high]``."""
    if high.ndim != 5 or left.ndim != 5 or right.ndim != 5:
        raise ValueError("RobotWin latents must be [B,C,T,H,W].")
    if high.shape[:3] != left.shape[:3] or left.shape != right.shape:
        raise ValueError(
            "RobotWin latent batch/channel/time shapes do not match: "
            f"high={tuple(high.shape)}, left={tuple(left.shape)}, "
            f"right={tuple(right.shape)}."
        )
    if high.shape[-1] != left.shape[-1] + right.shape[-1]:
        raise ValueError(
            "High latent width must equal the combined wrist widths: "
            f"high={high.shape[-1]}, wrists={left.shape[-1]}+{right.shape[-1]}."
        )
    if high.shape[-2] != left.shape[-2] + right.shape[-2]:
        raise ValueError(
            "High latent height must equal the combined wrist heights: "
            f"high={high.shape[-2]}, wrists={left.shape[-2]}+{right.shape[-2]}."
        )
    output = torch.cat(
        [torch.cat([left, right], dim=-1), high],
        dim=-2,
    ).contiguous()
    if tuple(output.shape[-2:]) != LATENT_SIZE:
        raise ValueError(
            f"RobotWin latent must end in {LATENT_SIZE}, got {tuple(output.shape)}."
        )
    return output


__all__ = [
    "CAMERA_IMAGE_SIZES",
    "CAMERA_ORDER",
    "LATENT_LAYOUT",
    "LATENT_SIZE",
    "RGB_LAYOUT",
    "RGB_TRANSPORT_SIZE",
    "VAE_STREAM_IDS",
    "compose_robotwin_latents",
    "compose_robotwin_rgb",
    "split_robotwin_rgb",
]
