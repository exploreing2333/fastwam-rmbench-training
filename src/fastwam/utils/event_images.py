"""Shared event-image mosaic after the RMBench JPEG decode boundary."""
import torch
import torchvision.transforms.functional as tvf


def event_mosaic_from_decoded(views):
    if len(views) != 3:
        raise ValueError("Event memory requires head, left-wrist, right-wrist RGB views.")
    resized = []
    for rgb, size in zip(views, ((256, 320), (128, 160), (128, 160))):
        if rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError("Decoded event view must be uint8 [3,H,W].")
        resized.append(tvf.resize(rgb.float().div(255).unsqueeze(0), list(size),
                                  interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)[0])
    return torch.cat([resized[0], torch.cat(resized[1:], dim=-1)], dim=-2).sub(0.5).div(0.5).contiguous()
