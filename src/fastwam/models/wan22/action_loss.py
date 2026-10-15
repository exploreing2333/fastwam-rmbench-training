"""Action loss reduction with an optional supervision-only mask."""
from __future__ import annotations

import torch


def reduce_action_loss(
    token_error: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
    action_loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask supervision without increasing the weight of surviving tokens.

    The loss mask does not change the denominator. GT terminal hold keeps all
    32 carrier tokens valid, so excluded hold targets contribute zero / 32.
    Without this optional mask, retain the original episode-padding reduction.
    """
    if token_error.ndim != 2:
        raise ValueError("Action token errors must have shape [B, T].")
    if action_loss_mask is not None:
        if action_loss_mask.shape != token_error.shape:
            raise ValueError("action_loss_mask must match the action horizon.")
        mask = action_loss_mask.to(device=token_error.device, dtype=torch.bool)
        token_error = token_error.masked_fill(~mask, 0)
    if action_is_pad is None:
        return token_error.mean(dim=1)
    if action_is_pad.shape != token_error.shape:
        raise ValueError("action_is_pad must match the action horizon.")
    valid = (~action_is_pad).to(device=token_error.device, dtype=token_error.dtype)
    return (token_error * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
