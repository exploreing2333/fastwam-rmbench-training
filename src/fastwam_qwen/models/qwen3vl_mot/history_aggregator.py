from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SpatialTemporalHistoryAggregator(nn.Module):
    """Reduce independently encoded history frames to one condition latent.

    The input history is ordered from oldest to newest.  A learned age embedding
    and a per-spatial-location gate let the module retain a different history
    frame for each latent location, while the output remains a single Wan-latent
    frame compatible with the existing two-slot Video DiT condition interface.
    """

    def __init__(
        self,
        latent_dim: int,
        num_history_frames: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_history_frames = int(num_history_frames)
        hidden_dim = int(hidden_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {self.latent_dim}.")
        if self.num_history_frames <= 0:
            raise ValueError(
                f"`num_history_frames` must be positive, got {self.num_history_frames}."
            )
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")

        self.age_embedding = nn.Parameter(
            torch.zeros(self.num_history_frames, self.latent_dim)
        )
        # A small recency preference is a stable starting point; training can
        # override it spatially through ``gate_network``.
        self.age_logit = nn.Parameter(
            torch.linspace(-0.3, 0.3, self.num_history_frames)
        )
        self.value_projection = nn.Conv2d(
            self.latent_dim,
            self.latent_dim,
            kernel_size=1,
            bias=False,
        )
        self.gate_network = nn.Sequential(
            nn.Conv2d(self.latent_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Start from an identity value path so the aggregator initially only
        # performs a (slightly recency-biased) convex combination of VAE latents.
        with torch.no_grad():
            self.value_projection.weight.zero_()
            diagonal = min(
                self.value_projection.weight.shape[0],
                self.value_projection.weight.shape[1],
            )
            indices = torch.arange(diagonal, device=self.value_projection.weight.device)
            self.value_projection.weight[indices, indices, 0, 0] = 1.0
            nn.init.normal_(self.gate_network[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.gate_network[0].bias)
            nn.init.normal_(self.gate_network[2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.gate_network[2].bias)

    def forward(
        self,
        history_latents: torch.Tensor,
        history_is_pad: Optional[torch.Tensor] = None,
        fallback_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate ``[B,K,C,H,W]`` history into ``[B,C,H,W]``.

        ``history_is_pad`` marks synthetic episode-boundary history slots.  They
        receive zero gate mass.  If every history slot is padded, the supplied
        current-frame ``fallback_latent`` is used instead, avoiding a fabricated
        temporal observation at the beginning of an episode.
        """

        if history_latents.ndim != 5:
            raise ValueError(
                "`history_latents` must be [B,K,C,H,W], "
                f"got {tuple(history_latents.shape)}."
            )
        batch_size, num_history_frames, latent_dim, height, width = history_latents.shape
        if num_history_frames != self.num_history_frames:
            raise ValueError(
                "History frame count mismatch: "
                f"got K={num_history_frames}, expected {self.num_history_frames}."
            )
        if latent_dim != self.latent_dim:
            raise ValueError(
                "History latent channel mismatch: "
                f"got C={latent_dim}, expected {self.latent_dim}."
            )

        if history_is_pad is None:
            valid_history = torch.ones(
                (batch_size, num_history_frames),
                device=history_latents.device,
                dtype=torch.bool,
            )
        else:
            if history_is_pad.shape != (batch_size, num_history_frames):
                raise ValueError(
                    "`history_is_pad` must be [B,K] matching history latents, "
                    f"got {tuple(history_is_pad.shape)} vs {(batch_size, num_history_frames)}."
                )
            valid_history = ~history_is_pad.to(
                device=history_latents.device,
                dtype=torch.bool,
            )

        all_history_padded = ~valid_history.any(dim=1)
        # Softmax requires at least one finite logit.  This temporary valid slot
        # is overwritten with the current-frame fallback below.
        effective_valid_history = valid_history.clone()
        if bool(all_history_padded.any()):
            effective_valid_history[all_history_padded, 0] = True

        age = self.age_embedding.to(dtype=history_latents.dtype).view(
            1, num_history_frames, latent_dim, 1, 1
        )
        gate_input = (history_latents + age).reshape(
            batch_size * num_history_frames, latent_dim, height, width
        )
        gate_logits = self.gate_network(gate_input).reshape(
            batch_size, num_history_frames, 1, height, width
        )
        gate_logits = gate_logits + self.age_logit.to(dtype=gate_logits.dtype).view(
            1,
            -1,
            1,
            1,
            1,
        )
        gate_logits = gate_logits.masked_fill(
            ~effective_valid_history.view(batch_size, num_history_frames, 1, 1, 1),
            -torch.finfo(gate_logits.dtype).max,
        )
        gate = torch.softmax(gate_logits, dim=1)

        values = self.value_projection(
            history_latents.reshape(
                batch_size * num_history_frames,
                latent_dim,
                height,
                width,
            )
        ).reshape(batch_size, num_history_frames, latent_dim, height, width)
        aggregated = (gate * values).sum(dim=1)

        if bool(all_history_padded.any()) and fallback_latent is not None:
            if fallback_latent.ndim == 5:
                if fallback_latent.shape[2] != 1:
                    raise ValueError(
                        "5D `fallback_latent` must have one temporal frame, "
                        f"got {tuple(fallback_latent.shape)}."
                    )
                fallback_latent = fallback_latent[:, :, 0]
            expected_fallback_shape = (batch_size, latent_dim, height, width)
            if fallback_latent.shape != expected_fallback_shape:
                raise ValueError(
                    "`fallback_latent` must match the aggregated latent shape, "
                    f"got {tuple(fallback_latent.shape)}, expected {expected_fallback_shape}."
                )
            fallback_latent = fallback_latent.to(
                device=aggregated.device,
                dtype=aggregated.dtype,
            )
            aggregated = torch.where(
                all_history_padded.view(batch_size, 1, 1, 1),
                fallback_latent,
                aggregated,
            )
        return aggregated
