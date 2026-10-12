from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


EVENT_ANCHOR_CACHE_FORMAT = "fastwam_event_anchor_latents_v1"


class EventAnchorLatentStore:
    """Read VAE latents for causal Progress event-boundary anchors.

    Each episode payload stores one latent per V72 condition transition.  A
    sample at frame ``t`` receives only anchors whose event frame is ``<= t``;
    the returned tensor is fixed-width so the normal FastWAM collator can stack
    it without introducing a second variable-length batch path.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_anchors: int = 4,
        max_cached_episodes: int = 8,
        include_candidates: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Event-anchor manifest does not exist: {manifest_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != EVENT_ANCHOR_CACHE_FORMAT:
            raise ValueError(
                "Unsupported event-anchor cache format: "
                f"{self.manifest.get('format')!r}."
            )
        if not bool(self.manifest.get("complete", False)):
            raise RuntimeError(f"Event-anchor cache is incomplete: {manifest_path}")
        self.max_anchors = int(max_anchors)
        if self.max_anchors <= 0:
            raise ValueError(f"max_anchors must be positive, got {max_anchors}.")
        self.max_cached_episodes = max(1, int(max_cached_episodes))
        self.include_candidates = bool(include_candidates)
        if self.include_candidates and not self.manifest.get("candidate_offsets"):
            raise ValueError("Temporal memory augmentation requires a candidate-latent cache.")
        self._payloads: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_payloads"] = OrderedDict()
        return state

    def episode_path(self, episode_index: int) -> Path:
        return self.root / "episodes" / f"episode_{int(episode_index):06d}.pt"

    def _payload(self, episode_index: int) -> dict[str, Any]:
        episode_index = int(episode_index)
        payload = self._payloads.pop(episode_index, None)
        if payload is None:
            path = self.episode_path(episode_index)
            if not path.is_file():
                raise FileNotFoundError(f"Missing event-anchor cache: {path}")
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        self._payloads[episode_index] = payload
        while len(self._payloads) > self.max_cached_episodes:
            self._payloads.popitem(last=False)
        return payload

    def load(self, episode_index: int, frame_index: int) -> dict[str, torch.Tensor]:
        payload = self._payload(int(episode_index))
        event_indices = torch.as_tensor(payload["event_indices"], dtype=torch.long)
        event_latents = payload["event_latents"]
        if not torch.is_tensor(event_latents) or event_latents.ndim != 4:
            raise ValueError(
                "Event-anchor latents must have shape [E,C,H,W], got "
                f"{type(event_latents)} / {getattr(event_latents, 'shape', None)}."
            )
        if event_indices.ndim != 1 or event_indices.shape[0] != event_latents.shape[0]:
            raise ValueError("Event indices and latent count do not match.")
        if event_indices.numel() == 0 or int(event_indices[0]) != 0:
            raise ValueError("Every event-anchor episode must contain frame 0.")

        visible = torch.nonzero(event_indices <= int(frame_index), as_tuple=False).flatten()
        if visible.numel() == 0:
            raise RuntimeError(
                f"No causal event anchor for episode={episode_index}, frame={frame_index}."
            )
        visible_indices = visible.tolist()
        # Keep the initial scene and the newest event boundaries.  This is a
        # fixed-size approximation of the unbounded causal cache.
        selected = [visible_indices[0]]
        if self.max_anchors > 1:
            selected.extend(visible_indices[-(self.max_anchors - 1) :])
        deduplicated: list[int] = []
        for index in selected:
            if index not in deduplicated:
                deduplicated.append(index)
        valid_count = len(deduplicated)
        padded_indices = list(deduplicated)
        while len(padded_indices) < self.max_anchors:
            padded_indices.append(padded_indices[0])

        # Keep a fixed-width transport tensor for the ordinary collator, but
        # never duplicate init into the model memory. The explicit mask is
        # consumed immediately before KV prefill, where the sequence is
        # compacted to its true variable length.
        selected_tensor = torch.as_tensor(padded_indices, dtype=torch.long)
        valid_tensor = torch.as_tensor(deduplicated, dtype=torch.long)
        anchors = torch.zeros(
            (self.max_anchors, *event_latents.shape[1:]),
            dtype=event_latents.dtype,
        )
        anchors[:valid_count] = event_latents.index_select(0, valid_tensor)
        returned_indices = torch.full(
            (self.max_anchors,), -1, dtype=torch.long
        )
        returned_indices[:valid_count] = event_indices.index_select(0, valid_tensor)
        valid_mask = torch.zeros((self.max_anchors,), dtype=torch.bool)
        valid_mask[:valid_count] = True
        result = {
            "event_anchor_latents": anchors.contiguous(),
            "event_anchor_frame_indices": returned_indices,
            "event_anchor_valid_mask": valid_mask,
        }
        if self.include_candidates:
            candidates = payload["candidate_latents"]
            candidate_frames = payload["candidate_frame_indices"]
            if candidates.shape[:2] != candidate_frames.shape or candidates.shape[0] != len(event_indices):
                raise ValueError("Invalid event-candidate cache shape.")
            result["event_anchor_candidate_latents"] = torch.zeros(
                (self.max_anchors, *candidates.shape[1:]), dtype=candidates.dtype
            )
            result["event_anchor_candidate_indices"] = torch.full(
                (self.max_anchors, candidate_frames.shape[1]), -1, dtype=torch.long
            )
            result["event_anchor_candidate_latents"][:valid_count] = candidates.index_select(0, valid_tensor)
            result["event_anchor_candidate_indices"][:valid_count] = candidate_frames.index_select(0, valid_tensor)
            # A second, group-wide causality check is mandatory at augmentation
            # time. Returning candidate metadata is not making it visible to DiT.
        return result


__all__ = ["EVENT_ANCHOR_CACHE_FORMAT", "EventAnchorLatentStore"]
