from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from TMD3.multiview import (
    CAMERA_IMAGE_SIZES,
    LATENT_LAYOUT,
    LATENT_SIZE,
    RGB_LAYOUT,
    RGB_TRANSPORT_SIZE,
    VAE_STREAM_IDS,
)

CACHE_FORMAT = "tmd3_lingbot_robotwin_episode_stream_v3"
CACHE_FORMAT_STRIDE1 = "tmd3_lingbot_robotwin_phase_stream_v4"
MANIFEST_NAME = "manifest.json"


class TMD3LatentStore:
    """Read VAE latents that were streamed per episode before sample shuffle."""

    def __init__(
        self,
        root: str | Path,
        *,
        camera_order: list[str],
        history_stride: int,
        action_horizon: int,
        anchor_stride: int = 16,
        condition_latents: int = 4,
        max_mapped_episodes: int = 8,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "TMD3 streaming latent manifest is missing: "
                f"{manifest_path}. Run scripts/precompute_tmd3_streaming_latents.py first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cache_format = self.manifest.get("format")
        if self.cache_format not in {CACHE_FORMAT, CACHE_FORMAT_STRIDE1}:
            raise ValueError(
                "Unsupported TMD3 streaming latent cache format: "
                f"{self.cache_format!r}."
            )
        self.anchor_stride = int(anchor_stride)
        expected_format = (
            CACHE_FORMAT_STRIDE1 if self.anchor_stride == 1 else CACHE_FORMAT
        )
        expected = {
            "format": expected_format,
            "camera_order": list(camera_order),
            "history_stride": int(history_stride),
            "action_horizon": int(action_horizon),
            "condition_latents": int(condition_latents),
            "camera_image_sizes": {
                key: list(CAMERA_IMAGE_SIZES[key]) for key in camera_order
            },
            "rgb_layout": RGB_LAYOUT,
            "rgb_transport_size": list(RGB_TRANSPORT_SIZE),
            "latent_layout": LATENT_LAYOUT,
            "latent_size": list(LATENT_SIZE),
            "vae_stream_ids": list(VAE_STREAM_IDS),
        }
        if self.cache_format == CACHE_FORMAT_STRIDE1:
            expected.update(
                {
                    "anchor_stride": 1,
                    "phase_count": int(action_horizon),
                }
            )
        mismatches = {
            key: {"actual": self.manifest.get(key), "expected": value}
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "TMD3 streaming latent cache contract mismatch: "
                f"{mismatches}."
            )
        if not bool(self.manifest.get("complete", False)):
            raise RuntimeError(
                f"TMD3 streaming latent cache is incomplete: {manifest_path}"
            )
        total_episodes = int(self.manifest.get("total_episodes", 0))
        if total_episodes <= 0:
            raise ValueError(
                "TMD3 streaming latent manifest must contain a positive "
                f"`total_episodes`, got {total_episodes}."
            )
        episode_dir = self.root / "episodes"
        episode_indices = self.manifest.get("episode_indices")
        if episode_indices is None:
            episode_indices = range(total_episodes)
        missing = [
            self.episode_path(episode_index)
            for episode_index in episode_indices
            if not self.episode_path(episode_index).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "TMD3 streaming latent cache is marked complete but is missing "
                f"{len(missing)} episode file(s); first missing: {missing[0]}"
            )
        self.condition_latents = int(condition_latents)
        self.max_mapped_episodes = max(1, int(max_mapped_episodes))
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
                raise FileNotFoundError(
                    f"Missing TMD3 streaming latent episode cache: {path}"
                )
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        self._payloads[episode_index] = payload
        while len(self._payloads) > self.max_mapped_episodes:
            self._payloads.popitem(last=False)
        return payload

    def load(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        path = self.episode_path(episode_index)
        payload = self._payload(episode_index)
        if payload.get("format") != self.cache_format:
            raise ValueError(
                f"Unexpected TMD3 cache format in {path}: {payload.get('format')!r}."
            )
        if self.cache_format == CACHE_FORMAT_STRIDE1:
            return self._load_stride1_payload(
                payload,
                episode_index=int(episode_index),
                frame_index=int(frame_index),
            )
        anchors = torch.as_tensor(payload["anchor_frame_indices"], dtype=torch.long)
        matches = torch.nonzero(anchors == int(frame_index), as_tuple=False).flatten()
        if matches.numel() != 1:
            raise KeyError(
                "TMD3 cache has no unique replan-aligned anchor: "
                f"episode={episode_index}, frame={frame_index}, matches={matches.numel()}."
            )
        sample_index = int(matches.item())
        stream_latents = payload["stream_latents"]
        if not torch.is_tensor(stream_latents) or stream_latents.ndim != 4:
            raise ValueError(
                f"`stream_latents` must be [C,T,H,W] in {path}."
            )
        if tuple(stream_latents.shape[-2:]) != LATENT_SIZE:
            raise ValueError(
                f"Stored TMD3 latent must end in {LATENT_SIZE}, got "
                f"{tuple(stream_latents.shape)} in {path}."
            )
        # v3 stores the real episode stream as [anchor,z1,z2,...] without
        # manufacturing repeated-image history at the episode boundary.
        future_index = sample_index + 1
        if future_index >= int(stream_latents.shape[1]):
            raise ValueError(
                "TMD3 cache cannot provide H1..H4 plus Future at "
                f"episode={episode_index}, frame={frame_index}."
            )
        history_start = max(0, sample_index - self.condition_latents + 1)
        valid_history = stream_latents[
            :, history_start : sample_index + 1
        ]
        valid_count = int(valid_history.shape[1])
        if not 1 <= valid_count <= self.condition_latents:
            raise RuntimeError("Unexpected valid TMD3 history length.")
        sample_latents = torch.zeros(
            (
                int(stream_latents.shape[0]),
                self.condition_latents + 1,
                int(stream_latents.shape[2]),
                int(stream_latents.shape[3]),
            ),
            dtype=stream_latents.dtype,
            device=stream_latents.device,
        )
        sample_latents[
            :, self.condition_latents - valid_count : self.condition_latents
        ] = valid_history
        sample_latents[:, self.condition_latents :] = stream_latents[
            :, future_index : future_index + 1
        ]
        condition_valid_mask = torch.zeros(
            self.condition_latents,
            dtype=torch.bool,
        )
        condition_valid_mask[-valid_count:] = True
        return {
            "precomputed_video_latents": sample_latents,
            "condition_latent_valid_mask": condition_valid_mask,
            "tmd3_latent_cache_sample_index": sample_index,
        }

    def _load_stride1_payload(
        self,
        payload: dict[str, Any],
        *,
        episode_index: int,
        frame_index: int,
    ) -> dict[str, Any]:
        episode_length = int(payload["episode_length"])
        if not 0 <= frame_index < episode_length - int(
            self.manifest["action_horizon"]
        ):
            raise KeyError(
                "TMD3 stride-1 cache anchor is outside the legal range: "
                f"episode={episode_index}, frame={frame_index}, "
                f"length={episode_length}."
            )
        phase_count = int(self.manifest["phase_count"])
        phase = frame_index % phase_count
        local_index = frame_index // phase_count
        phase_offsets = torch.as_tensor(
            payload["phase_offsets"], dtype=torch.long
        )
        if phase_offsets.shape != (phase_count + 1,):
            raise ValueError(
                f"Invalid phase offsets in episode {episode_index}: "
                f"{tuple(phase_offsets.shape)}."
            )
        phase_start = int(phase_offsets[phase].item())
        phase_stop = int(phase_offsets[phase + 1].item())
        # Every phase stream contains one latent at each real anchor plus the
        # next latent used as that phase's final future target.
        if not 0 <= local_index < phase_stop - phase_start - 1:
            raise KeyError(
                "TMD3 stride-1 cache has no unique phase-stream anchor: "
                f"episode={episode_index}, frame={frame_index}, phase={phase}."
            )
        stream_latents = payload["stream_latents"]
        if not torch.is_tensor(stream_latents) or stream_latents.ndim != 4:
            raise ValueError("`stream_latents` must be [C,T,H,W].")
        if tuple(stream_latents.shape[-2:]) != LATENT_SIZE:
            raise ValueError(
                f"Stored TMD3 latent must end in {LATENT_SIZE}, got "
                f"{tuple(stream_latents.shape)}."
            )
        sample_index = phase_start + local_index
        history_start = max(
            phase_start,
            sample_index - self.condition_latents + 1,
        )
        valid_history = stream_latents[:, history_start : sample_index + 1]
        valid_count = int(valid_history.shape[1])
        sample_latents = torch.zeros(
            (
                int(stream_latents.shape[0]),
                self.condition_latents + 1,
                int(stream_latents.shape[2]),
                int(stream_latents.shape[3]),
            ),
            dtype=stream_latents.dtype,
            device=stream_latents.device,
        )
        sample_latents[
            :, self.condition_latents - valid_count : self.condition_latents
        ] = valid_history
        sample_latents[:, self.condition_latents :] = stream_latents[
            :, sample_index + 1 : sample_index + 2
        ]
        condition_valid_mask = torch.zeros(
            self.condition_latents,
            dtype=torch.bool,
        )
        condition_valid_mask[-valid_count:] = True
        return {
            "precomputed_video_latents": sample_latents,
            "condition_latent_valid_mask": condition_valid_mask,
            "tmd3_latent_cache_sample_index": frame_index,
        }

    def load_long_image_clip(
        self,
        episode_index: int,
        frame_index: int,
        *,
        max_history_latents: int,
        max_future_latents: int,
    ) -> dict[str, Any]:
        """Load a right-aligned causal history and consecutive future targets.

        The stride-1 cache stores sixteen independent causal VAE streams, one
        for every control-step phase.  A long clip must stay inside the phase
        selected by ``frame_index``; adjacent latent entries are therefore
        exactly sixteen control steps apart and share one continuous VAE
        history.  Missing episode-start history is zero-filled and masked,
        never manufactured by repeating the first observation.  Real history
        and Future positions remain the phase stream's global latent indices.
        """
        max_history_latents = int(max_history_latents)
        max_future_latents = int(max_future_latents)
        if max_history_latents <= 0 or max_future_latents <= 0:
            raise ValueError(
                "Long-image history/future latent counts must be positive, "
                f"got history={max_history_latents}, future={max_future_latents}."
            )
        if self.cache_format != CACHE_FORMAT_STRIDE1:
            raise ValueError(
                "Long-image clips require the stride-1 phase-stream cache, "
                f"got format={self.cache_format!r}."
            )

        episode_index = int(episode_index)
        frame_index = int(frame_index)
        payload = self._payload(episode_index)
        path = self.episode_path(episode_index)
        if payload.get("format") != self.cache_format:
            raise ValueError(
                f"Unexpected TMD3 cache format in {path}: "
                f"{payload.get('format')!r}."
            )
        episode_length = int(payload["episode_length"])
        future_control_span = (
            max_future_latents * int(self.manifest["action_horizon"])
        )
        if not 0 <= frame_index < episode_length - future_control_span:
            raise KeyError(
                "Long-image anchor cannot provide the requested consecutive "
                f"future: episode={episode_index}, frame={frame_index}, "
                f"length={episode_length}, future_control_span={future_control_span}."
            )

        phase_count = int(self.manifest["phase_count"])
        phase = frame_index % phase_count
        local_index = frame_index // phase_count
        phase_offsets = torch.as_tensor(
            payload["phase_offsets"], dtype=torch.long
        )
        if phase_offsets.shape != (phase_count + 1,):
            raise ValueError(
                f"Invalid phase offsets in episode {episode_index}: "
                f"{tuple(phase_offsets.shape)}."
            )
        phase_start = int(phase_offsets[phase].item())
        phase_stop = int(phase_offsets[phase + 1].item())
        sample_index = phase_start + local_index
        if sample_index + max_future_latents >= phase_stop:
            raise KeyError(
                "Phase stream is too short for the requested future clip: "
                f"episode={episode_index}, frame={frame_index}, phase={phase}, "
                f"sample={sample_index}, stop={phase_stop}, "
                f"future={max_future_latents}."
            )

        stream_latents = payload["stream_latents"]
        if not torch.is_tensor(stream_latents) or stream_latents.ndim != 4:
            raise ValueError(
                f"`stream_latents` must be [C,T,H,W] in {path}."
            )
        if tuple(stream_latents.shape[-2:]) != LATENT_SIZE:
            raise ValueError(
                f"Stored TMD3 latent must end in {LATENT_SIZE}, got "
                f"{tuple(stream_latents.shape)} in {path}."
            )

        history_start = max(
            phase_start,
            sample_index - max_history_latents + 1,
        )
        valid_history = stream_latents[
            :, history_start : sample_index + 1
        ]
        valid_count = int(valid_history.shape[1])
        history = torch.zeros(
            (
                int(stream_latents.shape[0]),
                max_history_latents,
                int(stream_latents.shape[2]),
                int(stream_latents.shape[3]),
            ),
            dtype=stream_latents.dtype,
            device=stream_latents.device,
        )
        history[:, max_history_latents - valid_count :] = valid_history
        history_valid = torch.zeros(max_history_latents, dtype=torch.bool)
        history_valid[-valid_count:] = True
        history_frame_indices = torch.full(
            (max_history_latents,),
            -1,
            dtype=torch.long,
        )
        history_frame_indices[-valid_count:] = torch.arange(
            local_index - valid_count + 1,
            local_index + 1,
            dtype=torch.long,
        )
        future = stream_latents[
            :, sample_index + 1 : sample_index + 1 + max_future_latents
        ].contiguous()
        if int(future.shape[1]) != max_future_latents:
            raise RuntimeError(
                "Long-image cache returned a truncated future: "
                f"expected={max_future_latents}, got={future.shape[1]}."
            )
        return {
            "long_image_history_latents": history,
            "long_image_history_valid_mask": history_valid,
            "long_image_history_frame_indices": history_frame_indices,
            "long_image_future_latents": future,
            "long_image_future_frame_indices": torch.arange(
                local_index + 1,
                local_index + 1 + max_future_latents,
                dtype=torch.long,
            ),
            "long_image_anchor_frame": frame_index,
            "long_image_phase": phase,
        }


__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_STRIDE1",
    "MANIFEST_NAME",
    "TMD3LatentStore",
]
