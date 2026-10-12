from __future__ import annotations

from typing import Any

import torch

from fastwam.models.wan22.fastwam import FastWAM
from TMD2.model import ProgressTMD as TMD2ProgressTMD
from TMD3.multiview import (
    CAMERA_ORDER,
    LATENT_SIZE,
    RGB_TRANSPORT_SIZE,
    compose_robotwin_latents,
    split_robotwin_rgb,
)


TMD3_CAMERA_ORDER = CAMERA_ORDER


class ProgressTMD(TMD2ProgressTMD):
    """Four-latent memory with LingBot RobotWin multi-view VAE encoding."""

    independent_local_reset_endpoints = False
    camera_order = TMD3_CAMERA_ORDER
    num_cameras = len(TMD3_CAMERA_ORDER)
    condition_latents = 4
    future_latents = 1
    action_horizon = 16
    contract_name = "TMD3"

    def _prepare_training_condition_latents(
        self,
        condition_latents: torch.Tensor,
        condition_latent_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply LingBot-style stochastic noise to the four history latents."""
        if condition_latents.ndim != 5:
            raise ValueError(
                "TMD3 condition latents must be [B,C,4,H,W], got "
                f"{tuple(condition_latents.shape)}."
            )
        batch_size = int(condition_latents.shape[0])
        num_condition_latents = int(condition_latents.shape[2])
        if num_condition_latents != int(self.condition_latents):
            raise ValueError(
                "TMD3 condition latent count mismatch: "
                f"got={num_condition_latents}, "
                f"expected={self.condition_latents}."
            )
        if condition_latent_valid_mask is None:
            condition_latent_valid_mask = torch.ones(
                (batch_size, num_condition_latents),
                device=condition_latents.device,
                dtype=torch.bool,
            )
        else:
            condition_latent_valid_mask = condition_latent_valid_mask.to(
                device=condition_latents.device,
                dtype=torch.bool,
            )
            if tuple(condition_latent_valid_mask.shape) != (
                batch_size,
                num_condition_latents,
            ):
                raise ValueError(
                    "TMD3 condition valid mask must be [B,4], got "
                    f"{tuple(condition_latent_valid_mask.shape)}."
                )

        if (
            self.history_condition_noise_prob <= 0.0
            or torch.rand((), device=condition_latents.device).item()
            >= self.history_condition_noise_prob
        ):
            return condition_latents, None

        noise_sampling = str(
            getattr(
                self,
                "history_condition_noise_sampling",
                "lingbot_timestep",
            )
        )
        # MemoryWAM linearly mixes every clean conditioning latent with
        # Gaussian noise at an independently sampled ratio.  The ratio is the
        # actual flow sigma, rather than an index into a shifted scheduler.
        if noise_sampling == "uniform_mix_per_sample":
            u = torch.rand(
                (batch_size, num_condition_latents),
                device=condition_latents.device,
                dtype=torch.float32,
            )
            sigma = self.history_condition_noise_u_min + u * (
                self.history_condition_noise_u_max
                - self.history_condition_noise_u_min
            )
            condition_timestep = sigma * float(
                self.train_video_scheduler.num_train_timesteps
            )
            sigma = sigma.to(dtype=condition_latents.dtype).view(
                batch_size,
                1,
                num_condition_latents,
                1,
                1,
            )
            noisy_condition_latents = (
                (1.0 - sigma) * condition_latents
                + sigma * torch.randn_like(condition_latents)
            )
            valid = condition_latent_valid_mask[:, None, :, None, None]
            noisy_condition_latents = torch.where(
                valid,
                noisy_condition_latents,
                condition_latents,
            )
            condition_timestep = condition_timestep.to(
                dtype=condition_latents.dtype
            )
            condition_timestep = torch.where(
                condition_latent_valid_mask,
                condition_timestep,
                torch.zeros_like(condition_timestep),
            )
            return noisy_condition_latents, condition_timestep
        if noise_sampling != "lingbot_timestep":
            raise ValueError(
                "Unsupported TMD3 history condition-noise sampling mode: "
                f"{noise_sampling!r}."
            )

        # LingBot samples one condition-timestep vector for the trajectory
        # sample and repeats it across the training batch.  Gaussian noise is
        # still independent for every tensor element.
        u = torch.rand(
            (num_condition_latents,),
            device=condition_latents.device,
            dtype=torch.float32,
        )
        u = self.history_condition_noise_u_min + u * (
            self.history_condition_noise_u_max
            - self.history_condition_noise_u_min
        )
        # Official LingBot converts u to an index into a *descending* sigma
        # table.  With extra_one_step training sigmas this is exactly
        # base_sigma = 1 - floor(u*N)/N, not base_sigma = u.
        num_train_timesteps = int(
            self.train_video_scheduler.num_train_timesteps
        )
        timestep_ids = torch.floor(u * num_train_timesteps).clamp(
            min=0,
            max=num_train_timesteps - 1,
        )
        base_sigma = 1.0 - timestep_ids / float(num_train_timesteps)
        sigma = self.train_video_scheduler._phi(
            base_sigma,
            self.train_video_scheduler.shift,
        )
        condition_timestep = sigma * float(num_train_timesteps)
        sigma = sigma.to(dtype=condition_latents.dtype).view(
            1,
            1,
            num_condition_latents,
            1,
            1,
        )
        noisy_condition_latents = (
            (1.0 - sigma) * condition_latents
            + sigma * torch.randn_like(condition_latents)
        )
        valid = condition_latent_valid_mask[:, None, :, None, None]
        noisy_condition_latents = torch.where(
            valid,
            noisy_condition_latents,
            condition_latents,
        )
        condition_timestep = condition_timestep.to(
            dtype=condition_latents.dtype
        ).unsqueeze(0).expand(batch_size, -1)
        condition_timestep = torch.where(
            condition_latent_valid_mask,
            condition_timestep,
            torch.zeros_like(condition_timestep),
        )
        return noisy_condition_latents, condition_timestep

    def reset_streaming_history(self) -> None:
        """Clear both named VAE cache banks and history at episode reset."""
        reset_streaming_encode = getattr(
            getattr(self, "vae", None),
            "reset_streaming_encode",
            None,
        )
        if not callable(reset_streaming_encode):
            raise RuntimeError(
                "TMD3 online streaming requires WanVideoVAE.reset_streaming_encode()."
            )
        reset_streaming_encode()
        self._streaming_condition_latents = None
        self._streaming_condition_valid_mask = None
        self._streaming_condition_latents_by_id = {}
        self._streaming_condition_valid_masks_by_id = {}

    @torch.no_grad()
    def _encode_streaming_camera_chunk(
        self,
        input_images: torch.Tensor,
        *,
        reset: bool,
        stream_id: str = "default",
    ) -> torch.Tensor:
        """Stream high and wrist views through independent VAE cache banks."""
        input_images = self._normalize_inference_images(input_images)
        video = input_images.to(
            device=self.device,
            dtype=self.torch_dtype,
        ).permute(0, 2, 1, 3, 4).contiguous()
        batch_size = int(video.shape[0])
        if batch_size != 1:
            raise ValueError(
                "TMD3 online streaming currently requires batch size one, "
                f"got B={batch_size}."
            )
        high_video, left_video, right_video = split_robotwin_rgb(video)
        wrist_video = torch.stack(
            [left_video, right_video],
            dim=1,
        ).flatten(0, 1).contiguous()
        streaming_encode = getattr(
            getattr(self, "vae", None),
            "streaming_encode",
            None,
        )
        if not callable(streaming_encode):
            raise RuntimeError(
                "TMD3 online streaming requires WanVideoVAE.streaming_encode()."
            )
        stream_id = str(stream_id)
        stream_prefix = "" if stream_id == "default" else f"{stream_id}:"
        high_latent = streaming_encode(
            high_video,
            device=self.device,
            reset=reset,
            stream_id=f"{stream_prefix}high",
        )
        wrist_latents = streaming_encode(
            wrist_video,
            device=self.device,
            reset=reset,
            stream_id=f"{stream_prefix}wrists",
        )
        if int(high_latent.shape[0]) != batch_size:
            raise RuntimeError(
                "Streaming high latent batch mismatch: "
                f"got={high_latent.shape[0]}, expected={batch_size}."
            )
        if int(wrist_latents.shape[0]) != batch_size * 2:
            raise RuntimeError(
                "Streaming wrist latent batch mismatch: "
                f"got={wrist_latents.shape[0]}, expected={batch_size * 2}."
            )
        wrist_latents = wrist_latents.reshape(
            batch_size,
            2,
            *wrist_latents.shape[1:],
        )
        return compose_robotwin_latents(
            high_latent,
            wrist_latents[:, 0],
            wrist_latents[:, 1],
        )

    @torch.no_grad()
    def initialize_streaming_history(
        self,
        input_images: torch.Tensor,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        """Initialize one real anchor and three masked history slots."""
        input_images = self._normalize_inference_images(input_images)
        expected_frames = 1
        if int(input_images.shape[1]) != expected_frames:
            raise ValueError(
                "TMD3 official streaming initialization requires one image, "
                f"got T={input_images.shape[1]}."
            )
        all_latents = self._encode_streaming_camera_chunk(
            input_images,
            reset=True,
            stream_id=stream_id,
        )
        expected_latents = 1
        if int(all_latents.shape[2]) != expected_latents:
            raise RuntimeError(
                "TMD3 streaming initialization must emit one anchor latent, "
                f"got {tuple(all_latents.shape)}."
            )
        condition_latents = torch.zeros(
            (
                int(all_latents.shape[0]),
                int(all_latents.shape[1]),
                int(self.condition_latents),
                int(all_latents.shape[3]),
                int(all_latents.shape[4]),
            ),
            device=all_latents.device,
            dtype=all_latents.dtype,
        )
        condition_latents[:, :, -1:] = all_latents
        condition_valid_mask = torch.zeros(
            (int(all_latents.shape[0]), int(self.condition_latents)),
            device=all_latents.device,
            dtype=torch.bool,
        )
        condition_valid_mask[:, -1] = True
        latents_by_id = getattr(
            self, "_streaming_condition_latents_by_id", None
        )
        masks_by_id = getattr(
            self, "_streaming_condition_valid_masks_by_id", None
        )
        if latents_by_id is None or masks_by_id is None:
            latents_by_id = {}
            masks_by_id = {}
            self._streaming_condition_latents_by_id = latents_by_id
            self._streaming_condition_valid_masks_by_id = masks_by_id
        stream_id = str(stream_id)
        latents_by_id[stream_id] = condition_latents
        masks_by_id[stream_id] = condition_valid_mask
        if stream_id == "default":
            self._streaming_condition_latents = condition_latents
            self._streaming_condition_valid_mask = condition_valid_mask
        return condition_latents

    @torch.no_grad()
    def update_streaming_history(
        self,
        input_images: torch.Tensor,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        """Consume four new stride-4 observations and shift H1..H4."""
        stream_id = str(stream_id)
        latents_by_id = getattr(
            self, "_streaming_condition_latents_by_id", None
        )
        masks_by_id = getattr(
            self, "_streaming_condition_valid_masks_by_id", None
        )
        previous = (
            None if latents_by_id is None else latents_by_id.get(stream_id)
        )
        previous_valid = (
            None if masks_by_id is None else masks_by_id.get(stream_id)
        )
        # Preserve checkpoints/callers created before named history banks.
        if stream_id == "default" and previous is None:
            previous = getattr(self, "_streaming_condition_latents", None)
            previous_valid = getattr(
                self, "_streaming_condition_valid_mask", None
            )
        if previous is None or previous_valid is None:
            raise RuntimeError(
                "TMD3 streaming history must be initialized before update: "
                f"stream_id={stream_id!r}."
            )
        input_images = self._normalize_inference_images(input_images)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if int(input_images.shape[1]) != temporal_factor:
            raise ValueError(
                "A TMD3 streaming update requires four new observations, "
                f"got T={input_images.shape[1]}."
            )
        newest = self._encode_streaming_camera_chunk(
            input_images,
            reset=False,
            stream_id=stream_id,
        )
        if int(newest.shape[2]) != 1:
            raise RuntimeError(
                "A four-image streaming update must emit one latent, "
                f"got {tuple(newest.shape)}."
            )
        condition_latents = torch.cat(
            [previous[:, :, 1:], newest],
            dim=2,
        ).contiguous()
        condition_valid_mask = torch.cat(
            [
                previous_valid[:, 1:],
                torch.ones(
                    (int(previous_valid.shape[0]), 1),
                    device=previous_valid.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        if latents_by_id is None or masks_by_id is None:
            latents_by_id = {}
            masks_by_id = {}
            self._streaming_condition_latents_by_id = latents_by_id
            self._streaming_condition_valid_masks_by_id = masks_by_id
        latents_by_id[stream_id] = condition_latents
        masks_by_id[stream_id] = condition_valid_mask
        if stream_id == "default":
            self._streaming_condition_latents = condition_latents
            self._streaming_condition_valid_mask = condition_valid_mask
        return condition_latents

    def get_streaming_condition_latents(
        self,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        stream_id = str(stream_id)
        by_id = getattr(self, "_streaming_condition_latents_by_id", None)
        condition_latents = None if by_id is None else by_id.get(stream_id)
        if stream_id == "default" and condition_latents is None:
            condition_latents = getattr(
                self, "_streaming_condition_latents", None
            )
        if condition_latents is None:
            raise RuntimeError(
                "TMD3 streaming history is not initialized: "
                f"stream_id={stream_id!r}."
            )
        return condition_latents

    def get_streaming_condition_valid_mask(
        self,
        *,
        stream_id: str = "default",
    ) -> torch.Tensor:
        stream_id = str(stream_id)
        by_id = getattr(
            self, "_streaming_condition_valid_masks_by_id", None
        )
        valid_mask = None if by_id is None else by_id.get(stream_id)
        if stream_id == "default" and valid_mask is None:
            valid_mask = getattr(
                self,
                "_streaming_condition_valid_mask",
                None,
            )
        if valid_mask is None:
            raise RuntimeError(
                "TMD3 streaming history is not initialized: "
                f"stream_id={stream_id!r}."
            )
        return valid_mask

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        """Encode the RobotWin three-view transport with independent streams."""
        if video_tensor.ndim != 5 or int(video_tensor.shape[1]) != 3:
            raise ValueError(
                "TMD3 VAE input must be [B,3,T,384,320], got "
                f"{tuple(video_tensor.shape)}."
            )
        batch_size = int(video_tensor.shape[0])
        high_video, left_video, right_video = split_robotwin_rgb(video_tensor)
        wrist_video = torch.stack(
            [left_video, right_video],
            dim=1,
        ).flatten(0, 1).contiguous()

        def encode_branch(branch_video: torch.Tensor) -> torch.Tensor:
            single_encode = getattr(
                getattr(self, "vae", None),
                "single_encode",
                None,
            )
            if tiled or not callable(single_encode):
                return FastWAM._encode_video_latents(
                    self,
                    branch_video,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            encode_batch_size = int(getattr(self, "vae_encode_batch_size", 4))
            if encode_batch_size <= 0:
                raise ValueError(
                    "`vae_encode_batch_size` must be positive, got "
                    f"{encode_batch_size}."
                )
            chunks = [
                single_encode(
                    branch_video[start : start + encode_batch_size],
                    device=self.device,
                )
                for start in range(0, branch_video.shape[0], encode_batch_size)
            ]
            return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

        high_latent = encode_branch(high_video)
        wrist_latents = encode_branch(wrist_video)
        if int(high_latent.shape[0]) != batch_size:
            raise RuntimeError(
                f"High VAE batch mismatch: got={high_latent.shape[0]}, "
                f"expected={batch_size}."
            )
        if int(wrist_latents.shape[0]) != batch_size * 2:
            raise RuntimeError(
                f"Wrist VAE batch mismatch: got={wrist_latents.shape[0]}, "
                f"expected={batch_size * 2}."
            )
        wrist_latents = wrist_latents.reshape(
            batch_size,
            2,
            *wrist_latents.shape[1:],
        )
        return compose_robotwin_latents(
            high_latent,
            wrist_latents[:, 0],
            wrist_latents[:, 1],
        )

    def build_inputs(self, sample: dict[str, Any], tiled: bool = False):
        video = sample.get("video")
        if not torch.is_tensor(video) or video.ndim != 5:
            raise ValueError(
                "TMD3 training expects RobotWin RGB transport video "
                "[B,3,T,384,320]."
            )
        precomputed_latents = sample.get("precomputed_video_latents")
        if precomputed_latents is None:
            if tuple(video.shape[-2:]) != RGB_TRANSPORT_SIZE:
                raise ValueError(
                    "TMD3 training video spatial layout mismatch: "
                    f"expected={RGB_TRANSPORT_SIZE}, got={tuple(video.shape)}."
                )
        else:
            # The fast-I/O dataset deliberately transports only B/C/T in a
            # 1x1 RGB placeholder when the VAE result is already cached.  The
            # parent implementation never encodes this placeholder; it uses
            # the cached tensor and validates its full B/C/T shape against the
            # VAE contract.  Keep a TMD3-specific spatial guard here so an
            # ordinary single-camera latent cache cannot silently enter the
            # RobotWin [left|right;high] path.
            if not torch.is_tensor(precomputed_latents):
                raise TypeError(
                    "`precomputed_video_latents` must be a tensor, got "
                    f"{type(precomputed_latents)}."
                )
            if precomputed_latents.ndim != 5:
                raise ValueError(
                    "TMD3 precomputed latents must be [B,C,T,H,W], got "
                    f"{tuple(precomputed_latents.shape)}."
                )
            if tuple(precomputed_latents.shape[-2:]) != LATENT_SIZE:
                raise ValueError(
                    "TMD3 precomputed latent layout mismatch: expected "
                    f"{LATENT_SIZE}, got "
                    f"{tuple(precomputed_latents.shape)}."
                )
            if tuple(video.shape[-2:]) not in {
                RGB_TRANSPORT_SIZE,
                (1, 1),
            }:
                raise ValueError(
                    "TMD3 cached-latent RGB carrier must use its configured "
                    "real layout or the 1x1 fast-I/O "
                    f"placeholder, got {tuple(video.shape)}."
                )
        if int(video.shape[1]) != 3:
            raise ValueError(
                "TMD3 training RGB carrier must have three channels, got "
                f"{tuple(video.shape)}."
            )
        return super().build_inputs(sample, tiled=tiled)


RobometerTMD = ProgressTMD


__all__ = [
    "ProgressTMD",
    "RobometerTMD",
    "TMD3_CAMERA_ORDER",
]
