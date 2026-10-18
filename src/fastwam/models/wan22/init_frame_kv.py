"""Add one episode-start visual KV prefix to the original subtask FastWAM.

The dataset, sampler, 32-control target, noise schedulers and loss are inherited
unchanged. Only the input-conditioning path is extended. Init is encoded once
per episode/subtask group within a batch, with gradients through the video DiT.
At inference its KV can be reused until the subtask or model weights change.
"""
from __future__ import annotations

import torch

from .fastwam import FastWAM
from .memorywam_init import MemoryWAMInit


class InitFrameKVFastWAM(FastWAM):
    # Share the published 3D frequency basis, without inheriting episode training.
    shared_rope = MemoryWAMInit.shared_rope

    def configure_init_frame_kv(self, config):
        self.init_frame_kv_config = dict(config)
        if self.init_frame_context_injection or not self.event_memory_video_conditioning:
            raise ValueError("Init KV requires video conditioning, without T5-token injection.")
        if self.init_frame_kv_config != {
            "anchor_frames": 1, "recent_frames": 0, "gist_tokens": 0,
            "control_steps_per_latent": 16, "position_encoding": "shared_video_3d_rope",
        }:
            raise ValueError("Expected exactly one init frame and the original stride-4 VAE cadence.")
        if self.event_memory_augmentation or self.progress_conditioning_bins is not None:
            raise ValueError("This init-only experiment preserves the original GT subtask conditions.")
        self._init_current_frame = None
        self._init_current_proprio = None

    def _frame_times(self, batch, device):
        if self._init_current_frame is None:
            raise ValueError("Episode-relative current frame indices are required for init KV RoPE.")
        frames = torch.as_tensor(self._init_current_frame, device=device).reshape(-1)
        if frames.numel() != batch:
            raise ValueError("Current frame indices must match the sample batch.")
        return frames.to(torch.float64) / 16

    def _video_freqs(self, pre, frame_times):
        frames, height, width = pre["meta"]["grid_size"]
        device = frame_times.device
        time = (frame_times[:, None] + torch.arange(frames, device=device)[None])[:, :, None, None]
        row = torch.arange(height, device=device)[None, None, :, None]
        col = torch.arange(width, device=device)[None, None, None, :]
        return self.shared_rope(time, row, col).reshape(frame_times.numel(), -1, 1,
                                                     sum(x.shape[1] for x in self.video_expert.freqs))

    def _action_freqs(self, frame_times, horizon):
        time = frame_times[:, None] + torch.arange(horizon, device=frame_times.device)[None] / 16
        # The paper specifies the common basis, but not exact action coordinates.
        return self.shared_rope(time, torch.zeros_like(time), torch.zeros_like(time))

    def training_loss(self, sample, tiled=False):
        self._init_current_frame = sample["frame_index"]
        try:
            # Retain the original per-clip timesteps, target construction, masks,
            # terminal-hold supervision and loss weighting exactly.
            return super().training_loss(sample, tiled=tiled)
        finally:
            self._init_current_frame = None

    def _grouped_event_predictions(
        self, *, current_latents, event_anchor_latents, event_anchor_frame_indices,
        event_anchor_valid_mask, event_anchor_group_id, event_context,
        event_context_mask, action, timestep_action, noisy_action,
        progress_context, progress_context_mask, progress_modulations,
        fuse_vae_embedding_in_latents, noisy_future_latents=None, timestep_video=None,
    ):
        batch = current_latents.shape[0]
        if event_anchor_latents.shape[2] != 1 or event_anchor_frame_indices.shape != (batch, 1):
            raise ValueError("Only one full frame-zero anchor is allowed.")
        if not bool((event_anchor_frame_indices == 0).all()) or not bool(event_anchor_valid_mask.all()):
            raise ValueError("Every sample must carry its valid episode frame zero.")
        if noisy_future_latents is None or noisy_action.shape[1] != 32:
            raise ValueError("Keep the original 32-control subtask training target.")
        if self.video_expert.action_conditioned:
            raise ValueError("This experiment requires the original action_conditioned=false.")

        # This shares identical init computations; it neither reorders nor resamples
        # examples. Different episodes/subtasks always have separate prefixes.
        representatives, lookup, inverse = [], {}, []
        for i, group in enumerate(event_anchor_group_id.detach().cpu().tolist()):
            if group not in lookup:
                lookup[group] = len(representatives)
                representatives.append(i)
            inverse.append(lookup[group])
        device = current_latents.device
        representatives = torch.tensor(representatives, device=device)
        inverse = torch.tensor(inverse, device=device)
        init_pre = self.video_expert.pre_dit(
            x=event_anchor_latents.index_select(0, representatives),
            timestep=torch.zeros(len(representatives), device=device, dtype=current_latents.dtype),
            context=event_context.index_select(0, representatives),
            context_mask=event_context_mask.index_select(0, representatives),
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_pre = self.video_expert.pre_dit(
            x=torch.cat([current_latents, noisy_future_latents], dim=2),
            timestep=timestep_video, context=progress_context,
            context_mask=progress_context_mask, action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=progress_context, context_mask=progress_context_mask,
        )
        frame_times = self._frame_times(batch, device)
        video_pre["freqs"] = self._video_freqs(video_pre, frame_times)
        action_pre["freqs"] = self._action_freqs(frame_times, noisy_action.shape[1])
        self._apply_progress_modulation(video_pre, "video", progress_modulations)
        self._apply_progress_modulation(action_pre, "action", progress_modulations)
        ix, vx, ax = init_pre["tokens"], video_pre["tokens"], action_pre["tokens"]
        p, nv, na = ix.shape[1], vx.shape[1], ax.shape[1]
        old_mask = self._build_mot_attention_mask(nv, na, p, device)
        # Original current/future/action visibility is unchanged; all additionally
        # read init. Init itself cannot read current, future or action tokens.
        mask = torch.cat([torch.ones(nv + na, p, dtype=torch.bool, device=device), old_mask], dim=1)
        init_mask = torch.ones(p, p, dtype=torch.bool, device=device)
        ipayload = {"context": init_pre["context"], "mask": init_pre["context_mask"]}
        vpayload = {"context": video_pre["context"], "mask": video_pre["context_mask"]}
        apayload = {"context": action_pre["context"], "mask": action_pre["context_mask"]}
        for layer in range(self.mot.num_layers):
            vb, ab = self.video_expert.blocks[layer], self.action_expert.blocks[layer]
            ii = self.mot._build_expert_attention_io(self.video_expert, vb, ix, init_pre["freqs"], init_pre["t_mod"])
            vi = self.mot._build_expert_attention_io(self.video_expert, vb, vx, video_pre["freqs"], video_pre["t_mod"])
            ai = self.mot._build_expert_attention_io(self.action_expert, ab, ax, action_pre["freqs"], action_pre["t_mod"])
            init_out = self.mot._mixed_attention(ii[0], ii[1], ii[2], init_mask)
            mixed = self.mot._mixed_attention(
                torch.cat([vi[0], ai[0]], dim=1),
                torch.cat([ii[1].index_select(0, inverse), vi[1], ai[1]], dim=1),
                torch.cat([ii[2].index_select(0, inverse), vi[2], ai[2]], dim=1), mask,
            )
            ix = self.mot._apply_post_with_optional_checkpoint(vb, *ii[3:], init_out, ipayload)
            vx = self.mot._apply_post_with_optional_checkpoint(vb, *vi[3:], mixed[:, :nv], vpayload)
            ax = self.mot._apply_post_with_optional_checkpoint(ab, *ai[3:], mixed[:, nv:], apayload)
        video = self.video_expert.post_dit(vx, video_pre)[:, :, 1:]
        return video, self.action_expert.post_dit(ax, action_pre)

    def _build_event_memory_cache(
        self, *, current_latents, event_anchor_latents, context, context_mask,
        fuse_vae_embedding_in_latents, event_anchor_valid_mask=None,
        event_memory_cache_key=None, event_memory_mask_mode="none",
    ):
        if event_anchor_latents is None or event_anchor_latents.shape[2] != 1:
            raise ValueError("Inference requires the complete episode-start frame latent.")
        if event_anchor_valid_mask is not None and not bool(event_anchor_valid_mask.all()):
            raise ValueError("The initial frame must be valid.")
        if event_memory_mask_mode not in {"none", "all", "rolling"}:
            raise ValueError("Unknown memory ablation mode.")
        prefix, count, _ = self._event_prefix_for_inference(
            cache_key=event_memory_cache_key, event_anchor_latents=event_anchor_latents,
            context=context, context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        # Preserve the original current-video proprioception conditioning.
        if self._init_current_proprio is not None:
            proprio = torch.as_tensor(self._init_current_proprio, device=context.device, dtype=context.dtype)
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio.reshape(context.shape[0], -1))
        pre = self.video_expert.pre_dit(
            x=current_latents, timestep=torch.zeros(current_latents.shape[0], device=current_latents.device, dtype=current_latents.dtype),
            context=context, context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        pre["freqs"] = self._video_freqs(pre, self._frame_times(current_latents.shape[0], current_latents.device))
        n = pre["tokens"].shape[1]
        mask = torch.ones(n, count+n, dtype=torch.bool, device=current_latents.device)
        if event_memory_mask_mode == "all":
            mask[:, :count] = False
        cache = self.mot.append_video_with_cache(
            pre["tokens"], pre["freqs"], pre["t_mod"],
            {"context": pre["context"], "mask": pre["context_mask"]}, prefix, mask, count,
        )
        return cache, count+n

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self, latents_action, timestep_action, context, context_mask,
        video_kv_cache, attention_mask, video_seq_len, progress_modulations=None,
    ):
        pre = self.action_expert.pre_dit(latents_action, timestep_action, context, context_mask)
        freqs = self._action_freqs(self._frame_times(latents_action.shape[0], latents_action.device), latents_action.shape[1])
        self._apply_progress_modulation(pre, "action", progress_modulations)
        out = self.mot.forward_action_with_video_cache(
            pre["tokens"], freqs, pre["t_mod"],
            {"context": pre["context"], "mask": pre["context_mask"]},
            video_kv_cache, attention_mask, video_seq_len,
        )
        return self.action_expert.post_dit(out, pre)

    @torch.no_grad()
    def infer_action(self, *args, current_frame_index=None, **kwargs):
        if current_frame_index is None:
            raise ValueError("Pass current_frame_index in episode control steps for init KV RoPE.")
        self._init_current_frame = [int(current_frame_index)]
        self._init_current_proprio = kwargs.get("proprio")
        try:
            return super().infer_action(*args, **kwargs)
        finally:
            self._init_current_frame = self._init_current_proprio = None

    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError("Use infer_action with current_frame_index for this subtask init-KV run.")
