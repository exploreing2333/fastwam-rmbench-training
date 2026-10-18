"""MemoryWAM attention with exactly one persistent, full-resolution init frame.

The paper's other memory slots are removed. Each autoregressive block sees
init + its current observation, and never another block's observations/actions.
Grouping those independent blocks is an exact factorization of that mask, not
truncated backpropagation. The init branch remains differentiable.

The public paper does not specify action spatial markers/sub-frame coordinates
or the logit-normal loss-weight formula. These choices are explicit in config.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fastwam import FastWAM
from .wan_video_dit import sinusoidal_embedding_1d


class MemoryWAMInit(FastWAM):
    def configure_memorywam(self, config):
        self.memorywam_config = dict(config)
        if self.init_frame_context_injection or self.event_memory_video_conditioning:
            raise ValueError("MemoryWAM requires its own native video-KV path.")
        if self.memorywam_config.get("anchor_frames", 1) != 1:
            raise ValueError("Exactly one initial frame is supported.")
        if self.memorywam_config.get("recent_frames", 0) or self.memorywam_config.get("gist_tokens", 0):
            raise ValueError("Additional persistent memories must be disabled.")
        self.memorywam_horizon = int(self.memorywam_config.get("action_horizon", 16))
        if self.memorywam_horizon != 16:
            raise ValueError("MemoryWAM uses 16 actions per stride-4/VAE-stride-4 frame.")
        self._memorywam_anchor_cache = None

    def shared_rope(self, time, row, col):
        """Use the pretrained video's actual temporal/height/width frequency bases."""
        coordinates = torch.broadcast_tensors(time, row, col)
        parts = []
        for coordinate, basis in zip(coordinates, self.video_expert.freqs):
            frequency = torch.angle(basis[1]).to(coordinate.device)
            phase = coordinate.to(frequency.dtype).unsqueeze(-1) * frequency
            parts.append(torch.polar(torch.ones_like(phase), phase))
        return torch.cat(parts, dim=-1).unsqueeze(-2)

    def video_rope(self, frame_times, height, width):
        device = frame_times.device
        time = frame_times[:, None, None].expand(-1, height, width)
        row = torch.arange(height, device=device)[None, :, None].expand_as(time)
        col = torch.arange(width, device=device)[None, None, :].expand_as(time)
        return self.shared_rope(time.flatten(), row.flatten(), col.flatten())

    def action_rope(self, frame_times, horizon):
        # The paper specifies a shared 3D basis, but not exact action coordinates.
        # Use time t+j/16 and the explicitly configured fixed spatial marker.
        time = frame_times[:, None] + torch.arange(horizon, device=frame_times.device)[None] / self.memorywam_horizon
        marker = torch.full_like(time, float(self.memorywam_config.get("action_spatial_marker", 0)))
        return self.shared_rope(time, marker, marker)

    def _video_pre(self, latents, times, frame_times, context, context_mask):
        batch, _, frames, height, width = latents.shape
        pre = self.video_expert.pre_dit(
            x=latents, timestep=times[:, 0], context=context,
            context_mask=context_mask, action=None,
            fuse_vae_embedding_in_latents=True, num_clean_frames=0,
        )
        patch_h, patch_w = self.video_expert.patch_size[1:]
        tokens = pre["meta"]["tokens_per_frame"]
        # Embed one timestep per frame, then expand; this avoids embedding the
        # same timestep independently for all 120 spatial patches.
        t = self.video_expert.time_embedding(sinusoidal_embedding_1d(
            self.video_expert.freq_dim, times.flatten()))
        t = t.reshape(batch, frames, 1, -1).expand(-1, -1, tokens, -1).reshape(batch, frames * tokens, -1)
        pre["t"] = t
        pre["t_mod"] = self.video_expert.time_projection(t).unflatten(-1, (6, self.video_expert.hidden_dim))
        pre["freqs"] = self.video_rope(frame_times, height // patch_h, width // patch_w)
        return pre

    @staticmethod
    def block_mask(tokens_per_frame, horizon, *, device, first_block=False, include_noisy=True):
        """Rows: current[, noisy], action. Columns: init + those same tokens."""
        p = tokens_per_frame
        video = p * (2 if include_noisy else 1)
        mask = torch.zeros(video + horizon, p + video + horizon, dtype=torch.bool, device=device)
        if not first_block:
            mask[:, :p] = True
        mask[:, p:2*p] = True
        if include_noisy:
            mask[p:2*p, 2*p:3*p] = True
        mask[video:, p+video:] = True
        return mask

    def forward_sequence(self, clean, noisy_next, noisy_actions, video_times, action_times,
                         context, context_mask, proprio, *, use_checkpoint=None):
        """Parallel, exact init-only autoregression over one or more episodes."""
        batch, channels, frames, height, width = clean.shape
        horizon = noisy_actions.shape[2]
        if noisy_next.shape != clean.shape or noisy_actions.shape[:2] != (batch, frames):
            raise ValueError("Clean/noisy video and action chunk alignment differs.")
        if horizon != self.memorywam_horizon:
            raise ValueError("Each autoregressive block requires 16 action tokens.")
        # Video sequence: anchor, clean_0, noisy_1, clean_1, noisy_2, ...
        interleaved = torch.stack([clean, noisy_next], dim=3).flatten(2, 3)
        latents = torch.cat([clean[:, :, :1], interleaved], dim=2)
        times = torch.stack([torch.zeros_like(video_times), video_times], dim=2).flatten(1)
        times = torch.cat([times[:, :1] * 0, times], dim=1)
        ids = torch.arange(frames, device=clean.device)
        frame_times = torch.cat([ids[:1], torch.stack([ids, ids + 1], dim=1).flatten()])
        vp = self._video_pre(latents, times, frame_times, context, context_mask)
        p = vp["meta"]["tokens_per_frame"]
        action_context = context[:, None].expand(-1, frames, -1, -1).flatten(0, 1)
        action_context_mask = context_mask[:, None].expand(-1, frames, -1).flatten(0, 1)
        action_context, action_context_mask = self._append_proprio_to_context(
            action_context, action_context_mask, proprio.flatten(0, 1))
        ap = self.action_expert.pre_dit(noisy_actions.flatten(0, 1), action_times.flatten(),
                                      action_context, action_context_mask)
        ap["freqs"] = self.action_rope(ids.repeat(batch), horizon)
        mask = self.block_mask(p, horizon, device=clean.device)
        masks = mask[None, None].expand(batch * frames, 1, -1, -1).clone()
        masks[torch.arange(batch, device=clean.device) * frames, :, :, :p] = False
        anchor_mask = torch.ones(p, p, dtype=torch.bool, device=clean.device)
        vx, ax = vp["tokens"], ap["tokens"]
        use_checkpoint = self.memorywam_config.get("checkpoint_layers", True) if use_checkpoint is None else use_checkpoint
        for layer in range(self.mot.num_layers):
            vb, ab = self.video_expert.blocks[layer], self.action_expert.blocks[layer]

            def layer_forward(vx, ax, vt, at, vc, ac, vb=vb, ab=ab):
                vi = self.mot._build_expert_attention_io(self.video_expert, vb, vx, vp["freqs"], vt)
                ai = self.mot._build_expert_attention_io(self.action_expert, ab, ax, ap["freqs"], at)
                vq, vk, vv = vi[:3]
                aq, ak, av = ai[:3]
                anchor_out = self.mot._mixed_attention(vq[:, :p], vk[:, :p], vv[:, :p], anchor_mask)

                def grouped(x):
                    return x[:, p:].reshape(batch * frames, 2*p, -1)

                def prefix(x):
                    return x[:, :p, None].transpose(1, 2).expand(-1, frames, -1, -1).reshape(batch * frames, p, -1)

                mixed = self.mot._mixed_attention(
                    torch.cat([grouped(vq), aq], dim=1),
                    torch.cat([prefix(vk), grouped(vk), ak], dim=1),
                    torch.cat([prefix(vv), grouped(vv), av], dim=1), masks)
                video_out = torch.cat([anchor_out, mixed[:, :2*p].reshape(batch, frames * 2*p, -1)], dim=1)
                vx = self.mot._apply_expert_post_block(vb, vi[3], video_out, *vi[4:8],
                    {"context": vc, "mask": vp["context_mask"]})
                ax = self.mot._apply_expert_post_block(ab, ai[3], mixed[:, 2*p:], *ai[4:8],
                    {"context": ac, "mask": ap["context_mask"]})
                return vx, ax

            args = (vx, ax, vp["t_mod"], ap["t_mod"], vp["context"], ap["context"])
            vx, ax = checkpoint(layer_forward, *args, use_reentrant=False) if use_checkpoint and torch.is_grad_enabled() else layer_forward(*args)
        # Predict only noisy video blocks. Clean/anchor outputs have no loss.
        out_tokens = vx[:, p:].reshape(batch, frames, 2, p, -1)[:, :, 1].reshape(batch, frames*p, -1)
        out_t = vp["t"][:, p:].reshape(batch, frames, 2, p, -1)[:, :, 1].reshape(batch, frames*p, -1)
        pred_video = self.video_expert.head(out_tokens, out_t)
        pred_video = self.video_expert.unpatchify(pred_video, (frames, height // self.video_expert.patch_size[1], width // self.video_expert.patch_size[2]))
        pred_action = self.action_expert.post_dit(ax, ap).reshape(batch, frames, horizon, -1)
        return pred_video, pred_action

    def sample_logit_normal(self, shape, shift, device):
        u = torch.randn(shape, device=device, dtype=torch.float32).sigmoid()
        return shift * u / (1 + (shift - 1) * u)

    @staticmethod
    def logit_normal_weight(sigma, shift):
        # Explicit interpretation of the paper's unspecified "logit-normal
        # training weight": the density of the shifted logistic-normal.
        sigma = sigma.float().clamp(1e-5, 1-1e-5)
        denominator = shift - (shift-1) * sigma
        u = (sigma / denominator).clamp(1e-5, 1-1e-5)
        return torch.exp(-0.5 * torch.logit(u).square()) / (math.sqrt(2*math.pi) * u * (1-u)) * shift / denominator.square()

    def training_loss(self, sample, **kwargs):
        del kwargs
        latent = sample["episode_latents"].to(self.device, self.torch_dtype)
        action = sample["action"].to(self.device, self.torch_dtype)
        proprio = sample["proprio"].to(self.device, self.torch_dtype)
        context = sample["context"].to(self.device, self.torch_dtype)
        context_mask = sample["context_mask"].to(self.device, torch.bool)
        batch, _, total, _, _ = latent.shape
        frames = total - 1
        sigma_v = self.sample_logit_normal((batch, frames), 5.0, latent.device)
        sigma_a = self.sample_logit_normal((batch, frames), 1.0, latent.device)
        clean, target = latent[:, :, :-1], latent[:, :, 1:]
        # p=1, independent Uniform[0,1] corruption of each clean condition.
        ratio = torch.rand(batch, 1, frames, 1, 1, device=latent.device, dtype=latent.dtype)
        clean = (1-ratio) * clean + ratio * torch.randn_like(clean)
        nv, na = torch.randn_like(target), torch.randn_like(action)
        sv = sigma_v[:, None, :, None, None].to(latent.dtype)
        sa = sigma_a[:, :, None, None].to(action.dtype)
        pv, pa = self.forward_sequence(clean, (1-sv)*target + sv*nv,
            (1-sa)*action + sa*na, (sigma_v*1000).to(latent.dtype), (sigma_a*1000).to(action.dtype),
            context, context_mask, proprio)
        lv = F.mse_loss(pv.float(), (nv-target).float(), reduction="none").mean((1, 3, 4))
        la = F.mse_loss(pa.float(), (na-action).float(), reduction="none").mean(-1)
        valid_a = ~sample["action_is_pad"].to(action.device)
        valid_v = ~sample["video_is_pad"].to(action.device)
        wv = self.logit_normal_weight(sigma_v, 5.0)
        wa = self.logit_normal_weight(sigma_a, 1.0)
        lv = (lv * wv * valid_v).sum(1) / valid_v.sum(1).clamp_min(1)
        la = ((la * valid_a).sum(2) / valid_a.sum(2).clamp_min(1)) * wa
        la = (la * valid_v).sum(1) / valid_v.sum(1).clamp_min(1)
        weight = sample.get("sample_weight", torch.ones(batch, device=latent.device)).to(latent.device)
        lv, la = (lv*weight).mean(), (la*weight).mean()
        loss = self.loss_lambda_video * lv + self.loss_lambda_action * la
        return loss, {"loss_video": lv.detach(), "loss_action": la.detach(),
                      "action_chunks": valid_v.sum().detach().float()}

    def clear_event_memory_cache(self):
        super().clear_event_memory_cache()
        self._memorywam_anchor_cache = None

    @torch.no_grad()
    def encode_observation_chunk(self, video, *, reset=False):
        """RGB is sampled every four controls; first call has one frame,
        subsequent calls have four frames per 16-control action chunk.
        Input mosaic is wrists on top, head on bottom, normalized to [-1,1].
        The frozen VAE's causal convolution state follows the paper's encoder;
        only the init frame is retained as transformer memory.
        """
        if reset:
            self.clear_event_memory_cache()
        video = video.to(self.device, self.torch_dtype)
        return self.vae.model.encode_streaming(video, self.vae.scale, reset=reset)

    @torch.no_grad()
    def infer_from_latents(self, *, init_latent, current_latent, frame_index,
                           context, context_mask, proprio, seed=42,
                           num_inference_steps=50, episode_key=None):
        """Streaming caller supplies causal-VAE latents and resets per episode."""
        if torch.is_grad_enabled() or self.video_expert.training:
            raise RuntimeError("Call eval() before inference KV caching.")
        device, dtype = current_latent.device, current_latent.dtype
        context = context.to(device, dtype)
        context_mask = context_mask.to(device, torch.bool)
        if init_latent.shape[2] != 1 or current_latent.shape[2] != 1:
            raise ValueError("One full init frame and one current frame are required.")
        zero = torch.zeros(1, 1, device=device, dtype=dtype)
        saved = self._memorywam_anchor_cache
        signature = (episode_key, tuple(p._version for p in self.video_expert.parameters()))
        if saved is None or saved[0] != signature or not torch.equal(saved[1], init_latent) or not torch.equal(saved[2], context) or not torch.equal(saved[3], context_mask):
            ip = self._video_pre(init_latent, zero, zero[0], context, context_mask)
            p = ip["tokens"].shape[1]
            cache = self.mot.prefill_video_cache(ip["tokens"], ip["freqs"], ip["t_mod"],
                {"context": ip["context"], "mask": ip["context_mask"]},
                torch.ones(p, p, dtype=torch.bool, device=device))
            saved = (signature, init_latent.clone(), context.clone(), context_mask.clone(), cache)
            self._memorywam_anchor_cache = saved
        cache = saved[4]
        p = cache[0]["k"].shape[1]
        if frame_index != 0:
            cp = self._video_pre(current_latent, zero, zero[0] + frame_index, context, context_mask)
            cache = self.mot.append_video_with_cache(cp["tokens"], cp["freqs"], cp["t_mod"],
                {"context": cp["context"], "mask": cp["context_mask"]}, cache,
                torch.ones(p, 2*p, dtype=torch.bool, device=device), p)
        count = cache[0]["k"].shape[1]
        act_context, act_mask = self._append_proprio_to_context(context, context_mask, proprio.to(device, dtype))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        actions = torch.randn(1, self.memorywam_horizon, self.action_expert.action_dim, generator=generator).to(device, dtype)
        times, deltas = self.infer_action_scheduler.build_inference_schedule(num_inference_steps, device, dtype)
        freqs = self.action_rope(torch.tensor([frame_index], device=device), self.memorywam_horizon)
        mask = self._build_cache_attention_mask(video_seq_len=count, action_seq_len=self.memorywam_horizon, device=device)
        for t, delta in zip(times, deltas):
            ap = self.action_expert.pre_dit(actions, t[None], act_context, act_mask)
            out = self.mot.forward_action_with_video_cache(ap["tokens"], freqs, ap["t_mod"],
                {"context": ap["context"], "mask": ap["context_mask"]}, cache, mask, count)
            pred = self.action_expert.post_dit(out, ap)
            actions = self.infer_action_scheduler.step(pred, delta, actions)
        return {"action": actions[0].float().cpu()}

    def infer_action(self, **kwargs):
        raise RuntimeError("MemoryWAM requires infer_from_latents with causal VAE streaming, fixed task text, and horizon=16.")
