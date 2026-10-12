"""Causal, group-shared augmentation of *inputs*, never of final-layer KV.

The init anchor is immutable. Current images, state, text and action/video
targets are not altered. No cross-episode replacement or privileged labels.
"""
from __future__ import annotations

import torch


def augment_event_memory(
    *, latents, frame_indices, valid_mask, group_ids, current_frames,
    candidate_latents, candidate_indices,
    probabilities=(0.5, 0.2, 0.2, 0.1), noise_max=0.1,
):
    p = torch.tensor(probabilities, dtype=torch.float64)
    if p.shape != (4,) or (p < 0).any() or not torch.isclose(p.sum(), p.new_tensor(1.0)):
        raise ValueError("Memory probabilities must be four nonnegative values summing to one.")
    if not 0 < float(noise_max) <= 0.2:
        raise ValueError("Memory noise_max must be in (0, 0.2].")
    b, c, k, h, w = latents.shape
    indices = frame_indices.detach().cpu().long()
    valid = valid_mask.detach().cpu().bool()
    ids = group_ids.detach().cpu().long().reshape(-1)
    current = current_frames.detach().cpu().long().reshape(-1)
    if indices.shape != (b, k) or valid.shape != (b, k) or current.shape != (b,) or ids.shape != (b,):
        raise ValueError("Memory metadata does not match [B,K].")
    if not valid[:, 0].all() or not (indices[:, 0] == 0).all():
        raise ValueError("Every sample must retain fixed init at frame 0.")
    if ((indices > current[:, None]) & valid).any():
        raise ValueError("Original memory contains a future observation.")
    if candidate_latents is None or candidate_indices is None:
        raise ValueError("Memory augmentation requires the precomputed candidate bank.")
    ci = candidate_indices.detach().cpu().long()
    if ci.ndim != 3 or ci.shape[:2] != (b, k) or tuple(candidate_latents.shape) != (b, k, ci.shape[2], c, h, w):
        raise ValueError("Candidate bank must be [B,K,M,C,H,W] with [B,K,M] indices.")

    out = latents.clone()
    out_idx, out_valid = indices.clone(), valid.clone()
    requested = torch.zeros(4, dtype=torch.float64)
    applied = torch.zeros(4, dtype=torch.float64)
    shift_sum = shifted_count = positive_count = late_count = 0.0
    alpha_sum = noise_count = dropped_count = 0.0
    for group in torch.unique(ids, sorted=True):
        rows = torch.nonzero(ids == group).flatten()
        r, n = int(rows[0]), len(rows)
        if not torch.equal(indices[rows], indices[r].expand(n, -1)) or not torch.equal(valid[rows], valid[r].expand(n, -1)):
            raise ValueError("Shared memory group has inconsistent anchors/masks.")
        if not torch.equal(ci[rows], ci[r].expand(n, -1, -1)):
            raise ValueError("Shared memory group has inconsistent candidate banks.")
        group_limit = int(current[rows].min())
        positions = torch.nonzero(valid[r]).flatten().tolist()
        if positions != sorted(positions) or not torch.all(indices[r, positions][1:] > indices[r, positions][:-1]):
            raise ValueError("Historical anchors must be strictly chronological.")
        z = latents[r].clone()
        ix, keep = indices[r].clone(), valid[r].clone()
        mode = int(torch.multinomial(p, 1))
        requested[mode] += n
        effective = 0
        rolling = positions[1:]
        if rolling and mode == 1:
            previous = 0
            for ordinal, slot in enumerate(rolling):
                next_original = int(indices[r, rolling[ordinal + 1]]) if ordinal + 1 < len(rolling) else group_limit + 1
                candidates = ci[r, slot]
                ok = (candidates > previous) & (candidates < next_original) & (candidates <= group_limit) & (candidates != indices[r, slot])
                choices = torch.nonzero(ok).flatten()
                # Give already-observed delayed captures explicit coverage.
                # This uses timestamps only, never wall/target/GT stage labels.
                delayed = torch.nonzero(ok & (candidates >= indices[r, slot] + 4)).flatten()
                if len(delayed) and float(torch.rand(())) < 0.5:
                    choices = delayed
                if len(choices):
                    choice = int(choices[int(torch.randint(len(choices), ()))])
                    ix[slot] = candidates[choice]
                    z[:, slot] = candidate_latents[r, slot, choice].to(z.device, z.dtype)
                    delta = int(ix[slot] - indices[r, slot])
                    shift_sum += abs(delta) * n
                    shifted_count += n
                    positive_count += (delta > 0) * n
                    late_count += (delta >= 4) * n
                    effective = 1
                previous = int(ix[slot])
        elif rolling and mode == 2:
            count = int(torch.randint(1, len(rolling) + 1, ()))
            for selected in torch.randperm(len(rolling))[:count].tolist():
                slot = rolling[selected]
                keep[slot] = False
                ix[slot] = -1
                z[:, slot] = 0
            dropped_count += count * n
            effective = 2
        elif rolling and mode == 3:
            for slot in rolling:
                alpha = float(torch.rand(())) * float(noise_max)
                z[:, slot] = (1.0 - alpha) * z[:, slot] + alpha * torch.randn_like(z[:, slot])
                alpha_sum += alpha * n
                noise_count += n
            effective = 3
        applied[effective] += n
        out.index_copy_(0, rows.to(out.device), z.unsqueeze(0).expand(n, -1, -1, -1, -1))
        out_idx[rows], out_valid[rows] = ix, keep

    if ((out_idx > current[:, None]) & out_valid).any():
        raise AssertionError("Augmented memory leaked a future frame.")
    init_unchanged = (out[:, :, 0] == latents[:, :, 0]).all()
    if latents.device.type == "cuda":
        # Keep the invariant check without synchronizing the training stream
        # back to Python each batch. This does not touch any RNG state.
        torch._assert_async(init_unchanged, "Augmentation altered init.")
    elif not bool(init_unchanged):
        raise AssertionError("Augmentation altered init.")
    names = ("raw", "time", "drop", "noise")
    metrics = {f"mem_requested_{name}": float(requested[i] / b) for i, name in enumerate(names)}
    metrics.update({f"mem_{name}": float(applied[i] / b) for i, name in enumerate(names)})
    metrics.update(
        mem_valid=float(out_valid.sum()) / b,
        mem_shift_sum=shift_sum / b, mem_shift_count=shifted_count / b,
        mem_positive_shift_count=positive_count / b, mem_late_shift_count=late_count / b,
        mem_noise_alpha_sum=alpha_sum / b, mem_noise_count=noise_count / b,
        mem_dropped_count=dropped_count / b,
    )
    return out, out_idx.to(frame_indices.device), out_valid.to(valid_mask.device), metrics
