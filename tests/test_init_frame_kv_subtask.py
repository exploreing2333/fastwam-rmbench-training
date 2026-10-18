"""Init information reaches both losses without changing subtask supervision."""
from unittest.mock import patch

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.init_frame_kv import InitFrameKVFastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


@pytest.fixture
def model():
    torch.manual_seed(42)
    common = dict(hidden_dim=32, ffn_dim=64, text_dim=16, freq_dim=8,
                  num_heads=2, attn_head_dim=8, num_layers=3, eps=1e-6,
                  use_gradient_checkpointing=False)
    video = WanVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
                       has_image_input=False, seperated_timestep=True,
                       video_attention_mask_mode="first_frame_causal")
    action = ActionDiT(**common, action_dim=14)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    vae = torch.nn.Identity()
    vae.temporal_downsample_factor = 4
    result = InitFrameKVFastWAM(video, action, mot, vae, text_dim=16,
                               proprio_dim=14, torch_dtype=torch.float32,
                               event_memory_video_conditioning=True)
    result.configure_init_frame_kv(dict(anchor_frames=1, recent_frames=0, gist_tokens=0,
        control_steps_per_latent=16, position_encoding="shared_video_3d_rope"))
    return result


def inputs(model):
    torch.manual_seed(7)
    context = torch.randn(1, 5, 16).expand(2, -1, -1).clone()
    mask = torch.ones(2, 5, dtype=torch.bool)
    proprio = torch.randn(2, 14)
    pc, pm = model._append_proprio_to_context(context, mask, proprio)
    anchors = torch.randn(1, 4, 1, 4, 4).expand(2, -1, -1, -1, -1).clone().requires_grad_()
    model._init_current_frame = torch.tensor([32, 155])
    return dict(current_latents=torch.randn(2, 4, 1, 4, 4),
        event_anchor_latents=anchors, event_anchor_frame_indices=torch.zeros(2, 1, dtype=torch.long),
        event_anchor_valid_mask=torch.ones(2, 1, dtype=torch.bool),
        event_anchor_group_id=torch.tensor([5, 5]), event_context=context,
        event_context_mask=mask, action=torch.randn(2, 32, 14),
        timestep_action=torch.full((2,), 600.), noisy_action=torch.randn(2, 32, 14),
        progress_context=pc, progress_context_mask=pm, progress_modulations=None,
        fuse_vae_embedding_in_latents=True, noisy_future_latents=torch.randn(2, 4, 2, 4, 4),
        timestep_video=torch.full((2,), 400.)), proprio


def test_init_gradients_and_no_future_leakage(model):
    kw, _ = inputs(model)
    video, action = model._grouped_event_predictions(**kw)
    for pred in (video, action):
        grad = torch.autograd.grad(pred.square().mean(), kw["event_anchor_latents"], retain_graph=True)[0]
        assert grad[0].abs().sum() > 0 and torch.isfinite(grad).all()
    _, changed = model._grouped_event_predictions(**dict(kw, noisy_future_latents=kw["noisy_future_latents"] + 10))
    torch.testing.assert_close(action, changed, rtol=0, atol=0)
    _, changed = model._grouped_event_predictions(**dict(kw, event_anchor_latents=kw["event_anchor_latents"] + 5))
    assert not torch.allclose(action, changed)
    (video.square().mean() + action.square().mean()).backward()
    for param in (model.video_expert.patch_embedding.weight,
                  model.action_expert.action_encoder.weight, model.proprio_encoder.weight):
        assert param.grad is not None and param.grad.abs().sum() > 0


@torch.no_grad()
def test_train_matches_cached_inference_with_proprio_and_absolute_time(model):
    model.eval()
    kw, proprio = inputs(model)
    _, parallel = model._grouped_event_predictions(**kw)
    model._init_current_proprio = proprio
    cache, count = model._build_event_memory_cache(
        current_latents=kw["current_latents"], event_anchor_latents=kw["event_anchor_latents"],
        context=kw["event_context"], context_mask=kw["event_context_mask"],
        fuse_vae_embedding_in_latents=True)
    actual = model._predict_action_noise_with_cache(
        kw["noisy_action"], kw["timestep_action"], kw["progress_context"], kw["progress_context_mask"],
        cache, model._build_cache_attention_mask(video_seq_len=count, action_seq_len=32, device=torch.device("cpu")), count)
    torch.testing.assert_close(actual, parallel, atol=2e-6, rtol=2e-5)
    # Changing absolute time changes init/current temporal separation.
    model._init_current_frame = torch.tensor([32, 200])
    _, shifted = model._grouped_event_predictions(**kw)
    torch.testing.assert_close(parallel[0], shifted[0], atol=0, rtol=0)
    assert not torch.allclose(parallel[1], shifted[1])


def test_original_scheduler_targets_masks_and_loss_reduction_are_inherited(model):
    kw, _ = inputs(model)
    built = dict(input_latents=torch.randn(2, 4, 3, 4, 4),
        context=kw["progress_context"], context_mask=kw["progress_context_mask"],
        action=kw["action"], action_is_pad=torch.zeros(2, 32, dtype=torch.bool),
        image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
        action_real_mask=torch.arange(32)[None].expand(2, -1) < 13,
        progress_probabilities=None, progress_present=None,
        progress_augmentation=None, progress_shift_bins=None,
        first_frame_latents=kw["current_latents"],
        event_memory_context=kw["event_context"], event_memory_context_mask=kw["event_context_mask"],
        fuse_vae_embedding_in_latents=True,
        **{k: v for k, v in kw.items() if k.startswith("event_anchor_")})
    sample = dict(frame_index=torch.tensor([32, 155]), sample_weight=torch.ones(2))
    with patch.object(model, "build_inputs", return_value=built):
        torch.manual_seed(88)
        loss, metrics = model.training_loss(sample)
        model._init_current_frame = sample["frame_index"]
        torch.manual_seed(88)
        expected, expected_metrics = FastWAM.training_loss(model, sample)
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        assert metrics.keys() == expected_metrics.keys()
        for k in metrics:
            torch.testing.assert_close(metrics[k], expected_metrics[k], atol=0, rtol=0)
        assert metrics["action_real_tokens"] == 26
        assert metrics["action_hold_tokens"] == 38
        loss.backward()


def test_original_video_spatial_rope_and_uncompressed_initial_tokens(model):
    kw, _ = inputs(model)
    pre = model.video_expert.pre_dit(kw["noisy_future_latents"], kw["timestep_video"],
        kw["event_context"], kw["event_context_mask"], fuse_vae_embedding_in_latents=True)
    ours = model._video_freqs(pre, torch.zeros(2))
    torch.testing.assert_close(ours, pre["freqs"][None].expand_as(ours), atol=1e-12, rtol=1e-12)
    assert kw["event_anchor_latents"].shape[2:] == (1, 4, 4)
    with pytest.raises(ValueError, match="frame-zero"):
        model._grouped_event_predictions(**dict(kw, event_anchor_latents=kw["event_anchor_latents"].expand(-1, -1, 2, -1, -1)))
