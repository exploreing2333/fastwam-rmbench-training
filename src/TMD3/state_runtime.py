from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


def _plain(value, default=None):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {} if default is None else default
    return value


def create_tmd(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    action_dit_config,
    progress_checkpoint_path: str | None = None,
    progress_hidden_dim: int = 2560,
    tokenizer_max_len: int = 128,
    action_dit_pretrained_path: str | None = None,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    condition_encoder_config=None,
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = False,
    proprio_dim: int | None = None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    del load_text_encoder
    from TMD3.state_model import StateConditionedProgressTMD

    video_dit_config = _plain(video_dit_config)
    action_dit_config = _plain(action_dit_config)
    video_scheduler = _plain(video_scheduler)
    action_scheduler = _plain(action_scheduler)
    loss = _plain(loss)
    condition_encoder_config = _plain(condition_encoder_config)
    state_dim = int(condition_encoder_config.get("state_dim", 14))
    if proprio_dim is not None and int(proprio_dim) != state_dim:
        raise ValueError(
            f"proprio_dim={proprio_dim} does not match state_dim={state_dim}."
        )
    missing = {"train_shift", "infer_shift", "num_train_timesteps"} - set(
        action_scheduler
    )
    if missing:
        raise ValueError(f"Action scheduler is missing {sorted(missing)}")
    return StateConditionedProgressTMD.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        progress_checkpoint_path=progress_checkpoint_path,
        progress_hidden_dim=int(progress_hidden_dim),
        redirect_common_files=bool(redirect_common_files),
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(
            video_scheduler.get("num_train_timesteps", 1000)
        ),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        condition_encoder_config=condition_encoder_config,
    )


__all__ = ["create_tmd"]
