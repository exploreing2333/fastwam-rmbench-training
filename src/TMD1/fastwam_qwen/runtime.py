from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(value, default=None):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {} if default is None else default
    return value


def create_tmd(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = False,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    qwen_model_path: str = "/home/cenzl/VLA/base_model/Qwen3-VL-2B-Instruct",
    qwen_num_mot_layers: int | None = None,
    qwen_attn_implementation: str = "sdpa",
    freeze_qwen: bool = False,
    freeze_qwen_vision: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from TMD1.fastwam_qwen.models.qwen3vl_mot import TMD

    video_dit_config = _to_plain_dict(video_dit_config)
    action_dit_config = _to_plain_dict(action_dit_config)
    video_scheduler = _to_plain_dict(video_scheduler)
    action_scheduler = _to_plain_dict(action_scheduler)
    loss = _to_plain_dict(loss)

    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    missing = {"train_shift", "infer_shift", "num_train_timesteps"} - set(action_scheduler.keys())
    if missing:
        raise ValueError(f"`action_scheduler` missing required keys: {sorted(missing)}")
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return TMD.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        qwen_model_path=str(qwen_model_path),
        qwen_num_mot_layers=(None if qwen_num_mot_layers is None else int(qwen_num_mot_layers)),
        qwen_attn_implementation=str(qwen_attn_implementation),
        freeze_qwen=bool(freeze_qwen),
        freeze_qwen_vision=bool(freeze_qwen_vision),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


create_fastwam_qwen3vl = create_tmd
