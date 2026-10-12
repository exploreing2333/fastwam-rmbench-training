from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from TMD1.fastwam.models.wan22.action_dit import ActionDiT
from TMD1.fastwam.models.wan22.fastwam import FastWAM
from TMD1.fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from TMD1.fastwam.utils.logging_config import get_logger

from .dit_experts import ContextFreeWanVideoExpert, load_action_dit_for_mot_layers
from .qwen_expert import Qwen3VLExpert
from .trimot import TriMoT

logger = get_logger(__name__)


class TMD(FastWAM):
    """TMD: FastWAM side path that replaces T5 conditioning with Qwen3-VL MoT tokens."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        qwen_expert: Qwen3VLExpert,
        mot: TriMoT,
        vae,
        text_dim: int,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            text_dim=int(text_dim),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(device=self.device, dtype=self.torch_dtype)

    @property
    def qwen_expert(self) -> Qwen3VLExpert:
        return self.mot.mixtures["qwen"]

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = False,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        qwen_model_path: str = "/home/cenzl/VLA/base_model/Qwen3-VL-2B-Instruct",
        qwen_num_mot_layers: Optional[int] = None,
        qwen_attn_implementation: str = "sdpa",
        freeze_qwen: bool = False,
        freeze_qwen_vision: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for TMD.")
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for TMD.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required by the underlying Wan DiT constructor.")
        if load_text_encoder:
            logger.warning("TMD ignores `load_text_encoder=True`; T5 is not loaded in this path.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=False,
        )
        video_expert = ContextFreeWanVideoExpert(components.dit)
        action_expert = load_action_dit_for_mot_layers(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for TriMoT.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for TriMoT.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        qwen_expert = Qwen3VLExpert(
            model_path=qwen_model_path,
            shared_num_heads=int(video_expert.num_heads),
            shared_attn_head_dim=int(video_expert.attn_head_dim),
            num_mot_layers=qwen_num_mot_layers,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=qwen_attn_implementation,
            freeze_qwen=freeze_qwen,
            freeze_vision=freeze_qwen_vision,
        )
        if int(qwen_expert.num_layers) > int(len(video_expert.blocks)):
            raise ValueError(
                "Qwen MoT layer count cannot exceed video/action DiT layers in this path: "
                f"qwen={qwen_expert.num_layers}, video={len(video_expert.blocks)}. "
                "Use fewer Qwen MoT layers or a deeper DiT."
            )

        mot = TriMoT(
            qwen_expert=qwen_expert,
            video_expert=video_expert,
            action_expert=action_expert,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            qwen_expert=qwen_expert,
            mot=mot,
            vae=components.vae,
            text_dim=int(qwen_expert.hidden_dim),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": "DISABLED_QWEN3VL_PATH",
            "tokenizer": "DISABLED_QWEN3VL_PATH",
            "qwen3vl": qwen_model_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def _normalize_prompts(self, prompt: Union[str, Sequence[str]], batch_size: int) -> list[str]:
        if isinstance(prompt, str):
            return [prompt] * batch_size
        prompts = list(prompt)
        if len(prompts) != batch_size:
            raise ValueError(f"Prompt batch mismatch: got {len(prompts)} prompts for batch={batch_size}")
        return [str(p) for p in prompts]

    @staticmethod
    def _append_text_position_id(
        position_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones(
                position_ids.shape[-2:],
                dtype=torch.bool,
                device=position_ids.device,
            )
        valid_mask = valid_mask.to(device=position_ids.device, dtype=torch.bool)

        if position_ids.ndim == 3:
            mask = valid_mask.unsqueeze(0)
            masked_position_ids = torch.where(
                mask,
                position_ids,
                torch.full_like(position_ids, -1),
            )
            next_position = masked_position_ids.amax(dim=(0, 2)).clamp_min(-1) + 1
            append_position = next_position.to(dtype=position_ids.dtype).view(1, position_ids.shape[1], 1)
            append_position = append_position.expand(position_ids.shape[0], -1, -1)
            return torch.cat([position_ids, append_position], dim=2)

        if position_ids.ndim == 2:
            masked_position_ids = torch.where(
                valid_mask,
                position_ids,
                torch.full_like(position_ids, -1),
            )
            next_position = masked_position_ids.amax(dim=1).clamp_min(-1) + 1
            append_position = next_position.to(dtype=position_ids.dtype).view(position_ids.shape[0], 1)
            return torch.cat([position_ids, append_position], dim=1)

        raise ValueError(f"`position_ids` must be 2D or 3D, got shape {tuple(position_ids.shape)}")

    def _prepare_training_proprio(self, sample, batch_size: int) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            return None
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
        if proprio.ndim != 3:
            raise ValueError(f"`sample['proprio']` must be 3D [B,T,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[0] != batch_size:
            raise ValueError(f"`sample['proprio']` batch must be {batch_size}, got {proprio.shape[0]}")
        if self.proprio_dim is None or proprio.shape[2] != self.proprio_dim:
            raise ValueError(f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}")
        return proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

    def _prepare_inference_proprio(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            return None
        if proprio is None:
            raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
        if proprio.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"1D `proprio` only supports batch_size=1, got batch_size={batch_size}")
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        else:
            raise ValueError(f"`proprio` must be [D], [B,D], or [B,T,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[0] != batch_size:
            raise ValueError(f"`proprio` batch must be {batch_size}, got {proprio.shape[0]}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        return proprio.to(device=self.device, dtype=self.torch_dtype)

    def _append_proprio_to_qwen_pre(
        self,
        qwen_pre: dict[str, Any],
        proprio: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if self.proprio_encoder is None or proprio is None:
            return qwen_pre

        tokens = qwen_pre["tokens"]
        attention_mask = qwen_pre.get("attention_mask")
        if attention_mask is None:
            old_attention_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            old_attention_mask = attention_mask.to(device=tokens.device, dtype=torch.bool)

        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=tokens.dtype).unsqueeze(1)
        ).to(device=tokens.device, dtype=tokens.dtype)
        tokens_with_proprio = torch.cat([tokens, proprio_token], dim=1)
        proprio_mask = torch.ones(
            (old_attention_mask.shape[0], 1),
            dtype=torch.bool,
            device=old_attention_mask.device,
        )
        attention_mask_with_proprio = torch.cat([old_attention_mask, proprio_mask], dim=1)
        if tokens_with_proprio.shape[-1] != self.qwen_expert.hidden_dim:
            raise ValueError(
                "Qwen proprio token dim mismatch: "
                f"got {tokens_with_proprio.shape[-1]}, expected {self.qwen_expert.hidden_dim}"
            )

        out = dict(qwen_pre)
        out["tokens"] = tokens_with_proprio
        out["attention_mask"] = attention_mask_with_proprio

        rope_position_ids = qwen_pre.get("rope_position_ids")
        if rope_position_ids is None:
            seq = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long)
            rope_position_ids = seq.view(1, 1, -1).expand(3, tokens.shape[0], -1)
        rope_position_ids = self._append_text_position_id(rope_position_ids, old_attention_mask)
        out["rope_position_ids"] = rope_position_ids
        out["position_embeddings"] = self.qwen_expert.qwen.model.language_model.rotary_emb(
            tokens_with_proprio,
            rope_position_ids,
        )

        text_position_ids = qwen_pre.get("text_position_ids")
        if text_position_ids is not None:
            out["text_position_ids"] = self._append_text_position_id(text_position_ids, old_attention_mask)

        visual_pos_masks = qwen_pre.get("visual_pos_masks")
        if visual_pos_masks is not None:
            visual_pos_masks = visual_pos_masks.to(device=tokens.device, dtype=torch.bool)
            proprio_visual_mask = torch.zeros(
                (visual_pos_masks.shape[0], 1),
                dtype=torch.bool,
                device=visual_pos_masks.device,
            )
            out["visual_pos_masks"] = torch.cat([visual_pos_masks, proprio_visual_mask], dim=1)

        meta = dict(qwen_pre.get("meta", {}))
        meta["seq_len"] = int(tokens_with_proprio.shape[1])
        meta["proprio_appended"] = True
        out["meta"] = meta
        return out

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got {tuple(video.shape)}")
        if "prompt" not in sample:
            raise ValueError("TMD training requires `sample['prompt']`.")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAMQwen3VL training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B,T,D], got {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                "`sample['action']` temporal dimension must be divisible by video transitions "
                f"({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or action_is_pad.shape != action.shape[:2]:
                raise ValueError(
                    f"`action_is_pad` must be [B,T]={tuple(action.shape[:2])}, got {tuple(action_is_pad.shape)}"
                )
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            if image_is_pad.ndim != 2 or image_is_pad.shape != (batch_size, num_frames):
                raise ValueError(
                    f"`image_is_pad` must be [B,T]={batch_size, num_frames}, got {tuple(image_is_pad.shape)}"
                )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        prompt = self._normalize_prompts(sample["prompt"], batch_size=batch_size)
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        proprio = self._prepare_training_proprio(sample, batch_size=batch_size)

        return {
            "prompt": prompt,
            "qwen_image": input_video[:, :, 0],
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "proprio": proprio,
        }

    @torch.no_grad()
    def _build_trimot_attention_mask(
        self,
        qwen_seq_len: int,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = qwen_seq_len + video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        q0, q1 = 0, qwen_seq_len
        v0, v1 = q1, q1 + video_seq_len
        a0, a1 = v1, total_seq_len

        mask[q0:q1, q0:q1] = True
        mask[v0:v1, q0:q1] = True
        mask[v0:v1, v0:v1] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[a0:a1, q0:q1] = True
        mask[a0:a1, a0:a1] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[a0:a1, v0 : v0 + first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_dit_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        v0, v1 = 0, video_seq_len
        a0, a1 = v1, total_seq_len

        mask[v0:v1, v0:v1] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[a0:a1, a0:a1] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[a0:a1, v0 : v0 + first_frame_tokens] = True
        return mask

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        qwen_pre = self.qwen_expert.pre_dit(images=inputs["qwen_image"], prompts=inputs["prompt"])
        qwen_pre = self._append_proprio_to_qwen_pre(qwen_pre, inputs["proprio"])
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        attention_mask = self._build_trimot_attention_mask(
            qwen_seq_len=qwen_pre["tokens"].shape[1],
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        dit_attention_mask = self._build_dit_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "qwen": qwen_pre["tokens"],
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            dit_attention_mask=dit_attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            qwen_pre_state=qwen_pre,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        qwen_image: torch.Tensor,
        prompts: Sequence[str],
        proprio: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qwen_pre = self.qwen_expert.pre_dit(images=qwen_image, prompts=prompts)
        qwen_pre = self._append_proprio_to_qwen_pre(qwen_pre, proprio)
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
        )
        attention_mask = self._build_trimot_attention_mask(
            qwen_seq_len=qwen_pre["tokens"].shape[1],
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        dit_attention_mask = self._build_dit_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "qwen": qwen_pre["tokens"],
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            dit_attention_mask=dit_attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            qwen_pre_state=qwen_pre,
        )
        return (
            self.video_expert.post_dit(tokens_out["video"], video_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[Union[str, Sequence[str]]],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
    ) -> dict[str, Any]:
        del context, context_mask, negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()
        if prompt is None:
            raise ValueError("TMD inference requires raw `prompt` text.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f"`input_image` must have H/W multiples of 16, got HxW=({height},{width})")
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(f"`action` must be [1,T,D] or [T,D], got {tuple(action.shape)}")
            action = action.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        prompts = self._normalize_prompts(prompt, batch_size=1)
        proprio = self._prepare_inference_proprio(proprio, batch_size=1)

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                qwen_image=input_image,
                prompts=prompts,
                proprio=proprio,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del context, context_mask, negative_prompt, text_cfg_scale
        out = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=1,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        return {"action": out["action"]}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        del action_cfg_scale
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )


FastWAMQwen3VL = TMD
