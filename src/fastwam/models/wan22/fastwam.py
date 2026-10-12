from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,     
        action_expert: ActionDiT,
        mot: MoT,  # 混合 video/action transformer blocks 的模块
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,     # context embedding 维度
        proprio_dim: Optional[int] = None,  # 机器人本体状态维度
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,           # video scheduler 训练 shift
        video_infer_shift: float = 5.0,           # video scheduler 推理 shift
        video_num_train_timesteps: int = 1000,    # video scheduler 时间步数量
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,           # video loss 权重
        loss_lambda_action: float = 1.0,
        progress_conditioning_bins: Optional[int] = None,
        progress_conditioning_token_index: int = 127,
        progress_conditioning_mode: str = "context_token",
        train_progress_adapter_only: bool = False,
        event_memory_video_conditioning: bool = False,
        event_memory_augmentation: Optional[dict] = None,
    ):
        """
        T  视频帧数
        L  text/context token 数
        D  	context embedding dim
        Sa action 序列长度，也就是 action_horizon
        Sv video token 序列长度

        z_dim            VAE latent channel 数
        latent_T/H/W     VAE 压缩后的视频时空尺寸

        video:        [B, 3, T, H, W]
        context:      [B, L, D]
        context_mask: [B, L]
        action:       [B, Sa, action_dim]
        """

        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        # 计算 text_dim
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)

        # 机器人状态处理
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        # 视频和动作是两套 flow matching 调度器，训练时采样 timestep、加噪、算 target；
        # 推理时生成采样 schedule 并 step 更新 latent
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )  # 训练时给 video latent 加噪、算 target、loss weight
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )  # 推理时更新 video latent

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )  # 训练时给 action 加噪、算 target、loss weight
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        # Optional aliases for consistency with Wan22Core naming.  兼容别名
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)  # video loss 系数
        self.loss_lambda_action = float(loss_lambda_action)
        self.progress_conditioning_bins = (
            None
            if progress_conditioning_bins is None
            else int(progress_conditioning_bins)
        )
        self.progress_conditioning_token_index = int(
            progress_conditioning_token_index
        )
        self.progress_conditioning_mode = str(
            progress_conditioning_mode
        ).strip().lower()
        self.train_progress_adapter_only = bool(
            train_progress_adapter_only
        )
        self.event_memory_video_conditioning = bool(event_memory_video_conditioning)
        self.event_memory_augmentation = dict(event_memory_augmentation or {})
        self._inference_event_cache = None
        self.event_cache_hits = 0
        self.event_cache_misses = 0

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,       # tokenizer 最大长度
        load_text_encoder: bool = True,     # 是否加载 text encoder/tokenizer
        proprio_dim: Optional[int] = None,  # 是否启用 proprio 条件
        redirect_common_files: bool = True, # loader 是否重定向共用文件
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,   # MoT 混合注意力是否用 gradient checkpoint
        video_train_shift: float = 5.0,           # scheduler 参数
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        progress_conditioning_bins: Optional[int] = None,
        progress_conditioning_token_index: int = 127,
        progress_conditioning_mode: str = "context_token",
        progress_modulation_hidden_dim: int = 128,
        progress_action_modulation_scale: float = 1.0,
        progress_video_modulation_scale: float = 0.1,
        train_progress_adapter_only: bool = False,
        event_memory_video_conditioning: bool = False,
        event_memory_augmentation: Optional[dict] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        # 加载 Wan2.2 组件(dit,vae,text_encoder,tokenizer,dit_path,vae_path,tokenizer_path,text_encoder_path)
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        # 检查 video/action expert 结构一致  num_heads  attn_head_dim  block
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            progress_conditioning_bins=progress_conditioning_bins,
            progress_conditioning_dim=int(video_dit_config["text_dim"]),
            progress_conditioning_mode=progress_conditioning_mode,
            progress_modulation_hidden_dim=progress_modulation_hidden_dim,
            progress_action_modulation_scale=progress_action_modulation_scale,
            progress_video_modulation_scale=progress_video_modulation_scale,
            train_progress_adapter_only=train_progress_adapter_only,
        )
        """
        分别从 video/action block 里生成 Q/K/V
        concat video/action Q/K/V
        用 joint attention_mask 做一次混合 attention
        再把结果切回 video/action
        分别走各自 block 的 cross-attn + MLP
        """

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
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
            progress_conditioning_bins=progress_conditioning_bins,
            progress_conditioning_token_index=progress_conditioning_token_index,
            progress_conditioning_mode=progress_conditioning_mode,
            train_progress_adapter_only=train_progress_adapter_only,
            event_memory_video_conditioning=event_memory_video_conditioning,
            event_memory_augmentation=event_memory_augmentation,
        )

        # 保存模型来源路径
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        # 把自然语言 prompt 编码成 context embedding
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)

        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()  # 每个样本真实 token 长度

        # 把 padding 后面的 embedding 置 0
        for i, v in enumerate(seq_lens):  
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)  # cross-attention 里 padding token 位置不会被 mask 掉
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,            # [B, L, D]
        context_mask: torch.Tensor,       # [B, L]
        proprio: Optional[torch.Tensor],  # [B, proprio_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 把机器人 proprio/state 变成一个额外 context token  
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )

        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, proprio_dim] -> [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _validate_progress_inputs(
        self,
        batch_size: int,
        probabilities: Optional[torch.Tensor],
        present_mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.progress_conditioning_bins is None:
            if probabilities is not None or present_mask is not None:
                raise ValueError(
                    "Progress was provided to a model with progress conditioning disabled."
                )
            return None, None
        if probabilities is None:
            raise ValueError(
                "Progress-conditioned FastWAM requires progress probabilities."
            )
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        if probabilities.ndim != 2 or probabilities.shape != (
            batch_size,
            self.progress_conditioning_bins,
        ):
            raise ValueError(
                "Progress probabilities must have shape "
                f"({batch_size}, {self.progress_conditioning_bins}), got "
                f"{tuple(probabilities.shape)}."
            )
        if present_mask is None:
            present_mask = torch.ones(
                batch_size, dtype=torch.bool, device=probabilities.device
            )
        elif present_mask.ndim == 0:
            present_mask = present_mask.unsqueeze(0)
        if present_mask.ndim != 1 or present_mask.shape[0] != batch_size:
            raise ValueError(
                f"Progress present mask must be [B], got {tuple(present_mask.shape)}."
            )
        probabilities = probabilities.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        probability_sums = probabilities.sum(dim=1)
        active = present_mask.to(device=self.device, dtype=torch.bool)
        if bool(active.any()) and not torch.allclose(
            probability_sums[active],
            torch.ones_like(probability_sums[active]),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("Active progress probabilities must sum to one.")
        return probabilities, active

    def _condition_context_with_progress(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        probabilities: Optional[torch.Tensor],
        present_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply context-token conditioning or validate modulation inputs."""
        probabilities, active = self._validate_progress_inputs(
            batch_size=context.shape[0],
            probabilities=probabilities,
            present_mask=present_mask,
        )
        if probabilities is None:
            return context, context_mask
        if self.progress_conditioning_mode == "modulation":
            return context, context_mask

        assert active is not None
        progress_token = self.mot.encode_progress(probabilities, active)
        if self.progress_conditioning_mode == "append_context_token":
            # Keep the subtask/proprio context untouched.  A present progress
            # condition is one additional cross-attention token; exact dropout
            # appends a zero token with a false mask and is therefore exactly
            # equivalent to the no-progress context.
            progress_mask = active.to(
                device=context_mask.device,
                dtype=torch.bool,
            ).unsqueeze(1)
            return (
                torch.cat(
                    [context, progress_token.unsqueeze(1).to(dtype=context.dtype)],
                    dim=1,
                ),
                torch.cat([context_mask, progress_mask], dim=1),
            )

        token_index = self.progress_conditioning_token_index
        if not 0 <= token_index < context.shape[1]:
            raise ValueError(
                f"Progress token index {token_index} is outside context length "
                f"{context.shape[1]}."
            )
        # This invariant makes the zero-initialized adapter exactly equivalent
        # to a subtask-only checkpoint at initialization.
        if torch.count_nonzero(context[:, token_index, :]).item() != 0:
            raise ValueError(
                f"Context slot {token_index} is not zero padding; refusing to overwrite it."
            )
        conditioned = context.clone()
        conditioned[:, token_index, :] = progress_token.to(dtype=context.dtype)
        return conditioned, context_mask

    def _progress_modulations(
        self,
        *,
        batch_size: int,
        probabilities: Optional[torch.Tensor],
        present_mask: Optional[torch.Tensor],
    ) -> Optional[dict[str, torch.Tensor]]:
        if self.progress_conditioning_mode != "modulation":
            return None
        probabilities, active = self._validate_progress_inputs(
            batch_size=batch_size,
            probabilities=probabilities,
            present_mask=present_mask,
        )
        if probabilities is None or active is None:
            return None
        return self.mot.encode_progress_modulations(probabilities, active)

    @staticmethod
    def _apply_progress_modulation(
        pre_state: dict[str, Any],
        expert_name: str,
        modulations: Optional[dict[str, torch.Tensor]],
    ) -> None:
        if modulations is None:
            return
        delta = modulations.get(expert_name)
        if delta is None:
            return
        t_mod = pre_state["t_mod"]
        if t_mod.ndim == 4:
            delta = delta.unsqueeze(1)
        if delta.ndim != t_mod.ndim:
            raise ValueError(
                f"Progress modulation rank mismatch for {expert_name}: "
                f"delta={tuple(delta.shape)} t_mod={tuple(t_mod.shape)}."
            )
        pre_state["t_mod"] = t_mod + delta.to(
            device=t_mod.device,
            dtype=t_mod.dtype,
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,  # [B, 3, T, H, W]
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )  # [B, z_dim, latent_T, latent_H, latent_W]
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        # 把推理时输入的单张图像编码成 VAE latent

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )# 推理只支持 batch size 1

        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)  # [1, z_dim, 1, latent_H, latent_W]
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        # 把 video latent 解码成 PIL 图像帧列表
        # [1, 3, T, H, W]
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)  # [3, T, H, W]
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()   # 把 [-1, 1] 映射到 [0, 255]

        # 每一帧转 PIL
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        # 训练前的 batch 预处理函数
        """
        video	        [B, 3, T, H, W]	    训练视频
        context	        [B, L, D]	        文本/context embedding
        context_mask	[B, L]	            context mask
        action	        [B, Sa, action_dim]	动作序列
        proprio	        [B, T, proprio_dim]	机器人状态
        action_is_pad	[B, Sa]	            哪些 action token 是 padding
        image_is_pad	[B, T]	            哪些 image frame 是 padding

        """
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)

        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        cached_latents = sample.get("fastwam_video_latents")
        if cached_latents is not None:
            if tiled or self.vae.training or any(p.requires_grad for p in self.vae.parameters()):
                raise ValueError("Full-clip latent cache requires a frozen eval VAE and tiled=False.")
            shapes = sample["fastwam_video_shape"].detach().cpu()
            if shapes.shape != (batch_size, 4) or not torch.equal(shapes, shapes[:1].expand_as(shapes)):
                raise ValueError("Inconsistent cached RGB clip shape metadata.")
            channels, cached_t, height, width = map(int, shapes[0])
            if channels != 3 or cached_t != num_frames or tuple(video.shape[-2:]) not in {(1, 1), (height, width)}:
                raise ValueError("Cached clip and RGB carrier shape mismatch.")
            expected = (batch_size, int(self.vae.model.z_dim),
                        1 + (num_frames - 1) // int(self.vae.temporal_downsample_factor),
                        height // int(self.vae.upsampling_factor), width // int(self.vae.upsampling_factor))
            if tuple(cached_latents.shape) != expected or cached_latents.dtype != self.torch_dtype:
                raise ValueError(f"Invalid full-clip latent shape/dtype: {cached_latents.shape}, {cached_latents.dtype}, expected {expected}/{self.torch_dtype}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:  # VAE temporal downsample 相关，后面 latent 时间长度按 transition 分组
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        # 动作长度必须能对齐视频 transition
        # 视频有 T 帧
        # transition 数 = T - 1
        # action_horizon 必须能被 T - 1 整除
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        # 如果存在，要求：action_is_pad: [B, action_horizon] 后面 action loss 会跳过 padding token
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        # image_is_pad: [B, num_frames]  ,video loss 会跳过 padding frame 对应的 latent step。
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        if cached_latents is None:
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
        else:
            input_latents = cached_latents.to(device=self.device, non_blocking=True)

        # 是否固定第一帧
        # 如果开启这个模式，训练时第一帧作为条件，不对第一帧预测
        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        # Event-anchor visual memory is persistent across control samples in a
        # subtask segment.  Keep its static text/subtask context separate from
        # the per-sample proprioception context used by the action branch.
        event_memory_context = context
        event_memory_context_mask = context_mask

        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            # 只取第 0 帧
            proprio = proprio[:, 0, :] # [B, T, proprio_dim] -> [B, proprio_dim]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        event_anchor_latents = sample.get("event_anchor_latents", None)
        event_anchor_frame_indices = sample.get("event_anchor_frame_indices", None)
        event_anchor_valid_mask = sample.get("event_anchor_valid_mask", None)
        event_anchor_group_id = sample.get("event_anchor_group_id", None)
        if event_anchor_latents is not None:
            if not torch.is_tensor(event_anchor_latents):
                raise TypeError(
                    "`sample['event_anchor_latents']` must be a tensor, got "
                    f"{type(event_anchor_latents)}."
                )
            if event_anchor_latents.ndim != 5:
                raise ValueError(
                    "`event_anchor_latents` must be [B,C,K,H,W], got "
                    f"{tuple(event_anchor_latents.shape)}."
                )
            if event_anchor_latents.shape[0] != batch_size:
                raise ValueError(
                    "`event_anchor_latents` batch mismatch: "
                    f"got {event_anchor_latents.shape[0]}, expected {batch_size}."
                )
            # Dataset collation emits [B,K,C,H,W], while the model memory
            # path consumes [B,C,K,H,W]. Normalize the layout once here.
            if (
                event_anchor_latents.shape[1] != int(self.vae.model.z_dim)
                and event_anchor_latents.shape[2] == int(self.vae.model.z_dim)
            ):
                event_anchor_latents = event_anchor_latents.permute(0, 2, 1, 3, 4).contiguous()
            if event_anchor_latents.shape[1] != int(self.vae.model.z_dim):
                raise ValueError(
                    "`event_anchor_latents` channel dimension mismatch after layout normalization: "
                    f"got {tuple(event_anchor_latents.shape)}, expected C={int(self.vae.model.z_dim)}."
                )
            event_anchor_latents = event_anchor_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            if event_anchor_frame_indices is None or event_anchor_group_id is None:
                raise ValueError(
                    "Event-anchor training requires `event_anchor_frame_indices` and "
                    "`event_anchor_group_id` for causal cache sharing."
                )
            if event_anchor_valid_mask is None:
                # Backward-compatible fallback for old fixed-width caches.
                event_anchor_valid_mask = event_anchor_frame_indices >= 0
            if event_anchor_valid_mask.ndim != 2 or tuple(
                event_anchor_valid_mask.shape
            ) != tuple(event_anchor_frame_indices.shape):
                raise ValueError(
                    "`event_anchor_valid_mask` must match [B,K] frame indices, got "
                    f"mask={tuple(event_anchor_valid_mask.shape)} and "
                    f"indices={tuple(event_anchor_frame_indices.shape)}."
                )
            # These are Python/grouping metadata, not attention masks. The
            # event-memory collator keeps them on CPU through Accelerate;
            # callers supplying device tensors remain supported via this
            # fallback. Never enqueue a nonblocking D2H read before CPU use.
            event_anchor_frame_indices = event_anchor_frame_indices.detach().to(
                device="cpu", dtype=torch.long
            )
            event_anchor_valid_mask = event_anchor_valid_mask.detach().to(
                device="cpu", dtype=torch.bool
            )
            event_anchor_group_id = event_anchor_group_id.detach().to(
                device="cpu", dtype=torch.long
            )

        progress_probabilities = sample.get("wam_progress_probabilities")
        progress_present = sample.get("wam_progress_present")
        progress_augmentation = sample.get("wam_progress_augmentation")
        progress_shift_bins = sample.get("wam_progress_shift_bins")
        if progress_probabilities is not None:
            progress_probabilities = progress_probabilities.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        if progress_present is not None:
            progress_present = progress_present.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        if progress_augmentation is not None:
            progress_augmentation = progress_augmentation.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
        if progress_shift_bins is not None:
            progress_shift_bins = progress_shift_bins.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )

        return {
            "context": context,
            "context_mask": context_mask,
            "event_memory_context": event_memory_context,
            "event_memory_context_mask": event_memory_context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "event_anchor_latents": event_anchor_latents,
            "event_anchor_frame_indices": event_anchor_frame_indices,
            "event_anchor_valid_mask": event_anchor_valid_mask,
            "event_anchor_group_id": event_anchor_group_id,
            "event_anchor_candidate_latents": sample.get("event_anchor_candidate_latents"),
            "event_anchor_candidate_indices": sample.get("event_anchor_candidate_indices"),
            "event_current_frame": sample.get("frame_index"),
            "action_real_mask": sample.get("action_real_mask"),
            "progress_probabilities": progress_probabilities,
            "progress_present": progress_present,
            "progress_augmentation": progress_augmentation,
            "progress_shift_bins": progress_shift_bins,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        # 构造 video/action 拼接后的 self-attention mask。 [ video tokens | action tokens ]
    
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video  视频 token 之间的可见性由 video expert 决定
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        # action -> action  所有 action token 可以互相看
        mask[video_seq_len:, video_seq_len:] = True

        # action -> first-frame video only  action token 只能看第一帧 video tokens
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True

        # video -> action 默认不允许

        return mask

    @staticmethod
    def _build_cache_attention_mask(
        video_seq_len: int,
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Attention mask for event-anchor KV memory plus action tokens.

        During a cache-backed pass all cached visual tokens are already
        observed.  Action queries may therefore read every anchor and the
        current frame, while the video cache itself is built separately.
        """
        if video_seq_len <= 0 or action_seq_len <= 0:
            raise ValueError(
                "Cache attention lengths must be positive, got "
                f"video={video_seq_len}, action={action_seq_len}."
            )
        total = int(video_seq_len) + int(action_seq_len)
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = True
        mask[video_seq_len:, :video_seq_len] = True
        mask[video_seq_len:, video_seq_len:] = True
        return mask

    @staticmethod
    def _mask_event_history_columns(
        mask: torch.Tensor, *, event_seq_len: int, event_num_frames: int,
        mode: str,
    ) -> torch.Tensor:
        """Inference ablation: block history reads without changing cache/RoPE layout.

        Apply to both current-frame prefill and action queries, otherwise masked
        history could still reach actions indirectly through the current cache.
        Causal anchor prefill ensures init does not contain rolling-event content.
        """
        if mode not in {"none", "rolling", "all"}:
            raise ValueError(f"Unknown event memory mask mode: {mode!r}")
        if mode == "none" or event_seq_len == 0:
            return mask
        if event_num_frames <= 0 or event_seq_len % event_num_frames:
            raise ValueError("History tokens must comprise equal-sized visual frames.")
        if event_seq_len > mask.shape[-1]:
            raise ValueError("History prefix exceeds attention mask width.")
        first_blocked = event_seq_len // event_num_frames if mode == "rolling" else 0
        masked = mask.clone()
        masked[..., first_blocked:event_seq_len] = False
        return masked

    @staticmethod
    def _build_frame_causal_mask(
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_frames <= 0 or tokens_per_frame <= 0:
            raise ValueError(
                f"Invalid frame-causal mask dimensions: frames={num_frames}, "
                f"tokens_per_frame={tokens_per_frame}."
            )
        frame_mask = torch.tril(
            torch.ones((num_frames, num_frames), dtype=torch.bool, device=device)
        )
        return frame_mask.repeat_interleave(tokens_per_frame, dim=0).repeat_interleave(
            tokens_per_frame, dim=1
        )

    def _build_event_anchor_cache(
        self,
        *,
        event_anchor_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[list[dict[str, torch.Tensor]], int, int]:
        """Prefill only immutable event anchors with a causal frame mask."""
        if event_anchor_latents.ndim != 5 or event_anchor_latents.shape[2] <= 0:
            raise ValueError(
                "`event_anchor_latents` must be [B,C,K,H,W] with K>0, got "
                f"{tuple(event_anchor_latents.shape)}."
            )
        if event_anchor_latents.shape[0] != context.shape[0]:
            raise ValueError("Event-anchor and context batches must match.")
        zero_timestep = torch.zeros(
            (event_anchor_latents.shape[0],),
            dtype=event_anchor_latents.dtype,
            device=event_anchor_latents.device,
        )
        anchor_pre = self.video_expert.pre_dit(
            x=event_anchor_latents,
            timestep=zero_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            temporal_offset=0,
        )
        anchor_seq_len = int(anchor_pre["tokens"].shape[1])
        tokens_per_frame = int(anchor_pre["meta"]["tokens_per_frame"])
        num_frames = int(anchor_pre["meta"]["grid_size"][0])
        anchor_mask = self._build_frame_causal_mask(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            device=anchor_pre["tokens"].device,
        )
        cache = self.mot.prefill_video_cache(
            video_tokens=anchor_pre["tokens"],
            video_freqs=anchor_pre["freqs"],
            video_t_mod=anchor_pre["t_mod"],
            video_context_payload={
                "context": anchor_pre["context"],
                "mask": anchor_pre["context_mask"],
            },
            video_attention_mask=anchor_mask,
        )
        return cache, anchor_seq_len, num_frames

    def _append_current_video_cache(
        self,
        *,
        current_latents: torch.Tensor,
        event_cache: list[dict[str, torch.Tensor]],
        event_seq_len: int,
        event_num_frames: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        event_memory_mask_mode: str = "none",
    ) -> tuple[list[dict[str, torch.Tensor]], int]:
        """Append current frame to an immutable event-anchor KV prefix."""
        if current_latents.ndim != 5 or current_latents.shape[2] != 1:
            raise ValueError(
                "`current_latents` must be [B,C,1,H,W], got "
                f"{tuple(current_latents.shape)}."
            )
        zero_timestep = torch.zeros(
            (current_latents.shape[0],),
            dtype=current_latents.dtype,
            device=current_latents.device,
        )
        current_pre = self.video_expert.pre_dit(
            x=current_latents,
            timestep=zero_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            temporal_offset=int(event_num_frames),
        )
        current_seq_len = int(current_pre["tokens"].shape[1])
        current_mask = torch.ones(
            (current_seq_len, int(event_seq_len) + current_seq_len),
            dtype=torch.bool,
            device=current_pre["tokens"].device,
        )
        current_mask = self._mask_event_history_columns(
            current_mask, event_seq_len=int(event_seq_len),
            event_num_frames=int(event_num_frames), mode=event_memory_mask_mode,
        )
        cache = self.mot.append_video_with_cache(
            video_tokens=current_pre["tokens"],
            video_freqs=current_pre["freqs"],
            video_t_mod=current_pre["t_mod"],
            video_context_payload={
                "context": current_pre["context"],
                "mask": current_pre["context_mask"],
            },
            video_kv_cache=event_cache,
            attention_mask=current_mask,
            video_seq_len=int(event_seq_len),
        )
        return cache, int(event_seq_len) + current_seq_len

    def clear_event_memory_cache(self):
        self._inference_event_cache = None

    def _event_prefix_for_inference(self, *, cache_key, event_anchor_latents,
                                    context, context_mask, fuse_vae_embedding_in_latents):
        """Bounded one-entry KV reuse, valid only for immutable inference state.

        Rolling eviction/subtask changes rebuild the entire prefix, avoiding
        residual information from evicted frames in later-layer keys/values.
        """
        if cache_key is None:
            return self._build_event_anchor_cache(
                event_anchor_latents=event_anchor_latents, context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents)
        if torch.is_grad_enabled() or self.video_expert.training:
            raise RuntimeError("Persistent KV reuse is forbidden during training.")
        revision = tuple(p._version for p in self.video_expert.parameters())
        signature = (cache_key, revision, str(event_anchor_latents.device), event_anchor_latents.dtype,
                     bool(fuse_vae_embedding_in_latents))
        saved = self._inference_event_cache
        if (saved is not None and saved["signature"] == signature
                and torch.equal(saved["anchors"], event_anchor_latents)
                and torch.equal(saved["context"], context)
                and torch.equal(saved["context_mask"], context_mask)):
            self.event_cache_hits += 1
            return saved["result"]
        result = self._build_event_anchor_cache(
            event_anchor_latents=event_anchor_latents, context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents)
        self._inference_event_cache = dict(
            signature=signature, anchors=event_anchor_latents.detach().clone(),
            context=context.detach().clone(), context_mask=context_mask.detach().clone(), result=result)
        self.event_cache_misses += 1
        return result

    def _build_event_memory_cache(
        self,
        *,
        current_latents: torch.Tensor,
        event_anchor_latents: Optional[torch.Tensor],
        event_anchor_valid_mask: Optional[torch.Tensor] = None,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        event_memory_cache_key=None,
        event_memory_mask_mode: str = "none",
    ) -> tuple[list[dict[str, torch.Tensor]], int]:
        """Build a causal event-prefix cache and append the current frame."""
        if current_latents.ndim != 5 or current_latents.shape[2] != 1:
            raise ValueError(
                "`current_latents` must be [B,C,1,H,W], got "
                f"{tuple(current_latents.shape)}."
            )
        if event_anchor_latents is not None:
            if event_anchor_latents.ndim != 5:
                raise ValueError(
                    "`event_anchor_latents` must be [B,C,K,H,W], got "
                    f"{tuple(event_anchor_latents.shape)}."
                )
            if event_anchor_latents.shape[0] != current_latents.shape[0]:
                raise ValueError("Event-anchor and current batches must match.")
            if (
                event_anchor_latents.shape[1] != current_latents.shape[1]
                and event_anchor_latents.shape[2] == current_latents.shape[1]
            ):
                event_anchor_latents = event_anchor_latents.permute(0, 2, 1, 3, 4).contiguous()
            if event_anchor_latents.shape[1] != current_latents.shape[1]:
                raise ValueError(
                    "`event_anchor_latents` channel dimension mismatch: "
                    f"got {tuple(event_anchor_latents.shape)}, current={tuple(current_latents.shape)}."
                )
            if event_anchor_valid_mask is not None:
                if event_anchor_valid_mask.ndim == 1:
                    event_anchor_valid_mask = event_anchor_valid_mask.unsqueeze(0)
                expected = (
                    int(event_anchor_latents.shape[0]),
                    int(event_anchor_latents.shape[2]),
                )
                if tuple(event_anchor_valid_mask.shape) != expected:
                    raise ValueError(
                        "`event_anchor_valid_mask` must be [B,K], got "
                        f"mask={tuple(event_anchor_valid_mask.shape)}, expected={expected}."
                    )
                valid_counts = event_anchor_valid_mask.to(torch.int64).sum(dim=1)
                if torch.any(valid_counts <= 0):
                    raise ValueError("Each event-anchor sample must retain init cache.")
                if not torch.all(valid_counts == valid_counts[0]):
                    raise ValueError(
                        "Variable-length event anchors require a grouped or single-sample "
                        "cache build; mixed lengths cannot share one dense prefill."
                    )
                valid_positions = torch.nonzero(
                    event_anchor_valid_mask[0], as_tuple=False
                ).flatten()
                event_anchor_latents = event_anchor_latents.index_select(
                    2, valid_positions
                )
            event_cache, event_seq_len, event_num_frames = self._event_prefix_for_inference(
                cache_key=event_memory_cache_key,
                event_anchor_latents=event_anchor_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            return self._append_current_video_cache(
                current_latents=current_latents,
                event_cache=event_cache,
                event_seq_len=event_seq_len,
                event_num_frames=event_num_frames,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                event_memory_mask_mode=event_memory_mask_mode,
            )

        # No historical prefix: current frame is still encoded once, with a
        # one-frame cache and the same causal append path.
        current_pre = self.video_expert.pre_dit(
            x=current_latents,
            timestep=torch.zeros(
                (current_latents.shape[0],),
                dtype=current_latents.dtype,
                device=current_latents.device,
            ),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            temporal_offset=0,
        )
        seq_len = int(current_pre["tokens"].shape[1])
        one_frame_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=current_pre["tokens"].device
        )
        return self.mot.prefill_video_cache(
            video_tokens=current_pre["tokens"],
            video_freqs=current_pre["freqs"],
            video_t_mod=current_pre["t_mod"],
            video_context_payload={
                "context": current_pre["context"],
                "mask": current_pre["context_mask"],
            },
            video_attention_mask=one_frame_mask,
        ), seq_len

    def _predict_future_video_with_cache(
        self, *, latents, timestep, context, context_mask, video_cache,
        video_seq_len, fuse_vae_embedding_in_latents, progress_modulations=None,
    ):
        tokens_per_frame = (latents.shape[-2] // self.video_expert.patch_size[1]) * (latents.shape[-1] // self.video_expert.patch_size[2])
        if video_seq_len % tokens_per_frame:
            raise ValueError("Clean memory must contain complete visual frames.")
        pre = self.video_expert.pre_dit(
            x=latents, timestep=timestep, context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            temporal_offset=video_seq_len // tokens_per_frame, num_clean_frames=0)
        self._apply_progress_modulation(pre, "video", progress_modulations)
        query_len = pre["tokens"].shape[1]
        mask = torch.ones(query_len, video_seq_len + query_len, dtype=torch.bool, device=latents.device)
        out = self.mot.forward_video_with_cache(
            video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_kv_cache=video_cache, attention_mask=mask, video_seq_len=video_seq_len)
        return self.video_expert.post_dit(out, pre)

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        # 计算每个 sample 的 video loss，支持 padding frame mask
        """
        pred_video	    [B, C, latent_T, latent_H, latent_W]	模型预测的 video target
        target_video	同上	                                 flow matching target
        image_is_pad	[B, T] 或 None	                         原始视频帧是否是 padding
        include_initial_video_step	bool	                    是否把第一个 latent step 算进 loss
        """

        # 原始[B, C, latent_T, latent_H, latent_W] ，此处对 channel 和空间维度求平均
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            # 直接对时间维求均值
            return video_loss_token.mean(dim=1)  # 返回每个 sample 一个 loss 【B】

        # 对齐 image_is_pad 到 latent 时间轴
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]  # 去掉第一帧，因为后面的 latent step 对应后续 frame groups
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum  # 每个 sample 的视频 loss

    def _grouped_event_action_prediction(self, **kwargs) -> torch.Tensor:
        """Backward-compatible action-only path for previous checkpoints."""
        _, action = self._grouped_event_predictions(**kwargs)
        return action

    def _grouped_event_predictions(
        self,
        *,
        current_latents: torch.Tensor,
        event_anchor_latents: torch.Tensor,
        event_anchor_frame_indices: torch.Tensor,
        event_anchor_valid_mask: torch.Tensor,
        event_anchor_group_id: torch.Tensor,
        event_context: torch.Tensor,
        event_context_mask: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        noisy_action: torch.Tensor,
        progress_context: torch.Tensor,
        progress_context_mask: torch.Tensor,
        progress_modulations: Optional[dict[str, torch.Tensor]],
        fuse_vae_embedding_in_latents: bool,
        noisy_future_latents: Optional[torch.Tensor] = None,
        timestep_video: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Predict video/actions with the same differentiable clean memory.

        Event anchors are encoded once for each unique episode/subtask group in
        the batch.  Current-frame video tokens remain per-sample and are
        appended causally, so this is equivalent to independent streaming
        cache construction up to floating-point ordering. Future video queries
        read this same prefix but never enter it, so action conditioning is
        independent of future video targets/noise. No KV is detached.
        """
        if event_anchor_latents.ndim != 5:
            raise ValueError(
                "`event_anchor_latents` must be [B,C,K,H,W], got "
                f"{tuple(event_anchor_latents.shape)}."
            )
        if event_anchor_group_id.ndim != 1:
            raise ValueError("`event_anchor_group_id` must be [B].")
        if event_anchor_valid_mask.ndim != 2:
            raise ValueError("`event_anchor_valid_mask` must be [B,K].")
        batch_size = int(current_latents.shape[0])
        if event_anchor_group_id.shape[0] != batch_size:
            raise ValueError("Event-anchor group ids must match the batch size.")
        if event_anchor_valid_mask.shape[0] != batch_size:
            raise ValueError("Event-anchor valid-mask batch must match current latents.")
        if noisy_future_latents is not None:
            if (noisy_future_latents.ndim != 5
                    or noisy_future_latents.shape[0] != batch_size
                    or noisy_future_latents.shape[2] <= 0):
                raise ValueError("Future video queries must be [B,C,T,H,W], T>0.")
            if timestep_video is None or timestep_video.shape[0] != batch_size:
                raise ValueError("Future video queries require per-sample video timesteps.")
            if getattr(self.video_expert, "action_conditioned", False):
                raise ValueError("Shared event-memory training requires action_conditioned=false.")

        # Do sorting, dynamic nonzero, and consistency checks on CPU metadata.
        # Moving a tiny index tensor to the model device does not require the
        # GPU->CPU synchronization formerly incurred for each group's .item()
        # and torch.equal(). Group order and representative selection are the
        # same as before, including after augmentation.
        event_anchor_group_id = event_anchor_group_id.detach().to("cpu")
        event_anchor_valid_mask = event_anchor_valid_mask.detach().to("cpu")
        event_anchor_frame_indices = event_anchor_frame_indices.detach().to("cpu")
        group_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        video_outputs: list[torch.Tensor] = []
        for group_id in torch.unique(event_anchor_group_id, sorted=True):
            group_indices = torch.nonzero(
                event_anchor_group_id == group_id, as_tuple=False
            ).flatten()
            representative = int(group_indices[0].item())
            valid_positions = torch.nonzero(
                event_anchor_valid_mask[representative], as_tuple=False
            ).flatten()
            if valid_positions.numel() <= 0:
                raise RuntimeError(
                    "Every event-anchor segment must contain its fixed init cache."
                )
            # Segment sharing is exact only when every member has the same
            # chronological anchors and valid slots.
            if not torch.equal(
                event_anchor_frame_indices.index_select(0, group_indices),
                event_anchor_frame_indices[representative].expand(group_indices.numel(), -1),
            ) or not torch.equal(
                event_anchor_valid_mask.index_select(0, group_indices),
                event_anchor_valid_mask[representative].expand(group_indices.numel(), -1),
            ):
                raise ValueError("Samples sharing an event group must have identical anchor indices/masks.")
            group_indices = group_indices.to(current_latents.device, non_blocking=True)
            valid_positions = valid_positions.to(event_anchor_latents.device, non_blocking=True)
            anchor_latents = event_anchor_latents[representative : representative + 1].index_select(
                2, valid_positions
            )
            anchor_context = event_context[representative : representative + 1]
            anchor_context_mask = event_context_mask[representative : representative + 1]
            event_cache, event_seq_len, event_num_frames = self._build_event_anchor_cache(
                event_anchor_latents=anchor_latents,
                context=anchor_context,
                context_mask=anchor_context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )

            group_size = int(group_indices.numel())
            if group_size > 1:
                event_cache = [
                    {
                        "k": layer["k"].expand(group_size, -1, -1),
                        "v": layer["v"].expand(group_size, -1, -1),
                    }
                    for layer in event_cache
                ]

            current_group = current_latents.index_select(0, group_indices)
            event_context_group = event_context.index_select(0, group_indices)
            event_context_mask_group = event_context_mask.index_select(0, group_indices)
            video_cache, video_seq_len = self._append_current_video_cache(
                current_latents=current_group,
                event_cache=event_cache,
                event_seq_len=event_seq_len,
                event_num_frames=event_num_frames,
                context=event_context_group,
                context_mask=event_context_mask_group,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )

            if noisy_future_latents is not None:
                video_outputs.append(self._predict_future_video_with_cache(
                    latents=noisy_future_latents.index_select(0, group_indices),
                    timestep=timestep_video.index_select(0, group_indices),
                    context=progress_context.index_select(0, group_indices),
                    context_mask=progress_context_mask.index_select(0, group_indices),
                    fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    video_cache=video_cache,
                    video_seq_len=video_seq_len,
                    progress_modulations=(None if progress_modulations is None else {
                        k: v.index_select(0, group_indices) for k, v in progress_modulations.items()
                    }),
                ))

            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action.index_select(0, group_indices),
                timestep=timestep_action.index_select(0, group_indices),
                context=progress_context.index_select(0, group_indices),
                context_mask=progress_context_mask.index_select(0, group_indices),
            )
            group_modulations = None
            if progress_modulations is not None:
                group_modulations = {
                    name: value.index_select(0, group_indices)
                    for name, value in progress_modulations.items()
                }
            self._apply_progress_modulation(action_pre, "action", group_modulations)
            cache_attention_mask = self._build_cache_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_pre["tokens"].shape[1],
                device=action_pre["tokens"].device,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=video_cache,
                attention_mask=cache_attention_mask,
                video_seq_len=video_seq_len,
            )
            group_outputs.append(
                (
                    group_indices,
                    self.action_expert.post_dit(action_tokens, action_pre),
                )
            )

        ordered_indices = torch.cat([item[0] for item in group_outputs], dim=0)
        ordered_predictions = torch.cat([item[1] for item in group_outputs], dim=0)
        inverse = torch.argsort(ordered_indices)
        pred_video = (
            torch.cat(video_outputs, dim=0).index_select(0, inverse)
            if video_outputs else None
        )
        return pred_video, ordered_predictions.index_select(0, inverse)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]  # clean video latent
        batch_size = input_latents.shape[0]
        context = inputs["context"]              # 文本/proprio context
        context_mask = inputs["context_mask"]
        action = inputs["action"]                # clean GT action
        action_is_pad = inputs["action_is_pad"]  # padding mask
        image_is_pad = inputs["image_is_pad"]
        progress_context, progress_context_mask = (
            self._condition_context_with_progress(
                context=context,
                context_mask=context_mask,
                probabilities=inputs["progress_probabilities"],
                present_mask=inputs["progress_present"],
            )
        )
        progress_modulations = self._progress_modulations(
            batch_size=batch_size,
            probabilities=inputs["progress_probabilities"],
            present_mask=inputs["progress_present"],
        )

        # 视频加噪
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)  # 加噪后的视频 latent
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)  # 模型需要预测的视频 flow target

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

        event_anchor_latents = inputs["event_anchor_latents"]
        memory_metrics = {}
        if self.event_memory_augmentation.get("enabled", False) and self.mot.training:
            from .event_memory_augmentation import augment_event_memory
            if event_anchor_latents is None:
                raise ValueError("Memory augmentation requires historical inputs.")
            event_anchor_latents, aug_indices, aug_valid, memory_metrics = augment_event_memory(
                latents=event_anchor_latents,
                frame_indices=inputs["event_anchor_frame_indices"],
                valid_mask=inputs["event_anchor_valid_mask"],
                group_ids=inputs["event_anchor_group_id"],
                current_frames=inputs["event_current_frame"],
                candidate_latents=inputs["event_anchor_candidate_latents"],
                candidate_indices=inputs["event_anchor_candidate_indices"],
                probabilities=self.event_memory_augmentation.get("probabilities", (0.5, 0.2, 0.2, 0.1)),
                noise_max=self.event_memory_augmentation.get("noise_max", 0.1),
            )
            inputs["event_anchor_frame_indices"] = aug_indices
            inputs["event_anchor_valid_mask"] = aug_valid
        joint_event_video = getattr(self, "event_memory_video_conditioning", False)
        if joint_event_video and event_anchor_latents is None:
            raise ValueError("event_memory_video_conditioning requires event-anchor inputs.")
        video_prediction_has_current = True
        if event_anchor_latents is None:
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=progress_context,
                context_mask=progress_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )

            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=progress_context,
                context_mask=progress_context_mask,
            )
            self._apply_progress_modulation(video_pre, "video", progress_modulations)
            self._apply_progress_modulation(action_pre, "action", progress_modulations)

            video_tokens = video_pre["tokens"]
            action_tokens = action_pre["tokens"]
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_tokens.shape[1],
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_tokens.device,
            )
            tokens_out = self.mot(
                embeds_all={"video": video_tokens, "action": action_tokens},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        elif joint_event_video:
            current_latents = inputs["first_frame_latents"]
            if current_latents is None:
                raise ValueError("Joint event video/action training requires a clean current frame.")
            pred_video, pred_action = self._grouped_event_predictions(
                current_latents=current_latents,
                event_anchor_latents=event_anchor_latents,
                event_anchor_frame_indices=inputs["event_anchor_frame_indices"],
                event_anchor_valid_mask=inputs["event_anchor_valid_mask"],
                event_anchor_group_id=inputs["event_anchor_group_id"],
                event_context=inputs["event_memory_context"],
                event_context_mask=inputs["event_memory_context_mask"],
                action=action,
                timestep_action=timestep_action,
                noisy_action=noisy_action,
                progress_context=progress_context,
                progress_context_mask=progress_context_mask,
                progress_modulations=progress_modulations,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                noisy_future_latents=latents[:, :, 1:],
                timestep_video=timestep_video,
            )
            video_prediction_has_current = False
        else:
            # Cache-backed training mirrors deployment: video prediction uses
            # the current/future clip, while action queries attend a causal
            # event-anchor prefix shared within each subtask segment.
            pred_video = self.video_expert(
                x=latents,
                timestep=timestep_video,
                context=progress_context,
                context_mask=progress_context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            current_latents = inputs["first_frame_latents"]
            if current_latents is None:
                current_latents = input_latents[:, :, :1]
            pred_action = self._grouped_event_action_prediction(
                current_latents=current_latents,
                event_anchor_latents=event_anchor_latents,
                event_anchor_frame_indices=inputs["event_anchor_frame_indices"],
                event_anchor_valid_mask=inputs["event_anchor_valid_mask"],
                event_anchor_group_id=inputs["event_anchor_group_id"],
                event_context=inputs["event_memory_context"],
                event_context_mask=inputs["event_memory_context_mask"],
                action=action,
                timestep_action=timestep_action,
                noisy_action=noisy_action,
                progress_context=progress_context,
                progress_context_mask=progress_context_mask,
                progress_modulations=progress_modulations,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            if video_prediction_has_current:
                pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        sample_weight = sample.get("sample_weight")
        if sample_weight is None:
            sample_weight = torch.ones_like(loss_video_per_sample)
        else:
            sample_weight = sample_weight.to(loss_video_per_sample).reshape(-1)
            if sample_weight.shape != loss_video_per_sample.shape:
                raise ValueError("sample_weight must match the training batch.")
        # Explicit DDP epoch-padding sentinels must contribute exactly zero.
        loss_video = (loss_video_per_sample * video_weight * sample_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight * sample_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            # Detached device scalars are accumulated by the trainer and read
            # only at log intervals. Float64 scaling preserves the old Python
            # float multiplication before the trainer's float32 conversion.
            "loss_video": (loss_video.detach().double() * self.loss_lambda_video).float(),
            "loss_action": (loss_action.detach().double() * self.loss_lambda_action).float(),
        }
        loss_dict.update(memory_metrics)
        real_mask = inputs.get("action_real_mask")
        if real_mask is not None:
            real_mask = real_mask.to(action_loss_token.device, dtype=torch.bool)
            if real_mask.shape != action_loss_token.shape:
                raise ValueError("action_real_mask must match the action horizon.")
            valid_tokens = torch.ones_like(real_mask) if action_is_pad is None else ~action_is_pad
            valid_tokens = valid_tokens & (sample_weight[:, None] > 0)
            # The scheduler returns a scalar for B=1 (offline evaluation).
            # Keep batch broadcasting without changing the loss or B>1 metrics.
            weighted_error = action_loss_token.detach() * action_weight.reshape(-1, 1) * self.loss_lambda_action
            for name, mask in (("real", real_mask & valid_tokens), ("hold", ~real_mask & valid_tokens)):
                # Trainer reduces sums/counts across the logging interval and
                # ranks before taking their ratio; no rank-average bias.
                loss_dict[f"action_{name}_loss_sum"] = (weighted_error * mask).sum()
                loss_dict[f"action_{name}_tokens"] = mask.sum()
        augmentation = inputs["progress_augmentation"]
        if augmentation is not None:
            loss_dict.update(
                progress_clean_fraction=(augmentation == 0).float().mean(),
                progress_shift_fraction=(augmentation == 1).float().mean(),
                progress_dropout_fraction=(augmentation == 2).float().mean(),
            )
            shift_bins = inputs["progress_shift_bins"]
            if shift_bins is not None:
                shifted = augmentation == 1
                loss_dict["progress_abs_shift_bins"] = (
                    shift_bins[shifted].abs().mean()
                    if bool(shifted.any())
                    else shift_bins.new_zeros(())
                )
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        progress_modulations: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        self._apply_progress_modulation(video_pre, "video", progress_modulations)
        self._apply_progress_modulation(action_pre, "action", progress_modulations)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        progress_modulations: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        self._apply_progress_modulation(video_pre, "video", progress_modulations)
        self._apply_progress_modulation(action_pre, "action", progress_modulations)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        progress_modulations: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        self._apply_progress_modulation(action_pre, "action", progress_modulations)
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        event_anchor_latents: Optional[torch.Tensor] = None,
        event_anchor_valid_mask: Optional[torch.Tensor] = None,
        event_memory_cache_key=None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        progress_probabilities: Optional[torch.Tensor] = None,
        progress_present: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if self.event_memory_video_conditioning and event_anchor_latents is None:
            raise ValueError("Event-conditioned joint inference requires event anchors (including init).")
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                event_anchor_latents=(
                    event_anchor_latents.clone()
                    if event_anchor_latents is not None
                    else None
                ),
                event_anchor_valid_mask=(
                    event_anchor_valid_mask.clone()
                    if event_anchor_valid_mask is not None
                    else None
                ),
                event_memory_cache_key=event_memory_cache_key,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                progress_probabilities=(
                    progress_probabilities.clone()
                    if progress_probabilities is not None
                    else None
                ),
                progress_present=(
                    progress_present.clone()
                    if progress_present is not None
                    else None
                ),
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

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

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        event_video_context, event_video_context_mask = context, context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        context, context_mask = self._condition_context_with_progress(
            context=context,
            context_mask=context_mask,
            probabilities=progress_probabilities,
            present_mask=progress_present,
        )
        progress_modulations = self._progress_modulations(
            batch_size=context.shape[0],
            probabilities=progress_probabilities,
            present_mask=progress_present,
        )

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
        use_event_video = self.event_memory_video_conditioning and event_anchor_latents is not None
        if use_event_video:
            if latents_video.shape[2] < 2:
                raise ValueError("Joint event-video inference requires at least one future latent frame.")
            if event_anchor_latents.ndim == 4:
                event_anchor_latents = event_anchor_latents.unsqueeze(0)
            event_anchor_latents = event_anchor_latents.to(self.device, self.torch_dtype)
            if event_anchor_valid_mask is not None:
                event_anchor_valid_mask = event_anchor_valid_mask.reshape(1, -1).to(self.device)
            video_kv_cache, video_seq_len = self._build_event_memory_cache(
                current_latents=first_frame_latents, event_anchor_latents=event_anchor_latents,
                event_anchor_valid_mask=event_anchor_valid_mask,
                context=event_video_context, context_mask=event_video_context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                event_memory_cache_key=event_memory_cache_key)
            cache_mask = self._build_cache_attention_mask(
                video_seq_len=video_seq_len, action_seq_len=action_horizon, device=self.device)
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if use_event_video:
                future_pred = self._predict_future_video_with_cache(
                    latents=latents_video[:, :, 1:], timestep=timestep_video,
                    context=context, context_mask=context_mask,
                    video_cache=video_kv_cache, video_seq_len=video_seq_len,
                    fuse_vae_embedding_in_latents=fuse_flag, progress_modulations=progress_modulations)
                pred_video_posi = torch.cat([torch.zeros_like(first_frame_latents), future_pred], dim=2)
                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=latents_action, timestep_action=timestep_action,
                    context=context, context_mask=context_mask,
                    video_kv_cache=video_kv_cache, attention_mask=cache_mask,
                    video_seq_len=video_seq_len, progress_modulations=progress_modulations)
            else:
                pred_video_posi, pred_action_posi = self._predict_joint_noise(
                    latents_video=latents_video, latents_action=latents_action,
                    timestep_video=timestep_video, timestep_action=timestep_action,
                    context=context, context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag, gt_action=action,
                    progress_modulations=progress_modulations)
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        event_anchor_latents: Optional[torch.Tensor] = None,
        event_anchor_valid_mask: Optional[torch.Tensor] = None,
        event_memory_cache_key=None,
        event_memory_mask_mode: str = "none",
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        progress_probabilities: Optional[torch.Tensor] = None,
        progress_present: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if event_memory_mask_mode not in {"none", "rolling", "all"}:
            raise ValueError("event_memory_mask_mode must be none, rolling, or all.")
        if event_memory_mask_mode != "none" and event_anchor_latents is None:
            raise ValueError("History masking requires anchors to preserve visual context and RoPE layout.")
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )

        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # 从高斯噪声开始生成 action
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),  
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        if event_anchor_latents is not None:
            if event_anchor_latents.ndim == 4:
                event_anchor_latents = event_anchor_latents.unsqueeze(0)
            if event_anchor_latents.ndim != 5:
                raise ValueError(
                    "`event_anchor_latents` must be [C,K,H,W] or [1,C,K,H,W], got "
                    f"{tuple(event_anchor_latents.shape)}."
                )
            event_anchor_latents = event_anchor_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            if event_anchor_valid_mask is not None:
                if event_anchor_valid_mask.ndim == 1:
                    event_anchor_valid_mask = event_anchor_valid_mask.unsqueeze(0)
                if tuple(event_anchor_valid_mask.shape) != (
                    int(event_anchor_latents.shape[0]),
                    int(event_anchor_latents.shape[2]),
                ):
                    raise ValueError(
                        "`event_anchor_valid_mask` must match [B,K] event anchors, got "
                        f"mask={tuple(event_anchor_valid_mask.shape)} and "
                        f"anchors={tuple(event_anchor_latents.shape)}."
                    )
                event_anchor_valid_mask = event_anchor_valid_mask.to(
                    device=self.device, dtype=torch.bool
                )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        # Grouped event-cache training conditions both visual prefills on
        # subtask only; current proprio is added only to the action context.
        event_video_context, event_video_context_mask = context, context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        context, context_mask = self._condition_context_with_progress(
            context=context,
            context_mask=context_mask,
            probabilities=progress_probabilities,
            present_mask=progress_present,
        )
        progress_modulations = self._progress_modulations(
            batch_size=context.shape[0],
            probabilities=progress_probabilities,
            present_mask=progress_present,
        )

        video_kv_cache, video_seq_len = self._build_event_memory_cache(
            current_latents=first_frame_latents,
            event_anchor_latents=event_anchor_latents,
            event_anchor_valid_mask=event_anchor_valid_mask,
            context=event_video_context if event_anchor_latents is not None else context,
            context_mask=event_video_context_mask if event_anchor_latents is not None else context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            event_memory_cache_key=event_memory_cache_key,
            event_memory_mask_mode=event_memory_mask_mode,
        )
        attention_mask = self._build_cache_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            device=first_frame_latents.device,
        )
        if event_memory_mask_mode != "none":
            history_frames = (
                int(event_anchor_valid_mask[0].sum().item())
                if event_anchor_valid_mask is not None else int(event_anchor_latents.shape[2])
            )
            if history_frames <= 0 or video_seq_len % (history_frames + 1):
                raise ValueError("Masked inference requires equal-sized history/current frames.")
            history_tokens = video_seq_len // (history_frames + 1) * history_frames
            attention_mask = self._mask_event_history_columns(
                attention_mask, event_seq_len=history_tokens,
                event_num_frames=history_frames, mode=event_memory_mask_mode,
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                progress_modulations=progress_modulations,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        event_anchor_latents: Optional[torch.Tensor] = None,
        event_anchor_valid_mask: Optional[torch.Tensor] = None,
        event_memory_cache_key=None,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        progress_probabilities: Optional[torch.Tensor] = None,
        progress_present: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            event_anchor_latents=event_anchor_latents,
            event_anchor_valid_mask=event_anchor_valid_mask,
            event_memory_cache_key=event_memory_cache_key,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            progress_probabilities=progress_probabilities,
            progress_present=progress_present,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        self.clear_event_memory_cache()
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
