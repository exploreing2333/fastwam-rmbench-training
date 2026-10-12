import logging
import copy
import hashlib
import json
import inspect
import os
import re
import shutil
from math import ceil, cos, pi
from pathlib import Path
import time

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def _loss_metric_vector(loss, loss_dict):
    """Detach metrics without device reads; upload host scalars in one copy."""
    keys = ["loss"] + sorted(loss_dict)
    values = [loss] + [loss_dict[key] for key in keys[1:]]
    host_positions = [i for i, value in enumerate(values) if not torch.is_tensor(value)]
    if host_positions:
        host_values = torch.tensor(
            [float(values[i]) for i in host_positions], dtype=torch.float32,
            device="cpu",
        ).to(device=loss.device, non_blocking=True)
        for j, i in enumerate(host_positions):
            values[i] = host_values[j]
    vector = torch.stack([
        value.detach().to(device=loss.device, dtype=torch.float32).reshape(())
        for value in values
    ])
    return keys, vector


def _stack_labeled_eval_panels(
    rollout: Image.Image,
    vae_reconstruction: Image.Image,
    ground_truth: Image.Image,
    *,
    time_label: str,
    rollout_label: str = "MODEL ROLLOUT",
) -> Image.Image:
    """Stack the same three comparison panels as the original FastWAM eval."""
    panels = (
        (rollout.convert("RGB"), rollout_label),
        (vae_reconstruction.convert("RGB"), "VAE RECONSTRUCTION"),
        (ground_truth.convert("RGB"), "GROUND TRUTH"),
    )
    panel_size = panels[0][0].size
    if any(panel.size != panel_size for panel, _ in panels):
        raise ValueError("Evaluation comparison panels must have identical sizes.")
    title_height = 32
    output = Image.new(
        "RGB",
        (
            panel_size[0],
            len(panels) * (panel_size[1] + title_height),
        ),
        color=(15, 15, 15),
    )
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for panel_index, (panel, label) in enumerate(panels):
        top = panel_index * (panel_size[1] + title_height)
        output.paste(panel, (0, top + title_height))
        draw.text(
            (8, top + 7),
            f"{label}  |  {time_label}",
            fill=(255, 255, 255),
            font=font,
        )
    return output


def _save_robotwin_action_plot(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    path: str,
    *,
    episode_index: int,
    frame_index: int,
) -> None:
    """Plot denormalized 16-D RobotWin endpose predictions against GT."""
    if tuple(pred_action.shape) != tuple(gt_action.shape):
        raise ValueError(
            "Action plot shape mismatch: "
            f"pred={tuple(pred_action.shape)}, gt={tuple(gt_action.shape)}."
        )
    if pred_action.ndim != 2 or pred_action.shape[1] != 16:
        raise ValueError(
            "RobotWin action plot expects [T,16], got "
            f"{tuple(pred_action.shape)}."
        )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pred = pred_action.detach().float().cpu()
    gt = gt_action.detach().float().cpu()
    # q and -q encode the same rotation. Align the predicted sign to GT before
    # plotting components so an equivalent quaternion is not shown as a false
    # channel-wise disagreement.
    for pose_start in (0, 8):
        quat_slice = slice(pose_start + 3, pose_start + 7)
        pred_q = torch.nn.functional.normalize(pred[:, quat_slice], dim=-1)
        gt_q = torch.nn.functional.normalize(gt[:, quat_slice], dim=-1)
        sign = torch.where(
            (pred_q * gt_q).sum(dim=-1, keepdim=True) < 0,
            -torch.ones_like(pred_q[:, :1]),
            torch.ones_like(pred_q[:, :1]),
        )
        pred[:, quat_slice] = pred_q * sign
        gt[:, quat_slice] = gt_q
    steps = list(range(int(pred.shape[0])))
    channel_names = (
        "Left position x",
        "Left position y",
        "Left position z",
        "Left quaternion x",
        "Left quaternion y",
        "Left quaternion z",
        "Left quaternion w",
        "Left gripper",
        "Right position x",
        "Right position y",
        "Right position z",
        "Right quaternion x",
        "Right quaternion y",
        "Right quaternion z",
        "Right quaternion w",
        "Right gripper",
    )
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    for channel, (axis, channel_name) in enumerate(
        zip(axes.reshape(-1), channel_names)
    ):
        axis.plot(
            steps,
            gt[:, channel],
            "--",
            color="black",
            linewidth=2.0,
            label="GT",
        )
        axis.plot(
            steps,
            pred[:, channel],
            "-",
            color="tab:red",
            linewidth=1.8,
            label="Pred",
        )
        axis.set_title(channel_name, fontsize=10)
        axis.set_xlabel("step", fontsize=8)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
        if "quaternion" in channel_name.lower():
            axis.set_ylim(-1.05, 1.05)
    fig.suptitle(
        "RobotWin endpose action — pairwise Pred vs GT\n"
        f"episode {episode_index}, anchor {frame_index}"
    )
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.persistent_workers = bool(cfg.get("persistent_workers", False))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        raw_save_at_steps = cfg.get("save_at_steps")
        if raw_save_at_steps in (None, "null", ""):
            self.save_at_steps = frozenset()
        else:
            self.save_at_steps = frozenset(
                int(step) for step in raw_save_at_steps
            )
        invalid_save_steps = sorted(
            step for step in self.save_at_steps if step <= 0
        )
        if invalid_save_steps:
            raise ValueError(
                "`save_at_steps` must contain only positive steps, got "
                f"{invalid_save_steps}."
            )
        self.max_checkpoints_to_keep = int(cfg.get("max_checkpoints_to_keep", 0))
        checkpoint_output_dir = cfg.get("checkpoint_output_dir")
        self.checkpoint_output_dir = (
            None
            if checkpoint_output_dir in (None, "null", "")
            else os.path.abspath(str(checkpoint_output_dir))
        )
        self.clear_existing_checkpoints_before_final_save = bool(
            cfg.get("clear_existing_checkpoints_before_final_save", False)
        )
        final_checkpoint_output_dir = cfg.get("final_checkpoint_output_dir")
        self.final_checkpoint_output_dir = (
            None
            if final_checkpoint_output_dir in (None, "null", "")
            else os.path.abspath(str(final_checkpoint_output_dir))
        )
        if self.max_checkpoints_to_keep < 0:
            raise ValueError(
                "`max_checkpoints_to_keep` must be >= 0 "
                f"(0 means unlimited), got {self.max_checkpoints_to_keep}."
            )
        self.eval_every = int(cfg.eval_every)
        eval_once_at_step = cfg.get("eval_once_at_step")
        self.eval_once_at_step = (
            None
            if eval_once_at_step in (None, "null", "")
            else int(eval_once_at_step)
        )
        if self.eval_once_at_step is not None and self.eval_once_at_step <= 0:
            raise ValueError(
                "`eval_once_at_step` must be positive when configured, got "
                f"{self.eval_once_at_step}."
            )
        self.eval_on_start = bool(cfg.get("eval_on_start", False))
        self.eval_save_artifacts = bool(cfg.get("eval_save_artifacts", True))
        eval_fixed_anchor_indices = cfg.get("eval_fixed_anchor_indices")
        eval_fixed_anchor_index = cfg.get("eval_fixed_anchor_index")
        if eval_fixed_anchor_indices not in (None, "null", ""):
            if eval_fixed_anchor_index not in (None, "null", ""):
                raise ValueError(
                    "Configure only one of `eval_fixed_anchor_indices` and "
                    "`eval_fixed_anchor_index`."
                )
            self.eval_fixed_anchor_indices = tuple(
                int(index) for index in eval_fixed_anchor_indices
            )
            if not self.eval_fixed_anchor_indices:
                raise ValueError("`eval_fixed_anchor_indices` must not be empty.")
        elif eval_fixed_anchor_index not in (None, "null", ""):
            self.eval_fixed_anchor_indices = (int(eval_fixed_anchor_index),)
        else:
            self.eval_fixed_anchor_indices = None
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_full_trajectory = bool(cfg.get("eval_full_trajectory", False))
        self.eval_rollout_chunks = int(cfg.get("eval_rollout_chunks", 4))
        if self.eval_rollout_chunks <= 0:
            raise ValueError(
                "`eval_rollout_chunks` must be positive, got "
                f"{self.eval_rollout_chunks}."
            )
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.restart_lr_scheduler_on_resume = bool(
            cfg.get("restart_lr_scheduler_on_resume", False)
        )
        scheduler_total_steps = cfg.get("lr_scheduler_total_steps")
        scheduler_offset_steps = cfg.get("lr_scheduler_offset_steps")
        scheduler_warmup_steps = cfg.get("lr_scheduler_warmup_steps")
        self.lr_scheduler_total_steps = (
            None
            if scheduler_total_steps in (None, "null", "")
            else int(scheduler_total_steps)
        )
        self.lr_scheduler_offset_steps = (
            0
            if scheduler_offset_steps in (None, "null", "")
            else int(scheduler_offset_steps)
        )
        self.lr_scheduler_warmup_steps = (
            None
            if scheduler_warmup_steps in (None, "null", "")
            else int(scheduler_warmup_steps)
        )
        if self.lr_scheduler_offset_steps < 0:
            raise ValueError("`lr_scheduler_offset_steps` must be non-negative.")
        resume_lr_scheduler_steps = cfg.get("resume_lr_scheduler_steps")
        self.resume_lr_scheduler_steps = (
            None
            if resume_lr_scheduler_steps in (None, "null", "")
            else int(resume_lr_scheduler_steps)
        )
        if self.resume_lr_scheduler_steps is not None and self.resume_lr_scheduler_steps <= 0:
            raise ValueError(
                "`resume_lr_scheduler_steps` must be positive when configured, got "
                f"{self.resume_lr_scheduler_steps}."
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        # A standalone weight checkpoint must be loaded before Adam/DeepSpeed
        # creates FP32 master parameters.  Loading it after `accelerator.prepare`
        # updates only the model copy; the first optimizer step can then restore
        # stale pre-checkpoint masters and silently erase the requested weights.
        self._weights_preloaded_before_optimizer = False
        if self.resume:
            early_resume_path = Path(str(self.resume))
            if early_resume_path.is_file():
                logger.info(
                    "Preloading weight checkpoint before optimizer/DeepSpeed initialization: %s",
                    early_resume_path,
                )
                self.model.load_checkpoint(str(early_resume_path), optimizer=None)
                self._weights_preloaded_before_optimizer = True
            elif not early_resume_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {early_resume_path}"
                )
        trainable_params = [
            parameter
            for parameter in self.model.dit.parameters()
            if parameter.requires_grad
        ]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(
                parameter
                for parameter in proprio_encoder.parameters()
                if parameter.requires_grad
            )
        # The independent-history condition encoder lives outside `model.dit` so
        # it can be switched off for legacy checkpoints.  Include it explicitly
        # when present; all other frozen VAE components remain excluded.
        history_aggregator = getattr(self.model, "history_aggregator", None)
        if history_aggregator is not None:
            trainable_params.extend(
                parameter
                for parameter in history_aggregator.parameters()
                if parameter.requires_grad
            )
        if not trainable_params:
            raise RuntimeError("Model exposes no trainable parameters.")
        logger.info(
            "Optimizer trainable parameters: %.3f M across %d tensors.",
            sum(parameter.numel() for parameter in trainable_params) / 1e6,
            len(trainable_params),
        )
        optimizer_cls = torch.optim.AdamW
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None:
            zero_config = deepspeed_plugin.deepspeed_config.get("zero_optimization", {})
            offload_optimizer = zero_config.get("offload_optimizer", {})
            if str(offload_optimizer.get("device", "none")).lower() == "cpu":
                from deepspeed.ops.adam import DeepSpeedCPUAdam

                optimizer_cls = DeepSpeedCPUAdam
                logger.info("Using DeepSpeedCPUAdam for ZeRO optimizer CPU offload.")
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.95),
        }
        if optimizer_cls is torch.optim.AdamW and bool(
            cfg.get("optimizer_fused", False)
        ):
            optimizer_kwargs.update(fused=True, foreach=False)
            logger.info("Using fused AdamW optimizer.")
        self.optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        scheduler_train_steps = (
            self.resume_lr_scheduler_steps
            if self.restart_lr_scheduler_on_resume
            and self.resume_lr_scheduler_steps is not None
            else total_train_steps
        )
        warmup_steps = int(scheduler_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.total_samples_seen = 0

        checkpoint_output_root = (
            self.output_dir
            if self.checkpoint_output_dir is None
            else self.checkpoint_output_dir
        )
        self.checkpoint_root = os.path.join(
            checkpoint_output_root,
            "checkpoints",
        )
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        fresh_scheduler_state = None
        fresh_scheduler_lrs = None
        if self.restart_lr_scheduler_on_resume:
            resume_path = Path(str(self.resume)) if self.resume else None
            if resume_path is None or not resume_path.is_dir():
                raise ValueError(
                    "`restart_lr_scheduler_on_resume=true` requires `resume` to be a "
                    "full training-state directory."
                )
            fresh_scheduler_state = copy.deepcopy(self.scheduler.state_dict())
            fresh_scheduler_lrs = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
        self._resume_or_load_checkpoint()
        if fresh_scheduler_state is not None:
            self.scheduler.load_state_dict(fresh_scheduler_state)
            if len(fresh_scheduler_lrs) != len(self.optimizer.param_groups):
                raise RuntimeError(
                    "Optimizer parameter-group count changed while restoring the full "
                    "training state; cannot restart the LR scheduler safely."
                )
            for group, fresh_lr in zip(
                self.optimizer.param_groups, fresh_scheduler_lrs, strict=True
            ):
                group["lr"] = fresh_lr
            logger.info(
                "Restarted LR scheduler after full-state resume: schedule_steps=%d "
                "warmup_steps=%d initial_lr=%.8g.",
                scheduler_train_steps,
                warmup_steps,
                fresh_scheduler_lrs[0],
            )

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)
        logger.info(
            "Training sampling: data_epoch_anchors=%d sampler_slots=%d "
            "zero_loss_padding=%d contract=%s persistent_workers=%s",
            self._anchor_population_size(),
            len(self.train_sampler),
            int(getattr(self.train_sampler, "num_padding_samples", 0)),
            self._sampling_contract(),
            self.persistent_workers,
        )

    def _should_save_periodic_checkpoint(self) -> bool:
        return (
            self.global_step in self.save_at_steps
            or (
                self.save_every > 0
                and self.global_step % self.save_every == 0
            )
        )

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        api_key = self.cfg.wandb.get("api_key")
        if api_key not in (None, "null", ""):
            os.environ["WANDB_API_KEY"] = str(api_key)
        run_id = self.cfg.wandb.get("id")
        if run_id in (None, "null", ""):
            run_id = hashlib.sha1(
                os.path.abspath(self.output_dir).encode("utf-8")
            ).hexdigest()[:16]
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=os.environ.get("WANDB_DIR", self.output_dir),
            id=str(run_id),
            resume=str(self.cfg.wandb.get("resume", "allow")),
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {}
        if self.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=bool(
                    getattr(self, "persistent_workers", False)
                ),
                prefetch_factor=4,
            )
        collate_fn = getattr(dataset, "collate_fn", None)
        if getattr(self.model, "event_memory_video_conditioning", False):
            from .utils.event_memory_batch import EventMemoryCollator
            collate_fn = EventMemoryCollator(collate_fn)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        """
        :param total_train_steps: 更新次数（不是epoch）
        """
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine_offset":
            schedule_total = self.lr_scheduler_total_steps
            if schedule_total is None or schedule_total <= 0:
                raise ValueError(
                    "`lr_scheduler_type=cosine_offset` requires positive "
                    "`lr_scheduler_total_steps`."
                )
            schedule_warmup = (
                int(self.lr_scheduler_warmup_steps)
                if self.lr_scheduler_warmup_steps is not None
                else int(schedule_total * 0.05)
            )
            if not 0 <= schedule_warmup < schedule_total:
                raise ValueError(
                    "Offset cosine warmup must satisfy 0 <= warmup < total, got "
                    f"warmup={schedule_warmup}, total={schedule_total}."
                )
            if self.lr_scheduler_offset_steps + total_train_steps > schedule_total:
                raise ValueError(
                    "Offset cosine schedule would run past its global horizon: "
                    f"offset={self.lr_scheduler_offset_steps}, local={total_train_steps}, "
                    f"total={schedule_total}."
                )

            def lr_factor(local_step: int) -> float:
                global_step = self.lr_scheduler_offset_steps + int(local_step)
                if schedule_warmup > 0 and global_step < schedule_warmup:
                    return max(global_step, 1) / schedule_warmup
                progress = min(
                    max(
                        (global_step - schedule_warmup)
                        / max(schedule_total - schedule_warmup, 1),
                        0.0,
                    ),
                    1.0,
                )
                return 0.01 + 0.99 * 0.5 * (1.0 + cos(pi * progress))

            scheduler = LambdaLR(self.optimizer, lr_lambda=lr_factor)
            logger.info(
                "Offset cosine LR: global_offset=%d global_total=%d warmup=%d "
                "initial_lr=%.8g final_local_lr=%.8g.",
                self.lr_scheduler_offset_steps,
                schedule_total,
                schedule_warmup,
                self.learning_rate * lr_factor(0),
                self.learning_rate * lr_factor(total_train_steps),
            )
            return scheduler
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'cosine_offset', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _anchor_population_size(self) -> int:
        size = getattr(self.train_dataset, "anchor_population_size", None)
        if size is None:
            return max(len(self.train_dataset), 1)
        return max(int(size), 1)

    def _sampling_contract(self):
        contract = getattr(self.train_dataset, "sampling_contract", None)
        return None if contract is None else str(contract)

    def _set_training_data_epoch(self):
        self.train_sampler.set_epoch(self.epoch)
        # DataLoaderShard.__iter__ calls set_epoch(self.iteration). After a
        # process restart iteration defaults to zero; synchronize the wrapper
        # as well or it silently overwrites the restored sampler epoch.
        set_epoch = getattr(getattr(self, "train_loader", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.epoch)

    def _data_epoch_progress(self) -> float:
        global_microbatch = self.batch_size * self.accelerator.num_processes
        current_real_samples = min(
            int(self.batch_in_epoch) * global_microbatch,
            self._anchor_population_size(),
        )
        return float(self.epoch) + (
            current_real_samples / self._anchor_population_size()
        )

    def _real_samples_in_current_batch(self) -> int:
        global_microbatch = self.batch_size * self.accelerator.num_processes
        if not bool(getattr(self.train_dataset, "pad_to_global_batch", False)):
            return global_microbatch
        batch_start = (int(self.batch_in_epoch) - 1) * global_microbatch
        return max(
            min(self._anchor_population_size() - batch_start, global_microbatch),
            0,
        )

    def _infer_total_samples_seen(self) -> int:
        completed = int(self.epoch) * self._anchor_population_size()
        current = min(
            int(self.batch_in_epoch)
            * self.batch_size
            * self.accelerator.num_processes,
            self._anchor_population_size(),
        )
        return completed + current

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        samples_per_sec = max(
            self.total_samples_seen - self.run_start_samples,
            0,
        ) / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return (
            f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}",
            steps_per_sec,
            samples_per_sec,
        )

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if self._weights_preloaded_before_optimizer:
            logger.info(
                "Weight checkpoint was loaded before optimizer initialization; "
                "skipping post-prepare reload: %s",
                resume,
            )
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        configure_trainable = getattr(model.dit, "configure_trainable_parameters", None)
        if callable(configure_trainable):
            configure_trainable()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        adapter_only = bool(
            getattr(model, "train_progress_adapter_only", False)
        )
        if proprio_encoder is not None and not adapter_only:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        history_aggregator = getattr(model, "history_aggregator", None)
        if history_aggregator is not None:
            history_aggregator.train()
            history_aggregator.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        # 快速评估，把输入数据整理成模型 forward 需要的格式
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        qwen_images = sample.get("qwen_images", None)
        qwen_proprio = sample.get("qwen_proprio", None)
        qwen_proprio_is_pad = sample.get("qwen_proprio_is_pad", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        progress_memory_layers = sample.get("progress_memory_layers", None)
        progress_memory_mask = sample.get("progress_memory_mask", None)
        progress_current_layers = sample.get("progress_current_layers", None)
        progress_current_mask = sample.get("progress_current_mask", None)
        progress_logits = sample.get("progress_logits", None)
        progress_instruction_layers = sample.get(
            "progress_instruction_layers", None
        )
        progress_instruction_mask = sample.get(
            "progress_instruction_mask", None
        )
        progress_subtask_tokens = sample.get("progress_subtask_tokens", None)
        progress_subtask_layers = sample.get("progress_subtask_layers", None)
        progress_subtask_mask = sample.get("progress_subtask_mask", None)
        progress_active_idx = sample.get("progress_active_idx", None)
        progress_plan_length = sample.get("progress_plan_length", None)
        precomputed_video_latents = sample.get(
            "precomputed_video_latents", None
        )
        condition_latent_valid_mask = sample.get(
            "condition_latent_valid_mask", None
        )
        state_history = sample.get("state_history", None)
        state_history_mask = sample.get("state_history_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if qwen_images is not None:
            if not isinstance(qwen_images, torch.Tensor):
                raise TypeError(f"`sample['qwen_images']` must be a torch.Tensor, got {type(qwen_images)}")
            if qwen_images.ndim == 4:
                qwen_images = qwen_images.unsqueeze(0)
            if qwen_images.ndim != 5 or qwen_images.shape[0] != video.shape[0]:
                raise ValueError(
                    "`sample['qwen_images']` must be [B,K,3,H,W], "
                    f"got {tuple(qwen_images.shape)}."
                )

        num_condition_video_frames = sample.get(
            "num_condition_video_frames",
            1 if qwen_images is None else qwen_images.shape[1],
        )
        if torch.is_tensor(num_condition_video_frames):
            num_condition_video_frames = int(num_condition_video_frames.reshape(-1)[0].item())
        else:
            num_condition_video_frames = int(num_condition_video_frames)
        future_video_frames = num_video_frames - num_condition_video_frames
        if future_video_frames <= 0:
            raise ValueError(
                f"Eval video must contain future frames, got video_T={num_video_frames}, K={num_condition_video_frames}."
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % future_video_frames != 0:
                raise ValueError(
                    "`sample['action']` temporal dimension must be divisible by future video frames "
                    f"{future_video_frames}, got {action.shape[1]}."
                )
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if qwen_proprio is not None:
            if not isinstance(qwen_proprio, torch.Tensor):
                raise TypeError(f"`sample['qwen_proprio']` must be a torch.Tensor, got {type(qwen_proprio)}")
            if qwen_proprio.ndim == 2:
                qwen_proprio = qwen_proprio.unsqueeze(0)
            if qwen_proprio.ndim != 3:
                raise ValueError(
                    "`sample['qwen_proprio']` must be [B,T_state,D], "
                    f"got {tuple(qwen_proprio.shape)}."
                )
        if qwen_proprio_is_pad is not None:
            if qwen_proprio_is_pad.ndim == 1:
                qwen_proprio_is_pad = qwen_proprio_is_pad.unsqueeze(0)
            expected_proprio_shape = (
                None if qwen_proprio is None else qwen_proprio.shape[:2]
            )
            if expected_proprio_shape is None or qwen_proprio_is_pad.shape != expected_proprio_shape:
                raise ValueError(
                    "`sample['qwen_proprio_is_pad']` shape mismatch: "
                    f"got {tuple(qwen_proprio_is_pad.shape)}, "
                    f"expected {expected_proprio_shape}."
                )

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        progress_values = {
            "progress_memory_layers": progress_memory_layers,
            "progress_memory_mask": progress_memory_mask,
            "progress_current_layers": progress_current_layers,
            "progress_current_mask": progress_current_mask,
            "progress_logits": progress_logits,
            "progress_instruction_layers": progress_instruction_layers,
            "progress_instruction_mask": progress_instruction_mask,
            "progress_subtask_tokens": progress_subtask_tokens,
            "progress_subtask_layers": progress_subtask_layers,
            "progress_subtask_mask": progress_subtask_mask,
            "progress_active_idx": progress_active_idx,
            "progress_plan_length": progress_plan_length,
        }
        if any(value is not None for value in progress_values.values()):
            compact_progress = progress_logits is not None
            state_progress = progress_current_layers is not None
            required_progress = (
                {
                    "progress_logits": progress_logits,
                    "progress_subtask_tokens": progress_subtask_tokens,
                    "progress_subtask_mask": progress_subtask_mask,
                    "state_history": state_history,
                    "state_history_mask": state_history_mask,
                }
                if compact_progress
                else {
                    "progress_current_layers": progress_current_layers,
                    "progress_current_mask": progress_current_mask,
                    "progress_instruction_layers": progress_instruction_layers,
                    "progress_instruction_mask": progress_instruction_mask,
                    "progress_subtask_layers": progress_subtask_layers,
                    "progress_subtask_mask": progress_subtask_mask,
                    "state_history": state_history,
                    "state_history_mask": state_history_mask,
                }
                if state_progress
                else {
                    "progress_memory_layers": progress_memory_layers,
                    "progress_memory_mask": progress_memory_mask,
                    "progress_subtask_mask": progress_subtask_mask,
                }
            )
            missing_progress = [
                name for name, value in required_progress.items() if value is None
            ]
            if missing_progress:
                raise ValueError(
                    "Incomplete Progress fields in eval sample; missing "
                    f"{missing_progress}."
                )
            if not compact_progress and not state_progress and (progress_subtask_tokens is None) == (
                progress_subtask_layers is None
            ):
                raise ValueError(
                    "Eval sample must provide exactly one Progress subtask "
                    "representation: final or layerwise."
                )

            def batch_progress_tensor(name, value, unbatched_ndim):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"`sample['{name}']` must be a torch.Tensor, "
                        f"got {type(value)}."
                    )
                if value.ndim == unbatched_ndim:
                    value = value.unsqueeze(0)
                if value.ndim != unbatched_ndim + 1:
                    raise ValueError(
                        f"`sample['{name}']` must have {unbatched_ndim} "
                        f"unbatched dimensions, got {tuple(value.shape)}."
                    )
                if value.shape[0] != video.shape[0]:
                    raise ValueError(
                        f"Progress batch mismatch for {name}: "
                        f"{value.shape[0]} vs video batch {video.shape[0]}."
                    )
                return value

            if compact_progress:
                progress_logits = batch_progress_tensor(
                    "progress_logits", progress_logits, 1
                )
                progress_subtask_tokens = batch_progress_tensor(
                    "progress_subtask_tokens", progress_subtask_tokens, 2
                )
                state_history = batch_progress_tensor(
                    "state_history", state_history, 2
                )
                state_history_mask = batch_progress_tensor(
                    "state_history_mask", state_history_mask, 1
                )
                if progress_active_idx is not None or progress_plan_length is not None:
                    if progress_active_idx is None or progress_plan_length is None:
                        raise ValueError(
                            "progress_active_idx and progress_plan_length must "
                            "be provided together."
                        )
                    progress_active_idx = batch_progress_tensor(
                        "progress_active_idx", progress_active_idx, 0
                    )
                    progress_plan_length = batch_progress_tensor(
                        "progress_plan_length", progress_plan_length, 0
                    )
            elif state_progress:
                progress_current_layers = batch_progress_tensor(
                    "progress_current_layers", progress_current_layers, 3
                )
                progress_current_mask = batch_progress_tensor(
                    "progress_current_mask", progress_current_mask, 1
                )
                progress_instruction_layers = batch_progress_tensor(
                    "progress_instruction_layers",
                    progress_instruction_layers,
                    3,
                )
                progress_instruction_mask = batch_progress_tensor(
                    "progress_instruction_mask",
                    progress_instruction_mask,
                    1,
                )
                state_history = batch_progress_tensor(
                    "state_history", state_history, 2
                )
                state_history_mask = batch_progress_tensor(
                    "state_history_mask", state_history_mask, 1
                )
            else:
                progress_memory_layers = batch_progress_tensor(
                    "progress_memory_layers", progress_memory_layers, 3
                )
                progress_memory_mask = batch_progress_tensor(
                    "progress_memory_mask", progress_memory_mask, 1
                )
            progress_subtask_mask = batch_progress_tensor(
                "progress_subtask_mask", progress_subtask_mask, 1
            )
            if compact_progress:
                pass
            elif progress_subtask_layers is not None:
                progress_subtask_layers = batch_progress_tensor(
                    "progress_subtask_layers", progress_subtask_layers, 3
                )
            else:
                progress_subtask_tokens = batch_progress_tensor(
                    "progress_subtask_tokens", progress_subtask_tokens, 2
                )

        if precomputed_video_latents is not None:
            if not isinstance(precomputed_video_latents, torch.Tensor):
                raise TypeError(
                    "`sample['precomputed_video_latents']` must be a tensor, "
                    f"got {type(precomputed_video_latents)}."
                )
            if precomputed_video_latents.ndim == 4:
                precomputed_video_latents = precomputed_video_latents.unsqueeze(0)
            if (
                precomputed_video_latents.ndim != 5
                or precomputed_video_latents.shape[0] != video.shape[0]
            ):
                raise ValueError(
                    "`precomputed_video_latents` must be [B,C,T,H,W], got "
                    f"{tuple(precomputed_video_latents.shape)}."
                )
            if condition_latent_valid_mask is None:
                raise ValueError(
                    "Precomputed eval latents require "
                    "`condition_latent_valid_mask`."
                )
            if not isinstance(condition_latent_valid_mask, torch.Tensor):
                raise TypeError(
                    "`sample['condition_latent_valid_mask']` must be a tensor, "
                    f"got {type(condition_latent_valid_mask)}."
                )
            if condition_latent_valid_mask.ndim == 1:
                condition_latent_valid_mask = (
                    condition_latent_valid_mask.unsqueeze(0)
                )
            if (
                condition_latent_valid_mask.ndim != 2
                or condition_latent_valid_mask.shape[0] != video.shape[0]
            ):
                raise ValueError(
                    "`condition_latent_valid_mask` must be [B,T], got "
                    f"{tuple(condition_latent_valid_mask.shape)}."
                )
        elif condition_latent_valid_mask is not None:
            raise ValueError(
                "`condition_latent_valid_mask` requires "
                "`precomputed_video_latents`."
            )

        event_anchors = sample.get("event_anchor_latents")
        event_valid = sample.get("event_anchor_valid_mask")
        event_indices = sample.get("event_anchor_frame_indices")
        event_group = sample.get("event_anchor_group_id")
        if event_anchors is not None:
            if event_anchors.ndim == 4:
                event_anchors = event_anchors.unsqueeze(0)
            if event_anchors.ndim != 5 or event_anchors.shape[0] != video.shape[0]:
                raise ValueError("Dataset event anchors must be [B,K,C,H,W].")
            # Dataset transport is [B,K,C,H,W]; model inference is [B,C,K,H,W].
            event_anchors = event_anchors.transpose(1, 2).contiguous()
            if event_valid is None or event_indices is None or event_group is None:
                raise ValueError("Event-memory evaluation requires valid mask, frame indices, and group id.")
            if event_valid.ndim == 1:
                event_valid = event_valid.unsqueeze(0)
            event_indices = torch.as_tensor(event_indices)
            if event_indices.ndim == 1:
                event_indices = event_indices.unsqueeze(0)
            event_group = torch.as_tensor(event_group).reshape(-1)
            expected = (video.shape[0], event_anchors.shape[2])
            if (tuple(event_valid.shape) != expected
                    or tuple(event_indices.shape) != expected
                    or event_group.shape != (video.shape[0],)):
                raise ValueError("Event-memory metadata does not match its [B,K] anchors.")
        batched_sample = {
            "event_anchor_latents": event_anchors,
            "event_anchor_valid_mask": event_valid,
            "event_anchor_frame_indices": event_indices,
            "event_anchor_group_id": event_group,
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "qwen_images": qwen_images,
            "qwen_proprio": qwen_proprio,
            "qwen_proprio_is_pad": qwen_proprio_is_pad,
            "num_condition_video_frames": num_condition_video_frames,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "progress_memory_layers": progress_memory_layers,
            "progress_memory_mask": progress_memory_mask,
            "progress_current_layers": progress_current_layers,
            "progress_current_mask": progress_current_mask,
            "progress_logits": progress_logits,
            "progress_instruction_layers": progress_instruction_layers,
            "progress_instruction_mask": progress_instruction_mask,
            "progress_subtask_tokens": progress_subtask_tokens,
            "progress_subtask_layers": progress_subtask_layers,
            "progress_subtask_mask": progress_subtask_mask,
            "progress_active_idx": progress_active_idx,
            "progress_plan_length": progress_plan_length,
            "precomputed_video_latents": precomputed_video_latents,
            "condition_latent_valid_mask": condition_latent_valid_mask,
            "state_history": state_history,
            "state_history_mask": state_history_mask,
        }
        for mask_name in ("image_is_pad", "qwen_image_is_pad", "action_is_pad", "proprio_is_pad"):
            mask = sample.get(mask_name)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            batched_sample[mask_name] = mask
        # Keep causal/augmentation metadata and loss weights in the eval-loss
        # contract. Candidate tensors retain the dataset's [B,K,N,C,H,W] layout.
        for name, unbatched_ndim in (
            ("frame_index", 0), ("episode_index", 0), ("sample_weight", 0),
            ("action_real_mask", 1), ("event_anchor_candidate_latents", 5),
            ("event_anchor_candidate_indices", 2),
        ):
            value = sample.get(name)
            if value is not None:
                value = torch.as_tensor(value)
                if value.ndim == unbatched_ndim:
                    value = value.unsqueeze(0)
                if value.ndim != unbatched_ndim + 1 or value.shape[0] != video.shape[0]:
                    raise ValueError(f"Invalid evaluation batch shape for `{name}`: {tuple(value.shape)}")
                batched_sample[name] = value
        return batched_sample

    @staticmethod
    def _eval_input_image(model, condition_images):
        """Respect the native FastWAM contract without changing TMD/Qwen inputs."""
        from .models.wan22.fastwam import FastWAM

        if getattr(type(model), "infer_joint", None) is FastWAM.infer_joint:
            if condition_images.shape[1] != 1:
                raise ValueError("Native FastWAM evaluation requires exactly one current image.")
            return condition_images[:, 0]
        return condition_images

    @staticmethod
    def _eval_images_to_pil(images):
        """Model-independent conversion; native FastWAM has no TMD image helper."""
        if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3:
            raise ValueError(f"Expected [1,T,3,H,W], got {tuple(images.shape)}")
        pixels = images[0].detach().float().cpu().clamp(-1, 1)
        pixels = pixels.add(1).mul(127.5).round().to(torch.uint8)
        return [Image.fromarray(frame.permute(1, 2, 0).contiguous().numpy()) for frame in pixels]

    @torch.no_grad()
    def evaluate(self):
        # 随即抽取一个sample，计算和GT之间的 loss，保存视频

        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        aligned_anchor_fn = getattr(
            self.val_dataset, "_aligned_anchor_indices", None
        )
        aligned_anchors = (
            aligned_anchor_fn() if callable(aligned_anchor_fn) else None
        )
        if self.eval_fixed_anchor_indices is not None:
            eval_index = int(
                self.eval_fixed_anchor_indices[
                    self.accelerator.process_index
                    % len(self.eval_fixed_anchor_indices)
                ]
            )
            if aligned_anchors is not None and eval_index not in aligned_anchors:
                raise ValueError(
                    "Configured fixed eval anchor is not a legal future "
                    f"window: anchor={eval_index}."
                )
            if not 0 <= eval_index < len(self.val_dataset):
                raise ValueError(
                    "Configured fixed eval anchor is outside the held-out "
                    f"dataset: anchor={eval_index}, size={len(self.val_dataset)}."
                )
        elif aligned_anchors is not None:
            rng = torch.Generator(device="cpu").manual_seed(
                self.global_step + self.accelerator.process_index
            )
            aligned_anchors = aligned_anchor_fn()
            anchor_offset = torch.randint(
                0, len(aligned_anchors), (1,), generator=rng
            ).item()
            eval_index = int(aligned_anchors[anchor_offset])
        else:
            rng = torch.Generator(device="cpu").manual_seed(
                self.global_step + self.accelerator.process_index
            )
            eval_index = torch.randint(
                0, len(self.val_dataset), (1,), generator=rng
            ).item()
        raw_sample = self.val_dataset[eval_index]
        episode_index = int(
            torch.as_tensor(raw_sample.get("episode_index", -1)).item()
        )
        frame_index = int(
            torch.as_tensor(raw_sample.get("frame_index", eval_index)).item()
        )
        if self.accelerator.is_main_process:
            if self.eval_full_trajectory:
                logger.info(
                    "Eval held-out full trajectory: episode=%d start_frame=%d",
                    episode_index,
                    frame_index,
                )
            else:
                logger.info(
                    "Eval held-out segments: episode=%d local_anchors=%s "
                    "rank0_frame=%d",
                    episode_index,
                    (
                        list(self.eval_fixed_anchor_indices)
                        if self.eval_fixed_anchor_indices is not None
                        else [eval_index]
                    ),
                    frame_index,
                )
        sample = self._to_batched_eval_sample(raw_sample)

        # 1. training loss  拟合
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio_is_pad = None
        proprio_mask = None
        num_condition_frames = int(sample["num_condition_video_frames"])
        vae_input_image = video0[:, :num_condition_frames].permute(1, 0, 2, 3).unsqueeze(0)
        if sample.get("state_history") is not None:
            input_image = vae_input_image
            proprio = sample["state_history"][0]
            proprio_mask = sample["state_history_mask"][0]
        elif sample["qwen_images"] is not None:
            input_image = sample["qwen_images"][0:1]
            proprio = sample["qwen_proprio"][0:1] if sample["qwen_proprio"] is not None else None
            proprio_is_pad = sample["qwen_proprio_is_pad"]
        elif num_condition_frames > 1:
            input_image = video0[:, :num_condition_frames].permute(1, 0, 2, 3).unsqueeze(0)
            proprio = sample["qwen_proprio"][0:1] if sample["qwen_proprio"] is not None else None
            proprio_is_pad = sample["qwen_proprio_is_pad"]
        else:
            proprio = sample["proprio"][0, 0] if sample["proprio"] is not None else None
            input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving  解算动作
        infer_kwargs = {
            "input_image": self._eval_input_image(model, vae_input_image),
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        joint_infer = getattr(model, "infer_joint", model.infer)
        if "proprio_mask" in inspect.signature(model.infer).parameters:
            infer_kwargs["proprio_mask"] = proprio_mask
        if sample.get("event_anchor_latents") is not None:
            infer_kwargs["event_anchor_latents"] = sample["event_anchor_latents"][:1]
            infer_kwargs["event_anchor_valid_mask"] = sample["event_anchor_valid_mask"][:1]
        if "action_cfg_scale" in inspect.signature(joint_infer).parameters:
            infer_kwargs["action_cfg_scale"] = 1.0
        if sample["qwen_images"] is not None and "qwen_images" in inspect.signature(model.infer).parameters:
            infer_kwargs["qwen_images"] = input_image
        if proprio_is_pad is not None and "proprio_is_pad" in inspect.signature(model.infer).parameters:
            infer_kwargs["proprio_is_pad"] = proprio_is_pad
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt
        for progress_name in (
            "progress_memory_layers",
            "progress_memory_mask",
            "progress_subtask_tokens",
            "progress_subtask_layers",
            "progress_subtask_mask",
            "progress_active_idx",
            "progress_plan_length",
        ):
            progress_value = sample.get(progress_name)
            if progress_value is not None:
                infer_kwargs[progress_name] = progress_value
        state_progress_mapping = {
            "progress_current_layers": "progress_memory_layers",
            "progress_current_mask": "progress_memory_mask",
            "progress_instruction_layers": "progress_instruction_layers",
            "progress_instruction_mask": "progress_instruction_mask",
        }
        for sample_name, infer_name in state_progress_mapping.items():
            progress_value = sample.get(sample_name)
            if progress_value is not None:
                infer_kwargs[infer_name] = progress_value
        if sample.get("progress_logits") is not None:
            infer_kwargs["progress_memory_layers"] = sample["progress_logits"]

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        future_frames_per_chunk = num_frames - num_condition_frames
        if future_frames_per_chunk <= 0:
            raise ValueError(
                "Long-rollout evaluation requires at least one future frame."
            )
        action_horizon = int(sample["action_horizon"])
        if action_horizon % future_frames_per_chunk != 0:
            raise ValueError(
                "Action horizon must be divisible by generated image count: "
                f"action_horizon={action_horizon}, future_images={future_frames_per_chunk}."
            )
        image_control_stride = action_horizon // future_frames_per_chunk
        rollout_future = list(pred_video[num_condition_frames:])
        rollout_prediction_frames = list(rollout_future)
        comparison_samples = [sample]
        rollout_action_chunks = [pred_action]
        gt_action_chunks = [action]

        if self.eval_full_trajectory:
            if aligned_anchors is None:
                raise ValueError(
                    "Full-trajectory eval requires legal aligned anchors."
                )
            aligned_anchor_set = set(int(anchor) for anchor in aligned_anchors)
            rollout_chunk_anchors = []
            next_anchor = int(eval_index)
            while next_anchor in aligned_anchor_set:
                rollout_chunk_anchors.append(next_anchor)
                next_anchor += action_horizon
        else:
            rollout_chunk_anchors = [
                int(eval_index) + chunk_index * action_horizon
                for chunk_index in range(self.eval_rollout_chunks)
            ]
        if not rollout_chunk_anchors or rollout_chunk_anchors[0] != int(eval_index):
            raise RuntimeError("Full rollout did not retain its initial anchor.")
        if self.accelerator.is_main_process:
            logger.info(
                "Eval rollout anchors=%s oracle_progress_per_replan=%s",
                rollout_chunk_anchors,
                self.eval_full_trajectory,
            )

        # RGB conditions are autoregressive, with the model's actual window
        # length (one current image for native FastWAM). Per-replan state/text
        # and event memory are dataset/oracle conditions: this is an offline
        # prediction diagnostic, NOT a closed-loop success-rate evaluation.
        rollout_condition = vae_input_image.detach().to(device="cpu")
        rollout_action = pred_action
        completed_chunk_anchors = [int(eval_index)]
        for chunk_index, comparison_anchor in enumerate(
            rollout_chunk_anchors[1:], start=1
        ):
            if not 0 <= comparison_anchor < len(self.val_dataset):
                break
            if aligned_anchors is not None and comparison_anchor not in aligned_anchors:
                break
            comparison_raw = self.val_dataset[comparison_anchor]
            if int(torch.as_tensor(comparison_raw.get("episode_index", episode_index)).item()) != episode_index:
                break
            future_condition = (
                pil_frames_to_video_tensor(rollout_future)
                .permute(1, 0, 2, 3)
                .unsqueeze(0)
                .mul(2.0)
                .sub(1.0)
            )
            rollout_condition = torch.cat(
                [rollout_condition, future_condition],
                dim=1,
            )[:, -num_condition_frames:].contiguous()
            comparison_sample = self._to_batched_eval_sample(
                comparison_raw
            )
            comparison_samples.append(comparison_sample)
            chunk_infer_kwargs = dict(infer_kwargs)
            chunk_infer_kwargs["input_image"] = self._eval_input_image(model, rollout_condition)
            chunk_infer_kwargs["action"] = rollout_action
            chunk_infer_kwargs["seed"] = 42 + chunk_index
            if comparison_sample["context"] is not None:
                chunk_infer_kwargs["prompt"] = None
                chunk_infer_kwargs["context"] = comparison_sample["context"][0]
                chunk_infer_kwargs["context_mask"] = comparison_sample[
                    "context_mask"
                ][0]
            else:
                chunk_infer_kwargs["prompt"] = comparison_sample["prompt"][0]
                chunk_infer_kwargs.pop("context", None)
                chunk_infer_kwargs.pop("context_mask", None)
            for progress_name in (
                "progress_memory_layers",
                "progress_memory_mask",
                "progress_subtask_tokens",
                "progress_subtask_layers",
                "progress_subtask_mask",
                "progress_active_idx",
                "progress_plan_length",
            ):
                progress_value = comparison_sample.get(progress_name)
                if progress_value is None:
                    chunk_infer_kwargs.pop(progress_name, None)
                else:
                    chunk_infer_kwargs[progress_name] = progress_value
            for sample_name, infer_name in state_progress_mapping.items():
                progress_value = comparison_sample.get(sample_name)
                if progress_value is None:
                    chunk_infer_kwargs.pop(infer_name, None)
                else:
                    chunk_infer_kwargs[infer_name] = progress_value
            # The compact TMD3 contract carries the full-attention Progress
            # condition as logits rather than layerwise memory.  Refresh it at
            # every rollout anchor just like the subtask and state conditions;
            # otherwise the generic cleanup above removes the initial logits
            # before the second autoregressive chunk.
            progress_logits = comparison_sample.get("progress_logits")
            if progress_logits is not None:
                chunk_infer_kwargs["progress_memory_layers"] = progress_logits
            if comparison_sample.get("state_history") is not None:
                chunk_infer_kwargs["proprio"] = comparison_sample[
                    "state_history"
                ][0]
                chunk_infer_kwargs["proprio_mask"] = comparison_sample[
                    "state_history_mask"
                ][0]
            elif infer_kwargs["input_image"].ndim == 4:
                state = comparison_sample.get("proprio")
                chunk_infer_kwargs["proprio"] = state[0, 0] if state is not None else None
            for name in ("event_anchor_latents", "event_anchor_valid_mask"):
                value = comparison_sample.get(name)
                if value is None:
                    chunk_infer_kwargs.pop(name, None)
                else:
                    chunk_infer_kwargs[name] = value[:1]
            chunk_pred = model.infer(**chunk_infer_kwargs)
            rollout_future = list(
                chunk_pred["video"][num_condition_frames:]
            )
            if len(rollout_future) != future_frames_per_chunk:
                raise RuntimeError(
                    "Autoregressive eval chunk returned the wrong number of "
                    f"future images: chunk={chunk_index}, "
                    f"got={len(rollout_future)}, expected={future_frames_per_chunk}."
                )
            rollout_action = chunk_pred.get("action", rollout_action)
            rollout_action_chunks.append(rollout_action)
            gt_action_chunks.append(comparison_sample.get("action"))
            rollout_prediction_frames.extend(rollout_future)
            completed_chunk_anchors.append(comparison_anchor)
        rollout_chunk_anchors = completed_chunk_anchors
        oracle_event_memory = sample.get("event_anchor_latents") is not None and len(comparison_samples) > 1
        if oracle_event_memory:
            logger.info("Offline eval refreshes GT event memory and state per replan; not closed-loop evaluation.")

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()  # normalization

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        # The condition prefix is copied into the rollout and is therefore not
        # a prediction. Including it would make 17 exact history frames swamp
        # the four generated future frames in PSNR/SSIM.
        pred_future_tensor = pred_video_tensor[:, num_condition_frames:]
        gt_future_tensor = gt_video_tensor[:, num_condition_frames:]
        if pred_future_tensor.shape[1] == 0:
            raise ValueError(
                "Evaluation video contains no generated future frames: "
                f"video_T={pred_video_tensor.shape[1]}, condition_T={num_condition_frames}."
            )
        psnr_rollout_vs_gt = video_psnr(
            pred=pred_future_tensor,
            target=gt_future_tensor,
        )
        ssim_rollout_vs_gt = video_ssim(
            pred=pred_future_tensor,
            target=gt_future_tensor,
        )

        action_l1 = None
        action_l2 = None
        pred_action_absolute = None
        gt_action_absolute = None
        if action is not None and pred_action is not None:
            if any(value is None for value in rollout_action_chunks):
                raise RuntimeError("Full rollout is missing a predicted action chunk.")
            if any(value is None for value in gt_action_chunks):
                raise RuntimeError("Full rollout is missing a GT action chunk.")

            def _concatenate_action_chunks(chunks):
                normalized_chunks = []
                for value in chunks:
                    if value.ndim == 2:
                        value_td = value
                    elif value.ndim == 3 and value.shape[0] == 1:
                        value_td = value[0]
                    else:
                        raise ValueError(
                            "Action chunk must be [T,D] or [1,T,D], got "
                            f"{tuple(value.shape)}."
                        )
                    normalized_chunks.append(value_td)
                return torch.cat(normalized_chunks, dim=0)

            full_pred_action = _concatenate_action_chunks(
                rollout_action_chunks
            )
            full_gt_action = _concatenate_action_chunks(gt_action_chunks)
            packed_actions = {}
            for action_name, raw_action in (
                ("pred", full_pred_action),
                ("gt", full_gt_action),
            ):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_td = raw_action
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_td = raw_action[0]
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                packed_actions[action_name] = action_td.detach().to(
                    device="cpu", dtype=torch.float32
                )

            endpose_store = getattr(
                self.val_dataset, "endpose_action_store", None
            )
            if endpose_store is not None:
                from TMD3.endpose_action import unpack_robotwin_action

                pred_model_action = unpack_robotwin_action(
                    packed_actions["pred"]
                )
                gt_model_action = unpack_robotwin_action(
                    packed_actions["gt"]
                )
                # L1/L2 stay in the exact normalized 16-D model space, while
                # the plot below uses physically meaningful absolute endpose.
                action_diff = pred_model_action - gt_model_action
                episode_anchor = endpose_store.load_episode_anchor(
                    episode_index
                )
                pred_action_absolute = endpose_store.denormalize_to_absolute(
                    packed_actions["pred"], episode_anchor
                )
                gt_action_absolute = endpose_store.denormalize_to_absolute(
                    packed_actions["gt"], episode_anchor
                )
            else:
                if sample["proprio"] is None:
                    raise ValueError(
                        "Eval sample must contain `proprio` for generic action denormalization."
                    )
                proprio = sample["proprio"].detach().to(
                    device="cpu", dtype=torch.float32
                )
                processor = self.val_dataset.lerobot_dataset.processor
                action_meta = processor.shape_meta["action"]
                state_meta = processor.shape_meta["state"]
                denorm_actions = {}
                for action_name, action_td in packed_actions.items():
                    batch = {
                        "action": action_td.unsqueeze(0),
                        "state": proprio,
                    }
                    batch = processor.action_state_merger.backward(batch)
                    batch = processor.normalizer.backward(batch)
                    merged_batch = {
                        "action": {
                            meta["key"]: batch["action"][meta["key"]].squeeze(0)
                            for meta in action_meta
                        },
                        "state": {
                            meta["key"]: batch["state"][meta["key"]].squeeze(0)
                            for meta in state_meta
                        },
                    }
                    merged_batch = processor.action_state_merger.forward(
                        merged_batch
                    )
                    denorm_actions[action_name] = merged_batch["action"]
                action_diff = denorm_actions["pred"] - denorm_actions["gt"]
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        if num_condition_frames > 1 or sample["qwen_images"] is not None:
            current_and_future = torch.cat(
                [video0[:, num_condition_frames - 1 : num_condition_frames], video0[:, num_condition_frames:]],
                dim=1,
            ).unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
            vae_latents = model._encode_video_latents(current_and_future, tiled=False)
            decoded_current_and_future = model._decode_latents(vae_latents, tiled=False)
            if len(decoded_current_and_future) != 1 + video0.shape[1] - num_condition_frames:
                raise RuntimeError(
                    "Sparse eval VAE reconstruction length mismatch: "
                    f"decoded={len(decoded_current_and_future)}, video_T={video0.shape[1]}, "
                    f"condition_T={num_condition_frames}."
                )
            vae_recon_video = self._eval_images_to_pil(vae_input_image)
            vae_recon_video.extend(decoded_current_and_future[1:])
        else:
            gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
            vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
            vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        vae_future_tensor = vae_video_tensor[:, num_condition_frames:]
        psnr_decode_vs_gt = video_psnr(
            pred=vae_future_tensor,
            target=gt_future_tensor,
        )
        ssim_decode_vs_gt = video_ssim(
            pred=vae_future_tensor,
            target=gt_future_tensor,
        )

        psnr_rollout_vs_decode = video_psnr(
            pred=pred_future_tensor,
            target=vae_future_tensor,
        )
        ssim_rollout_vs_decode = video_ssim(
            pred=pred_future_tensor,
            target=vae_future_tensor,
        )

        # Preserve the original three-panel comparison. Full-trajectory mode
        # displays the episode-initial current image followed by every complete
        # replan chunk through the final legal t=144 image.
        display_history_frames = min(num_condition_frames, 1 if self.eval_full_trajectory else 4)
        rollout_display = list(
            pred_video[
                num_condition_frames - display_history_frames : num_condition_frames
            ]
        )
        rollout_display.extend(rollout_prediction_frames)
        vae_display = list(
            vae_recon_video[
                num_condition_frames - display_history_frames :
            ]
        )
        gt_pil_video = self._eval_images_to_pil(
            video0.permute(1, 0, 2, 3).unsqueeze(0)
        )
        gt_display = list(
            gt_pil_video[
                num_condition_frames - display_history_frames :
            ]
        )
        for chunk_index, comparison_sample in enumerate(
            comparison_samples[1:], start=1
        ):
            comparison_anchor = rollout_chunk_anchors[chunk_index]
            comparison_video = comparison_sample["video"][0]
            comparison_condition_frames = int(
                comparison_sample["num_condition_video_frames"]
            )
            if comparison_condition_frames != num_condition_frames:
                raise ValueError(
                    "Long eval comparison changed the history length: "
                    f"base={num_condition_frames}, "
                    f"chunk={comparison_condition_frames}."
                )
            comparison_current_future = torch.cat(
                [
                    comparison_video[
                        :,
                        comparison_condition_frames - 1 : comparison_condition_frames,
                    ],
                    comparison_video[:, comparison_condition_frames:],
                ],
                dim=1,
            ).unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
            comparison_latents = model._encode_video_latents(
                comparison_current_future,
                tiled=False,
            )
            comparison_decoded = model._decode_latents(
                comparison_latents,
                tiled=False,
            )
            if len(comparison_decoded) != 1 + future_frames_per_chunk:
                raise RuntimeError(
                    "Long eval VAE comparison returned the wrong length: "
                    f"chunk={chunk_index}, got={len(comparison_decoded)}, "
                    f"expected={1 + future_frames_per_chunk}."
                )
            vae_display.extend(comparison_decoded[1:])
            comparison_gt = self._eval_images_to_pil(
                comparison_video.permute(1, 0, 2, 3).unsqueeze(0)
            )
            gt_display.extend(
                comparison_gt[comparison_condition_frames:]
            )

        rollout_chunk_count = len(rollout_chunk_anchors)
        expected_display_frames = display_history_frames + (
            rollout_chunk_count * future_frames_per_chunk
        )
        if not (
            len(rollout_display)
            == len(vae_display)
            == len(gt_display)
            == expected_display_frames
        ):
            raise RuntimeError(
                "Long eval panel lengths do not match: "
                f"rollout={len(rollout_display)}, vae={len(vae_display)}, "
                f"gt={len(gt_display)}, expected={expected_display_frames}."
            )
        time_offsets = [
            -image_control_stride * offset
            for offset in range(display_history_frames - 1, -1, -1)
        ] + [
            image_control_stride * offset
            for offset in range(
                1,
                rollout_chunk_count * future_frames_per_chunk + 1,
            )
        ]

        # Report future-only metrics across the same complete trajectory shown
        # in the video, not merely the first four generated images.
        full_rollout_tensor = pil_frames_to_video_tensor(
            rollout_prediction_frames
        )
        full_vae_tensor = pil_frames_to_video_tensor(
            vae_display[display_history_frames:]
        )
        full_gt_tensor = pil_frames_to_video_tensor(
            gt_display[display_history_frames:]
        )
        psnr_rollout_vs_gt = video_psnr(
            pred=full_rollout_tensor, target=full_gt_tensor
        )
        ssim_rollout_vs_gt = video_ssim(
            pred=full_rollout_tensor, target=full_gt_tensor
        )
        psnr_rollout_vs_decode = video_psnr(
            pred=full_rollout_tensor, target=full_vae_tensor
        )
        ssim_rollout_vs_decode = video_ssim(
            pred=full_rollout_tensor, target=full_vae_tensor
        )
        psnr_decode_vs_gt = video_psnr(
            pred=full_vae_tensor, target=full_gt_tensor
        )
        ssim_decode_vs_gt = video_ssim(
            pred=full_vae_tensor, target=full_gt_tensor
        )
        comparison_frames = []
        for rollout_frame, vae_frame, gt_frame, time_offset in zip(
            rollout_display,
            vae_display,
            gt_display,
            time_offsets,
        ):
            if time_offset < 0:
                time_label = f"HISTORY t-{abs(time_offset):03d}"
            elif time_offset == 0:
                time_label = "CURRENT t+000"
            else:
                time_label = f"FUTURE t+{time_offset:03d}"
            comparison_frames.append(
                _stack_labeled_eval_panels(
                    rollout_frame,
                    vae_frame,
                    gt_frame,
                    time_label=time_label,
                    rollout_label=("MODEL (GT STATE / EVENT MEMORY)" if oracle_event_memory else "MODEL ROLLOUT"),
                )
            )

        video_suffix = "full_trajectory" if self.eval_full_trajectory else (
            f"frame_{frame_index:04d}"
        )
        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_episode_{episode_index:03d}_"
            f"{video_suffix}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        # Fixed anchors are assigned modulo rank for metric gathering. Persist
        # exactly one artifact for each distinct configured anchor (ranks
        # 0..N-1), rather than dropping ranks 1/2 or writing five duplicates.
        save_eval_artifacts = self.eval_save_artifacts and (
            self.accelerator.is_main_process
            or (
                self.eval_fixed_anchor_indices is not None
                and self.accelerator.process_index
                < len(self.eval_fixed_anchor_indices)
            )
        )
        if save_eval_artifacts:
            try:
                save_mp4(comparison_frames, video_path, fps=8)
            except OSError as exc:
                # Evaluation media are auxiliary. In particular, a full disk
                # or project quota must not discard in-memory training state.
                logger.warning(
                    "Skipping eval video after write failure: path=%s error=%s",
                    video_path,
                    exc,
                )
                try:
                    os.remove(video_path)
                except OSError:
                    pass
                save_eval_artifacts = False
        action_plot_path = None
        if (
            save_eval_artifacts
            and pred_action_absolute is not None
            and gt_action_absolute is not None
        ):
            action_plot_path = os.path.splitext(video_path)[0] + "_actions.png"
            try:
                _save_robotwin_action_plot(
                    pred_action_absolute,
                    gt_action_absolute,
                    action_plot_path,
                    episode_index=episode_index,
                    frame_index=frame_index,
                )
            except OSError as exc:
                logger.warning(
                    "Skipping eval action plot after write failure: path=%s error=%s",
                    action_plot_path,
                    exc,
                )
                try:
                    os.remove(action_plot_path)
                except OSError:
                    pass
                action_plot_path = None

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        gathered_frames = self.accelerator.gather_for_metrics(
            torch.tensor(
                [frame_index],
                device=self.accelerator.device,
                dtype=torch.long,
            )
        ).reshape(-1)
        segment_vectors = []
        segment_metrics = {}
        for segment_frame in sorted(set(gathered_frames.tolist())):
            segment_mask = gathered_frames == int(segment_frame)
            segment_mean = gathered_metrics[segment_mask].mean(dim=0)
            segment_vectors.append(segment_mean)
            segment_result = {
                "val_loss": float(segment_mean[0].item()),
                "psnr_rd": float(segment_mean[3].item()),
                "ssim_rd": float(segment_mean[4].item()),
            }
            if action_l2 is not None:
                segment_result["action_l2"] = float(segment_mean[7].item())
            if action_l1 is not None:
                segment_result["action_l1"] = float(segment_mean[8].item())
            segment_metrics[int(segment_frame)] = segment_result
        mean_metrics = torch.stack(segment_vectors, dim=0).mean(dim=0)
        action_l2_mean = mean_metrics[7].item() if action_l2 is not None else None
        action_l1_mean = mean_metrics[8].item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
            "segments": segment_metrics,
            "rollout_chunk_anchors": rollout_chunk_anchors,
            "oracle_event_memory_per_replan": oracle_event_memory,
        }
        if action_plot_path is not None:
            result["action_plot_path"] = action_plot_path
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _record_evaluation(self, metrics):
        if metrics is None or not self.accelerator.is_main_process:
            return
        description = (
            "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f"
            % (
                self.global_step,
                metrics["val_loss"],
                metrics["psnr_rd"],
                metrics["ssim_rd"],
            )
        )
        if "action_l2" in metrics:
            description += " action_l2=%.4f" % metrics["action_l2"]
        if "action_l1" in metrics:
            description += " action_l1=%.4f" % metrics["action_l1"]
        logger.info(description)
        eval_payload = {
            "eval/val_loss": float(metrics["val_loss"]),
            "eval/psnr_rg": float(metrics["psnr_rg"]),
            "eval/ssim_rg": float(metrics["ssim_rg"]),
            "eval/psnr_rd": float(metrics["psnr_rd"]),
            "eval/ssim_rd": float(metrics["ssim_rd"]),
            "eval/psnr_dg": float(metrics["psnr_dg"]),
            "eval/ssim_dg": float(metrics["ssim_dg"]),
        }
        if "action_l2" in metrics:
            eval_payload["eval/action_l2"] = float(metrics["action_l2"])
        if "action_l1" in metrics:
            eval_payload["eval/action_l1"] = float(metrics["action_l1"])
        for frame_index, segment in metrics.get("segments", {}).items():
            logger.info(
                "[eval-segment] step=%d frame=%d val_loss=%.4f "
                "infer_psnr=%.4f infer_ssim=%.4f action_l2=%.4f "
                "action_l1=%.4f",
                self.global_step,
                int(frame_index),
                segment["val_loss"],
                segment["psnr_rd"],
                segment["ssim_rd"],
                segment.get("action_l2", float("nan")),
                segment.get("action_l1", float("nan")),
            )
            prefix = f"eval/segment_{int(frame_index):04d}"
            for name, value in segment.items():
                eval_payload[f"{prefix}/{name}"] = float(value)
        self._wandb_log(eval_payload)

    def _save_weights_checkpoint(self, step_tag: str, *, weights_dir=None):
        model = self.accelerator.unwrap_model(self.model)
        destination = self.weights_dir if weights_dir is None else str(weights_dir)
        ensure_dir(destination)
        ckpt_path = os.path.join(destination, f"{step_tag}.pt")
        tmp_path = f"{ckpt_path}.tmp.{os.getpid()}"
        try:
            model.save_checkpoint(tmp_path, optimizer=None, step=self.global_step)
            os.replace(tmp_path, ckpt_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "total_samples_seen": int(self.total_samples_seen),
            "sample_offset_in_epoch": int(
                self.batch_in_epoch
                * self.batch_size
                * self.accelerator.num_processes
            ),
            "checkpoint_batch_size": int(self.batch_size),
            "checkpoint_world_size": int(self.accelerator.num_processes),
            "checkpoint_gradient_accumulation_steps": int(
                self.gradient_accumulation_steps
            ),
            "sampling_contract": self._sampling_contract(),
        }
        sampler = getattr(self, "train_sampler", None)
        if (sampler is not None and bool(getattr(
            self.train_dataset, "event_anchor_microbatch_shuffle", False
        ))):
            payload["sampler_reorder_state"] = sampler.reorder_state_dict()
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _prune_old_checkpoints(self):
        """Keep the newest checkpoint steps across weights and trainer state."""
        if self.max_checkpoints_to_keep <= 0:
            return

        checkpoint_steps = set()
        state_root = Path(self.state_dir)
        weights_root = Path(self.weights_dir)
        for path in state_root.iterdir():
            match = re.fullmatch(r"step_(\d+)", path.name)
            if path.is_dir() and match:
                checkpoint_steps.add(int(match.group(1)))
        for path in weights_root.iterdir():
            match = re.fullmatch(r"step_(\d+)\.pt", path.name)
            if path.is_file() and match:
                checkpoint_steps.add(int(match.group(1)))

        steps_to_remove = sorted(checkpoint_steps)[: -self.max_checkpoints_to_keep]
        removed_tags = []
        for step in steps_to_remove:
            step_tag = f"step_{step:06d}"
            state_path = state_root / step_tag
            weights_path = weights_root / f"{step_tag}.pt"
            if state_path.is_dir():
                shutil.rmtree(state_path)
            elif state_path.exists():
                state_path.unlink()
            weights_path.unlink(missing_ok=True)
            removed_tags.append(step_tag)

        if removed_tags:
            logger.info(
                "Pruned old checkpoints (keeping newest %d): %s",
                self.max_checkpoints_to_keep,
                ", ".join(removed_tags),
            )

    def _clear_existing_checkpoints(self):
        """Remove only recognized checkpoint artifacts before the final save."""
        removed = []
        state_root = Path(self.state_dir)
        weights_root = Path(self.weights_dir)
        for path in state_root.iterdir():
            if not re.fullmatch(r"step_\d+", path.name):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
        for path in weights_root.iterdir():
            if not re.fullmatch(r"step_\d+\.pt", path.name):
                continue
            path.unlink()
            removed.append(str(path))
        logger.info(
            "Cleared %d existing checkpoint artifacts before final save.",
            len(removed),
        )

    def _clear_existing_checkpoints_distributed(self):
        """Run final checkpoint cleanup once while keeping all ranks aligned."""
        self.accelerator.wait_for_everyone()
        local_cleanup_ok = True
        if self.accelerator.is_main_process:
            try:
                self._clear_existing_checkpoints()
            except Exception:
                local_cleanup_ok = False
                logger.exception(
                    "Failed to clear existing checkpoints before final save."
                )
        cleanup_status = torch.tensor(
            [1 if local_cleanup_ok else 0],
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        cleanup_ok = bool(
            torch.all(self.accelerator.gather(cleanup_status) == 1).item()
        )
        if not cleanup_ok:
            raise RuntimeError(
                "Existing checkpoint cleanup failed; refusing the final save."
            )
        self.accelerator.wait_for_everyone()

    def _prepare_final_checkpoint_output_distributed(self):
        """Validate the optional final-only destination before old state is removed."""
        if self.final_checkpoint_output_dir is None:
            return None

        final_checkpoint_root = os.path.join(
            self.final_checkpoint_output_dir, "checkpoints"
        )
        local_prepare_ok = True
        if self.accelerator.is_main_process:
            probe_path = None
            try:
                final_state_dir = os.path.join(final_checkpoint_root, "state")
                final_weights_dir = os.path.join(final_checkpoint_root, "weights")
                ensure_dir(final_state_dir)
                ensure_dir(final_weights_dir)
                probe_path = os.path.join(
                    final_checkpoint_root,
                    f".final_checkpoint_write_probe.{os.getpid()}",
                )
                with open(probe_path, "wb") as probe:
                    probe.write(b"FastWAM final checkpoint write probe\n")
                    probe.flush()
                    os.fsync(probe.fileno())
                os.remove(probe_path)
            except Exception:
                local_prepare_ok = False
                if probe_path is not None:
                    Path(probe_path).unlink(missing_ok=True)
                logger.exception(
                    "Final checkpoint destination is not writable: %s",
                    final_checkpoint_root,
                )

        prepare_status = torch.tensor(
            [1 if local_prepare_ok else 0],
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        prepare_ok = bool(
            torch.all(self.accelerator.gather(prepare_status) == 1).item()
        )
        if not prepare_ok:
            raise RuntimeError(
                "Final checkpoint destination preflight failed; preserving "
                "existing checkpoints and refusing the final save."
            )
        self.accelerator.wait_for_everyone()
        return final_checkpoint_root

    def save_checkpoint(self, *, checkpoint_root=None):
        step_tag = f"step_{self.global_step:06d}"
        destination_root = (
            self.checkpoint_root
            if checkpoint_root is None
            else os.path.abspath(str(checkpoint_root))
        )
        destination_state_dir = os.path.join(destination_root, "state")
        destination_weights_dir = os.path.join(destination_root, "weights")

        # The distributed state is the only artifact that can resume the
        # optimizer, scheduler, RNG, and dataloader exactly.  Persist it before
        # exporting the redundant standalone inference weights.
        self.accelerator.wait_for_everyone()
        state_path = os.path.join(destination_state_dir, step_tag)
        ensure_dir(state_path)

        local_state_save_ok = True
        try:
            self.accelerator.save_state(output_dir=state_path)
        except Exception:
            local_state_save_ok = False
            if self.accelerator.is_main_process:
                logger.exception(
                    "Failed to save full training state to %s; attempting the "
                    "standalone weights checkpoint.",
                    state_path,
                )

        local_state_status = torch.tensor(
            [1 if local_state_save_ok else 0],
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        gathered_state_status = self.accelerator.gather(local_state_status)
        state_save_ok = bool(torch.all(gathered_state_status == 1).item())

        if state_save_ok and self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        elif self.accelerator.is_main_process:
            logger.warning(
                "Skipping trainer_state.json for incomplete full training state at %s.",
                state_path,
            )
        self.accelerator.wait_for_everyone()

        ckpt_path = None
        local_weights_save_ok = True
        if self.accelerator.is_main_process:
            try:
                ckpt_path = self._save_weights_checkpoint(
                    step_tag=step_tag,
                    weights_dir=destination_weights_dir,
                )
            except Exception:
                local_weights_save_ok = False
                logger.exception(
                    "Failed to export standalone weights for %s. Full training "
                    "state is available at %s=%s.",
                    step_tag,
                    state_path,
                    state_save_ok,
                )

        local_weights_status = torch.tensor(
            [1 if local_weights_save_ok else 0],
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        gathered_weights_status = self.accelerator.gather(local_weights_status)
        weights_save_ok = bool(torch.all(gathered_weights_status == 1).item())
        self.accelerator.wait_for_everyone()

        if not state_save_ok and not weights_save_ok:
            raise RuntimeError(
                f"Failed to save both full training state and standalone weights for {step_tag}."
            )

        if self.accelerator.is_main_process:
            try:
                self._prune_old_checkpoints()
            except Exception:
                # Retention cleanup must not strand the other distributed ranks
                # at the following barrier if a filesystem deletion fails.
                logger.exception(
                    "Failed to enforce checkpoint retention limit=%d.",
                    self.max_checkpoints_to_keep,
                )
        self.accelerator.wait_for_everyone()

        return {
            "weights_path": ckpt_path if weights_save_ok else None,
            "state_path": state_path if state_save_ok else None,
        }

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            saved_sampling_contract = payload.get("sampling_contract")
            current_sampling_contract = self._sampling_contract()
            order_only_transition = (
                bool(self.cfg.get("allow_microbatch_sampler_transition", False))
                and bool(getattr(self.train_dataset, "event_anchor_microbatch_shuffle", False))
                and isinstance(saved_sampling_contract, str)
                and current_sampling_contract == saved_sampling_contract + "_microbatchshuffle_v4"
                and "epoch" in payload and "sample_offset_in_epoch" in payload
            )
            if bool(getattr(self.train_dataset, "event_anchor_microbatch_shuffle", False)):
                expected = (self.batch_size, self.accelerator.num_processes, self.gradient_accumulation_steps)
                saved = tuple(payload.get(key) for key in (
                    "checkpoint_batch_size", "checkpoint_world_size", "checkpoint_gradient_accumulation_steps"
                ))
                if saved != expected:
                    raise ValueError("Microbatch resume requires unchanged batch/world/accumulation.")
                if saved_sampling_contract != current_sampling_contract and not order_only_transition:
                    raise ValueError("Unrecognized sampling transition; refusing to reset the data epoch.")
            if (
                current_sampling_contract is not None
                and saved_sampling_contract != current_sampling_contract
                and not order_only_transition
            ):
                self.epoch = 0
                self.batch_in_epoch = 0
                self.total_samples_seen = int(
                    payload.get("total_samples_seen", 0)
                )
                self.train_sampler.set_epoch(0)
                self.train_sampler.set_epoch_offset(0)
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "Sampling contract changed from %r to %r; optimizer, "
                    "scheduler, RNG, global step, and sample count were "
                    "restored, but data progress starts at a fresh complete "
                    "epoch to avoid skipping anchors.",
                    saved_sampling_contract,
                    current_sampling_contract,
                )
                self.accelerator.wait_for_everyone()
                return

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                saved_sample_offset = payload.get("sample_offset_in_epoch")
                if saved_sample_offset is None:
                    self.batch_in_epoch = int(payload["batch_in_epoch"])
                    saved_sample_offset = (
                        self.batch_in_epoch
                        * self.batch_size
                        * self.accelerator.num_processes
                    )
                saved_sample_offset = int(saved_sample_offset)
                if order_only_transition:
                    self.train_sampler.set_reorder_transition(self.epoch, saved_sample_offset)
                    logger.info(
                        "Order-only sampler transition: preserving epoch=%d consumed_samples=%d; "
                        "shuffling only remaining samples. Optimizer/scheduler/RNG restored unchanged.",
                        self.epoch, saved_sample_offset,
                    )
                elif "sampler_reorder_state" in payload:
                    self.train_sampler.load_reorder_state_dict(payload["sampler_reorder_state"])
                current_global_microbatch = (
                    self.batch_size * self.accelerator.num_processes
                )
                if saved_sample_offset % current_global_microbatch:
                    raise ValueError(
                        "Checkpoint sample offset cannot be represented by the "
                        "current global micro-batch without replay or skipping: "
                        f"offset={saved_sample_offset}, "
                        f"global_microbatch={current_global_microbatch}."
                    )
                self.batch_in_epoch = (
                    saved_sample_offset // current_global_microbatch
                )
                self.total_samples_seen = int(
                    payload.get(
                        "total_samples_seen",
                        self._infer_total_samples_seen(),
                    )
                )
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_epoch_offset(0)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    saved_sample_offset,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.total_samples_seen = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.total_samples_seen = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        self._set_training_data_epoch()
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_samples = self.total_samples_seen
        self.run_start_time = time.perf_counter()

        if self.eval_on_start and self.val_dataset is not None:
            metrics = self.evaluate()
            self.accelerator.wait_for_everyone()
            self._record_evaluation(metrics)
            self.run_start_time = time.perf_counter()

        interval_metrics = bool(self.cfg.get("log_interval_metrics", False))
        metric_sum, metric_count, metric_keys = None, 0, None
        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
                self.total_samples_seen += self._real_samples_in_current_batch()
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self._set_training_data_epoch()
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                if interval_metrics:
                    keys, vector = _loss_metric_vector(loss, loss_dict)
                    if metric_keys is not None and keys != metric_keys:
                        raise RuntimeError("Loss metric schema changed within training.")
                    metric_keys = keys
                    metric_sum = vector if metric_sum is None else metric_sum + vector
                    metric_count += 1
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    should_log = (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                    )
                    if should_log:
                        if interval_metrics:
                            values = self.accelerator.gather((metric_sum / metric_count).unsqueeze(0)).mean(0).cpu().tolist()
                            global_loss = values[0]
                            global_loss_metrics = dict(zip(metric_keys[1:], values[1:]))
                            metric_sum, metric_count = None, 0
                        else:
                            global_loss = float(self.accelerator.gather(loss.detach().float().reshape(1)).mean().item())
                            global_loss_metrics = {}
                            for key, value in loss_dict.items():
                                metric_tensor = torch.as_tensor(value, device=loss.device, dtype=torch.float32).detach().reshape(1)
                                global_loss_metrics[key] = float(self.accelerator.gather(metric_tensor).mean().item())
                        m = global_loss_metrics
                        if "action_real_tokens" in m:
                            for kind in ("real", "hold"):
                                m[f"loss_action_{kind}"] = m[f"action_{kind}_loss_sum"] / max(m[f"action_{kind}_tokens"], 1e-12)
                            m["action_hold_fraction"] = m["action_hold_tokens"] / max(m["action_real_tokens"] + m["action_hold_tokens"], 1e-12)
                        if "mem_shift_count" in m:
                            m["mem_shift_abs_frames"] = m["mem_shift_sum"] / max(m["mem_shift_count"], 1e-12)
                            m["mem_noise_alpha"] = m["mem_noise_alpha_sum"] / max(m["mem_noise_count"], 1e-12)
                        if torch.is_tensor(grad_norm):
                            grad_norm_tensor = grad_norm.detach().to(
                                device=loss.device,
                                dtype=torch.float32,
                            ).reshape(1)
                        else:
                            grad_norm_tensor = torch.tensor(
                                [float(grad_norm)],
                                device=loss.device,
                                dtype=torch.float32,
                            )
                        global_grad_norm = float(
                            self.accelerator.gather(grad_norm_tensor).mean().item()
                        )
                        current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if should_log and self.accelerator.is_main_process:
                        eta_str, steps_per_sec, samples_per_sec = self._estimate_eta()
                        data_epoch = self._data_epoch_progress()
                        description = (
                            "[train] data_epoch=%.4f "
                            "samples_seen=%d step=%d/%d loss=%.4f "
                        ) % (
                            data_epoch,
                            self.total_samples_seen,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([
                                f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())
                                if not (k.endswith(("_sum", "_count", "_tokens")) or k.startswith("mem_requested_"))
                            ])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            samples_per_sec,
                            eta_str,
                        )
                        logger.info(description)
                        if self.cfg.get("metrics_jsonl", False):
                            with open(os.path.join(self.output_dir, "training_metrics.jsonl"), "a") as metrics_file:
                                metrics_file.write(json.dumps(dict(
                                    step=self.global_step, loss=global_loss,
                                    lr=current_lr, grad_norm=global_grad_norm,
                                    samples_per_sec=samples_per_sec, data_epoch=data_epoch,
                                    interval_mean=interval_metrics, **global_loss_metrics,
                                )) + "\n")

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "train/data_epoch": data_epoch,
                            "train/samples_seen": self.total_samples_seen,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        self._record_evaluation(metrics)

                    if self.global_step >= self.max_steps:
                        final_checkpoint_root = (
                            self._prepare_final_checkpoint_output_distributed()
                        )
                        if self.clear_existing_checkpoints_before_final_save:
                            self._clear_existing_checkpoints_distributed()
                        ckpt_info = self.save_checkpoint(
                            checkpoint_root=final_checkpoint_root
                        )
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

                    if self._should_save_periodic_checkpoint():
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if (
                        self.eval_once_at_step is not None
                        and self.val_dataset is not None
                        and self.global_step == self.eval_once_at_step
                        and not (
                            self.eval_every > 0
                            and self.global_step % self.eval_every == 0
                        )
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        self._record_evaluation(metrics)

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
