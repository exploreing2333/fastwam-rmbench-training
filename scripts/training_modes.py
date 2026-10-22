"""Build a full-instruction variant of each task's existing training config."""
from pathlib import Path

from omegaconf import OmegaConf


def task_run_root(root, task, instruction_mode):
    directory = Path(root) / 'runs' / task
    return directory / 'full_instruction' if instruction_mode == 'full' else directory


def configure_instruction_mode(cfg, instruction_mode):
    if instruction_mode not in ('subtask', 'full'):
        raise ValueError(f'Unknown instruction mode: {instruction_mode}')
    cfg.instruction_mode = instruction_mode
    if instruction_mode == 'subtask':
        return cfg
    data = cfg.data.train
    for name in list(data):
        if name.startswith(('gt_subtask_', 'subtask_')) or name in (
            'press_button_label_mode', 'exclude_terminal_hold_loss_subtasks',
        ):
            del data[name]
    init_kv = OmegaConf.select(cfg, 'model.init_frame_kv_config') is not None
    target = ('EpisodeInitFrameFullInstructionRobotVideoDataset' if init_kv
              else 'FullInstructionRobotVideoDataset')
    data._target_ = 'fastwam.datasets.lerobot.full_instruction_video_dataset.' + target
    data.training_shuffle_unit = 'frame'
    data.trajectory_anchor_stride = 1
    data.text_embedding_cache_dir = '${oc.env:FASTWAM_RUNTIME_ROOT}/full_instruction_text_embeds'
    data.text_embedding_use_true_mask = False
    cfg.output_dir = str(task_run_root('${oc.env:FASTWAM_BUNDLE_ROOT}', cfg.rmbench_task, 'full'))
    cfg.wandb.name = str(cfg.wandb.name).replace('_gt_', '_full_instruction_')
    return cfg
