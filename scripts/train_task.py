"""One command per task; resume the latest complete task state by default."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import uuid

from omegaconf import OmegaConf
from training_processes import conflicting_training_processes, find_training_processes
from checkpoint_resume import inspect_training_state, latest_training_state

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'data/task_manifest.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=list(manifest['tasks']))
    parser.add_argument('--gpus', type=int, choices=(1, 2, 4, 8))
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--grad-accum', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--run-id')
    parser.add_argument('--master-port', type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument('--resume', nargs='?', const='latest', default='latest',
                              help='Latest complete task checkpoint (default), or an explicit full-state directory.')
    resume_group.add_argument('--fresh', action='store_true', help='Start a new run from Wan initialization.')
    parser.add_argument('--max-steps', type=int, default=200000)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--cached-stats', action='store_true', help='Use bundled statistics computed with the unchanged reference algorithm.')
    parser.add_argument('--subtask-segments', choices=('v72_semantic', 'raw_annotation'),
                        help='Override the default V72 semantic spans or reproduce the original raw GT spans.')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    selected, skipped = None, []
    if not cli.fresh:
        try:
            if cli.resume == 'latest':
                selected, skipped = latest_training_state(ROOT/'runs'/cli.task)
            else:
                selected = inspect_training_state(Path(cli.resume))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    saved_cfg = None
    if selected is not None:
        previous_config = selected.run_dir/'launch_config.yaml'
        if previous_config.is_file():
            saved_cfg = OmegaConf.load(previous_config)
        if cli.gpus is not None and cli.gpus != selected.world_size:
            parser.error(f'Full-state resume requires the saved GPU count {selected.world_size}; requested {cli.gpus}')
        if cli.microbatch is not None and cli.microbatch != selected.microbatch:
            parser.error(f'Full-state resume requires saved microbatch {selected.microbatch}')
        if cli.grad_accum is not None and cli.grad_accum != selected.accumulation:
            parser.error(f'Full-state resume requires saved accumulation {selected.accumulation}')
        cli.gpus, cli.microbatch, cli.grad_accum = selected.world_size, selected.microbatch, selected.accumulation
    else:
        cli.gpus = cli.gpus or 8
        cli.microbatch = 8 if cli.microbatch is None else cli.microbatch
    if cli.num_workers is None:
        cli.num_workers = int(saved_cfg.get('num_workers',8)) if saved_cfg is not None else 8
    if cli.microbatch <= 0 or cli.num_workers < 0 or cli.max_steps <= 0:
        parser.error('microbatch/max-steps must be positive and num-workers must be nonnegative')
    if cli.master_port is not None and not 1 <= cli.master_port <= 65535:
        parser.error('master-port must be between 1 and 65535')
    unit = cli.gpus * cli.microbatch
    accumulation = cli.grad_accum
    if accumulation is None:
        if 64 % unit:
            parser.error('gpus * microbatch must divide the reference global batch size 64')
        accumulation = 64 // unit
    if accumulation <= 0 or unit * accumulation != 64:
        parser.error('gpus * microbatch * grad-accum must equal the reference global batch size 64')
    launch_id = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S') + '_' + uuid.uuid4().hex[:6]
    task_root = (ROOT/'runs'/cli.task).resolve()
    reusable_run = selected is not None and selected.run_dir.parent == task_root
    run_id = cli.run_id or (selected.run_dir.name if reusable_run else launch_id)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_id):
        parser.error('run-id may contain letters, digits, dots, underscores, and hyphens')
    cfg = OmegaConf.load(ROOT / f'configs/portable_{cli.task}.yaml')
    cfg.batch_size = cli.microbatch
    cfg.gradient_accumulation_steps = accumulation
    cfg.num_workers = cli.num_workers
    cfg.max_steps = cli.max_steps
    cfg.output_dir = str(ROOT / 'runs' / cli.task / run_id)
    if cli.subtask_segments:
        cfg.data.train.gt_subtask_segment_mode = cli.subtask_segments
    cfg.resume = str(selected.path) if selected is not None else None
    for key in ('model.mot_checkpoint_mixed_attn',
                'model.video_dit_config.use_gradient_checkpointing',
                'model.action_dit_config.use_gradient_checkpointing'):
        saved = bool(OmegaConf.select(saved_cfg,key,default=False)) if saved_cfg is not None else False
        if cli.gradient_checkpointing or saved:
            OmegaConf.update(cfg,key,True)
    if cli.cached_stats:
        stats = ROOT / f'data/task_norm_stats/{cli.task}.json'
        if not stats.is_file():
            parser.error(f'task statistics are missing: {stats}')
        cfg.data.train.pretrained_norm_stats = str(stats)
    cfg_data = OmegaConf.to_container(cfg, resolve=True)
    info = manifest['tasks'][cli.task]
    assert cfg_data['data']['train']['episode_indices'] == info['episode_indices']
    assert cfg_data['data']['train']['progress_cache_root'] is None
    assert cfg_data['data']['train']['progress_subtask_text_embedding_cache_dir'] is None
    assert cfg_data['model']['load_text_encoder'] is False
    assert cfg_data['model']['action_scheduler']['train_shift'] == 5.0
    port = cli.master_port or (29800 + list(manifest['tasks']).index(cli.task))
    run_dir = Path(cfg_data['output_dir'])
    if selected is not None and selected.step >= cli.max_steps:
        print(f'[complete] task={cli.task} latest_step={selected.step} target={cli.max_steps}; no training needed.', flush=True)
        return 0
    if selected is None and (run_dir/'checkpoints').exists():
        parser.error(f'Fresh run must use a new run-id; checkpoints already exist in {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    # Dry runs must not replace the live run's launch configuration.
    config_dir = run_dir/'dry_runs'/launch_id if cli.dry_run else run_dir
    config = config_dir/'launch_config.yaml'
    command = [sys.executable, '-u', '-m', 'accelerate.commands.launch',
               '--config_file', str(ROOT / 'scripts/accelerate_configs/accelerate_zero1_ds.yaml'),
               '--num_processes', str(cli.gpus), '--num_machines', '1', '--machine_rank', '0',
               '--main_process_ip', '127.0.0.1', '--main_process_port', str(port),
               str(ROOT / 'scripts/train.py'), '--config-path', str(config_dir), '--config-name', 'launch_config']
    record = dict(task=cli.task, run_id=run_id, command=command, gpus=cli.gpus,
                  microbatch=cli.microbatch, gradient_accumulation_steps=accumulation,
                  num_workers=cli.num_workers,
                  global_batch_size=64, max_steps=cli.max_steps, resume=cfg_data['resume'],
                  gt_subtask_segment_mode=cfg_data['data']['train']['gt_subtask_segment_mode'],
                  exclude_terminal_hold_loss_subtasks=cfg_data['data']['train'].get('exclude_terminal_hold_loss_subtasks', []),
                  resume_mode='full_state' if selected is not None else 'fresh',
                  resume_step=selected.step if selected is not None else 0,
                  resume_skipped_incomplete=skipped, launch_id=launch_id,
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                  run_dir=str(run_dir), created_at=datetime.now(timezone.utc).isoformat())
    def save_launch():
        config_dir.mkdir(parents=True, exist_ok=True)
        if not cli.dry_run and config.exists():
            history = run_dir/'launch_history'/launch_id
            history.mkdir(parents=True, exist_ok=True)
            for name in ('launch_config.yaml','launch_command.json'):
                previous = run_dir/name
                if previous.is_file():
                    shutil.copy2(previous,history/name)
        OmegaConf.save(OmegaConf.create(cfg_data),config)
        (config_dir/'launch_command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(shlex.join(command), flush=True)
    print(f'[training] task={cli.task} episodes=50 frames={info["frames"]} '
          f'global_batch=64 ({cli.gpus} GPUs x {cli.microbatch} x accumulation {accumulation})', flush=True)
    print('[subtasks] mode=' + cfg_data['data']['train']['gt_subtask_segment_mode']
          + ' event_instance_tasks=' + ','.join(cfg_data['data']['train']['gt_subtask_event_instance_tasks']), flush=True)
    print('[action loss] exclude_terminal_hold_subtasks='
          + json.dumps(cfg_data['data']['train'].get('exclude_terminal_hold_loss_subtasks', [])), flush=True)
    print(f'[resume] mode={record["resume_mode"]} step={record["resume_step"]} path={cfg_data["resume"]}', flush=True)
    for skipped_state in skipped:
        print('[resume] skipped incomplete state: ' + json.dumps(skipped_state), flush=True)
    if cli.dry_run:
        save_launch()
        print('[dry-run] No GPU training process was launched.', flush=True)
        return 0
    import torch
    available = torch.cuda.device_count()
    if available < cli.gpus:
        parser.error(f'requested {cli.gpus} GPUs but only {available} are visible; use --gpus to select a supported count')
    if not torch.cuda.is_bf16_supported():
        parser.error('the reference training requires BF16-capable GPUs')
    check = [sys.executable, str(ROOT / 'scripts/check_runtime.py'), '--gpus', str(cli.gpus),
             '--microbatch', str(cli.microbatch), '--num-workers', str(cli.num_workers),
             '--master-port', str(port), '--output', str(run_dir / 'runtime_preflight.json')]
    if subprocess.call(check, cwd=ROOT) != 0:
        return 1
    uuids = [str(getattr(torch.cuda.get_device_properties(i), 'uuid', '')) for i in range(cli.gpus)]
    record['gpu_uuids'] = uuids if all(uuids) else []
    # Serialize check+spawn so two launchers cannot pass the same idle check.
    with (ROOT / 'runs/.launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX)
        same_run = [row for row in find_training_processes(ROOT)
                    if Path(row['run_dir']).resolve() == run_dir.resolve()]
        if same_run:
            parser.error(f'This run is already training: {run_dir}; refusing to overwrite its launch configuration')
        conflicts = conflicting_training_processes(ROOT, cli.gpus, record['gpu_uuids'])
        if conflicts:
            parser.error('training is already running on these GPUs: '
                         + ', '.join(sorted({row['run_dir'] for row in conflicts}))
                         + '; inspect with bash stop_training.sh --list')
        save_launch()
        if cli.foreground:
            process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        else:
            log = run_dir / 'stdout.log'
            with log.open('ab', buffering=0) as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            record['log'] = str(log)
        (run_dir / 'training.pid').write_text(str(process.pid) + '\n')
        (run_dir / 'training.pgid').write_text(str(process.pid) + '\n')
        record['pid'] = process.pid
        (run_dir / 'launch_command.json').write_text(json.dumps(record, indent=2) + '\n')
    if cli.foreground:
        try:
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            return process.wait()
    print(f'[started] pid={process.pid} log={record["log"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
