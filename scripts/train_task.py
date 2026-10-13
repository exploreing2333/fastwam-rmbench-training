"""One command per task; fresh GT-current-only reference training by default."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid

from omegaconf import OmegaConf
from training_processes import conflicting_training_processes

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'data/task_manifest.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=list(manifest['tasks']))
    parser.add_argument('--gpus', type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--grad-accum', type=int)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--run-id')
    parser.add_argument('--master-port', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int, default=200000)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--cached-stats', action='store_true', help='Use bundled statistics computed with the unchanged reference algorithm.')
    parser.add_argument('--subtask-segments', choices=('v72_semantic', 'raw_annotation'),
                        help='Override the default V72 semantic spans or reproduce the original raw GT spans.')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
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
    run_id = cli.run_id or (datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S') + '_' + uuid.uuid4().hex[:6])
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
    if cli.resume is not None:
        if not cli.resume.exists():
            parser.error(f'resume checkpoint does not exist: {cli.resume}')
        cfg.resume = str(cli.resume.resolve())
    if cli.gradient_checkpointing:
        cfg.model.mot_checkpoint_mixed_attn = True
        cfg.model.video_dit_config.use_gradient_checkpointing = True
        cfg.model.action_dit_config.use_gradient_checkpointing = True
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
    run_dir.mkdir(parents=True, exist_ok=True)
    config = run_dir / 'launch_config.yaml'
    OmegaConf.save(OmegaConf.create(cfg_data), config)
    command = [sys.executable, '-u', '-m', 'accelerate.commands.launch',
               '--config_file', str(ROOT / 'scripts/accelerate_configs/accelerate_zero1_ds.yaml'),
               '--num_processes', str(cli.gpus), '--num_machines', '1', '--machine_rank', '0',
               '--main_process_ip', '127.0.0.1', '--main_process_port', str(port),
               str(ROOT / 'scripts/train.py'), '--config-path', str(run_dir), '--config-name', 'launch_config']
    record = dict(task=cli.task, run_id=run_id, command=command, gpus=cli.gpus,
                  microbatch=cli.microbatch, gradient_accumulation_steps=accumulation,
                  global_batch_size=64, max_steps=cli.max_steps, resume=cfg_data['resume'],
                  gt_subtask_segment_mode=cfg_data['data']['train']['gt_subtask_segment_mode'],
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                  run_dir=str(run_dir), created_at=datetime.now(timezone.utc).isoformat())
    (run_dir / 'launch_command.json').write_text(json.dumps(record, indent=2) + '\n')
    print(shlex.join(command), flush=True)
    print(f'[training] task={cli.task} episodes=50 frames={info["frames"]} '
          f'global_batch=64 ({cli.gpus} GPUs x {cli.microbatch} x accumulation {accumulation})', flush=True)
    print('[subtasks] mode=' + cfg_data['data']['train']['gt_subtask_segment_mode']
          + ' event_instance_tasks=' + ','.join(cfg_data['data']['train']['gt_subtask_event_instance_tasks']), flush=True)
    if cli.dry_run:
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
        conflicts = conflicting_training_processes(ROOT, cli.gpus, record['gpu_uuids'])
        if conflicts:
            parser.error('training is already running on these GPUs: '
                         + ', '.join(sorted({row['run_dir'] for row in conflicts}))
                         + '; inspect with bash stop_training.sh --list')
        (run_dir / 'launch_command.json').write_text(json.dumps(record, indent=2) + '\n')
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
