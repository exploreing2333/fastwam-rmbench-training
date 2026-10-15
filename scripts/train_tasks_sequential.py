"""Train remaining tasks sequentially; one distributed task at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'data/task_manifest.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tasks', nargs='+', choices=list(manifest['tasks']))
parser.add_argument('--gpus', type=int, choices=(1,2,4,8))
parser.add_argument('--microbatch', type=int)
parser.add_argument('--num-workers', type=int)
parser.add_argument('--subtask-segments', choices=('v72_semantic', 'raw_annotation'))
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--fresh', action='store_true')
parser.add_argument('--max-steps', type=int, default=200000)
args = parser.parse_args()
tasks = args.tasks or manifest['remaining_tasks']
for task in tasks:
    command = [sys.executable, str(root / 'scripts/train_task.py'), task, '--foreground',
               '--max-steps', str(args.max_steps)]
    if args.num_workers is not None:
        command.extend(['--num-workers',str(args.num_workers)])
    if args.gpus is not None:
        command.extend(['--gpus',str(args.gpus)])
    if args.microbatch is not None:
        command.extend(['--microbatch',str(args.microbatch)])
    if args.fresh:
        command.append('--fresh')
    if args.dry_run:
        command.append('--dry-run')
    if args.subtask_segments:
        command.extend(['--subtask-segments', args.subtask_segments])
    print(f'[queue] Starting task {task}', flush=True)
    subprocess.run(command, cwd=root, check=True)
print('[queue] All requested tasks finished.', flush=True)
