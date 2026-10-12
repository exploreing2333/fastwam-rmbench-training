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
parser.add_argument('--gpus', type=int, default=8)
parser.add_argument('--microbatch', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--subtask-segments', choices=('v72_semantic', 'raw_annotation'))
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
tasks = args.tasks or manifest['remaining_tasks']
for task in tasks:
    command = [sys.executable, str(root / 'scripts/train_task.py'), task, '--foreground',
               '--gpus', str(args.gpus), '--microbatch', str(args.microbatch),
               '--num-workers', str(args.num_workers)]
    if args.dry_run:
        command.append('--dry-run')
    if args.subtask_segments:
        command.extend(['--subtask-segments', args.subtask_segments])
    print(f'[queue] Starting task {task}', flush=True)
    subprocess.run(command, cwd=root, check=True)
print('[queue] All requested tasks finished.', flush=True)
