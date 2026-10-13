"""List or stop one selected training run belonging to this checkout."""
import argparse
import os
from pathlib import Path
import signal
import time

from training_processes import find_training_processes, still_same_process

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--all', action='store_true', help='Select all training runs in this checkout')
    parser.add_argument('--stop', action='store_true', help='Send SIGTERM to selected training processes')
    parser.add_argument('--force', action='store_true', help='After SIGTERM, use SIGKILL for selected processes still alive')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.stop and not (args.run_dir or args.all):
        parser.error('--stop requires --run-dir or --all')
    if args.force and not args.stop:
        parser.error('--force requires --stop')
    selected = args.run_dir.resolve() if args.run_dir else None
    rows = [row for row in find_training_processes(ROOT) if selected is None or row['run_dir'] == str(selected)]
    for row in rows:
        print(f"pid={row['pid']} state={row['state']} tracer={row['tracer_pid']} run={row['run_dir']}", flush=True)
    if not args.stop:
        print(f'{len(rows)} training processes found; listing only.', flush=True)
        return 0
    for row in rows:
        if still_same_process(row):
            try:
                os.kill(row['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(still_same_process(row) for row in rows):
        time.sleep(0.2)
    remaining = [row for row in rows if still_same_process(row)]
    if remaining and args.force:
        for row in remaining:
            if still_same_process(row):
                try:
                    os.kill(row['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(1)
        remaining = [row for row in rows if still_same_process(row)]
    if remaining:
        for row in remaining:
            print(f"Still alive: pid={row['pid']} tracer={row['tracer_pid']} run={row['run_dir']}", flush=True)
        print('Inspect the process state and any attached tracer before restarting.', flush=True)
        return 1
    print('Selected training processes stopped.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
