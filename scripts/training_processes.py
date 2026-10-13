"""Identify this checkout's training processes without matching unrelated jobs."""
import json
import os
from pathlib import Path


def find_training_processes(root):
    root = Path(root).resolve()
    script = str(root / 'scripts/train.py')
    runs = root / 'runs'
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = directory.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if script not in args or '--config-path' not in args:
                continue
            run = Path(args[args.index('--config-path') + 1]).resolve()
            if runs not in run.parents:
                continue
            status = directory.joinpath('status').read_text()
            fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
            if fields['State'].strip().startswith('Z'):
                continue
            stat = directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            matches.append(dict(pid=int(directory.name), run_dir=str(run),
                                start_ticks=stat[19], state=fields['State'].strip(),
                                tracer_pid=int(fields['TracerPid'])))
        except (OSError, ValueError, IndexError, KeyError):
            continue
    return sorted(matches, key=lambda row: row['pid'])


def _selector_keys(visible, count):
    if visible in (None, '', 'all'):
        return {'index:' + str(i) for i in range(count)}
    values = [part.strip() for part in visible.split(',')][:count]
    return {('index:' + str(int(value))) if value.isdigit() else ('uuid:' + value) for value in values}


def conflicting_training_processes(root, gpus, gpu_uuids=None):
    own_keys = _selector_keys(os.environ.get('CUDA_VISIBLE_DEVICES'), gpus)
    conflicts = []
    records = {}
    for row in find_training_processes(root):
        if row['run_dir'] not in records:
            try:
                records[row['run_dir']] = json.loads((Path(row['run_dir']) / 'launch_command.json').read_text())
            except (OSError, ValueError):
                records[row['run_dir']] = None
        record = records[row['run_dir']]
        if not record:
            conflicts.append(row)
            continue
        if gpu_uuids and record.get('gpu_uuids'):
            overlap = bool(set(gpu_uuids) & set(record['gpu_uuids']))
        else:
            keys = _selector_keys(record.get('CUDA_VISIBLE_DEVICES'), int(record.get('gpus', 8)))
            mixed_names = {key.split(':', 1)[0] for key in keys | own_keys}
            overlap = bool(keys & own_keys) or len(mixed_names) > 1
        if overlap:
            conflicts.append(row)
    return conflicts


def still_same_process(row):
    try:
        stat = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return stat[19] == row['start_ticks'] and stat[0] != 'Z'
    except (OSError, IndexError):
        return False
