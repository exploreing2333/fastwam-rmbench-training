"""Locate complete task-local training states without loading model tensors."""
from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class TrainingState:
    path: Path
    run_dir: Path
    step: int
    world_size: int
    microbatch: int
    accumulation: int
    saved_at_ns: int


def inspect_training_state(path: Path) -> TrainingState:
    path = path.expanduser().resolve()
    marker = path / 'trainer_state.json'
    payload = json.loads(marker.read_text())
    step = int(payload['global_step'])
    match = re.fullmatch(r'step_(\d+)', path.name)
    if not match or int(match.group(1)) != step or step < 0:
        raise ValueError('Checkpoint directory and trainer global_step disagree')
    world = int(payload['checkpoint_world_size'])
    micro = int(payload['checkpoint_batch_size'])
    accumulation = int(payload['checkpoint_gradient_accumulation_steps'])
    if min(world, micro, accumulation) < 1:
        raise ValueError('Invalid checkpoint batch/world/accumulation metadata')
    # trainer_state.json is written only after every rank saves successfully.
    # Also reject deleted/truncated shards and a stale completion marker.
    expected = [path/'scheduler.bin'] + [path/f'random_states_{i}.pkl' for i in range(world)]
    model = list(path.glob('**/*_model_states.pt'))
    optim = list(path.glob('**/*_optim_states.pt'))
    if model:
        if len(optim) < world:
            raise ValueError('Missing DeepSpeed optimizer shards')
        expected += model + optim
    else:
        # Accept Accelerate's non-DeepSpeed layout as well.
        weights = [p for p in (path/'model.safetensors', path/'pytorch_model.bin') if p.is_file()]
        if not weights:
            raise ValueError('Missing saved model state')
        expected += weights + [path/'optimizer.bin']
    if any(not p.is_file() or p.stat().st_size <= 0 for p in expected):
        raise ValueError('Missing or empty state artifact')
    return TrainingState(path, path.parent.parent.parent, step, world, micro,
                         accumulation, marker.stat().st_mtime_ns)


def latest_training_state(task_root: Path):
    candidates, skipped = [], []
    for path in task_root.glob('*/checkpoints/state/step_*'):
        try:
            candidates.append(inspect_training_state(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({'path':str(path),'reason':str(exc)})
    # Recency matters across runs: a deliberately fresh run at 2k must win
    # over an older completed run at 200k. Step breaks timestamp ties.
    latest = max(candidates, key=lambda c:(c.saved_at_ns,c.step,str(c.path))) if candidates else None
    if latest is None:
        weights = list(task_root.glob('*/checkpoints/weights/step_*.pt'))
        if skipped or weights:
            raise ValueError(
                f'Checkpoints exist under {task_root}, but none has a complete training state. '
                'A weights-only .pt cannot restore optimizer/step/sampler. '
                'Restore the matching checkpoints/state directory, or use --fresh explicitly.'
            )
    return latest, skipped
