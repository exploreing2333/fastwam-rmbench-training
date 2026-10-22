"""Check full task text, continuous targets and shuffled coverage using real data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from fastwam.datasets.lerobot.full_instruction_video_dataset import FullInstructionRobotVideoDataset
from fastwam.utils import misc
from fastwam.utils.samplers import ResumableEpochSampler
from prepare_full_instruction_text_cache import prepare
from training_modes import configure_instruction_mode

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/lerobot/rmbench_all_12tasks_demo_clean_50')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/RMBench/data/data')
    parser.add_argument('--anchor-cache', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/full_instruction_training_audit.json')
    args = parser.parse_args()
    os.environ.setdefault('FASTWAM_BUNDLE_ROOT', str(ROOT))
    os.environ.setdefault('FASTWAM_RUNTIME_ROOT', str(ROOT / '.runtime'))
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / 'data/task_manifest.json').read_text())
    source = json.loads((ROOT / 'data/source_map_template.json').read_text())
    source_rows = {row['episode_index']: row for row in source['entries']}
    checks = []
    with tempfile.TemporaryDirectory(prefix='full-instruction-audit-', dir='/tmp') as temporary:
        for task, info in manifest['tasks'].items():
            cfg = configure_instruction_mode(OmegaConf.load(ROOT / f'configs/portable_{task}.yaml'), 'full')
            cfg.data.train.dataset_dirs = [str(args.dataset_root.resolve())]
            cfg.data.train.pretrained_norm_stats = str(ROOT / f'data/task_norm_stats/{task}.json')
            prepare(task, cfg.data.train.text_embedding_cache_dir)
            episode = info['episode_indices'][0]
            cfg.data.train.episode_indices = [episode]
            if args.anchor_cache and cfg.model.get('init_frame_kv_config') is not None:
                cfg.data.train.event_anchor_latent_cache_root = str(args.anchor_cache.resolve())
            misc.register_work_dir(Path(temporary) / task)
            dataset = instantiate(cfg.data.train)
            row = source_rows[episode]
            length = len(dataset)
            assert length == row['length']
            # Raw annotations are read solely by this audit to locate a boundary.
            # The training dataset never receives their paths.
            annotations = json.loads((args.raw_root / task / 'demo_clean/language_annotation.json').read_text())
            segments = annotations[f"episode_{row['source_episode_id']}"]
            boundary = 0
            for index, (text, duration) in enumerate(segments[:-1]):
                boundary += int(duration)
                if boundary < length - 32 and text != segments[index + 1][0]:
                    break
            assert 0 < boundary < length - 1, (task, boundary, length)
            anchor = max(0, boundary - 8)
            assert anchor < boundary < anchor + 32
            for frame in (anchor, length - 1):
                sample = dataset[frame]
                expected = dataset.lerobot_dataset[frame]
                assert sample['task'] == row['task']
                assert sample['prompt'].endswith(row['task'])
                assert not any(key in sample for key in ('current_subtask', 'gt_subtask_index', 'gt_subtask_end_frame', 'action_loss_mask'))
                torch.testing.assert_close(sample['action'], expected['action'], rtol=0, atol=0)
                torch.testing.assert_close(sample['action_is_pad'], expected['action_is_pad'], rtol=0, atol=0)
                assert tuple(sample['action'].shape) == (32, 14)
                assert tuple(sample['video'].shape) == (3, 9, 384, 320)
                assert tuple(sample['proprio'].shape) == (1, 14)
                assert tuple(sample['context'].shape) == (128, 4096)
                assert sample['context_mask'].all()
                if frame == anchor:
                    assert not sample['action_is_pad'].any()
                else:
                    assert sample['action_is_pad'].any()
                if cfg.model.get('init_frame_kv_config') is not None:
                    assert sample['event_anchor_frame_indices'].tolist() == [0]
                    assert sample['event_anchor_group_id'].item() == episode
            # Check all 50 episodes' anchor counts without decoding every RGB.
            class Frames:
                def __len__(self):
                    return info['frames']
            plan = FullInstructionRobotVideoDataset.__new__(FullInstructionRobotVideoDataset)
            plan.lerobot_dataset = Frames()
            plan.episode_indices = tuple(info['episode_indices'])
            plan.context_len = 128
            plan.training_shuffle_unit = 'frame'
            sampler = ResumableEpochSampler(plan, seed=42, batch_size=8, num_processes=8)
            epoch_hashes = []
            for epoch in range(3):
                sampler.set_epoch(epoch)
                order = list(sampler)
                real = [index for index in order if index >= 0]
                assert sorted(real) == list(range(info['frames']))
                expected_order = torch.randperm(info['frames'], generator=torch.Generator().manual_seed(42 + epoch)).tolist()
                assert real == expected_order
                sampler.set_resume_batch_offset(2)
                assert list(sampler) == order[128:]
                sampler.clear_resume_batch_offset()
                epoch_hashes.append(hashlib.sha256(torch.tensor(real).numpy().tobytes()).hexdigest())
            assert len(set(epoch_hashes)) == 3
            checks.append(dict(task=task, episode=episode, sampled_frames=[anchor, length - 1],
                               crossed_subtask_boundary=boundary, instruction=row['task'],
                               actual_targets_equal_continuous_base_dataset=True,
                               only_episode_end_padding=True, shuffled_real_frames_per_epoch=info['frames'],
                               three_epochs_match_fastwam_randperm=True, exact_sampler_resume=True,
                               init_kv_preserved=cfg.model.get('init_frame_kv_config') is not None))
            print(f'[audit] {task}: continuous targets across boundary {boundary}; full text and 3 shuffled epochs verified', flush=True)
    result = dict(passed=True, tasks=len(checks), training_episodes=600,
                  shuffled_training_frames=sum(row['shuffled_real_frames_per_epoch'] for row in checks),
                  real_rgb_action_checks=2 * len(checks), gpu_jobs_started=0, checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
