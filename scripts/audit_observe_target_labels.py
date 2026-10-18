"""Verify historical target-specific label assets (not the current init-KV default)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from fastwam.datasets.lerobot.gt_subtask_video_dataset import GroundTruthSubtaskRobotVideoDataset
from fastwam.runtime import _configure_runtime_cache
from fastwam.utils import misc
from fastwam.utils.samplers import ResumableEpochSampler

ROOT = Path(__file__).resolve().parents[1]
TASK = 'observe_and_pickup'


class MetadataEpisodes:
    def __init__(self, lengths):
        stops = torch.tensor(lengths).cumsum(0)
        self.episode_data_index = {'from': torch.cat([torch.tensor([0]), stops[:-1]]), 'to': stops}
        self.frames = int(stops[-1])

    def __len__(self):
        return self.frames


def metadata_dataset(episodes, rows, source_map, cache):
    dataset = GroundTruthSubtaskRobotVideoDataset.__new__(GroundTruthSubtaskRobotVideoDataset)
    dataset.episode_indices = tuple(episodes)
    dataset.lerobot_dataset = MetadataEpisodes([rows[e]['length'] for e in episodes])
    dataset.subtask_source_map = source_map
    dataset.gt_subtask_text_embedding_cache_dir = cache
    dataset.context_len = 128
    dataset.training_shuffle_unit = 'trajectory'
    dataset.trajectory_anchor_stride = 1
    dataset.gt_subtask_segment_mode = 'v72_semantic'
    dataset.gt_subtask_event_instance_tasks = ('press_button',)
    dataset._gt_stage_index = {}
    dataset._subtask_segment_index = {}
    dataset._subtask_segment_end = {}
    dataset._gt_texts_by_episode = {}
    dataset._gt_raw_ranges_by_episode = {}
    dataset._load_gt_subtask_trajectories()
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/RMBench/data/data')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/lerobot/rmbench_all_12tasks_demo_clean_50')
    parser.add_argument('--text-cache', type=Path, default=ROOT / 'data/gt_subtask_text_embeds')
    parser.add_argument('--rgb-samples', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/observe_target_labels_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.raw_root / TASK / 'demo_clean'
    annotations = json.loads((directory / 'language_annotation.json').read_text())
    backup = directory / 'language_annotation.before_target_labels.json'
    original = json.loads(backup.read_text())
    labels = json.loads((directory / 'target_object_labels.json').read_text())
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == labels['source_annotation_sha256']
    assert labels['episode_count'] == 50
    assert set(labels['records']) == set(annotations) == set(original)
    source = json.loads((ROOT / 'data/source_map_template.json').read_text())
    source['raw_root'] = str(args.raw_root.resolve())
    rows = {row['episode_index']: row for row in source['entries']}
    task_info = json.loads((ROOT / 'data/task_manifest.json').read_text())['tasks'][TASK]
    episodes = task_info['episode_indices']
    contexts = {}
    changed_frames = 0
    with tempfile.TemporaryDirectory(prefix='fastwam-target-label-audit-') as temporary:
        temporary = Path(temporary)
        source_map = temporary / 'source_map.json'
        source_map.write_text(json.dumps(source))
        new = metadata_dataset(episodes, rows, source_map, args.text_cache.resolve())
        old_directory = temporary / 'raw' / TASK / 'demo_clean'
        old_directory.mkdir(parents=True)
        (old_directory / 'language_annotation.json').write_bytes(backup.read_bytes())
        old_source = dict(source, raw_root=str(temporary / 'raw'))
        old_map = temporary / 'old_source_map.json'
        old_map.write_text(json.dumps(old_source))
        old = metadata_dataset(episodes, rows, old_map, args.text_cache.resolve())
        for episode in episodes:
            row = rows[episode]
            key = f"episode_{row['source_episode_id']}"
            description = labels['records'][key]['target_description']
            expected = (
                f'Observe the {description} on the shelf and wait for the wall to appear.',
                f'Pick up the matching {description} from the table.',
            )
            assert [r[1] for r in annotations[key]] == [r[1] for r in original[key]]
            assert [r[0] for r in annotations[key]] == [expected[0]] + [expected[1]] * 3
            assert new._gt_texts_by_episode[episode] == expected
            np.testing.assert_array_equal(new._gt_stage_index[episode], old._gt_stage_index[episode])
            np.testing.assert_array_equal(new._subtask_segment_end[episode], old._subtask_segment_end[episode])
            for text in expected:
                if text not in contexts:
                    context = new._load_gt_context(text)
                    assert context.dtype == torch.bfloat16
                    contexts[text] = context
            stages = new._gt_stage_index[episode]
            for stage, text in enumerate(expected):
                assert description in text
                changed_frames += int(np.count_nonzero(stages == stage))
        assert changed_frames == task_info['frames']
        assert len(contexts) == 38
        assert new.sampling_contract != old.sampling_contract
        old_sampler = ResumableEpochSampler(old, seed=42, batch_size=8, num_processes=8)
        new_sampler = ResumableEpochSampler(new, seed=42, batch_size=8, num_processes=8)
        order = list(new_sampler)
        assert order == list(old_sampler)
        assert len([i for i in order if i >= 0]) == len(set(i for i in order if i >= 0)) == changed_frames
        new_sampler.set_resume_batch_offset(2)
        assert list(new_sampler) == order[128:]
        checks = 0
        selected = []
        if args.rgb_samples:
            seen = set()
            for episode in episodes:
                key = f"episode_{rows[episode]['source_episode_id']}"
                kind = labels['records'][key]['target_modelname']
                if kind not in seen:
                    seen.add(kind)
                    selected.append(episode)
            cfg = OmegaConf.load(ROOT / f'configs/portable_{TASK}.yaml')
            # This audit deliberately checks the historical target-label setup.
            cfg.data.train._target_ = 'fastwam.datasets.lerobot.gt_subtask_video_dataset.GroundTruthSubtaskRobotVideoDataset'
            for key in ('event_anchor_latent_cache_root', 'event_anchor_max_frames',
                        'event_anchor_batch_grouping', 'event_anchor_cache_max_open_episodes',
                        'init_frame_anchor_only', 'gt_subtask_annotation_files'):
                cfg.data.train.pop(key, None)
            cfg.data.train.episode_indices = selected
            cfg.data.train.dataset_dirs = [str(args.dataset_root.resolve())]
            cfg.data.train.subtask_source_map = str(source_map)
            cfg.data.train.gt_subtask_text_embedding_cache_dir = str(args.text_cache.resolve())
            cfg.data.train.pretrained_norm_stats = str(ROOT / f'data/task_norm_stats/{TASK}.json')
            _configure_runtime_cache(cfg)
            misc.register_work_dir(temporary / 'rgb_samples')
            dataset = instantiate(cfg.data.train)
            normalizer = dataset.lerobot_dataset.processor.normalizer.normalizers['action']['default']
            offsets = dataset.lerobot_dataset.episode_data_index['from'].tolist()
            for position, episode in enumerate(selected):
                stage = dataset._gt_stage_index[episode]
                ends = dataset._subtask_segment_end[episode]
                path = args.dataset_root / f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
                actions = pq.read_table(path, columns=['action'])['action'].to_pylist()
                normalized = normalizer.forward(torch.tensor(actions, dtype=torch.float32))
                key = f"episode_{rows[episode]['source_episode_id']}"
                raw_starts = np.cumsum([0] + [r[1] for r in annotations[key][:-1]])
                frames = {0, len(stage) - 1}
                for start in raw_starts:
                    frames.update({max(0, int(start) - 8), int(start)})
                for frame in sorted(f for f in frames if f < len(stage)):
                    sample = dataset[int(offsets[position]) + frame]
                    text = new._gt_texts_by_episode[episode][int(stage[frame])]
                    assert sample['current_subtask'] == text
                    assert labels['records'][key]['target_description'] in sample['prompt']
                    torch.testing.assert_close(sample['context'], contexts[text], rtol=0, atol=0)
                    target = np.minimum(frame + np.arange(32), ends[frame])
                    torch.testing.assert_close(sample['action'], normalized[torch.from_numpy(target)], rtol=0, atol=0)
                    assert tuple(sample['video'].shape) == (3, 9, 384, 320)
                    assert tuple(sample['proprio'].shape) == (1, 14)
                    assert sample['context_mask'].all()
                    assert torch.isfinite(sample['video']).all()
                    last_valid = 0
                    for i, offset in enumerate(range(0, 33, 4)):
                        if frame + offset <= ends[frame]:
                            last_valid = i
                        else:
                            torch.testing.assert_close(sample['video'][:, i], sample['video'][:, last_valid], rtol=0, atol=0)
                    checks += 1
        result = dict(passed=True, task=TASK, episodes=50, frames=changed_frames,
                      target_descriptions=19, unique_new_contexts=len(contexts),
                      stages_per_episode=2, original_durations_and_stage_boundaries_preserved=True,
                      all_frames_use_target_specific_labels=True, sampling_order_and_resume_preserved=True,
                      sampling_contract=new.sampling_contract, previous_sampling_contract=old.sampling_contract,
                      real_sample_episodes=selected, actual_getitem_checks=checks,
                      exact_real_action_and_context_checks=bool(checks), gpu_jobs_started=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
