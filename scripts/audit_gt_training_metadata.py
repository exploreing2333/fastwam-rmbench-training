"""Audit every task's authoritative labels, scalar alignment, and sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.datasets.lerobot.gt_subtask_video_dataset import GroundTruthSubtaskRobotVideoDataset
from fastwam.utils.samplers import ResumableEpochSampler

ROOT = Path(__file__).resolve().parents[1]


class MetadataEpisodes:
    def __init__(self, lengths):
        end = torch.tensor(lengths).cumsum(0)
        self.episode_data_index = {'from': torch.cat([torch.tensor([0]), end[:-1]]), 'to': end}
        self.frames = int(end[-1])

    def __len__(self):
        return self.frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/RMBench/data/data')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/lerobot/rmbench_all_12tasks_demo_clean_50')
    parser.add_argument('--subtask-segments', choices=('v72_semantic', 'raw_annotation'), default='v72_semantic')
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = json.loads((ROOT / 'data/source_map_template.json').read_text())
    source['raw_root'] = str(args.raw_root.resolve())
    temporary_map = ROOT / 'reports/metadata_audit_source_map.json'
    temporary_map.write_text(json.dumps(source, indent=2)+'\n')
    tasks = json.loads((ROOT / 'data/task_manifest.json').read_text())['tasks']
    rows = {row['episode_index']: row for row in source['entries']}
    reports = {}
    for task, info in tasks.items():
        dataset = GroundTruthSubtaskRobotVideoDataset.__new__(GroundTruthSubtaskRobotVideoDataset)
        selected = [rows[ep] for ep in info['episode_indices']]
        dataset.episode_indices = tuple(info['episode_indices'])
        dataset.lerobot_dataset = MetadataEpisodes([row['length'] for row in selected])
        dataset.subtask_source_map = temporary_map
        dataset.training_shuffle_unit = 'trajectory'
        dataset.trajectory_anchor_stride = 1
        dataset._gt_stage_index = {}
        dataset._subtask_segment_end = {}
        dataset._subtask_segment_index = {}
        dataset._gt_texts_by_episode = {}
        dataset._gt_raw_ranges_by_episode = {}
        dataset.gt_subtask_segment_mode = args.subtask_segments
        dataset.gt_subtask_event_instance_tasks = ('press_button',)
        dataset._load_gt_subtask_trajectories()
        gt_segments = held = targets_count = 0
        for row in selected:
            ep, raw_id = row['episode_index'], row['source_episode_id']
            raw = args.raw_root / task / f'demo_clean/data/episode{raw_id}.hdf5'
            with h5py.File(raw) as handle:
                qpos = np.asarray(handle['joint_action/vector'], dtype=np.float32)
            assert qpos.shape == (row['raw_length'], 14)
            assert hashlib.sha256(qpos[:row['length']].tobytes()).hexdigest() == row['state_sha256']
            parquet = args.dataset_root / f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
            table = pq.read_table(parquet, columns=['observation.state', 'action'])
            states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
            actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            assert np.array_equal(states, qpos[:-1])
            assert np.array_equal(actions, qpos[1:])
            stage, end = dataset._gt_stage_index[ep], dataset._subtask_segment_end[ep]
            offsets = np.arange(len(stage))[:, None] + np.arange(32)
            target = np.minimum(offsets, end[:, None])
            assert np.all(stage[target] == stage[:, None])
            gt_segments += len(np.unique(stage))
            held += int((offsets > end[:, None]).sum())
            targets_count += int(target.size)
        sampler = ResumableEpochSampler(dataset, seed=42, batch_size=8, num_processes=8)
        first = list(sampler)
        real = [index for index in first if index >= 0]
        assert len(real) == len(set(real)) == info['frames']
        assert sorted(real) == list(range(info['frames']))
        assert len(first) % 64 == 0
        assert first.count(-1) == (-info['frames']) % 64
        sampler.set_resume_batch_offset(2)
        assert list(sampler) == first[128:]
        sampler.clear_resume_batch_offset()
        sampler.set_epoch(1)
        second = list(sampler)
        assert second != first and sorted(index for index in second if index >= 0) == sorted(real)
        reports[task] = dict(episodes=50, frames=info['frames'], raw_qpos_and_next_qpos_actions_exact=True,
                            gt_segments=gt_segments, action_targets=targets_count, cross_subtask_action_targets=0,
                            terminal_hold_action_targets=held, epoch_real_anchors=len(real),
                            ddp_zero_loss_padding_slots=first.count(-1), resume_order_exact=True,
                            sampling_contract=dataset.sampling_contract)
        print('METADATA_AUDIT_OK '+task+' '+json.dumps(reports[task]), flush=True)
        (ROOT / 'reports/gt_metadata_audit_partial.json').write_text(json.dumps(reports, indent=2)+'\n')
    report = dict(passed=True, tasks=12, episodes=600, frames=sum(info['frames'] for info in tasks.values()),
                  input_raw_root=str(args.raw_root), input_dataset_root=str(args.dataset_root),
                  gt_subtask_segment_mode=args.subtask_segments, tasks_report=reports)
    (ROOT / 'reports/gt_metadata_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('ALL_METADATA_AUDITS_PASSED', flush=True)


if __name__ == '__main__':
    main()
