"""Exercise actual packaged GT dataset samples across subtask boundaries."""
import argparse
import json
import os
from pathlib import Path
import time

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from fastwam.runtime import _configure_runtime_cache
from fastwam.utils import misc

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-copy', action='store_true')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/RMBench/data/data')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/lerobot/rmbench_all_12tasks_demo_clean_50')
    parser.add_argument('--text-cache', type=Path, default=ROOT / 'data/gt_subtask_text_embeds')
    parser.add_argument('--episodes-per-task', type=int, default=2)
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        parser.error('episodes-per-task must be positive')
    if args.wait_for_copy:
        while True:
            state = json.loads((ROOT / 'reports/build_status.json').read_text())
            if state['phase'] == 'copy_complete':
                break
            if (ROOT / 'reports/build_error.json').exists():
                raise RuntimeError((ROOT / 'reports/build_error.json').read_text())
            time.sleep(30)
    torch.set_num_threads(1)
    tasks = json.loads((ROOT / 'data/task_manifest.json').read_text())['tasks']
    source = json.loads((ROOT / 'data/source_map_template.json').read_text())
    source['raw_root'] = str(args.raw_root.resolve())
    source_rows = {row['episode_index']: row for row in source['entries']}
    source_map = ROOT / 'reports/rgb_audit_source_map.json'
    source_map.write_text(json.dumps(source, indent=2) + '\n')
    reports = {}
    for task, info in tasks.items():
        cfg = OmegaConf.load(ROOT / f'configs/portable_{task}.yaml')
        cfg.data.train.episode_indices = info['episode_indices'][:args.episodes_per_task]
        cfg.data.train.dataset_dirs = [str(args.dataset_root.resolve())]
        cfg.data.train.subtask_source_map = str(source_map)
        cfg.data.train.gt_subtask_text_embedding_cache_dir = str(args.text_cache.resolve())
        cfg.data.train.pretrained_norm_stats = str(ROOT / f'data/task_norm_stats/{task}.json')
        _configure_runtime_cache(cfg)
        work = ROOT / 'reports/rgb_samples' / task
        work.mkdir(parents=True, exist_ok=True)
        misc.register_work_dir(work)
        dataset = instantiate(cfg.data.train)
        assert dataset.history_num_frames == 1 and dataset.current_obs_index == 0
        assert dataset.progress_feature_store is None
        assert dataset.action_horizon == 32
        assert dataset.video_sample_indices == list(range(0,33,4))
        normalizer = dataset.lerobot_dataset.processor.normalizer.normalizers['action']['default']
        offsets = dataset.lerobot_dataset.episode_data_index['from'].tolist()
        checked = 0
        for position, episode in enumerate(dataset.episode_indices):
            stage = dataset._gt_stage_index[episode]
            ends = dataset._subtask_segment_end[episode]
            path = args.dataset_root / f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
            table = pq.read_table(path, columns=['action'])
            normalized = normalizer.forward(torch.tensor(table['action'].to_pylist(), dtype=torch.float32))
            starts = np.r_[0, np.flatnonzero(stage[1:] != stage[:-1])+1]
            stops = np.r_[starts[1:],len(stage)]
            row = source_rows[episode]
            raw = json.loads((args.raw_root / task / 'demo_clean/language_annotation.json').read_text())[f"episode_{row['source_episode_id']}"]
            raw_starts = np.cumsum([0] + [int(r[1]) for r in raw[:-1]])
            test_frames = set()
            for start, stop in zip(starts.tolist(), stops.tolist()):
                test_frames.update({start, max(start, stop - 32), stop - 1})
            # Removed raw boundaries must continue the same instruction, action
            # trajectory and RGB future until the true semantic stage ends.
            for start in raw_starts:
                test_frames.update({max(0, int(start) - 8), int(start)})
            for frame in sorted(f for f in test_frames if 0 <= f < len(stage)):
                sample = dataset[int(offsets[position])+frame]
                target = np.minimum(np.arange(32)+frame,ends[frame])
                torch.testing.assert_close(sample['action'],normalized[torch.from_numpy(target)],rtol=0,atol=0)
                assert sample['current_subtask'] == dataset._gt_texts_by_episode[episode][int(stage[frame])]
                assert tuple(sample['video'].shape)==(3,9,384,320)
                assert tuple(sample['action'].shape)==(32,14)
                assert tuple(sample['proprio'].shape)==(1,14)
                assert tuple(sample['context'].shape)==(128,4096)
                assert sample['context_mask'].all()
                assert torch.isfinite(sample['video']).all() and torch.isfinite(sample['context']).all()
                assert not any(k.startswith(('event_','progress_','wam_progress','qwen_images')) for k in sample)
                last_valid = 0
                for i,video_offset in enumerate(range(0,33,4)):
                    if frame+video_offset <= ends[frame]:
                        last_valid=i
                    else:
                        torch.testing.assert_close(sample['video'][:,i],sample['video'][:,last_valid],rtol=0,atol=0)
                        assert not bool(sample['image_is_pad'][i])
                valid_steps=min(32,int(ends[frame]-frame+1))
                assert not sample['action_is_pad'][valid_steps:].any()
                checked+=1
        reports[task]=dict(passed=True,episodes_sampled=len(dataset.episode_indices),actual_getitem_boundary_checks=checked,
                           shape_contract='video3x9x384x320_action32x14_proprio1x14_context128x4096',
                           same_subtask_action_hold_exact=True,same_subtask_video_hold_exact=True,
                           current_only=True,gt_subtask_only=True,
                           gt_subtask_segment_mode=dataset.gt_subtask_segment_mode,
                           removed_raw_boundary_checks=True)
        print('RGB_SAMPLE_AUDIT_OK '+task+' '+json.dumps(reports[task]),flush=True)
        (ROOT/'reports/gt_rgb_samples_partial.json').write_text(json.dumps(reports,indent=2)+'\n')
        del dataset
    result=dict(passed=True,tasks=12,episodes_sampled=sum(row['episodes_sampled'] for row in reports.values()),total_getitem_checks=sum(row['actual_getitem_boundary_checks'] for row in reports.values()),
                input_root=str(ROOT),tasks_report=reports)
    (ROOT/'reports/gt_rgb_sample_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL_RGB_SAMPLE_AUDITS_PASSED',flush=True)


if __name__=='__main__':
    main()
