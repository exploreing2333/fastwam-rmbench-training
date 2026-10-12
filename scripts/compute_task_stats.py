"""Compute reference normalization from scalar columns, avoiding image decode."""
import argparse
import json
import os
from pathlib import Path

from hydra.utils import instantiate
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

ROOT = Path(__file__).resolve().parents[1]


class ScalarEpisodes:
    def __init__(self, rows):
        self.rows = rows
        self.num_episodes = len(rows)
        self.num_frames = sum(len(row['action']) for row in rows)

    def get_episode_data(self, index):
        return self.rows[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/lerobot/rmbench_all_12tasks_demo_clean_50')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    tasks = json.loads((ROOT / 'data/task_manifest.json').read_text())['tasks']
    output = ROOT / 'data/task_norm_stats'
    output.mkdir(exist_ok=True)
    for task, info in tasks.items():
        cfg = OmegaConf.load(ROOT / f'configs/portable_{task}.yaml')
        provider = BaseLerobotDataset.__new__(BaseLerobotDataset)
        provider.action_size = 32
        provider.state_meta = [dict(meta, lerobot_key='observation.state') for meta in
                               OmegaConf.to_container(cfg.data.train.shape_meta.state, resolve=True)]
        provider.action_meta = [dict(meta, lerobot_key='action') for meta in
                                OmegaConf.to_container(cfg.data.train.shape_meta.action, resolve=True)]
        rows = []
        for episode in info['episode_indices']:
            path = cli.dataset_root / f'data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet'
            table = pq.read_table(path, columns=['observation.state', 'action'])
            rows.append({key: torch.tensor(table[key].to_pylist(), dtype=torch.float32)
                         for key in ('observation.state', 'action')})
        provider.multi_dataset = ScalarEpisodes(rows)
        processor = instantiate(cfg.data.train.processor)
        # Execute the actual reference method, with the same 32-control windows,
        # transforms, per-episode weighting, quantiles, and replication rules.
        stats = provider.get_dataset_stats(processor, action_horizon=32)
        assert stats['num_episodes'] == 50 and stats['num_transition'] == info['frames']
        save_dataset_stats_to_json(stats, str(output / f'{task}.json'))
        print(f'STATS_OK task={task} frames={info["frames"]}', flush=True)
    expected = json.loads((ROOT / 'provenance/reference_cover_dataset_stats.json').read_text())
    actual = json.loads((output / 'cover_blocks.json').read_text())
    maximum_difference = 0.0
    for namespace in ('action', 'state'):
        for field, values in expected[namespace]['default'].items():
            a, b = torch.tensor(values), torch.tensor(actual[namespace]['default'][field])
            assert a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
            maximum_difference = max(maximum_difference, float((a-b).abs().max()))
    report = dict(passed=True, tasks=len(tasks), native_reference_method_used=True,
                  reference_cover_stats_match=True, maximum_cover_absolute_difference=maximum_difference,
                  default_training_still_recomputes_stats=True,
                  optional_flag='--cached-stats')
    (ROOT / 'reports/normalization_equivalence.json').write_text(json.dumps(report, indent=2)+'\n')
    print('NORMALIZATION_EQUIVALENCE_OK '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
