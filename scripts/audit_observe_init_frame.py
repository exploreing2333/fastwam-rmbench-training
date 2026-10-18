"""Audit original Observe/Pickup labels, unchanged subtask sampling, and episode-zero memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

from audit_observe_target_labels import metadata_dataset
from fastwam.datasets.lerobot.event_anchor_cache import EventAnchorLatentStore
from fastwam.datasets.lerobot.gt_subtask_video_dataset import GroundTruthSubtaskRobotVideoDataset
from fastwam.runtime import _configure_runtime_cache
from fastwam.utils import misc
from fastwam.utils.samplers import ResumableEpochSampler
from precompute_init_frame_latents import validate_payload

ROOT = Path(__file__).resolve().parents[1]
TASK = "observe_and_pickup"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/lerobot/rmbench_all_12tasks_demo_clean_50")
    parser.add_argument("--anchor-cache", type=Path)
    parser.add_argument("--rgb-samples", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/observe_init_frame_audit.json")
    args = parser.parse_args()
    os.environ.setdefault("FASTWAM_BUNDLE_ROOT", str(ROOT))
    os.environ.setdefault("FASTWAM_RUNTIME_ROOT", str(ROOT / ".runtime"))
    torch.set_num_threads(1)
    cfg = OmegaConf.load(ROOT / f"configs/portable_{TASK}.yaml")
    backup = Path(cfg.data.train.gt_subtask_annotation_files[TASK])
    original = json.loads(backup.read_text())
    canonical = backup.with_name("language_annotation.json")
    target_labels = json.loads(canonical.read_text())
    expected_texts = {"Wait for wall appear.", "Pick up the target object."}
    assert {row[0] for rows in original.values() for row in rows} == expected_texts
    assert len(original) == 50 and original.keys() == target_labels.keys()
    for episode in original:
        assert [row[1] for row in original[episode]] == [row[1] for row in target_labels[episode]]
    baseline_cfg = OmegaConf.load(ROOT / "provenance/observe_initkv_config_before.yaml")
    # Only input plumbing, annotation selection, and the descriptive run name may change.
    before = OmegaConf.to_container(baseline_cfg, resolve=True)
    after = OmegaConf.to_container(cfg, resolve=True)
    data_changes = {"_target_", "gt_subtask_annotation_files", "event_anchor_latent_cache_root",
                    "event_anchor_max_frames", "event_anchor_batch_grouping",
                    "event_anchor_cache_max_open_episodes", "init_frame_anchor_only"}
    for key in data_changes:
        before["data"]["train"].pop(key, None)
        after["data"]["train"].pop(key, None)
    for key in ("event_memory_video_conditioning", "init_frame_context_injection", "init_frame_kv_config"):
        before["model"].pop(key, None)
        after["model"].pop(key, None)
    before["wandb"].pop("name")
    after["wandb"].pop("name")
    assert before == after, "An unrelated training setting changed."
    assert cfg.batch_size == 8 and cfg.gradient_accumulation_steps == 1 and cfg.max_steps == 200000
    source = json.loads((ROOT / "data/source_map_template.json").read_text())
    source["raw_root"] = str(ROOT / "data/RMBench/data/data")
    rows = {int(row["episode_index"]): row for row in source["entries"]}
    episodes = list(cfg.data.train.episode_indices)
    cache = Path(args.anchor_cache or cfg.data.train.event_anchor_latent_cache_root).resolve()
    anchors = EventAnchorLatentStore(cache, max_anchors=1)
    assert anchors.manifest["anchor_policy"] == "episode_frame_zero_only"
    assert Path(anchors.manifest["dataset_root"]).resolve() == args.dataset_root.resolve()
    actual_checks = 0
    with tempfile.TemporaryDirectory(prefix="observe-init-audit-", dir="/tmp") as temporary:
        source_map = Path(temporary) / "source_map.json"
        source_map.write_text(json.dumps(source))
        old = metadata_dataset(episodes, rows, source_map, Path(cfg.data.train.gt_subtask_text_embedding_cache_dir))
        new = metadata_dataset(episodes, rows, source_map, Path(cfg.data.train.gt_subtask_text_embedding_cache_dir))
        new.gt_subtask_annotation_files = {TASK: backup}
        new._load_gt_subtask_trajectories()
        frames = 0
        for episode in episodes:
            np.testing.assert_array_equal(old._gt_stage_index[episode], new._gt_stage_index[episode])
            np.testing.assert_array_equal(old._subtask_segment_end[episode], new._subtask_segment_end[episode])
            assert set(new._gt_texts_by_episode[episode]) == expected_texts
            frames += len(new._gt_stage_index[episode])
            validate_payload(anchors.episode_path(episode), episode)
            first = anchors.load(episode, 0)["event_anchor_latents"]
            for frame in (0, int(np.flatnonzero(new._gt_stage_index[episode] == 1)[0]), len(new._gt_stage_index[episode]) - 1):
                item = anchors.load(episode, frame)
                assert item["event_anchor_frame_indices"].tolist() == [0]
                assert item["event_anchor_valid_mask"].tolist() == [True]
                torch.testing.assert_close(item["event_anchor_latents"], first, atol=0, rtol=0)
        for text in expected_texts:
            context = new._load_gt_context(text)
            assert tuple(context.shape) == (128, 4096) and torch.isfinite(context).all()
        for epoch in range(3):
            old_sampler, new_sampler = [ResumableEpochSampler(ds, seed=42, batch_size=8, num_processes=8) for ds in (old, new)]
            old_sampler.set_epoch(epoch)
            new_sampler.set_epoch(epoch)
            assert list(old_sampler) == list(new_sampler)
            new_sampler.set_resume_batch_offset(2)
            assert list(new_sampler) == list(old_sampler)[128:]
        if args.rgb_samples:
            cfg.data.train.episode_indices = [episodes[0], episodes[-1]]
            cfg.data.train.dataset_dirs = [str(args.dataset_root.resolve())]
            cfg.data.train.subtask_source_map = str(source_map)
            cfg.data.train.event_anchor_latent_cache_root = str(cache)
            cfg.data.train.pretrained_norm_stats = str(ROOT / f"data/task_norm_stats/{TASK}.json")
            _configure_runtime_cache(cfg)
            misc.register_work_dir(Path(temporary) / "rgb")
            dataset = instantiate(cfg.data.train)
            for i, episode in enumerate(cfg.data.train.episode_indices):
                stage = dataset._gt_stage_index[episode]
                transition = int(np.flatnonzero(stage == 1)[0])
                start = int(dataset.lerobot_dataset.episode_data_index["from"][i])
                for frame in sorted({0, transition - 1, transition, min(transition + 1, len(stage) - 1), len(stage) - 1}):
                    item = dataset._get(start + frame)
                    parent = GroundTruthSubtaskRobotVideoDataset._get(dataset, start + frame)
                    for key, value in parent.items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(item[key], value, rtol=0, atol=0)
                        elif isinstance(value, np.ndarray):
                            np.testing.assert_array_equal(item[key], value)
                        else:
                            assert item[key] == value, key
                    assert item["current_subtask"] == new._gt_texts_by_episode[episode][int(stage[frame])]
                    assert item["event_anchor_frame_indices"].tolist() == [0]
                    assert int(item["event_anchor_group_id"]) == episode * 1000000 + int(stage[frame])
                    assert tuple(item["video"].shape) == (3, 9, 384, 320)
                    assert tuple(item["action"].shape) == (32, 14)
                    torch.testing.assert_close(item["event_anchor_latents"], anchors.load(episode, 0)["event_anchor_latents"], atol=0, rtol=0)
                    actual_checks += 1
        result = dict(passed=True, task=TASK, episodes=len(episodes), frames=frames,
                      annotation_file=str(backup), annotation_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                      canonical_annotation_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
                      active_subtasks=sorted(expected_texts), same_stage_boundaries_all_frames=True,
                      same_sampler_order_three_epochs_and_resume=True, unrelated_training_settings_unchanged=True,
                      init_frame=0, full_vae_shape=[1, 48, 24, 20], full_patch_tokens=120,
                      cache_episodes_verified=len(episodes), actual_rgb_action_proprio_context_checks=actual_checks,
                      dataset_root=str(args.dataset_root.resolve()), cache_root=str(cache), gpu_jobs_started=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
