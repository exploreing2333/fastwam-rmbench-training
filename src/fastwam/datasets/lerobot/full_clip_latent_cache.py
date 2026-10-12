"""Exact, independently encoded FastWAM training clips (not episode slices)."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FORMAT = "fastwam_independent_full_clip_bf16_v1"


def clip_contract(dataset):
    """Bind the cache to RGB sources, preprocessing and deployed V72 boundaries."""
    if not (dataset.subtask_trajectory_mode and dataset.subtask_terminal_hold
            and dataset.subtask_segment_source == "progress_condition_index"
            and dataset.history_num_frames == 1 and not dataset.skip_padding_as_possible):
        raise ValueError("Full-clip cache requires current-only V72 terminal-hold training.")
    processor = dataset.lerobot_dataset.processor
    transforms = processor.train_transforms
    if not all(type(t).__name__ in {"ToTensor", "Resize"} for t in transforms):
        raise ValueError("Cannot cache stochastic or unrecognized RGB preprocessing.")
    repo = Path(__file__).resolve().parents[4]
    code = ["src/fastwam/datasets/lerobot/progress_subtask_video_dataset.py",
            "src/fastwam_qwen/datasets/robot_video_dataset.py",
            "src/fastwam/datasets/lerobot/processors/fastwam_processor.py",
            "src/fastwam/models/wan22/wan_video_vae.py"]
    sources = []
    for raw in dataset.lerobot_dataset.dataset_dirs:
        root = Path(raw).resolve()
        info = json.loads((root / "meta/info.json").read_text())
        for ep in dataset.episode_indices:
            path = root / "data" / f"chunk-{ep // info['chunks_size']:03d}" / f"episode_{ep:06d}.parquet"
            stat = path.stat()
            sources.append([str(path), stat.st_size, stat.st_mtime_ns])
    return dict(format=FORMAT, sources=sources, episodes=list(dataset.episode_indices),
                video_size=list(dataset.video_size), video_indices=list(dataset.video_sample_indices),
                current_obs_index=dataset.current_obs_index,
                concat_multi_camera=dataset.concat_multi_camera,
                image_meta=dataset.lerobot_dataset.shape_meta["images"],
                transforms=repr(transforms),
                segment_ends={str(ep): hashlib.sha256(np.asarray(ends, dtype=np.int64).tobytes()).hexdigest()
                              for ep, ends in dataset._subtask_segment_end.items()},
                code={name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in code})


class FullClipLatentStore:
    def __init__(self, root, *, expected_contract, max_cached_episodes=8):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT or not self.manifest.get("complete"):
            raise ValueError("Full-clip cache is incomplete or has an unsupported format.")
        if self.manifest.get("contract") != expected_contract:
            raise ValueError("Full-clip cache RGB/preprocessing/V72 contract mismatch.")
        if "vae_path" in self.manifest:
            if self.manifest.get("autocast") != "disabled_deepspeed_bf16":
                raise ValueError("Full-clip precision regime must match DeepSpeed native BF16.")
            stat = Path(self.manifest["vae_path"]).stat()
            if (stat.st_size != self.manifest["vae_size"]
                    or stat.st_mtime_ns != self.manifest["vae_mtime_ns"]):
                raise ValueError("Frozen VAE checkpoint changed after latent precomputation.")
        self.max_cached_episodes = int(max_cached_episodes)
        self.payloads = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["payloads"] = OrderedDict()
        return state

    def load(self, episode, frame):
        payload = self.payloads.pop(episode, None)
        if payload is None:
            payload = torch.load(self.root / "episodes" / f"episode_{episode:06d}.pt",
                                 map_location="cpu", weights_only=True, mmap=True)
            if payload.get("format") != FORMAT or payload.get("episode_index") != episode:
                raise ValueError("Invalid full-clip episode payload.")
            frames, z = payload["frame_indices"], payload["latents"]
            if (z.dtype != torch.bfloat16 or z.ndim != 5 or len(z) != len(frames)
                    or frames.ndim != 1 or not bool((frames[1:] > frames[:-1]).all())):
                raise ValueError("Malformed independently encoded latent clips.")
            payload["lookup"] = {int(t): i for i, t in enumerate(frames.tolist())}
        self.payloads[episode] = payload
        while len(self.payloads) > self.max_cached_episodes:
            self.payloads.popitem(last=False)
        if frame not in payload["lookup"]:
            raise KeyError(f"Full-clip cache missing episode={episode}, frame={frame}.")
        return payload["latents"][payload["lookup"][frame]].clone()
