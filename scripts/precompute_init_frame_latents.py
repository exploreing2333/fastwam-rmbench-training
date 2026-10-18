"""Prepare one full, frozen Wan VAE latent per episode, with the training RGB transform."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
import torchvision.transforms.functional as tvf

from fastwam.datasets.lerobot.event_anchor_cache import EVENT_ANCHOR_CACHE_FORMAT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = tuple(f"observation.images.{name}" for name in
                ("cam_high", "cam_left_wrist", "cam_right_wrist"))
CACHE_PREPROCESS = "JPEG decoded RGB, ToTensor, resize240x320, robotwin384x320, [-1,1]"


def load_training_mosaic(table):
    views = []
    for column in CAMERAS:
        with Image.open(io.BytesIO(table[column][0].as_py()["bytes"])) as image:
            rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
        views.append(tvf.resize(rgb.float().div(255), [240, 320], antialias=True))
    high = tvf.resize(views[0], [256, 320], antialias=True)
    wrists = [tvf.resize(view, [128, 160], antialias=True) for view in views[1:]]
    return torch.cat([high, torch.cat(wrists, dim=-1)], dim=-2).sub(0.5).div(0.5).contiguous()


def validate_payload(path, episode):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != EVENT_ANCHOR_CACHE_FORMAT
            or payload.get("episode_index") != episode
            or payload.get("preprocess") != CACHE_PREPROCESS
            or payload["event_indices"].tolist() != [0]
            or tuple(payload["event_latents"].shape) != (1, 48, 24, 20)
            or not bool(torch.isfinite(payload["event_latents"]).all())):
        raise ValueError(f"Invalid init-frame cache: {path}")


def prepare(dataset_root, output_root, episodes, device):
    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve()
    episodes = sorted(set(map(int, episodes)))
    if not episodes:
        raise ValueError("At least one episode is required.")
    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata is missing: {info_path}; restore bundle assets with bash setup.sh first.")
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if (previous.get("dataset_root") != str(dataset_root)
                or previous.get("anchor_policy") != "episode_frame_zero_only"
                or previous.get("preprocess") != CACHE_PREPROCESS
                or previous.get("episode_indices") != episodes):
            raise ValueError(f"Incompatible init cache; choose a separate output root: {output_root}")
    directory = output_root / "episodes"
    directory.mkdir(parents=True, exist_ok=True)
    missing = []
    for episode in episodes:
        path = directory / f"episode_{episode:06d}.pt"
        if path.is_file():
            validate_payload(path, episode)
        else:
            missing.append(episode)
    manifest = dict(format=EVENT_ANCHOR_CACHE_FORMAT, anchor_policy="episode_frame_zero_only",
                    dataset_root=str(dataset_root), model_id="Wan-AI/Wan2.2-TI2V-5B",
                    episode_indices=episodes, preprocess=CACHE_PREPROCESS, complete=False)

    def save_manifest():
        pending = manifest_path.with_suffix(".pending.json")
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(manifest_path)

    if missing:
        save_manifest()
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "models"))
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        _, _, config, _ = _resolve_configs("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B", True)
        config.download_if_necessary()
        if not config.path:
            raise FileNotFoundError("Wan VAE weights are missing; restore bundle assets with bash setup.sh first.")
        vae = _load_registered_model(config.path, "wan_video_vae", torch_dtype=torch.bfloat16,
                                     device=device).eval().requires_grad_(False)
        manifest["vae_path"] = str(config.path)
        chunk_size = int(json.loads(info_path.read_text())["chunks_size"])
        with torch.inference_mode():
            for episode in missing:
                parquet = dataset_root / "data" / f"chunk-{episode // chunk_size:03d}" / f"episode_{episode:06d}.parquet"
                table = pq.ParquetFile(parquet).read_row_group(0, columns=list(CAMERAS)).slice(0, 1)
                image = load_training_mosaic(table).to(device=device, dtype=torch.bfloat16).unsqueeze(1)
                latent = vae.encode([image], device=device, tiled=False,
                                    tile_size=(30, 52), tile_stride=(15, 26))
                if isinstance(latent, list):
                    latent = latent[0].unsqueeze(0)
                if tuple(latent.shape) != (1, 48, 1, 24, 20):
                    raise ValueError(f"Unexpected VAE shape: {tuple(latent.shape)}")
                path = directory / f"episode_{episode:06d}.pt"
                pending = path.with_suffix(".pending.pt")
                torch.save(dict(format=EVENT_ANCHOR_CACHE_FORMAT, episode_index=episode,
                                event_indices=torch.tensor([0]),
                                event_latents=latent[0, :, 0].detach().cpu().unsqueeze(0),
                                preprocess=CACHE_PREPROCESS), pending)
                pending.replace(path)
                validate_payload(path, episode)
                print(f"[init-frame] episode={episode} full VAE latent=(1,48,24,20)", flush=True)
    manifest["complete"] = True
    save_manifest()
    print(f"[init-frame] verified {len(episodes)} trajectory-frame-zero anchors: {output_root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    prepare(args.dataset_root, args.output_root, args.episode_indices, args.device)
