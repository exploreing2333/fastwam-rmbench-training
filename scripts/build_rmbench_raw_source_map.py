#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map FastWAM RMBench episodes to original Progress Model inputs."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--description-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    return parser.parse_args()


def state_key(value: np.ndarray):
    value = np.asarray(value, dtype=np.float32)
    return len(value), hashlib.sha256(value.tobytes()).hexdigest()


def task_name_map(description_root: Path):
    result = {}
    for path in description_root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = list(payload.get("seen") or [])
        candidates.extend(payload.get("unseen") or [])
        if payload.get("full_description"):
            candidates.append(payload["full_description"])
        for text in candidates:
            result[str(text)] = path.stem
    return result


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    processed_root = args.processed_root.resolve()
    raw_root = args.raw_root.resolve()
    descriptions = task_name_map(args.description_root.resolve())
    info = json.loads((dataset_root / "meta/info.json").read_text())
    records = [
        json.loads(line)
        for line in (dataset_root / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    processed_indices = {}
    for task_text in {str(record["tasks"][0]) for record in records}:
        if task_text not in descriptions:
            raise ValueError(f"No raw RMBench task matches {task_text!r}")
        task_name = descriptions[task_text]
        index = defaultdict(list)
        task_root = processed_root / f"{task_name}-demo_clean-50"
        for directory in task_root.glob("episode_*"):
            source_id = int(directory.name.rsplit("_", 1)[1])
            path = directory / f"episode_{source_id}.hdf5"
            with h5py.File(path, "r") as handle:
                states = np.asarray(handle["observations/qpos"], dtype=np.float32)
            index[state_key(states)].append((source_id, path))
        processed_indices[task_name] = index

    entries = []
    for position, record in enumerate(records):
        episode_index = int(record["episode_index"])
        if episode_index != position:
            raise ValueError("LeRobot episode indices must be contiguous and ordered")
        task_text = str(record["tasks"][0])
        task_name = descriptions[task_text]
        parquet = dataset_root / info["data_path"].format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
        )
        states = np.asarray(
            pq.read_table(parquet, columns=["observation.state"])[
                "observation.state"
            ].to_pylist(),
            dtype=np.float32,
        )
        matches = processed_indices[task_name].get(state_key(states), [])
        if len(matches) != 1:
            raise ValueError(
                f"Episode {episode_index} has {len(matches)} exact raw-source matches"
            )
        source_id, processed_path = matches[0]
        raw_path = (
            raw_root
            / task_name
            / "demo_clean/data"
            / f"episode{source_id}.hdf5"
        )
        with h5py.File(raw_path, "r") as handle:
            raw_states = np.asarray(
                handle["joint_action/vector"][: len(states)], dtype=np.float32
            )
            raw_frames = handle["observation/head_camera/rgb"]
            if len(raw_frames) < len(states):
                raise ValueError(
                    f"Raw episode is shorter than LeRobot episode {episode_index}"
                )
            image = cv2.imdecode(
                np.frombuffer(raw_frames[0], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            raw_length = len(raw_frames)
        if not np.array_equal(raw_states, states):
            raise ValueError(
                f"Raw state prefix differs for LeRobot episode {episode_index}"
            )
        if image is None:
            raise ValueError(f"Cannot decode first raw frame for episode {episode_index}")
        entries.append(
            {
                "episode_index": episode_index,
                "task": task_text,
                "source_task_name": task_name,
                "source_episode_id": source_id,
                "source_hdf5": str(raw_path),
                "processed_hdf5": str(processed_path),
                "length": len(states),
                "raw_length": raw_length,
                "state_sha256": state_key(states)[1],
                "raw_image_shape": list(image.shape),
            }
        )

    manifest = {
        "format": "rmbench_lerobot_to_raw_progress_v1",
        "dataset_root": str(dataset_root),
        "processed_root": str(processed_root),
        "raw_root": str(raw_root),
        "source_camera": "head_camera",
        "source_state": "joint_action/vector",
        "source_fps": float(args.source_fps),
        "total_episodes": len(entries),
        "total_frames": sum(item["length"] for item in entries),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
