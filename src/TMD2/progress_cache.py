from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


CACHE_FORMAT = "fastwam_progress_model_cache_v2"


def bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return value.view(torch.uint16).numpy()


def uint16_to_bf16(value: np.ndarray) -> torch.Tensor:
    # ``h5py_dataset[...]`` already returns an owned, writable, contiguous
    # ndarray.  Avoid copying that entire layer cache a second time on every
    # training sample; retain the defensive copy only for read-only or
    # non-contiguous callers.  Reinterpreting the same uint16 bits preserves
    # the archived bfloat16 values exactly.
    array = np.asarray(value, dtype=np.uint16)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array).view(torch.bfloat16)


class ProgressFeatureStore:
    """Worker-local random access to episode-sharded Progress Model features."""

    def __init__(
        self,
        root: str | Path,
        max_open_episodes: int = 8,
        memory_mode: str = "all",
        subtask_mode: str = "final",
    ):
        self.root = Path(root).expanduser().resolve()
        self.max_open_episodes = max(1, int(max_open_episodes))
        self.memory_mode = str(memory_mode).lower()
        self.subtask_mode = str(subtask_mode).lower()
        if self.memory_mode not in {"all", "state_only"}:
            raise ValueError(
                "Progress cache memory_mode must be `all` or `state_only`, "
                f"got {memory_mode!r}."
            )
        if self.subtask_mode not in {"final", "layerwise"}:
            raise ValueError(
                "Progress cache subtask_mode must be `final` or `layerwise`, "
                f"got {subtask_mode!r}."
            )
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Progress cache manifest does not exist: {manifest_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError(
                f"Unsupported Progress cache format: {self.manifest.get('format')!r}"
            )
        if not bool(self.manifest.get("complete", False)):
            raise RuntimeError(f"Progress cache is incomplete: {manifest_path}")
        expected_memory = [
            "current_state",
            "instruction",
        ]
        if self.manifest.get("memory_tokens") != expected_memory:
            raise RuntimeError(
                "Progress cache does not satisfy the TMD3 full-attention "
                f"memory contract: expected={expected_memory}, "
                f"actual={self.manifest.get('memory_tokens')}"
            )
        expected_cross_attention = ["generated_subtask"]
        if self.manifest.get("cross_attention_tokens") != expected_cross_attention:
            raise RuntimeError(
                "Progress cache does not satisfy the TMD3 cross-attention "
                f"contract: expected={expected_cross_attention}, "
                f"actual={self.manifest.get('cross_attention_tokens')}"
            )
        excluded = self.manifest.get("excluded_cross_attention_tokens", [])
        if not {"current_progress", "plan"}.issubset(excluded):
            raise RuntimeError(
                "Progress cache must explicitly exclude current-progress and "
                "plan tokens from TMD3 cross-attention."
            )
        if self.subtask_mode == "layerwise":
            if self.manifest.get("subtask_hidden_layout") != "layer_input":
                raise RuntimeError(
                    "Layerwise Progress subtask conditioning requires "
                    "subtask_hidden_layout=layer_input."
                )
            if self.manifest.get("subtask_hidden_dataset") != "subtask_layer_hidden":
                raise RuntimeError(
                    "Layerwise Progress subtask conditioning requires the "
                    "subtask_layer_hidden dataset."
                )
        self.num_layers = int(self.manifest["num_layers"])
        self.hidden_size = int(self.manifest["hidden_size"])
        self.state_memory_frames = int(
            self.manifest.get("state_memory_frames", 1)
        )
        if not 1 <= self.state_memory_frames <= 8:
            raise ValueError(
                "Progress cache state_memory_frames must be in [1, 8]"
            )
        self.total_episodes = int(self.manifest["total_episodes"])
        self.total_frames = int(self.manifest["total_frames"])
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _episode_path(self, episode_index: int) -> Path:
        return self.root / "episodes" / f"episode_{episode_index:06d}.h5"

    def _open(self, episode_index: int) -> h5py.File:
        episode_index = int(episode_index)
        handle = self._handles.pop(episode_index, None)
        if handle is None:
            path = self._episode_path(episode_index)
            if not path.is_file():
                raise FileNotFoundError(f"Missing Progress cache episode: {path}")
            handle = h5py.File(path, "r", libver="latest", swmr=True)
            if handle.attrs.get("format") != CACHE_FORMAT:
                handle.close()
                raise ValueError(f"Invalid Progress cache episode format: {path}")
        self._handles[episode_index] = handle
        while len(self._handles) > self.max_open_episodes:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        return handle

    def load(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        handle = self._open(episode_index)
        frame_index = int(frame_index)
        episode_length = int(handle.attrs["length"])
        if not 0 <= frame_index < episode_length:
            raise IndexError(
                f"Frame {frame_index} is outside cached episode {episode_index} "
                f"with length {episode_length}."
            )
        condition_index = int(handle["condition_index"][frame_index])
        if condition_index < 0:
            raise RuntimeError(
                f"Episode {episode_index} frame {frame_index} has no valid subtask condition."
            )
        condition = handle["conditions"][f"{condition_index:06d}"]
        state_layers = uint16_to_bf16(
            handle["prompt_layer_hidden"][frame_index]
        )
        expected_state_shape = (
            self.num_layers,
            self.state_memory_frames,
            self.hidden_size,
        )
        if tuple(state_layers.shape) != expected_state_shape:
            raise ValueError(
                "Invalid state-history cache shape: "
                f"{tuple(state_layers.shape)}"
            )
        if self.memory_mode == "state_only":
            memory_layers = state_layers
        else:
            instruction_layers = uint16_to_bf16(
                handle["instruction_layer_hidden"][...]
            )
            if (
                instruction_layers.ndim != 3
                or instruction_layers.shape[0] != self.num_layers
                or instruction_layers.shape[-1] != self.hidden_size
            ):
                raise ValueError(
                    "Invalid instruction cache shape: "
                    f"{tuple(instruction_layers.shape)}"
                )
            memory_layers = torch.cat(
                [state_layers, instruction_layers],
                dim=1,
            )
        result = {
            "progress_memory_layers": memory_layers,
            "progress_memory_mask": torch.ones(
                memory_layers.shape[1], dtype=torch.bool
            ),
            "progress_condition_index": condition_index,
            "progress_decode_source_frame": int(condition.attrs["source_frame"]),
            "progress_parse_valid": bool(condition.attrs["parse_valid"]),
        }
        if self.subtask_mode == "layerwise":
            if "subtask_layer_hidden" not in condition:
                raise ValueError(
                    "Layerwise generated-subtask cache is missing from condition."
                )
            subtask_layers = uint16_to_bf16(
                condition["subtask_layer_hidden"][...]
            )
            if (
                subtask_layers.ndim != 3
                or subtask_layers.shape[0] != self.num_layers
                or subtask_layers.shape[-1] != self.hidden_size
            ):
                raise ValueError(
                    "Invalid generated-subtask layer hidden shape: "
                    f"{tuple(subtask_layers.shape)}"
                )
            result["progress_subtask_layers"] = subtask_layers
            subtask_length = int(subtask_layers.shape[1])
        else:
            cross_subtask = uint16_to_bf16(condition["final_hidden"][...])
            if (
                cross_subtask.ndim != 2
                or cross_subtask.shape[-1] != self.hidden_size
            ):
                raise ValueError(
                    "Invalid generated-subtask final hidden shape: "
                    f"{tuple(cross_subtask.shape)}"
                )
            result["progress_subtask_tokens"] = cross_subtask
            subtask_length = int(cross_subtask.shape[0])
        result["progress_subtask_mask"] = torch.ones(
            subtask_length,
            dtype=torch.bool,
        )
        return result


def pad_progress_features(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not samples:
        return {}
    layers = samples[0]["progress_memory_layers"].shape[0]
    hidden = samples[0]["progress_memory_layers"].shape[-1]
    max_memory = max(item["progress_memory_layers"].shape[1] for item in samples)
    has_final_subtask = [
        "progress_subtask_tokens" in item for item in samples
    ]
    has_layer_subtask = [
        "progress_subtask_layers" in item for item in samples
    ]
    if all(has_final_subtask) == all(has_layer_subtask):
        raise ValueError(
            "Progress samples must provide exactly one subtask representation: "
            "final or layerwise."
        )
    if any(has_final_subtask) and not all(has_final_subtask):
        raise ValueError("Cannot mix final and layerwise subtask cache samples.")
    if any(has_layer_subtask) and not all(has_layer_subtask):
        raise ValueError("Cannot mix final and layerwise subtask cache samples.")
    layerwise_subtask = all(has_layer_subtask)
    max_subtask = max(
        (
            item["progress_subtask_layers"].shape[1]
            if layerwise_subtask
            else item["progress_subtask_tokens"].shape[0]
        )
        for item in samples
    )
    memory = torch.zeros(
        len(samples), layers, max_memory, hidden, dtype=torch.bfloat16
    )
    memory_mask = torch.zeros(len(samples), max_memory, dtype=torch.bool)
    subtask_mask = torch.zeros(len(samples), max_subtask, dtype=torch.bool)
    if layerwise_subtask:
        subtask_layers = torch.zeros(
            len(samples),
            layers,
            max_subtask,
            hidden,
            dtype=torch.bfloat16,
        )
        subtask = None
    else:
        subtask = torch.zeros(
            len(samples), max_subtask, hidden, dtype=torch.bfloat16
        )
        subtask_layers = None
    for batch_idx, item in enumerate(samples):
        memory_len = item["progress_memory_layers"].shape[1]
        subtask_len = int(item["progress_subtask_mask"].shape[0])
        memory[batch_idx, :, :memory_len] = item["progress_memory_layers"]
        memory_mask[batch_idx, :memory_len] = item["progress_memory_mask"]
        subtask_mask[batch_idx, :subtask_len] = item["progress_subtask_mask"]
        if subtask_layers is not None:
            subtask_layers[batch_idx, :, :subtask_len] = item[
                "progress_subtask_layers"
            ]
        else:
            subtask[batch_idx, :subtask_len] = item[
                "progress_subtask_tokens"
            ]
    result = {
        "progress_memory_layers": memory,
        "progress_memory_mask": memory_mask,
        "progress_subtask_mask": subtask_mask,
        "progress_condition_index": torch.tensor(
            [item["progress_condition_index"] for item in samples],
            dtype=torch.long,
        ),
        "progress_decode_source_frame": torch.tensor(
            [item["progress_decode_source_frame"] for item in samples],
            dtype=torch.long,
        ),
    }
    if subtask_layers is not None:
        result["progress_subtask_layers"] = subtask_layers
    else:
        result["progress_subtask_tokens"] = subtask
    return result
