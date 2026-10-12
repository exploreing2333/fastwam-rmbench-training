from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ACTION_FORMAT = "rmbench_robotwin_endpose_tplus1_v3"
CURRENT_DELTA_ACTION_FORMAT = (
    "rmbench_robotwin_current_se3_delta_actual_eef_state_tplus1_v4"
)
SUPPORTED_ACTION_FORMATS = (ACTION_FORMAT, CURRENT_DELTA_ACTION_FORMAT)
SOURCE_MAP_FORMAT = "rmbench_lerobot_to_raw_progress_v1"
TARGET_OFFSET = 1
VALID_CHANNELS = tuple(range(7)) + (28,) + tuple(range(7, 14)) + (29,)
CURRENT_DELTA_VALID_CHANNELS = tuple(range(16))


def _quat_xyzw_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., :3] = -result[..., :3]
    return result


def _quat_xyzw_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dim=-1,
    )


def _normalize_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _canonicalize_quaternion_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _normalize_quaternion_xyzw(quaternion)
    return torch.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)


def _quat_xyzw_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D vectors without changing their leading dimensions."""
    quaternion = _normalize_quaternion_xyzw(quaternion)
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + w * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def rmbench_wxyz_to_robotwin_xyzw(absolute_action: torch.Tensor) -> torch.Tensor:
    """Convert RMBench's raw endpose quaternion fields to RobotWin xyzw."""
    if absolute_action.shape[-1] != 16:
        raise ValueError("RMBench endpose action must end in 16 channels.")
    result = absolute_action.clone()
    for pose_start in (0, 8):
        quaternion = absolute_action[..., pose_start + 3 : pose_start + 7]
        result[..., pose_start + 3 : pose_start + 7] = quaternion[..., [1, 2, 3, 0]]
    return result


def robotwin_xyzw_to_rmbench_wxyz(absolute_action: torch.Tensor) -> torch.Tensor:
    """Convert RobotWin xyzw quaternion fields back to RMBench raw wxyz."""
    if absolute_action.shape[-1] != 16:
        raise ValueError("RobotWin endpose action must end in 16 channels.")
    result = absolute_action.clone()
    for pose_start in (0, 8):
        quaternion = absolute_action[..., pose_start + 3 : pose_start + 7]
        result[..., pose_start + 3 : pose_start + 7] = quaternion[..., [3, 0, 1, 2]]
    return result


def make_relative_endpose(
    absolute_action: torch.Tensor,
    anchor_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply official LingBot's first-frame-relative RobotWin xyzw pose."""
    if absolute_action.ndim != 2 or absolute_action.shape[1] != 16:
        raise ValueError(
            f"Absolute endpose action must be [T,16], got {tuple(absolute_action.shape)}."
        )
    anchor_action = (
        absolute_action[:1]
        if anchor_action is None
        else anchor_action.reshape(1, 16).to(absolute_action)
    )
    result = absolute_action.clone()
    for pose_start in (0, 8):
        result[:, pose_start : pose_start + 3] -= anchor_action[
            :, pose_start : pose_start + 3
        ]
        quaternion = _normalize_quaternion_xyzw(
            absolute_action[:, pose_start + 3 : pose_start + 7]
        )
        anchor = _normalize_quaternion_xyzw(
            anchor_action[:, pose_start + 3 : pose_start + 7]
        )
        # This is scipy Rotation.from_quat(first).inv() * Rotation.from_quat(q),
        # including its xyzw convention and without an extra sign canonicalizer.
        relative = _quat_xyzw_multiply(
            _quat_xyzw_inverse(anchor).expand_as(quaternion),
            quaternion,
        )
        relative = _normalize_quaternion_xyzw(relative)
        result[:, pose_start + 3 : pose_start + 7] = relative
    return result


def restore_absolute_endpose(
    relative_action: torch.Tensor,
    anchor_action: torch.Tensor,
) -> torch.Tensor:
    if relative_action.ndim != 2 or relative_action.shape[1] != 16:
        raise ValueError("Relative endpose action must be [T,16].")
    anchor_action = anchor_action.reshape(1, 16).to(relative_action)
    result = relative_action.clone()
    for pose_start in (0, 8):
        result[:, pose_start : pose_start + 3] += anchor_action[
            :, pose_start : pose_start + 3
        ]
        anchor_quaternion = _normalize_quaternion_xyzw(
            anchor_action[:, pose_start + 3 : pose_start + 7]
        )
        relative_quaternion = _normalize_quaternion_xyzw(
            relative_action[:, pose_start + 3 : pose_start + 7]
        )
        result[:, pose_start + 3 : pose_start + 7] = _normalize_quaternion_xyzw(
            _quat_xyzw_multiply(
                anchor_quaternion.expand_as(relative_quaternion),
                relative_quaternion,
            )
        )
    return result


def make_current_se3_delta_endpose(
    absolute_action: torch.Tensor,
    anchor_action: torch.Tensor,
) -> torch.Tensor:
    """Express absolute EEF targets in the current EEF coordinate frames.

    For each arm this computes ``inv(T_current) @ T_target``. Grippers remain
    absolute commands, matching the simulator's 16-D end-pose interface.
    """
    if absolute_action.ndim != 2 or absolute_action.shape[1] != 16:
        raise ValueError(
            f"Absolute endpose action must be [T,16], got {tuple(absolute_action.shape)}."
        )
    anchor_action = anchor_action.reshape(1, 16).to(absolute_action)
    result = absolute_action.clone()
    for pose_start in (0, 8):
        quaternion = _normalize_quaternion_xyzw(
            absolute_action[:, pose_start + 3 : pose_start + 7]
        )
        anchor_quaternion = _normalize_quaternion_xyzw(
            anchor_action[:, pose_start + 3 : pose_start + 7]
        )
        world_translation = (
            absolute_action[:, pose_start : pose_start + 3]
            - anchor_action[:, pose_start : pose_start + 3]
        )
        result[:, pose_start : pose_start + 3] = _quat_xyzw_rotate(
            _quat_xyzw_inverse(anchor_quaternion).expand_as(quaternion),
            world_translation,
        )
        result[:, pose_start + 3 : pose_start + 7] = (
            _canonicalize_quaternion_xyzw(
                _quat_xyzw_multiply(
                    _quat_xyzw_inverse(anchor_quaternion).expand_as(quaternion),
                    quaternion,
                )
            )
        )
    return result


def restore_current_se3_delta_endpose(
    relative_action: torch.Tensor,
    anchor_action: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct absolute EEF targets from current-frame SE(3) deltas."""
    if relative_action.ndim != 2 or relative_action.shape[1] != 16:
        raise ValueError("Relative endpose action must be [T,16].")
    anchor_action = anchor_action.reshape(1, 16).to(relative_action)
    result = relative_action.clone()
    for pose_start in (0, 8):
        anchor_quaternion = _normalize_quaternion_xyzw(
            anchor_action[:, pose_start + 3 : pose_start + 7]
        )
        relative_quaternion = _normalize_quaternion_xyzw(
            relative_action[:, pose_start + 3 : pose_start + 7]
        )
        result[:, pose_start : pose_start + 3] = (
            anchor_action[:, pose_start : pose_start + 3]
            + _quat_xyzw_rotate(
                anchor_quaternion.expand_as(relative_quaternion),
                relative_action[:, pose_start : pose_start + 3],
            )
        )
        result[:, pose_start + 3 : pose_start + 7] = _normalize_quaternion_xyzw(
            _quat_xyzw_multiply(
                anchor_quaternion.expand_as(relative_quaternion),
                relative_quaternion,
            )
        )
    return result


def pack_robotwin_action(relative_action: torch.Tensor) -> torch.Tensor:
    if relative_action.ndim != 2 or relative_action.shape[1] != 16:
        raise ValueError("Relative RobotWin action must be [T,16].")
    packed = torch.zeros(
        (relative_action.shape[0], 30),
        dtype=relative_action.dtype,
        device=relative_action.device,
    )
    packed[:, 0:7] = relative_action[:, 0:7]
    packed[:, 28] = relative_action[:, 7]
    packed[:, 7:14] = relative_action[:, 8:15]
    packed[:, 29] = relative_action[:, 15]
    return packed


def canonicalize_robotwin_endpose_quaternions(
    absolute_state: torch.Tensor,
) -> torch.Tensor:
    """Choose the deterministic positive-w representative for physical state."""
    result = absolute_state.clone()
    for pose_start in (0, 8):
        result[:, pose_start + 3 : pose_start + 7] = (
            _canonicalize_quaternion_xyzw(
                result[:, pose_start + 3 : pose_start + 7]
            )
        )
    return result


def unpack_robotwin_action(packed_action: torch.Tensor) -> torch.Tensor:
    if packed_action.ndim != 2 or packed_action.shape[1] != 30:
        raise ValueError("Packed RobotWin action must be [T,30].")
    result = torch.empty(
        (packed_action.shape[0], 16),
        dtype=packed_action.dtype,
        device=packed_action.device,
    )
    result[:, 0:7] = packed_action[:, 0:7]
    result[:, 7] = packed_action[:, 28]
    result[:, 8:15] = packed_action[:, 7:14]
    result[:, 15] = packed_action[:, 29]
    return result


class RMBenchEndposeActionStore:
    """Load raw RMBench endpose actions with LingBot RobotWin preprocessing."""

    def __init__(
        self,
        source_map_path: str | Path,
        stats_path: str | Path,
        *,
        max_open_episodes: int = 8,
        max_cached_episodes: int = 8,
    ) -> None:
        self.source_map_path = Path(source_map_path).expanduser().resolve()
        self.stats_path = Path(stats_path).expanduser().resolve()
        source_map = json.loads(self.source_map_path.read_text(encoding="utf-8"))
        if source_map.get("format") != SOURCE_MAP_FORMAT:
            raise ValueError(f"Unexpected RMBench source-map format: {source_map.get('format')!r}")
        stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.format = str(stats.get("format", ""))
        if self.format not in SUPPORTED_ACTION_FORMATS:
            raise ValueError(f"Unexpected endpose action stats format: {stats.get('format')!r}")
        if self.format == ACTION_FORMAT:
            expected_semantics = {
                "source_quaternion_order": "wxyz",
                "model_quaternion_order": "xyzw",
                "relative_anchor": "episode_initial",
                "normalization": "q01_q99_plus_1e-6_clip_1.5",
                "target_offset": TARGET_OFFSET,
            }
            self.state_representation = "match_episode_relative_action"
            self.model_dim = 30
            self.valid_channels = VALID_CHANNELS
        else:
            expected_semantics = {
                "source_quaternion_order": "wxyz",
                "model_quaternion_order": "xyzw",
                "relative_anchor": "current_observation",
                "state_representation": "absolute_actual_eef",
                "translation_delta_frame": "current_eef_local",
                "normalization": "q01_q99_plus_1e-6_clip_1.5",
                "target_offset": TARGET_OFFSET,
            }
            self.state_representation = str(stats.get("state_representation"))
            self.model_dim = 16
            self.valid_channels = CURRENT_DELTA_VALID_CHANNELS
        mismatches = {
            key: {"actual": stats.get(key), "expected": value}
            for key, value in expected_semantics.items()
            if stats.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Endpose stats semantics do not match LingBot RobotWin: {mismatches}."
            )
        if int(stats.get("model_dim", self.model_dim)) != self.model_dim:
            raise ValueError("Endpose stats use a different model action width.")
        if tuple(stats.get("valid_channels", ())) != self.valid_channels:
            raise ValueError("Endpose stats use a different RobotWin channel mapping.")
        self.entries = {
            int(entry["episode_index"]): entry
            for entry in source_map["entries"]
        }
        self.relative_anchor = str(stats["relative_anchor"])
        self.q01 = torch.tensor(stats["q01"], dtype=torch.float32)
        self.q99 = torch.tensor(stats["q99"], dtype=torch.float32)
        if self.q01.shape != (self.model_dim,) or self.q99.shape != (self.model_dim,):
            raise ValueError(
                f"Endpose q01/q99 stats must contain {self.model_dim} channels."
            )
        # Match the official quantile formula exactly, including +1e-6.
        self.scale = 2.0 / (self.q99 - self.q01 + 1e-6)
        self.offset = -1.0 - self.q01 * self.scale
        invalid = torch.ones(self.model_dim, dtype=torch.bool)
        invalid[list(self.valid_channels)] = False
        self.scale[invalid] = 1.0
        self.offset[invalid] = 0.0
        if self.state_representation == "absolute_actual_eef":
            self.state_q01 = torch.tensor(stats["state_q01"], dtype=torch.float32)
            self.state_q99 = torch.tensor(stats["state_q99"], dtype=torch.float32)
            if (
                self.state_q01.shape != (self.model_dim,)
                or self.state_q99.shape != (self.model_dim,)
            ):
                raise ValueError(
                    "Endpose state q01/q99 stats must contain "
                    f"{self.model_dim} channels."
                )
            self.state_scale = 2.0 / (self.state_q99 - self.state_q01 + 1e-6)
            self.state_offset = -1.0 - self.state_q01 * self.state_scale
            self.state_scale[invalid] = 1.0
            self.state_offset[invalid] = 0.0
        else:
            self.state_q01 = self.q01
            self.state_q99 = self.q99
            self.state_scale = self.scale
            self.state_offset = self.offset
        self.max_open_episodes = max(1, int(max_open_episodes))
        self.max_cached_episodes = max(1, int(max_cached_episodes))
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._episode_actions: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._episode_anchors: dict[int, torch.Tensor] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        state["_episode_actions"] = OrderedDict()
        state["_episode_anchors"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._episode_actions.clear()
        self._episode_anchors.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _open(self, episode_index: int) -> h5py.File:
        if episode_index in self._handles:
            handle = self._handles.pop(episode_index)
            self._handles[episode_index] = handle
            return handle
        entry = self.entries.get(int(episode_index))
        if entry is None:
            raise KeyError(f"Unknown RMBench episode {episode_index}.")
        handle = h5py.File(entry["source_hdf5"], "r")
        self._handles[episode_index] = handle
        while len(self._handles) > self.max_open_episodes:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def _absolute_episode(self, episode_index: int) -> torch.Tensor:
        episode_index = int(episode_index)
        absolute = self._episode_actions.pop(episode_index, None)
        if absolute is None:
            entry = self.entries[episode_index]
            episode_length = int(entry["length"])
            handle = self._open(episode_index)
            left = np.asarray(
                handle["endpose/left_endpose"][:episode_length],
                dtype=np.float32,
            )
            left_grip = np.asarray(
                handle["endpose/left_gripper"][:episode_length],
                dtype=np.float32,
            )[:, None]
            right = np.asarray(
                handle["endpose/right_endpose"][:episode_length],
                dtype=np.float32,
            )
            right_grip = np.asarray(
                handle["endpose/right_gripper"][:episode_length],
                dtype=np.float32,
            )[:, None]
            absolute = rmbench_wxyz_to_robotwin_xyzw(
                torch.from_numpy(
                    np.concatenate(
                        [left, left_grip, right, right_grip], axis=1
                    )
                )
            )
            if tuple(absolute.shape) != (episode_length, 16):
                raise ValueError(
                    "RMBench endpose episode shape mismatch: "
                    f"episode={episode_index}, shape={tuple(absolute.shape)}, "
                    f"expected={(episode_length, 16)}."
                )
        self._episode_actions[episode_index] = absolute
        while len(self._episode_actions) > self.max_cached_episodes:
            evicted_index, _ = self._episode_actions.popitem(last=False)
            self._episode_anchors.pop(evicted_index, None)
        return absolute

    def load_absolute_window(
        self,
        episode_index: int,
        frame_index: int,
        horizon: int,
    ) -> torch.Tensor:
        absolute_episode = self._absolute_episode(int(episode_index))
        episode_length = int(absolute_episode.shape[0])
        # LeRobot's contract is observation[t] -> action[t] == source[t+1].
        start = min(
            max(int(frame_index) + TARGET_OFFSET, 0),
            episode_length - 1,
        )
        stop = min(start + int(horizon), episode_length)
        absolute = absolute_episode[start:stop]
        if int(absolute.shape[0]) < int(horizon):
            absolute = torch.cat(
                [absolute, absolute[-1:].expand(int(horizon) - int(absolute.shape[0]), -1)],
                dim=0,
            )
        return absolute

    def load_episode_anchor(self, episode_index: int) -> torch.Tensor:
        """Return the unshifted episode-initial pose used as relative anchor."""
        return self._absolute_episode(int(episode_index))[0]

    def normalize_absolute(
        self,
        absolute_action: torch.Tensor,
        anchor_action: torch.Tensor,
    ) -> torch.Tensor:
        """Map absolute RobotWin xyzw endposes into the model's 30-D space."""
        absolute_action = torch.as_tensor(
            absolute_action,
            dtype=torch.float32,
            device="cpu",
        )
        if absolute_action.ndim == 1:
            absolute_action = absolute_action.unsqueeze(0)
        if absolute_action.ndim != 2 or absolute_action.shape[1] != 16:
            raise ValueError(
                "Absolute RobotWin endpose must be [T,16], got "
                f"{tuple(absolute_action.shape)}."
            )
        anchor_action = torch.as_tensor(
            anchor_action,
            dtype=torch.float32,
            device="cpu",
        )
        relative = (
            make_current_se3_delta_endpose(absolute_action, anchor_action)
            if getattr(self, "relative_anchor", "episode_initial")
            == "current_observation"
            else make_relative_endpose(absolute_action, anchor_action)
        )
        model_action = (
            relative
            if getattr(self, "model_dim", 30) == 16
            else pack_robotwin_action(relative)
        )
        return (model_action * self.scale + self.offset).clamp(-1.5, 1.5)

    def normalize_observed_absolute(
        self,
        absolute_state: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize physical EEF observations without treating them as actions."""
        absolute_state = torch.as_tensor(
            absolute_state,
            dtype=torch.float32,
            device="cpu",
        )
        if absolute_state.ndim == 1:
            absolute_state = absolute_state.unsqueeze(0)
        if absolute_state.ndim != 2 or absolute_state.shape[1] != 16:
            raise ValueError(
                "Absolute RobotWin endpose state must be [T,16], got "
                f"{tuple(absolute_state.shape)}."
            )
        if getattr(self, "state_representation", "match_episode_relative_action") != (
            "absolute_actual_eef"
        ):
            raise RuntimeError(
                "Legacy episode-relative state normalization requires its episode anchor."
            )
        canonical_state = canonicalize_robotwin_endpose_quaternions(
            absolute_state
        )
        model_state = (
            canonical_state
            if getattr(self, "model_dim", 30) == 16
            else pack_robotwin_action(canonical_state)
        )
        return (model_state * self.state_scale + self.state_offset).clamp(-1.5, 1.5)

    def normalize_states(
        self,
        episode_index: int,
        frame_indices: list[int] | tuple[int, ...] | torch.Tensor,
    ) -> torch.Tensor:
        """Normalize observed EEF states with the exact action representation."""
        absolute_episode = self._absolute_episode(int(episode_index))
        episode_length = int(absolute_episode.shape[0])
        indices = torch.as_tensor(frame_indices, dtype=torch.long).reshape(-1)
        indices = indices.clamp(min=0, max=episode_length - 1)
        states = absolute_episode.index_select(0, indices)
        if getattr(self, "state_representation", "match_episode_relative_action") == (
            "absolute_actual_eef"
        ):
            return self.normalize_observed_absolute(states)
        return self.normalize_absolute(states, absolute_episode[0])

    def normalize_window(
        self,
        episode_index: int,
        frame_index: int,
        horizon: int,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        absolute = self.load_absolute_window(episode_index, frame_index, horizon)
        episode_index = int(episode_index)
        if getattr(self, "relative_anchor", "episode_initial") == "current_observation":
            absolute_episode = self._absolute_episode(episode_index)
            current_index = min(max(int(frame_index), 0), len(absolute_episode) - 1)
            anchor = absolute_episode[current_index]
        else:
            anchor = self._episode_anchors.get(episode_index)
            if anchor is None:
                anchor = self._absolute_episode(episode_index)[0]
                self._episode_anchors[episode_index] = anchor
        normalized = self.normalize_absolute(absolute, anchor)
        if action_is_pad is not None:
            pad = torch.as_tensor(action_is_pad, dtype=torch.bool)[:horizon]
            normalized[pad] = 0.0
        return normalized

    def denormalize_to_absolute(
        self,
        normalized_action: torch.Tensor,
        anchor_action: torch.Tensor,
    ) -> torch.Tensor:
        normalized_action = normalized_action.to(dtype=torch.float32, device="cpu")
        model_action = (normalized_action - self.offset) / self.scale
        invalid = torch.ones(self.model_dim, dtype=torch.bool)
        invalid[list(self.valid_channels)] = False
        model_action[:, invalid] = 0.0
        relative = (
            model_action
            if self.model_dim == 16
            else unpack_robotwin_action(model_action)
        )
        anchor_action = anchor_action.to(dtype=torch.float32, device="cpu")
        if getattr(self, "relative_anchor", "episode_initial") == "current_observation":
            return restore_current_se3_delta_endpose(relative, anchor_action)
        return restore_absolute_endpose(relative, anchor_action)


__all__ = [
    "ACTION_FORMAT",
    "CURRENT_DELTA_ACTION_FORMAT",
    "CURRENT_DELTA_VALID_CHANNELS",
    "RMBenchEndposeActionStore",
    "SUPPORTED_ACTION_FORMATS",
    "TARGET_OFFSET",
    "VALID_CHANNELS",
    "canonicalize_robotwin_endpose_quaternions",
    "make_relative_endpose",
    "make_current_se3_delta_endpose",
    "pack_robotwin_action",
    "rmbench_wxyz_to_robotwin_xyzw",
    "restore_absolute_endpose",
    "restore_current_se3_delta_endpose",
    "robotwin_xyzw_to_rmbench_wxyz",
    "unpack_robotwin_action",
]
