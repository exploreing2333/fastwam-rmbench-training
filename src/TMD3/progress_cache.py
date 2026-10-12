from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from TMD2.progress_cache import uint16_to_bf16


STATE_PROGRESS_CACHE_FORMAT = "fastwam_progress_model_cache_v5"
STATE_PROGRESS_MEMORY_CONTRACT = "tmd3_progress_logits_decoded_subtask_v2"
STATE_PROGRESS_COMPACT_CACHE_FORMAT = "fastwam_progress_model_cache_v6_compact"
STATE_PROGRESS_COMPACT_MEMORY_CONTRACT = "tmd4_v56_c51_v53protocol_subtask_v1"
STATE_PROGRESS_V61_COMPACT_MEMORY_CONTRACT = (
    "tmd4_v61_c51_causal_etg_subtask_v1"
)
STATE_PROGRESS_ALIGNED_CACHE_FORMAT = "fastwam_progress_model_cache_v7_aligned"
STATE_PROGRESS_V72_ALIGNED_MEMORY_CONTRACT = (
    "tmd4_v72_postupdate_aligned_c51_deployed_subtask_v1"
)
SUPPORTED_STATE_PROGRESS_CACHE_FORMATS = {
    STATE_PROGRESS_CACHE_FORMAT,
    STATE_PROGRESS_COMPACT_CACHE_FORMAT,
    STATE_PROGRESS_ALIGNED_CACHE_FORMAT,
}
SUBTASK_T5_CACHE_FORMAT = "fastwam_progress_subtask_t5_v1"
ORDINAL_CONTEXT_CONTRACT = {
    "active_idx_attr": "planner_active_idx_after",
    "plan_length_attr": "planner_plan_length_after",
    "bootstrap_active_idx": -1,
    "terminal_active_idx_equals_plan_length": True,
}


def subtask_token_ids_sha256(token_ids: np.ndarray) -> str:
    """Return a stable content key for Progress-model subtask token IDs."""
    canonical = np.asarray(token_ids, dtype="<i4").reshape(-1)
    if canonical.size == 0:
        raise ValueError("Decoded subtask token IDs cannot be empty.")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


class StateProgressFeatureStore:
    """Random access to compact progress logits and decoded subtask tokens."""

    def __init__(
        self,
        root: str | Path,
        max_open_episodes: int = 8,
        max_cached_instruction_episodes: int = 8,
        subtask_text_embedding_cache_dir: str | Path | None = None,
        require_ordinal_context: bool = False,
        require_gt_subtask_exact: bool = False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.max_open_episodes = max(1, int(max_open_episodes))
        self.max_cached_subtask_contexts = max(
            1, int(max_cached_instruction_episodes)
        )
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Progress cache manifest does not exist: {manifest_path}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cache_format = str(self.manifest.get("format", ""))
        if self.cache_format not in SUPPORTED_STATE_PROGRESS_CACHE_FORMATS:
            raise ValueError(
                "Unsupported StateProgress cache format: "
                f"{self.manifest.get('format')!r}."
            )
        if not bool(self.manifest.get("complete", False)):
            raise RuntimeError(f"Progress cache is incomplete: {manifest_path}")
        compact_cache = self.cache_format == STATE_PROGRESS_COMPACT_CACHE_FORMAT
        aligned_cache = self.cache_format == STATE_PROGRESS_ALIGNED_CACHE_FORMAT
        text_condition_cache = compact_cache or aligned_cache
        memory_contract = self.manifest.get("memory_contract")
        expected_memory_contracts = (
            {STATE_PROGRESS_V72_ALIGNED_MEMORY_CONTRACT}
            if aligned_cache
            else
            {
                STATE_PROGRESS_COMPACT_MEMORY_CONTRACT,
                STATE_PROGRESS_V61_COMPACT_MEMORY_CONTRACT,
            }
            if compact_cache
            else {STATE_PROGRESS_MEMORY_CONTRACT}
        )
        if memory_contract not in expected_memory_contracts:
            raise RuntimeError(
                "Progress cache memory contract mismatch: "
                f"{memory_contract!r}."
            )
        expected = {
            "full_attention_tokens": [],
            "cross_attention_tokens": [
                "decoded_subtask",
                "compressed_progress_logits",
            ],
            "excluded_conditioning_tokens": [
                "instruction",
                "current_progress_hidden",
                "progress_state_hidden",
                "layerwise_subtask_hidden",
            ],
            "progress_logits_dataset": "progress_logits",
            "progress_logits_bins": 10,
            "invalid_generation_policy": (
                "preserve_features_without_planner_update_official_rollout"
            ),
        }
        if text_condition_cache:
            expected.update(
                {
                    "subtask_conditioning": "deployed_current_subtask_text",
                    "condition_storage": "deduplicated_token_sequence_per_episode",
                    "invalid_generation_policy": (
                        "preserve_features_without_planner_update_v61_causal_etg"
                        if memory_contract
                        == STATE_PROGRESS_V61_COMPACT_MEMORY_CONTRACT
                        else "preserve_features_without_planner_update_v53_rollout"
                    ),
                }
            )
            if aligned_cache:
                expected.update(
                    {
                        "progress_state_alignment": (
                            "post_runtime_update_same_frame_reforward"
                        ),
                        "invalid_generation_policy": (
                            "preserve_state_and_features_v72_language_delta"
                        ),
                    }
                )
        else:
            expected.update(
                {
                    "subtask_hidden_layout": "final_backbone_output",
                    "subtask_hidden_dataset": "decoded_subtask_hidden",
                }
            )
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise RuntimeError(
                    f"Progress cache {key} mismatch: expected={value!r}, "
                    f"actual={self.manifest.get(key)!r}."
                )
        state_contract = self.manifest.get("state_history_contract", {})
        if state_contract != {
            "source": "normalized_dataset_state",
            "frames": 4,
            "control_offsets": [-48, -32, -16, 0],
            "latent_alignment": ["H1", "H2", "H3", "H4"],
            "padding": "episode_start_masked",
        }:
            raise RuntimeError(
                f"Progress cache state history contract mismatch: {state_contract!r}."
            )
        if not text_condition_cache and int(
            self.manifest.get("transition_confirm_observations", -1)
        ) != 1:
            raise RuntimeError("Legacy TMD3 Progress cache must use confirm1.")
        self.num_layers = int(self.manifest["num_layers"])
        self.hidden_size = int(self.manifest["hidden_size"])
        self.progress_bins = int(self.manifest["progress_logits_bins"])
        self.total_episodes = int(self.manifest["total_episodes"])
        self.total_frames = int(self.manifest["total_frames"])
        self.require_ordinal_context = bool(require_ordinal_context)
        self.require_gt_subtask_exact = bool(require_gt_subtask_exact)
        if (
            self.require_ordinal_context
            and self.manifest.get("ordinal_context_contract")
            != ORDINAL_CONTEXT_CONTRACT
        ):
            raise RuntimeError(
                "Progress cache does not contain the required active-ordinal/"
                "plan-length contract. Regenerate it with the updated builder."
            )
        if self.require_gt_subtask_exact:
            admission = self.manifest.get("training_admission")
            expected_admission = {
                "format": "rmbench_gt_subtask_exact_mask_v1",
                "complete": True,
                "dataset": "gt_subtask_exact",
                "policy": "normalized_deployed_subtask_equals_frame_gt_subtask",
            }
            if not isinstance(admission, dict) or any(
                admission.get(key) != value
                for key, value in expected_admission.items()
            ):
                raise RuntimeError(
                    "Progress cache lacks a complete GT-exact training admission "
                    f"contract: {manifest_path}."
                )
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self.subtask_text_embedding_cache_dir = (
            None
            if subtask_text_embedding_cache_dir in (None, "")
            else Path(subtask_text_embedding_cache_dir).expanduser().resolve()
        )
        if text_condition_cache and self.subtask_text_embedding_cache_dir is None:
            raise ValueError(
                "This compact Progress cache contains semantic token IDs, not "
                "Qwen hidden states; progress_subtask_text_embedding_cache_dir is "
                "required."
            )
        self._subtask_context_entries: dict[str, dict[str, Any]] = {}
        self._subtask_contexts: OrderedDict[
            str, tuple[torch.Tensor, torch.BoolTensor]
        ] = OrderedDict()
        if self.subtask_text_embedding_cache_dir is not None:
            text_manifest_path = (
                self.subtask_text_embedding_cache_dir / "manifest.json"
            )
            if not text_manifest_path.is_file():
                raise FileNotFoundError(
                    "Decoded-subtask T5 manifest does not exist: "
                    f"{text_manifest_path}"
                )
            text_manifest = json.loads(
                text_manifest_path.read_text(encoding="utf-8")
            )
            if text_manifest.get("format") != SUBTASK_T5_CACHE_FORMAT:
                raise ValueError(
                    "Unsupported decoded-subtask T5 cache format: "
                    f"{text_manifest.get('format')!r}."
                )
            if not bool(text_manifest.get("complete", False)):
                raise RuntimeError(
                    f"Decoded-subtask T5 cache is incomplete: {text_manifest_path}"
                )
            source_root = Path(
                str(text_manifest.get("source_progress_cache_root", ""))
            ).expanduser().resolve()
            if source_root != self.root:
                raise RuntimeError(
                    "Decoded-subtask T5 cache was built from a different "
                    f"Progress cache: expected={self.root}, actual={source_root}."
                )
            if int(text_manifest.get("context_dim", -1)) != 4096:
                raise RuntimeError("Decoded-subtask T5 context width must be 4096.")
            entries = text_manifest.get("entries")
            if not isinstance(entries, dict) or not entries:
                raise RuntimeError(
                    "Decoded-subtask T5 manifest contains no token mappings."
                )
            self._subtask_context_entries = entries

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        state["_subtask_contexts"] = OrderedDict()
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
        return self.root / "episodes" / f"episode_{int(episode_index):06d}.h5"

    def _open(self, episode_index: int) -> h5py.File:
        episode_index = int(episode_index)
        handle = self._handles.pop(episode_index, None)
        if handle is None:
            path = self._episode_path(episode_index)
            if not path.is_file():
                raise FileNotFoundError(f"Missing Progress cache episode: {path}")
            handle = h5py.File(path, "r", libver="latest", swmr=True)
            if handle.attrs.get("format") != self.cache_format:
                handle.close()
                raise ValueError(f"Invalid StateProgress episode format: {path}")
            if not bool(handle.attrs.get("complete", False)):
                handle.close()
                raise RuntimeError(f"Incomplete StateProgress episode: {path}")
        self._handles[episode_index] = handle
        while len(self._handles) > self.max_open_episodes:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        return handle

    def _load_subtask_text_context(
        self,
        condition: h5py.Group,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        token_ids = np.asarray(condition["subtask_token_ids"][...], dtype=np.int32)
        token_key = subtask_token_ids_sha256(token_ids)
        cached = self._subtask_contexts.pop(token_key, None)
        if cached is not None:
            self._subtask_contexts[token_key] = cached
            return cached
        entry = self._subtask_context_entries.get(token_key)
        if not isinstance(entry, dict):
            raise KeyError(
                "Decoded subtask has no training-compatible T5 embedding: "
                f"token_sha256={token_key}."
            )
        relative_path = entry.get("file")
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(
                f"Decoded-subtask T5 entry has no file: token_sha256={token_key}."
            )
        cache_path = self.subtask_text_embedding_cache_dir / relative_path
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Decoded-subtask T5 payload is missing: {cache_path}"
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context = payload["context"].to(dtype=torch.bfloat16).contiguous()
        mask = payload["mask"].to(dtype=torch.bool).contiguous()
        if context.ndim != 2 or int(context.shape[-1]) != 4096:
            raise ValueError(
                f"Decoded-subtask T5 context must be [L,4096], got "
                f"{tuple(context.shape)} from {cache_path}."
            )
        if tuple(mask.shape) != (int(context.shape[0]),):
            raise ValueError(
                f"Decoded-subtask T5 mask shape does not match {cache_path}."
            )
        active = torch.nonzero(mask, as_tuple=False)
        if int(active.numel()) == 0:
            raise ValueError(f"Decoded-subtask T5 context is empty: {cache_path}.")
        keep_length = int(active[-1].item()) + 1
        if not bool(mask[:keep_length].all()):
            raise ValueError(
                f"Decoded-subtask T5 mask is not right padded: {cache_path}."
            )
        cached = (
            context[:keep_length].contiguous(),
            mask[:keep_length].contiguous(),
        )
        self._subtask_contexts[token_key] = cached
        while len(self._subtask_contexts) > self.max_cached_subtask_contexts:
            self._subtask_contexts.popitem(last=False)
        return cached

    def load(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        handle = self._open(episode_index)
        frame_index = int(frame_index)
        length = int(handle.attrs["length"])
        if not 0 <= frame_index < length:
            raise IndexError(
                f"Frame {frame_index} is outside episode {episode_index} length {length}."
            )
        condition_index = int(handle["condition_index"][frame_index])
        if condition_index < 0:
            raise RuntimeError(
                f"Episode {episode_index} frame {frame_index} has no condition."
            )
        condition = handle["conditions"][f"{condition_index:06d}"]
        progress_logits = torch.from_numpy(
            handle["progress_logits"][frame_index]
        ).float()
        if tuple(progress_logits.shape) != (self.progress_bins,):
            raise ValueError(
                f"Invalid progress-logit cache shape: {tuple(progress_logits.shape)}."
            )
        if self.subtask_text_embedding_cache_dir is None:
            subtask = uint16_to_bf16(condition["decoded_subtask_hidden"][...])
            if (
                subtask.ndim != 2
                or int(subtask.shape[-1]) != self.hidden_size
            ):
                raise ValueError(
                    f"Invalid decoded subtask cache shape: {tuple(subtask.shape)}."
                )
            subtask_mask = torch.ones(int(subtask.shape[0]), dtype=torch.bool)
        else:
            subtask, subtask_mask = self._load_subtask_text_context(condition)
        result = {
            "progress_logits": progress_logits,
            "progress_subtask_tokens": subtask,
            "progress_subtask_mask": subtask_mask,
            "progress_condition_index": condition_index,
            "progress_decode_source_frame": int(condition.attrs["source_frame"]),
            "progress_parse_valid": bool(condition.attrs["parse_valid"]),
        }
        if self.require_ordinal_context:
            active_idx = int(condition.attrs["planner_active_idx_after"])
            plan_length = int(condition.attrs["planner_plan_length_after"])
            if plan_length < 0 or not -1 <= active_idx <= plan_length:
                raise ValueError(
                    "Invalid cached planner ordinal: "
                    f"active_idx={active_idx}, plan_length={plan_length}."
                )
            result["progress_active_idx"] = torch.tensor(
                active_idx, dtype=torch.long
            )
            result["progress_plan_length"] = torch.tensor(
                plan_length, dtype=torch.long
            )
        return result

    def load_training_valid_mask(self, episode_index: int) -> np.ndarray:
        """Load the audited frame mask used to admit anchors for training."""
        if not self.require_gt_subtask_exact:
            raise RuntimeError(
                "GT-exact admission was not required when this store was created."
            )
        handle = self._open(episode_index)
        dataset_name = str(self.manifest["training_admission"]["dataset"])
        if dataset_name not in handle:
            raise RuntimeError(
                f"Episode {episode_index} lacks training mask {dataset_name!r}."
            )
        mask = np.asarray(handle[dataset_name][...], dtype=np.bool_)
        length = int(handle.attrs["length"])
        if tuple(mask.shape) != (length,):
            raise ValueError(
                f"Episode {episode_index} training mask has shape {mask.shape}, "
                f"expected {(length,)}."
            )
        return mask


def pad_state_progress_features(
    samples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    if not samples:
        return {}
    hidden = int(samples[0]["progress_subtask_tokens"].shape[-1])
    progress_bins = int(samples[0]["progress_logits"].shape[-1])
    max_subtask = max(
        int(item["progress_subtask_tokens"].shape[0]) for item in samples
    )
    logits = torch.empty(len(samples), progress_bins, dtype=torch.float32)
    subtask = torch.zeros(
        len(samples),
        max_subtask,
        hidden,
        dtype=torch.bfloat16,
    )
    subtask_mask = torch.zeros(len(samples), max_subtask, dtype=torch.bool)
    for row, item in enumerate(samples):
        if tuple(item["progress_logits"].shape) != (progress_bins,):
            raise ValueError("Progress-logit width differs within a batch.")
        subtask_len = int(item["progress_subtask_tokens"].shape[0])
        logits[row] = item["progress_logits"]
        subtask[row, :subtask_len] = item["progress_subtask_tokens"]
        subtask_mask[row, :subtask_len] = item["progress_subtask_mask"]
    result = {
        "progress_logits": logits,
        "progress_subtask_tokens": subtask,
        "progress_subtask_mask": subtask_mask,
        "progress_condition_index": torch.tensor(
            [item["progress_condition_index"] for item in samples],
            dtype=torch.long,
        ),
        "progress_decode_source_frame": torch.tensor(
            [item["progress_decode_source_frame"] for item in samples],
            dtype=torch.long,
        ),
        "progress_parse_valid": torch.tensor(
            [item["progress_parse_valid"] for item in samples],
            dtype=torch.bool,
        ),
    }
    has_ordinal = "progress_active_idx" in samples[0]
    if any(("progress_active_idx" in item) != has_ordinal for item in samples):
        raise ValueError("Ordinal context presence differs within a batch.")
    if has_ordinal:
        result["progress_active_idx"] = torch.stack(
            [torch.as_tensor(item["progress_active_idx"]).long() for item in samples]
        ).reshape(-1)
        result["progress_plan_length"] = torch.stack(
            [
                torch.as_tensor(item["progress_plan_length"]).long()
                for item in samples
            ]
        ).reshape(-1)
    return result


__all__ = [
    "STATE_PROGRESS_CACHE_FORMAT",
    "STATE_PROGRESS_COMPACT_CACHE_FORMAT",
    "STATE_PROGRESS_ALIGNED_CACHE_FORMAT",
    "SUPPORTED_STATE_PROGRESS_CACHE_FORMATS",
    "STATE_PROGRESS_MEMORY_CONTRACT",
    "STATE_PROGRESS_COMPACT_MEMORY_CONTRACT",
    "STATE_PROGRESS_V61_COMPACT_MEMORY_CONTRACT",
    "STATE_PROGRESS_V72_ALIGNED_MEMORY_CONTRACT",
    "SUBTASK_T5_CACHE_FORMAT",
    "ORDINAL_CONTEXT_CONTRACT",
    "StateProgressFeatureStore",
    "pad_state_progress_features",
    "subtask_token_ids_sha256",
]
