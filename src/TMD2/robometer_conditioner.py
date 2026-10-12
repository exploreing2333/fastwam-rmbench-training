from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from robometer.configs.experiment_configs import ModelConfig
from robometer.data.collators.subtask_rbm import SubtaskRBMBatchCollator
from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.data.datasets.rmbench_subtask import (
    PREPROGRESS_SUPERVISION_SLOTS,
    TASKSTART1_TRANSITION4_SUBTASK3_SUPERVISION_SLOTS,
    UNIFORM16S_THEN_GAP6S_DISJOINT_RANDOM4S3_SUPERVISION_SLOTS,
    taskstart1_transition4_subtask3_layout,
)
from robometer.robot_state import ROBOT_STATE_SCHEMA
from robometer.subtask_planner import AbsoluteOrdinalState, TASK_COMPLETE
from robometer.utils.setup_utils import setup_model_and_processor

from TMD1.fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_V56_FORWARD_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "position_ids",
    "cache_position",
    "second_per_grid_ts",
    "planner_protocol_id",
    "observation_attention_token_mask",
}


@dataclass
class RobometerCondition:
    memory_tokens: torch.Tensor
    memory_mask: torch.BoolTensor
    current_progress_tokens: torch.Tensor
    current_progress_mask: torch.BoolTensor
    subtask_tokens: torch.Tensor
    subtask_layer_tokens: torch.Tensor | None
    subtask_mask: torch.BoolTensor
    instruction_tokens: torch.Tensor
    instruction_mask: torch.BoolTensor
    subtask_token_ids: torch.LongTensor
    sequences: torch.LongTensor
    decoded_active_idx: int
    planner_active_idx: int
    planner_plan_frozen: bool
    planner_committed_idx: int
    current_subtask: str
    plan: tuple[str, ...]
    execution_history: tuple[str, ...]
    generated_text: str
    parse_valid: bool
    parse_error: str
    planner_repair: str
    accepted_incomplete_plan: bool
    planner_update_applied: bool
    planner_committed_now: bool
    local_progress: float
    global_progress: float
    progress_bin: int
    progress_confidence: float
    progress_logits: torch.Tensor
    done_probability: float
    language_invoked: bool
    state_event: str


@dataclass
class CompactCausalState:
    """Deployed symbolic state used by the V56/V53 compact cache builder."""

    current_subtask: str | None = None
    current_start_frame: int = 0
    completed_subtasks: list[str] = field(default_factory=list)
    committed_idx: int = -1


@dataclass
class V72CortexState(CompactCausalState):
    """Exact recurrent state of the deployed V72 Cortex planner."""

    transition_history_frames: list[int] = field(default_factory=list)
    last_frame: int = -1


class FrozenRobometerConditioner(nn.Module):
    """Online Robometer path that is identical to the TMD2 cache builder."""

    def __init__(
        self,
        checkpoint_path: str,
        run_config_path: str,
        max_new_tokens: int = 512,
        num_layers: int = 30,
        transition_confirm_observations: int = 1,
        capture_subtask_layers: bool = False,
    ):
        super().__init__()
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.run_config_path = str(Path(run_config_path).resolve())
        self.max_new_tokens = int(max_new_tokens)
        self.num_layers = int(num_layers)
        self.transition_confirm_observations = int(
            transition_confirm_observations
        )
        self.capture_subtask_layers = bool(capture_subtask_layers)
        if self.max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be positive.")
        if self.num_layers <= 0:
            raise ValueError("`num_layers` must be positive.")
        if self.transition_confirm_observations <= 0:
            raise ValueError("`transition_confirm_observations` must be positive.")

        with open(self.run_config_path, "r", encoding="utf-8") as handle:
            saved_config = yaml.safe_load(handle)
        saved_data = saved_config.get("data") or {}
        saved_model = saved_config.get("model") or {}
        self.frame_sampling = str(
            saved_data.get("rmbench_frame_sampling", "causal_uniform")
        ).lower()
        self.planner_protocol = str(
            saved_data.get("rmbench_planner_protocol", "absolute_ordinal")
        ).lower()
        self.progress_target_space = str(
            saved_data.get("rmbench_progress_target", "local")
        ).lower()
        self.keyframe_stride = int(saved_data.get("rmbench_keyframe_stride", 30))
        self.planner_query_stride = int(
            saved_data.get("rmbench_planner_query_stride", 8)
        )
        self.dense_current_subtask_conditioning = bool(
            saved_data.get("rmbench_dense_current_subtask_conditioning", False)
        )
        self.completion_skill_start_anchor = bool(
            saved_data.get("rmbench_completion_skill_start_anchor", False)
        )
        self.compact_v56_protocol = (
            self.planner_protocol == "causal_planner"
            and self.frame_sampling
            == "uniform16s_then_gap6s_disjoint_random4s3"
        )
        self.v72_cortex_protocol = (
            self.planner_protocol == "cortex_progress_planner"
            and self.frame_sampling == "taskstart1_transition4_subtask3"
        )
        self.compact_online_protocol = bool(
            self.compact_v56_protocol or self.v72_cortex_protocol
        )
        if self.compact_v56_protocol:
            compact_contract = {
                "max_new_tokens": (self.max_new_tokens, 64),
                "progress_target": (self.progress_target_space, "global"),
                "max_frames": (int(saved_data.get("max_frames", 0)), 8),
                "planner_query_stride": (self.planner_query_stride, 8),
                "keyframe_stride": (self.keyframe_stride, 30),
                "dense_current_subtask_conditioning": (
                    self.dense_current_subtask_conditioning,
                    False,
                ),
                "completion_skill_start_anchor": (
                    self.completion_skill_start_anchor,
                    False,
                ),
                "use_success_head": (
                    bool(saved_model.get("use_success_head", False)),
                    True,
                ),
            }
            mismatches = {
                key: {"actual": actual, "expected": expected}
                for key, (actual, expected) in compact_contract.items()
                if actual != expected
            }
            if mismatches:
                raise ValueError(
                    f"Invalid V56 compact online contract: {mismatches}"
                )
        if self.v72_cortex_protocol:
            v72_contract = {
                "max_new_tokens": (self.max_new_tokens, 96),
                "progress_target": (self.progress_target_space, "local"),
                "max_frames": (int(saved_data.get("max_frames", 0)), 8),
                "planner_query_stride": (self.planner_query_stride, 8),
                "keyframe_stride": (self.keyframe_stride, 30),
                "dense_current_subtask_conditioning": (
                    self.dense_current_subtask_conditioning,
                    True,
                ),
                "use_robot_state": (
                    bool(saved_data.get("rmbench_use_robot_state", False)),
                    False,
                ),
                "use_success_head": (
                    bool(saved_model.get("use_success_head", False)),
                    False,
                ),
                "use_causal_etg_decision": (
                    bool(saved_model.get("use_causal_etg_decision", False)),
                    True,
                ),
                "causal_etg_binary_advance": (
                    bool(saved_model.get("causal_etg_binary_advance", False)),
                    True,
                ),
                "causal_language_schema": (
                    str(saved_data.get("rmbench_causal_language_schema", "")),
                    "current_subtask_memory_update",
                ),
                "uniform_terminal_prompt": (
                    bool(saved_data.get("rmbench_uniform_terminal_prompt", False)),
                    True,
                ),
                "completed_start_history": (
                    bool(
                        saved_data.get(
                            "rmbench_transition_history_completed_subtask_starts",
                            False,
                        )
                    ),
                    True,
                ),
                "transition_confirm_observations": (
                    self.transition_confirm_observations,
                    1,
                ),
            }
            mismatches = {
                key: {"actual": actual, "expected": expected}
                for key, (actual, expected) in v72_contract.items()
                if actual != expected
            }
            if mismatches:
                raise ValueError(
                    f"Invalid V72 Cortex online contract: {mismatches}"
                )
        self.preprogress_history_stride = int(
            saved_data.get("rmbench_preprogress_history_stride", 4)
        )
        model_config = ModelConfig(**saved_config["model"])
        self.use_robot_state = bool(model_config.use_robot_state)
        tokenizer, processor, model = setup_model_and_processor(
            model_config,
            hf_model_id=self.checkpoint_path,
        )
        tokenizer.padding_side = "left"
        processor.tokenizer.padding_side = "left"
        model.gradient_checkpointing_disable()
        model.eval().requires_grad_(False)
        if self.num_layers > len(model.model.model.language_model.layers):
            raise ValueError("Requested cache depth exceeds Progress Model depth.")

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        collator_protocol_kwargs = {}
        if self.v72_cortex_protocol:
            collator_protocol_kwargs = {
                "causal_language_schema": str(
                    saved_data["rmbench_causal_language_schema"]
                ),
                "uniform_terminal_prompt": bool(
                    saved_data["rmbench_uniform_terminal_prompt"]
                ),
                "event_instance_memory": bool(
                    saved_data.get("rmbench_event_instance_tasks", [])
                ),
            }
        self.collator = SubtaskRBMBatchCollator(
            processor=processor,
            tokenizer=tokenizer,
            max_length=4096,
            resized_height=None,
            resized_width=None,
            base_model_id=model_config.base_model_id,
            load_embeddings=False,
            use_multi_image=True,
            use_per_frame_progress_token=True,
            inference=True,
            use_visual_subtask_grounding_token=bool(
                saved_model.get("use_visual_subtask_grounding_token", False)
            ),
            dense_current_subtask_conditioning=(
                self.dense_current_subtask_conditioning
            ),
            completion_skill_start_anchor=self.completion_skill_start_anchor,
            causal_done_token_position=str(
                saved_data.get("rmbench_causal_done_token_position", "user_query")
            ),
            **collator_protocol_kwargs,
        )
        self.hidden_dim = int(model.native_hidden_size)
        self.planner_states: list[AbsoluteOrdinalState | CompactCausalState] = []
        self.v72_transactional_mode = False
        self._v72_pending_transition: dict[str, Any] | None = None
        self.reset(1)
        logger.info(
            "Loaded frozen Robometer checkpoint=%s hidden_dim=%d layers=%d "
            "decode_limit=%d protocol=%s transition_confirm=%d "
            "subtask_layers=%s",
            self.checkpoint_path,
            self.hidden_dim,
            self.num_layers,
            self.max_new_tokens,
            self.planner_protocol,
            self.transition_confirm_observations,
            self.capture_subtask_layers,
        )

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.model.eval()
        return self

    def reset(self, batch_size: int = 1) -> None:
        if int(batch_size) <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.v72_cortex_protocol:
            self.planner_states = [
                V72CortexState() for _ in range(int(batch_size))
            ]
        elif self.compact_v56_protocol:
            self.planner_states = [
                CompactCausalState() for _ in range(int(batch_size))
            ]
        else:
            self.planner_states = [
                AbsoluteOrdinalState(
                    transition_confirm_observations=(
                        self.transition_confirm_observations
                    )
                )
                for _ in range(int(batch_size))
            ]
        self._v72_pending_transition = None
        self._gpt_memory_version = 0
        self._gpt_memory_fingerprint = None

    def enable_v72_transactional_mode(self) -> None:
        """Hold V72 semantic transitions until an external arbiter commits them."""
        if not self.v72_cortex_protocol:
            raise RuntimeError("Transactional transitions require the V72 Cortex protocol")
        state = self.planner_states[0]
        if not isinstance(state, V72CortexState):
            raise RuntimeError("V72 Cortex state was not initialized")
        if state.last_frame >= 0:
            raise RuntimeError(
                "Transactional V72 mode must be enabled before the first observation"
            )
        self.v72_transactional_mode = True
        self._v72_pending_transition = None

    def v72_pending_transition(self) -> dict[str, Any] | None:
        pending = self._v72_pending_transition
        return None if pending is None else copy.deepcopy(pending)

    def commit_v72_transition(
        self,
        *,
        proposal_id: str,
        task: str,
        source_task_name: str,
        frame_history: Sequence[Any],
    ) -> RobometerCondition:
        """Commit a pending successor without consuming another observation."""
        if not self.v72_transactional_mode:
            raise RuntimeError("Transactional V72 mode is not enabled")
        pending = self._v72_pending_transition
        if pending is None:
            raise RuntimeError("No V72 transition proposal is pending")
        if str(proposal_id) != str(pending["proposal_id"]):
            raise ValueError(
                "Stale V72 transition proposal: "
                f"expected={pending['proposal_id']!r}, got={proposal_id!r}"
            )
        state = self.planner_states[0]
        if not isinstance(state, V72CortexState):
            raise RuntimeError("V72 Cortex state was not initialized")
        if int(state.committed_idx) != int(pending["from_committed_idx"]):
            raise RuntimeError("V72 committed state changed while proposal was pending")
        if self._v72_normalize(state.current_subtask) != self._v72_normalize(
            pending["from_subtask"]
        ):
            raise RuntimeError("V72 active subtask changed while proposal was pending")
        if len(frame_history) != state.last_frame + 1:
            raise ValueError(
                "V72 commit must use the already-ingested contiguous frame history: "
                f"history={len(frame_history)}, last_frame={state.last_frame}"
            )

        previous_start = int(state.current_start_frame)
        state.completed_subtasks.append(str(pending["memory_update"]).strip())
        state.current_subtask = str(pending["candidate_subtask"]).strip()
        state.current_start_frame = int(pending["proposed_at_frame"])
        state.committed_idx = int(pending["to_committed_idx"])
        if previous_start != 0:
            state.transition_history_frames.append(previous_start)
            state.transition_history_frames = state.transition_history_frames[-4:]

        query_frame = int(state.last_frame)
        sample = self._v72_sample(
            task=task,
            source_task_name=source_task_name,
            frame_history=frame_history,
            state=state,
            query_frame=query_frame,
        )
        device = next(self.model.parameters()).device
        progress, etg = self._v72_forward_heads(sample, device)
        etg_probabilities = torch.softmax(etg.float(), dim=-1)
        generated_text = str(pending.get("generated_text", ""))
        self._v72_pending_transition = None
        return self._compact_condition(
            state=state,
            progress_logits=progress[0],
            generated_text=generated_text,
            parse_valid=True,
            parse_error="",
            planner_update_applied=True,
            done_probability=float(etg_probabilities[0, 1].item()),
            language_invoked=False,
            state_event="external_transition_commit",
        )

    def repair_v72_state(
        self, *, expected_subtask: str, expected_committed_idx: int,
        approved_current_subtask: str, approved_completed_subtasks: list[str],
        advance: bool, task: str, source_task_name: str, memory_version=None,
        frame_history: Sequence[Any],
    ) -> RobometerCondition:
        """Apply an externally evidenced semantic patch without replaying frames.

        The host owns visual completion decisions. This method checks identity
        and format only; it does not query task state or claim semantic proof.
        """
        if not self.v72_transactional_mode:
            raise RuntimeError("Semantic repair requires transactional V72")
        state = self.planner_states[0]
        if not isinstance(state, V72CortexState) or state.last_frame < 0:
            raise RuntimeError("V72 must have ingested an initial observation")
        if (state.committed_idx != expected_committed_idx or
                state.current_subtask != expected_subtask):
            raise ValueError("Stale semantic repair identity")
        if len(frame_history) != state.last_frame + 1:
            raise ValueError("Semantic repair must retain the contiguous observation cursor")
        if not isinstance(approved_current_subtask, str) or not approved_current_subtask.strip():
            raise ValueError("Approved active subtask must be nonempty")
        if (type(advance) is not bool or not isinstance(approved_completed_subtasks, list) or
                any(not isinstance(text, str) or not text.strip()
                    for text in approved_completed_subtasks)):
            raise ValueError("Explicit advance and completed-text list required")
        if (memory_version is not None and
                (type(memory_version) is not int or
                 memory_version < getattr(self, '_gpt_memory_version', 0))):
            raise ValueError("Stale GPT memory version")
        state = copy.deepcopy(state)
        previous_start = state.current_start_frame
        state.completed_subtasks = [text.strip() for text in approved_completed_subtasks]
        state.current_subtask = approved_current_subtask.strip()
        if advance:
            state.committed_idx += 1
            state.current_start_frame = state.last_frame
            if previous_start != 0:
                state.transition_history_frames.append(previous_start)
                state.transition_history_frames = state.transition_history_frames[-4:]
        sample = self._v72_sample(task=task, source_task_name=source_task_name,
            frame_history=frame_history, state=state, query_frame=state.last_frame,
            memory_version=memory_version)
        device = next(self.model.parameters()).device
        progress, etg = self._v72_forward_heads(sample, device)
        probabilities = torch.softmax(etg.float(), dim=-1)
        self.planner_states[0] = state
        if memory_version is not None:
            self._gpt_memory_version = memory_version
            self._gpt_memory_fingerprint = (
                state.committed_idx, state.current_subtask,
                tuple(state.completed_subtasks))
        self._v72_pending_transition = None
        return self._compact_condition(state=state, progress_logits=progress[0],
            generated_text="", parse_valid=True, parse_error="",
            planner_update_applied=True,
            done_probability=float(probabilities[0, 1].item()),
            language_invoked=False, state_event="external_semantic_repair")

    def sync_gpt_memory(self, *, expected_subtask, expected_committed_idx,
                        completed_subtasks, memory_version, clear_pending=False):
        """Write approved history into the real V72 prompt state, without frames/ACK."""
        state = self.planner_states[0]
        if not self.v72_transactional_mode or not isinstance(state, V72CortexState):
            raise RuntimeError("GPT memory synchronization requires transactional V72")
        if state.current_subtask != expected_subtask or state.committed_idx != expected_committed_idx:
            raise ValueError("Stale GPT memory identity")
        if (not isinstance(completed_subtasks, list) or len(completed_subtasks) != state.committed_idx
                or any(not isinstance(text, str) or not text.strip() for text in completed_subtasks)):
            raise ValueError("Completed memory must match committed stage count")
        previous_version = getattr(self, '_gpt_memory_version', 0)
        fingerprint = (state.committed_idx, state.current_subtask,
                       tuple(text.strip() for text in completed_subtasks))
        if type(memory_version) is not int or memory_version < previous_version:
            raise ValueError("Stale GPT memory version")
        if (memory_version == previous_version and
                getattr(self, '_gpt_memory_fingerprint', None) not in (None, fingerprint)):
            raise ValueError("Same GPT memory version cannot change approved memory")
        state.completed_subtasks = list(completed_subtasks)
        self._gpt_memory_version = memory_version
        self._gpt_memory_fingerprint = fingerprint
        if clear_pending:
            self._v72_pending_transition = None
        return dict(completed_subtasks=list(state.completed_subtasks),
            current_subtask=state.current_subtask, committed_idx=state.committed_idx,
            memory_version=memory_version, last_frame=state.last_frame,
            pending_transition=self.v72_pending_transition())

    def propose_after_gpt_completion(self, *, completed_subtasks, memory_version, task,
                                     source_task_name, frame_history):
        """Read-only language query on actual RGB and GPT's completed-stage list.

        V72 was trained to receive completed *previous* stages plus the current
        stage in its active slot.  GPT's canonical list already includes the
        just-completed active stage, so translate that list back to V72's
        training-time representation for this read-only successor query.  GPT's
        completion remains authoritative: the ETG head cannot veto it, and the
        deployed state/index remain unchanged until external approval.
        """
        state = self.planner_states[0]
        if not self.v72_transactional_mode or len(frame_history) != state.last_frame + 1:
            raise ValueError("Successor planning requires current contiguous V72 observations")
        if (len(completed_subtasks) != state.committed_idx + 1 or
                completed_subtasks[-1] != state.current_subtask):
            raise ValueError("Successor memory must include the entire completed active stage")
        proposed = copy.deepcopy(state)
        proposed.completed_subtasks = list(completed_subtasks[:-1])
        proposed.current_subtask = str(completed_subtasks[-1])
        # This query still describes the completed active stage. Retain its
        # real interval and transition frames; reset the start only when the
        # approved successor is committed by repair_v72_state/commit_v72_transition.
        sample = self._v72_sample(task=task, source_task_name=source_task_name,
            frame_history=frame_history, state=proposed, query_frame=proposed.last_frame,
            memory_version=memory_version)
        device = next(self.model.parameters()).device
        _, _, text, payload, error, _ = self._v72_generate(sample, device)
        candidate = None if payload is None else payload.get('current_subtask')
        memory_update = None if payload is None else payload.get('memory_update')
        successor_error = error
        if successor_error is None and (
                not isinstance(candidate, str) or not candidate.strip()):
            successor_error = 'successor query did not return a current_subtask'
        if successor_error is None and self._v72_normalize(candidate) == self._v72_normalize(
                state.current_subtask):
            successor_error = 'successor query repeated the completed current subtask'
        if successor_error is None and self._v72_normalize(memory_update) != self._v72_normalize(
                state.current_subtask):
            successor_error = 'successor query did not acknowledge the completed current subtask'
        return dict(proposal_id=f"gptdone-f{state.last_frame}-i{state.committed_idx}",
            from_committed_idx=state.committed_idx, from_subtask=state.current_subtask,
            candidate_subtask=candidate, memory_update=memory_update,
            generated_text=text, parse_error=successor_error,
            parse_valid=successor_error is None,
            completed_input=list(completed_subtasks),
            model_completed_input=list(proposed.completed_subtasks),
            model_active_input=proposed.current_subtask, source_frame=state.last_frame,
            deployed_state_unchanged=True, input_receipt=self._v72_last_input_receipt)

    def _to_progress_sample(
        self,
        task: str,
        frames: torch.Tensor,
        states: torch.Tensor,
        history_indices: torch.Tensor,
        planner_state: AbsoluteOrdinalState | CompactCausalState,
    ) -> ProgressSample:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Robometer frames must be [K,3,H,W], got {tuple(frames.shape)}"
            )
        frame_count = int(frames.shape[0])
        if not 1 <= frame_count <= 8:
            raise ValueError(f"Robometer requires 1..8 real frames, got {frame_count}.")
        if states.shape != (frame_count, 14):
            raise ValueError(
                f"Robometer states must be [K,14], got {tuple(states.shape)}"
            )
        if history_indices.shape != (frame_count,):
            raise ValueError(
                "Robometer history indices must align with frames, got "
                f"{tuple(history_indices.shape)} for K={frame_count}."
            )

        frames_np = (
            frames.detach()
            .to(device="cpu", dtype=torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        states_np = states.detach().to(device="cpu", dtype=torch.float32).numpy()
        indices = history_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        if self.compact_v56_protocol:
            if frame_count != 8:
                raise ValueError(
                    "Online V56 compact inference requires exactly eight images"
                )
            progress_slots = list(
                UNIFORM16S_THEN_GAP6S_DISJOINT_RANDOM4S3_SUPERVISION_SLOTS
            )
        elif self.frame_sampling == "preprogress":
            if frame_count != 8:
                raise ValueError(
                    "Online preprogress requires exactly eight sampled observations"
                )
            progress_slots = list(PREPROGRESS_SUPERVISION_SLOTS)
        else:
            progress_slots = list(range(frame_count))
        if self.compact_v56_protocol:
            assert isinstance(planner_state, CompactCausalState)
            frozen = True
            strategy = "subtask_forward"
        else:
            assert isinstance(planner_state, AbsoluteOrdinalState)
            frozen = planner_state.plan_frozen
            strategy = "subtask_forward" if frozen else "subtask_plan_refinement"
        trajectory = Trajectory(
            frames=frames_np,
            frames_shape=tuple(frames_np.shape),
            robot_states=(states_np if self.use_robot_state else None),
            robot_state_schema=(ROBOT_STATE_SCHEMA if self.use_robot_state else None),
            task=str(task),
            target_progress=[0.0] * len(progress_slots),
            predict_last_frame_mask=[1.0] * len(progress_slots),
            frame_indices=indices,
            data_source="rmbench",
            quality_label="successful",
            data_gen_strategy=strategy,
            metadata={
                "source": "fastwam_tmd2_online",
                "temporal_strategy": "forward" if frozen else "plan_refinement",
                "frame_sampling": self.frame_sampling,
                "progress_supervision_slots": progress_slots,
                "progress_target_space": self.progress_target_space,
            },
        )
        if self.compact_v56_protocol:
            assert isinstance(planner_state, CompactCausalState)
            return ProgressSample(
                trajectory=trajectory,
                sample_type="progress",
                data_gen_strategy=strategy,
                completed_subtasks=list(planner_state.completed_subtasks),
                active_subtask=planner_state.current_subtask,
                plan_subtasks=[],
                prompt_plan_subtasks=[],
                execution_history=[],
                target_done=None,
                target_action=None,
                target_instruction=None,
                target_subtask_idx=None,
                query_frame_idx=int(indices[-1]),
                active_subtask_idx=None,
                planner_plan_source="model_commit",
                sample_role=(
                    "causal_initial_plan"
                    if planner_state.current_subtask is None
                    else "causal_completion_plan"
                ),
                planner_bootstrap=False,
                planner_protocol=self.planner_protocol,
                planner_plan_frozen=True,
                planner_forward_only=True,
            )
        assert isinstance(planner_state, AbsoluteOrdinalState)
        return ProgressSample(
            trajectory=trajectory,
            sample_type="progress",
            data_gen_strategy=strategy,
            completed_subtasks=[],
            active_subtask=None,
            plan_subtasks=planner_state.prompt_plan,
            prompt_plan_subtasks=planner_state.prompt_plan,
            execution_history=planner_state.prompt_history,
            target_done=None,
            target_action=None,
            target_instruction=None,
            target_subtask_idx=None,
            query_frame_idx=int(indices[-1]),
            active_subtask_idx=None,
            planner_plan_source=("model_commit" if frozen else "uninitialized"),
            sample_role="frozen_state" if frozen else "plan_refinement",
            planner_bootstrap=not frozen,
            planner_protocol="absolute_ordinal",
            planner_plan_frozen=frozen,
        )

    @staticmethod
    def _v72_normalize(value: object) -> str:
        text = " ".join(str(value or "").casefold().replace("_", " ").split())
        return "task complete" if text in {"task complete", "task completed"} else text

    @staticmethod
    def _v72_parse_json(text: str) -> tuple[dict | None, str | None, bool]:
        fields = ("current_subtask", "memory_update")
        stripped = text.strip()
        decoder = json.JSONDecoder()
        candidates = [0] + [
            index for index, character in enumerate(stripped) if character == "{"
        ]
        payload = None
        parse_error = None
        for start in dict.fromkeys(candidates):
            try:
                candidate, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            return None, parse_error or "no JSON object", False
        key_order_valid = tuple(payload.keys()) == fields
        if set(payload) != set(fields):
            return (
                payload,
                f"expected fields {fields}, got {tuple(payload)}",
                key_order_valid,
            )
        for field_name in fields:
            if field_name == "memory_update" and payload[field_name] is None:
                continue
            value = payload[field_name]
            if not isinstance(value, str) or not value.strip():
                return (
                    payload,
                    f"{field_name} is not a non-empty string",
                    key_order_valid,
                )
            payload[field_name] = value.strip()
        return payload, None, key_order_valid

    def _v72_sample(
        self,
        *,
        task: str,
        source_task_name: str,
        frame_history: Sequence[Any],
        state: V72CortexState,
        query_frame: int,
        memory_version=None,
    ) -> ProgressSample:
        indices, frame_valid, progress_valid = (
            taskstart1_transition4_subtask3_layout(
                query_frame,
                int(state.current_start_frame),
                state.transition_history_frames,
                preserve_fixed_first_with_masked_padding=True,
            )
        )
        selected = []
        for index in indices:
            frame = frame_history[int(index)]
            if torch.is_tensor(frame):
                frame = (
                    frame.detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .numpy()
                )
            selected.append(np.asarray(frame, dtype=np.uint8))
        frames_np = np.stack(selected)
        strategy = "subtask_forward"
        trajectory = Trajectory(
            frames=frames_np,
            frames_shape=tuple(frames_np.shape),
            robot_states=None,
            robot_state_schema=None,
            id=f"{source_task_name}/online/t_{query_frame}",
            task=str(task),
            data_source="rmbench",
            quality_label="successful",
            target_progress=[0.0]
            * len(TASKSTART1_TRANSITION4_SUBTASK3_SUPERVISION_SLOTS),
            predict_last_frame_mask=[float(value) for value in progress_valid],
            metadata={
                "temporal_strategy": "forward",
                "frame_sampling": self.frame_sampling,
                "progress_supervision_slots": list(
                    TASKSTART1_TRANSITION4_SUBTASK3_SUPERVISION_SLOTS
                ),
                "progress_target_space": self.progress_target_space,
                "frame_valid_mask": frame_valid,
                "progress_target_valid_mask": progress_valid,
                "transition_history_frames": list(
                    state.transition_history_frames
                ),
                "external_task_termination": source_task_name
                in {"battery_try", "blocks_ranking_try"},
                "current_subtask_start_frame": int(state.current_start_frame),
            },
            data_gen_strategy=strategy,
            frame_indices=indices,
        )
        self._v72_last_input_receipt = dict(query_frame=int(query_frame),
            current_subtask=state.current_subtask,
            completed_subtasks=list(state.completed_subtasks),
            current_subtask_start_frame=int(state.current_start_frame),
            transition_history_frames=list(state.transition_history_frames),
            frame_indices=list(indices), frame_valid_mask=list(frame_valid),
            memory_version=(getattr(self, '_gpt_memory_version', 0)
                            if memory_version is None else memory_version),
            memory_sha256=hashlib.sha256(json.dumps(state.completed_subtasks,
                ensure_ascii=False).encode()).hexdigest())
        return ProgressSample(
            trajectory=trajectory,
            sample_type="progress",
            data_gen_strategy=strategy,
            completed_subtasks=list(state.completed_subtasks),
            active_subtask=state.current_subtask,
            plan_subtasks=[],
            prompt_plan_subtasks=[],
            execution_history=[],
            target_done=None,
            target_action=None,
            target_instruction=None,
            causal_language_schema="current_subtask_memory_update",
            target_subtask_idx=None,
            query_frame_idx=int(query_frame),
            active_subtask_idx=None,
            planner_plan_source="model_commit",
            sample_role=(
                "cortex_initial"
                if query_frame == 0
                and state.current_subtask is None
                and not state.completed_subtasks
                else "cortex_stable"
            ),
            planner_bootstrap=False,
            planner_protocol=self.planner_protocol,
            planner_plan_frozen=True,
            planner_forward_only=True,
        )

    def _v72_extract_heads(self, output) -> tuple[torch.Tensor, torch.Tensor]:
        progress = output.progress_logits.float()[:, -1].detach().cpu()
        etg = output.etg_decision_logits
        if etg is None:
            raise RuntimeError("V72 online inference requires causal ETG logits")
        etg = etg.float().detach().cpu()
        if tuple(progress.shape) != (1, 10):
            raise ValueError(f"Expected V72 C51 logits [1,10], got {progress.shape}")
        if tuple(etg.shape) != (1, 2):
            raise ValueError(f"Expected V72 ETG logits [1,2], got {etg.shape}")
        return progress, etg

    def _v72_forward_heads(
        self,
        sample: ProgressSample,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(**self._compact_inputs([sample], device))
        return self._v72_extract_heads(output)

    def _v72_generate(
        self,
        sample: ProgressSample,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, str, dict | None, str | None, bool]:
        inputs = self._compact_inputs([sample], device)
        prompt_length = int(inputs["input_ids"].shape[1])
        output = self.model.generate_with_progress(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated_ids = output.sequences[:, prompt_length:]
        text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        payload, error, key_order = self._v72_parse_json(text)
        progress, etg = self._v72_extract_heads(output)
        return progress, etg, text, payload, error, key_order

    def _forward_v72(
        self,
        *,
        task: str,
        source_task_name: str,
        frame_history: Sequence[Any],
        query_frame: int,
    ) -> RobometerCondition:
        state = self.planner_states[0]
        if not isinstance(state, V72CortexState):
            raise RuntimeError("V72 Cortex state was not initialized")
        if query_frame != state.last_frame + 1:
            raise RuntimeError(
                "V72 numeric inference must consume every source frame in order: "
                f"previous={state.last_frame}, current={query_frame}."
            )
        if len(frame_history) != query_frame + 1:
            raise ValueError(
                "V72 online frame history must end at the current query: "
                f"history={len(frame_history)}, query={query_frame}."
            )
        state.last_frame = int(query_frame)
        device = next(self.model.parameters()).device
        sample = self._v72_sample(
            task=task,
            source_task_name=source_task_name,
            frame_history=frame_history,
            state=state,
            query_frame=query_frame,
        )
        is_query = query_frame % self.planner_query_stride == 0
        language_invoked = bool(
            is_query
            and self._v72_normalize(state.current_subtask) != "task complete"
        )
        generated_text = ""
        parse_valid = True
        parse_error = ""
        planner_update_applied = False
        state_event = (
            "terminal_frozen"
            if self._v72_normalize(state.current_subtask) == "task complete"
            else "nonquery"
        )
        if language_invoked and (
            self.v72_transactional_mode
            and self._v72_pending_transition is not None
        ):
            # A semantic successor is awaiting the external arbiter. Continue
            # consuming the numeric stream, but do not generate/commit another
            # successor from a later frame and do not rewind the frame cursor.
            progress, etg = self._v72_forward_heads(sample, device)
            language_invoked = False
            state_event = "pending_transition_hold"
        elif language_invoked:
            (
                progress,
                etg,
                generated_text,
                payload,
                raw_error,
                key_order_valid,
            ) = self._v72_generate(sample, device)
            del key_order_valid
            etg_decision = "advance" if int(etg.argmax(dim=-1).item()) == 1 else "keep"
            deployed = None if payload is None else dict(payload)
            deployed_error = raw_error
            current_before = state.current_subtask
            if current_before is not None and etg_decision == "keep":
                deployed = {
                    "current_subtask": current_before,
                    "memory_update": None,
                }
                deployed_error = None
            elif current_before is not None and etg_decision == "advance":
                predicted = (
                    None if deployed is None else deployed.get("current_subtask")
                )
                if (
                    deployed is None
                    or deployed_error is not None
                    or self._v72_normalize(predicted)
                    == self._v72_normalize(current_before)
                ):
                    deployed_error = (
                        "causal ETG predicted advance but successor language was invalid"
                    )

            predicted = (
                None if deployed is None else deployed.get("current_subtask")
            )
            memory_update = (
                None if deployed is None else deployed.get("memory_update")
            )
            valid = deployed_error is None and bool(predicted)
            if current_before is None:
                if valid:
                    state.current_subtask = str(predicted)
                    state.current_start_frame = int(query_frame)
                    state.committed_idx = 0
                    state_event = "initialize"
                    planner_update_applied = True
                else:
                    state_event = "initial_parse_error"
            elif not valid:
                state_event = "dense_parse_error"
            elif self._v72_normalize(predicted) == self._v72_normalize(
                current_before
            ):
                state_event = (
                    "dense_language_keep_with_invalid_memory"
                    if memory_update
                    else "dense_language_keep"
                )
            elif (
                not memory_update
                or self._v72_normalize(memory_update)
                != self._v72_normalize(current_before)
            ):
                state_event = "dense_language_change_invalid_memory"
            elif self.v72_transactional_mode:
                proposal_id = (
                    f"v72-f{int(query_frame):06d}-"
                    f"i{int(state.committed_idx):03d}-"
                    f"to{int(state.committed_idx) + 1:03d}"
                )
                self._v72_pending_transition = {
                    "schema": "v72.transition_candidate.v1",
                    "proposal_id": proposal_id,
                    "proposed_at_frame": int(query_frame),
                    "from_committed_idx": int(state.committed_idx),
                    "to_committed_idx": int(state.committed_idx) + 1,
                    "from_subtask": str(current_before),
                    "candidate_subtask": str(predicted).strip(),
                    "memory_update": str(memory_update).strip(),
                    "generated_text": generated_text,
                    "pre_update_done_probability": float(
                        torch.softmax(etg.float(), dim=-1)[0, 1].item()
                    ),
                }
                state_event = "transition_proposed"
            else:
                previous_start = int(state.current_start_frame)
                state.completed_subtasks.append(str(memory_update).strip())
                state.current_subtask = str(predicted).strip()
                state.current_start_frame = int(query_frame)
                state.committed_idx += 1
                if previous_start != 0:
                    state.transition_history_frames.append(previous_start)
                    state.transition_history_frames = (
                        state.transition_history_frames[-4:]
                    )
                state_event = "dense_language_advance"
                planner_update_applied = True
            parse_valid = deployed_error is None
            parse_error = "" if deployed_error is None else str(deployed_error)
        else:
            progress, etg = self._v72_forward_heads(sample, device)

        if state.current_subtask is None:
            raise RuntimeError(
                "V72 initialization did not produce a deployable current subtask: "
                f"error={parse_error!r}, raw={generated_text!r}."
            )
        if planner_update_applied:
            aligned_sample = self._v72_sample(
                task=task,
                source_task_name=source_task_name,
                frame_history=frame_history,
                state=state,
                query_frame=query_frame,
            )
            progress, _aligned_etg = self._v72_forward_heads(
                aligned_sample,
                device,
            )
        etg_probabilities = torch.softmax(etg.float(), dim=-1)
        return self._compact_condition(
            state=state,
            progress_logits=progress[0],
            generated_text=generated_text,
            parse_valid=parse_valid,
            parse_error=parse_error,
            planner_update_applied=planner_update_applied,
            done_probability=float(etg_probabilities[0, 1].item()),
            language_invoked=language_invoked,
            state_event=state_event,
        )

    def _compact_condition(
        self,
        *,
        state: CompactCausalState,
        progress_logits: torch.Tensor,
        generated_text: str,
        parse_valid: bool,
        parse_error: str,
        planner_update_applied: bool,
        done_probability: float,
        language_invoked: bool,
        state_event: str,
    ) -> RobometerCondition:
        if state.current_subtask is None:
            raise RuntimeError("V56 compact planner has no deployed subtask")
        token_ids = np.asarray(
            self.tokenizer.encode(
                state.current_subtask,
                add_special_tokens=False,
            ),
            dtype=np.int64,
        )
        if token_ids.size == 0:
            raise RuntimeError("V56 compact planner produced an empty subtask")
        logits = progress_logits.float().reshape(-1).cpu()
        if tuple(logits.shape) != (10,):
            raise ValueError(f"Expected ten C51 logits, got {tuple(logits.shape)}")
        probabilities = torch.softmax(logits, dim=-1)
        centers = torch.linspace(0.0, 1.0, 10, dtype=probabilities.dtype)
        local_progress = float((probabilities * centers).sum().item())
        progress_bin = int(probabilities.argmax().item())
        empty_hidden = torch.empty(0, self.hidden_dim, dtype=torch.bfloat16)
        empty_layers = torch.empty(
            self.num_layers, 0, self.hidden_dim, dtype=torch.bfloat16
        )
        return RobometerCondition(
            memory_tokens=empty_layers,
            memory_mask=torch.empty(0, dtype=torch.bool),
            current_progress_tokens=empty_layers,
            current_progress_mask=torch.empty(0, dtype=torch.bool),
            subtask_tokens=empty_hidden,
            subtask_layer_tokens=None,
            subtask_mask=torch.empty(0, dtype=torch.bool),
            instruction_tokens=empty_layers,
            instruction_mask=torch.empty(0, dtype=torch.bool),
            subtask_token_ids=torch.from_numpy(token_ids),
            sequences=torch.from_numpy(token_ids.copy()),
            decoded_active_idx=int(state.committed_idx),
            planner_active_idx=int(state.committed_idx),
            planner_plan_frozen=True,
            planner_committed_idx=int(state.committed_idx),
            current_subtask=state.current_subtask,
            plan=(),
            execution_history=tuple(state.completed_subtasks),
            generated_text=generated_text,
            parse_valid=parse_valid,
            parse_error=parse_error,
            planner_repair="",
            accepted_incomplete_plan=False,
            planner_update_applied=planner_update_applied,
            planner_committed_now=planner_update_applied,
            local_progress=local_progress,
            global_progress=local_progress,
            progress_bin=progress_bin,
            progress_confidence=float(probabilities[progress_bin].item()),
            progress_logits=logits,
            done_probability=float(done_probability),
            language_invoked=bool(language_invoked),
            state_event=str(state_event),
        )

    def _forward_compact_v56(
        self,
        samples: list[ProgressSample],
        history_indices: Sequence[torch.Tensor],
    ) -> RobometerCondition:
        """Reproduce the cache builder's V53 two-pass deployed protocol."""
        if len(samples) != 1:
            raise ValueError("V56 compact online inference requires batch size one")
        state = self.planner_states[0]
        if not isinstance(state, CompactCausalState):
            raise RuntimeError("V56 compact state was not initialized")
        device = next(self.model.parameters()).device
        generated_text = ""
        parse_valid = True
        parse_error = ""
        planner_update_applied = False
        language_invoked = False
        state_event = "keep"
        query_frame = int(history_indices[0][-1].item())

        if state.current_subtask is None:
            progress, _success, texts, parsed = self._compact_run_generation(
                samples,
                device,
            )
            generated_text = texts[0]
            language_invoked = True
            payload, error, _key_order = parsed[0]
            if error is not None or payload is None:
                raise RuntimeError(
                    "V56 initialization parse failure: "
                    f"error={error}; raw={generated_text!r}"
                )
            state.current_subtask = str(payload["subtask"])
            state.current_start_frame = query_frame
            state.committed_idx = 0
            planner_update_applied = True
            state_event = "initialize"
            progress_logits = progress[0]
            done_probability = float(_success[0])
        else:
            progress, success = self._compact_run_forward(
                samples,
                device,
            )
            progress_logits = progress[0]
            done_probability = float(success[0])
            is_query = query_frame % self.planner_query_stride == 0
            should_generate = (
                is_query
                and state.current_subtask != TASK_COMPLETE
                and float(success[0]) >= 0.5
            )
            if should_generate:
                _generated_progress, _generated_success, texts, parsed = (
                    self._compact_run_generation(
                    samples,
                    device,
                    )
                )
                # V56's deployed event-gated validator treats the first
                # numeric readout as the sole completion decision. The second
                # prefill only obtains language; it must not re-gate or
                # overwrite that query's C51/success snapshot.
                generated_text = texts[0]
                language_invoked = True
                payload, error, _key_order = parsed[0]
                parse_valid = error is None and payload is not None
                parse_error = "" if error is None else str(error)
                if parse_valid:
                    state.completed_subtasks.append(state.current_subtask)
                    state.current_subtask = str(payload["subtask"])
                    state.current_start_frame = query_frame
                    state.committed_idx += 1
                    planner_update_applied = True
                    state_event = "advance"
                else:
                    state_event = "done_without_valid_next"
            elif state.current_subtask == TASK_COMPLETE:
                state_event = "terminal_frozen"

        return self._compact_condition(
            state=state,
            progress_logits=progress_logits,
            generated_text=generated_text,
            parse_valid=parse_valid,
            parse_error=parse_error,
            planner_update_applied=planner_update_applied,
            done_probability=done_probability,
            language_invoked=language_invoked,
            state_event=state_event,
        )

    def _compact_inputs(
        self,
        samples: list[ProgressSample],
        device: torch.device,
    ) -> dict:
        batch = self.collator(samples)["progress_inputs"]
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key in _V56_FORWARD_KEYS and value is not None
        }

    @staticmethod
    def _compact_extract_heads(output) -> tuple[torch.Tensor, torch.Tensor]:
        progress = output.progress_logits.float()[:, -1].detach().cpu()
        success_logits = output.done_success_component_logits
        if success_logits is None:
            success_logits = output.done_logits
        if success_logits is None:
            raise RuntimeError("V56 compact inference requires a success head")
        success = torch.sigmoid(success_logits.float())[:, -1].detach().cpu()
        if progress.ndim != 2 or tuple(progress.shape[1:]) != (10,):
            raise ValueError(
                f"Expected final C51 logits [B,10], got {tuple(progress.shape)}"
            )
        return progress, success

    @staticmethod
    def _compact_parse_json(text: str) -> tuple[dict | None, str | None, bool]:
        stripped = text.strip()
        decoder = json.JSONDecoder()
        starts = [0] + [
            index for index, character in enumerate(stripped) if character == "{"
        ]
        payload = None
        parse_error = None
        for start in dict.fromkeys(starts):
            try:
                candidate, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            return None, parse_error or "no JSON object", False
        key_order_valid = tuple(payload.keys()) == ("subtask",)
        if set(payload) != {"subtask"}:
            return (
                payload,
                f"expected fields ('subtask',), got {tuple(payload)}",
                key_order_valid,
            )
        value = payload["subtask"]
        if not isinstance(value, str) or not value.strip():
            return payload, "subtask is not a non-empty string", key_order_valid
        payload["subtask"] = value.strip()
        return payload, None, key_order_valid

    def _compact_run_forward(
        self,
        samples: list[ProgressSample],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(**self._compact_inputs(samples, device))
        return self._compact_extract_heads(output)

    def _compact_run_generation(
        self,
        samples: list[ProgressSample],
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[str],
        list[tuple[dict | None, str | None, bool]],
    ]:
        inputs = self._compact_inputs(samples, device)
        prompt_length = int(inputs["input_ids"].shape[1])
        output = self.model.generate_with_progress(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated_ids = output.sequences[:, prompt_length:]
        texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        parsed = [self._compact_parse_json(text) for text in texts]
        progress, success = self._compact_extract_heads(output)
        return progress, success, texts, parsed

    @torch.inference_mode()
    def forward(
        self,
        tasks: Sequence[str],
        frames: Sequence[Any],
        states: Sequence[Any],
        history_indices: Sequence[torch.Tensor],
        source_task_names: Sequence[str] | None = None,
    ) -> RobometerCondition:
        # These are the exact functions used to create and augment the 1.1 TB
        # training cache. Import lazily so ordinary TMD2 training does not load
        # the cache-build CLI module in every data worker.
        from scripts.build_rmbench_progress_cache import decode_conditions

        batch_size = len(tasks)
        if not (
            len(frames) == len(states) == len(history_indices) == batch_size
        ):
            raise ValueError("Robometer task/history batch lengths do not match.")
        if batch_size != 1:
            raise ValueError(
                "Online RMBench rollout currently requires batch size 1 so each "
                "episode owns an independent transactional planner."
            )
        if len(self.planner_states) != batch_size:
            raise RuntimeError(
                "Planner batch does not match inputs; call conditioner.reset(batch_size)."
            )
        if self.v72_cortex_protocol:
            if source_task_names is None or len(source_task_names) != batch_size:
                raise ValueError(
                    "V72 online inference requires one RMBench source task name."
                )
            if batch_size != 1:
                raise ValueError("V72 online inference requires batch size one")
            return self._forward_v72(
                task=str(tasks[0]),
                source_task_name=str(source_task_names[0]),
                frame_history=frames[0],
                query_frame=int(history_indices[0][-1].item()),
            )
        samples = [
            self._to_progress_sample(task, frame, state, indices, planner_state)
            for task, frame, state, indices, planner_state in zip(
                tasks,
                frames,
                states,
                history_indices,
                self.planner_states,
                strict=True,
            )
        ]
        if self.compact_v56_protocol:
            return self._forward_compact_v56(samples, history_indices)
        device = next(self.model.parameters()).device
        decoded = decode_conditions(
            self.model,
            self.tokenizer,
            self.collator,
            samples,
            self.planner_states,
            device,
            self.num_layers,
            self.max_new_tokens,
            capture_selected_prompt_memory=self.use_robot_state,
            capture_subtask_layers=self.capture_subtask_layers,
            capture_prompt_state_layers=self.use_robot_state,
            capture_instruction_layers=self.use_robot_state,
            capture_prompt_progress_layers=self.use_robot_state,
            capture_progress_logits=True,
        )

        generation_state_layers, condition = decoded[0]
        if self.use_robot_state:
            state_layers = condition["current_state_layers"]
            current_progress_layers = condition["current_progress_layers"]
            instruction_layers = condition["instruction_layers"]
            if not torch.equal(state_layers, generation_state_layers):
                raise RuntimeError(
                    "Online Progress state memory is not the generation prefill state."
                )
        else:
            empty = torch.empty(
                self.num_layers,
                0,
                self.hidden_dim,
                dtype=torch.bfloat16,
            )
            state_layers = empty
            current_progress_layers = empty
            instruction_layers = empty

        final_subtask = condition["final_hidden"].to(torch.bfloat16)
        subtask_layer_tokens = (
            condition["subtask_layer_hidden"].to(torch.bfloat16)
            if self.capture_subtask_layers
            else None
        )
        if self.capture_subtask_layers and subtask_layer_tokens is None:
            raise RuntimeError(
                "Online Progress generation did not capture layerwise subtask tokens."
            )
        memory_tokens = torch.cat(
            [
                state_layers.to(torch.bfloat16),
                instruction_layers.to(torch.bfloat16),
            ],
            dim=1,
        )
        memory_mask = torch.ones(memory_tokens.shape[1], dtype=torch.bool)
        subtask_mask = torch.ones(final_subtask.shape[0], dtype=torch.bool)
        instruction_mask = torch.ones(instruction_layers.shape[1], dtype=torch.bool)
        token_ids = np.concatenate(
            [condition["subtask_token_ids"], condition["plan_token_ids"]]
        )
        planner_state = self.planner_states[0]
        progress_logits = condition["progress_logits"].float()
        progress_probs = torch.softmax(progress_logits, dim=-1)
        progress_centers = torch.linspace(
            0.0,
            1.0,
            progress_probs.shape[-1],
            dtype=progress_probs.dtype,
        )
        local_progress = float((progress_probs * progress_centers).sum().item())
        global_progress = (
            planner_state.compose_progress(local_progress)
            if planner_state.active_idx is not None
            and (planner_state.frozen_plan or planner_state.candidate_plan)
            else local_progress
        )
        progress_bin = int(progress_probs.argmax().item())
        return RobometerCondition(
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            current_progress_tokens=current_progress_layers.to(torch.bfloat16),
            current_progress_mask=torch.ones(
                current_progress_layers.shape[1], dtype=torch.bool
            ),
            subtask_tokens=final_subtask,
            subtask_layer_tokens=subtask_layer_tokens,
            subtask_mask=subtask_mask,
            instruction_tokens=instruction_layers,
            instruction_mask=instruction_mask,
            subtask_token_ids=torch.from_numpy(
                np.asarray(condition["subtask_token_ids"], dtype=np.int64)
            ),
            sequences=torch.from_numpy(token_ids.astype(np.int64, copy=False)),
            decoded_active_idx=int(condition["decoded_active_idx"]),
            planner_active_idx=(
                -1 if planner_state.active_idx is None else int(planner_state.active_idx)
            ),
            planner_plan_frozen=bool(planner_state.plan_frozen),
            planner_committed_idx=(
                -1
                if planner_state.committed_active_idx is None
                else int(planner_state.committed_active_idx)
            ),
            current_subtask=str(planner_state.active_instruction or ""),
            plan=tuple(planner_state.frozen_plan or planner_state.candidate_plan or ()),
            execution_history=tuple(planner_state.execution_history or ()),
            generated_text=str(condition["generated_text"]),
            parse_valid=bool(condition["parse_valid"]),
            parse_error=str(condition["parse_error"]),
            planner_repair=str(condition["planner_repair"]),
            accepted_incomplete_plan=bool(condition["accepted_incomplete_plan"]),
            planner_update_applied=bool(condition["planner_update_applied"]),
            planner_committed_now=bool(condition["planner_committed_now"]),
            local_progress=local_progress,
            global_progress=float(global_progress),
            progress_bin=progress_bin,
            progress_confidence=float(progress_probs[progress_bin].item()),
            progress_logits=progress_logits.cpu(),
            done_probability=float("nan"),
            language_invoked=True,
            state_event=(
                "advance" if bool(condition["planner_update_applied"]) else "keep"
            ),
        )
