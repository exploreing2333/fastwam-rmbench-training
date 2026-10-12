from __future__ import annotations

import argparse
import atexit
from dataclasses import fields
import os
from multiprocessing.connection import Client, Listener
from pathlib import Path
import subprocess
import threading
import time
import uuid
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from TMD2.robometer_conditioner import (
    FrozenRobometerConditioner,
    RobometerCondition,
)


ROBOMETER_RUNTIME = Path(
    "/home/cenzl/conda_env/robometer_runtime/bin/python"
)
AUTHKEY = b"fastwam-v72-progress"


def _serialize_tensor(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    dtype = str(tensor.dtype).removeprefix("torch.")
    if tensor.dtype == torch.bfloat16:
        array = tensor.float().reshape(-1).numpy()
    else:
        array = tensor.reshape(-1).numpy()
    return {
        "shape": tuple(tensor.shape),
        "dtype": dtype,
        "array": array,
    }


def _deserialize_tensor(payload: dict[str, Any]) -> torch.Tensor:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype_name = str(payload["dtype"])
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported IPC tensor dtype: {dtype_name}")
    return (
        torch.from_numpy(np.asarray(payload["array"]))
        .to(dtype=dtypes[dtype_name])
        .reshape(tuple(payload["shape"]))
        .contiguous()
    )


def _serialize_condition(condition: RobometerCondition) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(RobometerCondition):
        value = getattr(condition, item.name)
        result[item.name] = (
            {"tensor": _serialize_tensor(value)}
            if torch.is_tensor(value)
            else value
        )
    return result


def _deserialize_condition(payload: dict[str, Any]) -> RobometerCondition:
    values = {
        key: (
            _deserialize_tensor(value["tensor"])
            if isinstance(value, dict) and set(value) == {"tensor"}
            else value
        )
        for key, value in payload.items()
    }
    return RobometerCondition(**values)


class V72SubprocessRobometerConditioner(nn.Module):
    """Run V72 in its pinned Torch/FlashAttention runtime on the second GPU."""

    compact_v56_protocol = False
    v72_cortex_protocol = True
    compact_online_protocol = True
    planner_query_stride = 8
    keyframe_stride = 30
    frame_sampling = "taskstart1_transition4_subtask3"
    preprogress_history_stride = 4

    def __init__(
        self,
        *,
        checkpoint_path: str,
        run_config_path: str,
        max_new_tokens: int,
        num_layers: int,
        transition_confirm_observations: int,
        capture_subtask_layers: bool,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not ROBOMETER_RUNTIME.is_file():
            raise FileNotFoundError(
                f"Pinned Robometer runtime is missing: {ROBOMETER_RUNTIME}"
            )
        if str(dtype) != "torch.bfloat16":
            raise ValueError("V72 deployed inference requires bfloat16.")
        self.device = str(device)
        self.dtype = dtype
        self._lock = threading.Lock()
        self._closed = False
        self._last_pending_transition: dict[str, Any] | None = None
        self._socket_path = Path(
            f"/tmp/fw_v72_{os.getpid()}_{uuid.uuid4().hex[:8]}.sock"
        )
        env = os.environ.copy()
        env.update(
            {
                "ROBOMETER_REQUIRE_FLASH_ATTN": "1",
                "USE_TF": "0",
                "USE_FLAX": "0",
                "OMP_NUM_THREADS": "2",
                # Match the promoted V72 rollout runtime exactly. Its local
                # Qwen processor/base-model snapshot lives in this cache.
                "HF_HOME": "/home/cenzl/conda_env/huggingface",
                "HF_HUB_CACHE": "/home/cenzl/conda_env/huggingface/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONNOUSERSITE": "1",
            }
        )
        # The promoted V72 launcher intentionally does not set the deprecated
        # TRANSFORMERS_CACHE variable.  If inherited from FastWAM it takes
        # precedence over HF_HOME and hides the pinned local model snapshot.
        env.pop("TRANSFORMERS_CACHE", None)
        self._process = subprocess.Popen(
            [
                str(ROBOMETER_RUNTIME),
                "-m",
                "TMD2.robometer_subprocess",
                "--worker",
                str(self._socket_path),
            ],
            env=env,
        )
        # Importing the pinned Robometer runtime can take longer than one
        # minute when the host has just loaded the multi-billion-parameter
        # FastWAM checkpoint and its page cache is cold.  The worker has not
        # touched the simulator at this point, so waiting longer is safe and
        # avoids misclassifying slow process startup as an IPC failure.
        startup_timeout = float(os.environ.get("FASTWAM_V72_IPC_START_TIMEOUT", "300"))
        deadline = time.monotonic() + startup_timeout
        while not self._socket_path.exists():
            returncode = self._process.poll()
            if returncode is not None:
                raise RuntimeError(
                    "V72 Robometer worker exited before opening IPC: "
                    f"returncode={returncode}"
                )
            if time.monotonic() >= deadline:
                self._process.terminate()
                raise TimeoutError(
                    "Timed out waiting for V72 Robometer IPC socket "
                    f"after {startup_timeout:.1f}s."
                )
            time.sleep(0.1)
        self._connection = Client(str(self._socket_path), authkey=AUTHKEY)
        ready = self._request(
            {
                "op": "initialize",
                "checkpoint_path": str(checkpoint_path),
                "run_config_path": str(run_config_path),
                "max_new_tokens": int(max_new_tokens),
                "num_layers": int(num_layers),
                "transition_confirm_observations": int(
                    transition_confirm_observations
                ),
                "capture_subtask_layers": bool(capture_subtask_layers),
                "device": self.device,
            },
            timeout=900.0,
        )
        if ready.get("attention_backend") != "flash_attention_2":
            self.close()
            raise RuntimeError(
                "V72 worker did not activate FlashAttention 2: "
                f"{ready.get('attention_backend')!r}"
            )
        self.hidden_dim = int(ready["hidden_dim"])
        atexit.register(self.close)

    def _request(
        self,
        request: dict[str, Any],
        *,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("V72 Robometer worker is closed.")
            self._connection.send(request)
            if not self._connection.poll(timeout):
                returncode = self._process.poll()
                raise TimeoutError(
                    "V72 Robometer worker did not respond: "
                    f"op={request.get('op')}, returncode={returncode}"
                )
            reply = self._connection.recv()
        if not bool(reply.get("ok", False)):
            raise RuntimeError(
                "V72 Robometer worker failed: "
                f"op={request.get('op')}, error={reply.get('error')}"
            )
        return reply

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        return self

    def to(self, *args, **kwargs):
        requested_device = kwargs.get("device")
        requested_dtype = kwargs.get("dtype")
        if requested_device is not None and str(requested_device) != self.device:
            raise ValueError(
                f"V72 worker device is fixed at {self.device}, got {requested_device}."
            )
        if requested_dtype is not None and requested_dtype != self.dtype:
            raise ValueError(
                f"V72 worker dtype is fixed at {self.dtype}, got {requested_dtype}."
            )
        return self

    def reset(self, batch_size: int = 1) -> None:
        if int(batch_size) != 1:
            raise ValueError("V72 deployed worker requires batch size one.")
        self._request({"op": "reset"})
        self._last_pending_transition = None

    def enable_transactional_v72(self) -> None:
        self._request({"op": "enable_transactional_v72"})
        self._last_pending_transition = None

    @property
    def pending_transition(self) -> dict[str, Any] | None:
        pending = self._last_pending_transition
        return None if pending is None else dict(pending)

    def commit_pending_transition(
        self,
        *,
        proposal_id: str,
        instruction: str,
        source_task_name: str,
    ) -> RobometerCondition:
        reply = self._request(
            {
                "op": "commit_transition",
                "proposal_id": str(proposal_id),
                "instruction": str(instruction),
                "source_task_name": str(source_task_name),
            }
        )
        self._last_pending_transition = reply.get("pending_transition")
        return _deserialize_condition(reply["condition"])

    def repair_semantic_state(self, *, instruction: str, source_task_name: str,
                              patch: dict[str, Any]) -> RobometerCondition:
        reply = self._request(dict(op="repair_semantic_state",
            instruction=instruction, source_task_name=source_task_name, patch=patch))
        self._last_pending_transition = reply.get("pending_transition")
        self.last_input_receipt = reply.get('input_receipt')
        return _deserialize_condition(reply["condition"])

    def sync_gpt_memory(self, **patch):
        reply = self._request(dict(op='sync_gpt_memory', patch=patch))
        self._last_pending_transition = reply.get('pending_transition')
        return reply['memory']

    def propose_after_gpt_completion(self, *, instruction, completed_subtasks,
                                     memory_version, source_task_name):
        reply = self._request(dict(op='propose_after_gpt_completion',
            instruction=instruction, source_task_name=source_task_name,
            completed_subtasks=completed_subtasks, memory_version=memory_version))
        self.last_input_receipt = reply['proposal']['input_receipt']
        return reply['proposal']

    @torch.inference_mode()
    def forward(
        self,
        tasks: Sequence[str],
        frames: Sequence[Any],
        states: Sequence[Any],
        history_indices: Sequence[torch.Tensor],
        source_task_names: Sequence[str] | None = None,
    ) -> RobometerCondition:
        del states
        if (
            len(tasks) != 1
            or len(frames) != 1
            or len(history_indices) != 1
            or source_task_names is None
            or len(source_task_names) != 1
        ):
            raise ValueError("V72 deployed worker requires one complete sample.")
        query_frame = int(history_indices[0][-1].item())
        latest = frames[0][-1]
        if torch.is_tensor(latest):
            latest = latest.detach().cpu().numpy()
        image = np.asarray(latest, dtype=np.uint8)
        if image.ndim != 3:
            raise ValueError(f"V72 frame must be rank three, got {image.shape}.")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError(f"V72 frame must be HWC RGB, got {image.shape}.")
        reply = self._request(
            {
                "op": "forward",
                "instruction": str(tasks[0]),
                "source_task_name": str(source_task_names[0]),
                "query_frame": query_frame,
                "frame": np.ascontiguousarray(image),
            }
        )
        self._last_pending_transition = reply.get("pending_transition")
        self.last_input_receipt = reply.get('input_receipt')
        return _deserialize_condition(reply["condition"])

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self._request({"op": "shutdown"}, timeout=10.0)
        except Exception:
            pass
        self._closed = True
        try:
            self._connection.close()
        except Exception:
            pass
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10.0)
        self._socket_path.unlink(missing_ok=True)


def _attention_backend(conditioner: FrozenRobometerConditioner) -> str:
    objects = [conditioner.model]
    current = conditioner.model
    for _ in range(4):
        current = getattr(current, "model", None)
        if current is None:
            break
        objects.append(current)
    for item in objects:
        config = getattr(item, "config", None)
        implementation = getattr(config, "_attn_implementation", None)
        if implementation:
            return str(implementation)
    return "unknown"


def _run_worker(socket_path: Path) -> None:
    socket_path.unlink(missing_ok=True)
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
    connection = listener.accept()
    conditioner: FrozenRobometerConditioner | None = None
    frame_history: list[np.ndarray] = []
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            operation = request.get("op")
            try:
                if operation == "initialize":
                    if conditioner is not None:
                        raise RuntimeError("V72 worker was initialized twice.")
                    conditioner = FrozenRobometerConditioner(
                        checkpoint_path=request["checkpoint_path"],
                        run_config_path=request["run_config_path"],
                        max_new_tokens=int(request["max_new_tokens"]),
                        num_layers=int(request["num_layers"]),
                        transition_confirm_observations=int(
                            request["transition_confirm_observations"]
                        ),
                        capture_subtask_layers=bool(
                            request["capture_subtask_layers"]
                        ),
                    )
                    conditioner.to(
                        device=str(request["device"]),
                        dtype=torch.bfloat16,
                    ).eval()
                    backend = _attention_backend(conditioner)
                    if backend != "flash_attention_2":
                        raise RuntimeError(
                            "Pinned V72 worker requires flash_attention_2, "
                            f"got {backend!r}."
                        )
                    connection.send(
                        {
                            "ok": True,
                            "hidden_dim": int(conditioner.hidden_dim),
                            "attention_backend": backend,
                            "torch_version": torch.__version__,
                        }
                    )
                elif operation == "reset":
                    if conditioner is None:
                        raise RuntimeError("V72 worker is not initialized.")
                    frame_history.clear()
                    conditioner.reset(1)
                    connection.send({"ok": True})
                elif operation == "enable_transactional_v72":
                    if conditioner is None:
                        raise RuntimeError("V72 worker is not initialized.")
                    conditioner.enable_v72_transactional_mode()
                    connection.send({"ok": True})
                elif operation == "forward":
                    if conditioner is None:
                        raise RuntimeError("V72 worker is not initialized.")
                    query_frame = int(request["query_frame"])
                    if query_frame != len(frame_history):
                        raise RuntimeError(
                            "V72 IPC frames must be consecutive: "
                            f"next={len(frame_history)}, query={query_frame}."
                        )
                    frame_history.append(
                        np.ascontiguousarray(request["frame"], dtype=np.uint8)
                    )
                    indices = torch.arange(query_frame + 1, dtype=torch.long)
                    condition = conditioner(
                        tasks=[str(request["instruction"])],
                        frames=[tuple(frame_history)],
                        states=[()],
                        history_indices=[indices],
                        source_task_names=[str(request["source_task_name"])],
                    )
                    connection.send(
                        {
                            "ok": True,
                            "condition": _serialize_condition(condition),
                            "pending_transition": conditioner.v72_pending_transition(),
                            "input_receipt": getattr(conditioner, '_v72_last_input_receipt', None),
                        }
                    )
                elif operation == "commit_transition":
                    if conditioner is None:
                        raise RuntimeError("V72 worker is not initialized.")
                    condition = conditioner.commit_v72_transition(
                        proposal_id=str(request["proposal_id"]),
                        task=str(request["instruction"]),
                        source_task_name=str(request["source_task_name"]),
                        frame_history=tuple(frame_history),
                    )
                    connection.send(
                        {
                            "ok": True,
                            "condition": _serialize_condition(condition),
                            "pending_transition": conditioner.v72_pending_transition(),
                        }
                    )
                elif operation == "repair_semantic_state":
                    if conditioner is None:
                        raise RuntimeError("V72 worker is not initialized.")
                    condition = conditioner.repair_v72_state(
                        **request["patch"], task=str(request["instruction"]),
                        source_task_name=str(request["source_task_name"]),
                        frame_history=tuple(frame_history))
                    connection.send(dict(ok=True, condition=_serialize_condition(condition),
                        pending_transition=conditioner.v72_pending_transition(),
                        input_receipt=getattr(conditioner, '_v72_last_input_receipt', None)))
                elif operation == 'sync_gpt_memory':
                    memory = conditioner.sync_gpt_memory(**request['patch'])
                    connection.send(dict(ok=True, memory=memory,
                        pending_transition=conditioner.v72_pending_transition()))
                elif operation == 'propose_after_gpt_completion':
                    proposal = conditioner.propose_after_gpt_completion(
                        completed_subtasks=request['completed_subtasks'],
                        memory_version=request['memory_version'],
                        task=str(request['instruction']),
                        source_task_name=str(request['source_task_name']),
                        frame_history=tuple(frame_history))
                    connection.send(dict(ok=True, proposal=proposal))
                elif operation == "shutdown":
                    connection.send({"ok": True})
                    break
                else:
                    raise ValueError(f"Unknown V72 worker operation: {operation!r}")
            except Exception as error:
                connection.send(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    finally:
        connection.close()
        listener.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    _run_worker(args.worker)


if __name__ == "__main__":
    main()
