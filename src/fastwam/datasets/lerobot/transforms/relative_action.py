from typing import List, Dict

import torch
from .rotation import quaternion_to_matrix, matrix_to_quaternion

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


class RelativePoseTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        
        return batch
        
    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])

        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]] # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0: 3, 0: 3] = rotation
        matrix[..., 0: 3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0: 3, 0: 3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]] # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose
    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix
    

class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]
        
        return batch


class MaskedRelativeJointTransform:
    """Convert selected joint dimensions relative to the current state."""

    def __init__(self, dim_masks: Dict[str, List[bool]]):
        if not dim_masks:
            raise ValueError("`dim_masks` must contain at least one action key.")
        self.dim_masks = {
            key: torch.as_tensor(mask, dtype=torch.bool)
            for key, mask in dim_masks.items()
        }

    def _apply(self, batch: Dict, *, inverse: bool) -> Dict:
        if "action" not in batch:
            return batch

        missing_action = sorted(set(self.dim_masks) - set(batch["action"]))
        missing_state = sorted(set(self.dim_masks) - set(batch["state"]))
        if missing_action or missing_state:
            raise KeyError(
                "Masked joint transform keys are missing: "
                f"action={missing_action}, state={missing_state}."
            )

        for key, cpu_mask in self.dim_masks.items():
            action = batch["action"][key]
            state = batch["state"][key]
            if action.shape[-1] != cpu_mask.numel():
                raise ValueError(
                    f"Action {key!r} has dim={action.shape[-1]}, "
                    f"expected mask dim={cpu_mask.numel()}."
                )
            if state.shape[-1] != action.shape[-1]:
                raise ValueError(
                    f"Action/state dimensions differ for {key!r}: "
                    f"action={action.shape[-1]}, state={state.shape[-1]}."
                )
            mask = cpu_mask.to(device=action.device)
            anchor = state[..., :1, :].to(device=action.device, dtype=action.dtype)
            relative = action + anchor if inverse else action - anchor
            batch["action"][key] = torch.where(mask, relative, action)
        return batch

    def forward(self, batch: Dict) -> Dict:
        return self._apply(batch, inverse=False)

    def backward(self, batch: Dict) -> Dict:
        return self._apply(batch, inverse=True)


class MaskedDeltaJointTransform:
    """Convert selected joints to consecutive per-control-step deltas."""

    def __init__(self, dim_masks: Dict[str, List[bool]]):
        if not dim_masks:
            raise ValueError("`dim_masks` must contain at least one action key.")
        self.dim_masks = {
            key: torch.as_tensor(mask, dtype=torch.bool)
            for key, mask in dim_masks.items()
        }

    def _validate(self, batch: Dict) -> None:
        missing_action = sorted(set(self.dim_masks) - set(batch["action"]))
        missing_state = sorted(set(self.dim_masks) - set(batch["state"]))
        if missing_action or missing_state:
            raise KeyError(
                "Masked joint transform keys are missing: "
                f"action={missing_action}, state={missing_state}."
            )

    def forward(self, batch: Dict) -> Dict:
        if "action" not in batch:
            return batch
        self._validate(batch)

        for key, cpu_mask in self.dim_masks.items():
            action = batch["action"][key]
            state = batch["state"][key]
            if action.shape[-1] != cpu_mask.numel():
                raise ValueError(
                    f"Action {key!r} has dim={action.shape[-1]}, "
                    f"expected mask dim={cpu_mask.numel()}."
                )
            if state.shape[-1] != action.shape[-1]:
                raise ValueError(
                    f"Action/state dimensions differ for {key!r}: "
                    f"action={action.shape[-1]}, state={state.shape[-1]}."
                )
            if action.shape[-2] == 0:
                raise ValueError(f"Action {key!r} must contain at least one timestep.")

            anchor = state[..., :1, :].to(device=action.device, dtype=action.dtype)
            delta = torch.empty_like(action)
            delta[..., :1, :] = action[..., :1, :] - anchor
            delta[..., 1:, :] = action[..., 1:, :] - action[..., :-1, :]
            mask = cpu_mask.to(device=action.device)
            batch["action"][key] = torch.where(mask, delta, action)
        return batch

    def backward(self, batch: Dict) -> Dict:
        if "action" not in batch:
            return batch
        self._validate(batch)

        for key, cpu_mask in self.dim_masks.items():
            action = batch["action"][key]
            state = batch["state"][key]
            if action.shape[-1] != cpu_mask.numel():
                raise ValueError(
                    f"Action {key!r} has dim={action.shape[-1]}, "
                    f"expected mask dim={cpu_mask.numel()}."
                )
            if state.shape[-1] != action.shape[-1]:
                raise ValueError(
                    f"Action/state dimensions differ for {key!r}: "
                    f"action={action.shape[-1]}, state={state.shape[-1]}."
                )

            anchor = state[..., :1, :].to(device=action.device, dtype=action.dtype)
            absolute = anchor + torch.cumsum(action, dim=-2)
            mask = cpu_mask.to(device=action.device)
            batch["action"][key] = torch.where(mask, absolute, action)
        return batch
