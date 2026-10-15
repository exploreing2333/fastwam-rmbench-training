"""Selected subtask padding must contribute zero action gradient."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from fastwam.datasets.lerobot.gt_subtask_video_dataset import GroundTruthSubtaskRobotVideoDataset
from fastwam.models.wan22.action_loss import reduce_action_loss
from fastwam_qwen.datasets.robot_video_dataset import RobotVideoDatasetNoT5


class SelectedHoldLossTests(unittest.TestCase):
    def dataset(self):
        ds = GroundTruthSubtaskRobotVideoDataset.__new__(GroundTruthSubtaskRobotVideoDataset)
        ds.exclude_terminal_hold_loss_subtasks = frozenset({"press the button."})
        ds._gt_stage_index = {0: np.array([0]*2 + [1]*4 + [2]*3)}
        ds._subtask_segment_end = {0: np.array([1]*2 + [5]*4 + [8]*3)}
        ds._gt_texts_by_episode = {0: ("Pick up the block.", "Press the button.", "Return the block.")}
        ds._gt_contexts = {text: torch.ones(2,4) for text in ds._gt_texts_by_episode[0]}
        ds.context_len = 2
        ds.current_obs_index = 0
        ds.video_sample_indices = list(range(0,33,4))
        ds.trajectory_anchor_stride = 1
        ds.training_shuffle_unit = "trajectory"
        ds.lerobot_dataset = SimpleNamespace(
            episode_data_index={"from": torch.tensor([0]), "to": torch.tensor([9])},
        )
        return ds

    def sample(self, frame):
        return dict(episode_index=torch.tensor(0),frame_index=torch.tensor(frame),
            action=torch.arange(frame,frame+32).clamp(max=8)[:,None].expand(32,14).float().clone(),
            action_is_pad=torch.arange(frame,frame+32)>=9,
            video=torch.arange(frame,frame+33,4).clamp(max=8)[None,:,None,None].expand(3,9,1,1).float().clone(),
            image_is_pad=torch.arange(frame,frame+33,4)>=9)

    def test_only_selected_artificial_suffix_is_masked(self):
        ds = self.dataset()
        ds._validate_terminal_hold_loss_subtasks()
        for frame in range(9):
            with self.subTest(frame=frame):
                with patch.object(RobotVideoDatasetNoT5, "_get", return_value=self.sample(frame)):
                    result = ds._get(frame)
                end = int(ds._subtask_segment_end[0][frame])
                torch.testing.assert_close(result["action"][:,0],torch.arange(frame,frame+32).clamp(max=end).float())
                self.assertFalse(result["action_is_pad"].any())
                expected = torch.arange(32)<end-frame+1 if 2<=frame<6 else torch.ones(32,dtype=torch.bool)
                self.assertTrue(torch.equal(result["action_loss_mask"],expected))
                self.assertTrue(result["action_loss_mask"][0])
        ds.exclude_terminal_hold_loss_subtasks = frozenset()
        with patch.object(RobotVideoDatasetNoT5, "_get", return_value=self.sample(5)):
            self.assertNotIn("action_loss_mask",ds._get(5))

    def test_exclusion_preserves_sampler_order_and_boundaries(self):
        ds = self.dataset()
        before = ds.shuffled_epoch_indices(torch.Generator().manual_seed(42))
        contract = ds.sampling_contract
        ds.exclude_terminal_hold_loss_subtasks = frozenset()
        after = ds.shuffled_epoch_indices(torch.Generator().manual_seed(42))
        self.assertEqual(before,after)
        self.assertEqual(contract,ds.sampling_contract)

    def test_typo_is_rejected(self):
        ds = self.dataset()
        ds.exclude_terminal_hold_loss_subtasks = frozenset({"press a missing button."})
        with self.assertRaisesRegex(ValueError,"no matching GT subtask"):
            ds._validate_terminal_hold_loss_subtasks()

    def test_zero_gradient_on_padding_and_no_short_suffix_upweight(self):
        pred = torch.ones(3,32,14,requires_grad=True)
        mask = torch.ones(3,32,dtype=torch.bool)
        mask[0,1:] = False
        mask[1,8:] = False
        losses = reduce_action_loss(pred.square().mean(dim=-1),torch.zeros_like(mask),mask)
        torch.testing.assert_close(losses,torch.tensor([1/32,8/32,1.]))
        losses.sum().backward()
        self.assertEqual(torch.count_nonzero(pred.grad[0,1:]).item(),0)
        self.assertEqual(torch.count_nonzero(pred.grad[1,8:]).item(),0)
        torch.testing.assert_close(pred.grad[0,0],pred.grad[2,0],rtol=0,atol=0)
        self.assertTrue((pred.grad[2]!=0).all())

    def test_mask_is_loss_only_and_legacy_episode_padding_is_unchanged(self):
        errors = torch.arange(64,dtype=torch.float32).view(2,32)
        pads = torch.zeros(2,32,dtype=torch.bool)
        pads[1,8:] = True
        expected = torch.stack([errors[0].mean(),errors[1,:8].mean()])
        torch.testing.assert_close(reduce_action_loss(errors,pads),expected,rtol=0,atol=0)
        torch.testing.assert_close(reduce_action_loss(errors,pads,torch.ones_like(pads)),expected,rtol=0,atol=0)
        torch.testing.assert_close(reduce_action_loss(errors),errors.mean(dim=1),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            reduce_action_loss(errors,pads,torch.ones(2,8,dtype=torch.bool))

    def test_eval_adapter_keeps_loss_mask(self):
        from fastwam.trainer import Wan22Trainer
        sample = self.sample(5)
        sample.update(proprio=torch.zeros(1,14),context=torch.zeros(2,4),
            context_mask=torch.ones(2,dtype=torch.bool),prompt="Press the button.",
            action_loss_mask=torch.arange(32)<1)
        for raw in (sample,torch.utils.data.default_collate([sample,sample])):
            batch = Wan22Trainer._to_batched_eval_sample(raw)
            expected = raw["action_loss_mask"]
            if expected.ndim==1:
                expected = expected.unsqueeze(0)
            self.assertTrue(torch.equal(batch["action_loss_mask"],expected))


if __name__=="__main__":
    unittest.main()
