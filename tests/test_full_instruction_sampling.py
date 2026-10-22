"""Frame permutations cover episode tails and resume at the exact global batch."""
import pytest
import torch

from fastwam.datasets.lerobot.full_instruction_video_dataset import FullInstructionRobotVideoDataset
from fastwam.utils.samplers import ResumableEpochSampler


@pytest.fixture
def dataset():
    class Frames:
        def __len__(self):
            return 149
    data=FullInstructionRobotVideoDataset.__new__(FullInstructionRobotVideoDataset)
    data.lerobot_dataset=Frames()
    data.episode_indices=(10,11)
    data.context_len=128
    data.training_shuffle_unit='frame'
    return data


def test_shuffle_matches_fastwam_randperm_and_keeps_all_tail_frames(dataset):
    sampler=ResumableEpochSampler(dataset,seed=42,batch_size=8,num_processes=8)
    orders=[]
    for epoch in range(3):
        sampler.set_epoch(epoch)
        order=list(sampler)
        expected=torch.randperm(149,generator=torch.Generator().manual_seed(42+epoch)).tolist()
        assert order[:149]==expected
        assert sorted(i for i in order if i>=0)==list(range(149))
        assert order[149:]==[-1]*43
        sampler.set_resume_batch_offset(1)
        assert list(sampler)==order[64:]
        sampler.clear_resume_batch_offset()
        orders.append(expected)
    assert orders[0]!=orders[1]!=orders[2]


def test_failed_anchor_is_not_replaced_and_ddp_sentinel_has_zero_weight(dataset):
    dataset._get=lambda index: {'frame_index':index}
    assert dataset[-1]['sample_weight'].item()==0
    assert dataset[148]['sample_weight'].item()==1
    assert dataset[148]['frame_index']==148
    def broken(index):
        raise FileNotFoundError('missing frame')
    dataset._get=broken
    with pytest.raises(FileNotFoundError):
        dataset[13]
