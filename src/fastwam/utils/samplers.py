from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self.reorder_transition_epoch = None
        self.reorder_transition_sample_offset = 0

    def set_reorder_transition(self, epoch: int, sample_offset: int):
        if epoch < 0 or sample_offset < 0 or sample_offset % (self.batch_size * self.num_processes):
            raise ValueError("Reorder transition must be a nonnegative global batch boundary.")
        self.reorder_transition_epoch = int(epoch)
        self.reorder_transition_sample_offset = int(sample_offset)

    def reorder_state_dict(self):
        return dict(
            version=1, epoch=self.reorder_transition_epoch,
            sample_offset=self.reorder_transition_sample_offset,
            batch_size=self.batch_size, num_processes=self.num_processes,
        )

    def load_reorder_state_dict(self, state):
        if state.get("version") != 1 or (state["batch_size"], state["num_processes"]) != (
            self.batch_size, self.num_processes
        ):
            raise ValueError("Microbatch sampler resume requires unchanged batch/world geometry.")
        if state["epoch"] is not None:
            self.set_reorder_transition(state["epoch"], state["sample_offset"])

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    @property
    def base_epoch_size(self) -> int:
        shuffled_epoch_size = getattr(
            self.dataset,
            "shuffled_epoch_size",
            None,
        )
        if shuffled_epoch_size is not None:
            return int(shuffled_epoch_size)
        return len(self.dataset)

    @property
    def num_padding_samples(self) -> int:
        if not bool(getattr(self.dataset, "pad_to_global_batch", False)):
            return 0
        global_microbatch = self.batch_size * self.num_processes
        return (-self.base_epoch_size) % global_microbatch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        shuffled_epoch_indices = getattr(
            self.dataset,
            "shuffled_epoch_indices",
            None,
        )
        if callable(shuffled_epoch_indices):
            indices = shuffled_epoch_indices(g)
        else:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if len(indices) != self.base_epoch_size:
            raise RuntimeError(
                "Dataset epoch index count does not match its declared size: "
                f"indices={len(indices)}, declared={self.base_epoch_size}."
            )
        if bool(getattr(self.dataset, "event_anchor_microbatch_shuffle", False)):
            # Generate the SAME v3 multiset first. On a v3->v4 transition only
            # reorder the unconsumed suffix; future resumes reconstruct this
            # exact transition-epoch plan before applying their own offset.
            start = (
                min(self.reorder_transition_sample_offset, len(indices))
                if self.epoch == self.reorder_transition_epoch else 0
            )
            indices = indices[:start] + self.dataset.reorder_event_microbatches(
                indices[start:], generator=g, batch_size=self.batch_size,
                num_processes=self.num_processes,
            )
        if bool(getattr(self.dataset, "pad_to_global_batch", False)):
            # Weighted task-balanced epochs intentionally draw with replacement.
            # Keep the uniqueness assertion for the original aligned-anchor
            # sampler, but do not reject a documented weighted epoch.
            requires_unique = bool(
                getattr(self.dataset, "requires_unique_epoch_indices", True)
            )
            if requires_unique and len(set(indices)) != len(indices):
                raise RuntimeError(
                    "A no-replacement epoch contains duplicate real anchors."
                )
            # Accelerate otherwise repeats real samples to make every rank see
            # the same number of batches. Negative indices are explicit
            # zero-loss samples handled by RobotVideoDatasetNoT5.
            indices.extend([-1] * self.num_padding_samples)
        # ``resume_batch_offset`` is local to the first iterator constructed
        # after loading a checkpoint.  A checkpoint can be written in any
        # trajectory epoch, so restricting this skip to epoch zero silently
        # replays the beginning of later epochs after resume.
        if self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return self.base_epoch_size + self.num_padding_samples
