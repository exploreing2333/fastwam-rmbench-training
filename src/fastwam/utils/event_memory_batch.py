"""Keep event-memory bookkeeping on the host during Accelerate placement.

The collator delegates sampling/collation unchanged. Accelerate's supported
``.to(device)`` batch interface moves the numerical inputs normally, but leaves
the small indices/masks used by Python grouping and augmentation on the CPU.
No random draws, model tensors, or trainable caches are handled here.
"""
from dataclasses import dataclass

from accelerate.utils import send_to_device
from torch.utils.data import default_collate


EVENT_HOST_KEYS = frozenset({
    "event_anchor_frame_indices",
    "event_anchor_valid_mask",
    "event_anchor_group_id",
    "event_anchor_candidate_indices",
    "frame_index",
    "fastwam_video_shape",
})


class EventMemoryBatch(dict):
    def to(self, device, non_blocking=False):
        # Pass a plain dict to avoid recursively calling this .to() method.
        return type(self)(send_to_device(
            dict(self), device, non_blocking=non_blocking,
            skip_keys=EVENT_HOST_KEYS,
        ))


@dataclass
class EventMemoryCollator:
    """Picklable for worker processes; preserve any existing custom collator."""

    collate_fn: object = None

    def __call__(self, samples):
        batch = (self.collate_fn or default_collate)(samples)
        if not isinstance(batch, dict):
            raise TypeError("Event-memory training requires a dictionary batch.")
        return EventMemoryBatch(batch)
