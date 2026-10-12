from __future__ import annotations

from TMD3.instruction_mot import InstructionStateMoT
from TMD4.long_image_mot import LongImageStateProgressMoT


class LongImageInstructionStateMoT(
    LongImageStateProgressMoT,
    InstructionStateMoT,
):
    """TMD4 long-Image attention conditioned by full T5 instruction.

    Multiple inheritance deliberately composes TMD4's long-Image attention
    and GigaWorld parameter layout with TMD3's Progress-free instruction/state
    cross-attention.  No Progress or decoded-subtask modules survive
    ``InstructionStateMoT.__init__``.
    """


__all__ = ["LongImageInstructionStateMoT"]
