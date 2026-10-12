from __future__ import annotations

from .fastwam_qwen.runtime import create_tmd

create_fastwam_qwen3vl = create_tmd

__all__ = ["create_tmd", "create_fastwam_qwen3vl"]
