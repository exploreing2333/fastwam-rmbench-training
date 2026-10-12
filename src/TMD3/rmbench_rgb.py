"""RMBench RGB transport helpers shared by Progress cache and deployment.

RMBench's raw HDF5 writer passes environment RGB arrays directly to OpenCV's
JPEG encoder.  Consequently, ``cv2.imdecode`` returns the original numerical
environment RGB order even though OpenCV calls the returned layout BGR.  The
V56 Progress checkpoint was trained with that ``semantic_rgb`` contract.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


LEGACY_BGR2RGB = "legacy_bgr2rgb"
SEMANTIC_RGB = "semantic_rgb"
SUPPORTED_RMBENCH_RGB_DECODE_MODES = frozenset(
    {LEGACY_BGR2RGB, SEMANTIC_RGB}
)


def normalize_rmbench_rgb_decode_mode(mode: Any) -> str:
    """Validate and canonicalize a saved Progress RGB decode mode."""
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_RMBENCH_RGB_DECODE_MODES:
        raise ValueError(
            "rmbench_rgb_decode_mode must be one of "
            f"{sorted(SUPPORTED_RMBENCH_RGB_DECODE_MODES)}, got {mode!r}."
        )
    return normalized


def decode_rmbench_jpeg(encoded: Any, mode: Any) -> np.ndarray:
    """Decode one raw RMBench JPEG using the checkpoint's saved contract."""
    image = cv2.imdecode(
        np.frombuffer(encoded, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("OpenCV could not decode the RMBench RGB frame.")
    if normalize_rmbench_rgb_decode_mode(mode) == LEGACY_BGR2RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image)


def prepare_rmbench_environment_rgb(image: Any, mode: Any) -> np.ndarray:
    """Map a live environment RGB array to the Progress checkpoint contract."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            "Expected an RMBench RGB image with shape [H,W,3], got "
            f"{array.shape}."
        )
    if array.dtype != np.uint8:
        raise ValueError(
            f"Expected an RMBench uint8 RGB image, got dtype={array.dtype}."
        )
    if normalize_rmbench_rgb_decode_mode(mode) == LEGACY_BGR2RGB:
        # cv2's RGB2BGR and BGR2RGB operations are both the same R/B swap;
        # this spelling documents that the input is an environment RGB array.
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(array)
