"""=== OPTIONAL REFERENCE VERIFICATION - NOT PART OF THE SUBMITTED ALGORITHM ===

Uses NumPy's built-in reshape (C/row-major order) purely as an independent
check that the from-scratch implementation in tensor_flat.py produces the same
B x (CHW) layout.  The assignment permits this only as a reference after the
required implementation is complete.
"""
from __future__ import annotations

import numpy as np


def reference_flatten(I: np.ndarray) -> np.ndarray:
    B, C, H, W = I.shape
    return np.reshape(I, (B, C * H * W))          # library reshape: reference only


def matches_reference(I: np.ndarray, flat: np.ndarray) -> bool:
    return bool(np.array_equal(reference_flatten(I), flat))
