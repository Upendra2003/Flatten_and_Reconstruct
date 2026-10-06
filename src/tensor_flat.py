"""BCHW <-> B x (CHW) conversion, implemented from scratch.

No reshape()/view()/flatten()/ravel() (or equivalents) are used here.  Every
destination address is generated explicitly from the lecture mapping:

    Flatten      (b, c, h, w)  ->  (b, k)      k = c*H*W + h*W + w

    Reconstruct  (b, k)        ->  (b, c, h, w)
                 c = k // (H*W)
                 r = k %  (H*W)          # position inside channel c
                 h = r // W
                 w = r %  W

Two variants are provided:
  * *_loop       - the main algorithm: explicit nested loops, one element at a
                   time.  This is the submitted implementation.
  * *_vectorized - the SAME formulas, but the index arithmetic is evaluated on
                   whole index arrays and the copy is done with NumPy fancy
                   indexing (a gather/scatter, still no reshape).  Used only to
                   cross-check the loop version and to show it scales.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Index mapping (scalar)
# --------------------------------------------------------------------------
def bchw_to_k(c: int, h: int, w: int, H: int, W: int) -> int:
    """Column index k in the B x (CHW) tensor for element (c, h, w)."""
    return c * H * W + h * W + w


def k_to_chw(k: int, H: int, W: int) -> tuple[int, int, int]:
    """Inverse of bchw_to_k: recover (c, h, w) from column index k."""
    HW = H * W
    c = k // HW
    r = k % HW
    h = r // W
    w = r % W
    return c, h, w


# --------------------------------------------------------------------------
# Main algorithm: explicit loops
# --------------------------------------------------------------------------
def flatten_loop(I: np.ndarray) -> np.ndarray:
    """B x C x H x W  ->  B x (CHW) using explicit loops."""
    B, C, H, W = I.shape
    # NaN-initialised so any address the loops fail to write shows up as error
    flat = np.full((B, C * H * W), np.nan, dtype=np.float64)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    k = c * H * W + h * W + w
                    flat[b, k] = I[b, c, h, w]
    return flat


def reconstruct_loop(flat: np.ndarray, C: int, H: int, W: int) -> np.ndarray:
    """B x (CHW)  ->  B x C x H x W using explicit loops."""
    B, CHW = flat.shape
    assert CHW == C * H * W, f"CHW mismatch: {CHW} != {C}*{H}*{W}"
    HW = H * W
    I_hat = np.full((B, C, H, W), np.nan, dtype=np.float64)
    for b in range(B):
        for k in range(CHW):
            c = k // HW
            r = k % HW
            h = r // W
            w = r % W
            I_hat[b, c, h, w] = flat[b, k]
    return I_hat


# --------------------------------------------------------------------------
# Same formulas, evaluated on index arrays (cross-check only)
# --------------------------------------------------------------------------
def flatten_vectorized(I: np.ndarray) -> np.ndarray:
    """B x C x H x W -> B x (CHW): k computed for every (c,h,w), then scatter."""
    B, C, H, W = I.shape
    c = np.arange(C)[:, None, None]
    h = np.arange(H)[None, :, None]
    w = np.arange(W)[None, None, :]
    k = c * H * W + h * W + w                  # shape (C, H, W), integer
    flat = np.full((B, C * H * W), np.nan, dtype=np.float64)
    flat[:, k] = I                             # flat[b, k[c,h,w]] = I[b,c,h,w]
    return flat


def reconstruct_vectorized(flat: np.ndarray, C: int, H: int, W: int) -> np.ndarray:
    """B x (CHW) -> B x C x H x W: (c,h,w) computed for every k, then scatter."""
    B, CHW = flat.shape
    assert CHW == C * H * W, f"CHW mismatch: {CHW} != {C}*{H}*{W}"
    HW = H * W
    k = np.arange(CHW)
    c = k // HW
    r = k % HW
    h = r // W
    w = r % W
    I_hat = np.full((B, C, H, W), np.nan, dtype=np.float64)
    I_hat[:, c, h, w] = flat                   # I_hat[b,c[k],h[k],w[k]] = flat[b,k]
    return I_hat


# --------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------
def reconstruction_error(I: np.ndarray, I_hat: np.ndarray) -> tuple[float, float]:
    """(E_max, MAE) of E = I - I_hat.  NaN means some element was never written."""
    E = np.abs(I.astype(np.float64) - I_hat)
    return float(E.max()), float(E.sum() / E.size)
