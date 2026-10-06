"""Load MNIST / CIFAR test images as BCHW tensors, and generate synthetic fmaps.

The datasets are cached as uint8 .npz files (images, labels):
    mnist_test.npz     images (N, 28, 28)       -> 1 channel
    cifar10_test.npz   images (N, 32, 32, 3)    -> HWC, RGB
    cifar100_test.npz  images (N, 32, 32, 3)

They are looked up in DATA_DIRS (Assignment 1 already prepared them).  If none
is found they are downloaded from the HuggingFace parquet mirrors into ./data.
Conversion to BCHW is done by copying channel by channel into a pre-allocated
array, so no reshape/transpose is involved here either.
"""
from __future__ import annotations

import io
import os
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRS = [
    os.path.join(ROOT, "data"),
    os.path.join(ROOT, "..", "Assignment_1", "dnn-precision-analysis", "data", "prepared"),
]

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "mnist": ("ylecun/mnist", "mnist/test-00000-of-00001.parquet", ("image", "img")),
    "cifar10": ("uoft-cs/cifar10", "plain_text/test-00000-of-00001.parquet", ("img", "image")),
    "cifar100": ("uoft-cs/cifar100", "cifar100/test-00000-of-00001.parquet", ("img", "image")),
}


def _download_test_split(name: str, out_npz: str) -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    repo, path, image_cols = SOURCES[name]
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    pq_path = out_npz.replace(".npz", ".parquet")
    url = HF.format(repo=repo, path=path)
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, pq_path)

    tbl = pq.read_table(pq_path)
    icol = next(c for c in image_cols if c in tbl.column_names)
    imgs = []
    for rec in tbl.column(icol).to_pylist():
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        imgs.append(np.asarray(Image.open(io.BytesIO(raw))))
    np.savez_compressed(out_npz, images=np.stack(imgs))
    os.remove(pq_path)
    print(f"[data] saved {out_npz}")


def find_or_fetch(name: str) -> str:
    fname = f"{name}_test.npz"
    for d in DATA_DIRS:
        p = os.path.normpath(os.path.join(d, fname))
        if os.path.exists(p):
            return p
    out = os.path.join(DATA_DIRS[0], fname)
    _download_test_split(name, out)
    return out


def load_dataset_bchw(name: str, B: int) -> np.ndarray:
    """First B test images of `name` as a float64 BCHW tensor (raw 0-255 values)."""
    path = find_or_fetch(name)
    imgs = np.load(path)["images"][:B]
    if imgs.ndim == 3:                                   # (B, H, W) grayscale
        Bn, H, W = imgs.shape
        x = np.empty((Bn, 1, H, W), dtype=np.float64)
        x[:, 0, :, :] = imgs
    else:                                                # (B, H, W, C) colour
        Bn, H, W, C = imgs.shape
        x = np.empty((Bn, C, H, W), dtype=np.float64)
        for c in range(C):
            x[:, c, :, :] = imgs[:, :, :, c]
    return x


def synthetic_fmap(B: int, C: int, H: int, W: int, seed: int = 0) -> np.ndarray:
    """Random feature map standing in for an intermediate DNN activation."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, C, H, W))
