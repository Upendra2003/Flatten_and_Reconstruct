"""Assignment 3: BCHW -> B x (CHW) -> BCHW round-trip experiments.

    python src/run_experiments.py                 # all experiments, loop algorithm
    python src/run_experiments.py --batch 4       # smaller batch (faster)

Writes results/manual_example.md, results/results.csv and results/results.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from data import ROOT, load_dataset_bchw, synthetic_fmap
from reference_check import matches_reference          # optional reference only
from tensor_flat import (bchw_to_k, flatten_loop, flatten_vectorized, k_to_chw,
                         reconstruct_loop, reconstruct_vectorized,
                         reconstruction_error)

RESULTS = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------
# Manual indexing example: B=1, C=2, H=2, W=3, values 1..12
# --------------------------------------------------------------------------
def manual_example() -> str:
    B, C, H, W = 1, 2, 2, 3
    I = np.empty((B, C, H, W))
    v = 1
    for c in range(C):
        for h in range(H):
            for w in range(W):
                I[0, c, h, w] = v
                v += 1

    flat = flatten_loop(I)
    I_hat = reconstruct_loop(flat, C, H, W)

    lines = [f"# Manual indexing example (B={B}, C={C}, H={H}, W={W})", "",
             f"Flatten: k = c*H*W + h*W + w = c*{H*W} + h*{W} + w", "",
             "| I[b,c,h,w] | value | k = c*6 + h*3 + w | flat[b,k] |",
             "|---|---|---|---|"]
    for c in range(C):
        for h in range(H):
            for w in range(W):
                k = bchw_to_k(c, h, w, H, W)
                lines.append(f"| I[0,{c},{h},{w}] | {I[0,c,h,w]:.0f} | "
                             f"{c}*6 + {h}*3 + {w} = {k} | flat[0,{k}] = {flat[0,k]:.0f} |")

    lines += ["", f"flat[0] = {[int(x) for x in flat[0]]}", "",
              "Reconstruct: c = k // (H*W), r = k % (H*W), h = r // W, w = r % W", "",
              "| k | c = k//6 | r = k%6 | h = r//3 | w = r%3 | I_hat[b,c,h,w] |",
              "|---|---|---|---|---|---|"]
    for k in (0, 5, 7, 11):
        c, h, w = k_to_chw(k, H, W)
        r = k % (H * W)
        lines.append(f"| {k} | {c} | {r} | {h} | {w} | I_hat[0,{c},{h},{w}] = {I_hat[0,c,h,w]:.0f} |")

    emax, mae = reconstruction_error(I, I_hat)
    lines += ["", f"E_max = {emax}, MAE = {mae}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def experiments(B: int):
    """(name, tensor) pairs. Synthetic fmaps keep B, H=W=32 fixed and vary C."""
    yield "MNIST / sample", load_dataset_bchw("mnist", B)
    yield "CIFAR-10 / sample", load_dataset_bchw("cifar10", B)
    yield "CIFAR-100 / sample", load_dataset_bchw("cifar100", B)
    for C in (8, 16, 32, 64, 128, 256, 500):
        yield f"Synthetic fmap C={C}", synthetic_fmap(B, C, 32, 32, seed=C)
    # non-square, odd-sized case: catches any H/W or C/H mix-up in the indexing
    yield "Synthetic odd-shape", synthetic_fmap(3, 5, 7, 11, seed=1)


def run_one(name: str, I: np.ndarray, mode: str) -> dict:
    B, C, H, W = I.shape
    flatten, reconstruct = ((flatten_loop, reconstruct_loop) if mode == "loop"
                            else (flatten_vectorized, reconstruct_vectorized))
    t0 = time.perf_counter()
    flat = flatten(I)
    t1 = time.perf_counter()
    I_hat = reconstruct(flat, C, H, W)
    t2 = time.perf_counter()
    emax, mae = reconstruction_error(I, I_hat)
    return {"input": name, "mode": mode, "B": B, "C": C, "H": H, "W": W,
            "flat_shape": f"{flat.shape[0]}x{flat.shape[1]}",
            "E_max": emax, "MAE": mae,
            "ref_match": matches_reference(I, flat),
            "t_flatten_s": round(t1 - t0, 4), "t_reconstruct_s": round(t2 - t1, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8, help="B for every experiment")
    ap.add_argument("--mode", choices=("loop", "vectorized", "both"), default="both",
                    help="loop = main algorithm; vectorized = same formulas on index arrays")
    args = ap.parse_args()
    modes = ("loop", "vectorized") if args.mode == "both" else (args.mode,)
    os.makedirs(RESULTS, exist_ok=True)

    example = manual_example()
    print(example, "\n")
    with open(os.path.join(RESULTS, "manual_example.md"), "w") as f:
        f.write(example + "\n")

    rows = []
    hdr = (f"{'Dataset/Input':<22}{'mode':<11}{'B':>3}{'C':>5}{'H':>4}{'W':>4}"
           f"{'B x CHW':>13}{'E_max':>8}{'MAE':>8}{'ref':>6}{'t_flat':>9}{'t_rec':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, I in experiments(args.batch):
        for mode in modes:
            r = run_one(name, I, mode)
            rows.append(r)
            print(f"{r['input']:<22}{mode:<11}{r['B']:>3}{r['C']:>5}{r['H']:>4}{r['W']:>4}"
                  f"{r['flat_shape']:>13}{r['E_max']:>8.1g}{r['MAE']:>8.1g}"
                  f"{str(r['ref_match']):>6}{r['t_flatten_s']:>9.3f}{r['t_reconstruct_s']:>9.3f}",
                  flush=True)

    with open(os.path.join(RESULTS, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    md = ["| Dataset/Input | B | C | H | W | B x (CHW) | E_max | MAE | flatten (s) | reconstruct (s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["mode"] == "loop":
            md.append(f"| {r['input']} | {r['B']} | {r['C']} | {r['H']} | {r['W']} | "
                      f"{r['flat_shape']} | {r['E_max']:g} | {r['MAE']:g} | "
                      f"{r['t_flatten_s']} | {r['t_reconstruct_s']} |")
    with open(os.path.join(RESULTS, "results.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    ok = all(r["E_max"] == 0 and r["MAE"] == 0 and r["ref_match"] for r in rows)
    print(f"\nALL ROUND-TRIPS EXACT: {ok}")
    print(f"results written to {RESULTS}/")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
