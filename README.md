# Assignment 3: BCHW ↔ B×(CHW) Tensor Flattening

From-scratch flattening and reconstruction of NCHW tensors. Neither direction uses `reshape`, `view`, `flatten` or `ravel`.

| Direction | Mapping |
|---|---|
| Flatten `B×C×H×W → B×(CHW)` | `k = c·H·W + h·W + w` |
| Reconstruct `B×(CHW) → B×C×H×W` | `c = k // (HW)`, `r = k % (HW)`, `h = r // W`, `w = r % W` |

## Layout
```
src/tensor_flat.py       flatten_loop / reconstruct_loop  (main algorithm, explicit loops)
                         *_vectorized                     (same formulas on index arrays, cross-check)
src/data.py              MNIST / CIFAR-10 / CIFAR-100 loaders -> BCHW, synthetic fmaps
src/reference_check.py   OPTIONAL np.reshape reference (verification only, not the algorithm)
src/run_experiments.py   manual example + all experiments, writes results/
scripts/                 setup_venv.sh, run.sh, slurm_job.sbatch
```

## Run
```bash
./scripts/setup_venv.sh            # once: creates .venv, installs requirements.txt
./scripts/run.sh                   # all experiments (B=8), log in logs/run.log
./scripts/run.sh --batch 16 --mode loop
sbatch scripts/slurm_job.sbatch    # same thing on SLURM (CPU only, <1 min)
```

The script looks for datasets in `./data/` first and then in `../Assignment_1/dnn-precision-analysis/data/prepared/`. If it finds neither, it downloads the test splits from the HuggingFace mirrors into `./data/`.

## Outputs
- `results/manual_example.md`: the B=1, C=2, H=2, W=3 mapping, worked by hand
- `results/results.md`, `results/results.csv`: B, C, H, W, E_max and MAE for every experiment

Each flattened tensor is also compared with `np.reshape` (column `ref` in the console output). This check catches a wrong mapping even when the reconstruction formula makes the same mistake in reverse.
