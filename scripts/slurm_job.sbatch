#!/usr/bin/env bash
#SBATCH --job-name=tensor-flat
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# ---------------------------------------------------------------------------
# BCHW -> B x (CHW) -> BCHW round-trip experiments (CPU only, no GPU needed;
# a full run takes well under a minute).
#   sbatch scripts/slurm_job.sbatch
#   sbatch scripts/slurm_job.sbatch --batch 16      # args forwarded
#   tail -f logs/slurm-<jobid>.out
# Add  #SBATCH --partition=<name>  if your cluster has no default partition.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

echo "job     : ${SLURM_JOB_ID:-local} on $(hostname)"
echo "started : $(date -Is)"

[[ -x .venv/bin/python ]] || ./scripts/setup_venv.sh
export PYTHONUNBUFFERED=1
.venv/bin/python src/run_experiments.py "$@"

echo "finished: $(date -Is)"
