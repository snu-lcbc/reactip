#!/bin/bash
#SBATCH --job-name=reactip_se_gsm
#SBATCH --output=logs/reactip_se_gsm_%j.out
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

REACTIP_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYGSM_DIR="${REACTIP_PYGSM_DIR:-$REACTIP_ROOT/../se-gsm/pyGSM}"
ENV_NAME="${REACTIP_ENV:-allegro}"
MODEL_PATH="${1:-${REACTIP_MODEL:-}}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: pass a model path as the first sbatch argument or set REACTIP_MODEL."
    exit 2
fi

cd "$REACTIP_ROOT"
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$PYGSM_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_FLOAT32_MATMUL_PRECISION=high

# Use only one GPU for SE-GSM (pyGSM is single-process).
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

nvidia-smi -L || true

python run_se_gsm.py \
    --model  "$MODEL_PATH" \
    --xyz    examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --isomers examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/isomers.txt \
    --device cuda \
    --num-nodes 12 \
    --max-iters 15 \
    --max-opt-steps 10 \
    --optimizer eigenvector_follow \
    --rtype 2 \
    --output-dir "gsm_runs/${SLURM_JOB_ID}"
