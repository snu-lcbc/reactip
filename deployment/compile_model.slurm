#!/bin/bash
#SBATCH --job-name=compile_model
#SBATCH --output=log.compile_model_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#
# Compile a NequIP .nequip.zip package into a device-specific .nequip.pt2.
#
# The .nequip.zip is portable (built on the training server). This script
# compiles it for the current GPU architecture using AOTInductor.
#
# Usage (from the reactip/ directory):
#   sbatch -p g3090_veryshort --qos=veryshort deployment/compile_model.slurm
#   sbatch -p g4090_veryshort --qos=veryshort deployment/compile_model.slurm
#
# To compile a specific package:
#   sbatch -p g3090_veryshort --qos=veryshort \
#       --export=ALL,PACKAGE=models/model_e1f9_l2_f32.nequip.zip deployment/compile_model.slurm

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODELS_DIR="$PROJECT_ROOT/models"
ENV_NAME="${REACTIP_ENV:-reactip}"

cd "$PROJECT_ROOT"

export CUDA_HOME="${CUDA_HOME:-/appl/cuda/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"

eval "$("$(command -v conda)" shell.bash hook)"
conda activate "$ENV_NAME"

COMPILE_ARGS=()
if [[ -n "${PACKAGE:-}" ]]; then
    COMPILE_ARGS+=("$PACKAGE")
fi

exec bash deployment/compile_from_package.sh "${COMPILE_ARGS[@]}"
