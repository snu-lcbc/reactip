#!/usr/bin/env bash
# Compile .nequip.zip packages into device-specific .nequip.pt2 artifacts.
#
# Run this on the TARGET machine where the model will be used.
# The .nequip.zip package bundles example data, so no dataset access is needed.
#
# Usage:
#   bash deployment/compile_from_package.sh                                    # compile all .zip in models/, CUDA
#   bash deployment/compile_from_package.sh --device cpu                       # compile all, CPU
#   bash deployment/compile_from_package.sh models/model_e1f9_l2_f32.nequip.zip   # compile one, CUDA
#   bash deployment/compile_from_package.sh --device cpu models/model.nequip.zip   # compile one, CPU
#
# Prerequisites:
#   conda activate reactip   # needs nequip + torch with CUDA
#   export CUDA_HOME=/path/to/cuda  # for CUDA compilation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${REACTIP_MODELS_DIR:-$PROJECT_ROOT/models}"
ENV_NAME="${REACTIP_ENV:-reactip}"

DEVICE="cuda"
INPUT_PATHS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            [[ $# -ge 2 ]] || { echo "--device requires cpu or cuda" >&2; exit 2; }
            DEVICE="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            INPUT_PATHS+=("$1")
            shift
            ;;
    esac
done

cd "$PROJECT_ROOT"

eval "$("$(command -v conda)" shell.bash hook)"
conda activate "$ENV_NAME"

export CUDA_HOME="${CUDA_HOME:-/appl/cuda/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "============================================================"
echo "ReactIP Model Compilation (from .nequip.zip package)"
echo "  Device:    $DEVICE"
echo "  Models:    $MODELS_DIR"
echo "  Env:       $CONDA_DEFAULT_ENV"
echo "  nequip:    $(python -c 'import nequip; print(nequip.__version__)')"
echo "  PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
if [[ "$DEVICE" == "cuda" ]]; then
    echo "  CUDA:      $(python -c 'import torch; print(torch.version.cuda)')"
    nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null || true
fi
echo "============================================================"

# Discover packages to compile
PACKAGES=()
if [[ ${#INPUT_PATHS[@]} -gt 0 ]]; then
    for path in "${INPUT_PATHS[@]}"; do
        if [[ -f "$path" ]]; then
            PACKAGES+=("$path")
        else
            echo "WARNING: file not found: $path"
        fi
    done
else
    while IFS= read -r pkg; do
        PACKAGES+=("$pkg")
    done < <(find "$MODELS_DIR" -name '*.nequip.zip' -type f | sort)
fi

if [[ ${#PACKAGES[@]} -eq 0 ]]; then
    echo "No .nequip.zip packages found in $MODELS_DIR"
    exit 1
fi

# Detect GPU architecture for CUDA filename
ARCH_TAG=""
if [[ "$DEVICE" == "cuda" ]]; then
    SM_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [[ -n "$SM_CAP" ]]; then
        ARCH_TAG="_sm${SM_CAP}"
    fi
fi

pass=0
fail=0
skip=0

for pkg in "${PACKAGES[@]}"; do
    base="$(basename "$pkg" .nequip.zip)"
    if [[ "$DEVICE" == "cpu" ]]; then
        out="$MODELS_DIR/${base}_cpu.nequip.pt2"
    else
        out="$MODELS_DIR/${base}_cuda${ARCH_TAG}.nequip.pt2"
    fi

    if [[ -f "$out" ]]; then
        echo "SKIP $(basename "$pkg") -- already compiled: $(basename "$out")"
        (( skip++ )) || true
        continue
    fi

    echo "Compiling $(basename "$pkg") -> $(basename "$out") [device=$DEVICE] ..."

    if nequip-compile "$pkg" "$out" \
            --mode aotinductor --device "$DEVICE" --target ase 2>&1; then
        echo "OK  $(basename "$out")  $(ls -lh "$out" | awk '{print $5}')"
        (( pass++ )) || true
    else
        echo "FAIL $(basename "$pkg")"
        rm -f "$out"
        (( fail++ )) || true
    fi
    printf '\n'
done

printf '\n'
echo '========================================='
echo 'Compilation summary'
echo "  device : $DEVICE"
echo "  passed : $pass"
echo "  failed : $fail"
echo "  skipped: $skip"
echo '========================================='
echo "Compiled models in $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.nequip.pt2 2>/dev/null || echo '  (none)'
