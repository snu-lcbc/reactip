#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REACTIP_ROOT=$(cd "$HERE/../../.." && pwd)
MODEL=${1:-${REACTIP_MODEL:-}}
OUT_DIR=${2:-"$REACTIP_ROOT/exploratory_runs/2_4_diphenylpentane__C17H20/aryl_shift_left"}

if [[ -z "$MODEL" ]]; then
  echo "ERROR: pass a model path as the first argument or set REACTIP_MODEL."
  exit 2
fi

python "$REACTIP_ROOT/run_se_gsm.py" \
  --model "$MODEL" \
  --xyz "$HERE/reactant.xyz" \
  --isomers "$HERE/isomers_aryl_shift_left.txt" \
  --label "2_4_diphenylpentane__C17H20__aryl_shift_left" \
  --reaction-label "2,4-Diphenylpentane exploratory 1,2-aryl shift" \
  --formula "C17H20" \
  --case-kind "exploratory" \
  --source-fixture "PubChem CID 244014 3D conformer" \
  --device cpu \
  --num-nodes 15 \
  --max-iters 25 \
  --max-opt-steps 15 \
  --optimizer lbfgs \
  --rtype 0 \
  --max-force 300.0 \
  --output-dir "$OUT_DIR" \
  --ID 1
