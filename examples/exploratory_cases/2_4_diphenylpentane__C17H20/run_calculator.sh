#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REACTIP_ROOT=$(cd "$HERE/../../.." && pwd)
MODEL=${1:-${REACTIP_MODEL:-}}

if [[ -z "$MODEL" ]]; then
  echo "ERROR: pass a model path as the first argument or set REACTIP_MODEL."
  exit 2
fi

python "$REACTIP_ROOT/reactip_calculator.py" \
  --model "$MODEL" \
  --xyz "$HERE/reactant.xyz"
