#!/usr/bin/env python3
"""Fallback NequIP gradient script for the molecularGSM C++ ASE bridge.

This script remains a compatibility path only. The production SE-GSM path is
the Python/pyGSM workflow in :mod:`reactip.se_gsm`, which keeps the ML model
loaded in-process. The C++ bridge still launches a fresh Python process for
every gradient call, so it is intentionally not optimized further here.

Usage::

    export NEQUIP_MODEL=/path/to/model.nequip.pt2
    export NEQUIP_DEVICE=cuda   # optional, default: cuda
    cp reactip/reactip/grad_nequip.py se-gsm/molecularGSM/GSM/grad.py
    chmod +x se-gsm/molecularGSM/GSM/grad.py
"""

from __future__ import annotations

import os
import sys

from ase.io import read


def get_calculator():
    model_path = os.environ.get("NEQUIP_MODEL")
    if model_path is None:
        print("ERROR: NEQUIP_MODEL environment variable not set", file=sys.stderr)
        sys.exit(1)

    device = os.environ.get("NEQUIP_DEVICE", "cuda")

    from reactip import ReactIPCalculator

    return ReactIPCalculator(model_path, device=device)


def main():
    if len(sys.argv) < 2:
        print("Usage: grad_nequip.py <run_id> [ncpu] [charge]", file=sys.stderr)
        sys.exit(1)

    run_id = sys.argv[1]
    structure_file = f"scratch/structure{run_id}"
    grad_file = f"scratch/GRAD{run_id}"

    atoms = read(structure_file)
    calculator = get_calculator()
    result = calculator.calculate(atoms)

    with open(grad_file, "w") as f:
        f.write(str(result["energy"]))
        f.write("\n")
        f.write(str(result["forces"]))
        f.write("\n")


if __name__ == "__main__":
    main()
