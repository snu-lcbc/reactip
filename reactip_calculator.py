"""ReactIPCalculator — energy and force prediction entry point.

Computes energy and forces for a given XYZ structure using a trained model.

Run::

    python reactip_calculator.py --model path/to/model.nequip.pt2 --xyz structure.xyz
    python reactip_calculator.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reactip import ReactIPCalculator

def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run ReactIPCalculator energy/force prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("REACTIP_MODEL"),
        help="Model path (.ckpt, .nequip.pth, or .nequip.pt2). Required unless REACTIP_MODEL is set.",
    )
    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument("--device", default=_default_device(), help="PyTorch device")
    parser.add_argument(
        "--units",
        default="eV",
        choices=["eV", "kcal/mol"],
        help="Output energy units",
    )
    args = parser.parse_args(argv)

    if not args.model:
        parser.error("--model is required unless REACTIP_MODEL is set.")

    calc = ReactIPCalculator(args.model, device=args.device)
    result = calc.calculate_file(args.xyz)
    if args.units == "kcal/mol":
        result = ReactIPCalculator.convert_results_to_units(result, energy_units="kcal/mol")

    print(f"Energy : {result['energy']:.6f} {args.units}")
    print(f"Forces ({result['forces'].shape[0]} atoms, {args.units}/Å):")
    for i, f in enumerate(result["forces"]):
        print(f"  atom {i:3d}: [{f[0]:10.5f}  {f[1]:10.5f}  {f[2]:10.5f}]")


if __name__ == "__main__":
    main()
