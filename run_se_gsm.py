"""Research-focused ReactIP SE-GSM runner with reporting/export support.

Run from the ``reactip/`` directory::

    python run_se_gsm.py \
        --model models/model_e1f9_l2_f32.nequip.pt2 \
        --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
        --isomers examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/isomers.txt \
        --label butadiene_ethylene_diels_alder__C6H10 \
        --reaction-label "Butadiene + ethylene Diels-Alder" \
        --output-dir runs/diels_alder

In addition to the raw pyGSM files, this runner writes:

- ``summary.json``
- ``trajectory.sdf``
- ``trajectory.gif`` when optional plotting dependencies are installed
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reactip.utils import (
    export_trajectory_artifacts,
    find_preferred_trajectory_source,
    write_summary_json,
)


CHEMICAL_SYMBOLS: tuple[str, ...] = ("H", "C", "N", "O", "F", "S", "Cl", "Br")
_GENERIC_XYZ_STEMS = {"reactant", "struc", "structure", "input", "geom", "geometry"}


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def infer_formula_from_xyz(path: str | Path) -> str:
    xyz_path = Path(path)
    lines = [line.split() for line in xyz_path.read_text().splitlines()[2:] if line.strip()]
    counts = Counter(parts[0] for parts in lines)
    ordered = ["C", "H"] + sorted(symbol for symbol in counts if symbol not in {"C", "H"})
    return "".join(
        f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}"
        for symbol in ordered
        if symbol in counts
    )


def infer_label_from_xyz(path: str | Path) -> str:
    xyz_path = Path(path)
    stem = xyz_path.stem
    if stem.lower() in _GENERIC_XYZ_STEMS and xyz_path.parent.name:
        return xyz_path.parent.name
    return stem


def infer_reaction_label(label: str) -> str:
    return label.split("__", maxsplit=1)[0].replace("_", " ")


def resolve_run_metadata(
    *,
    xyz_file: str | Path,
    label: str | None,
    reaction_label: str | None,
    formula: str | None,
    case_kind: str,
    source_fixture: str | None,
) -> dict[str, str | None]:
    xyz_path = Path(xyz_file)
    resolved_label = label or infer_label_from_xyz(xyz_path)
    resolved_formula = formula or infer_formula_from_xyz(xyz_path)
    resolved_reaction_label = reaction_label or infer_reaction_label(resolved_label)
    return {
        "case": resolved_label,
        "case_kind": case_kind,
        "reaction_label": resolved_reaction_label,
        "formula": resolved_formula,
        "source_fixture": source_fixture,
    }


def collect_raw_output_paths(run_dir: str | Path, run_id: int) -> dict[str, str | None]:
    run_dir = Path(run_dir)
    paths = {
        "opt_converged_xyz": run_dir / f"opt_converged_{run_id:03d}.xyz",
        "grown_string1_xyz": run_dir / f"grown_string1_{run_id:03d}.xyz",
        "grown_string_xyz": run_dir / f"grown_string_{run_id:03d}.xyz",
        "ts_node_xyz": run_dir / f"TSnode_{run_id}.xyz",
        "scratch_dir": run_dir / "scratch",
    }
    return {
        key: str(path) if path.exists() else None
        for key, path in paths.items()
    }


def _export_artifacts_or_warning(
    *,
    run_dir: Path,
    run_id: int,
    case_name: str,
    reaction_label: str,
    formula: str,
) -> dict:
    try:
        trajectory_source = find_preferred_trajectory_source(run_dir, run_id)
        return export_trajectory_artifacts(
            trajectory_source,
            run_dir,
            case_name=case_name,
            reaction_label=reaction_label,
            formula=formula,
        )
    except Exception as exc:
        return {
            "trajectory_source": None,
            "frame_count": 0,
            "energies": [],
            "artifact_paths": {
                "trajectory_sdf": None,
                "trajectory_gif": None,
            },
            "warnings": [f"Artifact export failed: {exc}"],
        }


def run_se_gsm_core(**kwargs):
    from reactip.se_gsm import run_se_gsm as _run_se_gsm

    return _run_se_gsm(**kwargs)


def collect_runtime_provenance() -> dict:
    """Capture the active Python/pyGSM runtime provenance for this run."""
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cwd": str(Path.cwd()),
        "which_gsm": shutil.which("gsm"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
    try:
        import pyGSM

        payload["pygsm_package"] = inspect.getfile(pyGSM)
    except Exception as exc:
        payload["pygsm_import_error"] = f"{exc.__class__.__name__}: {exc}"

    try:
        from pyGSM.level_of_theories import xtb_lot

        payload["xtb_lot_module"] = inspect.getfile(xtb_lot)
    except Exception as exc:
        payload["xtb_lot_import_error"] = f"{exc.__class__.__name__}: {exc}"
    return payload


def run_with_reporting(
    *,
    model_path: str | Path,
    xyz_file: str | Path,
    isomers_file: str | Path,
    run_dir: str | Path,
    label: str | None = None,
    reaction_label: str | None = None,
    formula: str | None = None,
    case_kind: str = "exploratory",
    source_fixture: str | None = None,
    device: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    adiabatic_state: int = 0,
    num_nodes: int = 20,
    max_gsm_iters: int = 100,
    max_opt_steps: int = 20,
    conv_tol: float = 0.0005,
    optimizer: str = "eigenvector_follow",
    coordinate_type: str = "TRIC",
    rtype: int = 2,
    max_force: float = 100.0,
    max_abs_energy: float = 10000.0,
    reactant_geom_fixed: bool = False,
    run_id: int = 0,
) -> tuple[dict, Path, int]:
    model_path = Path(model_path).resolve()
    xyz_path = Path(xyz_file).resolve()
    isomers_path = Path(isomers_file).resolve()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = resolve_run_metadata(
        xyz_file=xyz_path,
        label=label,
        reaction_label=reaction_label,
        formula=formula,
        case_kind=case_kind,
        source_fixture=source_fixture,
    )

    run_error: str | None = None
    run_traceback: str | None = None
    result: dict | None = None
    started_at = perf_counter()
    previous_cwd = Path.cwd()

    try:
        os.chdir(run_dir)
        try:
            result = run_se_gsm_core(
                model_path=str(model_path),
                xyz_file=str(xyz_path),
                driving_coords=str(isomers_path),
                device=device or default_device(),
                charge=charge,
                multiplicity=multiplicity,
                adiabatic_state=adiabatic_state,
                num_nodes=num_nodes,
                max_gsm_iters=max_gsm_iters,
                max_opt_steps=max_opt_steps,
                conv_tol=conv_tol,
                optimizer=optimizer,
                coordinate_type=coordinate_type,
                rtype=rtype,
                max_force=max_force,
                max_abs_energy=max_abs_energy,
                reactant_geom_fixed=reactant_geom_fixed,
                chemical_symbols=CHEMICAL_SYMBOLS,
                ID=run_id,
            )
        except Exception as exc:
            run_error = f"{exc.__class__.__name__}: {exc}"
            run_traceback = traceback.format_exc()
            print()
            print(f"SE-GSM raised an exception after partial output: {run_error}")
            print("Attempting to export artifacts from the latest saved trajectory instead.")

        artifact_info = _export_artifacts_or_warning(
            run_dir=run_dir,
            run_id=run_id,
            case_name=metadata["case"] or infer_label_from_xyz(xyz_path),
            reaction_label=metadata["reaction_label"] or infer_reaction_label(infer_label_from_xyz(xyz_path)),
            formula=metadata["formula"] or infer_formula_from_xyz(xyz_path),
        )
        elapsed_seconds = perf_counter() - started_at

        energies = artifact_info["energies"]
        if not energies and result is not None:
            energies = result["energies"]

        summary = {
            **metadata,
            "mlip_model_name": model_path.name,
            "mlip_model_path": str(model_path),
            "xyz_file": str(xyz_path),
            "isomers_file": str(isomers_path),
            "run_directory": str(run_dir),
            "trajectory_source": artifact_info["trajectory_source"],
            "frame_count": artifact_info["frame_count"],
            "nnodes": result["nnodes"] if result is not None else artifact_info["frame_count"],
            "status": (
                result["status"]
                if result is not None
                else "runtime_error_after_partial_output"
            ),
            "converged": result["converged"] if result is not None else False,
            "has_ts": result["has_ts"] if result is not None else False,
            "npeaks": result["npeaks"] if result is not None else None,
            "ts_node": result["ts_node"] if result is not None else None,
            "ts_energy": result["ts_energy"] if result is not None else None,
            "delta_e": result["delta_e"] if result is not None else None,
            "energies": energies,
            "artifact_paths": artifact_info["artifact_paths"],
            "raw_output_paths": collect_raw_output_paths(run_dir, run_id),
            "warnings": artifact_info["warnings"],
            "runtime_error": run_error,
            "runtime_error_traceback": run_traceback,
            "runtime_provenance": collect_runtime_provenance(),
            "se_gsm_parameters": {
                "device": device or default_device(),
                "charge": charge,
                "multiplicity": multiplicity,
                "adiabatic_state": adiabatic_state,
                "num_nodes": num_nodes,
                "max_gsm_iters": max_gsm_iters,
                "max_opt_steps": max_opt_steps,
                "conv_tol": conv_tol,
                "optimizer": optimizer,
                "coordinate_type": coordinate_type,
                "rtype": rtype,
                "max_force": max_force,
                "max_abs_energy": max_abs_energy,
                "reactant_geom_fixed": reactant_geom_fixed,
                "run_id": run_id,
            },
            "elapsed_seconds": elapsed_seconds,
        }

        summary_path = write_summary_json(summary, run_dir / "summary.json")
    finally:
        os.chdir(previous_cwd)

    exit_code = 0 if result is not None else 1
    return summary, summary_path, exit_code


def print_run_report(summary: dict, summary_path: str | Path) -> None:
    print(json.dumps(summary, indent=2))
    print()
    print(f"Summary JSON : {summary_path}")
    print(f"SDF          : {summary['artifact_paths']['trajectory_sdf']}")
    print(f"GIF          : {summary['artifact_paths']['trajectory_gif']}")
    ts_path = summary["raw_output_paths"].get("ts_node_xyz")
    if ts_path is not None:
        print(f"TS node XYZ  : {ts_path}")
    if summary.get("warnings"):
        print(f"Warnings     : {summary['warnings']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ReactIP-backed SE-GSM and export a machine-readable report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to compiled model or checkpoint")
    parser.add_argument("--xyz", required=True, help="Reactant XYZ file")
    parser.add_argument("--isomers", required=True, help="Driving coordinates (isomers) file")
    parser.add_argument("--label", default=None, help="Stable run/case ID used in reports")
    parser.add_argument("--reaction-label", default=None, help="Human-readable label used in reports and GIF title")
    parser.add_argument("--formula", default=None, help="Empirical formula stored in summary/SDF metadata")
    parser.add_argument(
        "--case-kind",
        default="exploratory",
        help="Short run category stored in summary.json, e.g. benchmark or exploratory",
    )
    parser.add_argument(
        "--source-fixture",
        default=None,
        help="Optional provenance note stored in summary.json",
    )
    parser.add_argument("--device", default=default_device(), help="PyTorch device")
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge (v1 supports only 0)")
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (v1 supports only 1)",
    )
    parser.add_argument(
        "--adiabatic-state",
        type=int,
        default=0,
        help="Adiabatic state index (v1 supports only 0)",
    )
    parser.add_argument("--num-nodes", type=int, default=20, help="Max string nodes")
    parser.add_argument("--max-iters", type=int, default=100, help="Max GSM iterations")
    parser.add_argument("--max-opt-steps", type=int, default=20, help="Max optimizer steps per cycle")
    parser.add_argument("--conv-tol", type=float, default=0.0005, help="TS convergence tolerance")
    parser.add_argument("--optimizer", default="eigenvector_follow", choices=["eigenvector_follow", "lbfgs"])
    parser.add_argument("--coord-type", default="TRIC", choices=["TRIC", "DLC", "HDLC"])
    parser.add_argument(
        "--rtype",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="0=no climb, 1=climb only, 2=find+climb",
    )
    parser.add_argument("--no-pre-opt", action="store_true", help="Skip reactant pre-optimization")
    parser.add_argument("--max-force", type=float, default=100.0, help="Safety cutoff for |F| in eV/A")
    parser.add_argument("--max-abs-energy", type=float, default=10000.0, help="Safety cutoff for |E| in eV")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory for pyGSM output and exported report artifacts. Defaults to the current working directory.",
    )
    parser.add_argument("--ID", type=int, default=0, help="String ID used in pyGSM filenames")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd().resolve()

    print("ReactIP SE-GSM research runner")
    print(f"  Label      : {args.label or infer_label_from_xyz(args.xyz)}")
    print(f"  XYZ        : {Path(args.xyz).resolve()}")
    print(f"  Isomers    : {Path(args.isomers).resolve()}")
    print(f"  Model      : {Path(args.model).resolve()}")
    print(f"  Output dir : {run_dir}")
    print(f"  Device     : {args.device}")
    print()

    summary, summary_path, exit_code = run_with_reporting(
        model_path=args.model,
        xyz_file=args.xyz,
        isomers_file=args.isomers,
        run_dir=run_dir,
        label=args.label,
        reaction_label=args.reaction_label,
        formula=args.formula,
        case_kind=args.case_kind,
        source_fixture=args.source_fixture,
        device=args.device,
        charge=args.charge,
        multiplicity=args.multiplicity,
        adiabatic_state=args.adiabatic_state,
        num_nodes=args.num_nodes,
        max_gsm_iters=args.max_iters,
        max_opt_steps=args.max_opt_steps,
        conv_tol=args.conv_tol,
        optimizer=args.optimizer,
        coordinate_type=args.coord_type,
        rtype=args.rtype,
        max_force=args.max_force,
        max_abs_energy=args.max_abs_energy,
        reactant_geom_fixed=args.no_pre_opt,
        run_id=args.ID,
    )
    print_run_report(summary, summary_path)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
