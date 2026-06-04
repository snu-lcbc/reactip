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
import math
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
    parse_molden_xyz_trajectory,
    write_summary_json,
)
from reactip.sampling import (
    DEFAULT_SAMPLE_MODES,
    compute_boltzmann_populations,
    driving_coords_to_lines,
    parse_sample_modes,
    sample_driving_coordinate_sets,
    write_isomers_file,
    write_xyz_frame,
)


CHEMICAL_SYMBOLS: tuple[str, ...] = ("H", "C", "N", "O", "F", "S", "Cl", "Br")
_GENERIC_XYZ_STEMS = {"reactant", "struc", "structure", "input", "geom", "geometry"}
SAMPLE_SCORE_MODES = ("thermodynamic", "kinetic")
SAMPLE_MIN_QUALITY_LEVELS = ("finite", "completed", "converged", "ts")
_COMPLETED_SAMPLE_STATUSES = {
    "completed_no_ts",
    "completed_with_ts_candidate",
    "converged_no_ts",
    "converged_ts",
}
_CONVERGED_SAMPLE_STATUSES = {"converged_no_ts", "converged_ts"}


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


def _collect_trajectory_info_or_warning(
    *,
    run_dir: Path,
    run_id: int,
) -> dict:
    try:
        trajectory_source = find_preferred_trajectory_source(run_dir, run_id)
        trajectory = parse_molden_xyz_trajectory(trajectory_source)
        return {
            "trajectory_source": str(trajectory_source),
            "frame_count": len(trajectory.frames),
            "energies": trajectory.energies,
            "artifact_paths": {
                "trajectory_sdf": None,
                "trajectory_gif": None,
            },
            "warnings": [],
        }
    except Exception as exc:
        return {
            "trajectory_source": None,
            "frame_count": 0,
            "energies": [],
            "artifact_paths": {
                "trajectory_sdf": None,
                "trajectory_gif": None,
            },
            "warnings": [f"Trajectory collection failed: {exc}"],
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
    export_artifacts: bool = True,
    calculator=None,
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
            run_kwargs = {
                "xyz_file": str(xyz_path),
                "driving_coords": str(isomers_path),
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
                "chemical_symbols": CHEMICAL_SYMBOLS,
                "ID": run_id,
            }
            if calculator is None:
                run_kwargs["model_path"] = str(model_path)
            else:
                run_kwargs["calculator"] = calculator
            result = run_se_gsm_core(
                **run_kwargs,
            )
        except Exception as exc:
            run_error = f"{exc.__class__.__name__}: {exc}"
            run_traceback = traceback.format_exc()
            print()
            print(f"SE-GSM raised an exception after partial output: {run_error}")
            print("Attempting to export artifacts from the latest saved trajectory instead.")

        if export_artifacts:
            artifact_info = _export_artifacts_or_warning(
                run_dir=run_dir,
                run_id=run_id,
                case_name=metadata["case"] or infer_label_from_xyz(xyz_path),
                reaction_label=metadata["reaction_label"] or infer_reaction_label(infer_label_from_xyz(xyz_path)),
                formula=metadata["formula"] or infer_formula_from_xyz(xyz_path),
            )
        else:
            artifact_info = _collect_trajectory_info_or_warning(
                run_dir=run_dir,
                run_id=run_id,
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
            "reactant_node": result["reactant_node"] if result is not None else None,
            "product_node": result["product_node"] if result is not None else None,
            "product_energy": result["product_energy"] if result is not None else None,
            "product_delta_e": result["product_delta_e"] if result is not None else None,
            "score_delta_e": result["score_delta_e"] if result is not None else None,
            "score_delta_e_source": (
                result["score_delta_e_source"] if result is not None else None
            ),
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
                "export_artifacts": export_artifacts,
            },
            "elapsed_seconds": elapsed_seconds,
        }

        summary_path = write_summary_json(summary, run_dir / "summary.json")
    finally:
        os.chdir(previous_cwd)

    exit_code = 0 if result is not None else 1
    return summary, summary_path, exit_code


def _finite_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _score_mode_description(score_mode: str) -> str:
    if score_mode == "kinetic":
        return "TST-like cumulative TS-barrier proxy"
    return "cumulative product dE"


def _annotate_candidate_path(candidate: dict, parent: dict, score_mode: str) -> None:
    parent_product_delta_e = _finite_score(parent.get("path_product_delta_e")) or 0.0
    parent_barrier_sum = _finite_score(parent.get("path_kinetic_barrier_sum")) or 0.0
    parent_rate_limiting_barrier = _finite_score(parent.get("path_rate_limiting_barrier_e"))

    local_product_delta_e = _finite_score(candidate.get("product_delta_e"))
    local_ts_barrier_e = _finite_score(candidate.get("ts_energy"))

    path_product_delta_e = (
        parent_product_delta_e + local_product_delta_e
        if local_product_delta_e is not None
        else None
    )
    path_kinetic_barrier_sum = (
        parent_barrier_sum + local_ts_barrier_e
        if local_ts_barrier_e is not None
        else None
    )
    if local_ts_barrier_e is None:
        path_rate_limiting_barrier_e = parent_rate_limiting_barrier
    elif parent_rate_limiting_barrier is None:
        path_rate_limiting_barrier_e = local_ts_barrier_e
    else:
        path_rate_limiting_barrier_e = max(parent_rate_limiting_barrier, local_ts_barrier_e)

    path_candidate_ids = list(parent.get("path_candidate_ids") or [])
    path_candidate_ids.append(str(candidate["candidate_id"]))

    candidate["parent_path_product_delta_e"] = parent_product_delta_e
    candidate["path_product_delta_e"] = path_product_delta_e
    candidate["local_ts_barrier_e"] = local_ts_barrier_e
    candidate["path_kinetic_barrier_sum"] = path_kinetic_barrier_sum
    candidate["path_rate_limiting_barrier_e"] = path_rate_limiting_barrier_e
    candidate["path_candidate_ids"] = path_candidate_ids
    candidate["path_depth"] = len(path_candidate_ids)

    if score_mode == "thermodynamic":
        candidate["ranking_score_delta_e"] = path_product_delta_e
        candidate["ranking_score_source"] = "cumulative_product_delta_e"
    elif score_mode == "kinetic":
        candidate["ranking_score_delta_e"] = path_kinetic_barrier_sum
        candidate["ranking_score_source"] = "cumulative_ts_barrier_sum"
    else:
        raise ValueError(f"Unknown sample score mode: {score_mode}")


def _ranking_exclusion_reason(
    candidate: dict,
    *,
    score_mode: str,
    min_quality: str,
) -> str | None:
    if candidate.get("runtime_error"):
        return "runtime error"
    exit_code = candidate.get("exit_code")
    if exit_code not in (0, None):
        return f"nonzero exit code {exit_code}"
    if candidate.get("product_xyz") is None:
        return "missing product XYZ"

    status = str(candidate.get("status") or "")
    has_ts = bool(candidate.get("has_ts"))
    if score_mode == "kinetic" and not has_ts:
        return "kinetic ranking requires a unique TS"
    score = _finite_score(candidate.get("ranking_score_delta_e"))
    if score is None:
        return "missing finite ranking score"

    if min_quality == "finite":
        return None
    if min_quality == "completed":
        if status not in _COMPLETED_SAMPLE_STATUSES:
            return f"status {status!r} is below completed quality"
        return None
    if min_quality == "converged":
        if status not in _CONVERGED_SAMPLE_STATUSES:
            return f"status {status!r} is below converged quality"
        return None
    if min_quality == "ts":
        if status != "converged_ts" or not has_ts:
            return "converged TS candidate required"
        return None
    raise ValueError(f"Unknown sample minimum quality: {min_quality}")


def _ranked_population_view(
    candidates: list[dict],
    *,
    temperature: float,
    score_mode: str,
    min_quality: str,
) -> list[dict]:
    eligible_candidates: list[dict] = []
    scores: list[float] = []
    for candidate in candidates:
        reason = _ranking_exclusion_reason(
            candidate,
            score_mode=score_mode,
            min_quality=min_quality,
        )
        candidate["ranking_included"] = reason is None
        candidate["ranking_exclusion_reason"] = reason
        if reason is not None:
            continue
        score = _finite_score(candidate.get("ranking_score_delta_e"))
        if score is None:
            continue
        eligible_candidates.append(candidate)
        scores.append(score)

    populations = compute_boltzmann_populations(scores, temperature=temperature)
    rows: list[dict] = []
    for candidate, score, population in zip(eligible_candidates, scores, populations):
        relative_population = population["relative_population"]
        if score is None or relative_population is None:
            continue
        row = {
            **candidate,
            "ranking_score_delta_e": score,
            "boltzmann_log_factor": population["boltzmann_log_factor"],
            "relative_population": relative_population,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["relative_population"]),
            float(row["ranking_score_delta_e"]),
            str(row["candidate_id"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _print_candidate_population_report(
    title: str,
    ranked_rows: list[dict],
    *,
    candidate_count: int,
    top_n: int,
    temperature: float,
    score_mode: str,
    min_quality: str,
) -> None:
    print(title)
    print(
        f"  score mode: {_score_mode_description(score_mode)}; "
        f"min quality: {min_quality}; T={temperature:.2f} K"
    )
    print(f"  ranked candidates: {len(ranked_rows)}/{candidate_count}")
    if not ranked_rows:
        print("  No candidates passed ranking score and quality filters.")
        return

    for row in ranked_rows[:top_n]:
        coords = "; ".join(row["driving_coords"])
        population = 100.0 * float(row["relative_population"])
        print(
            f"  {int(row['rank']):>2}. {row['candidate_id']} "
            f"parent={row['parent_id']} "
            f"score={float(row['ranking_score_delta_e']): .3f} kcal/mol "
            f"pop={population:6.2f}% "
            f"status={row['status']} "
            f"coords=[{coords}]"
        )


def _write_product_xyz_from_summary(
    summary: dict,
    output_path: str | Path,
    *,
    comment: str,
) -> tuple[str | None, str | None]:
    trajectory_source = summary.get("trajectory_source")
    if trajectory_source is None:
        return None, "No trajectory source was available for product extraction."

    try:
        trajectory = parse_molden_xyz_trajectory(trajectory_source)
    except Exception as exc:
        return None, f"Could not parse trajectory source for product extraction: {exc}"

    if not trajectory.frames:
        return None, "Trajectory source did not contain any frames."

    product_node = summary.get("product_node")
    if product_node is None:
        frame_index = len(trajectory.frames) - 1
    else:
        frame_index = max(0, min(int(product_node), len(trajectory.frames) - 1))

    frame = trajectory.frames[frame_index]
    path = write_xyz_frame(
        output_path,
        frame.symbols,
        frame.coordinates,
        comment=f"{comment}; source_node={frame_index}",
    )
    return str(path), None


def run_sampled_product_search(
    *,
    model_path: str | Path,
    xyz_file: str | Path,
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
    start_run_id: int = 0,
    sample_count: int = 10,
    sample_iterations: int = 3,
    resample_top_k: int = 3,
    print_top: int = 5,
    temperature: float = 298.15,
    sample_score_mode: str = "thermodynamic",
    sample_min_quality: str = "converged",
    sample_seed: int | None = 0,
    sample_modes: str | tuple[str, ...] = DEFAULT_SAMPLE_MODES,
    sample_include_hydrogen: bool = False,
    sample_add_max_distance: float = 5.0,
    sample_bond_scale: float = 1.20,
    sample_allow_shared_add_atoms: bool = False,
    sample_export_artifacts: bool = False,
    sample_reuse_calculator: bool = True,
) -> tuple[dict, Path, int]:
    if sample_iterations <= 0:
        raise ValueError("sample_iterations must be positive.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if resample_top_k <= 0:
        raise ValueError("resample_top_k must be positive.")
    if print_top <= 0:
        raise ValueError("print_top must be positive.")
    if sample_score_mode not in SAMPLE_SCORE_MODES:
        raise ValueError(
            "sample_score_mode must be one of: "
            + ", ".join(SAMPLE_SCORE_MODES)
        )
    if sample_min_quality not in SAMPLE_MIN_QUALITY_LEVELS:
        raise ValueError(
            "sample_min_quality must be one of: "
            + ", ".join(SAMPLE_MIN_QUALITY_LEVELS)
        )

    parsed_sample_modes = parse_sample_modes(sample_modes)
    model_path = Path(model_path).resolve()
    initial_xyz_path = Path(xyz_file).resolve()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = resolve_run_metadata(
        xyz_file=initial_xyz_path,
        label=label,
        reaction_label=reaction_label,
        formula=formula,
        case_kind=case_kind,
        source_fixture=source_fixture,
    )

    all_candidates: list[dict] = []
    iteration_reports: list[dict] = []
    shared_calculator = None
    shared_calculator_error = None
    if sample_reuse_calculator:
        from reactip import ReactIPCalculator

        try:
            shared_calculator = ReactIPCalculator(
                str(model_path),
                device=device or default_device(),
                chemical_symbols=CHEMICAL_SYMBOLS,
                max_force=max_force,
                max_abs_energy=max_abs_energy,
            )
        except Exception as exc:
            shared_calculator_error = f"{exc.__class__.__name__}: {exc}"
            print(
                "Could not initialize a shared ReactIP calculator for sampled "
                f"search; candidates will report this runtime error: {shared_calculator_error}"
            )
    frontier = [
        {
            "candidate_id": "root",
            "product_xyz": str(initial_xyz_path),
            "path_product_delta_e": 0.0,
            "path_kinetic_barrier_sum": 0.0,
            "path_rate_limiting_barrier_e": None,
            "path_candidate_ids": [],
        }
    ]
    next_run_id = start_run_id
    candidate_serial = 0
    started_at = perf_counter()

    for iteration in range(1, sample_iterations + 1):
        print()
        print(f"Sampling iteration {iteration}/{sample_iterations}")
        print(f"  Reactants in frontier: {len(frontier)}")

        iteration_candidates: list[dict] = []
        for parent_index, parent in enumerate(frontier):
            parent_xyz = Path(parent["product_xyz"]).resolve()
            parent_id = str(parent["candidate_id"])
            seed = None
            if sample_seed is not None:
                seed = int(sample_seed) + iteration * 100000 + parent_index * 1000

            sampled_sets = sample_driving_coordinate_sets(
                parent_xyz,
                sample_count=sample_count,
                random_seed=seed,
                include_hydrogen=sample_include_hydrogen,
                max_add_distance=sample_add_max_distance,
                bond_scale=sample_bond_scale,
                modes=parsed_sample_modes,
                allow_shared_add_atoms=sample_allow_shared_add_atoms,
            )
            print(
                f"  Parent {parent_id}: sampled {len(sampled_sets)} "
                f"candidate(s) from {parent_xyz}"
            )

            for sample_index, driving_coords in enumerate(sampled_sets, start=1):
                candidate_serial += 1
                candidate_id = f"cand_{candidate_serial:04d}"
                candidate_dir = (
                    run_dir
                    / f"iteration_{iteration:02d}"
                    / parent_id
                    / candidate_id
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                reactant_copy = candidate_dir / "reactant.xyz"
                shutil.copyfile(parent_xyz, reactant_copy)
                isomers_path = write_isomers_file(
                    candidate_dir / "isomers.txt",
                    driving_coords,
                )

                run_id = next_run_id
                next_run_id += 1
                if sample_reuse_calculator and shared_calculator is None:
                    summary = {
                        **metadata,
                        "status": "runtime_error_before_candidate_run",
                        "has_ts": False,
                        "ts_node": None,
                        "ts_energy": None,
                        "delta_e": None,
                        "product_node": None,
                        "product_energy": None,
                        "product_delta_e": None,
                        "score_delta_e": None,
                        "score_delta_e_source": None,
                        "trajectory_source": None,
                        "warnings": [],
                        "runtime_error": shared_calculator_error,
                    }
                    summary_path = write_summary_json(summary, candidate_dir / "summary.json")
                    candidate_exit_code = 1
                else:
                    summary, summary_path, candidate_exit_code = run_with_reporting(
                        model_path=model_path,
                        xyz_file=reactant_copy,
                        isomers_file=isomers_path,
                        run_dir=candidate_dir,
                        label=f"{metadata['case']}__{candidate_id}",
                        reaction_label=metadata["reaction_label"],
                        formula=metadata["formula"],
                        case_kind=metadata["case_kind"],
                        source_fixture=metadata["source_fixture"],
                        device=device,
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
                        run_id=run_id,
                        export_artifacts=sample_export_artifacts,
                        calculator=shared_calculator if sample_reuse_calculator else None,
                    )

                product_xyz, product_warning = _write_product_xyz_from_summary(
                    summary,
                    candidate_dir / "product.xyz",
                    comment=f"{candidate_id}; parent={parent_id}",
                )
                warnings = list(summary.get("warnings") or [])
                if product_warning is not None:
                    warnings.append(product_warning)

                candidate = {
                    "candidate_id": candidate_id,
                    "parent_id": parent_id,
                    "iteration": iteration,
                    "sample_index": sample_index,
                    "run_id": run_id,
                    "reactant_xyz": str(reactant_copy),
                    "isomers_file": str(isomers_path),
                    "driving_coords": driving_coords_to_lines(driving_coords),
                    "run_directory": str(candidate_dir),
                    "summary_path": str(summary_path),
                    "exit_code": candidate_exit_code,
                    "status": summary.get("status"),
                    "has_ts": summary.get("has_ts"),
                    "ts_node": summary.get("ts_node"),
                    "ts_energy": summary.get("ts_energy"),
                    "delta_e": summary.get("delta_e"),
                    "product_node": summary.get("product_node"),
                    "product_energy": summary.get("product_energy"),
                    "product_delta_e": summary.get("product_delta_e"),
                    "score_delta_e": summary.get("score_delta_e"),
                    "score_delta_e_source": summary.get("score_delta_e_source"),
                    "product_xyz": product_xyz,
                    "runtime_error": summary.get("runtime_error"),
                    "warnings": warnings,
                }
                _annotate_candidate_path(candidate, parent, sample_score_mode)
                iteration_candidates.append(candidate)
                all_candidates.append(candidate)

        ranked_iteration = _ranked_population_view(
            iteration_candidates,
            temperature=temperature,
            score_mode=sample_score_mode,
            min_quality=sample_min_quality,
        )
        _print_candidate_population_report(
            f"Top sampled candidates after iteration {iteration}",
            ranked_iteration,
            candidate_count=len(iteration_candidates),
            top_n=print_top,
            temperature=temperature,
            score_mode=sample_score_mode,
            min_quality=sample_min_quality,
        )

        iteration_reports.append(
            {
                "iteration": iteration,
                "parent_count": len(frontier),
                "candidate_count": len(iteration_candidates),
                "ranked_candidate_count": len(ranked_iteration),
                "top_candidates": ranked_iteration[:print_top],
            }
        )

        next_frontier = [
            {
                "candidate_id": row["candidate_id"],
                "product_xyz": row["product_xyz"],
                "path_product_delta_e": row.get("path_product_delta_e"),
                "path_kinetic_barrier_sum": row.get("path_kinetic_barrier_sum"),
                "path_rate_limiting_barrier_e": row.get("path_rate_limiting_barrier_e"),
                "path_candidate_ids": row.get("path_candidate_ids") or [],
            }
            for row in ranked_iteration
            if row.get("product_xyz") is not None
        ][:resample_top_k]
        if not next_frontier:
            print("  Stopping early: no ranked product XYZ files are available.")
            break
        frontier = next_frontier

    ranked_all = _ranked_population_view(
        all_candidates,
        temperature=temperature,
        score_mode=sample_score_mode,
        min_quality=sample_min_quality,
    )
    print()
    _print_candidate_population_report(
        "Overall top sampled candidates",
        ranked_all,
        candidate_count=len(all_candidates),
        top_n=print_top,
        temperature=temperature,
        score_mode=sample_score_mode,
        min_quality=sample_min_quality,
    )

    elapsed_seconds = perf_counter() - started_at
    search_summary = {
        **metadata,
        "mlip_model_name": model_path.name,
        "mlip_model_path": str(model_path),
        "initial_xyz_file": str(initial_xyz_path),
        "run_directory": str(run_dir),
        "search_parameters": {
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
            "start_run_id": start_run_id,
            "sample_count_per_reactant": sample_count,
            "sample_iterations": sample_iterations,
            "resample_top_k": resample_top_k,
            "print_top": print_top,
            "temperature": temperature,
            "sample_score_mode": sample_score_mode,
            "sample_min_quality": sample_min_quality,
            "sample_seed": sample_seed,
            "sample_modes": list(parsed_sample_modes),
            "sample_include_hydrogen": sample_include_hydrogen,
            "sample_add_max_distance": sample_add_max_distance,
            "sample_bond_scale": sample_bond_scale,
            "sample_allow_shared_add_atoms": sample_allow_shared_add_atoms,
            "sample_export_artifacts": sample_export_artifacts,
            "reuse_shared_calculator": sample_reuse_calculator and shared_calculator is not None,
            "shared_calculator_error": shared_calculator_error,
        },
        "iterations": iteration_reports,
        "candidate_count": len(all_candidates),
        "ranked_candidate_count": len(ranked_all),
        "candidates": all_candidates,
        "overall_top_candidates": ranked_all[:print_top],
        "elapsed_seconds": elapsed_seconds,
    }
    summary_path = write_summary_json(search_summary, run_dir / "candidate_search_summary.json")
    exit_code = 0 if ranked_all else 1
    return search_summary, summary_path, exit_code


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
    parser.add_argument(
        "--isomers",
        default=None,
        help="Driving coordinates (isomers) file. Required unless --sample-products is set.",
    )
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

    sampling = parser.add_argument_group("sampled product search")
    sampling.add_argument(
        "--sample-products",
        action="store_true",
        help="Sample candidate driving-coordinate sets and rank products instead of running one --isomers file.",
    )
    sampling.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of candidate products sampled per frontier reactant.",
    )
    sampling.add_argument(
        "--sample-iterations",
        type=int,
        default=3,
        help="Number of recursive sampling iterations.",
    )
    sampling.add_argument(
        "--resample-top-k",
        type=int,
        default=3,
        help="Number of top products used as reactants for the next sampling iteration.",
    )
    sampling.add_argument(
        "--print-top",
        type=int,
        default=5,
        help="Number of top candidates printed in Boltzmann population reports.",
    )
    sampling.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Temperature in K for relative populations proportional to exp(-beta dE).",
    )
    sampling.add_argument(
        "--sample-score-mode",
        choices=SAMPLE_SCORE_MODES,
        default="thermodynamic",
        help=(
            "Ranking score: thermodynamic uses cumulative product dE; "
            "kinetic uses a TST-like cumulative TS-barrier proxy."
        ),
    )
    sampling.add_argument(
        "--sample-min-quality",
        choices=SAMPLE_MIN_QUALITY_LEVELS,
        default="converged",
        help=(
            "Minimum candidate quality included in ranking. Use finite only for "
            "exploratory smokes; production defaults to converged."
        ),
    )
    sampling.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for deterministic candidate sampling.",
    )
    sampling.add_argument(
        "--sample-modes",
        default=",".join(DEFAULT_SAMPLE_MODES),
        help="Comma-separated sample modes: add, break, exchange, two_add.",
    )
    sampling.add_argument(
        "--sample-include-hydrogen",
        action="store_true",
        help="Allow sampled ADD/BREAK coordinates to include hydrogen atoms.",
    )
    sampling.add_argument(
        "--sample-add-max-distance",
        type=float,
        default=5.0,
        help="Maximum nonbonded atom distance considered for sampled ADD coordinates.",
    )
    sampling.add_argument(
        "--sample-bond-scale",
        type=float,
        default=1.20,
        help="Covalent-radius scale used to infer existing bonds before sampling.",
    )
    sampling.add_argument(
        "--sample-allow-shared-add-atoms",
        action="store_true",
        help="Allow two-ADD candidates where both new bonds share an atom.",
    )
    sampling.add_argument(
        "--sample-export-artifacts",
        action="store_true",
        help="Export SDF/GIF artifacts for every sampled candidate.",
    )
    sampling.add_argument(
        "--sample-reload-model-each-candidate",
        action="store_true",
        help="Disable shared-calculator reuse in sampled mode and reload the model for each candidate.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd().resolve()

    if not args.sample_products and not args.isomers:
        parser.error("--isomers is required unless --sample-products is set.")

    print("ReactIP SE-GSM research runner")
    print(f"  Label      : {args.label or infer_label_from_xyz(args.xyz)}")
    print(f"  XYZ        : {Path(args.xyz).resolve()}")
    if args.sample_products:
        print("  Isomers    : sampled")
        print(f"  Samples    : {args.sample_count} per reactant")
        print(f"  Iterations : {args.sample_iterations}")
        print(f"  Resample   : top {args.resample_top_k}")
        print(f"  Print top  : {args.print_top}")
        print(f"  Temperature: {args.temperature:.2f} K")
        print(f"  Score mode : {args.sample_score_mode}")
        print(f"  Min quality: {args.sample_min_quality}")
    else:
        print(f"  Isomers    : {Path(args.isomers).resolve()}")
    print(f"  Model      : {Path(args.model).resolve()}")
    print(f"  Output dir : {run_dir}")
    print(f"  Device     : {args.device}")
    print()

    if args.sample_products:
        summary, summary_path, exit_code = run_sampled_product_search(
            model_path=args.model,
            xyz_file=args.xyz,
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
            start_run_id=args.ID,
            sample_count=args.sample_count,
            sample_iterations=args.sample_iterations,
            resample_top_k=args.resample_top_k,
            print_top=args.print_top,
            temperature=args.temperature,
            sample_score_mode=args.sample_score_mode,
            sample_min_quality=args.sample_min_quality,
            sample_seed=args.sample_seed,
            sample_modes=args.sample_modes,
            sample_include_hydrogen=args.sample_include_hydrogen,
            sample_add_max_distance=args.sample_add_max_distance,
            sample_bond_scale=args.sample_bond_scale,
            sample_allow_shared_add_atoms=args.sample_allow_shared_add_atoms,
            sample_export_artifacts=args.sample_export_artifacts,
            sample_reuse_calculator=not args.sample_reload_model_each_candidate,
        )
        print()
        print(f"Candidate search summary JSON : {summary_path}")
        print(f"Candidate count               : {summary['candidate_count']}")
        raise SystemExit(exit_code)

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
