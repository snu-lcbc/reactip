"""SE-GSM (Single-Ended Growing String Method) with ReactIP.

This module exposes a Python-first SE-GSM workflow built on pyGSM and a
dedicated :class:`~reactip.reactip_lot.ReactIPLoT` adapter. The ML model is
loaded once per run and kept in-process for the full string optimization.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

def _append_local_pygsm_path() -> None:
    module_path = Path(__file__).resolve()
    candidates: list[Path] = []

    env_path = os.environ.get("REACTIP_PYGSM_DIR")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # Local development fallbacks (only used when pyGSM is not pip-installed and
    # REACTIP_PYGSM_DIR is unset). The pinned dependency is the snu-lcbc fork
    # (pyGSM @ git+https://github.com/snu-lcbc/pyGSM.git@reactip-compat); this
    # matches the local checkout layout used during development.
    workspace_root = module_path.parents[2]          # .../workdir_reactip
    candidates.append(workspace_root / "growth_string_method" / "pyGSM")
    candidates.append(workspace_root / "se-gsm" / "pyGSM")  # legacy layout

    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
            return


try:
    from pyGSM.coordinate_systems import (
        DelocalizedInternalCoordinates,
        Distance,
        PrimitiveInternalCoordinates,
        Topology,
    )
    from pyGSM.growing_string_methods import SE_GSM
    from pyGSM.level_of_theories.base_lot import Lot
    from pyGSM.molecule import Molecule
    from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
    from pyGSM.optimizers.lbfgs import lbfgs
    from pyGSM.potential_energy_surfaces import PES
    from pyGSM.utilities import elements, manage_xyz, nifty
    from pyGSM.utilities.cli_utils import get_driving_coord_prim
except ModuleNotFoundError as exc:
    if exc.name != "pyGSM":
        raise
    _append_local_pygsm_path()
    from pyGSM.coordinate_systems import (
        DelocalizedInternalCoordinates,
        Distance,
        PrimitiveInternalCoordinates,
        Topology,
    )
    from pyGSM.growing_string_methods import SE_GSM
    from pyGSM.level_of_theories.base_lot import Lot
    from pyGSM.molecule import Molecule
    from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
    from pyGSM.optimizers.lbfgs import lbfgs
    from pyGSM.potential_energy_surfaces import PES
    from pyGSM.utilities import elements, manage_xyz, nifty
    from pyGSM.utilities.cli_utils import get_driving_coord_prim

from .mlip_calculator import ReactIPCalculator  # noqa: F401 — re-exported
from .reactip_lot import DEFAULT_STATE_KEY, ReactIPLoT, normalize_calculator_registry

StateKey = tuple[int, int, int]


def read_isomers_file(filepath: str) -> list:
    """Parse a driving-coordinates file in pyGSM's native format."""
    with open(filepath) as f:
        lines = [line.rstrip() for line in f if line.strip()]

    driving_coordinates = []
    start = 1 if lines and lines[0] == "NEW" else 0

    for line in lines[start:]:
        dc = []
        two_ints = False
        three_ints = False
        four_ints = False
        for i, elem in enumerate(line.split()):
            if i == 0:
                dc.append(elem)
                if elem in ("ADD", "BREAK"):
                    two_ints = True
                elif elem in ("ANGLE", "ROTATE"):
                    three_ints = True
                elif elem in ("TORSION", "OOP"):
                    four_ints = True
            else:
                if two_ints and i > 2:
                    dc.append(float(elem))
                elif three_ints and i > 3:
                    dc.append(float(elem))
                elif four_ints and i > 4:
                    dc.append(float(elem))
                else:
                    dc.append(int(elem))
        driving_coordinates.append(dc)
    return driving_coordinates


def _require_supported_state(charge: int, multiplicity: int, adiabatic_state: int) -> None:
    """Validate (and permit) the requested electronic state.

    The ReactIP MLIP maps (atomic positions, species) -> energy/forces and has
    **no spin or charge input channel** (see ``ReactIPLoT.run`` and
    ``ReactIPCalculator.calculate``, which pass only the ASE ``atoms``). Charge,
    spin multiplicity and adiabatic-state index are therefore pyGSM *bookkeeping*
    only: for a given geometry the potential-energy surface is identical for
    every state.

    Non-default states used to hard-fail. They are now permitted so the
    singlet-trained model can be *probed* on open-shell systems (e.g. ``RH + 3O2``
    H-abstraction on the triplet surface), but a warning is emitted: the returned
    energies/forces are the singlet-trained values regardless of the requested
    ``(charge, multiplicity, state)``, so any open-shell run is purely
    extrapolative and the numbers will match the corresponding singlet run.
    """
    if not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError(f"multiplicity must be a positive integer, got {multiplicity!r}.")
    if not isinstance(adiabatic_state, int) or adiabatic_state < 0:
        raise ValueError(f"adiabatic_state must be a non-negative integer, got {adiabatic_state!r}.")
    requested = (charge, multiplicity, adiabatic_state)
    if requested != DEFAULT_STATE_KEY:
        warnings.warn(
            f"Requested electronic state {requested} differs from the trained "
            f"default {DEFAULT_STATE_KEY}. The ReactIP MLIP has no spin/charge "
            "input: energy and forces depend on geometry and species only, so the "
            "PES (barriers, products) is identical for every state. The requested "
            "state is bookkeeping only; results stay the singlet-trained values "
            "and are extrapolative for open-shell/charged systems.",
            stacklevel=2,
        )


def build_reactip_calculator(
    model_path: str,
    device: str = "cuda",
    chemical_symbols: Optional[Sequence[str] | Mapping[str, str]] = None,
    max_force: Optional[float] = 100.0,
    max_abs_energy: Optional[float] = 10000.0,
    energy_units: str = "eV",
) -> ReactIPCalculator:
    """Build the ASE-native ReactIP calculator used by SE-GSM."""
    return ReactIPCalculator(
        model_path,
        device=device,
        chemical_symbols=chemical_symbols,
        energy_units=energy_units,
        max_force=max_force,
        max_abs_energy=max_abs_energy,
    )


def build_reactip_lot(
    geom,
    *,
    model_path: Optional[str] = None,
    calculator: Optional[ReactIPCalculator] = None,
    calculator_registry: Optional[Mapping[StateKey, ReactIPCalculator]] = None,
    device: str = "cuda",
    chemical_symbols: Optional[Sequence[str] | Mapping[str, str]] = None,
    charge: int = 0,
    multiplicity: int = 1,
    adiabatic_state: int = 0,
    max_force: Optional[float] = 100.0,
    max_abs_energy: Optional[float] = 10000.0,
    energy_units: str = "eV",
    ID: int = 0,
) -> ReactIPLoT:
    """Build a pyGSM-native level of theory backed by the ML potential."""
    _require_supported_state(charge, multiplicity, adiabatic_state)

    loader_inputs = [model_path is not None, calculator is not None, calculator_registry is not None]
    if sum(loader_inputs) > 1:
        raise ValueError(
            "Provide exactly one of model_path, calculator, or calculator_registry "
            "when building a ReactIPLoT."
        )

    if calculator is None and calculator_registry is None:
        if model_path is None:
            raise ValueError(
                "build_reactip_lot requires one of model_path, calculator, or "
                "calculator_registry."
            )
        calculator = build_reactip_calculator(
            model_path=model_path,
            device=device,
            chemical_symbols=chemical_symbols,
            max_force=max_force,
            max_abs_energy=max_abs_energy,
            energy_units=energy_units,
        )

    registry = normalize_calculator_registry(
        calculator=calculator,
        calculator_registry=calculator_registry,
        default_state_key=(charge, multiplicity, adiabatic_state),
    )

    return ReactIPLoT.from_options(
        calculator_registry=registry,
        geom=geom,
        states=[(multiplicity, adiabatic_state)],
        charge=charge,
        ID=ID,
    )


def build_nequip_lot(*args, **kwargs) -> ReactIPLoT:
    """Backward-compatible alias for the previous public helper name."""
    return build_reactip_lot(*args, **kwargs)


def _build_internal_coordinates(geom, driving_coords: list, coordinate_type: str):
    nifty.printcool("Building topology and internal coordinates")
    element_table = elements.ElementData()
    atom_symbols = manage_xyz.get_atoms(geom)
    atoms = [element_table.from_symbol(symbol) for symbol in atom_symbols]
    xyz = manage_xyz.xyz_to_np(geom)

    top = Topology.build_topology(xyz, atoms)

    driving_coord_prims = []
    for dc in driving_coords:
        prim = get_driving_coord_prim(dc)
        if prim is not None:
            driving_coord_prims.append(prim)

    for prim in driving_coord_prims:
        if isinstance(prim, Distance):
            bond = (prim.atoms[0], prim.atoms[1])
            if bond not in top.edges() and (bond[1], bond[0]) not in top.edges():
                print(f"  Adding driving coord bond {bond} to topology")
                top.add_edge(bond[0], bond[1])

    connect = coordinate_type == "DLC"
    addtr = coordinate_type == "TRIC"
    addcart = coordinate_type == "HDLC"

    primitives = PrimitiveInternalCoordinates.from_options(
        xyz=xyz,
        atoms=atoms,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        topology=top,
    )

    for prim in driving_coord_prims:
        if not isinstance(prim, Distance) and prim not in primitives.Internals:
            print(f"  Adding driving coord prim {prim} to internals")
            primitives.append_prim_to_block(prim)

    coord_obj = DelocalizedInternalCoordinates.from_options(
        xyz=xyz,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=primitives,
    )
    return coord_obj


def _build_optimizer(name: str):
    nifty.printcool("Building optimizer")
    if name == "eigenvector_follow":
        return eigenvector_follow.from_options(
            print_level=1,
            Linesearch="NoLineSearch",
            update_hess_in_bg=True,
            DMAX=0.1,
            conv_Ediff=100.0,
            conv_gmax=100.0,
        )
    if name == "lbfgs":
        return lbfgs.from_options(
            print_level=1,
            Linesearch="NoLineSearch",
            update_hess_in_bg=False,
            DMAX=0.1,
            conv_Ediff=100.0,
            conv_gmax=100.0,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _ase_atoms_to_pygsm_geom(atoms) -> list[list[object]]:
    return [
        [symbol, float(x), float(y), float(z)]
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    ]


def _read_xyz_geometries(xyz_file: str | os.PathLike[str]) -> list[list[list[object]]]:
    """Read simple XYZ or ASE extended XYZ into pyGSM geometry records."""
    try:
        return manage_xyz.read_xyzs(str(xyz_file))
    except Exception as pygsm_exc:
        try:
            import ase.io

            frames = ase.io.read(str(xyz_file), index=":")
        except Exception as ase_exc:
            raise ValueError(
                f"Could not read XYZ file {xyz_file!s} with pyGSM or ASE. "
                f"pyGSM error: {pygsm_exc}; ASE error: {ase_exc}"
            ) from ase_exc

        if not isinstance(frames, list):
            frames = [frames]
        if not frames:
            raise ValueError(f"XYZ file {xyz_file!s} did not contain any frames")
        return [_ase_atoms_to_pygsm_geom(atoms) for atoms in frames]


def _determine_status(
    *,
    is_converged: bool,
    has_ts: bool,
    ran_out: bool,
    end_early: bool,
) -> str:
    if is_converged and has_ts:
        return "converged_ts"
    if is_converged and not has_ts:
        return "converged_no_ts"
    if ran_out and has_ts:
        return "ran_out_with_ts_candidate"
    if ran_out and not has_ts:
        return "ran_out_no_ts"
    if end_early and has_ts:
        return "ended_early_with_ts_candidate"
    if end_early and not has_ts:
        return "ended_early_no_ts"
    if has_ts:
        return "completed_with_ts_candidate"
    return "completed_no_ts"


def _analyze_gsm_result(gsm: SE_GSM) -> dict:
    energies = list(gsm.energies)
    nnodes = len(gsm.geometries)
    npeaks = int(gsm.npeaks)
    is_converged = bool(getattr(gsm, "isConverged", False))
    ran_out = bool(getattr(gsm, "ran_out", False))
    end_early = bool(getattr(gsm, "end_early", False))
    has_ts = npeaks == 1 and not end_early

    ts_node = int(gsm.TSnode) if has_ts else None
    ts_energy = None
    delta_e = None
    reactant_node = 0 if energies else None
    product_node = None
    product_energy = None
    product_delta_e = None
    score_delta_e = None
    score_delta_e_source = None

    if ts_node is not None:
        reactant_node = int(np.argmin(energies[: ts_node + 1]))
        product_node = ts_node + int(np.argmin(energies[ts_node:]))
        ts_energy = energies[ts_node] - energies[reactant_node]
        delta_e = energies[product_node] - energies[reactant_node]
        product_delta_e = delta_e
        score_delta_e = delta_e
        score_delta_e_source = "post_ts_minimum"
    elif len(energies) >= 2:
        product_node = len(energies) - 1
        product_delta_e = energies[product_node] - energies[reactant_node]
        score_delta_e = product_delta_e
        score_delta_e_source = "endpoint"

    if product_node is not None:
        product_energy = energies[product_node]

    status = _determine_status(
        is_converged=is_converged,
        has_ts=has_ts,
        ran_out=ran_out,
        end_early=end_early,
    )

    return {
        "status": status,
        "converged": is_converged,
        "has_ts": has_ts,
        "nnodes": nnodes,
        "npeaks": npeaks,
        "ran_out": ran_out,
        "end_early": end_early,
        "ts_node": ts_node,
        "ts_energy": ts_energy,
        "delta_e": delta_e,
        "reactant_node": reactant_node,
        "product_node": product_node,
        "product_energy": product_energy,
        "product_delta_e": product_delta_e,
        "score_delta_e": score_delta_e,
        "score_delta_e_source": score_delta_e_source,
        "energies": energies,
        "ts_imaginary_mode_count": None,
        "ts_is_first_order_saddle": None,
        "ts_imaginary_frequencies_cm": None,
        "ts_lowest_real_frequency_cm": None,
        "ts_frequency_threshold_cm": None,
        "ts_verification_error": None,
    }


def _verify_ts_or_warning(
    *,
    lot,
    geom,
    multiplicity: int,
    adiabatic_state: int,
    displacement: float,
) -> dict:
    """Run the MLIP Hessian / imaginary-frequency check on a TS node geometry.

    Returns a dict of ``ts_*`` fields. Failures are non-fatal: they are reported
    via ``ts_verification_error`` so the surrounding search keeps running.
    """
    from .ts_validation import verify_transition_state

    try:
        registry = getattr(lot, "calculator_registry", None)
        if not registry:
            return {"ts_verification_error": "no calculator available on the level of theory"}
        calculator = registry.get((getattr(lot, "charge", 0), multiplicity, adiabatic_state))
        if calculator is None:
            calculator = next(iter(registry.values()))
        symbols = manage_xyz.get_atoms(geom)
        coordinates = manage_xyz.xyz_to_np(geom)
        analysis = verify_transition_state(
            calculator, symbols, coordinates, displacement=displacement
        )
        payload = analysis.as_dict()
        payload["ts_verification_error"] = None
        print(
            f"  TS verification: {analysis.imaginary_mode_count} imaginary mode(s); "
            f"first-order saddle={analysis.is_first_order_saddle}"
        )
        return payload
    except Exception as exc:  # noqa: BLE001 — verification must never abort a run
        return {"ts_verification_error": f"{exc.__class__.__name__}: {exc}"}


def run_se_gsm(
    *,
    model_path: Optional[str] = None,
    xyz_file: str,
    driving_coords,
    calculator: Optional[ReactIPCalculator] = None,
    calculator_registry: Optional[Mapping[StateKey, ReactIPCalculator]] = None,
    lot: Optional[Lot] = None,
    device: str = "cuda",
    chemical_symbols: Optional[Sequence[str] | Mapping[str, str]] = None,
    energy_units: str = "eV",
    charge: int = 0,
    multiplicity: int = 1,
    adiabatic_state: int = 0,
    num_nodes: int = 20,
    max_gsm_iters: int = 100,
    max_opt_steps: int = 20,
    conv_tol: float = 0.0005,
    add_node_tol: float = 0.01,
    dqmag_max: float = 0.8,
    bdist_ratio: float = 0.5,
    optimizer: str = "eigenvector_follow",
    coordinate_type: str = "TRIC",
    rtype: int = 2,
    max_force: float = 100.0,
    max_abs_energy: float = 10000.0,
    reactant_geom_fixed: bool = False,
    verify_ts: bool = False,
    ts_hessian_displacement: float = 0.005,
    ID: int = 0,
) -> dict:
    """Run SE-GSM with the ML potential as the single-state backend.

    pyGSM writes all output files (``grown_string_*.xyz``, ``scratch/``,
    ``TSnode_*.xyz``) relative to the **current working directory**.  To direct
    output to a dedicated folder, ``os.chdir`` to it *before* calling this
    function (see ``example_se_gsm.py`` and ``run_se_gsm.slurm`` for the
    recommended pattern).
    """

    if isinstance(driving_coords, str):
        driving_coords = read_isomers_file(driving_coords)

    geoms = _read_xyz_geometries(xyz_file)
    geom = geoms[0]

    if lot is not None and any(
        value is not None for value in (model_path, calculator, calculator_registry)
    ):
        raise ValueError(
            "Provide either lot or one of model_path/calculator/calculator_registry, not both."
        )

    if lot is None:
        nifty.printcool("Loading ReactIP model")
        if model_path is not None:
            print(f"  model: {model_path}")
        elif calculator is not None:
            print("  model: using prebuilt calculator")
        else:
            print("  model: using calculator registry")
        print(f"  device: {device}")
        lot = build_reactip_lot(
            geom,
            model_path=model_path,
            calculator=calculator,
            calculator_registry=calculator_registry,
            device=device,
            chemical_symbols=chemical_symbols,
            charge=charge,
            multiplicity=multiplicity,
            adiabatic_state=adiabatic_state,
            max_force=max_force,
            max_abs_energy=max_abs_energy,
            energy_units=energy_units,
            ID=ID,
        )
    else:
        lot = type(lot).copy(lot, {"ID": ID, "node_id": 0})

    pes = PES.from_options(lot=lot, ad_idx=adiabatic_state, multiplicity=multiplicity)
    coord_obj = _build_internal_coordinates(geom, driving_coords, coordinate_type)

    nifty.printcool("Building reactant molecule")
    form_hessian = optimizer == "eigenvector_follow"
    reactant = Molecule.from_options(
        geom=geom,
        PES=pes,
        coord_obj=coord_obj,
        Form_Hessian=form_hessian,
    )

    opt = _build_optimizer(optimizer)

    if not reactant_geom_fixed:
        nifty.printcool("Pre-optimizing reactant geometry")
        path = os.path.join(os.getcwd(), f"scratch/{ID:03d}/0/")
        opt.optimize(
            molecule=reactant,
            refE=reactant.energy,
            opt_steps=100,
            path=path,
        )

    nifty.printcool("Building SE-GSM object")
    gsm = SE_GSM.from_options(
        reactant=reactant,
        nnodes=num_nodes,
        DQMAG_MAX=dqmag_max,
        BDIST_RATIO=bdist_ratio,
        CONV_TOL=conv_tol,
        ADD_NODE_TOL=add_node_tol,
        optimizer=opt,
        print_level=1,
        driving_coords=driving_coords,
        ID=ID,
        mp_cores=1,
        interp_method="DLC",
    )

    nifty.printcool("Running SE-GSM")
    gsm.go_gsm(max_gsm_iters, max_opt_steps, rtype=rtype)

    result = _analyze_gsm_result(gsm)
    result.update(
        {
            "geometries": gsm.geometries,
            "gsm": gsm,
            "lot": lot,
        }
    )

    if result["has_ts"]:
        print(f"\n  TS node: {result['ts_node']}")
        print(f"  TS barrier: {result['ts_energy']:.2f} kcal/mol")
        print(f"  Delta E (rxn): {result['delta_e']:.2f} kcal/mol")
        ts_geom = gsm.nodes[result["ts_node"]].geometry
        manage_xyz.write_xyz(f"TSnode_{ID}.xyz", ts_geom)
        if verify_ts:
            result.update(
                _verify_ts_or_warning(
                    lot=lot,
                    geom=ts_geom,
                    multiplicity=multiplicity,
                    adiabatic_state=adiabatic_state,
                    displacement=ts_hessian_displacement,
                )
            )
    else:
        print(f"\n  No unique TS was identified. status={result['status']}")
        if result["score_delta_e"] is not None:
            print(
                "  Candidate endpoint dE: "
                f"{result['score_delta_e']:.2f} kcal/mol "
                f"({result['score_delta_e_source']})"
            )

    return result


def run_se_gsm_with_calculator(
    calculator: ReactIPCalculator,
    *,
    xyz_file: str,
    driving_coords,
    **kwargs,
) -> dict:
    """Convenience wrapper for prebuilt-calculator injection."""
    return run_se_gsm(
        calculator=calculator,
        xyz_file=xyz_file,
        driving_coords=driving_coords,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SE-GSM transition state finding with ReactIP",
    )
    parser.add_argument("--model", required=True, help="Path to compiled model or checkpoint")
    parser.add_argument("--xyz", required=True, help="Reactant XYZ file")
    parser.add_argument("--isomers", required=True, help="Driving coordinates (isomers) file")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge (bookkeeping only; the MLIP has no charge input)",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help=(
            "Spin multiplicity, e.g. 3 for a triplet O2 system (bookkeeping only; "
            "the MLIP has no spin input so the PES is identical to the singlet run)"
        ),
    )
    parser.add_argument(
        "--adiabatic-state",
        type=int,
        default=0,
        help="Adiabatic state index (v1 supports only 0)",
    )
    parser.add_argument("--num-nodes", type=int, default=20, help="Max string nodes (default: 20)")
    parser.add_argument("--max-iters", type=int, default=100, help="Max GSM iterations (default: 100)")
    parser.add_argument("--max-opt-steps", type=int, default=20, help="Max opt steps per cycle (default: 20)")
    parser.add_argument("--conv-tol", type=float, default=0.0005, help="TS convergence tolerance (default: 0.0005)")
    parser.add_argument("--optimizer", default="eigenvector_follow", choices=["eigenvector_follow", "lbfgs"])
    parser.add_argument("--coord-type", default="TRIC", choices=["TRIC", "DLC", "HDLC"])
    parser.add_argument(
        "--rtype",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="0=no climb, 1=climb only, 2=find+climb (default: 2)",
    )
    parser.add_argument("--no-pre-opt", action="store_true", help="Skip reactant pre-optimization")
    parser.add_argument("--max-force", type=float, default=100.0, help="Safety: max force threshold eV/A (default: 100)")
    parser.add_argument("--max-abs-energy", type=float, default=10000.0, help="Safety: max |energy| threshold eV (default: 10000)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for pyGSM output (grown_string_*.xyz, scratch/, TSnode_*.xyz). "
             "Created if absent. Defaults to cwd.",
    )
    parser.add_argument("--ID", type=int, default=0, help="String ID (default: 0)")

    args = parser.parse_args()

    # Resolve input paths to absolute before any chdir.
    model_path   = os.path.abspath(args.model)
    xyz_file     = os.path.abspath(args.xyz)
    isomers_file = os.path.abspath(args.isomers)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)

    results = run_se_gsm(
        model_path=model_path,
        xyz_file=xyz_file,
        driving_coords=isomers_file,
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
        reactant_geom_fixed=args.no_pre_opt,
        max_force=args.max_force,
        max_abs_energy=args.max_abs_energy,
        ID=args.ID,
    )

    if results["has_ts"]:
        print(
            f"\nSE-GSM completed. status={results['status']} "
            f"TS barrier = {results['ts_energy']:.2f} kcal/mol"
        )
    else:
        print(f"\nSE-GSM completed. status={results['status']}")


if __name__ == "__main__":
    main()
