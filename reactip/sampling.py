"""Candidate driving-coordinate sampling and Boltzmann ranking helpers."""

from __future__ import annotations

import math
import random
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DrivingCoord = tuple[str, int, int]
DrivingCoordSet = tuple[DrivingCoord, ...]

R_KCAL_PER_MOL_K = 0.00198720425864083
DEFAULT_SAMPLE_MODES: tuple[str, ...] = ("exchange", "add", "break", "two_add")

_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
}


def read_xyz_frame(path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read the first frame from a simple XYZ/extXYZ file."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file is too short: {path}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First XYZ line must be an atom count: {path}") from exc

    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"XYZ file has {len(atom_lines)} atom lines, expected {natoms}: {path}")

    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line in {path}: {line!r}")
        symbols.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return tuple(symbols), np.asarray(coordinates, dtype=float)


def write_xyz_frame(
    path: str | Path,
    symbols: Sequence[str],
    coordinates: np.ndarray,
    *,
    comment: str,
) -> Path:
    """Write one XYZ frame."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), comment]
    for symbol, xyz in zip(symbols, coordinates):
        lines.append(f"{symbol:<2} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}")
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def infer_bond_pairs(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    *,
    scale: float = 1.20,
) -> set[tuple[int, int]]:
    """Infer zero-based covalent bond pairs from distances."""
    bonds: set[tuple[int, int]] = set()
    for i, symbol_i in enumerate(symbols):
        radius_i = _COVALENT_RADII.get(symbol_i, 0.77)
        for j in range(i + 1, len(symbols)):
            radius_j = _COVALENT_RADII.get(symbols[j], 0.77)
            cutoff = scale * (radius_i + radius_j)
            distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
            if distance <= cutoff:
                bonds.add((i, j))
    return bonds


def bond_signature(xyz_path: str | Path, *, scale: float = 1.20) -> tuple[str, ...]:
    """Return a hashable element-aware bond signature for one structure.

    Atom ordering is preserved throughout the sampled search (products are
    copied forward as the next reactant), so two structures that share this
    signature have the same covalent connectivity and are treated as the same
    product species for de-duplication.
    """
    symbols, coordinates = read_xyz_frame(xyz_path)
    bonds = infer_bond_pairs(symbols, coordinates, scale=scale)
    return tuple(
        sorted(f"{symbols[i]}{i}-{symbols[j]}{j}" for i, j in bonds)
    )


def count_bond_changes(
    reactant_xyz: str | Path,
    product_xyz: str | Path,
    *,
    scale: float = 1.20,
) -> tuple[int, int]:
    """Count covalent bonds added and removed between two XYZ structures.

    Returns ``(added, removed)`` from the reactant to the product bond graph.
    Raises ``ValueError`` when the structures have mismatched atoms.
    """
    r_symbols, r_coords = read_xyz_frame(reactant_xyz)
    p_symbols, p_coords = read_xyz_frame(product_xyz)
    if r_symbols != p_symbols:
        raise ValueError(
            "Reactant and product XYZ files have different atom symbols: "
            f"{reactant_xyz} vs {product_xyz}"
        )
    reactant_bonds = infer_bond_pairs(r_symbols, r_coords, scale=scale)
    product_bonds = infer_bond_pairs(p_symbols, p_coords, scale=scale)
    return (
        len(product_bonds - reactant_bonds),
        len(reactant_bonds - product_bonds),
    )


def normalize_driving_coord(coord: Sequence[object]) -> DrivingCoord:
    """Normalize ADD/BREAK driving coordinates to a stable 1-based tuple."""
    if len(coord) != 3:
        raise ValueError(f"Only ADD/BREAK coordinate triplets are supported: {coord!r}")
    op = str(coord[0]).upper()
    if op not in {"ADD", "BREAK"}:
        raise ValueError(f"Unsupported sampled driving coordinate operation: {op!r}")
    a = int(coord[1])
    b = int(coord[2])
    if a == b:
        raise ValueError(f"Driving coordinate cannot use the same atom twice: {coord!r}")
    lo, hi = sorted((a, b))
    return (op, lo, hi)


def normalize_driving_coord_set(coords: Iterable[Sequence[object]]) -> DrivingCoordSet:
    """Normalize a coordinate set while preserving BREAK-before-ADD order."""
    normalized = [normalize_driving_coord(coord) for coord in coords]
    break_coords = sorted(coord for coord in normalized if coord[0] == "BREAK")
    add_coords = sorted(coord for coord in normalized if coord[0] == "ADD")
    return tuple(break_coords + add_coords)


def driving_coords_to_lines(coords: Sequence[DrivingCoord]) -> list[str]:
    return [f"{op} {a} {b}" for op, a, b in coords]


def write_isomers_file(path: str | Path, coords: Sequence[DrivingCoord]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(driving_coords_to_lines(coords)) + "\n")
    return output_path


def parse_sample_modes(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SAMPLE_MODES
    if isinstance(value, str):
        modes = [mode.strip().lower() for mode in value.split(",") if mode.strip()]
    else:
        modes = [str(mode).strip().lower() for mode in value if str(mode).strip()]
    allowed = {"add", "break", "exchange", "two_add"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown sample mode(s): {', '.join(unknown)}")
    if not modes:
        raise ValueError("At least one sample mode is required.")
    return tuple(dict.fromkeys(modes))


def _allowed_atoms(symbols: Sequence[str], include_hydrogen: bool) -> list[int]:
    if include_hydrogen:
        return list(range(len(symbols)))
    heavy = [index for index, symbol in enumerate(symbols) if symbol != "H"]
    return heavy if len(heavy) >= 2 else list(range(len(symbols)))


def _pair_distance(coordinates: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(coordinates[i] - coordinates[j]))


def generate_driving_coordinate_pool(
    xyz_path: str | Path,
    *,
    include_hydrogen: bool = False,
    max_add_distance: float = 5.0,
    bond_scale: float = 1.20,
    modes: str | Sequence[str] | None = None,
    allow_shared_add_atoms: bool = False,
    max_pair_pool: int = 200,
    max_exchange_targets: int = 8,
    max_pool_size: int = 10000,
) -> list[DrivingCoordSet]:
    """Generate a deterministic pool of possible ADD/BREAK coordinate sets.

    The pool is heuristic: it proposes pyGSM driving-coordinate sets from the
    current geometry; SE-GSM still determines whether the proposal becomes a
    meaningful product candidate.
    """
    sample_modes = parse_sample_modes(modes)
    symbols, coordinates = read_xyz_frame(xyz_path)
    allowed_atoms = _allowed_atoms(symbols, include_hydrogen)
    allowed_set = set(allowed_atoms)
    bonds = infer_bond_pairs(symbols, coordinates, scale=bond_scale)

    pair_distances = [
        (i, j, _pair_distance(coordinates, i, j))
        for i, j in combinations(allowed_atoms, 2)
    ]
    pair_distances.sort(key=lambda item: (item[2], item[0], item[1]))

    break_pairs = [
        (i, j)
        for i, j in sorted(bonds)
        if i in allowed_set and j in allowed_set
    ][:max_pair_pool]
    add_pairs = [
        (i, j)
        for i, j, distance in pair_distances
        if (i, j) not in bonds and distance <= max_add_distance
    ]
    if not add_pairs:
        add_pairs = [(i, j) for i, j, _ in pair_distances if (i, j) not in bonds]
    add_pairs = add_pairs[:max_pair_pool]

    pool: set[DrivingCoordSet] = set()

    def add_to_pool(coords: Iterable[Sequence[object]]) -> None:
        if len(pool) >= max_pool_size:
            return
        pool.add(normalize_driving_coord_set(coords))

    if "add" in sample_modes:
        for i, j in add_pairs:
            add_to_pool((("ADD", i + 1, j + 1),))

    if "break" in sample_modes:
        for i, j in break_pairs:
            add_to_pool((("BREAK", i + 1, j + 1),))

    if "exchange" in sample_modes:
        for a, b in break_pairs:
            for old_anchor, moving_atom in ((a, b), (b, a)):
                new_anchors = [
                    (candidate, _pair_distance(coordinates, moving_atom, candidate))
                    for candidate in allowed_atoms
                    if candidate not in {old_anchor, moving_atom}
                    and tuple(sorted((moving_atom, candidate))) not in bonds
                    and _pair_distance(coordinates, moving_atom, candidate) <= max_add_distance
                ]
                new_anchors.sort(key=lambda item: (item[1], item[0]))
                for new_anchor, _ in new_anchors[:max_exchange_targets]:
                    add_to_pool(
                        (
                            ("BREAK", old_anchor + 1, moving_atom + 1),
                            ("ADD", new_anchor + 1, moving_atom + 1),
                        )
                    )

    if "two_add" in sample_modes:
        for (a, b), (c, d) in combinations(add_pairs, 2):
            if not allow_shared_add_atoms and len({a, b, c, d}) < 4:
                continue
            add_to_pool((("ADD", a + 1, b + 1), ("ADD", c + 1, d + 1)))
            if len(pool) >= max_pool_size:
                break

    return sorted(pool)


def sample_driving_coordinate_sets(
    xyz_path: str | Path,
    *,
    sample_count: int = 10,
    random_seed: int | None = None,
    **pool_kwargs,
) -> list[DrivingCoordSet]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    pool = generate_driving_coordinate_pool(xyz_path, **pool_kwargs)
    rng = random.Random(random_seed)
    rng.shuffle(pool)
    return pool[:sample_count]


def compute_boltzmann_populations(
    delta_e_values: Sequence[float | None],
    *,
    temperature: float = 298.15,
) -> list[dict[str, float | None]]:
    """Compute normalized populations proportional to exp(-beta dE)."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    beta = 1.0 / (R_KCAL_PER_MOL_K * temperature)
    valid: list[tuple[int, float, float]] = []
    for index, value in enumerate(delta_e_values):
        if value is None:
            continue
        delta_e = float(value)
        if math.isfinite(delta_e):
            valid.append((index, delta_e, -beta * delta_e))

    populations: list[dict[str, float | None]] = [
        {
            "boltzmann_log_factor": None,
            "relative_population": None,
        }
        for _ in delta_e_values
    ]
    if not valid:
        return populations

    max_log_factor = max(log_factor for _, _, log_factor in valid)
    shifted_weights = [
        (index, math.exp(log_factor - max_log_factor))
        for index, _, log_factor in valid
    ]
    total_weight = sum(weight for _, weight in shifted_weights)
    if total_weight <= 0.0:
        return populations

    for index, weight in shifted_weights:
        populations[index]["relative_population"] = weight / total_weight
    for index, _, log_factor in valid:
        populations[index]["boltzmann_log_factor"] = log_factor
    return populations
