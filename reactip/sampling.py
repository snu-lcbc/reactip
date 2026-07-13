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

# Pool-generation strategies selectable from run_se_gsm.py. ``rule_based`` is
# the recommended default (valence + symmetry pruned); ``exhaustive`` is the
# legacy geometric enumerator kept for benchmark reproducibility.
POOL_STRATEGIES: tuple[str, ...] = ("rule_based", "exhaustive")

# --------------------------------------------------------------------------- #
# Rule-based generation: valence connection limits
# --------------------------------------------------------------------------- #
# (min, max) number of bonded neighbours per element, irrespective of bond
# order (the ARD-GSM graph model). ``CONNECTION_LIMITS_CLOSED_SHELL`` is taken
# verbatim from the group's Dandelion prior work
# (``dandelion/segsm/ard_gsm/limits.py`` ``connection_limits``). The enforced
# MINIMUM makes these neutral closed-shell bounds: a step that would leave an
# atom under-coordinated (e.g. homolysing the only bond to H/F/Cl/Br without a
# compensating new bond) is rejected because it would create a radical.
CONNECTION_LIMITS_CLOSED_SHELL: dict[str, tuple[int, int]] = {
    "H": (1, 1), "C": (2, 4), "N": (1, 3), "O": (1, 2),
    "F": (1, 1), "S": (1, 4), "CL": (1, 1), "BR": (1, 1),
    "LI": (0, 1),
}

# Deliberately relaxed table for open-shell / radical chemistry (combustion,
# autoxidation, homolysis). This is NOT from Dandelion: it drops the minimum to
# 0 so bond homolysis (R-H -> R. + H., R-X -> R. + X.) and O-centred radical
# steps are permitted, while keeping the same maximum valences. Pair it with a
# spin-aware calculator; the shipped MLIP is neutral-singlet only (see
# reactip_lot.DEFAULT_STATE_KEY), so radical energetics are extrapolative.
CONNECTION_LIMITS_RADICAL: dict[str, tuple[int, int]] = {
    "H": (0, 1), "C": (1, 4), "N": (0, 3), "O": (0, 2),
    "F": (0, 1), "S": (0, 4), "CL": (0, 1), "BR": (0, 1),
    "LI": (0, 1),
}

# Fallback (min, max) when an element is absent from the table above.
_DEFAULT_CONNECTION_LIMIT = (0, 8)

# Single source of truth for covalent radii (Angstrom, Cordero et al. 2008) and
# the distance-cutoff scale used to perceive a covalent bond. utils.py and
# visualize.py import these so the sampler and every figure perceive IDENTICAL
# connectivity (previously the radii table was duplicated three times and the
# scale disagreed: 1.20 here vs 1.25 in visualize.py).
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
}
DEFAULT_COVALENT_RADIUS = 0.75
BOND_SCALE = 1.20

# Backwards-compatible private alias (older code referenced ``_COVALENT_RADII``).
_COVALENT_RADII = COVALENT_RADII


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
    scale: float = BOND_SCALE,
) -> set[tuple[int, int]]:
    """Infer zero-based covalent bond pairs from distances."""
    bonds: set[tuple[int, int]] = set()
    for i, symbol_i in enumerate(symbols):
        radius_i = COVALENT_RADII.get(symbol_i, DEFAULT_COVALENT_RADIUS)
        for j in range(i + 1, len(symbols)):
            radius_j = COVALENT_RADII.get(symbols[j], DEFAULT_COVALENT_RADIUS)
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


def canonical_bond_signature(
    xyz_path: str | Path, *, scale: float = BOND_SCALE
) -> tuple:
    """Return a symmetry-INVARIANT bond signature for one structure.

    Unlike :func:`bond_signature` (which embeds absolute atom indices and so
    treats two mirror-image relabelings of the same molecule as different
    products), this keys each bond on the *topological-symmetry classes* of its
    endpoints (Weisfeiler-Lehman colour refinement, element-aware) and returns
    the sorted bond multiset. Two structures with this signature equal are the
    same species up to graph automorphism, so they are correctly merged for
    Boltzmann de-duplication.

    Requires only that atom ORDERING is consistent within a structure (it is:
    products are copied forward as the next reactant), not across structures.
    """
    from collections import Counter

    symbols, coordinates = read_xyz_frame(xyz_path)
    bonds = infer_bond_pairs(symbols, coordinates, scale=scale)
    classes = _symmetry_classes(len(symbols), symbols, bonds)

    def endpoint(i: int) -> tuple[str, int]:
        return (symbols[i], classes[i])

    bond_keys = sorted(
        tuple(sorted((endpoint(i), endpoint(j)))) for i, j in bonds
    )
    return tuple(sorted(Counter(bond_keys).items()))


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


# --------------------------------------------------------------------------- #
# Rule-based driving-coordinate generation (ARD-GSM / Dandelion lineage)
# --------------------------------------------------------------------------- #
def _symmetry_classes(
    n_atoms: int,
    symbols: Sequence[str],
    bonds: set[tuple[int, int]],
    *,
    rounds: int = 4,
) -> list[int]:
    """Topological-symmetry classes via Weisfeiler-Lehman colour refinement.

    Atoms in the same class are related by a graph automorphism (the three H of
    a methyl, the two ortho C of a phenyl), so driving coordinates that differ
    only by swapping such atoms are the same reaction. Pure-Python and
    connectivity-only, so it is robust for radical / O2 systems where RDKit
    valence perception fails. Returns a dense integer label per atom.
    """
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    def _densify(labels: list[str]) -> list[int]:
        ordering = {label: rank for rank, label in enumerate(sorted(set(labels)))}
        return [ordering[label] for label in labels]

    # Initial colour: element + degree.
    colours = _densify([f"{symbols[i]}:{len(adjacency[i])}" for i in range(n_atoms)])
    for _ in range(rounds):
        refined = _densify(
            [
                f"{colours[i]}|" + ",".join(sorted(str(colours[j]) for j in adjacency[i]))
                for i in range(n_atoms)
            ]
        )
        if refined == colours:  # colour partition is stable -> done
            break
        colours = refined
    return colours


def _connection_limit(symbol: str, limits: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return limits.get(symbol.upper(), _DEFAULT_CONNECTION_LIMIT)


def _valence_ok_after(
    degrees: Sequence[int],
    symbols: Sequence[str],
    breaks: Sequence[tuple[int, int]],
    forms: Sequence[tuple[int, int]],
    limits: dict[str, tuple[int, int]],
) -> bool:
    """Would every atom whose connectivity changes stay within ``limits``?

    Only atoms whose connection count actually changes are tested: an atom left
    untouched keeps whatever the input perceived, so a perception artefact on an
    untouched atom must not veto every candidate.
    """
    delta: dict[int, int] = {}
    for i, j in breaks:
        delta[i] = delta.get(i, 0) - 1
        delta[j] = delta.get(j, 0) - 1
    for i, j in forms:
        delta[i] = delta.get(i, 0) + 1
        delta[j] = delta.get(j, 0) + 1
    for atom, change in delta.items():
        if change == 0:
            continue
        new_degree = degrees[atom] + change
        lo, hi = _connection_limit(symbols[atom], limits)
        if new_degree < lo or new_degree > hi:
            return False
    return True


def _symmetry_key(
    breaks: Sequence[tuple[int, int]],
    forms: Sequence[tuple[int, int]],
    classes: Sequence[int],
) -> tuple:
    """Canonical, symmetry-invariant key for a driving-coordinate set."""
    bk = tuple(sorted(tuple(sorted((classes[i], classes[j]))) for i, j in breaks))
    fk = tuple(sorted(tuple(sorted((classes[i], classes[j]))) for i, j in forms))
    return ("B", bk, "F", fk)


def generate_rule_based_pool(
    xyz_path: str | Path,
    *,
    include_hydrogen: bool = True,
    maxbreak: int = 1,
    maxform: int = 1,
    maxchange: int = 2,
    minchange: int = 1,
    single_change: bool = True,
    max_form_distance: float | None = 6.0,
    bond_scale: float = BOND_SCALE,
    symmetry_reduce: bool = True,
    open_shell: bool = False,
    connection_limits: dict[str, tuple[int, int]] | None = None,
) -> list[DrivingCoordSet]:
    """Valence-pruned, symmetry-reduced driving-coordinate sets from a geometry.

    A drop-in alternative to :func:`generate_driving_coordinate_pool` following
    the group's own ARD-GSM / Dandelion rules
    (``dandelion/segsm/ard_gsm/driving_coords.py`` + ``limits.py``):

    1. **Valence-bounded validity** — a bond change is proposed only if the
       atoms it touches stay within ``connection_limits`` (default
       :data:`CONNECTION_LIMITS_CLOSED_SHELL`; pass ``open_shell=True`` for the
       radical table). Forming a 5th bond to carbon or a 2nd bond to fluorine is
       rejected before it costs an SE-GSM run.
    2. **Bounded reaction complexity** — break ``<= maxbreak``, form
       ``<= maxform`` and change ``<= maxchange`` connections per elementary
       step (defaults 1/1/2). This removes the O(N**4) ``two_add`` pool by
       construction.
    3. **Topological-symmetry reduction** — driving-coordinate sets related by a
       graph automorphism collapse to a single representative.

    Unlike the geometric enumerator, hydrogen is INCLUDED by default: with
    valence + symmetry pruning the H-inclusive pool is small and chemically
    meaningful (H-abstraction / H-shift), whereas the geometric enumerator must
    exclude H to stay tractable.

    Output is byte-compatible with :func:`generate_driving_coordinate_pool`: a
    sorted list of :data:`DrivingCoordSet` with 1-based indices, BREAK before
    ADD.
    """
    if connection_limits is None:
        connection_limits = (
            CONNECTION_LIMITS_RADICAL if open_shell else CONNECTION_LIMITS_CLOSED_SHELL
        )
    symbols, coordinates = read_xyz_frame(xyz_path)
    n_atoms = len(symbols)
    bonds = infer_bond_pairs(symbols, coordinates, scale=bond_scale)
    degrees = [0] * n_atoms
    for i, j in bonds:
        degrees[i] += 1
        degrees[j] += 1

    if symmetry_reduce:
        classes = _symmetry_classes(n_atoms, symbols, bonds)
    else:
        classes = list(range(n_atoms))

    if include_hydrogen:
        allowed = set(range(n_atoms))
    else:
        heavy = {i for i, s in enumerate(symbols) if s != "H"}
        allowed = heavy if len(heavy) >= 2 else set(range(n_atoms))

    break_candidates = [
        (i, j) for i, j in sorted(bonds) if i in allowed and j in allowed
    ]
    form_candidates: list[tuple[int, int]] = []
    for i, j in combinations(sorted(allowed), 2):
        if (i, j) in bonds:
            continue
        if max_form_distance is not None:
            if _pair_distance(coordinates, i, j) > max_form_distance:
                continue
        form_candidates.append((i, j))

    pool: dict[tuple, DrivingCoordSet] = {}
    lower = max(1, minchange)
    for n_break in range(0, maxbreak + 1):
        for n_form in range(0, maxform + 1):
            n_change = n_break + n_form
            if n_change < lower or n_change > maxchange:
                continue
            if not single_change and n_change == 1:
                continue
            for breaks in combinations(break_candidates, n_break):
                for forms in combinations(form_candidates, n_form):
                    if set(breaks) & set(forms):
                        continue
                    if not _valence_ok_after(degrees, symbols, breaks, forms, connection_limits):
                        continue
                    normalized = normalize_driving_coord_set(
                        [("BREAK", i + 1, j + 1) for i, j in breaks]
                        + [("ADD", i + 1, j + 1) for i, j in forms]
                    )
                    key = (
                        _symmetry_key(breaks, forms, classes)
                        if symmetry_reduce
                        else normalized
                    )
                    if key not in pool:
                        pool[key] = normalized
    return sorted(pool.values())


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
