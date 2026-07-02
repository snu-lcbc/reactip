"""SMILES <-> XYZ conversion for ReactIP.

This module bridges the GUI (which speaks SMILES) and the SE-GSM engine (which
speaks XYZ). It provides four public helpers:

* :func:`smiles_to_xyz`           - one SMILES -> a single 3D geometry (XYZ text).
* :func:`smiles_to_reactant_xyz`  - one or more SMILES -> a single combined
  reactant frame with each molecule placed a few Angstrom apart. Crucially, the
  atoms of each fragment are written as a **contiguous block** (all of molecule
  A, then all of molecule B, ...). Interleaved multi-fragment numbering is what
  triggers the pyGSM ``get_hybrid_indices`` crash, so contiguous ordering here
  keeps GUI-generated bimolecular inputs (e.g. Diels-Alder) out of that failure
  mode.
* :func:`xyz_to_smiles`           - a 3D geometry -> SMILES via RDKit bond
  perception, for labelling ReactIP's XYZ products in the GUI reaction map.
* :func:`validate_domain`         - reject inputs outside the ReactIP MLIP's
  training domain (neutral, closed-shell, elements CHNOFSSClBr) with a clear
  message, instead of letting the run fail deep inside SE-GSM.

All geometry generation is deterministic for a fixed ``seed`` (ETKDG + MMFF/UFF).

RDKit is required. It is a dependency of the ReactIP conda environment; if it is
missing, :func:`_require_rdkit` raises a clear ImportError.

Command line
------------
    python -m reactip.smiles "C=CC=C.C=C" -o reactant.xyz          # bimolecular
    python -m reactip.smiles "C1=CC=C(C=C1)C=O" -o benzaldehyde.xyz
    python -m reactip.smiles --to-smiles product.xyz              # XYZ -> SMILES
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# MLIP training domain
# ---------------------------------------------------------------------------
# The ReactIP potential is trained on closed-shell, neutral molecules built from
# these elements only (matches reactip.utils._COVALENT_RADII).
ALLOWED_ELEMENTS: frozenset[str] = frozenset({"H", "C", "N", "O", "F", "S", "Cl", "Br"})
# NOTE: no molecule-size cap. NequIP is size-extensive (no architectural atom
# limit), and out-of-domain *geometry* is caught at runtime by the calculator's
# max-force guard. n_heavy_atoms is still reported for information.

# Default pre-reaction separation for placing multiple fragments (Angstrom,
# closest-approach gap between fragment surfaces along the placement axis).
DEFAULT_FRAGMENT_GAP: float = 4.0


class DomainError(ValueError):
    """Raised when a SMILES/molecule falls outside the ReactIP MLIP domain."""


class SmilesConversionError(ValueError):
    """Raised when RDKit cannot parse a SMILES or embed a 3D geometry."""


def _require_rdkit():
    """Import RDKit lazily with a clear error if it is missing."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        # Availability check: rdDetermineBonds is used in xyz_to_smiles and must be
        # present in this RDKit build. Probing it here gives one clear error site.
        from rdkit.Chem import rdDetermineBonds
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "reactip.smiles requires RDKit. Install it into the ReactIP "
            "environment (conda install -c conda-forge rdkit)."
        ) from exc
    assert rdDetermineBonds is not None
    return Chem, AllChem


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------
@dataclass
class DomainReport:
    """Structured result of a domain check (useful for GUI-side validation)."""

    ok: bool
    smiles: str
    n_heavy_atoms: int
    elements: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> "DomainReport":
        if not self.ok:
            raise DomainError(
                f"SMILES {self.smiles!r} is outside the ReactIP MLIP domain: "
                + "; ".join(self.reasons)
            )
        return self


def validate_domain(smiles: str, *, raise_on_error: bool = False) -> DomainReport:
    """Check a single SMILES against the ReactIP MLIP training domain.

    Domain: neutral, closed-shell (no unpaired electrons), only CHNOFSSClBr.
    Molecule size is NOT restricted (NequIP is size-extensive; distorted
    geometry is caught at runtime by the max-force guard). Returns a
    :class:`DomainReport`; pass ``raise_on_error=True`` to raise
    :class:`DomainError` instead of returning a failed report.
    """
    Chem, _ = _require_rdkit()
    reasons: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        report = DomainReport(False, smiles, 0, [], ["RDKit could not parse the SMILES"])
        return report.raise_if_bad() if raise_on_error else report

    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    n_heavy = mol.GetNumHeavyAtoms()

    bad_elements = sorted(set(elements) - ALLOWED_ELEMENTS)
    if bad_elements:
        reasons.append(
            f"unsupported element(s) {bad_elements} (allowed: "
            f"{sorted(ALLOWED_ELEMENTS)})"
        )
    total_charge = Chem.GetFormalCharge(mol)
    if total_charge != 0:
        reasons.append(f"net charge {total_charge:+d} (model is trained on neutrals)")
    n_radical = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if n_radical != 0:
        reasons.append(
            f"{n_radical} unpaired electron(s) (model is trained on closed-shell "
            "species)"
        )

    report = DomainReport(not reasons, smiles, n_heavy, elements, reasons)
    return report.raise_if_bad() if raise_on_error else report


# ---------------------------------------------------------------------------
# SMILES -> 3D geometry
# ---------------------------------------------------------------------------
def _embed_3d(smiles: str, *, seed: int, max_attempts: int):
    """Parse a SMILES, add explicit H, embed with ETKDGv3, optimize with MMFF
    (falling back to UFF). Returns an RDKit Mol with a single 3D conformer."""
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SmilesConversionError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Retry with random coordinates for stubborn small/strained systems.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise SmilesConversionError(
                f"RDKit failed to embed a 3D conformer for {smiles!r}"
            )

    # Geometry refinement: MMFF where parameterized, else UFF.
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_attempts)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_attempts)
    except Exception:  # pragma: no cover - optimizer numerical edge cases
        # A failed FF refinement is non-fatal; the embedded geometry is still a
        # reasonable starting point for SE-GSM pre-optimization.
        pass
    return mol


def _mol_to_symbols_coords(mol) -> tuple[list[str], np.ndarray]:
    """Extract (symbols, Nx3 coords) from an RDKit Mol's first conformer."""
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=float,
    )
    return symbols, coords


def _format_xyz(symbols: Sequence[str], coords: np.ndarray, comment: str = "") -> str:
    """Render (symbols, coords) as a standard XYZ text block."""
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:2s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(lines) + "\n"


def smiles_to_xyz(
    smiles: str,
    *,
    seed: int = 42,
    validate: bool = True,
    max_attempts: int = 200,
    comment: str | None = None,
) -> str:
    """Convert a single SMILES to an XYZ text block (deterministic for ``seed``).

    Parameters
    ----------
    smiles : str
        A single-molecule SMILES (no ``.`` fragment separators; for
        multi-molecule input use :func:`smiles_to_reactant_xyz`).
    validate : bool
        If True (default), enforce the MLIP domain and raise :class:`DomainError`
        on violations.
    """
    if "." in smiles:
        raise SmilesConversionError(
            f"smiles_to_xyz expects a single molecule but got {smiles!r} "
            "(contains '.'). Use smiles_to_reactant_xyz for multi-molecule input."
        )
    if validate:
        validate_domain(smiles, raise_on_error=True)
    mol = _embed_3d(smiles, seed=seed, max_attempts=max_attempts)
    symbols, coords = _mol_to_symbols_coords(mol)
    return _format_xyz(symbols, coords, comment if comment is not None else smiles)


def _fragment_radius(coords: np.ndarray) -> float:
    """Approximate radius of a fragment about its centroid (max atom distance)."""
    if len(coords) == 0:
        return 0.0
    centroid = coords.mean(axis=0)
    return float(np.max(np.linalg.norm(coords - centroid, axis=1)))


def smiles_to_reactant_xyz(
    smiles: str | Sequence[str],
    *,
    seed: int = 42,
    validate: bool = True,
    gap: float = DEFAULT_FRAGMENT_GAP,
    max_attempts: int = 200,
    comment: str | None = None,
) -> str:
    """Convert one or more SMILES into a single combined reactant XYZ frame.

    Accepts either a list of SMILES or a single dot-separated string
    (``"C=CC=C.C=C"``). Each molecule is embedded independently, then placed
    along the +x axis with its centroid separated from the previous fragment by
    ``gap`` Angstrom of clear space (sum of fragment radii + gap). Atoms are
    written as **contiguous per-fragment blocks** so the reactant never has the
    interleaved multi-fragment numbering that trips pyGSM's ``get_hybrid_indices``.

    For a single molecule this is equivalent to :func:`smiles_to_xyz`.
    """
    if isinstance(smiles, str):
        fragments = [s.strip() for s in smiles.split(".") if s.strip()]
    else:
        fragments = [s.strip() for part in smiles for s in str(part).split(".") if s.strip()]
    if not fragments:
        raise SmilesConversionError("No SMILES fragments provided.")

    if validate:
        for frag in fragments:
            validate_domain(frag, raise_on_error=True)

    if len(fragments) == 1:
        return smiles_to_xyz(
            fragments[0], seed=seed, validate=False, max_attempts=max_attempts,
            comment=comment if comment is not None else fragments[0],
        )

    all_symbols: list[str] = []
    all_coords: list[np.ndarray] = []
    cursor_x = 0.0  # x-position for the centroid of the next fragment
    prev_radius = 0.0
    for i, frag in enumerate(fragments):
        # Deterministic but distinct seed per fragment.
        mol = _embed_3d(frag, seed=seed + i, max_attempts=max_attempts)
        symbols, coords = _mol_to_symbols_coords(mol)
        coords = coords - coords.mean(axis=0)  # center at origin
        radius = _fragment_radius(coords)
        if i > 0:
            cursor_x += prev_radius + gap + radius
        coords = coords + np.array([cursor_x, 0.0, 0.0])
        all_symbols.extend(symbols)           # contiguous block for this fragment
        all_coords.append(coords)
        prev_radius = radius

    coords = np.vstack(all_coords)
    default_comment = " + ".join(fragments) + " (ReactIP reactant, contiguous fragments)"
    return _format_xyz(all_symbols, coords, comment if comment is not None else default_comment)


# ---------------------------------------------------------------------------
# 3D geometry -> SMILES  (for GUI product labels; see Q5)
# ---------------------------------------------------------------------------
def xyz_to_smiles(
    xyz: str | Path,
    *,
    charge: int = 0,
    canonical: bool = True,
    with_hs: bool = False,
) -> str:
    """Perceive bonds from a 3D geometry and return a SMILES label.

    Uses RDKit ``DetermineBonds`` (connectivity + bond orders from coordinates).
    Accepts an XYZ file path or an XYZ text block.

    NOTE: bond perception is a heuristic and can be ambiguous for strained or
    partial-bond geometries (e.g. a transition state). Treat the returned SMILES
    as a *display label*, not ground truth. Raises :class:`SmilesConversionError`
    if perception fails (e.g. on a TS-like geometry).
    """
    Chem, _ = _require_rdkit()
    from rdkit.Chem import rdDetermineBonds

    text = Path(xyz).read_text() if _looks_like_path(xyz) else str(xyz)
    mol = Chem.MolFromXYZBlock(text)
    if mol is None:
        raise SmilesConversionError("RDKit could not read the XYZ block.")
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
    except Exception as exc:
        raise SmilesConversionError(
            f"RDKit bond perception failed (geometry may be a TS / partial-bond "
            f"structure): {exc}"
        ) from exc
    if not with_hs:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=canonical)


def _looks_like_path(value) -> bool:
    if isinstance(value, Path):
        return True
    if isinstance(value, str):
        # An XYZ block starts with an integer atom count; a path does not span lines.
        return "\n" not in value.strip() and Path(value).exists()
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reactip.smiles",
        description="Convert SMILES <-> XYZ for ReactIP (with MLIP-domain checks).",
    )
    p.add_argument(
        "input",
        help="SMILES (single, or dot-separated for multiple molecules), or an XYZ "
        "file path when --to-smiles is set.",
    )
    p.add_argument("-o", "--output", default=None, help="Output XYZ path (default: stdout).")
    p.add_argument("--to-smiles", action="store_true", help="Reverse mode: XYZ -> SMILES.")
    p.add_argument("--seed", type=int, default=42, help="ETKDG random seed (deterministic).")
    p.add_argument("--gap", type=float, default=DEFAULT_FRAGMENT_GAP,
                   help="Clear separation (A) between fragments for multi-molecule input.")
    p.add_argument("--charge", type=int, default=0, help="Total charge for --to-smiles bond perception.")
    p.add_argument("--no-validate", action="store_true", help="Skip the MLIP-domain check.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.to_smiles:
        smi = xyz_to_smiles(args.input, charge=args.charge)
        print(smi)
        return 0

    xyz_text = smiles_to_reactant_xyz(
        args.input, seed=args.seed, validate=not args.no_validate, gap=args.gap,
    )
    if args.output:
        Path(args.output).write_text(xyz_text)
        n_atoms = xyz_text.splitlines()[0]
        print(f"Wrote {args.output} ({n_atoms} atoms)")
    else:
        sys.stdout.write(xyz_text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
