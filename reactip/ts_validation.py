"""Transition-state validation for ReactIP / SE-GSM candidates.

SE-GSM reports a TS whenever the optimized string has a single energy peak
(``npeaks == 1``). That is a *geometric* criterion: it does not guarantee the
peak node is a genuine first-order saddle point of the potential energy
surface. The Dandelion / RPS pipeline (Lee et al., Adv. Sci. 2025; Halo8,
Sci. Data 2025) only accepted reaction pathways whose transition states showed
exactly one imaginary frequency.

This module reproduces that check at MLIP cost. Given the ReactIP calculator and
a candidate TS geometry it builds the Cartesian Hessian by central finite
differences of the analytic MLIP forces, then uses ASE's vibrational machinery
(mass-weighting + unit handling) to obtain harmonic frequencies and count
imaginary modes. A true first-order saddle has exactly one imaginary mode above
a small wavenumber threshold (small near-zero modes are translations/rotations
and finite-difference noise).

The linear-algebra path here is deliberately self-contained and unit-tested on
synthetic Hessians; the finite-difference part is validated on real TS nodes
during sampled-search runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Modes below this wavenumber (cm^-1) are treated as translations, rotations, or
# finite-difference noise rather than a genuine reaction-coordinate mode.
DEFAULT_IMAGINARY_THRESHOLD_CM = 50.0


@dataclass(frozen=True)
class SaddleAnalysis:
    """Result of a TS frequency analysis."""

    imaginary_mode_count: int
    is_first_order_saddle: bool
    imaginary_frequencies_cm: tuple[float, ...]
    lowest_real_frequency_cm: float | None
    threshold_cm: float

    def as_dict(self) -> dict:
        return {
            "ts_imaginary_mode_count": self.imaginary_mode_count,
            "ts_is_first_order_saddle": self.is_first_order_saddle,
            "ts_imaginary_frequencies_cm": list(self.imaginary_frequencies_cm),
            "ts_lowest_real_frequency_cm": self.lowest_real_frequency_cm,
            "ts_frequency_threshold_cm": self.threshold_cm,
        }


def compute_force_field_hessian(
    calculator,
    symbols: Sequence[str],
    coordinates: np.ndarray,
    *,
    displacement: float = 0.005,
) -> np.ndarray:
    """Central finite-difference Cartesian Hessian (eV/Angstrom^2).

    ``calculator`` must expose ``calculate(atoms)`` returning a dict with a
    ``"forces"`` array in eV/Angstrom (the ReactIP calculator contract). The
    Hessian is symmetrized before being returned.
    """
    import ase

    coordinates = np.asarray(coordinates, dtype=float)
    n_atoms = len(symbols)
    n_dof = 3 * n_atoms
    base = ase.Atoms(symbols=list(symbols), positions=coordinates)

    def forces_at(positions: np.ndarray) -> np.ndarray:
        atoms = base.copy()
        atoms.set_positions(positions)
        result = calculator.calculate(atoms)
        return np.asarray(result["forces"], dtype=float).reshape(-1)

    hessian = np.zeros((n_dof, n_dof), dtype=float)
    flat = coordinates.reshape(-1)
    for dof in range(n_dof):
        plus = flat.copy()
        minus = flat.copy()
        plus[dof] += displacement
        minus[dof] -= displacement
        f_plus = forces_at(plus.reshape(n_atoms, 3))
        f_minus = forces_at(minus.reshape(n_atoms, 3))
        # Hessian = d^2E/dx^2 = -dF/dx
        hessian[dof] = -(f_plus - f_minus) / (2.0 * displacement)

    return 0.5 * (hessian + hessian.T)


def frequencies_cm_from_hessian(
    symbols: Sequence[str],
    coordinates: np.ndarray,
    hessian: np.ndarray,
) -> np.ndarray:
    """Harmonic frequencies in cm^-1 (imaginary modes returned as negative).

    Uses ASE's :class:`~ase.vibrations.VibrationsData` for vetted mass-weighting
    and unit conversion. ASE returns imaginary frequencies as complex numbers;
    here they are flattened to signed reals (negative == imaginary) for easy
    counting.
    """
    import ase
    from ase.vibrations import VibrationsData

    atoms = ase.Atoms(symbols=list(symbols), positions=np.asarray(coordinates, dtype=float))
    n_atoms = len(symbols)
    vib = VibrationsData.from_2d(atoms, np.asarray(hessian, dtype=float).reshape(3 * n_atoms, 3 * n_atoms))
    raw = vib.get_frequencies()  # complex ndarray, cm^-1
    signed = np.where(np.abs(raw.imag) > np.abs(raw.real), -np.abs(raw.imag), raw.real)
    return np.asarray(signed, dtype=float)


def analyze_frequencies(
    frequencies_cm: np.ndarray,
    *,
    threshold_cm: float = DEFAULT_IMAGINARY_THRESHOLD_CM,
) -> SaddleAnalysis:
    """Count imaginary modes and decide whether the point is a first-order saddle."""
    frequencies_cm = np.asarray(frequencies_cm, dtype=float)
    imaginary = frequencies_cm[frequencies_cm < -abs(threshold_cm)]
    real_positive = frequencies_cm[frequencies_cm > abs(threshold_cm)]
    lowest_real = float(np.min(real_positive)) if real_positive.size else None
    return SaddleAnalysis(
        imaginary_mode_count=int(imaginary.size),
        is_first_order_saddle=bool(imaginary.size == 1),
        imaginary_frequencies_cm=tuple(float(value) for value in sorted(imaginary)),
        lowest_real_frequency_cm=lowest_real,
        threshold_cm=float(abs(threshold_cm)),
    )


def verify_transition_state(
    calculator,
    symbols: Sequence[str],
    coordinates: np.ndarray,
    *,
    displacement: float = 0.005,
    threshold_cm: float = DEFAULT_IMAGINARY_THRESHOLD_CM,
) -> SaddleAnalysis:
    """Full TS check: build the MLIP Hessian and analyze its frequencies."""
    hessian = compute_force_field_hessian(
        calculator, symbols, coordinates, displacement=displacement
    )
    frequencies = frequencies_cm_from_hessian(symbols, coordinates, hessian)
    return analyze_frequencies(frequencies, threshold_cm=threshold_cm)
