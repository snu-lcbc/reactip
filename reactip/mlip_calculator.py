"""Standalone inference calculator for ReactIP (Reactive Interatomic Potential).

This module provides :class:`ReactIPCalculator`, a thin convenience wrapper
over the official :class:`~nequip.ase.nequip_calculator.NequIPCalculator`
that adds:

- A path-based constructor that auto-detects compiled vs. checkpoint format
- ``calculate()`` that *returns* a result dict (as well as storing in ``self.results``)
- Convenience wrappers: ``calculate_xyz()``, ``calculate_file()``
- A public ``from_checkpoint()`` classmethod
- Output validation for out-of-distribution geometries
- A pure helper to convert standalone result dicts to other units

Runtime dependencies: ``torch``, ``ase``, ``nequip`` (inference-only).
No Lightning, Hydra, or wandb are imported here.
"""

from __future__ import annotations

import io
import pathlib
from typing import Dict, List, Optional, Union

import numpy as np

import ase
import ase.io
from ase.calculators.calculator import CalculationFailed, all_changes

from nequip.ase.nequip_calculator import NequIPCalculator


class ReactIPCalculator(NequIPCalculator):
    """Inference calculator for ReactIP (Reactive Interatomic Potential).

    Inherits the full ASE Calculator interface from :class:`NequIPCalculator`
    (so it works transparently in ASE MD, BFGS, etc.) and adds convenience
    methods suited for SE-GSM and similar external callers.

    Quick start::

        # From a compiled model (preferred — no Lightning/Hydra needed)
        calc = ReactIPCalculator("model.nequip.pt2", device="cuda")
        result = calc.calculate_xyz(xyz_string)
        energy  = result["energy"]   # float, eV
        forces  = result["forces"]   # np.ndarray (n_atoms, 3), eV/Å

        # From a checkpoint (requires full nequip install)
        calc = ReactIPCalculator.from_checkpoint("best.ckpt", device="cuda")

        # As an ASE calculator
        atoms.calc = calc
        e = atoms.get_potential_energy()  # eV
        f = atoms.get_forces()            # (N, 3) eV/Å

    Args:
        model_path: Path to a compiled ``.nequip.pt2`` (AOTInductor) or
            ``.nequip.pth`` (TorchScript) file, **or** a raw ``.ckpt``
            checkpoint. Compiled models are preferred for deployment because
            they bundle architecture + weights with no training dependencies.
        device: PyTorch device string, e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``.
        chemical_symbols: Explicit list of chemical symbols (or symbol→type-name
            dict) for the model. Usually not needed — extracted from model
            metadata automatically.
        energy_units: Must be ``"eV"``. The calculator now stays strictly
            ASE-native; use :meth:`convert_results_to_units` for standalone
            result conversion.
        max_force: Safety threshold for force magnitude per component (eV/Å).
            If any force component exceeds this after inference, a
            :class:`CalculationFailed` error is raised.  Set to ``None`` to
            disable.  Default ``100.0``.
        max_abs_energy: Safety threshold for absolute energy (eV).  If
            ``|energy|`` exceeds this, a :class:`CalculationFailed` error is
            raised.  Set to ``None`` to disable.  Default ``10000.0``.
    """

    #: 1 eV = 23.0609 kcal/mol  (NIST 2018 CODATA)
    EV_TO_KCAL_MOL: float = 23.0609

    def __init__(
        self,
        model_path: Union[str, pathlib.Path],
        device: str = "cuda",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        energy_units: str = "eV",
        max_force: Optional[float] = 100.0,
        max_abs_energy: Optional[float] = 10000.0,
        **kwargs,
    ) -> None:
        self._max_force = max_force
        self._max_abs_energy = max_abs_energy
        model_path = str(model_path)

        if energy_units != "eV":
            raise ValueError(
                "ReactIPCalculator must remain ASE-native in eV/eV/Angstrom. "
                "Use ReactIPCalculator.convert_results_to_units(...) for standalone "
                "result conversion."
            )

        if model_path.endswith(".nequip.pth") or model_path.endswith(".nequip.pt2"):
            _base = NequIPCalculator.from_compiled_model(
                model_path,
                device=device,
                chemical_symbols=chemical_symbols,
                **kwargs,
            )
        else:
            # .ckpt checkpoint or .nequip.zip package — requires full nequip install
            _base = NequIPCalculator._from_saved_model(
                model_path,
                device=device,
                chemical_symbols=chemical_symbols,
                **kwargs,
            )

        NequIPCalculator.__init__(
            self,
            model=_base.model,
            device=_base.device,
            transforms=_base.transforms,
            energy_units_to_eV=_base.energy_units_to_eV,
            length_units_to_A=_base.length_units_to_A,
        )
    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Union[str, pathlib.Path],
        config_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ) -> "ReactIPCalculator":
        """Load from a raw NequIP checkpoint file.

        NequIP checkpoints are self-contained (architecture + weights embedded).
        The ``config_path`` argument is accepted for API symmetry but is *not*
        used — the config is always read from inside the checkpoint.

        Args:
            ckpt_path: Path to ``best.ckpt``, ``last.ckpt``, etc.
            config_path: Ignored. Kept for forward-compatibility.
            device: PyTorch device string.
        """
        return cls(ckpt_path, device=device, **kwargs)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def calculate(
        self,
        atoms=None,
        properties: List[str] = ["energy"],
        system_changes=all_changes,
    ) -> dict:
        """Run inference on a single structure.

        Accepts :class:`ase.Atoms`, an XYZ string, or a file path.  The method
        also stores results in ``self.results`` (standard ASE behaviour) so it
        integrates transparently with ASE optimisers and MD engines.

        Args:
            atoms: One of:

                - :class:`ase.Atoms` object
                - ``str`` — raw XYZ / extXYZ string content
                - :class:`pathlib.Path` — path to an ``.xyz`` file
                - ``str`` path that exists on disk — also treated as file

            properties: ASE properties list (for ASE Calculator compatibility).
            system_changes: ASE system_changes (for ASE Calculator compat.).

        Returns:
            Dict with keys:

            - ``"energy"`` — float, total energy
            - ``"forces"`` — :class:`numpy.ndarray` shape ``(n_atoms, 3)``
            - ``"stress"`` — :class:`numpy.ndarray` shape ``(6,)`` Voigt,
              only present when the structure has a cell

            Units are always ASE-native: eV and eV/Å.
        """
        atoms = self._to_ase_atoms(atoms)
        super().calculate(atoms, properties=properties, system_changes=system_changes)
        result = dict(self.results)
        # Drop stress for non-periodic structures: the model always computes a
        # stress/virial tensor, but for molecules without a cell the result is
        # ±inf (virial divided by zero volume) and should not be reported.
        if "stress" in result and not np.all(np.isfinite(result["stress"])):
            del result["stress"]
            self.results.pop("stress", None)
        # Validate outputs — ML potentials can produce garbage for
        # out-of-distribution geometries (NaN, huge forces, etc.).
        self._validate(result)

        # Keep self.results in ASE units; callers that want different units
        # should convert the returned copy explicitly.
        self.results = dict(result)
        return result

    def calculate_xyz(self, xyz: str) -> dict:
        """Run inference on a raw XYZ/extXYZ string.

        This is the primary entry-point for SE-GSM, which constructs geometry
        strings on the fly::

            result   = calc.calculate_xyz(xyz_string)
            energy   = result["energy"]    # eV
            gradient = -result["forces"]   # eV/Å (gradient = −forces)

        Args:
            xyz: Extended XYZ format string (ASE ``extxyz`` dialect).

        Returns:
            Same as :meth:`calculate`.
        """
        atoms = ase.io.read(io.StringIO(xyz), format="extxyz")
        return self.calculate(atoms)

    def calculate_file(self, path: Union[str, pathlib.Path], index: int = -1) -> dict:
        """Run inference on a frame from an XYZ file.

        Args:
            path: Path to a ``.xyz`` file (single- or multi-frame).
            index: Frame index (default ``-1`` = last frame).

        Returns:
            Same as :meth:`calculate`.
        """
        atoms = ase.io.read(str(path), index=index)
        return self.calculate(atoms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ase_atoms(atoms) -> ase.Atoms:
        """Normalise input to :class:`ase.Atoms`."""
        if isinstance(atoms, ase.Atoms):
            return atoms
        if isinstance(atoms, pathlib.Path):
            return ase.io.read(str(atoms))
        if isinstance(atoms, str):
            p = pathlib.Path(atoms)
            if p.exists():
                return ase.io.read(str(p))
            # treat as inline XYZ string
            return ase.io.read(io.StringIO(atoms), format="extxyz")
        raise TypeError(
            f"Unsupported atoms type: {type(atoms).__name__}. "
            "Expected ase.Atoms, str (XYZ content or file path), or pathlib.Path."
        )

    @classmethod
    def convert_results_to_units(
        cls,
        results: dict,
        energy_units: str = "eV",
    ) -> dict:
        """Return a converted copy of a standalone result dict.

        This helper is for reporting and post-processing only. It must not be
        used to mutate an attached ASE calculator's ``self.results`` because
        ASE assumes eV and eV/Angstrom throughout.
        """
        converted = dict(results)
        if energy_units == "eV":
            return converted
        if energy_units != "kcal/mol":
            raise ValueError(
                f"Unsupported energy_units={energy_units!r}. "
                "Expected 'eV' or 'kcal/mol'."
            )

        factor = cls.EV_TO_KCAL_MOL
        for key in ("energy", "forces", "energies", "free_energy", "stress"):
            if key in converted:
                converted[key] = converted[key] * factor
        return converted

    def _validate(self, results: dict) -> None:
        """Check for unphysical outputs and raise on failure.

        ML potentials can return NaN, Inf, or absurdly large values when
        evaluated on geometries far from the training distribution (e.g.
        atoms too close together, or beyond the radial cutoff).  This
        catches those cases early so callers like pyGSM can abort the
        current step rather than silently propagate garbage.

        Raises:
            CalculationFailed: If energy or forces are non-finite or exceed
                the configured thresholds.
        """
        energy = results.get("energy")
        forces = results.get("forces")

        if energy is not None:
            if not np.isfinite(energy):
                raise CalculationFailed(
                    f"ReactIPCalculator: energy is not finite ({energy})"
                )
            if self._max_abs_energy is not None and abs(energy) > self._max_abs_energy:
                raise CalculationFailed(
                    f"ReactIPCalculator: |energy| = {abs(energy):.1f} "
                    f"exceeds threshold {self._max_abs_energy}"
                )

        if forces is not None:
            if not np.all(np.isfinite(forces)):
                raise CalculationFailed(
                    "ReactIPCalculator: forces contain NaN or Inf"
                )
            if self._max_force is not None:
                fmax = np.max(np.abs(forces))
                if fmax > self._max_force:
                    raise CalculationFailed(
                        f"ReactIPCalculator: max force component = {fmax:.1f} "
                        f"exceeds threshold {self._max_force}"
                    )
