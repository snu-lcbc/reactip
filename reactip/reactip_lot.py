"""pyGSM level-of-theory adapter for ReactIP."""

from __future__ import annotations

from typing import Mapping, Tuple

from ase import units as ase_units
from ase.calculators.calculator import CalculationFailed

from pyGSM.level_of_theories.ase import xyz_to_ase
from pyGSM.level_of_theories.base_lot import LoTError, Lot

from .mlip_calculator import ReactIPCalculator

StateKey = Tuple[int, int, int]
DEFAULT_STATE_KEY: StateKey = (0, 1, 0)


def normalize_calculator_registry(
    calculator: ReactIPCalculator | None = None,
    calculator_registry: Mapping[StateKey, ReactIPCalculator] | None = None,
    default_state_key: StateKey = DEFAULT_STATE_KEY,
) -> dict[StateKey, ReactIPCalculator]:
    """Normalise calculator inputs into a state-keyed registry."""
    if calculator is not None and calculator_registry is not None:
        raise ValueError("Provide either calculator or calculator_registry, not both.")
    if calculator_registry is not None:
        return dict(calculator_registry)
    if calculator is None:
        raise ValueError("A calculator or calculator_registry is required.")
    return {default_state_key: calculator}


class ReactIPLoT(Lot):
    """Direct pyGSM ``Lot`` adapter for a ReactIP-backed ASE calculator.

    The adapter keeps the ML calculator loaded for the lifetime of the GSM run
    and performs the single required conversion from ASE units (eV, eV/Angstrom)
    to pyGSM units (Hartree, Hartree/Bohr).
    """

    def __init__(
        self,
        calculator_registry: Mapping[StateKey, ReactIPCalculator],
        options,
    ) -> None:
        super().__init__(options)
        self.calculator_registry = dict(calculator_registry)

    @classmethod
    def from_options(
        cls,
        calculator: ReactIPCalculator | None = None,
        calculator_registry: Mapping[StateKey, ReactIPCalculator] | None = None,
        **kwargs,
    ) -> "ReactIPLoT":
        states = kwargs.get("states", [(1, 0)])
        mult, ad_idx = states[0]
        registry = normalize_calculator_registry(
            calculator=calculator,
            calculator_registry=calculator_registry,
            default_state_key=(kwargs.get("charge", 0), mult, ad_idx),
        )
        return cls(registry, cls.default_options().set_values(kwargs))

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):
        assert isinstance(lot, ReactIPLoT)
        return cls(lot.calculator_registry, lot.options.copy().set_values(options))

    def _resolve_calculator(self, mult: int, ad_idx: int) -> ReactIPCalculator:
        key = (self.charge, mult, ad_idx)
        if key not in self.calculator_registry:
            available = ", ".join(str(k) for k in sorted(self.calculator_registry))
            raise LoTError(
                f"Unsupported state/model request {key}. "
                f"Available calculator keys: {available or 'none'}."
            )
        return self.calculator_registry[key]

    def run(self, geom, mult, ad_idx, runtype="gradient"):
        if runtype not in {"gradient", "energy"}:
            raise NotImplementedError(
                f"Run type {runtype!r} is not implemented for ReactIPLoT."
            )

        atoms = xyz_to_ase(geom)
        calculator = self._resolve_calculator(mult, ad_idx)

        try:
            result = calculator.calculate(atoms)
        except CalculationFailed as exc:
            raise LoTError(
                f"ReactIPLoT calculation failed for state "
                f"(charge={self.charge}, multiplicity={mult}, adiabatic_state={ad_idx}): {exc}"
            ) from exc

        self._Energies[(mult, ad_idx)] = self.Energy(
            float(result["energy"] / ase_units.Ha),
            "Hartree",
        )

        if runtype == "gradient":
            if "forces" not in result:
                raise LoTError("ReactIPLoT expected forces but calculator returned none.")
            self._Gradients[(mult, ad_idx)] = self.Gradient(
                -result["forces"] / ase_units.Ha * ase_units.Bohr,
                "Hartree/Bohr",
            )

        self.write_E_to_file()
        self.hasRanForCurrentCoords = True
