"""ReactIP — Reactive Interatomic Potential inference library for NequIP/Allegro models."""

__all__ = ["ReactIPCalculator"]


def __getattr__(name: str):
    if name == "ReactIPCalculator":
        from .mlip_calculator import ReactIPCalculator

        return ReactIPCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
