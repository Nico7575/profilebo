"""Design parameters: the coordinates the optimizer searches over.

An evaluator declares its own parameters (the Serpent evaluator derives
them from the particle batches in the config), so the BO layer never needs
to know what a parameter means physically — only its name, bounds and
whether it is integer-valued.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float
    # Integer parameters are optimized as continuous and rounded at
    # evaluation time; the rounded value is what gets recorded.
    integer: bool = False

    def __post_init__(self):
        if not self.lower < self.upper:
            raise ValueError(
                f"parameter {self.name}: lower bound {self.lower} must be "
                f"below upper bound {self.upper}"
            )
