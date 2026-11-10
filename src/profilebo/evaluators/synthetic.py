"""A cheap analytic stand-in for the Serpent example, for trying profilebo
without a transport code (and for the tests).

The model follows the physics of black absorber particles. Two batches
(radius R1, R2 in cm; particle counts N1, N2) each hold down reactivity in
proportion to their projected area, W_i = K N_i R_i^2, and burn as a
shrinking core: the absorbing core recedes at a steady rate, so a batch's
worth falls as W_i (1 - t/tau_i)^2 and is gone at tau_i = C R_i. Small
particles burn out early, large ones late; together they can compensate a
declining fuel reactivity, which is the point of a two-batch design.

The fuel's excess reactivity is constructed so that the REFERENCE design
holds the core exactly at 1000 pcm: the campaign has a known answer, and
the optimum lies inside the default bounds. Other designs in the box come
close (the hold-down/burn-out trade-off is not unique), as in the real
problem.

Config section:

    synthetic:
      noise: 25.0                  # pcm, 1-sigma of the Monte-Carlo-like noise
      bounds:                      # all four required, [lo, hi]
        R1: [0.005, 0.035]
        R2: [0.040, 0.090]
        N1: [100, 500]
        N2: [100, 400]

The noise is a deterministic function of x, so evaluating the same design
twice returns the same trajectory, like a seeded simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from profilebo.evaluators.base import Evaluation
from profilebo.parameters import Parameter

# Burnup grid, days.
T = np.array([0.025, 1, 5, 15, 30, 50, 90, 130, 180, 270, 365, 548, 730,
              913, 1095, 1278, 1460, 1643, 1825, 2008, 2190, 2373.0])

NAMES = ("R1", "R2", "N1", "N2")
K = 12000.0                 # pcm per particle per cm^2 of radius squared
C = 30000.0                 # burn-out time per cm of radius, days
REFERENCE = {"R1": 0.020, "R2": 0.065, "N1": 300.0, "N2": 250.0}
REFERENCE_LEVEL = 1000.0    # pcm, the flat reactivity REFERENCE achieves
_KEYS = {"noise", "bounds"}


def absorber_worth(r1: float, r2: float, n1: float, n2: float) -> np.ndarray:
    """Reactivity held down by both batches on the grid T, pcm."""
    return sum(K * n * r**2 * np.clip(1.0 - T / (C * r), 0.0, None) ** 2
               for r, n in ((r1, n1), (r2, n2)))


# Excess reactivity of the fuel alone: what REFERENCE holds at REFERENCE_LEVEL.
FUEL = REFERENCE_LEVEL + absorber_worth(*(REFERENCE[n] for n in NAMES))


def parse_parameters(section: dict) -> list[Parameter]:
    """The synthetic config section -> its four parameters, in NAMES order."""
    unknown = section.keys() - _KEYS
    if unknown:
        raise ValueError(f"synthetic: unknown keys {sorted(unknown)}; allowed: {sorted(_KEYS)}")
    bounds = section.get("bounds") or {}
    if set(bounds) != set(NAMES):
        raise ValueError(f"synthetic.bounds must give exactly {list(NAMES)}, got {sorted(bounds)}")
    return [Parameter(n, float(bounds[n][0]), float(bounds[n][1]), integer=n.startswith("N"))
            for n in NAMES]


class SyntheticEvaluator:
    """Analytic core reactivity trajectory (pcm) on the grid T (days)."""

    def __init__(self, section: dict | None = None):
        self.noise = float((section or {}).get("noise", 25.0))

    def evaluate(self, x: dict[str, float], case_dir: Path) -> Evaluation:
        mu = np.array([x[n] for n in NAMES], dtype=float)
        y = FUEL - absorber_worth(*mu)

        # Hashing the rounded design (floats hash deterministically in Python)
        # ties the noise to x and nothing else.
        rng = np.random.default_rng(abs(hash(tuple(np.round(mu, 6)))) % 2**32)
        y = y + rng.normal(0.0, self.noise, len(T))
        return Evaluation(t=T.copy(), y=y, std=np.full(len(T), self.noise))
