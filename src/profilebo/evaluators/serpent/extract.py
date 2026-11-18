"""Parse Serpent 2.2 _res.m output into burnup trajectories.

Reactivities are in pcm, rho = (k - 1)/k, and the statistical uncertainty is
propagated with the delta method: sigma_rho = sigma_k / k^2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PCM = 1.0e5

# e.g. "ABS_KEFF   (idx, [1:   2]) = [  1.18867E+00 0.00035 ] ;"
# (note: Serpent writes "] ;" with a space before the semicolon)
_ENTRY = re.compile(r"^(\w+)\s*\(idx[^)]*\)\s*=\s*\[([^\]]+)\]\s*;", re.MULTILINE)


@dataclass
class BurnupTrajectory:
    """k_eff and its reactivity over the burnup grid of one Serpent run."""

    days: np.ndarray            # burnup time (EFPD)
    burnup: np.ndarray          # MWd/kgU
    k_eff: np.ndarray
    k_eff_std: np.ndarray       # absolute 1-sigma

    @property
    def rho(self) -> np.ndarray:
        """Reactivity, pcm."""
        return (self.k_eff - 1.0) / self.k_eff * PCM

    @property
    def rho_std(self) -> np.ndarray:
        """1-sigma of the reactivity, pcm (delta method)."""
        return self.k_eff_std / self.k_eff**2 * PCM


def _collect(text: str) -> dict[str, list[list[float]]]:
    """All entries of the res.m file, in file (= burnup step) order."""
    out: dict[str, list[list[float]]] = {}
    for name, values in _ENTRY.findall(text):
        out.setdefault(name, []).append([float(v) for v in values.split()])
    return out


def parse_res_m(path: str | Path) -> BurnupTrajectory:
    """Extract k_eff (ABS_KEFF) vs burnup.

    Serpent reports [value, relative_std] pairs, one entry per burnup step.
    BURN_DAYS/BURNUP carry the step coordinates.
    """
    entries = _collect(Path(path).read_text())

    def series(name: str, col: int = 0) -> np.ndarray | None:
        if name not in entries:
            return None
        return np.array([row[col] for row in entries[name]])

    k_eff = series("ABS_KEFF")
    if k_eff is None:
        raise ValueError(f"No ABS_KEFF entries found in {path}")
    k_eff_rel = series("ABS_KEFF", 1)

    days = series("BURN_DAYS")
    burnup = series("BURNUP")
    if days is None:
        raise ValueError(f"No BURN_DAYS entries found in {path} (not a burnup run?)")

    n = len(k_eff)
    if not (len(days) == len(burnup) == n):
        raise ValueError(
            f"Inconsistent step counts in {path}: "
            f"ABS_KEFF={n}, BURN_DAYS={len(days)}, BURNUP={len(burnup)}"
        )

    return BurnupTrajectory(days=days, burnup=burnup, k_eff=k_eff,
                            k_eff_std=k_eff * k_eff_rel)
