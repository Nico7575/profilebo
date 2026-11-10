"""The evaluator interface: one design point in, one trajectory out.

An evaluator runs the expensive simulation for a design point x (a dict
{parameter name: value}) and returns the trajectory y on its grid t, with the
statistical 1-sigma of every point. The BO loop knows nothing else about it.

Contract:

  * Blocking: evaluate() returns when the trajectory is available.
  * Idempotent per case_dir: called again on the same case_dir (after a
    crash or a restart of the loop), it must reuse what is already there
    (re-attach to a running job, or read a finished result) rather than
    silently simulate again. This is what makes a campaign resumable.
  * Thread-safe: up to bo.doe_parallel calls run concurrently during the
    initial design.
  * Every evaluation of a campaign returns the same grid t.
  * If it uses randomness, it seeds it from the campaign seed and
    case_dir.name, so a reproducible campaign stays reproducible.

If the evaluator ends up reporting a different design from the one it was
asked for (for instance it adopted a finished case rendered for an earlier
proposal), it says so in Evaluation.x, and the loop records that design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Evaluation:
    t: np.ndarray                       # grid of the trajectory
    y: np.ndarray                       # trajectory on t
    std: np.ndarray                     # statistical 1-sigma of y
    x: dict[str, float] | None = None   # design actually simulated, if not the one asked for


class Evaluator(Protocol):
    def evaluate(self, x: dict[str, float], case_dir: Path) -> Evaluation:
        ...
