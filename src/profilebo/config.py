"""Typed configuration loaded from config.yaml.

Numerical layers (surrogate, optimization) get typed dataclasses; the
evaluator receives its own section as a plain dict (the Serpent one merely
forwards values into templates and shell commands).

A config has exactly one evaluator section: `serpent`, or `synthetic` for
the analytic toy model. The design parameters are not listed separately:
they are derived from that section (the Serpent evaluator's from its
particle batches, see evaluators/serpent/batches.py), so they cannot drift
out of sync with what is actually simulated.

Relative paths in the file (serpent.template, objective.target when it is
a file, paths.run_dir) are resolved against the config file's directory,
so a campaign behaves the same whichever directory it is launched from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from profilebo.parameters import Parameter

_EVALUATORS = ("serpent", "synthetic")
_SECTIONS = {*_EVALUATORS, "basis", "objective", "bo", "paths"}


@dataclass
class BasisConfig:
    # The POD rank is chosen after the DoE: the fewest modes whose truncation
    # error on the DoE trajectories (in the objective window) is at most
    # target_ratio times their statistical sigma, in RMS. See
    # surrogate.basis.select_pod_basis.
    max_modes: int = 10                 # upper limit; a warning is logged if hit
    target_ratio: float = 1.0
    n_coeff: int | None = None          # fixes the rank instead of choosing it


@dataclass
class ObjectiveConfig:
    alpha: float = 100.0
    target: float | Path = 0.0          # scalar, or path to a tabulated profile
    window: tuple[float, float] = (0.0, 2000.0)   # J evaluated for t in [start, end]
    # False: J is a plain mean over in-window grid POINTS (original).
    # True:  each point is weighted by the share of the window it represents,
    #        making J an average over t. On a front-loaded grid this shifts
    #        weight markedly toward late t; see objective.trapezoid_weights.
    time_weighted: bool = False
    # None: J is the distance-to-target form (original).
    # float: J = Var_w + beta*(mean_w - target)^2 + alpha*undershoot, which
    #        separates FLATNESS from OFFSET. beta=1 (alpha=0) is the symmetric
    #        original exactly; beta=0 optimises flatness alone. Composes with
    #        time_weighted, which is what stops the front-loaded grid from
    #        letting the early drop dominate the variance.
    flatness_beta: float | None = None


@dataclass
class BOConfig:
    n_init: int = 16
    budget: int = 40
    seed: int = 42
    num_restarts: int = 10
    raw_samples: int = 512
    mc_samples: int = 256
    acq_threshold: float | None = None
    doe_parallel: int = 1        # concurrent Serpent jobs during the DoE phase
    reproducible: bool = False

    # --- design-resolution floor ---
    # A proposal that differs from an evaluated design by less than the
    # resolution in EVERY parameter cannot be built or measured apart, so
    # it is redirected to the best distinguishable candidate instead.
    # Resolution = max(manufacturing tolerance, statistical resolution).
    # Leave design_tolerance empty and tolerance_streak None to disable.
    resolution_floor: bool = False   # master switch for the whole mechanism
    design_tolerance: dict = field(default_factory=dict)   # absolute units per param
    statistical_floor: bool = True      # add sigma_J / |dJ/dmu| per parameter
    tolerance_streak: int | None = None # stop after this many sub-resolution steps
    noise_stop_factor: float | None = None  # stop when EI_out < factor * sigma_J


@dataclass
class Config:
    parameters: list[Parameter]
    run_dir: Path
    evaluator: str                      # which evaluator section the config has
    serpent: dict = field(default_factory=dict)
    synthetic: dict = field(default_factory=dict)
    basis: BasisConfig = field(default_factory=BasisConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    bo: BOConfig = field(default_factory=BOConfig)

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def bounds_array(self) -> np.ndarray:
        """(2, d) array of [lower; upper] in parameter order."""
        return np.array([[p.lower for p in self.parameters],
                         [p.upper for p in self.parameters]], dtype=float)

    def round_integers(self, x: np.ndarray) -> np.ndarray:
        """Round the integer parameters (round at evaluation time, record
        the rounded x as the training input)."""
        x = np.asarray(x, dtype=float).copy()
        for i, p in enumerate(self.parameters):
            if p.integer:
                x[..., i] = np.round(x[..., i])
        return x

    def as_point(self, x: np.ndarray) -> dict[str, float]:
        """Array in parameter order -> {name: value}, as evaluators take it."""
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.parameters),):
            raise ValueError(f"expected {len(self.parameters)} values, got shape {x.shape}")
        return dict(zip(self.parameter_names, x.tolist()))


def _section(cls, raw: dict, name: str):
    """Build a section dataclass, naming the section if a key is unknown
    (a typo would otherwise surface as a bare TypeError from __init__)."""
    raw = raw.get(name) or {}
    allowed = {f.name for f in fields(cls)}
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    return cls(**raw)


def _evaluator(raw: dict) -> str:
    present = [name for name in _EVALUATORS if name in raw]
    if len(present) != 1:
        raise ValueError(f"the config needs exactly one evaluator section out of "
                         f"{list(_EVALUATORS)}, found {present}")
    return present[0]


def _parameters(evaluator: str, section: dict) -> list[Parameter]:
    """The design parameters, derived from the evaluator's section.
    Further evaluators plug in here (and in evaluators.make_evaluator)."""
    if evaluator == "serpent":
        from profilebo.evaluators.serpent.batches import batch_parameters, parse_batches
        return batch_parameters(parse_batches(section))
    from profilebo.evaluators.synthetic import parse_parameters
    return parse_parameters(section)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base = path.resolve().parent

    def resolve(p) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else base / p

    unknown = raw.keys() - _SECTIONS
    if unknown:
        raise ValueError(f"unknown sections {sorted(unknown)}; allowed: {sorted(_SECTIONS)}")

    evaluator = _evaluator(raw)
    section = raw[evaluator] or {}
    parameters = _parameters(evaluator, section)
    if evaluator == "serpent" and "template" in section:
        section["template"] = resolve(section["template"])

    objective = _section(ObjectiveConfig, raw, "objective")
    window = tuple(float(v) for v in objective.window)
    if len(window) != 2 or not window[0] < window[1]:
        raise ValueError(f"objective.window must be [start, end], got {window}")
    objective.window = window
    if isinstance(objective.target, str):
        objective.target = resolve(objective.target)

    bo = _section(BOConfig, raw, "bo")
    if bo.n_init > bo.budget:
        raise ValueError(f"bo.n_init ({bo.n_init}) exceeds bo.budget ({bo.budget})")

    basis = _section(BasisConfig, raw, "basis")
    if basis.max_modes < 1 or basis.target_ratio <= 0:
        raise ValueError("basis.max_modes must be >= 1 and basis.target_ratio > 0")
    # The basis is built from the n_init DoE trajectories; with their mean
    # removed they span at most n_init - 1 directions.
    if basis.n_coeff is not None and not 1 <= basis.n_coeff <= bo.n_init - 1:
        raise ValueError(f"basis.n_coeff must be between 1 and bo.n_init - 1 "
                         f"({bo.n_init - 1}), got {basis.n_coeff}")
    # Keys must name real parameters: with names derived from the batches,
    # a stale entry (say R2 after switching to one batch) would otherwise be
    # silently ignored.
    names = {p.name for p in parameters}
    stale = bo.design_tolerance.keys() - names
    if stale:
        raise ValueError(f"bo.design_tolerance: {sorted(stale)} are not design "
                         f"parameters; the parameters are {sorted(names)}")
    if bo.reproducible and evaluator == "serpent":
        section.setdefault("rng_seed", int(bo.seed))

    run_dir = (raw.get("paths") or {}).get("run_dir")
    if run_dir is None:
        # No default: resuming a campaign reuses whatever is in run_dir, so
        # two configs must never share one by accident.
        raise ValueError("paths.run_dir is required")

    return Config(
        parameters=parameters,
        run_dir=resolve(run_dir),
        evaluator=evaluator,
        **{evaluator: section},
        basis=basis,
        objective=objective,
        bo=bo,
    )
