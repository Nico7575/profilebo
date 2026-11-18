"""The Serpent evaluator: a design point in, a reactivity trajectory out.

The trajectory is the reactivity rho = (k_eff - 1)/k_eff in pcm, on the
burnup grid of serpent.burnup_days in days (plus the initial point), with
its statistical 1-sigma.
"""

from __future__ import annotations

from pathlib import Path

from profilebo.evaluators.base import Evaluation
from profilebo.evaluators.serpent.batches import batch_parameters, parse_batches, resolve
from profilebo.evaluators.serpent.inputs import burnup_days_of, design_from_case, render_input
from profilebo.evaluators.serpent.runner import run_case
from profilebo.evaluators.serpent.schedulers import BACKENDS, backend_name, local, slurm

SECTION_KEYS = {"executable", "template", "ba_particles", "burnup_days", "n_channels", "fuel",
                "cylinder", "disperser", "mpi_repro", "rng_seed", "scheduler", *BACKENDS}
_REQUIRED = {
    "fuel": {"universe", "radius", "count"},
    "cylinder": {"radius", "z_min", "z_max"},
}
_OPTIONAL = {"disperser": {"shake_factor", "growth_rate"}}
_BACKEND_SETTINGS = {"slurm": slurm.SETTINGS, "local": local.SETTINGS}


def check_section(serpent_cfg: dict) -> None:
    """Validate the serpent section of a config; raises ValueError.

    Only the structure is checked here: the template and the executable may
    live on another machine than the one reading the config.
    """
    def unknown(keys, allowed, where):
        extra = set(keys) - set(allowed)
        if extra:
            raise ValueError(f"{where}: unknown keys {sorted(extra)}; allowed: {sorted(allowed)}")

    unknown(serpent_cfg, SECTION_KEYS, "serpent")
    for key in ("executable", "template"):
        if not serpent_cfg.get(key):
            raise ValueError(f"serpent.{key} is required")
    for name, keys in _REQUIRED.items():
        sub = serpent_cfg.get(name)
        if not isinstance(sub, dict) or keys - sub.keys():
            raise ValueError(f"serpent.{name} needs the keys {sorted(keys)}")
        unknown(sub, keys, f"serpent.{name}")
    for name, keys in _OPTIONAL.items():
        unknown(serpent_cfg.get(name) or {}, keys, f"serpent.{name}")
    burnup_days_of(serpent_cfg)
    backend = backend_name(serpent_cfg)
    unknown(serpent_cfg.get(backend) or {}, _BACKEND_SETTINGS[backend], f"serpent.{backend}")


class SerpentEvaluator:
    def __init__(self, serpent_cfg: dict):
        check_section(serpent_cfg)
        self.cfg = serpent_cfg
        self.specs = parse_batches(serpent_cfg)
        template = Path(serpent_cfg["template"])
        if not template.is_file():
            raise FileNotFoundError(f"serpent.template not found: {template}")
        # Fill the template once now, so that a broken template stops the
        # campaign before any job is submitted rather than in every case.
        midpoint = {p.name: (p.lower + p.upper) / 2 for p in batch_parameters(self.specs)}
        try:
            render_input(resolve(self.specs, midpoint), "particles.pbed", template, serpent_cfg)
        except (KeyError, IndexError) as exc:
            raise ValueError(f"{template}: unknown placeholder {exc}; the placeholders are "
                             "ba_particles, ba_materials, ba_divs, pbed_filename and "
                             "burnup_days, and other curly braces must be doubled") from exc

    def evaluate(self, x: dict[str, float], case_dir: Path) -> Evaluation:
        traj = run_case(resolve(self.specs, x), self.specs, case_dir, self.cfg)
        # Always the design read back from the case: equal to x up to the
        # last digit when the case was run for x, and the design it was run
        # for when an earlier case was adopted.
        return Evaluation(t=traj.days, y=traj.rho, std=traj.rho_std,
                          x=design_from_case(case_dir, self.specs))
