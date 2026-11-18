"""Serpent input files: render one from the template, and read a rendered one
back.

The template is a plain Serpent input with Python str.format placeholders:
ba_particles, ba_materials and ba_divs (filled by batches.serpent_cards),
pbed_filename and burnup_days, each written between curly braces. The
depletion grid comes from serpent.burnup_days, so a campaign can refine it
without editing the template.

Reading back (design_from_case) recovers the absorber batches a case
directory was rendered for, from its particle cards and its pbed file. The
runner uses it to check that a case directory left over from an earlier run
really belongs to the design about to be evaluated.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from profilebo.evaluators.serpent.batches import (
    Batch,
    BatchSpec,
    batch_parameters,
    serpent_cards,
)

INPUT_NAME = "case.serp"     # the rendered input in every case directory


def format_burnup_days(days) -> str:
    """dep daytot argument list: space-separated, integers kept integral."""
    out = []
    for d in days:
        d = float(d)
        out.append(str(int(d)) if d.is_integer() else repr(d))
    if not out:
        raise ValueError("burnup_days is empty")
    return " ".join(out)


def burnup_days_of(serpent_cfg: dict) -> tuple[float, ...]:
    """The configured depletion grid, serpent.burnup_days.

    Strictly increasing and positive: Serpent's dep daytot is cumulative time,
    so a non-monotone list silently gives a wrong history.
    """
    days = serpent_cfg.get("burnup_days")
    if not days:
        raise ValueError("serpent.burnup_days is required: the depletion steps, in days")
    if isinstance(days, str):
        days = days.split()
    days = tuple(float(d) for d in days)
    if days[0] <= 0:
        raise ValueError(f"burnup_days must be positive, got {days[0]}")
    if any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError(f"burnup_days must be strictly increasing, got {days}")
    return days


def input_burnup_days(text: str) -> tuple[float, ...] | None:
    """The dep daytot grid of a rendered input (its text), or None."""
    for line in text.splitlines():
        if line.strip().startswith("dep daytot"):
            try:
                return tuple(float(v) for v in line.split()[2:])
            except ValueError:
                return None
    return None


def render_input(
    batches: list[Batch],
    pbed_filename: str,
    template_path: str | Path,
    serpent_cfg: dict,
    seed: int | None = None,
) -> str:
    """The Serpent input for these absorber batches, as text.

    seed: Serpent's random-number seed (set seed); None keeps Serpent's
    default. A fixed seed reproduces a run bit for bit on the SAME
    number of MPI tasks; a different task count is a different run (each
    history's random stream is offset by the task id). serpent.mpi_repro adds
    set repro 2, meant to remove that dependence, but on the model this was
    developed for it did NOT reproduce 2 vs 3 tasks, so keep a campaign on one
    fixed task layout instead of relying on it.
    """
    group = serpent_cfg["ba_particles"]
    if not group.get("matrix"):
        raise ValueError("serpent.ba_particles.matrix is required: the template material "
                         "around the outermost absorber shell (e.g. graphite)")
    params = serpent_cards(batches, int(serpent_cfg.get("n_channels", 1)), group["matrix"])
    params["pbed_filename"] = pbed_filename
    days = burnup_days_of(serpent_cfg)
    params["burnup_days"] = format_burnup_days(days)

    text = Path(template_path).read_text().format(**params)
    # The runner checks the grid of a case on disk against serpent.burnup_days
    # before reusing it, so the rendered card must be exactly that grid: a
    # template with its own hard-coded card would make every case look stale.
    if input_burnup_days(text) != days:
        raise ValueError(f"{template_path}: the template must contain the line "
                         "'dep daytot {burnup_days}' (the grid comes from "
                         "serpent.burnup_days)")
    extra = []
    if seed is not None:
        extra.append(f"set seed {int(seed)}")
    if serpent_cfg.get("mpi_repro", False):
        extra.append("set repro 2")
    if extra:
        text = text.rstrip("\n") + "\n\n% --- reproducibility (profilebo) ---\n" + "\n".join(extra) + "\n"
    return text


def generate_input(
    batches: list[Batch],
    pbed_filename: str,
    template_path: str | Path,
    out_path: str | Path,
    serpent_cfg: dict,
    seed: int | None = None,
) -> Path:
    """Render the template for these batches and write it to out_path."""
    out_path = Path(out_path)
    out_path.write_text(render_input(batches, pbed_filename, template_path, serpent_cfg, seed))
    return out_path


def batches_from_case(case_dir, specs: list[BatchSpec],
                      input_name: str = INPUT_NAME) -> list[tuple[float, int]]:
    """(radius, count) of every configured batch, as rendered in case_dir.

    Radii come from the innermost shell of each batch's particle card
    (R = r_1 x n_zones, the shells being of equal thickness); particle counts
    from the pbed file, which lists one particle per line with its universe in
    column 5. Raises ValueError if a configured batch is not in the case.
    """
    case_dir = Path(case_dir)
    text = (case_dir / input_name).read_text()
    univ = np.loadtxt(next(case_dir.glob("*.pbed")), usecols=4, dtype=int, ndmin=1)

    out = []
    for s in specs:
        # Particle card lines start with the shell name; mat and div cards
        # mention the same name after a keyword, so anchor at line start.
        shells = re.findall(rf"^burn_abs_{s.universe}_(\d+)\s+([-\d.eE+]+)", text, re.M)
        if not shells:
            raise ValueError(f"batch universe {s.universe} not found in {case_dir / input_name}")
        r_first = float(dict(shells)["1"])
        out.append((r_first * len(shells), int((univ == s.universe).sum())))
    return out


def design_from_case(case_dir, specs: list[BatchSpec],
                     input_name: str = INPUT_NAME) -> dict[str, float]:
    """The design point (free parameters only) a case directory was rendered for."""
    found = batches_from_case(case_dir, specs, input_name)
    names = {p.name for p in batch_parameters(specs)}
    x = {}
    for s, (radius, count) in zip(specs, found):
        if f"R{s.index}" in names:
            x[f"R{s.index}"] = radius
        if f"N{s.index}" in names:
            x[f"N{s.index}"] = float(count)
    return x


def case_matches(case_dir, batches: list[Batch], specs: list[BatchSpec],
                 input_name: str = INPUT_NAME) -> bool:
    """True if case_dir was rendered for (essentially) these batches, fixed
    values included: a case rendered under a different config does not match."""
    try:
        found = batches_from_case(case_dir, specs, input_name)
    except (OSError, ValueError, StopIteration):
        return False
    return all(abs(r - b.radius) <= 1e-6 and abs(n - b.count) < 0.5
               for (r, n), b in zip(found, batches))
