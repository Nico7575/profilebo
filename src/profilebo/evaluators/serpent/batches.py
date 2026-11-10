"""Burnable-absorber particle batches: config -> parameters -> Serpent cards.

A batch is one population of identical BA particles (radius R, N particles
per fuel channel), each particle divided into n_zones equal-thickness
concentric shells for self-shielding. Every shell is its own burnable
material whose declared volume is the total over the model:
shell volume x N x n_channels.

Each batch's radius and count is either a [lo, hi] range, which makes it a
design parameter (Ri, Ni for batch i, counting from 1), or a single
number, which fixes it. A single-batch study is a one-entry list; fixing a
radius turns a batch into a count-only study, without touching the code.

The template carries three placeholders that serpent_cards() fills:

    {ba_particles}   one 'particle <u>' definition per batch
    {ba_materials}   one 'mat burn_abs_<u>_<k>' per shell
    {ba_divs}        one 'div burn_abs_<u>_<k> sep 0' per shell

Batch i gets universe first_universe + i - 1, which must not collide with
any universe the template defines itself (the fuel particle in particular).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from profilebo.parameters import Parameter

_GROUP_KEYS = {"first_universe", "matrix", "n_zones", "material", "batches"}
_BATCH_KEYS = {"radius", "count", "n_zones", "material"}
_MATERIAL_KEYS = {"density", "temperature", "composition"}


@dataclass(frozen=True)
class Material:
    density: float                  # Serpent convention: negative = g/cm3
    temperature: float              # K
    composition: dict[str, float]   # nuclide (e.g. "64155.81c") -> fraction


@dataclass(frozen=True)
class BatchSpec:
    """One batch as configured: radius and count are fixed or a range."""

    index: int                                  # 1-based
    universe: int
    radius: float | tuple[float, float]         # cm
    count: int | tuple[int, int]                # particles per fuel channel
    n_zones: int
    material: Material


@dataclass(frozen=True)
class Batch:
    """One batch at a concrete design point: what Serpent needs."""

    index: int
    universe: int
    radius: float
    count: int
    n_zones: int
    material: Material


def _material(raw: dict, where: str) -> Material:
    _check_keys(raw, _MATERIAL_KEYS, where)
    missing = _MATERIAL_KEYS - raw.keys()
    if missing:
        raise ValueError(f"{where}: missing {sorted(missing)}")
    composition = {str(k): float(v) for k, v in raw["composition"].items()}
    if not composition:
        raise ValueError(f"{where}.composition is empty")
    return Material(float(raw["density"]), float(raw["temperature"]), composition)


def _range_or_fixed(value, where: str, integer: bool):
    """[lo, hi] -> (lo, hi) tuple; a single number -> that number."""
    cast = int if integer else float
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{where}: a range needs exactly [lo, hi], got {value}")
        lo, hi = cast(value[0]), cast(value[1])
        if not lo < hi:
            raise ValueError(f"{where}: lower bound {lo} must be below upper bound {hi}")
        return lo, hi
    return cast(value)


def _check_keys(raw: dict, allowed: set[str], where: str) -> None:
    # A typo would otherwise be silently ignored and fall back to a default.
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def parse_batches(serpent_cfg: dict) -> list[BatchSpec]:
    """The 'serpent.ba_particles' section -> one BatchSpec per batch.

    n_zones and material set at group level are defaults; a batch can
    override either with its own key.
    """
    group = serpent_cfg.get("ba_particles")
    if not group:
        raise ValueError("serpent.ba_particles is missing")
    _check_keys(group, _GROUP_KEYS, "serpent.ba_particles")
    raw_batches = group.get("batches") or []
    if not raw_batches:
        raise ValueError("serpent.ba_particles.batches needs at least one batch")

    first_universe = int(group.get("first_universe", 30))
    default_zones = group.get("n_zones")
    default_material = group.get("material")
    reserved = {int(serpent_cfg["fuel"]["universe"])} if "fuel" in serpent_cfg else set()

    specs = []
    for i, raw in enumerate(raw_batches, start=1):
        where = f"serpent.ba_particles.batches[{i}]"
        _check_keys(raw, _BATCH_KEYS, where)
        for key in ("radius", "count"):
            if key not in raw:
                raise ValueError(f"{where}: missing '{key}'")

        radius = _range_or_fixed(raw["radius"], f"{where}.radius", integer=False)
        count = _range_or_fixed(raw["count"], f"{where}.count", integer=True)
        if min(_as_tuple(radius)) <= 0:
            raise ValueError(f"{where}.radius must be positive")
        # The disperser reads a number below 1 as a packing fraction, and
        # rejects exactly 1, so a count must be at least 2.
        if min(_as_tuple(count)) < 2:
            raise ValueError(f"{where}.count must be at least 2")

        n_zones = raw.get("n_zones", default_zones)
        if n_zones is None or int(n_zones) < 1:
            raise ValueError(f"{where}: n_zones must be a positive integer "
                             "(set it per batch or for the whole group)")

        material = raw.get("material", default_material)
        if material is None:
            raise ValueError(f"{where}: no material (set it per batch or for the whole group)")

        universe = first_universe + i - 1
        if universe in reserved:
            raise ValueError(f"{where}: universe {universe} collides with the fuel "
                             "particle universe; change ba_particles.first_universe")

        specs.append(BatchSpec(i, universe, radius, count, int(n_zones),
                               _material(material, f"{where}.material")))
    return specs


def _as_tuple(value) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def batch_parameters(specs: list[BatchSpec]) -> list[Parameter]:
    """The design parameters the batches expose, in a fixed order:
    all radii first, then all counts (R1, R2, ..., N1, N2, ...)."""
    params = []
    for s in specs:
        if isinstance(s.radius, tuple):
            params.append(Parameter(f"R{s.index}", *s.radius))
    for s in specs:
        if isinstance(s.count, tuple):
            params.append(Parameter(f"N{s.index}", *s.count, integer=True))
    if not params:
        raise ValueError("every batch radius and count is fixed: nothing to optimize")
    return params


def resolve(specs: list[BatchSpec], x: dict[str, float]) -> list[Batch]:
    """Concrete batches at design point x (parameter name -> value).

    Fixed values come from the spec; free ones from x. Counts are rounded
    here, so whatever reaches Serpent is an integer even if x is not.
    """
    expected = {p.name for p in batch_parameters(specs)}
    if set(x) != expected:
        raise ValueError(f"design point has parameters {sorted(x)}, expected {sorted(expected)}")
    batches = []
    for s in specs:
        radius = float(x[f"R{s.index}"]) if isinstance(s.radius, tuple) else s.radius
        count = int(round(x[f"N{s.index}"])) if isinstance(s.count, tuple) else s.count
        batches.append(Batch(s.index, s.universe, radius, count, s.n_zones, s.material))
    return batches


def shell_radii(outer_radius: float, n_zones: int) -> list[float]:
    """Outer radii of n_zones equal-thickness concentric shells."""
    thickness = outer_radius / n_zones
    return [thickness * (k + 1) for k in range(n_zones)]


def shell_volumes(radii: list[float]) -> list[float]:
    """Volumes of the spherical shells bounded by consecutive radii."""
    volumes, prev = [], 0.0
    for r in radii:
        volumes.append(4.0 / 3.0 * math.pi * (r**3 - prev**3))
        prev = r
    return volumes


def serpent_cards(batches: list[Batch], n_channels: int, matrix: str) -> dict[str, str]:
    """Template text for {ba_particles}, {ba_materials} and {ba_divs}.

    matrix: the template material that fills the particle outside its
    outermost shell (in practice the compact's graphite).
    """
    particles, materials, divs = [], [], []
    for b in batches:
        radii = shell_radii(b.radius, b.n_zones)
        volumes = shell_volumes(radii)
        names = [f"burn_abs_{b.universe}_{k}" for k in range(1, b.n_zones + 1)]

        particles.append("\n".join(
            [f"particle {b.universe}"]
            + [f"{name} {r}" for name, r in zip(names, radii)]
            + [matrix]
        ))
        nuclides = "\n".join(f"{nuc} {frac}" for nuc, frac in b.material.composition.items())
        for name, v in zip(names, volumes):
            materials.append(
                f"mat {name} {b.material.density} vol {v * b.count * n_channels} "
                f"tmp {b.material.temperature:g} burn 1\n{nuclides}"
            )
        divs.append("\n".join(f"div {name} sep 0" for name in names))

    return {
        "ba_particles": "\n\n".join(particles),
        "ba_materials": "\n\n".join(materials),
        "ba_divs": "\n\n".join(divs),
    }
