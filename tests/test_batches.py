import math
from pathlib import Path

import pytest
import yaml

from profilebo.evaluators.serpent.batches import (
    batch_parameters,
    parse_batches,
    resolve,
    serpent_cards,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "config.yaml"


@pytest.fixture
def serpent_cfg():
    return yaml.safe_load(EXAMPLE.read_text())["serpent"]


def test_example_config_gives_dual_batch_parameters(serpent_cfg):
    params = batch_parameters(parse_batches(serpent_cfg))
    assert [p.name for p in params] == ["R1", "R2", "N1", "N2"]
    assert (params[1].lower, params[1].upper) == (0.041, 0.090)
    assert [p.integer for p in params] == [False, False, True, True]


def test_fixed_values_are_not_parameters(serpent_cfg):
    serpent_cfg["ba_particles"]["batches"][0]["radius"] = 0.03
    serpent_cfg["ba_particles"]["batches"][1]["count"] = 150
    params = batch_parameters(parse_batches(serpent_cfg))
    assert [p.name for p in params] == ["R2", "N1"]


def test_single_batch(serpent_cfg):
    del serpent_cfg["ba_particles"]["batches"][1]
    specs = parse_batches(serpent_cfg)
    assert [p.name for p in batch_parameters(specs)] == ["R1", "N1"]
    assert [s.universe for s in specs] == [30]


def test_resolve_rounds_counts_and_fills_fixed_values(serpent_cfg):
    serpent_cfg["ba_particles"]["batches"][1]["radius"] = 0.075
    specs = parse_batches(serpent_cfg)
    b1, b2 = resolve(specs, {"R1": 0.02, "N1": 783.4, "N2": 150.6})
    assert (b1.radius, b1.count, b1.universe) == (0.02, 783, 30)
    assert (b2.radius, b2.count, b2.universe) == (0.075, 151, 31)


def test_resolve_rejects_wrong_parameter_set(serpent_cfg):
    specs = parse_batches(serpent_cfg)
    with pytest.raises(ValueError, match="expected"):
        resolve(specs, {"R1": 0.02, "N1": 500})


def test_per_batch_overrides(serpent_cfg):
    b4c = {"density": -2.5, "temperature": 800, "composition": {"5010.81c": 1.2}}
    serpent_cfg["ba_particles"]["batches"][1].update(n_zones=40, material=b4c)
    s1, s2 = parse_batches(serpent_cfg)
    assert (s1.n_zones, s2.n_zones) == (20, 40)
    assert s1.material.density == -7.4 and s2.material.density == -2.5


@pytest.mark.parametrize("edit, match", [
    (lambda g: g["batches"][0].update(radius_typo=1), "unknown keys"),
    (lambda g: g["batches"][0].update(count=[1, 10]), "at least 2"),
    (lambda g: g["batches"][0].update(radius=[0.04, 0.01]), "below upper"),
    (lambda g: g.update(first_universe=29), "collides"),
    (lambda g: g.update(batches=[]), "at least one batch"),
    (lambda g: g.pop("n_zones"), "n_zones"),
])
def test_invalid_configs_are_rejected(serpent_cfg, edit, match):
    edit(serpent_cfg["ba_particles"])
    with pytest.raises(ValueError, match=match):
        parse_batches(serpent_cfg)


def test_cards_cover_every_shell_and_conserve_volume(serpent_cfg):
    serpent_cfg["ba_particles"]["batches"][1]["n_zones"] = 5
    specs = parse_batches(serpent_cfg)
    batches = resolve(specs, {"R1": 0.02, "R2": 0.07, "N1": 700, "N2": 150})
    cards = serpent_cards(batches, n_channels=14040, matrix="graphite")

    mats = cards["ba_materials"].split("\n\n")
    assert len(mats) == 20 + 5
    assert cards["ba_divs"].count("div burn_abs_") == 25
    assert cards["ba_particles"].count("particle ") == 2

    # The shells of each batch add up to whole particles x count x channels.
    for b in batches:
        total = sum(float(m.split()[4]) for m in mats
                    if m.startswith(f"mat burn_abs_{b.universe}_"))
        expected = 4 / 3 * math.pi * b.radius**3 * b.count * 14040
        assert total == pytest.approx(expected, rel=1e-12)
