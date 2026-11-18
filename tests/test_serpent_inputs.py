"""Serpent input rendering and reading back, without running Serpent."""

from pathlib import Path

import numpy as np
import pytest

from profilebo.evaluators.serpent.batches import parse_batches, resolve
from profilebo.evaluators.serpent.disperser import particle_specs
from profilebo.evaluators.serpent.inputs import (
    INPUT_NAME,
    burnup_days_of,
    case_matches,
    design_from_case,
    format_burnup_days,
    generate_input,
    render_input,
)

X = {"R1": 0.0205, "R2": 0.0749, "N1": 783.0, "N2": 151.0}


# --- depletion grid ------------------------------------------------------------

def test_integers_stay_integral_and_fractions_survive():
    assert format_burnup_days([1, 91.25, 2190]) == "1 91.25 2190"


def test_burnup_days_is_required():
    with pytest.raises(ValueError, match="burnup_days is required"):
        burnup_days_of({})


@pytest.mark.parametrize("bad", [[2190, 1825], [1, 1], [0, 5], [-1, 5]])
def test_non_monotone_or_nonpositive_grids_are_rejected(bad):
    with pytest.raises(ValueError):
        burnup_days_of({"burnup_days": bad})


# --- rendering ------------------------------------------------------------------

def test_every_placeholder_is_filled(serpent_cfg):
    batches = resolve(parse_batches(serpent_cfg), X)
    text = render_input(batches, "p.pbed", serpent_cfg["template"], serpent_cfg)
    assert "{" not in text and "}" not in text
    assert 'pbed 900 50 "p.pbed"' in text
    assert "dep daytot " + format_burnup_days(serpent_cfg["burnup_days"]) in text
    assert text.count("div burn_abs_") == 2 * serpent_cfg["ba_particles"]["n_zones"]


def test_seed_and_repro_cards(serpent_cfg):
    batches = resolve(parse_batches(serpent_cfg), X)
    render = lambda cfg, seed: render_input(batches, "p.pbed", cfg["template"], cfg, seed)
    text = render(serpent_cfg, 123456)
    assert "set seed 123456" in text and "set repro" not in text
    text = render({**serpent_cfg, "mpi_repro": True}, 123456)
    assert "set seed 123456" in text and "set repro 2" in text
    text = render(serpent_cfg, None)
    assert "set seed" not in text and "reproducibility" not in text


def test_a_hard_coded_depletion_card_is_rejected(serpent_cfg, tmp_path):
    """The grid must come from the config, or the restart check would see
    every case on disk as stale."""
    template = tmp_path / "t.inp"
    text = Path(serpent_cfg["template"]).read_text()
    template.write_text(text.replace("dep daytot {burnup_days}", "dep daytot 1 5 30"))
    batches = resolve(parse_batches(serpent_cfg), X)
    with pytest.raises(ValueError, match="dep daytot"):
        render_input(batches, "p.pbed", template, serpent_cfg)


def test_matrix_is_required(serpent_cfg):
    batches = resolve(parse_batches(serpent_cfg), X)
    del serpent_cfg["ba_particles"]["matrix"]
    with pytest.raises(ValueError, match="matrix is required"):
        render_input(batches, "p.pbed", serpent_cfg["template"], serpent_cfg)


# --- reading a case back ----------------------------------------------------------

def test_design_is_read_back_from_a_case(serpent_cfg, tmp_path, write_serpent_case):
    specs, _ = write_serpent_case(tmp_path / "c", serpent_cfg, X)
    x = design_from_case(tmp_path / "c", specs)
    assert x.keys() == X.keys()
    assert x["R1"] == pytest.approx(X["R1"], abs=1e-12)
    assert x["R2"] == pytest.approx(X["R2"], abs=1e-12)
    assert (x["N1"], x["N2"]) == (X["N1"], X["N2"])


def test_fixed_values_are_not_part_of_the_design(serpent_cfg, tmp_path, write_serpent_case):
    serpent_cfg["ba_particles"]["batches"][1]["radius"] = 0.0749
    x_free = {k: v for k, v in X.items() if k != "R2"}
    specs, _ = write_serpent_case(tmp_path / "c", serpent_cfg, x_free)
    assert design_from_case(tmp_path / "c", specs).keys() == x_free.keys()


def test_case_matches_its_own_design_only(serpent_cfg, tmp_path, write_serpent_case):
    specs, batches = write_serpent_case(tmp_path / "c", serpent_cfg, X)
    assert case_matches(tmp_path / "c", batches, specs)
    other = resolve(specs, {**X, "R1": X["R1"] + 1e-4})
    assert not case_matches(tmp_path / "c", other, specs)
    other = resolve(specs, {**X, "N2": X["N2"] + 1})
    assert not case_matches(tmp_path / "c", other, specs)


def test_a_case_from_another_config_does_not_match(serpent_cfg, tmp_path, write_serpent_case):
    """Same free parameters, but batch 2's radius fixed to another value."""
    specs, _ = write_serpent_case(tmp_path / "c", serpent_cfg, X)
    serpent_cfg["ba_particles"]["batches"][1]["radius"] = 0.05
    specs2 = parse_batches(serpent_cfg)
    x2 = {k: v for k, v in X.items() if k != "R2"}
    assert not case_matches(tmp_path / "c", resolve(specs2, x2), specs2)


def test_an_incomplete_case_does_not_match(serpent_cfg, tmp_path, write_serpent_case):
    specs, batches = write_serpent_case(tmp_path / "c", serpent_cfg, X)
    (tmp_path / "c" / "particles.pbed").unlink()
    assert not case_matches(tmp_path / "c", batches, specs)
    assert not case_matches(tmp_path / "missing", batches, specs)


def test_particle_specs_put_fuel_first(serpent_cfg):
    batches = resolve(parse_batches(serpent_cfg), X)
    specs = particle_specs(batches, serpent_cfg)
    assert [s.universe for s in specs] == [29, 30, 31]
    assert [s.count for s in specs] == [21000, 783, 151]
    assert specs[2].radius == pytest.approx(0.0749)
