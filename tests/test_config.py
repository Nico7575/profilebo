from pathlib import Path

import numpy as np
import pytest
import yaml

from profilebo.config import load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "config.yaml"


@pytest.fixture
def write_config(tmp_path):
    """Write an edited copy of the example config into tmp_path."""
    def write(edit=lambda raw: None):
        raw = yaml.safe_load(EXAMPLE.read_text())
        edit(raw)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(raw))
        return path
    return write


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.parameter_names == ["R1", "R2", "N1", "N2"]
    assert cfg.bounds_array.shape == (2, 4)
    assert cfg.serpent["template"].is_file()
    assert cfg.objective.window == (5.0, 1825.0)
    assert cfg.serpent["rng_seed"] == cfg.bo.seed     # reproducible: true


def test_round_integers_and_as_point():
    cfg = load_config(EXAMPLE)
    x = cfg.round_integers(np.array([0.0205, 0.0749, 783.4, 150.6]))
    assert x.tolist() == [0.0205, 0.0749, 783.0, 151.0]
    assert cfg.as_point(x) == {"R1": 0.0205, "R2": 0.0749, "N1": 783.0, "N2": 151.0}


def test_relative_paths_resolve_against_config_dir(write_config, tmp_path):
    def edit(raw):
        raw["serpent"]["template"] = "templates/core.inp"
        raw["objective"]["target"] = "goal.txt"
        raw["paths"]["run_dir"] = "runs/a"
    cfg = load_config(write_config(edit))
    assert cfg.serpent["template"] == tmp_path / "templates/core.inp"
    assert cfg.objective.target == tmp_path / "goal.txt"
    assert cfg.run_dir == tmp_path / "runs/a"


def test_single_batch_config(write_config):
    def edit(raw):
        del raw["serpent"]["ba_particles"]["batches"][1]
        del raw["bo"]["design_tolerance"]["R2"]
    assert load_config(write_config(edit)).parameter_names == ["R1", "N1"]


@pytest.mark.parametrize("edit, match", [
    (lambda r: r.update(bound={}), "unknown sections"),
    (lambda r: r["bo"].update(n_inti=16), r"bo: unknown keys \['n_inti'\]"),
    (lambda r: r["bo"].update(n_init=50), "exceeds"),
    (lambda r: del_batch(r, 1), "design_tolerance"),     # R2 tolerance now stale
    (lambda r: r["objective"].update(window=[1825, 5]), "objective.window"),
    (lambda r: r.pop("paths"), "run_dir is required"),
    (lambda r: r.update(basis={"n_coeff": 16}), "n_init - 1"),        # n_init = 16
    (lambda r: r["basis"].update(max_recon_error=200.0), "basis: unknown keys"),
])
def test_invalid_configs_are_rejected(write_config, edit, match):
    with pytest.raises(ValueError, match=match):
        load_config(write_config(edit))


def del_batch(raw, i):
    del raw["serpent"]["ba_particles"]["batches"][i]
