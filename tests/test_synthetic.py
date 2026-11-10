from pathlib import Path

import numpy as np
import pytest

from profilebo.config import load_config
from profilebo.evaluators import make_evaluator
from profilebo.evaluators.synthetic import (
    REFERENCE,
    REFERENCE_LEVEL,
    SyntheticEvaluator,
    parse_parameters,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic.yaml"
X = {"R1": 0.02, "R2": 0.06, "N1": 300.0, "N2": 200.0}


def test_example_config_loads_and_builds_its_evaluator():
    cfg = load_config(EXAMPLE)
    assert cfg.evaluator == "synthetic"
    assert cfg.parameter_names == ["R1", "R2", "N1", "N2"]
    assert [p.integer for p in cfg.parameters] == [False, False, True, True]
    assert isinstance(make_evaluator(cfg), SyntheticEvaluator)
    assert cfg.serpent == {}


def test_same_design_same_trajectory(tmp_path):
    ev = SyntheticEvaluator({"noise": 25.0})
    a, b = ev.evaluate(X, tmp_path), ev.evaluate(dict(X), tmp_path)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.x is None and np.all(a.std == 25.0)


def test_the_reference_design_is_flat_at_its_level(tmp_path):
    y = SyntheticEvaluator({"noise": 0.0}).evaluate(REFERENCE, tmp_path).y
    np.testing.assert_allclose(y, REFERENCE_LEVEL, atol=1e-9)


def test_reference_lies_inside_the_example_bounds():
    cfg = load_config(EXAMPLE)
    for p in cfg.parameters:
        assert p.lower < REFERENCE[p.name] < p.upper


def test_more_absorber_suppresses_more_and_burns_out(tmp_path):
    ev = SyntheticEvaluator({"noise": 0.0})
    light = ev.evaluate(X, tmp_path).y
    heavy = ev.evaluate({**X, "N2": 290.0}, tmp_path).y
    extra = light - heavy                         # extra suppression, pcm
    assert extra[0] > 1000.0                      # strong at BOL
    assert np.all(np.diff(extra) <= 0)            # decaying with time
    assert extra[-1] == 0.0                       # gone once the batch is burnt out


@pytest.mark.parametrize("section, match", [
    ({"bounds": {"R1": [0, 1]}}, "exactly"),
    ({"bounds": {}, "nosie": 1.0}, "unknown keys"),
])
def test_invalid_sections_are_rejected(section, match):
    with pytest.raises(ValueError, match=match):
        parse_parameters(section)


def test_a_config_needs_exactly_one_evaluator(tmp_path):
    text = EXAMPLE.read_text()
    both = tmp_path / "both.yaml"
    both.write_text(text + "\nserpent: {}\n")
    with pytest.raises(ValueError, match="exactly one evaluator"):
        load_config(both)
    none = tmp_path / "none.yaml"
    none.write_text("basis: {}\npaths: {run_dir: r}\n")
    with pytest.raises(ValueError, match="exactly one evaluator"):
        load_config(none)
