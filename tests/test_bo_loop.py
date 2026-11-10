"""End-to-end BO loop on cheap evaluators: the synthetic model, and a
two-parameter toy that checks nothing assumes four parameters."""

import json
from pathlib import Path

import numpy as np
import pytest

from profilebo.config import BasisConfig
from profilebo.evaluators.base import Evaluation
from profilebo.evaluators.synthetic import SyntheticEvaluator
from profilebo.optimization.bo_loop import BOLoop
from profilebo.parameters import Parameter


def test_bo_loop_end_to_end(synthetic_config, tmp_path):
    cfg = synthetic_config(tmp_path / "run", doe_parallel=2)
    loop = BOLoop(cfg, SyntheticEvaluator())
    x_star, j_star = loop.run()

    assert loop.n_evals == cfg.bo.budget
    assert set(x_star) == {"R1", "R2", "N1", "N2"}
    assert np.isfinite(j_star) and j_star >= 0.0

    # Integers stayed integral, bounds respected.
    lb, ub = cfg.bounds_array
    for x in loop.X:
        assert x[2] == round(x[2]) and x[3] == round(x[3])
        assert np.all(x >= lb - 1e-9) and np.all(x <= ub + 1e-9)

    # BO incumbent is at least as good as the best DoE point.
    obj = loop.fit_surrogate()[1]
    j_doe = min(obj.j_observed(y) for y, t in zip(loop.y, loop.tags) if t.startswith("doe"))
    assert j_star <= j_doe + 1e-9

    # Checkpoint artifacts exist, and the log records x by name.
    for name in ("history.npz", "basis.npz", "iterations.jsonl"):
        assert (cfg.run_dir / name).exists()
    first = json.loads((cfg.run_dir / "iterations.jsonl").read_text().splitlines()[0])
    assert set(first["x"]) == {"R1", "R2", "N1", "N2"}


def test_bo_loop_resume(synthetic_config, tmp_path):
    cfg = synthetic_config(tmp_path / "run")
    loop1 = BOLoop(cfg, SyntheticEvaluator())
    loop1.ensure_initial()
    assert loop1.n_evals == cfg.bo.n_init

    loop2 = BOLoop(cfg, SyntheticEvaluator())
    assert loop2.n_evals == cfg.bo.n_init          # picked up the checkpoint
    loop2.run()
    assert loop2.n_evals == cfg.bo.budget
    assert len(set(loop2.tags)) == len(loop2.tags)  # DoE not re-evaluated


def test_resume_refuses_a_changed_parameter_set(synthetic_config, tmp_path):
    BOLoop(synthetic_config(tmp_path / "run"), SyntheticEvaluator()).ensure_initial()
    params = [Parameter("a", 0.0, 1.0), Parameter("b", 0.0, 1.0)]
    with pytest.raises(ValueError, match="use a new run_dir"):
        BOLoop(synthetic_config(tmp_path / "run", parameters=params), SyntheticEvaluator())


def test_pod_rank_is_chosen_automatically(synthetic_config, tmp_path):
    cfg = synthetic_config(tmp_path / "run", basis=BasisConfig(max_modes=4), budget=7)
    loop = BOLoop(cfg, SyntheticEvaluator())
    loop.run()
    assert 1 <= loop.basis.n_coeff <= 4


# --- evaluator contract --------------------------------------------------------

T2 = np.linspace(0.0, 10.0, 12)


class TwoParam:
    """y(t) = a*t + b*sin(t); optionally reports a fixed design back."""

    def __init__(self, report=None):
        self.report = report      # design to report back instead of x, if set

    def evaluate(self, x, case_dir: Path) -> Evaluation:
        a, b = x["a"], x["b"]
        y = a * T2 + b * np.sin(T2)
        return Evaluation(T2, y, np.full(len(T2), 0.1), x=self.report)


def two_param_config(synthetic_config, run_dir, **bo):
    params = [Parameter("a", -1.0, 1.0), Parameter("b", 0.0, 3.0)]
    return synthetic_config(run_dir, parameters=params, basis=BasisConfig(n_coeff=2),
                            objective=None, n_init=5, budget=7, **bo)


def test_loop_is_not_tied_to_four_parameters(synthetic_config, tmp_path):
    loop = BOLoop(two_param_config(synthetic_config, tmp_path / "run"), TwoParam())
    x_star, _ = loop.run()
    assert set(x_star) == {"a", "b"}
    assert np.array(loop.X).shape == (7, 2)


def test_the_simulated_design_is_recorded(synthetic_config, tmp_path):
    """An evaluator that reports a different design (it adopted an earlier
    case) gets that design recorded, not the proposal."""
    loop = BOLoop(two_param_config(synthetic_config, tmp_path / "run"),
                  TwoParam(report={"a": 0.5, "b": 1.5}))
    loop.ensure_initial()
    assert all(np.array_equal(x, [0.5, 1.5]) for x in loop.X)


def test_an_incomplete_reported_design_is_an_error(synthetic_config, tmp_path):
    loop = BOLoop(two_param_config(synthetic_config, tmp_path / "run"),
                  TwoParam(report={"a": 0.5}))
    with pytest.raises(ValueError, match="without"):
        loop.ensure_initial()


def test_a_changing_grid_is_an_error(synthetic_config, tmp_path):
    class Drifting(TwoParam):
        calls = 0

        def evaluate(self, x, case_dir):
            ev = super().evaluate(x, case_dir)
            Drifting.calls += 1
            return Evaluation(ev.t + Drifting.calls, ev.y, ev.std)

    loop = BOLoop(two_param_config(synthetic_config, tmp_path / "run"), Drifting())
    with pytest.raises(ValueError, match="t grid"):
        loop.ensure_initial()
