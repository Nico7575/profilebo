"""bo.reproducible: a campaign is a deterministic function of its config.

Two fresh runs, an interrupted-and-resumed run, and a parallel DoE must all
propose identical designs, for the composite loop and the scalar baseline.
"""

import numpy as np
import pytest

from profilebo.evaluators.synthetic import SyntheticEvaluator
from profilebo.optimization.bo_loop import BOLoop
from profilebo.optimization.scalar_baseline import ScalarBOLoop

pytestmark = pytest.mark.slow

def run_campaign(cfg, loop_cls=BOLoop):
    loop = loop_cls(cfg, SyntheticEvaluator())
    loop.run()
    return loop


def bo_designs(loop):
    return np.array([x for x, t in zip(loop.X, loop.tags) if t.startswith("bo_")])


def test_two_fresh_campaigns_propose_identical_designs(synthetic_config, tmp_path):
    a = run_campaign(synthetic_config(tmp_path / "a", budget=10))
    b = run_campaign(synthetic_config(tmp_path / "b", budget=10))
    assert len(bo_designs(a)) == 4
    np.testing.assert_array_equal(bo_designs(a), bo_designs(b))


def test_interrupted_campaign_matches_uninterrupted(synthetic_config, tmp_path):
    full = run_campaign(synthetic_config(tmp_path / "full", budget=10))
    run_campaign(synthetic_config(tmp_path / "cut", budget=8))      # stops after 2 BO steps
    resumed = run_campaign(synthetic_config(tmp_path / "cut", budget=10))
    np.testing.assert_array_equal(bo_designs(full), bo_designs(resumed))


def test_step_state_is_rebuilt_from_the_archive(synthetic_config, tmp_path):
    cfg = synthetic_config(tmp_path / "s", budget=9)
    live = run_campaign(cfg)
    resumed = BOLoop(cfg, SyntheticEvaluator())
    resumed._restore_step_state(resumed.fit_surrogate()[1])
    assert resumed._sigma_hist == pytest.approx(live._sigma_hist)


@pytest.mark.parametrize("loop_cls", [BOLoop, ScalarBOLoop])
def test_parallel_doe_order_does_not_change_proposals(synthetic_config, tmp_path, loop_cls):
    """The scalar loop overrides step(): it must still seed each step and fit
    in tag order, so a parallel DoE cannot change it either."""
    seq = run_campaign(synthetic_config(tmp_path / "seq", budget=10), loop_cls)
    par = run_campaign(synthetic_config(tmp_path / "par", budget=10, doe_parallel=3), loop_cls)
    assert len(bo_designs(seq)) == 4
    np.testing.assert_array_equal(bo_designs(seq), bo_designs(par))
