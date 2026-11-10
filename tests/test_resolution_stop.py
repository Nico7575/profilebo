"""Regression tests for the practical-convergence rule.
"""

from profilebo.optimization.bo_loop import BOLoop


def make_loop(synthetic_config, tmp_path, streak, factor):
    cfg = synthetic_config(tmp_path / "run", resolution_floor=True,
                           design_tolerance={"R1": 2e-4},
                           tolerance_streak=streak, noise_stop_factor=factor)
    return BOLoop(cfg, evaluator=None)


def test_single_sub_resolution_step_does_not_stop(synthetic_config, tmp_path):
    """The original bug: tiny EI_out at streak 1 must NOT end the campaign."""
    loop = make_loop(synthetic_config, tmp_path, streak=3, factor=0.1)
    assert loop._check_stop(streak=1, ei_out=1.2e3, sigma_J=8.38e4) is None
    assert loop._check_stop(streak=2, ei_out=1.2e3, sigma_J=8.38e4) is None


def test_stops_once_the_streak_is_reached(synthetic_config, tmp_path):
    loop = make_loop(synthetic_config, tmp_path, streak=3, factor=0.1)
    reason = loop._check_stop(streak=3, ei_out=1.2e3, sigma_J=8.38e4)
    assert reason is not None and "3 consecutive" in reason


def test_streak_alone_is_not_enough_when_exploration_still_pays(synthetic_config, tmp_path):
    """A long streak with high EI_out means distinguishable designs are still
    worth running; the noise factor guards against stopping there."""
    loop = make_loop(synthetic_config, tmp_path, streak=3, factor=0.1)
    assert loop._check_stop(streak=5, ei_out=5.0e4, sigma_J=8.38e4) is None


def test_streak_only_mode(synthetic_config, tmp_path):
    """With no noise factor the streak alone decides."""
    loop = make_loop(synthetic_config, tmp_path, streak=2, factor=None)
    assert loop._check_stop(streak=1, ei_out=1e9, sigma_J=1e4) is None
    assert loop._check_stop(streak=2, ei_out=1e9, sigma_J=1e4) is not None


def test_disabled_never_stops(synthetic_config, tmp_path):
    loop = make_loop(synthetic_config, tmp_path, streak=None, factor=0.1)
    assert loop._check_stop(streak=99, ei_out=1.0, sigma_J=1e9) is None
