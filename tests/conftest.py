import pytest

from profilebo.config import BasisConfig, BOConfig, Config, ObjectiveConfig
from profilebo.evaluators.synthetic import parse_parameters

SYNTHETIC_BOUNDS = {"R1": [0.005, 0.035], "R2": [0.040, 0.090],
                    "N1": [100, 500], "N2": [100, 400]}


@pytest.fixture
def synthetic_config():
    """A small, fast campaign on the synthetic evaluator. Keyword arguments
    override bo settings; basis/objective/parameters can be replaced whole.
    The acquisition settings are deliberately cheap: these campaigns test
    bookkeeping and determinism, not optimization quality."""
    def make(run_dir, basis=None, objective=None, parameters=None, **bo):
        bo_kw = dict(n_init=6, budget=9, seed=7, num_restarts=2, raw_samples=32,
                     mc_samples=32, reproducible=True)
        bo_kw.update(bo)
        return Config(
            parameters=parameters or parse_parameters({"bounds": SYNTHETIC_BOUNDS}),
            run_dir=run_dir,
            evaluator="synthetic",
            basis=basis or BasisConfig(n_coeff=4),
            objective=objective or ObjectiveConfig(alpha=100.0, target=1000.0,
                                                   window=(5.0, 1825.0)),
            bo=BOConfig(**bo_kw),
        )
    return make
