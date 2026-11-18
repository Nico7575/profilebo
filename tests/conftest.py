from pathlib import Path

import pytest
import yaml

from profilebo.config import BasisConfig, BOConfig, Config, ObjectiveConfig
from profilebo.evaluators.serpent.batches import parse_batches, resolve
from profilebo.evaluators.serpent.inputs import INPUT_NAME, generate_input
from profilebo.evaluators.synthetic import parse_parameters

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

SYNTHETIC_BOUNDS = {"R1": [0.005, 0.035], "R2": [0.040, 0.090],
                    "N1": [100, 500], "N2": [100, 400]}


@pytest.fixture(scope="session")      # a pure factory, so shareable
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


@pytest.fixture
def serpent_cfg():
    """The serpent section of examples/config.yaml, template path resolved."""
    cfg = yaml.safe_load((EXAMPLES / "config.yaml").read_text())["serpent"]
    cfg["template"] = EXAMPLES / cfg["template"]
    return cfg


@pytest.fixture
def write_serpent_case():
    """Render a case directory for design x, with a matching fake pbed file
    (fuel particles plus N particles of every batch universe)."""
    def write(case_dir, serpent_cfg, x):
        case_dir.mkdir(parents=True, exist_ok=True)
        specs = parse_batches(serpent_cfg)
        batches = resolve(specs, x)
        generate_input(batches, "particles.pbed", serpent_cfg["template"],
                       case_dir / INPUT_NAME, serpent_cfg)
        rows = [f"0 0 0 0.05 {serpent_cfg['fuel']['universe']}"] * 10
        for b in batches:
            rows += [f"0 0 0 {b.radius} {b.universe}"] * b.count
        (case_dir / "particles.pbed").write_text("\n".join(rows) + "\n")
        return specs, batches
    return write
