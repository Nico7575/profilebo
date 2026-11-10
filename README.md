# profilebo

Bayesian optimization of expensive simulations whose output is a
**trajectory**: a quantity evolving over time (or burnup, or any other grid)
that should follow a target profile.

`profilebo` was developed to design burnable-absorber loadings for a
prismatic high-temperature gas-cooled reactor, where each evaluation is a
multi-hour Serpent 2 depletion calculation and the goal is a core reactivity
that stays flat, just above a target, over the whole cycle. The optimization
core is application-agnostic: any simulator that maps design parameters to a
trajectory with statistical uncertainties can be plugged in.

> **Status (November 2026): work in progress.** The optimization pipeline is
> complete and tested; the Serpent evaluator is being ported. See
> [Status](#status).

## How it works

1. **Initial design.** A Latin hypercube of `n_init` design points is
   simulated (several in parallel, if the simulator allows).
2. **Trajectory reduction.** Each simulated trajectory is compressed into a
   few coefficients of a Proper Orthogonal Decomposition (POD) basis built
   from the initial designs. The number of modes is chosen automatically:
   the fewest whose truncation error is at the level of the simulation noise.
3. **Surrogate.** One Gaussian process per coefficient, with the simulation's
   statistical uncertainty propagated into fixed observation noise.
4. **Acquisition.** Posterior samples of the coefficients are turned back into
   trajectories and scored by the objective, and the next design maximizes
   the Monte-Carlo expected improvement of that composite (qLogEI). The
   objective is therefore post-processing: the same archive of simulations
   can be re-optimized for a different target without new runs.
5. **Stopping.** At the evaluation budget, or earlier when proposals keep
   falling below the design resolution (the smallest change that can be
   manufactured or distinguished from noise), or when the expected
   improvement drops below a threshold.

The objective is a weighted mean squared deviation from the target over a
window of the trajectory, with an extra penalty (`alpha`) for falling below
it. Optional variants weight each point by the share of the window it
represents, and separate flatness from offset.

For comparison, a conventional baseline that fits a single GP directly to the
scalar objective is included (`ScalarBOLoop`).

## Installation

Python 3.11 or newer.

```bash
git clone https://github.com/Nico7575/profilebo.git
cd profilebo
python -m venv .venv
source .venv/bin/activate

# Optional, recommended without a GPU: the CPU-only PyTorch build.
# A plain install on Linux pulls the much larger CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Check the installation by running the quick tests (about a minute):

```bash
pytest -m "not slow"
```

`pytest` alone also runs the slower tests, which execute complete
optimization campaigns (a few minutes).

## Quick start: a campaign without Serpent

`examples/synthetic.yaml` runs a full campaign on a cheap analytic stand-in
for the reactor problem: two batches of absorber particles (radii `R1`, `R2`,
counts `N1`, `N2`) compensating a declining fuel reactivity. The model has a
known answer, `R1 = 0.020 cm`, `R2 = 0.065 cm`, `N1 = 300`, `N2 = 250`, which
holds the reactivity flat at the 1000 pcm target.

```python
import logging

from profilebo.config import load_config
from profilebo.evaluators import make_evaluator
from profilebo.optimization.bo_loop import BOLoop

logging.basicConfig(level=logging.INFO)

cfg = load_config("examples/synthetic.yaml")
loop = BOLoop(cfg, make_evaluator(cfg))
x_best, j_best = loop.run()
print(x_best, j_best)
```

It takes a few minutes and ends close to the known answer. A command-line
interface (`profilebo run config.yaml`) is planned.

The campaign writes everything to `paths.run_dir` (relative to the config
file, here `examples/runs/synthetic/`):

| File | Content |
|---|---|
| `history.npz` | every evaluated design with its trajectory and uncertainties |
| `basis.npz` | the POD basis, built once after the initial design |
| `iterations.jsonl` | one line per optimization step: design, acquisition value, objective, stopping state |
| `cases/<tag>/` | the simulator's working directory for each evaluation |

A campaign is resumable: running it again with the same `run_dir` continues
where it stopped. With `bo.reproducible: true` it is also deterministic, so
an interrupted and resumed campaign proposes exactly the same designs as an
uninterrupted one.

## Configuration

A campaign is described by one YAML file with these sections:

| Section | Purpose |
|---|---|
| `serpent` or `synthetic` | the simulator (exactly one); the design parameters are derived from it |
| `objective` | `target` (a number or a two-column file), `window`, `alpha`, and the optional variants |
| `basis` | `max_modes` and `target_ratio` for the automatic POD rank, or a fixed `n_coeff` |
| `bo` | initial design size, budget, seed, acquisition settings, stopping rules |
| `paths` | `run_dir` |

`examples/config.yaml` documents every key for the Serpent case. There the
design parameters come from a list of absorber particle batches: each batch
radius or count given as `[lo, hi]` is optimized, a single number is held
fixed, and the number of batches is free.

## Using your own simulator

An evaluator is any object with one method:

```python
from pathlib import Path
from profilebo.evaluators.base import Evaluation

class MyEvaluator:
    def evaluate(self, x: dict[str, float], case_dir: Path) -> Evaluation:
        # run the simulation for design x, working in case_dir
        return Evaluation(t=grid, y=trajectory, std=one_sigma)
```

It should return when the result is available, reuse what is already in
`case_dir` if called again for the same case (this is what makes campaigns
resumable), and be safe to call from several threads. The full contract is in
`src/profilebo/evaluators/base.py`.

## Status

| Component | State |
|---|---|
| Configuration, POD basis, Gaussian processes, objective, acquisition | ported and tested |
| BO loop and scalar-objective baseline | ported, tested, and checked against the original research code: identical proposals, step by step |
| Synthetic evaluator and example | done |
| Serpent evaluator | in progress: input generation for any number of particle batches is done; particle dispersion, job submission (local and SLURM) and result parsing are being ported |
| Command-line interface, monitoring script, optimization on the surrogate alone | planned |

`examples/templates/single_assembly_blueprint.inp` is a **placeholder** Serpent
model: a single fuel block with periodic boundaries, provided so that the
pipeline can be run end to end. It is not the full-core model of the paper.

## Relation to the paper

The results in the accompanying paper were produced with the original
research code. `profilebo` is its cleaned-up, generalized reimplementation of
the same method; on a test problem the optimization loop reproduces the
original's decisions exactly. The deliberate differences:

- The number of POD modes is chosen automatically by default (the paper
  fixed it by hand; `basis.n_coeff` still fixes it).
- In the optional flatness form of the objective, the offset term uses the
  same point weights for the target as for the trajectory. This only matters
  for a tabulated target with non-uniform weights; the paper used a constant
  target.
- The Gaussian-process fit has one more fallback noise level, used only when
  a fit would otherwise fail and stop the campaign.

> The code is related to a paper currently under review process in the NSE journal.

## License

MIT; see [LICENSE](LICENSE).
