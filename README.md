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

> **Status:** version 0.1.0 is complete and working: the whole method of the
> paper is ported, tested, and checked against the original research code.
> The reference to the paper, and a DOI for the software, will be added when
> the paper is published.

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
   can be re-optimized for a different target without new runs
   (`profilebo surrogate`).
5. **Stopping.** At the evaluation budget, or earlier when proposals keep
   falling below the design resolution (the smallest change that can be
   manufactured or distinguished from noise), or when the expected
   improvement drops below a threshold.

The objective is a weighted mean squared deviation from the target over a
window of the trajectory, with an extra penalty (`alpha`) for falling below
it. Optional variants weight each point by the share of the window it
represents, and separate flatness from offset.

For comparison, a conventional baseline that fits a single GP directly to the
scalar objective is included (`profilebo run --scalar`).

## Installation

You need Python 3.11 or newer and git. Serpent 2 is only needed for the
reactor example; it is licensed separately and must be installed on its own.
Choose one of the two ways below.

### With pip and a virtual environment

```bash
git clone https://github.com/Nico7575/profilebo.git
cd profilebo
python -m venv .venv            # create an isolated environment in .venv/
source .venv/bin/activate       # activate it (repeat in every new terminal)

# Optional, recommended on machines without a GPU: the CPU-only PyTorch build.
# A plain install on Linux pulls the much larger CUDA build.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev,plots]"
```

The last line installs profilebo and everything it depends on, as listed in
`pyproject.toml`:

- `.` means "the package in this folder".
- `-e` (editable) links the installation to this folder instead of copying
  it, so changes you make to the code take effect without reinstalling. If you
  only want to use the package, `pip install ".[dev,plots]"` works too.
- `[dev,plots]` adds two optional extras: `dev` brings `pytest` to run the
  tests, `plots` brings `matplotlib` for `profilebo report`. Without them,
  `pip install -e .` installs the optimizer alone.
- The quotes stop some shells (zsh, for instance) from interpreting the
  square brackets.

### With conda

```bash
git clone https://github.com/Nico7575/profilebo.git
cd profilebo
conda env create -f environment.yml    # Python 3.12, CPU-only PyTorch, profilebo
conda activate profilebo
```

`environment.yml` only sets up Python and then runs the same pip
installation as above (including the CPU-only PyTorch build), so both ways
install the package from the same list of dependencies in `pyproject.toml`.

### Check the installation

Run the quick tests (about a minute):

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

```bash
profilebo run examples/synthetic.yaml
```

It takes a few minutes and ends close to the known answer, printing the best
design and its objective `J`. Then look at the campaign:

```bash
profilebo report examples/synthetic.yaml
```

This prints a summary and writes it, with the figures described in
[The report](#the-report), to `examples/runs/synthetic/report/`.

The campaign writes everything to `paths.run_dir` (relative to the config
file, here `examples/runs/synthetic/`):

| File | Content |
|---|---|
| `history.npz` | every evaluated design with its trajectory and uncertainties |
| `basis.npz` | the POD basis, built once after the initial design |
| `iterations.jsonl` | one line per optimization step: design, acquisition value, objective, stopping state |
| `cases/<tag>/` | the simulator's working directory for each evaluation |
| `campaign.log` | the log of every run of the campaign |
| `report/` | the output of `profilebo report` |

A campaign is resumable: stop it with `Ctrl-C` at any time, and running the
same command again continues where it stopped. With `bo.reproducible: true` it is also deterministic, so
an interrupted and resumed campaign proposes exactly the same designs as an
uninterrupted one.

## Running with Serpent

Serpent 2 must be installed, with its nuclear data. The steps:

1. Copy `examples/config.yaml` and set `serpent.executable` (the `sss2`
   binary) and `serpent.template` (your model; see
   `examples/templates/single_assembly_blueprint.inp` for the placeholders
   it must contain). That example is a single fuel block with periodic
   boundaries, there to run the pipeline end to end; it is not the
   full-core model of the paper.
2. Choose where Serpent runs with `serpent.scheduler`:
   - `slurm`: every case is a SLURM job; fill in `serpent.slurm`
     (partition, account, tasks, time limit, modules, data paths).
   - `local`: every case runs in the background on this machine; fill in
     `serpent.local`. Keep `bo.doe_parallel` times the cores per case within
     the cores you have.
3. Start the campaign with `profilebo run my_config.yaml`, from a session
   that stays alive: inside `tmux` or `screen`, or, on a cluster, as a SLURM
   job of its own with [examples/optimizer.sbatch](examples/optimizer.sbatch)
   (fill in partition, account and paths, then `sbatch optimizer.sbatch`).
   The optimizer itself needs one core; the cases are the jobs. As a job,
   it submits the cases from a compute node, which the cluster must allow;
   put it where it does not take a slot the cases need, and keep
   `bo.doe_parallel` within what the cluster runs at once for you.

Each case directory `cases/<tag>/` holds the rendered input `case.serp`, the
particle positions `particles.pbed`, the job script, Serpent's output
(`OUT_pbo_<tag>.log`, `case.serp_res.m`) and the `job_id` of the run.

If the optimizer is stopped (`Ctrl-C`, `kill`, `scancel`, a lost connection), the running cases
are not: start it again with the same command, and it re-attaches to them, reuses the finished ones, and
goes on. A case that failed for a transient reason (a node fault, a time
limit) is run again up to `max_attempts` times; one that Serpent rejected
(an input error) stops the campaign with Serpent's message.

If a job fails at once with `error while loading shared libraries`, the job
cannot find the MPI library `sss2` was built with: load its module through
`serpent.slurm.modules` (`ldd $(which sss2)` lists what is missing).

## The report

`profilebo report config.yaml` only reads the run directory, so it can run at
any time, also while the campaign is running. It prints a summary (how far the
campaign is, why it stopped, the best design and its `J` with the statistical
error, which designs are a statistical tie with it, the parameters on a bound)
and writes it to `report/summary.txt`, with these figures:

| Figure | What it shows | What to look for |
|---|---|---|
| `trajectories` | the best trajectory with its 2-sigma band, the target, the other evaluations in grey | the best design follows the target inside the window (the shaded parts are outside it) |
| `convergence` | `J` of every evaluation, and the best so far | the optimization steps cluster low; a flat staircase means diminishing returns |
| `acquisition` | the improvement the surrogate still expects at every step, against the statistical error of `J` | when the expected improvement falls to the noise level, more steps cannot be told apart from the best |
| `parameters` | every evaluated design, parameter by parameter, with the bounds | the best design (star) or the late steps sitting on a bound: the optimum may lie outside the design box |
| `contributions` | where the best design's `J` comes from, point by point | with a large `alpha`, a small dip below the target can dominate `J` |
| `ranking` | the best evaluations with their statistical error bars | overlapping bars are designs that cannot be told apart |

`--gif` adds `convergence.gif`, one frame per evaluation (the initial design
from worst to best, then the optimization steps). `--show doe_004 bo_017`
draws those evaluations in the trajectory figure as well, `--prediction
FILE.npz` draws a prediction saved by `profilebo surrogate`, and `--format pdf`
writes the figures as PDF.

## Optimization on the surrogate alone

The surrogate models the trajectory, not `J`, so once a campaign has a
surrogate (after its initial design), the optimum of any objective can be
found on it in about a minute, with no new simulation:

```bash
profilebo surrogate config.yaml                       # the config's own objective
profilebo surrogate config.yaml --target 500          # another constant target
profilebo surrogate config.yaml --target profile.txt  # a tabulated target (t, target)
profilebo surrogate config.yaml --alpha 10 --window 5 2100 --time-weighted
profilebo surrogate config.yaml --flatness-beta 0 --save predictions/flat.npz
```

The options replace the objective of the config, and only for this command.
It prints:

- the best design already simulated, under this objective;
- the optimum on the surrogate, with its predicted `J` (median and 80%
  interval of the posterior);
- its distance to the nearest simulated design;
- a verdict. Close to the simulated designs with a tight interval, the
  surrogate's answer is likely final, and a campaign would only confirm it.
  Otherwise, a campaign for this objective is worth its budget.

It warns when a side of the design box reaches well beyond the simulated
designs, since an optimum there is extrapolation. With `bo.reproducible`, the
result is seeded from `bo.seed` and the target, so it can be regenerated
exactly. `--save` writes the optimum and its predicted trajectory, which
`profilebo report config.yaml --prediction FILE.npz` draws next to the
simulated ones. The command only reads the campaign, so it can also run
while the campaign is running.

## Command summary

| Task | Command |
|---|---|
| Install (in the cloned folder) | `pip install -e ".[dev,plots]"` or `conda env create -f environment.yml` |
| Activate the environment | `source .venv/bin/activate` or `conda activate profilebo` |
| Check the installation | `pytest -m "not slow"` |
| Run all tests | `pytest` |
| Run a campaign | `profilebo run config.yaml` |
| Stop it | `Ctrl-C`, or `kill`/`scancel` it (running simulations keep going) |
| Resume it | `profilebo run config.yaml` again |
| Start it over | delete its `run_dir`, then `profilebo run config.yaml` |
| Scalar-objective baseline | `profilebo run config.yaml --scalar` (with its own `run_dir`) |
| More detail in the log | `profilebo run config.yaml -v` |
| Summary and figures (any time) | `profilebo report config.yaml` |
| ... with an animation, some evaluations highlighted, as PDF | `profilebo report config.yaml --gif --show doe_004 bo_017 --format pdf` |
| Optimum on the surrogate, no simulation | `profilebo surrogate config.yaml` |
| ... for another objective, saved | `profilebo surrogate config.yaml --target 500 --save pred.npz` |
| ... drawn in the report | `profilebo report config.yaml --prediction pred.npz` |
| Help | `profilebo --help`, `profilebo run --help`, `profilebo report --help`, `profilebo surrogate --help` |

On a cluster, keep the optimizer alive after you log out, and follow the jobs
with the scheduler:

```bash
# either in a terminal multiplexer, if the cluster has one
tmux new -s campaign                      # detach: Ctrl-b d; back: tmux attach -t campaign
profilebo run config.yaml
# or as a SLURM job of its own (examples/optimizer.sbatch, filled in)
sbatch optimizer.sbatch                   # stop: scancel <its job id>

squeue --me                               # the jobs: pbo_optimizer, pbo_<tag>
tail -f runs/<name>/campaign.log          # the optimizer's log
profilebo report config.yaml              # where the campaign stands, with figures
```

From Python, the same campaign is:

```python
from profilebo.config import load_config
from profilebo.evaluators import make_evaluator
from profilebo.optimization.bo_loop import BOLoop

cfg = load_config("config.yaml")
x_best, j_best = BOLoop(cfg, make_evaluator(cfg)).run()
```

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
- `profilebo surrogate` seeds a tabulated target by the file's name; the
  original used the path as written. For a constant target the seed, and the
  result, are the same.

> The code is related to a paper currently under review process in the NSE journal.

## Citation

To cite the software, use [CITATION.cff](CITATION.cff) (on GitHub: "Cite
this repository"). The reference to the paper will be added once it is
published.

## License

MIT; see [LICENSE](LICENSE).
