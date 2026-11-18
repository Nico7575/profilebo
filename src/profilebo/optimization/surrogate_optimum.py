"""Optimization on the surrogate alone, with no new simulation.

The archive of a campaign does not depend on the objective: the GPs model the
coefficients of the trajectory, not J. So the optimum for any objective (a
different target, alpha, window, or form of J) can be found on the fitted
surrogate in seconds, and compared with the best design already simulated.
This tells whether a new campaign for that objective is worth its budget.

The optimum maximizes the posterior mean of -J (qSimpleRegret on the same
Monte-Carlo composite the campaign uses), and its predicted J and trajectory
are given as posterior quantiles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from profilebo.config import Config
from profilebo.optimization.bo_loop import BOLoop
from profilebo.rng import derive_seed

DTYPE = torch.double

ACQ_SAMPLES = 256        # MC samples of the acquisition
NUM_RESTARTS = 20        # multi-start of its optimization
RAW_SAMPLES = 1024
PREDICT_SAMPLES = 2048   # posterior samples for the predicted J and trajectory

# The surrogate's answer is likely final when the optimum is this close to an
# evaluated design (distance in the unit cube) and its J interval is this tight
# (90th over 50th percentile).
NEAR = 0.05
TIGHT = 1.5
# A side of the design box more than this fraction of its range beyond the
# evaluated designs is reported: an optimum out there is extrapolation.
SLACK = 0.10


@dataclass
class SurrogateOptimum:
    names: list[str]
    x: np.ndarray            # the optimum on the surrogate
    t: np.ndarray            # grid of the predicted trajectory
    y: np.ndarray            # predicted trajectory: posterior median
    y_p10: np.ndarray
    y_p90: np.ndarray
    y_std: np.ndarray        # posterior std, for a +-2 sigma band
    J_p10: float
    J_p50: float
    J_p90: float
    best_tag: str            # best design already simulated, under this objective
    best_J: float
    distance: float          # from x to the nearest evaluated design, in the unit cube
    n_evals: int
    seed: int | None
    extrapolation: list[str] # sides of the box well beyond the evaluated designs

    @property
    def settled(self) -> bool:
        """True when the surrogate's answer is likely final: close to an
        evaluated design, with a tight predicted J."""
        return self.distance < NEAR and self.J_p90 / self.J_p50 < TIGHT

    def save(self, path: str | Path, meta: dict) -> None:
        """Write the prediction to a .npz file; 'profilebo report --prediction'
        draws it next to the simulated trajectories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("seed")
        d.pop("extrapolation")
        np.savez(path, **{k: np.asarray(v) for k, v in d.items()},
                 meta=json.dumps({**meta, "seed": self.seed}))


def load_prediction(path: str | Path) -> dict:
    """A prediction saved by SurrogateOptimum.save, as a dict of arrays."""
    d = np.load(path, allow_pickle=False)
    if "J_p50" not in d:
        raise ValueError(f"{path} is not a prediction saved by profilebo surrogate")
    out = {k: d[k] for k in d.files}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def _target_key(target) -> str:
    # A file target is keyed by its name, so the seed does not depend on where
    # the file is or how its path was written.
    return target.name if isinstance(target, Path) else str(target)


def with_objective(cfg: Config, **changes) -> Config:
    """cfg with some objective settings replaced (those given and not None)."""
    changes = {k: v for k, v in changes.items() if v is not None}
    return replace(cfg, objective=replace(cfg.objective, **changes)) if changes else cfg


def optimize_on_surrogate(cfg: Config) -> SurrogateOptimum:
    """The optimum of cfg's objective on the surrogate of cfg's campaign.

    Read only: it needs the archive and the basis of the campaign (so the
    initial design must be complete), and writes nothing.
    """
    for name in ("history.npz", "basis.npz"):
        if not (cfg.run_dir / name).exists():
            raise FileNotFoundError(
                f"no surrogate yet: {cfg.run_dir / name} does not exist (the basis is "
                "built when the initial design is complete)")

    # With bo.reproducible, seed the GP fit, the MC sampler and the multi-start
    # from bo.seed and the target, so a prediction can be regenerated exactly.
    seed = None
    if cfg.bo.reproducible:
        seed = derive_seed(cfg.bo.seed, "surrogate_opt", _target_key(cfg.objective.target))
        torch.manual_seed(seed)
        np.random.seed(seed)

    loop = BOLoop(cfg, evaluator=None)
    gp, objective, _ = loop.fit_surrogate()
    X = np.array(loop.X)

    lb, ub = cfg.bounds_array
    extrapolation = []
    for k, name in enumerate(cfg.parameter_names):
        lo, hi = X[:, k].min(), X[:, k].max()
        slack = SLACK * (ub[k] - lb[k])
        if lb[k] < lo - slack or ub[k] > hi + slack:
            extrapolation.append(
                f"{name}: the design box [{lb[k]:g}, {ub[k]:g}] reaches well beyond the "
                f"evaluated range [{lo:g}, {hi:g}]")

    J_arch = np.array([objective.j_observed(y) for y in loop.y])
    i0 = int(np.argmin(J_arch))

    acq = qSimpleRegret(
        model=gp.model,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([ACQ_SAMPLES]), seed=seed),
        objective=GenericMCObjective(objective),
    )
    cand, _ = optimize_acqf(acq, bounds=gp.unit_bounds, q=1,
                            num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
    x = cfg.round_integers(gp.from_unit(cand.detach().numpy()[0]))

    with torch.no_grad():
        post = gp.model.posterior(torch.as_tensor(gp.to_unit(np.atleast_2d(x)), dtype=DTYPE))
        coeffs = post.rsample(torch.Size([PREDICT_SAMPLES])).squeeze(1)
        Js = objective.j_from_coeffs(coeffs).numpy()
        ys = objective.reconstruct(coeffs).numpy()

    q10, q50, q90 = np.percentile(Js, [10, 50, 90])
    p10, p50, p90 = np.percentile(ys, [10, 50, 90], axis=0)
    distance = float(np.min(np.linalg.norm((X - x) / (ub - lb), axis=1)))
    return SurrogateOptimum(
        names=list(cfg.parameter_names), x=x, t=np.asarray(loop.t, dtype=float),
        y=p50, y_p10=p10, y_p90=p90, y_std=ys.std(axis=0),
        J_p10=float(q10), J_p50=float(q50), J_p90=float(q90),
        best_tag=loop.tags[i0], best_J=float(J_arch[i0]), distance=distance,
        n_evals=loop.n_evals, seed=seed, extrapolation=extrapolation,
    )


def describe(opt: SurrogateOptimum, cfg: Config) -> str:
    """The result as text, for the command line."""
    o = cfg.objective
    form = []
    if o.time_weighted:
        form.append("time-weighted")
    if o.flatness_beta is not None:
        form.append(f"flatness form, beta = {o.flatness_beta:g}")
    lines = [
        f"archive        {cfg.run_dir}: {opt.n_evals} evaluations",
        f"objective      target {o.target}, alpha {o.alpha:g}, window "
        f"[{o.window[0]:g}, {o.window[1]:g}]" + (f" ({', '.join(form)})" if form else ""),
    ]
    lines += [f"warning        {w}; an optimum there is extrapolation" for w in opt.extrapolation]
    lines += [
        f"best evaluated {opt.best_tag}: J = {opt.best_J:.4g} (simulated)",
        "surrogate optimum, not simulated:",
        "               " + ", ".join(f"{n} = {v:.6g}" for n, v in zip(opt.names, opt.x)),
        f"predicted J    median {opt.J_p50:.4g}, 80% interval [{opt.J_p10:.4g}, "
        f"{opt.J_p90:.4g}]",
        f"distance to the nearest evaluated design: {opt.distance:.3f} (unit cube)",
    ]
    if opt.settled:
        lines.append("verdict        close to evaluated designs, with a tight interval: the "
                     "surrogate's answer is likely final; a short campaign would only confirm it")
    else:
        lines.append("verdict        extrapolating or uncertain: a campaign for this objective "
                     "is worth its budget")
    if opt.seed is not None:
        lines.append(f"seed           {opt.seed} (bo.reproducible)")
    return "\n".join(lines)
