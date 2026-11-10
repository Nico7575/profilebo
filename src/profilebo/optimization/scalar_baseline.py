"""Scalar-objective BO baseline: a GP fitted directly to J (ablation).

The main architecture models the trajectory (POD coefficients with a
composite Monte-Carlo acquisition), so that J is post-processing and the
archive is objective-agnostic. The conventional alternative is to fit one
GP straight to the scalar objective and use analytic expected improvement.
This module implements that alternative with everything else held fixed
(same DoE seed, bounds, kernel, restarts, budget), so the two can be
compared like for like.

Two differences from the composite route are intrinsic, not choices:

  * J spans several orders of magnitude across the design box, so the GP
    must model -log J. The composite route needs no such transform: it
    models trajectory coefficients, which are linear in y.
  * The per-point statistics of the simulation cannot be propagated into a
    scalar objective analytically (J is a nonlinear functional of the noisy
    trajectory), so the observation noise is estimated by Monte Carlo from
    the simulated sigma instead of coming from the WLS covariance.

Both are reported.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

from profilebo.optimization.bo_loop import BOLoop
from profilebo.optimization.objective import ProfileObjective, build_objective
from profilebo.surrogate.gp_model import TrajectoryGP

log = logging.getLogger(__name__)

DTYPE = torch.double
_J_FLOOR = 1e-12          # guards log(J) if a trajectory sits exactly on target


def log_j_with_noise(
    y: np.ndarray,
    y_std: np.ndarray,
    objective: ProfileObjective,
    n_mc: int = 512,
    seed: int = 0,
) -> tuple[float, float]:
    """(-log J, Var[-log J]) for one simulated trajectory.

    The variance is propagated by sampling the trajectory from its own
    statistics and re-evaluating J, because there is no analytic route from
    sigma_y to the variance of a nonlinear functional.
    """
    y = np.asarray(y, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    j0 = max(float(objective.j_observed(y)), _J_FLOOR)

    rng = np.random.default_rng(seed)
    draws = y[None, :] + rng.normal(size=(n_mc, len(y))) * y_std[None, :]
    j_samples = objective.j_from_y(torch.as_tensor(draws, dtype=DTYPE)).numpy()
    y_var = float(np.var(-np.log(np.clip(j_samples, _J_FLOOR, None)), ddof=1))
    return -float(np.log(j0)), y_var


def propose_candidate_scalar(
    model,
    best_f: float,
    unit_bounds: torch.Tensor,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> tuple[torch.Tensor, float]:
    """Argmax of analytic LogEI on the scalar surrogate.

    'model' is the one-output ModelListGP produced by TrajectoryGP; the
    single sub-model is handed to the analytic acquisition directly.
    """
    acqf = LogExpectedImprovement(
        model=model.models[0], best_f=best_f, maximize=True
    )
    candidates, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=unit_bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach(), float(acq_value)


class ScalarBOLoop(BOLoop):
    """BOLoop with the composite surrogate replaced by a GP on -log J.

    Everything else (evaluator, DoE, checkpointing, fault tolerance) is
    inherited unchanged, so a run differs from the composite campaign only
    in the surrogate and the acquisition. No trajectory basis is built. The
    stopping rule is a plain budget: the composite loop's resolution test
    depends on the trajectory posterior and has no scalar analogue, and a
    fixed budget is the fair comparison anyway.
    """

    def fit_surrogate_scalar(self) -> tuple[TrajectoryGP, ProfileObjective, float]:
        objective = build_objective(self.cfg.objective, t=self.t)
        order = self._training_order()
        obs = [log_j_with_noise(self.y[i], self.y_std[i], objective) for i in order]
        y = np.array([[o[0]] for o in obs])
        y_var = np.array([[o[1]] for o in obs])

        gp = TrajectoryGP(self.cfg.bounds_array).fit(np.array(self.X)[order], y, y_var)
        return gp, objective, float(y.max())

    def step(self) -> dict:
        tag = f"bo_{self.n_evals:03d}"
        self._seed_step(tag)
        gp, objective, best_f = self.fit_surrogate_scalar()
        cand_unit, acq_value = propose_candidate_scalar(
            gp.model,
            best_f,
            gp.unit_bounds,
            num_restarts=self.cfg.bo.num_restarts,
            raw_samples=self.cfg.bo.raw_samples,
        )
        x = self.cfg.round_integers(gp.from_unit(cand_unit.numpy()[0]))

        log.info("Iteration %s: logEI=%.4g, best -logJ=%.4g, x=%s",
                 tag, acq_value, best_f, self.cfg.as_point(x))

        self._evaluate(x, tag)
        j_new = objective.j_observed(self.y[-1])
        record = {
            "tag": tag,
            "x": self.cfg.as_point(self.X[-1]),   # the simulated x
            "acq_value": acq_value,
            "J_observed": j_new,
            "J_best": min(float(np.exp(-best_f)), j_new),
            "surrogate": "scalar_logJ",
        }
        self._log_iteration(record)
        return record
