"""Composite Monte-Carlo acquisition.

Posterior samples of the coefficient vector are pushed through the
deterministic trajectory -> J map (GenericMCObjective), and qLogEI is
optimized over the unit cube; callers map the result back to physical
units.

(qLogNoisyExpectedImprovement was tried as a remedy for proposals that
replicate evaluated designs, and removed: its acquisition surface
had two near-equal maxima on this objective, so proposals alternated
between the incumbent and a distant point across repeated runs on
identical data, with persistent L-BFGS convergence warnings. The
replicate behaviour is not an acquisition defect; see the design
resolution floor in bo_loop.)
"""

from __future__ import annotations

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from torch.quasirandom import SobolEngine

from profilebo.optimization.objective import ProfileObjective


def propose_candidate(
    model: ModelListGP,
    objective: ProfileObjective,
    best_f: float,
    unit_bounds: torch.Tensor,
    num_restarts: int = 10,
    raw_samples: int = 512,
    mc_samples: int = 256,
    q: int = 1,
) -> tuple[torch.Tensor, float]:
    """Argmax of composite qLogEI. Returns (candidates (q, d) in unit
    coordinates, acquisition value).

    best_f is the incumbent in maximization convention, i.e. -J_best
    of the simulated trajectories.
    """
    acqf = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples])),
        objective=GenericMCObjective(objective),
    )
    candidates, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=unit_bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach(), float(acq_value)


def propose_distinguishable(
    model: ModelListGP,
    objective: ProfileObjective,
    best_f: float,
    exclude_unit: torch.Tensor,
    tol_unit: torch.Tensor,
    mc_samples: int = 256,
    n_candidates: int = 8192,
) -> tuple[torch.Tensor | None, float]:
    """Best acquisition point that differs from every evaluated design by more
    than the resolution in at least one parameter.

    Used when the unconstrained argmax lands on a design already evaluated:
    the acquisition surface has a needle there (the incumbent), which a random
    sweep never samples (measured: 0 of 20000 Sobol points fell inside a ball),
    so this is a coarse sweep with the balls masked out, not a re-optimization.
    Returns (candidate (1, d) in unit coords, acquisition value), or (None, -inf)
    if every candidate is indistinguishable.
    """
    acqf = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples])),
        objective=GenericMCObjective(objective),
    )
    d = exclude_unit.shape[-1]
    Z = SobolEngine(d, scramble=True, seed=0).draw(n_candidates).to(exclude_unit)
    with torch.no_grad():
        vals = acqf(Z.unsqueeze(1))

    # inside a ball => within tolerance in EVERY parameter of some evaluated point
    close = (torch.abs(Z.unsqueeze(1) - exclude_unit.unsqueeze(0)) <= tol_unit).all(dim=-1)
    inside = close.any(dim=1)
    if bool(inside.all()):
        return None, float("-inf")

    masked = torch.where(inside, torch.full_like(vals, float("-inf")), vals)
    i = int(torch.argmax(masked))
    return Z[i : i + 1].detach(), float(masked[i])
