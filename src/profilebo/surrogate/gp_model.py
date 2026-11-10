"""Per-coefficient GPs over parameter-space.

One independent GP per basis coefficient, Matern-5/2 with ARD (the design
parameters generally live on distinct scales). Observation noise is fixed to
the WLS coefficient variances propagated from the simulator's statistical
uncertainties, rather than inferred from the data.

Inputs are normalized to the unit cube internally.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

log = logging.getLogger(__name__)

DTYPE = torch.double

# Relative noise floors (fraction of each coefficient's data variance).
# Late in a campaign the proposals cluster tightly; with the WLS noise at
# ~1e-6 of the outcome scale the Gram matrix becomes near-singular and
# MLL optimization aborts. Flooring the fixed noise restores conditioning
# without materially changing the posterior. The floors are tried in order
# and only escalate on failure; 1e-1 is a last resort for a coefficient that
# is barely above noise, so that one ill-conditioned fit does not end the
# campaign (a fit that succeeds at a lower floor never reaches it).
_NOISE_FLOORS = (1e-4, 1e-2, 1e-1)


class TrajectoryGP:
    """Independent fixed-noise GPs c_k(x), k = 1..n_coeff."""

    def __init__(self, bounds: np.ndarray):
        """bounds: (2, d) array of [lower; upper] physical bounds."""
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[1]
        self.model: ModelListGP | None = None

    # --- normalization -------------------------------------------------
    def to_unit(self, X: np.ndarray) -> np.ndarray:
        lb, ub = self.bounds
        return (np.asarray(X, dtype=float) - lb) / (ub - lb)

    def from_unit(self, X: np.ndarray) -> np.ndarray:
        lb, ub = self.bounds
        return np.asarray(X, dtype=float) * (ub - lb) + lb

    @property
    def unit_bounds(self) -> torch.Tensor:
        return torch.stack([torch.zeros(self.dim, dtype=DTYPE), torch.ones(self.dim, dtype=DTYPE)])

    # --- fitting ---------------------------------------------------------
    def fit(self, X: np.ndarray, C: np.ndarray, C_var: np.ndarray) -> "TrajectoryGP":
        """X: (n, d) design points in physical units; C, C_var: (n, n_coeff) coefficients and
        their observation-noise variances."""
        train_X = torch.as_tensor(self.to_unit(X), dtype=DTYPE)
        C = torch.as_tensor(np.atleast_2d(C), dtype=DTYPE)
        C_var = torch.as_tensor(np.atleast_2d(C_var), dtype=DTYPE)

        models = []
        for k in range(C.shape[1]):
            data_var = C[:, k].var().clamp_min(1e-12)
            gp = None
            for floor in _NOISE_FLOORS:
                yvar = C_var[:, k : k + 1].clamp_min(floor * data_var)
                gp = SingleTaskGP(
                    train_X,
                    C[:, k : k + 1],
                    train_Yvar=yvar,
                    covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim)),
                    outcome_transform=Standardize(m=1),
                )
                try:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
                    break
                except ModelFittingError:
                    log.warning(
                        "GP fit for coefficient %d failed at noise floor %g; "
                        "escalating", k, floor,
                    )
            else:
                raise ModelFittingError(
                    f"GP for coefficient {k} failed at every noise floor "
                    f"{_NOISE_FLOORS}"
                )
            models.append(gp)

        self.model = ModelListGP(*models)
        return self

    # --- prediction -------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance of the coefficients at design points X
        (physical units).
        Returns (mean, var), each (m, n_coeff)."""
        if self.model is None:
            raise RuntimeError("Call fit() first")
        test_X = torch.as_tensor(self.to_unit(np.atleast_2d(X)), dtype=DTYPE)
        with torch.no_grad():
            post = self.model.posterior(test_X)
            return post.mean.numpy(), post.variance.numpy()
