"""GP fit and acquisition on a small synthetic problem.

The GP learns the coefficients, and both proposal routines
return points inside the unit cube, the distinguishable
one outside every exclusion ball.
"""

import numpy as np
import torch

from profilebo.config import ObjectiveConfig
from profilebo.optimization.acquisition import propose_candidate, propose_distinguishable
from profilebo.optimization.objective import build_objective
from profilebo.surrogate.basis import PODBasis
from profilebo.surrogate.gp_model import TrajectoryGP

T = np.linspace(0.0, 100.0, 15)
BOUNDS = np.array([[0.0, 1.0], [10.0, 5.0]])     # x1 in [0, 10], x2 in [1, 5]


def simulate(X):
    """Trajectory y(t; x): an offset in x1 and a decay rate in x2."""
    X = np.atleast_2d(X)
    return X[:, :1] * 100.0 - 500.0 + 1000.0 * np.exp(-T[None, :] * X[:, 1:] / 50.0)


def fitted():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = BOUNDS[0] + rng.random((12, 2)) * (BOUNDS[1] - BOUNDS[0])
    Y = simulate(X)
    basis = PODBasis.build(T, Y, n_coeff=3)
    C, C_var = basis.fit_many(Y, np.full_like(Y, 1.0))
    gp = TrajectoryGP(BOUNDS).fit(X, C, C_var)
    obj = build_objective(ObjectiveConfig(alpha=10.0, target=0.0, window=(0.0, 100.0)),
                          basis=basis)
    J = np.array([obj.j_observed(y) for y in Y])
    return X, C, gp, obj, J


def test_gp_reproduces_training_coefficients():
    X, C, gp, _, _ = fitted()
    mean, var = gp.predict(X)
    assert mean.shape == C.shape and (var > 0).all()
    np.testing.assert_allclose(mean, C, atol=0.05 * np.abs(C).max())


def test_unit_mapping_roundtrip():
    gp = TrajectoryGP(BOUNDS)
    x = np.array([[2.5, 3.0]])
    np.testing.assert_allclose(gp.from_unit(gp.to_unit(x)), x)


def test_propose_candidate_is_in_the_unit_cube():
    _, _, gp, obj, J = fitted()
    cand, acq = propose_candidate(gp.model, obj, best_f=-J.min(), unit_bounds=gp.unit_bounds,
                                  num_restarts=2, raw_samples=32, mc_samples=32)
    assert cand.shape == (1, 2)
    assert ((cand >= 0) & (cand <= 1)).all()
    assert np.isfinite(acq)


def test_propose_distinguishable_avoids_the_excluded_balls():
    X, _, gp, obj, J = fitted()
    exclude = torch.as_tensor(gp.to_unit(X), dtype=torch.double)
    tol = torch.full((2,), 0.05, dtype=torch.double)
    cand, acq = propose_distinguishable(gp.model, obj, -J.min(), exclude, tol,
                                        mc_samples=32, n_candidates=512)
    assert cand is not None and np.isfinite(acq)
    assert ((cand - exclude).abs() > tol).any(dim=-1).all()


def test_propose_distinguishable_gives_up_when_everything_is_excluded():
    X, _, gp, obj, J = fitted()
    exclude = torch.as_tensor(gp.to_unit(X[:1]), dtype=torch.double)
    tol = torch.full((2,), 1.0, dtype=torch.double)      # one ball covers the cube
    cand, acq = propose_distinguishable(gp.model, obj, -J.min(), exclude, tol,
                                        mc_samples=16, n_candidates=64)
    assert cand is None and acq == float("-inf")
