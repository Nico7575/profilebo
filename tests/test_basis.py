import logging

import numpy as np
import pytest

from profilebo.surrogate.basis import PODBasis, TrajectoryBasis, noise_ratio, select_pod_basis

T = np.array([0.0, 1, 5, 15, 30, 50, 90, 130, 180, 270, 365, 548, 730,
              913, 1095, 1278, 1460, 1643, 1825, 2008, 2190, 2373])


def synthetic_trajectories(n=12, seed=0):
    """Three-parameter smooth family: constant + linear + exp decay."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2000, 2000, n)
    b = rng.uniform(-8, -2, n)
    c = rng.uniform(-15000, -5000, n)
    tau = 400.0
    return a[:, None] + b[:, None] * T + c[:, None] * np.exp(-T / tau)


def test_pod_reconstruction_exact_for_low_rank_family():
    y = synthetic_trajectories()
    basis = PODBasis.build(T, y, n_coeff=3)
    # The family is rank 3 around its mean -> near machine precision.
    assert basis.reconstruction_error(y) < 1e-6


def test_pod_reconstruction_improves_with_n():
    y = synthetic_trajectories() + np.random.default_rng(1).normal(0, 50, (12, len(T)))
    errs = [PODBasis.build(T, y, n).reconstruction_error(y) for n in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2]


def test_wls_variance_positive_and_scales_with_noise():
    y = synthetic_trajectories(n=6)
    basis = PODBasis.build(T, y, n_coeff=3)
    _, var_small = basis.fit_coefficients(y[0], std=np.full(len(T), 10.0))
    _, var_large = basis.fit_coefficients(y[0], std=np.full(len(T), 100.0))
    assert np.all(var_small > 0)
    # sigma x10 -> variance x100
    np.testing.assert_allclose(var_large / var_small, 100.0, rtol=1e-8)


def test_save_load_roundtrip(tmp_path):
    y = synthetic_trajectories()
    basis = PODBasis.build(T, y, 4)
    basis.save(tmp_path / "basis.npz")
    loaded = TrajectoryBasis.load(tmp_path / "basis.npz")
    C, _ = basis.fit_many(y)
    np.testing.assert_allclose(loaded.reconstruct(C), basis.reconstruct(C))
    np.testing.assert_array_equal(loaded.t, T)


def test_pod_needs_enough_snapshots():
    y = synthetic_trajectories(n=3)
    with pytest.raises(ValueError, match="snapshots"):
        PODBasis.build(T, y, n_coeff=5)


def test_phi_must_match_grid():
    with pytest.raises(ValueError, match="t grid"):
        TrajectoryBasis(T, np.zeros((len(T) - 1, 2)))


# --- rank selection -----------------------------------------------------------

def noisy_family(n=12, sigma=10.0, seed=0):
    """The rank-3 family plus white noise of known sigma."""
    y = synthetic_trajectories(n, seed)
    return y + np.random.default_rng(seed + 1).normal(0, sigma, y.shape), np.full(y.shape, sigma)


def test_selection_stops_at_the_noise_level():
    y, std = noisy_family()
    basis = select_pod_basis(T, y, std, max_modes=10)
    # Below 3 modes the truncation error dwarfs the noise; at 3 it is noise.
    assert basis.n_coeff == 3
    assert noise_ratio(basis, y, std) <= 1.0
    assert noise_ratio(PODBasis.build(T, y, 2), y, std) > 1.0


def test_selection_warns_when_max_modes_is_not_enough(caplog):
    y, std = noisy_family()
    with caplog.at_level(logging.WARNING):
        basis = select_pod_basis(T, y, std, max_modes=2)
    assert basis.n_coeff == 2
    assert "max_modes" in caplog.text


def test_selection_never_exceeds_snapshots_minus_one():
    y, std = noisy_family(n=4)
    basis = select_pod_basis(T, y, std, max_modes=10, target_ratio=1e-12)
    assert basis.n_coeff == 3


def test_fixed_rank_skips_the_search():
    y, std = noisy_family()
    assert select_pod_basis(T, y, std, max_modes=10, n_coeff=5).n_coeff == 5


def test_mask_restricts_the_check_to_the_window():
    y, std = noisy_family()
    # A tiny sigma outside the mask would dominate the ratio if those points
    # were counted; masked out, they are ignored.
    std_out = std.copy()
    std_out[:, :5] = 1e-3
    mask = np.ones(len(T), bool)
    mask[:5] = False
    b = PODBasis.build(T, y, 3)
    assert noise_ratio(b, y, std_out, mask) < noise_ratio(b, y, std_out)
