import numpy as np
import pytest
import torch

from profilebo.config import ObjectiveConfig
from profilebo.optimization.objective import (
    ProfileObjective,
    build_objective,
    j_of_measured,
    target_on_grid,
    trapezoid_weights,
)
from profilebo.surrogate.basis import TrajectoryBasis

T = np.linspace(0.0, 2000.0, 21)


def fixed_basis(n_coeff=3):
    """A hand-made basis (constant, linear, quadratic in t) so tests control it."""
    s = T / T[-1]
    return TrajectoryBasis(T, np.column_stack([s**k for k in range(n_coeff)]))


def make_objective(alpha=100.0, window=(0.0, 2000.0)):
    return ProfileObjective(
        target=np.zeros(len(T)),
        window_mask=(T >= window[0]) & (T <= window[1]),
        alpha=alpha,
        basis=fixed_basis(),
    )


def test_asymmetric_penalty():
    obj = make_objective(alpha=100.0)
    above = torch.full((len(T),), 100.0, dtype=torch.double)
    j_above = obj.j_from_y(above)
    j_below = obj.j_from_y(-above)
    torch.testing.assert_close(j_below / j_above, torch.tensor(100.0, dtype=torch.double))


def test_window_mask():
    obj = make_objective(window=(0.0, 1000.0))
    y = torch.zeros(len(T), dtype=torch.double)
    y[T > 1000.0] = 1.0e4  # huge deviation outside the window
    assert float(obj.j_from_y(y)) == 0.0


def test_call_matches_j_from_coeffs_and_maximizes():
    obj = make_objective()
    coeffs = torch.randn(8, 2, 1, 3, dtype=torch.double)  # (mc, batch, q, n_c)
    out = obj(coeffs)
    assert out.shape == (8, 2, 1)
    torch.testing.assert_close(out, -obj.j_from_coeffs(coeffs))
    assert torch.all(out <= 0)  # -J, J >= 0


def test_objective_without_basis_scores_trajectories_but_not_coefficients():
    obj = ProfileObjective(target=np.zeros(len(T)), window_mask=np.ones(len(T), bool), alpha=1.0)
    assert obj.j_observed(np.full(len(T), 3.0)) == pytest.approx(9.0)
    with pytest.raises(RuntimeError, match="no basis"):
        obj.j_from_coeffs(torch.zeros(3, dtype=torch.double))


def test_build_objective_from_config():
    cfg = ObjectiveConfig(alpha=50.0, target=200.0, window=(0.0, 1500.0))
    obj = build_objective(cfg, basis=fixed_basis())
    assert obj.j_observed(np.full(len(T), 200.0)) == 0.0   # exactly on target


def test_build_objective_rejects_a_grid_that_differs_from_the_basis():
    cfg = ObjectiveConfig(window=(0.0, 1500.0))
    with pytest.raises(ValueError, match="basis grid"):
        build_objective(cfg, t=T + 1.0, basis=fixed_basis())


# --- tabulated target ----------------------------------------------------------

def test_target_file_is_interpolated(tmp_path):
    f = tmp_path / "target.txt"
    np.savetxt(f, [[0.0, 0.0], [2000.0, 2000.0]])
    np.testing.assert_allclose(target_on_grid(f, T), T)
    # A path given as a string works the same.
    np.testing.assert_allclose(target_on_grid(str(f), T), T)


def test_target_file_must_cover_the_window(tmp_path):
    f = tmp_path / "target.txt"
    np.savetxt(f, [[100.0, 0.0], [1000.0, 0.0]])
    with pytest.raises(ValueError, match="window needs"):
        target_on_grid(f, T, window=(0.0, 1500.0))
    target_on_grid(f, T, window=(100.0, 1000.0))     # inside the table: fine


def test_measured_j_uses_a_target_file(tmp_path):
    f = tmp_path / "target.txt"
    np.savetxt(f, [[0.0, 1000.0], [2000.0, 3000.0]])
    cfg = ObjectiveConfig(alpha=1.0, target=f, window=(0.0, 2000.0))
    assert j_of_measured(1000.0 + T, T, cfg) == pytest.approx(0.0, abs=1e-18)


# --- time-weighted J (objective.time_weighted) ---------------------------
# A front-loaded grid makes a plain mean over POINTS over-weight early t.
# trapezoid_weights gives each point the share of the window it represents
# instead. Default stays uniform.

GRID = np.array([0, 1, 5, 30, 180, 365, 730, 1095, 1460, 1825, 2190], float)


def _cfg(time_weighted=False, beta=None, alpha=100.0):
    return ObjectiveConfig(alpha=alpha, target=1000.0, window=(5.0, 2100.0),
                           time_weighted=time_weighted, flatness_beta=beta)


def _j(y, cfg):
    return j_of_measured(y, GRID, cfg)


def test_trapezoid_weights_sum_to_one_and_span_the_window():
    w = trapezoid_weights(GRID, 5.0, 2100.0)
    assert w.shape == (8,)                       # the in-window points
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_trapezoid_downweights_the_front_loaded_points():
    w = trapezoid_weights(GRID, 5.0, 2100.0)
    uniform = 1.0 / len(w)
    assert w[0] < uniform / 10          # t = 5 covers a sliver of the window
    assert w[-1] > uniform              # t = 1825 covers far more
    # t = 5/30/180 carry 37.5 % of a uniform J but ~13 % of the window
    assert w[:3].sum() == pytest.approx(0.128, abs=0.01)


def test_uniform_is_the_default():
    y = np.linspace(3000, 500, len(GRID))
    m = (GRID >= 5) & (GRID <= 2100)
    dev = y[m] - 1000.0
    expect = float(np.mean(np.where(dev >= 0, 1.0, 100.0) * dev**2))
    assert _j(y, _cfg()) == pytest.approx(expect)


def test_time_weighted_matches_the_weighted_sum():
    y = np.linspace(3000, 500, len(GRID))
    m = (GRID >= 5) & (GRID <= 2100)
    dev = y[m] - 1000.0
    w = trapezoid_weights(GRID, 5.0, 2100.0)
    expect = float(np.sum(w * np.where(dev >= 0, 1.0, 100.0) * dev**2))
    got = _j(y, _cfg(time_weighted=True))
    assert got == pytest.approx(expect)
    assert got != pytest.approx(_j(y, _cfg()))


def test_a_late_deviation_costs_more_when_time_weighted():
    """The point of the reweighting: late excursions stop being cheap."""
    flat = np.full(len(GRID), 1000.0)
    early, late = flat.copy(), flat.copy()
    early[2] += 5000.0            # t = 5, a sliver of the window
    late[9] += 5000.0             # t = 1825, the largest share
    u, tw = _cfg(), _cfg(time_weighted=True)
    assert _j(early, u) == pytest.approx(_j(late, u))       # uniform: identical
    assert _j(late, tw) > 30 * _j(early, tw)


def test_single_point_window_is_degenerate_but_valid():
    assert trapezoid_weights(GRID, 300.0, 400.0) == pytest.approx([1.0])


# --- flatness form (objective.flatness_beta) -----------------------------
# J = Var_w + beta*(mean_w - target)^2 + alpha*sum w_i max(0, target - y_i)^2,
# which separates FLATNESS from OFFSET. beta=1 with alpha=0 is the symmetric
# original exactly; beta=0 optimises flatness alone.

@pytest.mark.parametrize("time_weighted", [False, True])
def test_beta_one_reproduces_the_symmetric_original(time_weighted):
    """sum w (y-target)^2 == Var_w + (mean_w-target)^2, the identity the form rests on."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = 1000.0 + rng.normal(0, 800, len(GRID)).cumsum()
        a = _j(y, _cfg(time_weighted, beta=None, alpha=1.0))
        b = _j(y, _cfg(time_weighted, beta=1.0, alpha=0.0))
        assert a == pytest.approx(b, rel=1e-12)


def test_beta_zero_ignores_the_absolute_level():
    """Pure flatness: shifting a trajectory that stays above target must not move J."""
    y = 4000.0 + np.linspace(0, 900, len(GRID))
    assert _j(y + 1500.0, _cfg(beta=0.0)) == pytest.approx(_j(y, _cfg(beta=0.0)))
    # ... whereas the distance-to-target form does care: deviations run
    # ~3000-3900, so a +1500 shift roughly doubles the squared deviation.
    assert _j(y + 1500.0, _cfg()) == pytest.approx(2.04 * _j(y, _cfg()), rel=0.02)


def test_beta_penalises_offset_again_once_nonzero():
    y = 4000.0 + np.linspace(0, 900, len(GRID))
    assert _j(y + 1500.0, _cfg(beta=0.3)) > _j(y, _cfg(beta=0.3))


def test_undershoot_is_still_punished_under_pure_flatness():
    """beta=0 must not let a perfectly flat trajectory sit below the target."""
    flat_high = np.full(len(GRID), 3000.0)
    flat_low = np.full(len(GRID), 400.0)          # flat, but 600 under target
    assert _j(flat_high, _cfg(beta=0.0)) == pytest.approx(0.0, abs=1e-9)
    assert _j(flat_low, _cfg(beta=0.0)) > 1e7


def test_flatness_composes_with_time_weighting():
    """The early drop dominates the variance unless the time weights damp it."""
    y = np.full(len(GRID), 2000.0)
    y[2] = 6000.0                                  # excursion at t = 5
    uniform = _j(y, _cfg(beta=0.0))
    timewtd = _j(y, _cfg(time_weighted=True, beta=0.0))
    assert timewtd < uniform / 10


def test_negative_beta_is_rejected():
    with pytest.raises(ValueError, match="flatness_beta"):
        ProfileObjective(target=np.zeros(3), window_mask=np.ones(3, bool),
                         alpha=1.0, flatness_beta=-1.0)


def test_flatness_offset_uses_the_weighted_target_mean():
    """A constant y at the weighted mean of a tabulated target has no offset.
    The unweighted target mean would leave one when the weights are not uniform."""
    target = np.linspace(500.0, 1500.0, len(GRID))
    m = (GRID >= 5) & (GRID <= 2100)
    w = trapezoid_weights(GRID, 5.0, 2100.0)
    obj = ProfileObjective(target=target, window_mask=m, alpha=0.0,
                           time_weights=w, flatness_beta=1.0)
    level = float(np.sum(w * target[m]))
    assert obj.j_observed(np.full(len(GRID), level)) == pytest.approx(0.0, abs=1e-9)
    assert obj.j_observed(np.full(len(GRID), target[m].mean())) > 1.0


@pytest.mark.parametrize("beta", [None, 0.0, 0.3, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_sigma_j_gradient_matches_finite_differences(beta, weighted):
    """sigma_J propagates dJ/dy_i, hand-derived for the flatness form; checked
    with a tabulated target and non-uniform weights too."""
    n = 6
    y = np.array([2600.0, 2100.0, 1500.0, 900.0, 1400.0, 2500.0])
    target = np.linspace(800.0, 1300.0, n) if weighted else np.full(n, 1000.0)
    w = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.3]) if weighted else None
    obj = ProfileObjective(target=target, window_mask=np.ones(n, bool),
                           alpha=100.0, time_weights=w, flatness_beta=beta)
    analytic = np.array([obj.sigma_j(y, np.eye(n)[i]) for i in range(n)])
    h, numeric = 1e-4, []
    for i in range(n):
        up, dn = y.copy(), y.copy()
        up[i] += h
        dn[i] -= h
        numeric.append(abs((obj.j_observed(up) - obj.j_observed(dn)) / (2 * h)))
    assert analytic == pytest.approx(np.array(numeric), rel=1e-5)
