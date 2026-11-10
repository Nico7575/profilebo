"""Composite objective J(x) evaluated on GP coefficient samples.

The GP posterior lives in coefficient space; J is a deterministic map

    c -> y(t) = mean(t) + Phi(t) c
      -> J = mean_i w_i (y_i - target_i)^2,  w_i = 1 if y_i >= target_i else alpha

so the acquisition must be a Monte-Carlo composite (Astudillo and Frazier
2019): posterior samples of c are pushed through this module and qLogEI
is taken over the resulting J samples.

J is the weighted mean squared deviation over the objective window, in the
squared units of y.

J is implemented once, in torch. A measured trajectory goes through the same
code (ProfileObjective without a basis, or j_of_measured), so the J that
ranks evaluated designs is exactly the J the acquisition optimizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from profilebo.surrogate.basis import TrajectoryBasis

DTYPE = torch.double


def target_on_grid(target: float | str | Path, t: np.ndarray, window=None) -> np.ndarray:
    """The target profile on the grid t.

    target: a number (constant target) or the path to a two-column text
    file (t, target), interpolated linearly onto t. Interpolation does not
    extrapolate, so the file must cover the objective window: outside its
    range np.interp would silently hold the end values.
    """
    t = np.asarray(t, dtype=float)
    if isinstance(target, (str, Path)):
        tab = np.loadtxt(target, ndmin=2)
        if tab.shape[1] != 2:
            raise ValueError(f"target file {target} must have two columns (t, target)")
        lo, hi = (t[0], t[-1]) if window is None else window
        inside = t[(t >= lo) & (t <= hi)]
        if inside.size and (inside[0] < tab[0, 0] or inside[-1] > tab[-1, 0]):
            raise ValueError(
                f"target file {target} spans [{tab[0, 0]}, {tab[-1, 0]}] but the "
                f"objective window needs [{inside[0]}, {inside[-1]}]")
        return np.interp(t, tab[:, 0], tab[:, 1])
    return np.full_like(t, float(target))


def trapezoid_weights(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalised trapezoidal weights for the in-window points.

    J is a mean over grid POINTS, but a grid is often front-loaded (a
    depletion grid such as 0 1 5 30 180 365 730 1095 1460 1825 2190 days), so
    early points are weighted far beyond the interval they represent: on the
    [5, 2100] window, the points 5/30/180 cover 13 % of the window yet carry
    37.5 % of a uniform J. These weights instead give each point the share of
    the window it represents (midpoint to midpoint, clipped to the window), so
    J becomes an average over t. They sum to 1, so a weighted SUM has the same
    units and comparable magnitude to the uniform MEAN it replaces.

    Returns an array over the in-window points only, in grid order.
    """
    t = np.asarray(t, dtype=float)
    d = t[(t >= lo) & (t <= hi)]
    if d.size == 0:
        raise ValueError("objective window excludes every point of the t grid")
    if d.size == 1:
        return np.ones(1)
    edges = np.concatenate([[lo], (d[1:] + d[:-1]) / 2.0, [hi]])
    w = np.diff(edges)
    return w / w.sum()


class ProfileObjective:
    def __init__(
        self,
        target: np.ndarray,       # (n_t,)
        window_mask: np.ndarray,  # (n_t,) bool, J evaluated where True
        alpha: float,
        time_weights: np.ndarray | None = None,   # over in-window points; sums to 1
        flatness_beta: float | None = None,       # None = distance-to-target J
        basis: TrajectoryBasis | None = None,     # needed only for coefficients
    ):
        as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=DTYPE)
        self.target = as_t(target)
        self.mask = torch.as_tensor(np.asarray(window_mask, dtype=bool))
        self.alpha = float(alpha)
        if self.target.shape != self.mask.shape:
            raise ValueError("target and window_mask must have the same length")
        if not self.mask.any():
            raise ValueError("objective window excludes every point of the t grid")
        # None = uniform (a plain mean over points).
        self.time_weights = None if time_weights is None else as_t(time_weights)
        if self.time_weights is not None and \
                self.time_weights.numel() != int(self.mask.sum()):
            raise ValueError(
                f"time_weights has {self.time_weights.numel()} entries but the "
                f"window holds {int(self.mask.sum())} points")
        self.flatness_beta = None if flatness_beta is None else float(flatness_beta)
        if self.flatness_beta is not None and self.flatness_beta < 0:
            raise ValueError("flatness_beta must be >= 0")

        self.phi = self.basis_mean = None
        if basis is not None:
            if basis.phi.shape[0] != self.target.numel():
                raise ValueError("basis grid and target have different lengths")
            self.phi = as_t(basis.phi)
            self.basis_mean = as_t(basis.mean)

    # --- trajectory reconstruction ------------------------------------
    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """(..., n_coeff) -> (..., n_t): the trajectory the coefficients encode."""
        if self.phi is None:
            raise RuntimeError("this objective has no basis; build it with one "
                               "to evaluate coefficients")
        return coeffs @ self.phi.T + self.basis_mean

    # --- J ----------------------------------------------------------------
    def _point_weights(self, n: int) -> torch.Tensor:
        """Per-point weights summing to 1 (uniform unless time_weights is set)."""
        if self.time_weights is None:
            return torch.full((n,), 1.0 / n, dtype=DTYPE)
        return self.time_weights

    def j_from_y(self, y: torch.Tensor) -> torch.Tensor:
        """(..., n_t) -> (...,): the objective, in squared units of y.

        Two forms. The default is the distance-to-target J, a weighted mean
        squared deviation with the asymmetric alpha on undershoot.

        With flatness_beta set, J instead splits the SAME quantity into its two
        parts. For symmetric weights and a constant target the identity is
        exact:

            sum_i w_i (y_i - target)^2  ==  Var_w(y) + (mean_w(y) - target)^2

        so the objective becomes

            J = Var_w(y) + beta * (mean_w(y) - mean_w(target))^2
                         + alpha * sum_i w_i * max(0, target_i - y_i)^2

        mean_w uses the same point weights for y and for the target, so the
        offset compares like with like when the target is tabulated and the
        weights are not uniform. Var_w is the flatness of y itself, also for a
        tabulated target.

        For a constant target, beta = 1 with alpha = 0 reproduces the symmetric
        original exactly;
        beta < 1 buys flatness at the cost of sitting further from the target;
        beta = 0 optimises flatness alone, with only the alpha term holding the
        trajectory above the target. alpha keeps its job either way: undershoot
        is the thing that must not happen.
        """
        y = y[..., self.mask]
        target = self.target[self.mask]
        if self.flatness_beta is None:
            dev = y - target
            w = torch.where(dev >= 0, 1.0, self.alpha)
            if self.time_weights is None:
                return (w * dev**2).mean(dim=-1)
            return (self.time_weights * w * dev**2).sum(dim=-1)

        wt = self._point_weights(y.shape[-1])
        mean_w = (wt * y).sum(dim=-1, keepdim=True)
        var_w = (wt * (y - mean_w) ** 2).sum(dim=-1)
        offset = (mean_w.squeeze(-1) - (wt * target).sum()) ** 2
        under = (wt * torch.clamp(target - y, min=0.0) ** 2).sum(dim=-1)
        return var_w + self.flatness_beta * offset + self.alpha * under

    def j_from_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.j_from_y(self.reconstruct(coeffs))

    def sigma_j(self, y: np.ndarray, y_std: np.ndarray) -> float:
        """First-order propagation of the trajectory's statistical uncertainty
        into J. Needs no replicates (every run carries its own sigma), and it
        automatically reflects the alpha amplification, since dJ/dy_i carries
        the weight w_i."""
        y = torch.as_tensor(np.asarray(y, float), dtype=DTYPE)
        dev = (y - self.target)[self.mask]
        sig = torch.as_tensor(np.asarray(y_std, float), dtype=DTYPE)[self.mask]
        wt = self._point_weights(int(self.mask.sum()))
        if self.flatness_beta is None:
            w = torch.where(dev >= 0, 1.0, self.alpha)
            grad = 2.0 * w * dev * wt
        else:
            # J = Var_w + beta*(mean_w(y) - mean_w(target))^2
            #     + alpha*sum w_i max(0, target_i - y_i)^2
            # dVar/dy_i = 2 w_i (y_i - mean_w)  (the mean's own term cancels,
            # since sum_j w_j (y_j - mean_w) = 0).
            target = self.target[self.mask]
            y_w = dev + target
            mean_w = (wt * y_w).sum()
            grad = 2.0 * wt * ((y_w - mean_w)
                               + self.flatness_beta * (mean_w - (wt * target).sum())
                               - self.alpha * torch.clamp(target - y_w, min=0.0))
        return float(torch.sqrt(torch.sum((grad * sig) ** 2)))

    def j_observed(self, y: np.ndarray) -> float:
        """J of a simulated trajectory on this objective's grid.
        Used for best_f and for reporting the incumbent."""
        return float(self.j_from_y(torch.as_tensor(np.asarray(y, dtype=float), dtype=DTYPE)))

    # --- botorch MC-objective interface -----------------------------------
    def __call__(self, samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        """samples: (sample_shape, batch, q, n_coeff) posterior draws of the
        coefficient vector -> negated J (botorch maximizes)."""
        return -self.j_from_coeffs(samples)


def build_objective(
    objective_cfg,
    t: np.ndarray | None = None,
    basis: TrajectoryBasis | None = None,
) -> ProfileObjective:
    """Assemble the objective from config on the grid t, or on the basis' grid.

    With a basis, the objective also evaluates GP coefficient samples (the
    acquisition); without one it scores measured trajectories only.
    """
    if basis is not None:
        if t is not None and not np.array_equal(np.asarray(t, float), basis.t):
            raise ValueError("t differs from the basis grid")
        t = basis.t
    if t is None:
        raise ValueError("build_objective needs a grid t or a basis")
    t = np.asarray(t, dtype=float)

    lo, hi = objective_cfg.window
    return ProfileObjective(
        target=target_on_grid(objective_cfg.target, t, (lo, hi)),
        window_mask=(t >= lo) & (t <= hi),
        alpha=objective_cfg.alpha,
        time_weights=trapezoid_weights(t, lo, hi) if objective_cfg.time_weighted else None,
        flatness_beta=objective_cfg.flatness_beta,
        basis=basis,
    )


def j_of_measured(y: np.ndarray, t: np.ndarray, objective_cfg) -> float:
    """J of a single measured trajectory on its own grid, with no basis
    involved: for single-case runs and reporting before a surrogate exists."""
    return build_objective(objective_cfg, t).j_observed(y)
