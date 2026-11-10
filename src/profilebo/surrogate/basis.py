"""Trajectory reduction: y(t) -> n_coeff coefficients.

A fixed basis Phi (n_t x n_coeff) is shared across all design points.
Coefficients are obtained by weighted least squares with weights 1/sigma^2
from the simulator's statistical uncertainties; the WLS covariance
(Phi^T W Phi)^-1 supplies the per-coefficient observation noise for the
fixed-noise GPs.

The basis is built by Proper Orthogonal Decomposition (POD) of the
simulated trajectories. Its rank is chosen from the data (select_pod_basis):
the fewest modes whose truncation error on the snapshots is down to the
statistical noise of the simulations, capped by a maximum.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class TrajectoryBasis:
    """Fixed linear basis: y(t) ~ mean(t) + Phi(t) c."""

    def __init__(self, t: np.ndarray, phi: np.ndarray, mean: np.ndarray | None = None):
        self.t = np.asarray(t, dtype=float)
        self.phi = np.asarray(phi, dtype=float)              # (n_t, n_coeff)
        self.mean = np.zeros(len(self.t)) if mean is None else np.asarray(mean, dtype=float)
        if self.phi.shape[0] != len(self.t):
            raise ValueError("phi rows must match the length of the t grid")

    @property
    def n_coeff(self) -> int:
        return self.phi.shape[1]

    # --- coefficient fitting -----------------------------------------
    def fit_coefficients(
        self, y: np.ndarray, std: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """WLS fit of one trajectory. Returns (c, var_c) where var_c is
        the diagonal of the WLS covariance (observation noise for the GP)."""
        y = np.asarray(y, dtype=float) - self.mean
        if std is None:
            std = np.ones_like(y)
        w = 1.0 / np.clip(np.asarray(std, dtype=float), 1e-6, None) ** 2

        A = self.phi * np.sqrt(w)[:, None]
        b = y * np.sqrt(w)
        gram = A.T @ A
        c = np.linalg.solve(gram, A.T @ b)
        var_c = np.diag(np.linalg.inv(gram))
        return c, var_c

    def fit_many(
        self, y: np.ndarray, std: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a stack of trajectories (n_samples, n_t). Returns
        (C, C_var), both (n_samples, n_coeff)."""
        y = np.atleast_2d(y)
        std = np.ones_like(y) if std is None else np.atleast_2d(std)
        out = [self.fit_coefficients(yi, si) for yi, si in zip(y, std)]
        C = np.array([c for c, _ in out])
        C_var = np.array([v for _, v in out])
        return C, C_var

    # --- reconstruction ----------------------------------------------
    def reconstruct(self, c: np.ndarray) -> np.ndarray:
        """(..., n_coeff) -> (..., n_t)."""
        return np.asarray(c) @ self.phi.T + self.mean

    def reconstruction_error(self, y: np.ndarray, std: np.ndarray | None = None) -> float:
        """Max abs reconstruction error over a stack of trajectories, in the
        units of y; compared against basis.max_recon_error."""
        y = np.atleast_2d(y)
        C, _ = self.fit_many(y, std)
        return float(np.max(np.abs(self.reconstruct(C) - y)))

    # --- persistence ---------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=self.t, phi=self.phi, mean=self.mean)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryBasis":
        d = np.load(path)
        return cls(t=d["t"], phi=d["phi"], mean=d["mean"])


class PODBasis(TrajectoryBasis):
    """POD/PCA modes of a snapshot set of trajectories."""

    @classmethod
    def build(cls, t: np.ndarray, trajectories: np.ndarray, n_coeff: int) -> "PODBasis":
        snaps = np.atleast_2d(trajectories)          # (n_samples, n_t)
        if snaps.shape[0] < n_coeff:
            raise ValueError(
                f"Need at least {n_coeff} snapshots to build {n_coeff} POD modes "
                f"(got {snaps.shape[0]})"
            )
        mean = snaps.mean(axis=0)
        _, s, vt = np.linalg.svd(snaps - mean, full_matrices=False)
        basis = cls(t=t, phi=vt[:n_coeff].T, mean=mean)
        basis.singular_values = s
        return basis


def noise_ratio(basis: TrajectoryBasis, y: np.ndarray, std: np.ndarray,
                mask: np.ndarray | None = None) -> float:
    """RMS of the reconstruction residual in units of the statistical sigma,
    over all trajectories and the points selected by mask.

    1 means the truncation error is as large as the simulation noise: more
    modes would start fitting noise rather than shape.
    """
    y, std = np.atleast_2d(y), np.atleast_2d(std)
    C, _ = basis.fit_many(y, std)
    z = (basis.reconstruct(C) - y) / np.clip(std, 1e-6, None)
    if mask is not None:
        z = z[:, np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(z**2)))


def select_pod_basis(
    t: np.ndarray,
    y: np.ndarray,
    std: np.ndarray,
    max_modes: int,
    target_ratio: float = 1.0,
    mask: np.ndarray | None = None,
    n_coeff: int | None = None,
) -> PODBasis:
    """POD basis with the fewest modes whose noise_ratio is <= target_ratio.

    The search stops at max_modes, and at n_snapshots - 1: with the mean
    removed, the snapshots span at most that many directions, and a basis of
    that rank reproduces them exactly whatever their noise. If no rank in
    range clears the target, the largest one is used and a warning is logged.

    n_coeff fixes the rank instead (no search); the ratio is still logged.
    mask restricts the check to the points the objective sees (its window).
    """
    y = np.atleast_2d(y)
    limit = min(max_modes, y.shape[0] - 1)
    if limit < 1:
        raise ValueError(f"need at least 2 snapshots to build a POD basis (got {y.shape[0]})")

    if n_coeff is not None:
        basis = PODBasis.build(t, y, n_coeff)
        log.info("POD basis: %d modes (fixed), noise ratio %.2f",
                 n_coeff, noise_ratio(basis, y, std, mask))
        return basis

    ratios = []
    for n in range(1, limit + 1):
        basis = PODBasis.build(t, y, n)
        ratios.append(noise_ratio(basis, y, std, mask))
        if ratios[-1] <= target_ratio:
            break
    log.info("POD rank scan (modes: noise ratio): %s",
             ", ".join(f"{k}: {r:.2f}" for k, r in enumerate(ratios, start=1)))
    if ratios[-1] > target_ratio:
        log.warning(
            "No POD rank up to %d brings the truncation error down to %.2f sigma "
            "(best: %.2f at %d modes). Using %d modes; consider a larger "
            "basis.max_modes or more initial designs.",
            limit, target_ratio, ratios[-1], limit, limit)
    else:
        log.info("POD basis: %d modes, noise ratio %.2f <= %.2f",
                 basis.n_coeff, ratios[-1], target_ratio)
    return basis
