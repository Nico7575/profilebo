"""Active GP+BO loop: DoE -> [fit -> acquire -> evaluate] -> x*.

The loop is checkpointed to disk after every evaluation (a loop of
hours-long simulations must be killable and resumable): 'history.npz' holds
all evaluated (x, trajectory) pairs, 'basis.npz' the frozen trajectory basis.
Restarting with the same run_dir resumes exactly where it stopped, including
half-finished evaluations, since evaluators are idempotent per case
directory (see evaluators/base.py).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from profilebo.config import Config
from profilebo.evaluators.base import Evaluation, Evaluator
from profilebo.optimization.acquisition import propose_candidate, propose_distinguishable
from profilebo.optimization.objective import ProfileObjective, build_objective
from profilebo.rng import derive_seed
from profilebo.surrogate.basis import TrajectoryBasis, select_pod_basis
from profilebo.surrogate.gp_model import TrajectoryGP

log = logging.getLogger(__name__)


class BOLoop:
    def __init__(
        self,
        config: Config,
        evaluator: Evaluator,
        run_dir: str | Path | None = None,
    ):
        self.cfg = config
        self.evaluator = evaluator
        self.run_dir = Path(run_dir if run_dir is not None else config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.t: np.ndarray | None = None         # shared trajectory grid
        self.X: list[np.ndarray] = []            # evaluated x (rounded, physical)
        self.y: list[np.ndarray] = []            # simulated trajectories
        self.y_std: list[np.ndarray] = []
        self.tags: list[str] = []
        self.basis: TrajectoryBasis | None = None
        self._lock = threading.Lock()   # guards history during parallel DoE
        self._tol_streak = 0            # consecutive sub-resolution proposals
        self._sigma_hist: list[float] = []   # for median-smoothing sigma_J
        self._step_state_restored = False    # see _restore_step_state
        self._stop_reason: str | None = None

        self._load_checkpoint()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @property
    def _history_file(self) -> Path:
        return self.run_dir / "history.npz"

    @property
    def _basis_file(self) -> Path:
        return self.run_dir / "basis.npz"

    def _load_checkpoint(self) -> None:
        if self._history_file.exists():
            d = np.load(self._history_file, allow_pickle=False)
            # X columns are only meaningful against the names they were saved
            # with: a config edited mid-campaign (a batch added, a radius fixed)
            # would otherwise pair old values with new parameters.
            names = [str(n) for n in d["names"]]
            if names != self.cfg.parameter_names:
                raise ValueError(
                    f"{self._history_file} was written for parameters {names}, but the "
                    f"config now defines {self.cfg.parameter_names}; use a new run_dir")
            self.t = d["t"]
            self.X = list(d["X"])
            self.y = list(d["y"])
            self.y_std = list(d["y_std"])
            self.tags = [str(t) for t in d["tags"]]
            log.info("Resumed %d evaluations from %s", len(self.X), self._history_file)
        if self._basis_file.exists():
            self.basis = TrajectoryBasis.load(self._basis_file)

    def _checkpoint(self) -> None:
        np.savez(
            self._history_file,
            t=self.t,
            X=np.array(self.X),
            y=np.array(self.y),
            y_std=np.array(self.y_std),
            tags=np.array(self.tags),
            names=np.array(self.cfg.parameter_names),
        )

    # ------------------------------------------------------------------
    # evaluation bookkeeping
    # ------------------------------------------------------------------
    @property
    def n_evals(self) -> int:
        return len(self.X)

    def _simulated_x(self, ev: Evaluation, proposed: np.ndarray, tag: str) -> np.ndarray:
        """The design the returned trajectory actually belongs to.

        An evaluator may adopt a case prepared for an earlier proposal (after a
        restart) instead of discarding its simulation; it then reports that
        design in ev.x, and the archive must record it, never the new proposal,
        or the trajectory would be paired with the wrong x.
        """
        if ev.x is None:
            return proposed
        missing = set(self.cfg.parameter_names) - set(ev.x)
        if missing:
            raise ValueError(f"evaluator reported a design for {tag} without {sorted(missing)}")
        found = np.array([ev.x[n] for n in self.cfg.parameter_names], dtype=float)
        if not np.allclose(found, proposed, rtol=0, atol=1e-6):
            log.warning("%s was simulated at x=%s, not the current proposal %s; "
                        "recording the simulated x", tag,
                        np.round(found, 5), np.round(proposed, 5))
        return found

    def _evaluate(self, x: np.ndarray, tag: str) -> None:
        x = self.cfg.round_integers(x)
        case_dir = self.run_dir / "cases" / tag
        ev = self.evaluator.evaluate(self.cfg.as_point(x), case_dir)
        x = self._simulated_x(ev, x, tag)

        with self._lock:
            if self.t is None:
                self.t = np.asarray(ev.t, dtype=float)
            elif not np.allclose(self.t, ev.t):
                raise ValueError(
                    f"Case {tag} returned a different t grid; all evaluations must "
                    "share one grid so trajectories live on one basis."
                )

            self.X.append(np.asarray(x, dtype=float))
            self.y.append(np.asarray(ev.y, dtype=float))
            self.y_std.append(np.asarray(ev.std, dtype=float))
            self.tags.append(tag)
            self._checkpoint()

    # ------------------------------------------------------------------
    # phase 1: initial design
    # ------------------------------------------------------------------
    def initial_design(self) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=len(self.cfg.parameters), seed=self.cfg.bo.seed)
        unit = sampler.random(self.cfg.bo.n_init)
        lb, ub = self.cfg.bounds_array
        return self.cfg.round_integers(qmc.scale(unit, lb, ub))

    def ensure_initial(self) -> None:
        """Evaluate the LHS design; DoE points are independent, so up to
        bo.doe_parallel evaluations run concurrently. The BO phase stays
        sequential (each acquisition depends on all previous results)."""
        design = self.initial_design()
        pending = [(i, x) for i, x in enumerate(design)
                   if f"doe_{i:03d}" not in self.tags]
        if not pending:
            return

        workers = max(1, int(self.cfg.bo.doe_parallel))
        log.info("DoE: %d/%d points to evaluate, %d concurrent",
                 len(pending), len(design), workers)
        if workers == 1:
            for i, x in pending:
                self._evaluate(x, f"doe_{i:03d}")
            return

        # Fail fast: if one evaluation raises, cancel everything still
        # queued instead of churning through the remaining design points
        # (running evaluations finish; their results are checkpointed).
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(self._evaluate, x, f"doe_{i:03d}"): i
                       for i, x in pending}
            for f in as_completed(futures):
                f.result()      # re-raise the first evaluation failure
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------
    # surrogate assembly
    # ------------------------------------------------------------------
    def _training_order(self) -> list[int]:
        """Archive indices in the order the surrogate is built from.

        The archive keeps completion order, which a parallel DoE makes
        timing-dependent; in reproducible mode the basis and GPs are built in
        tag order instead (DoE by index, then BO), so floating-point
        reductions cannot depend on which evaluation finished first.
        """
        idx = list(range(len(self.tags)))
        if not self.cfg.bo.reproducible:
            return idx
        def key(i):
            phase, _, num = self.tags[i].partition("_")
            return (0 if phase == "doe" else 1, int(num) if num.isdigit() else 0, self.tags[i])
        return sorted(idx, key=key)

    def _window_mask(self) -> np.ndarray:
        lo, hi = self.cfg.objective.window
        return (self.t >= lo) & (self.t <= hi)

    def _ensure_basis(self) -> TrajectoryBasis:
        """Basis is built once from the DoE snapshots and frozen."""
        if self.basis is None:
            order = self._training_order()
            self.basis = select_pod_basis(
                self.t,
                np.array(self.y)[order],
                np.array(self.y_std)[order],
                max_modes=self.cfg.basis.max_modes,
                target_ratio=self.cfg.basis.target_ratio,
                mask=self._window_mask(),
                n_coeff=self.cfg.basis.n_coeff,
            )
            self.basis.save(self._basis_file)
        return self.basis

    def fit_surrogate(self) -> tuple[TrajectoryGP, ProfileObjective, float]:
        basis = self._ensure_basis()
        order = self._training_order()
        C, C_var = basis.fit_many(np.array(self.y)[order], np.array(self.y_std)[order])

        gp = TrajectoryGP(self.cfg.bounds_array).fit(np.array(self.X)[order], C, C_var)
        objective = build_objective(self.cfg.objective, basis=basis)
        # Incumbent from *simulated* trajectories (maximization convention).
        best_f = max(-objective.j_observed(y) for y in self.y)
        return gp, objective, best_f

    # ------------------------------------------------------------------
    # design-resolution floor
    # ------------------------------------------------------------------
    def _predict_J(self, gp, objective, x: np.ndarray) -> float:
        Xu = torch.as_tensor(gp.to_unit(np.atleast_2d(x)), dtype=torch.double)
        with torch.no_grad():
            s = gp.model.posterior(Xu).rsample(torch.Size([256]))
        return float(np.median(objective.j_from_coeffs(s.squeeze(1)).numpy()))

    def _resolution(self, gp, objective, x0: np.ndarray, sigma_J: float) -> np.ndarray | None:
        """Per-parameter resolution: max(manufacturing tolerance, statistical
        resolution).

        The statistical part is the smallest step that moves predicted J by one
        sigma_J, found by a geometric scan, not a finite difference. A finite
        difference with a fixed step reaches into the alpha wall and reports a
        far steeper slope than the local one (measured: 0.05 vs 2.1 particles
        for N1), which would collapse the floor to nothing.
        """
        cfg = self.cfg.bo
        if not cfg.resolution_floor:
            return None
        manuf = np.array([float(cfg.design_tolerance.get(n, 0.0))
                          for n in self.cfg.parameter_names])
        if not cfg.statistical_floor:
            return manuf

        lb, ub = self.cfg.bounds_array
        d = len(self.cfg.parameters)
        J0 = self._predict_J(gp, objective, x0)
        cap = 0.05 * (ub - lb)          # a flat direction must not block everything
        stat = cap.copy()
        for k in range(d):
            span = ub[k] - lb[k]
            for frac in np.geomspace(0.002, 0.05, 12):
                delta = frac * span
                plus = np.clip(x0 + np.eye(d)[k] * delta, lb, ub)
                minus = np.clip(x0 - np.eye(d)[k] * delta, lb, ub)
                dJ = max(abs(self._predict_J(gp, objective, plus) - J0),
                         abs(self._predict_J(gp, objective, minus) - J0))
                if dJ >= sigma_J:
                    stat[k] = delta
                    break
        return np.maximum(manuf, stat)

    def _check_stop(self, streak: int, ei_out: float, sigma_J: float) -> str | None:
        """Practical-convergence test, applied only to sub-resolution steps.

        Requires PERSISTENCE: one sub-resolution proposal is not evidence of
        convergence, because both the acquisition argmax and sigma_J fluctuate
        step to step (sigma_J has been seen to move 8x between consecutive
        steps, which produced a false stop at streak 1). The noise test is a
        guard applied once the streak is reached, never an independent trigger.
        """
        k = self.cfg.bo.tolerance_streak
        f = self.cfg.bo.noise_stop_factor
        if k is None or streak < k:
            return None
        if f is not None and ei_out >= f * sigma_J:
            return None          # still worth exploring a distinguishable design
        reason = f"{streak} consecutive proposals below design resolution"
        if f is not None:
            reason += (f", with expected improvement {ei_out:.3g} below "
                       f"{f:g} x sigma_J ({f * sigma_J:.3g})")
        return reason

    def _indistinguishable(self, x: np.ndarray, tol: np.ndarray) -> bool:
        return bool(np.any(np.all(np.abs(np.array(self.X) - x) <= tol, axis=1)))

    def _restore_step_state(self, objective: ProfileObjective) -> None:
        """Rebuild the per-step state of the BO steps already in the archive.

        sigma_J smoothing and the sub-resolution streak are not checkpointed
        separately, so a restarted loop must not resume with an empty history:
        it could redirect, or stop, differently from an uninterrupted one. Both
        are functions of the archive: sigma_J of step p is that of the
        incumbent among the evaluations before it, and the streak is the
        trailing run of redirected steps in iterations.jsonl.
        """
        self._step_state_restored = True
        J = [objective.j_observed(y) for y in self.y]
        self._sigma_hist = []
        for p, t in enumerate(self.tags):
            if t.startswith("bo_") and p > 0:
                inc = int(np.argmin(J[:p]))
                self._sigma_hist.append(objective.sigma_j(self.y[inc], self.y_std[inc]))
        f = self.run_dir / "iterations.jsonl"
        if f.exists():
            streak = 0
            for line in f.read_text().splitlines():
                streak = streak + 1 if json.loads(line).get("redirected") else 0
            self._tol_streak = streak
        if self._sigma_hist or self._tol_streak:
            log.info("Restored step state: %d sigma_J values, streak %d",
                     len(self._sigma_hist), self._tol_streak)

    def _seed_step(self, tag: str) -> None:
        """In reproducible mode, seed the global RNGs for the step 'tag'.

        One seed per step, keyed on its tag: covers the GP-fit retries, the MC
        sampler, the multi-start initial conditions and the resolution scan,
        which all draw from the global torch/numpy RNG. Subclasses that
        override step() must call this first.
        """
        if self.cfg.bo.reproducible:
            s = derive_seed(self.cfg.bo.seed, tag, "acquisition")
            torch.manual_seed(s)
            np.random.seed(s)

    def _log_iteration(self, record: dict) -> None:
        with open(self.run_dir / "iterations.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # phase 2: active loop
    # ------------------------------------------------------------------
    def step(self) -> dict:
        tag = f"bo_{self.n_evals:03d}"
        self._seed_step(tag)
        gp, objective, best_f = self.fit_surrogate()
        if not self._step_state_restored:
            self._restore_step_state(objective)
        cand_unit, acq_value = propose_candidate(
            gp.model,
            objective,
            best_f,
            gp.unit_bounds,
            num_restarts=self.cfg.bo.num_restarts,
            raw_samples=self.cfg.bo.raw_samples,
            mc_samples=self.cfg.bo.mc_samples,
        )
        x = self.cfg.round_integers(gp.from_unit(cand_unit.numpy()[0]))

        # --- design-resolution floor -------------------------------------
        J_obs = [objective.j_observed(y) for y in self.y]
        inc = int(np.argmin(J_obs))
        # sigma_J jumps several-fold whenever the incumbent changes to one whose
        # trajectory dips below target (alpha amplification), so smooth it
        # before using it as a threshold: an unsmoothed value caused a false stop.
        self._sigma_hist.append(objective.sigma_j(self.y[inc], self.y_std[inc]))
        sigma_J = float(np.median(self._sigma_hist[-5:]))
        tol = self._resolution(gp, objective, self.X[inc], sigma_J)
        redirected = False
        if tol is not None and self._indistinguishable(x, tol):
            lb, ub = self.cfg.bounds_array
            alt, alt_acq = propose_distinguishable(
                gp.model, objective, best_f,
                torch.as_tensor(gp.to_unit(np.array(self.X)), dtype=torch.double),
                torch.as_tensor(tol / (ub - lb), dtype=torch.double),
                mc_samples=self.cfg.bo.mc_samples,
            )
            self._tol_streak += 1
            k = self.cfg.bo.tolerance_streak
            f = self.cfg.bo.noise_stop_factor
            if alt is None:
                self._stop_reason = "no distinguishable candidate remains"
            else:
                x = self.cfg.round_integers(gp.from_unit(alt.numpy()[0]))
                acq_value, redirected = alt_acq, True
            log.info("Proposal is below design resolution (streak %d/%s); "
                     "redirected, EI_out=%.3g vs %s",
                     self._tol_streak, k, np.exp(acq_value),
                     f"{f:g}*sigma_J={f * sigma_J:.3g}" if f is not None else "n/a")

            if self._stop_reason is None:
                self._stop_reason = self._check_stop(
                    self._tol_streak, float(np.exp(acq_value)), sigma_J)
        else:
            self._tol_streak = 0

        log.info("Iteration %s: acq=%.4g%s, sigma_J=%.3g, x=%s", tag, acq_value,
                 " (redirected)" if redirected else "", sigma_J, self.cfg.as_point(x))

        self._evaluate(x, tag)
        j_new = objective.j_observed(self.y[-1])
        record = {
            "tag": tag,
            "x": self.cfg.as_point(self.X[-1]),   # the simulated x
            "acq_value": acq_value,
            "J_observed": j_new,
            "J_best": min(-best_f, j_new),
            "sigma_J": sigma_J,
            "redirected": redirected,
            "stop_reason": self._stop_reason,
        }
        self._log_iteration(record)
        return record

    def run(self) -> tuple[dict[str, float], float]:
        """DoE + active loop until budget, practical convergence or the
        acquisition threshold."""
        self.ensure_initial()
        while self.n_evals < self.cfg.bo.budget:
            record = self.step()
            if self._stop_reason:
                log.info("Practical convergence: %s", self._stop_reason)
                break
            thr = self.cfg.bo.acq_threshold
            if thr is not None and record["acq_value"] < thr:
                log.info("Acquisition %.4g below threshold %.4g; stopping.",
                         record["acq_value"], thr)
                break
        return self.best()

    def best(self) -> tuple[dict[str, float], float]:
        """Incumbent (x, J) over all simulated trajectories."""
        objective = build_objective(self.cfg.objective, t=self.t)
        J = np.array([objective.j_observed(y) for y in self.y])
        i = int(np.argmin(J))
        return self.cfg.as_point(self.X[i]), float(J[i])
