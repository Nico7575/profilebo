"""The report of a campaign: a text summary and figures, from its run directory.

It only reads the campaign (history.npz and iterations.jsonl) and writes into
a folder of its own, so it can run at any time, also while the campaign is
running. J is computed with the objective of the config, so it is the same J
the optimizer ranks the designs by.

Figures:

    trajectories   the best design against the target, the others faint
    convergence    J of every evaluation and the best so far
    acquisition    expected improvement of every optimization step, against
                   the statistical resolution of J (the stopping signal)
    parameters     every evaluated design, parameter by parameter, with bounds
    contributions  where the best design's J comes from, point by point
    ranking        the best evaluations with their statistical error on J
    convergence.gif (optional) one frame per evaluation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from profilebo.config import Config
from profilebo.optimization.objective import build_objective, trapezoid_weights

FIGURES = ("trajectories", "convergence", "acquisition", "parameters", "contributions",
           "ranking")

# Colours, for a light background.
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
EXTRA_COLORS = (GREEN, MAGENTA, YELLOW, INK)
STYLE = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": BASE, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False,
    "legend.fontsize": 9,
}

# A parameter closer than this fraction of its range to a bound is "at" it.
AT_BOUND = 0.01


@dataclass
class Campaign:
    """The archive of a campaign, with J and its statistical error per evaluation."""

    t: np.ndarray            # (n_t,) grid
    X: np.ndarray            # (n, d) evaluated designs, in order of completion
    y: np.ndarray            # (n, n_t) trajectories
    y_std: np.ndarray        # (n, n_t) their statistical 1-sigma
    tags: list[str]
    names: list[str]
    J: np.ndarray            # (n,)
    sigma_J: np.ndarray      # (n,) first-order statistical error of J
    target: np.ndarray       # (n_t,) target on the grid
    window: np.ndarray       # (n_t,) bool, points where J is evaluated
    iterations: list[dict]   # one record per optimization step

    @property
    def initial(self) -> np.ndarray:
        """True for the evaluations of the initial design."""
        return np.array([t.startswith("doe_") for t in self.tags])

    @property
    def best(self) -> int:
        return int(np.argmin(self.J))

    @property
    def scalar(self) -> bool:
        """True for a campaign of the scalar-objective baseline."""
        return any(r.get("surrogate") == "scalar_logJ" for r in self.iterations)

    def index(self, tag: str) -> int:
        if tag not in self.tags:
            raise ValueError(f"no evaluation {tag!r} in this campaign; it has "
                             f"{', '.join(self.tags)}")
        return self.tags.index(tag)


def load_campaign(cfg: Config) -> Campaign:
    history = cfg.run_dir / "history.npz"
    if not history.exists():
        raise FileNotFoundError(f"no results yet: {history} does not exist")
    d = np.load(history, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    if names != cfg.parameter_names:
        raise ValueError(f"{history} was written for parameters {names}, but the config "
                         f"defines {cfg.parameter_names}; is it the campaign's config?")
    t, y, y_std = d["t"], d["y"], d["y_std"]
    objective = build_objective(cfg.objective, t)

    iterations = []
    log_file = cfg.run_dir / "iterations.jsonl"
    if log_file.exists():
        iterations = [json.loads(line) for line in log_file.read_text().splitlines()
                      if line.strip()]
    return Campaign(
        t=t, X=d["X"], y=y, y_std=y_std, tags=[str(s) for s in d["tags"]], names=names,
        J=np.array([objective.j_observed(r) for r in y]),
        sigma_J=np.array([objective.sigma_j(r, s) for r, s in zip(y, y_std)]),
        target=objective.target.numpy(), window=objective.mask.numpy(),
        iterations=iterations,
    )


# ----------------------------------------------------------------------
# text summary
# ----------------------------------------------------------------------
def at_bounds(c: Campaign, cfg: Config, i: int) -> list[str]:
    """The parameters of evaluation i that sit on a bound of the design box."""
    lo, hi = cfg.bounds_array
    tol = AT_BOUND * (hi - lo)
    out = []
    for k, name in enumerate(c.names):
        if c.X[i, k] <= lo[k] + tol[k]:
            out.append(f"{name} (lower)")
        elif c.X[i, k] >= hi[k] - tol[k]:
            out.append(f"{name} (upper)")
    return out


def status(c: Campaign, cfg: Config) -> str:
    last = c.iterations[-1] if c.iterations else {}
    thr = cfg.bo.acq_threshold
    if last.get("stop_reason"):
        return f"finished, converged: {last['stop_reason']}"
    if len(c.J) >= cfg.bo.budget:
        return "finished: budget reached"
    if thr is not None and last and last["acq_value"] < thr:
        return f"finished: acquisition value below acq_threshold ({thr:g})"
    return "not finished (still running, or stopped before the end)"


def summary(c: Campaign, cfg: Config) -> str:
    n, n0 = len(c.J), int(c.initial.sum())
    b = c.best
    lines = [
        f"run directory  {cfg.run_dir}",
        f"surrogate      {'scalar baseline (GP on -log J)' if c.scalar else 'trajectory (POD + GP)'}",
        f"evaluated      {n} of a budget of {cfg.bo.budget}: {n0} initial design, "
        f"{n - n0} optimization steps",
        f"status         {status(c, cfg)}",
        f"best           {c.tags[b]}: J = {c.J[b]:.4g} +/- {c.sigma_J[b]:.2g}",
        "               " + ", ".join(f"{k} = {v:.6g}" for k, v in zip(c.names, c.X[b])),
    ]
    edge = at_bounds(c, cfg, b)
    if edge:
        lines.append(f"               on a bound: {', '.join(edge)}; the optimum may lie "
                     "outside the design box")
    if 0 < n0 < n:
        b0 = int(np.argmin(np.where(c.initial, c.J, np.inf)))
        lines.append(f"initial design best {c.tags[b0]}: J = {c.J[b0]:.4g}; the optimization "
                     f"improved it by a factor {c.J[b0] / c.J[b]:.3g}")
    ties = [c.tags[i] for i in np.argsort(c.J)[1:]
            if c.J[i] - c.J[b] < c.sigma_J[i] + c.sigma_J[b]]
    if ties:
        lines.append(f"statistical tie with the best (error bars overlap): {', '.join(ties[:8])}"
                     + (" ..." if len(ties) > 8 else ""))
    if c.iterations:
        r = c.iterations[-1]
        if c.scalar:
            lines.append(f"last step      {r['tag']}: log EI = {r['acq_value']:.4g}")
        else:
            lines.append(f"last step      {r['tag']}: EI = {np.exp(r['acq_value']):.3g}, "
                         f"sigma_J = {r['sigma_J']:.3g}"
                         + (", redirected" if r.get("redirected") else ""))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def _figure(**kw):
    from matplotlib.figure import Figure
    return Figure(**kw)


def _shade_outside_window(ax, c: Campaign) -> None:
    lo, hi = c.t[c.window][0], c.t[c.window][-1]
    if c.t[0] < lo:
        ax.axvspan(c.t[0], lo, color=GRID, alpha=0.45, lw=0, zorder=0)
    if c.t[-1] > hi:
        ax.axvspan(hi, c.t[-1], color=GRID, alpha=0.45, lw=0, zorder=0)


def _phase_colors(c: Campaign) -> list[str]:
    return [MUTED if s else BLUE for s in c.initial]


def fig_trajectories(c: Campaign, labels, show=(), prediction: dict | None = None):
    t_label, y_label, _ = labels
    fig = _figure(figsize=(8.5, 4.4), layout="constrained")
    ax = fig.subplots()
    _shade_outside_window(ax, c)
    b = c.best
    for i in range(len(c.J)):
        if i != b:
            ax.plot(c.t, c.y[i], color=MUTED, lw=0.8, alpha=0.35, zorder=1)
    ax.plot(c.t, c.target, color=INK, lw=1.2, ls="--", zorder=2, label="target")
    focus = [c.y[b], c.target]
    for k, tag in enumerate(show):
        i = c.index(tag)
        ax.plot(c.t, c.y[i], color=EXTRA_COLORS[k % len(EXTRA_COLORS)], lw=1.6,
                marker="o", ms=3, zorder=3, label=f"{tag}, J = {c.J[i]:.3g}")
        focus.append(c.y[i])
    if prediction is not None:
        ax.fill_between(c.t, prediction["y_p10"], prediction["y_p90"], color=MAGENTA,
                        alpha=0.18, lw=0, zorder=3)
        ax.plot(c.t, prediction["y"], color=MAGENTA, lw=1.8, ls="--", zorder=3,
                label=f"surrogate optimum, not simulated: predicted J = "
                      f"{float(prediction['J_p50']):.3g} (10-90 % band)")
        focus.append(prediction["y"])
    sig = 2 * c.y_std[b]
    ax.fill_between(c.t, c.y[b] - sig, c.y[b] + sig, color=BLUE, alpha=0.18, lw=0, zorder=4)
    ax.plot(c.t, c.y[b], color=BLUE, lw=2, marker="o", ms=4, zorder=5,
            label=rf"best: {c.tags[b]}, J = {c.J[b]:.3g} ($\pm 2\sigma$ band)")
    # Scale on the curves that matter, inside the window; the others, and
    # everything outside the window, may run off the frame.
    focus = np.array(focus)[:, c.window]
    lo, hi = focus.min(), focus.max()
    pad = 0.15 * max(hi - lo, 4 * np.max(c.y_std[b]), 1e-12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(c.t[0], c.t[-1])
    ax.set_xlabel(t_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Trajectories: the best of {len(c.J)} evaluations (others in grey; "
                 "shaded: outside the objective window)")
    ax.legend(loc="best")
    return fig


def fig_convergence(c: Campaign, labels):
    j_label = labels[2]
    n = np.arange(1, len(c.J) + 1)
    init = c.initial
    fig = _figure(figsize=(8.5, 3.8), layout="constrained")
    ax = fig.subplots()
    ax.scatter(n[init], c.J[init], s=26, color=MUTED, zorder=3, label="initial design")
    ax.scatter(n[~init], c.J[~init], s=30, color=BLUE, zorder=4, label="optimization")
    ax.step(n, np.minimum.accumulate(c.J), where="post", color=INK, lw=1.4, zorder=5,
            label="best so far")
    if init.any() and (~init).any():
        ax.axvline(init.sum() + 0.5, color=BASE, lw=0.8)
    if np.all(c.J > 0):
        ax.set_yscale("log")
    ax.set_xlabel("evaluation (order of completion)")
    ax.set_ylabel(j_label)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_title("Convergence: J of every evaluation")
    ax.legend()
    return fig


def fig_acquisition(c: Campaign, cfg: Config, labels):
    """Trajectory surrogate: EI against sigma_J, the comparison that the
    noise_stop_factor rule makes. Scalar baseline: its log EI, which is on
    -log J and has no common scale with sigma_J."""
    fig = _figure(figsize=(8.5, 3.4), layout="constrained")
    ax = fig.subplots()
    recs = c.iterations
    if not recs:
        ax.text(0.5, 0.5, "no optimization step yet", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        ax.set_axis_off()
        return fig
    tags = [r["tag"] for r in recs]
    xs = np.arange(len(recs))
    acq = np.array([r["acq_value"] for r in recs])
    if c.scalar:
        ax.plot(xs, acq, color=BLUE, lw=2, marker="o", ms=5, label="log EI")
        ax.set_ylabel("log EI (on -log J)")
        ax.set_title("Acquisition value of every optimization step")
    else:
        ei = np.exp(acq)
        sig = np.array([r["sigma_J"] for r in recs])
        redirected = np.array([bool(r.get("redirected")) for r in recs])
        ax.plot(xs, ei, color=BLUE, lw=2, zorder=3, label="expected improvement of J")
        ax.scatter(xs[~redirected], ei[~redirected], s=30, color=BLUE, zorder=4)
        if redirected.any():
            ax.scatter(xs[redirected], ei[redirected], s=40, facecolor="white",
                       edgecolor=BLUE, lw=1.6, zorder=4,
                       label="redirected (proposal within resolution)")
        ax.plot(xs, sig, color=MUTED, lw=1.4, marker="s", ms=3, zorder=2,
                label=r"$\sigma_J$ of the best (statistical resolution)")
        f = cfg.bo.noise_stop_factor
        if f is not None:
            ax.plot(xs, f * sig, color=MUTED, lw=1.0, ls="--", zorder=2,
                    label=rf"stop below {f:g} $\sigma_J$")
        if cfg.bo.acq_threshold is not None:
            ax.axhline(np.exp(cfg.bo.acq_threshold), color=YELLOW, lw=1.0, ls="--",
                       label="acq_threshold")
        ax.set_yscale("log")
        ax.set_ylabel(labels[2])
        ax.set_title("Acquisition: improvement the surrogate still expects")
    stop = recs[-1].get("stop_reason")
    if stop:
        ax.annotate(f"stopped: {stop}", (xs[-1], acq[-1] if c.scalar else np.exp(acq[-1])),
                    xytext=(-6, 10), textcoords="offset points", ha="right",
                    fontsize=8, color=INK)
    step = max(1, len(xs) // 15)
    ax.set_xticks(xs[::step], tags[::step], rotation=30 if len(xs) > 8 else 0, fontsize=8)
    ax.set_xlabel("optimization step")
    ax.legend()
    return fig


def fig_parameters(c: Campaign, cfg: Config):
    d = len(c.names)
    ncols = 1 if d == 1 else 2
    nrows = -(-d // ncols)
    fig = _figure(figsize=(8.5, 2.5 * nrows + 0.6), layout="constrained")
    axes = np.atleast_1d(fig.subplots(nrows, ncols, sharex=True, squeeze=False)).ravel()
    n = np.arange(1, len(c.J) + 1)
    lo, hi = cfg.bounds_array
    b = c.best
    for k, ax in enumerate(axes):
        if k >= d:
            ax.set_axis_off()
            continue
        ax.axhline(lo[k], color=BASE, lw=1.0, ls="--")
        ax.axhline(hi[k], color=BASE, lw=1.0, ls="--")
        ax.scatter(n, c.X[:, k], s=24, c=_phase_colors(c), zorder=3)
        ax.scatter([b + 1], [c.X[b, k]], s=130, marker="*", color=INK, zorder=4)
        pad = 0.06 * (hi[k] - lo[k])
        ax.set_ylim(lo[k] - pad, hi[k] + pad)
        ax.set_ylabel(c.names[k])
        ax.xaxis.get_major_locator().set_params(integer=True)
    for ax in axes[-ncols:]:
        ax.set_xlabel("evaluation (order of completion)")
    fig.suptitle(f"Evaluated designs: grey initial design, blue optimization, star the "
                 f"best ({c.tags[b]}); dashes are the bounds", x=0.01, ha="left",
                 fontsize=11)
    return fig


def contributions(c: Campaign, cfg: Config, i: int) -> np.ndarray | None:
    """Each grid point's share of J for evaluation i (zero outside the window);
    they sum to J. None for the flatness form of J, which is not a sum over
    points."""
    if cfg.objective.flatness_beta is not None:
        return None
    m = c.window
    if cfg.objective.time_weighted:
        lo, hi = cfg.objective.window
        pw = trapezoid_weights(c.t, lo, hi)
    else:
        pw = np.full(int(m.sum()), 1.0 / m.sum())
    dev = (c.y[i] - c.target)[m]
    out = np.zeros_like(c.t, dtype=float)
    out[m] = pw * np.where(dev >= 0, 1.0, cfg.objective.alpha) * dev**2
    return out


def fig_contributions(c: Campaign, cfg: Config, labels):
    b = c.best
    contrib = contributions(c, cfg, b)
    if contrib is None:
        return None
    t_label, _, j_label = labels
    below = (c.y[b] < c.target) & c.window
    xs = np.arange(len(c.t))
    fig = _figure(figsize=(8.5, 3.4), layout="constrained")
    ax = fig.subplots()
    for k in xs[~c.window]:
        ax.axvspan(k - 0.5, k + 0.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    if (~below).any():
        ax.bar(xs[~below], contrib[~below], color=BLUE, width=0.62,
               label="above target (w = 1)")
    if below.any():
        ax.bar(xs[below], contrib[below], color=YELLOW, width=0.62,
               label=f"below target (w = alpha = {cfg.objective.alpha:g})")
    step = max(1, len(xs) // 20)
    ax.set_xticks(xs[::step], [f"{v:g}" for v in c.t[::step]], fontsize=8)
    ax.set_xlabel(f"{t_label} (grid points, evenly spaced)")
    ax.set_ylabel(f"share of {j_label}")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_title(f"Where J of the best design ({c.tags[b]}) comes from; the shaded points "
                 "are outside the window")
    ax.legend()
    return fig


def fig_ranking(c: Campaign, labels, top: int = 10):
    order = np.argsort(c.J)[:top][::-1]
    ys = np.arange(len(order))
    fig = _figure(figsize=(8.5, 0.32 * len(order) + 1.6), layout="constrained")
    ax = fig.subplots()
    ax.errorbar(c.J[order], ys, xerr=c.sigma_J[order], fmt="none", ecolor=BASE,
                elinewidth=1.6, capsize=3, zorder=2)
    ax.scatter(c.J[order], ys, s=40, c=[_phase_colors(c)[i] for i in order], zorder=3)
    ax.set_yticks(ys, [c.tags[i] for i in order], fontsize=9)
    if np.all(c.J[order] - c.sigma_J[order] > 0):
        ax.set_xscale("log")
    ax.set_xlabel(rf"{labels[2]}, $\pm 1\sigma$ from the simulation statistics")
    ax.set_title("Best evaluations: overlapping bars are a statistical tie "
                 "(grey initial design, blue optimization)")
    return fig


def animate(c: Campaign, labels, path: Path, fps: float = 2.0, trail: int = 6) -> None:
    """One frame per evaluation: the initial design from worst to best, then
    the optimization steps in order; a fading trail of the previous ones and
    the best so far."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    t_label, y_label, _ = labels
    init = c.initial
    order = sorted(np.flatnonzero(init), key=lambda i: -c.J[i]) + list(np.flatnonzero(~init))
    fig = _figure(figsize=(8.5, 4.6), layout="constrained")
    ax = fig.subplots()
    _shade_outside_window(ax, c)
    ax.plot(c.t, c.target, color=INK, lw=1.2, ls="--", zorder=2, label="target")
    # Scale on the better half of the evaluations: the worst initial designs
    # may leave the frame, but the late steps, close to the target, stay
    # readable.
    w = c.window
    good = c.y[c.J <= np.median(c.J)][:, w]
    lo = min(good.min(), c.target[w].min())
    hi = max(good.max(), c.target[w].max())
    pad = 0.08 * max(hi - lo, 1e-12)
    ax.set_xlim(c.t[0], c.t[-1])
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(t_label)
    ax.set_ylabel(y_label)
    ghosts = [ax.plot([], [], lw=1.2, color=MUTED, alpha=0.0, zorder=3)[0]
              for _ in range(trail)]
    best_line, = ax.plot([], [], lw=2.0, color=GREEN, zorder=4, label="best so far")
    current, = ax.plot([], [], lw=2.4, marker="o", ms=4, color=BLUE, zorder=5)
    text = ax.text(0.015, 0.965, "", transform=ax.transAxes, va="top", ha="left",
                   fontsize=9.5, color=INK,
                   bbox=dict(boxstyle="round,pad=0.45", fc="#fcfcfb", ec=GRID))
    ax.legend(loc="lower left")

    def frame(f):
        i = order[f]
        best = min(order[: f + 1], key=lambda k: c.J[k])
        prev = order[max(0, f - trail):f][::-1]
        for k, g in enumerate(ghosts):
            if k < len(prev):
                g.set_data(c.t, c.y[prev[k]])
                g.set_alpha(0.35 * (1.0 - k / trail))
            else:
                g.set_data([], [])
                g.set_alpha(0.0)
        current.set_data(c.t, c.y[i])
        current.set_color(MUTED if init[i] else BLUE)
        best_line.set_data(c.t, c.y[best])
        phase = "initial design" if init[i] else "optimization"
        text.set_text(f"{f + 1}/{len(order)}  {c.tags[i]} ({phase})\n"
                      f"J = {c.J[i]:.3g}, best so far {c.J[best]:.3g}")
        return [*ghosts, best_line, current, text]

    anim = FuncAnimation(fig, frame, frames=len(order), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=120)


# ----------------------------------------------------------------------
def write_report(cfg: Config, labels, out_dir: Path | None = None, fmt: str = "png",
                 show=(), gif: bool = False, prediction: Path | None = None
                 ) -> tuple[str, list[Path]]:
    """Write summary.txt and the figures into out_dir (default: the report
    folder of the run directory). Returns the summary and the files written.
    prediction: a file saved by 'profilebo surrogate --save', drawn in the
    trajectory figure."""
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("the report needs matplotlib: pip install 'profilebo[plots]'") from exc

    c = load_campaign(cfg)
    for tag in show:
        c.index(tag)                      # fail before writing anything
    pred = None
    if prediction is not None:
        from profilebo.optimization.surrogate_optimum import load_prediction
        pred = load_prediction(prediction)
        if pred["t"].shape != c.t.shape or not np.allclose(pred["t"], c.t):
            raise ValueError(f"{prediction} is on another grid than this campaign")
    out = Path(out_dir) if out_dir is not None else cfg.run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)

    text = summary(c, cfg)
    written = [out / "summary.txt"]
    written[0].write_text(text + "\n")
    with matplotlib.rc_context(STYLE):
        figs = {
            "trajectories": fig_trajectories(c, labels, show, pred),
            "convergence": fig_convergence(c, labels),
            "acquisition": fig_acquisition(c, cfg, labels),
            "parameters": fig_parameters(c, cfg),
            "contributions": fig_contributions(c, cfg, labels),
            "ranking": fig_ranking(c, labels),
        }
        for name, fig in figs.items():
            if fig is not None:
                path = out / f"{name}.{fmt}"
                fig.savefig(path, dpi=150)
                written.append(path)
        if gif:
            path = out / "convergence.gif"
            animate(c, labels, path)
            written.append(path)
    return text, written
