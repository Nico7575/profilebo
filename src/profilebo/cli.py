"""The profilebo command.

    profilebo run CONFIG            run a campaign, or resume it
    profilebo run CONFIG --scalar   the same with the scalar-objective baseline
    profilebo report CONFIG         summary and figures of a campaign
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import warnings
from pathlib import Path

from profilebo import __version__

log = logging.getLogger("profilebo")


def _setup_logging(log_file: Path, verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def _stop_on_ctrl_c(signum, frame) -> None:
    # Ctrl-C, kill, or scancel of an optimizer running as a SLURM job.
    # Exit at once: the loop would otherwise wait for every simulation in
    # progress, which can take hours. Simulations already submitted (Serpent
    # jobs, local or on the cluster) keep running, and the next run of the
    # same command re-attaches to them. The archive is written atomically, so
    # it is never left half written.
    log.warning("Interrupted. Simulations already started keep running; run the same "
                "command again to resume the campaign.")
    logging.shutdown()
    os._exit(130)


def _run(args) -> None:
    # Raised inside BoTorch's dependencies at import; nothing a user can act on.
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
    from profilebo.config import load_config
    from profilebo.evaluators import make_evaluator
    from profilebo.optimization.bo_loop import BOLoop
    from profilebo.optimization.scalar_baseline import ScalarBOLoop

    cfg = load_config(args.config)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(cfg.run_dir / "campaign.log", args.verbose)
    signal.signal(signal.SIGINT, _stop_on_ctrl_c)
    signal.signal(signal.SIGTERM, _stop_on_ctrl_c)

    loop_class = ScalarBOLoop if args.scalar else BOLoop
    log.info("profilebo %s: %s campaign, config %s, run directory %s", __version__,
             "scalar-baseline" if args.scalar else "trajectory", Path(args.config).resolve(),
             cfg.run_dir)
    x_best, j_best = loop_class(cfg, make_evaluator(cfg)).run()
    log.info("Best design: %s, J = %.6g",
             ", ".join(f"{k} = {v:.6g}" for k, v in x_best.items()), j_best)


def _report(args) -> None:
    from profilebo.analysis.report import write_report
    from profilebo.config import load_config
    from profilebo.evaluators import AXIS_LABELS

    cfg = load_config(args.config)
    try:
        text, written = write_report(cfg, AXIS_LABELS[cfg.evaluator], args.out, args.format,
                                     args.show, args.gif)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        sys.exit(f"profilebo report: {exc}")
    print(text)
    print(f"\nwritten to {written[0].parent}: {', '.join(p.name for p in written)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="profilebo",
        description="Bayesian optimization of simulated trajectories toward a target profile.")
    parser.add_argument("--version", action="version", version=f"profilebo {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="run a campaign, or resume it",
        description="Run the campaign described by CONFIG. If its run directory already "
                    "holds results, the campaign resumes where it stopped. The log goes "
                    "to the screen and to campaign.log in the run directory.")
    run.add_argument("config", help="the campaign's YAML file")
    run.add_argument("--scalar", action="store_true",
                     help="use the scalar-objective baseline (a GP on -log J) instead of "
                          "the trajectory surrogate; give it its own run_dir")
    run.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    run.set_defaults(func=_run)

    report = sub.add_parser(
        "report", help="summary and figures of a campaign",
        description="Print a summary of the campaign described by CONFIG and write it, "
                    "with figures, to the report folder of its run directory. It only "
                    "reads the campaign, so it can run while the campaign is running.")
    report.add_argument("config", help="the campaign's YAML file")
    report.add_argument("--out", type=Path, help="folder for the report (default: "
                        "report/ in the run directory)")
    report.add_argument("--format", default="png", choices=("png", "pdf", "svg"),
                        help="figure format (default png)")
    report.add_argument("--show", nargs="+", default=[], metavar="TAG",
                        help="also draw these evaluations in the trajectory figure, "
                             "e.g. --show doe_004 bo_017")
    report.add_argument("--gif", action="store_true",
                        help="also write convergence.gif, one frame per evaluation")
    report.set_defaults(func=_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
