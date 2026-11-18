"""One Serpent evaluation: batches, particle dispersion, input, job, trajectory.

Each evaluation works in its own case directory: a restarted optimizer re-attaches 
to a job still in flight, reuses acomplete result, and runs the case only when 
neither is there.

Two things decide whether a case directory can be reused:

  * the depletion grid. The same design on another grid is a different
    trajectory, and every trajectory of a campaign must share one grid, so a
    case rendered on another grid is always run again.
  * the design. A restarted optimizer re-derives its proposal, which can
    differ in the last digits from the design a case was rendered for (or
    entirely, if the settings changed). Such a case is adopted rather than
    discarded, and the evaluator reports the design it was rendered for, so
    the archive pairs each trajectory with the design that produced it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from profilebo.evaluators.serpent.batches import Batch, BatchSpec
from profilebo.evaluators.serpent.disperser import disperse_batches
from profilebo.evaluators.serpent.extract import BurnupTrajectory, parse_res_m
from profilebo.evaluators.serpent.inputs import (
    INPUT_NAME,
    burnup_days_of,
    case_matches,
    generate_input,
    input_burnup_days,
)
from profilebo.evaluators.serpent.schedulers import get_backend, job_config
from profilebo.rng import derive_seed

log = logging.getLogger(__name__)

JOB_ID_FILE = "job_id"       # written at submission, read to re-attach
JOB_PREFIX = "pbo_"          # job name = prefix + case directory name
RETRY_WAIT = 60.0            # seconds between attempts after a failed job


def _case_grid(input_file: Path) -> tuple[float, ...] | None:
    try:
        return input_burnup_days(input_file.read_text())
    except OSError:
        return None


def _expected_points(input_file: Path) -> int | None:
    """Burnup points a complete result of this input has (steps + BOL)."""
    days = _case_grid(input_file)
    return None if days is None else len(days) + 1


def _grid_matches(input_file: Path, serpent_cfg: dict) -> bool:
    """True if input_file was rendered on the configured depletion grid."""
    found = _case_grid(input_file)
    if found is None:
        return False
    want = burnup_days_of(serpent_cfg)
    return len(found) == len(want) and all(
        abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(found, want))


_SERPENT_ERRORS = ("Input error", "Fatal error", "Simulation aborted",
                   "Geometry error", "Memory allocation")

# Serpent reports a short read of an ACE file as a WARNING, not an error: the
# run continues and writes a complete-looking res.m built on partial
# cross-section data (seen during a file-system fault, where reads returned
# 0 bytes or a single 512 kB chunk). Its k_eff is plausible but wrong, so such
# a result is never parsed or reused.
_XS_DATA_WARNINGS = ("Error in XSS array",)


def _xs_data_error(case_dir: Path) -> str | None:
    """A truncated cross-section read reported for this case, if any.
    Serpent writes it to the .wrn file; it also reaches the output log."""
    for pattern in ("*.wrn", "OUT_*.log"):
        for path in sorted(case_dir.glob(pattern)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for marker in _XS_DATA_WARNINGS:
                i = text.find(marker)
                if i != -1:
                    return " ".join(text[i:i + 120].split())[:100]
    return None


def _serpent_error(case_dir: Path) -> str | None:
    """The error Serpent reported for this case, if it rejected the input.

    Tells a deterministic failure (a bad input: retrying is pointless) from an
    infrastructure fault (a launch timeout, a node failure), after which the
    Serpent log simply stops without complaint.
    """
    for log_file in case_dir.glob("OUT_*.log"):
        try:
            text = log_file.read_text(errors="replace")
        except OSError:
            continue
        for marker in _SERPENT_ERRORS:
            i = text.find(marker)
            if i != -1:
                return " ".join(text[i:i + 200].split())[:160]
    return None


def _complete_result(res_m: Path, expected: int | None) -> BurnupTrajectory | None:
    """The trajectory in res_m, only if it has every burnup point.

    Serpent writes res.m step by step, so the file of a running or killed job
    parses fine but must not be used.
    """
    bad_xs = _xs_data_error(res_m.parent)
    if bad_xs is not None:
        log.error("%s: Serpent read cross-section data incompletely (%s); refusing "
                  "the result, its k_eff is not trustworthy", res_m.parent.name, bad_xs)
        return None
    try:
        traj = parse_res_m(res_m)
    except (ValueError, OSError):
        return None
    if expected is not None and len(traj.days) != expected:
        log.warning("%s has %d of %s burnup points: incomplete", res_m, len(traj.days), expected)
        return None
    return traj


def _job_is_ours(sched, job_id: str, case_dir: Path) -> bool:
    """The recorded job is still alive AND carries this case's name.

    Job ids are not permanent (a cluster can reset its counter, a machine
    reuses process ids), so a recorded id may belong to someone else's job:
    never re-attach to, or cancel, that.
    """
    if sched.job_state(job_id) is None:
        return False
    return sched.job_name(job_id) == JOB_PREFIX + case_dir.name


def run_case(batches: list[Batch], specs: list[BatchSpec], case_dir: str | Path,
             serpent_cfg: dict) -> BurnupTrajectory:
    """Run Serpent for these absorber batches in case_dir; blocking.

    serpent_cfg is the serpent section of the config, with template resolved
    to a path. Raises RuntimeError if Serpent rejects the input, or if the
    case keeps failing after serpent.<backend>.max_attempts attempts.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_file = case_dir / INPUT_NAME
    res_m = case_dir / f"{INPUT_NAME}_res.m"
    job_id_file = case_dir / JOB_ID_FILE
    expected = _expected_points(input_file)

    sched = get_backend(serpent_cfg)
    sched_cfg = job_config(serpent_cfg)
    poll = float(sched_cfg.get("poll_seconds", 60))
    max_attempts = int(sched_cfg.get("max_attempts", 3))

    same_design = input_file.exists() and case_matches(case_dir, batches, specs)
    same_grid = input_file.exists() and _grid_matches(input_file, serpent_cfg)
    if same_design and not same_grid:
        log.warning("%s holds this design on another depletion grid (%d steps on disk, "
                    "%d configured); running it again", case_dir.name,
                    len(_case_grid(input_file) or ()), len(burnup_days_of(serpent_cfg)))

    # 1. A job may still be in flight (the optimizer was restarted): re-attach
    #    before trusting res.m, which grows while the job runs.
    if job_id_file.exists():
        job_id = job_id_file.read_text().strip()
        if _job_is_ours(sched, job_id, case_dir):
            if not same_grid:
                log.warning("Job %s in %s uses another depletion grid; cancelling it",
                            job_id, case_dir.name)
                sched.cancel(job_id)
            else:
                if same_design:
                    log.info("Re-attaching to job %s for %s", job_id, case_dir.name)
                else:
                    log.warning("Job %s in %s was started for an earlier proposal; adopting "
                                "it (its design is recorded)", job_id, case_dir.name)
                try:
                    sched.wait_for_job(job_id, res_m, poll_seconds=poll)
                except (RuntimeError, FileNotFoundError) as exc:
                    # A re-attached job can fail like any other (a time limit).
                    # Fall through and run the case again with the current
                    # settings rather than stopping the campaign.
                    log.warning("Re-attached job %s for %s failed (%s); running the case again",
                                job_id, case_dir.name, exc)
                else:
                    traj = _complete_result(res_m, expected)
                    if traj is not None:
                        return traj
                    log.warning("Job %s finished but its result in %s is unusable; "
                                "running the case again", job_id, case_dir.name)

    # 2. A finished result on disk: reusable if complete and on this grid.
    if res_m.exists():
        traj = _complete_result(res_m, expected)
        if traj is not None and same_grid:
            if same_design:
                log.info("Reusing the result in %s", case_dir)
            else:
                log.warning("%s holds a complete result for an earlier proposal; reusing "
                            "it (its design is recorded)", case_dir)
            return traj
        if traj is None:
            log.warning("Incomplete result in %s; running the case again", case_dir)

    # 3. Run the case. With serpent.rng_seed (set by bo.reproducible) the
    #    particle positions and the transport random numbers are fixed per
    #    case, keyed on its name, so retries and restarts rebuild it exactly.
    log.info("Case %s: %s", case_dir.name,
             ", ".join(f"R{b.index}={b.radius:.5f} N{b.index}={b.count}" for b in batches))
    rng_seed = serpent_cfg.get("rng_seed")
    disperser_seed = transport_seed = None
    if rng_seed is not None:
        disperser_seed = derive_seed(rng_seed, case_dir.name, "disperser")
        transport_seed = derive_seed(rng_seed, case_dir.name, "transport")

    job_name = JOB_PREFIX + case_dir.name
    for attempt in range(1, max_attempts + 1):
        pbed = disperse_batches(batches, "particles.pbed", serpent_cfg, workdir=case_dir,
                                seed=disperser_seed)
        generate_input(batches, pbed.name, serpent_cfg["template"], input_file, serpent_cfg,
                       seed=transport_seed)
        expected = _expected_points(input_file)
        # Serpent appends to the .wrn file; a warning left by an earlier
        # attempt would otherwise condemn every later one.
        for stale in case_dir.glob("*.wrn"):
            stale.unlink(missing_ok=True)

        script = sched.write_job_script(case_dir, job_name, INPUT_NAME, serpent_cfg)
        job_id = sched.submit(script)
        job_id_file.write_text(job_id + "\n")
        try:
            sched.wait_for_job(job_id, res_m, poll_seconds=poll)
        except (RuntimeError, FileNotFoundError) as exc:
            # A rejected input fails the same way every time; an
            # infrastructure fault is transient, and retrying is the fix.
            reason = _serpent_error(case_dir)
            if reason:
                raise RuntimeError(f"Serpent rejected {case_dir.name}: {reason}") from exc
            if attempt == max_attempts:
                raise RuntimeError(f"{case_dir.name} failed {max_attempts} times with no "
                                   f"Serpent error; last: {exc}") from exc
            log.warning("%s: attempt %d of %d failed (%s); retrying in %.0f s",
                        case_dir.name, attempt, max_attempts, exc, RETRY_WAIT)
            time.sleep(RETRY_WAIT)
            continue

        traj = _complete_result(res_m, expected)
        if traj is not None:
            return traj
        if attempt == max_attempts:
            bad_xs = _xs_data_error(case_dir)
            if bad_xs:
                raise RuntimeError(
                    f"{case_dir.name}: Serpent kept reading cross-section data incompletely "
                    f"({bad_xs}) in {max_attempts} attempts. This is a storage fault, not a "
                    "bad input: check that SERPENT_DATA is fully readable")
            raise RuntimeError(f"{case_dir.name}: the job finished but {res_m.name} is "
                               f"incomplete after {max_attempts} attempts")
        log.warning("%s: attempt %d of %d gave an incomplete result; retrying",
                    case_dir.name, attempt, max_attempts)

    raise AssertionError("unreachable")
