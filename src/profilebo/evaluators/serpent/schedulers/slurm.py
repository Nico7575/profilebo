"""SLURM backend: sbatch to submit, squeue and sacct to follow a job.

Every query that can fail transiently (the controller not answering) is
retried rather than read as an answer: an unanswered squeue must never be
mistaken for a finished job.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

SETTINGS = {"partition", "account", "ntasks", "cpus_per_task", "mem_per_cpu", "omp_threads",
            "time", "modules", "env_lines", "launcher", "extra_lines", "poll_seconds",
            "max_attempts"}

_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name={job_name}
{partition_line}#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
{time_line}{mem_line}{account_line}{extra_lines}
{module_lines}{env_lines}
cd "{workdir}"
{launcher} {executable}{omp_flag} {input_file} > {log_file}
"""


def write_job_script(workdir: str | Path, job_name: str, input_file: str, serpent_cfg: dict) -> Path:
    """Render the sbatch script into workdir.

    ntasks is the number of MPI ranks. The -omp flag is written only when
    omp_threads is set, which needs an sss2 built with OpenMP; cpus_per_task
    may exceed it, the extra CPUs then only reserve memory on partitions that
    cap memory per CPU. Directives left unset are omitted, so the cluster
    defaults apply.
    """
    slurm = serpent_cfg["slurm"]

    def line(option: str, key: str) -> str:
        value = slurm.get(key)
        return f"#SBATCH --{option}={value}\n" if value else ""

    omp_threads = slurm.get("omp_threads")
    script = _SCRIPT.format(
        job_name=job_name,
        partition_line=line("partition", "partition"),
        ntasks=slurm.get("ntasks", 1),
        cpus_per_task=int(slurm.get("cpus_per_task", 1)),
        time_line=line("time", "time"),
        mem_line=line("mem-per-cpu", "mem_per_cpu"),
        account_line=line("account", "account"),
        extra_lines="".join(f"#SBATCH {x}\n" for x in slurm.get("extra_lines") or []),
        module_lines="".join(f"module load {m}\n" for m in slurm.get("modules") or []),
        env_lines="".join(f"{x}\n" for x in slurm.get("env_lines") or []),
        launcher=slurm.get("launcher", "srun"),
        workdir=Path(workdir).resolve(),
        executable=serpent_cfg.get("executable", "sss2"),
        omp_flag=f" -omp {int(omp_threads)}" if omp_threads else "",
        input_file=input_file,
        log_file=f"OUT_{job_name}.log",
    )
    path = Path(workdir) / f"{job_name}.sbatch"
    path.write_text(script)
    return path


class TransientSchedulerError(RuntimeError):
    """The controller did not answer; retry, do not conclude."""


def submit(script: str | Path) -> str:
    """sbatch --parsable, returning the job id.

    sbatch can report a failure (a submit timeout) while the controller has
    accepted the job anyway. On an error, wait and look for a queued job
    with this script's name before raising, and adopt it if there is one.
    """
    script = Path(script).resolve()
    result = subprocess.run(["sbatch", "--parsable", str(script)],
                            capture_output=True, text=True, cwd=script.parent)
    if result.returncode == 0:
        job_id = result.stdout.strip().split(";")[0]
        log.info("Submitted %s as job %s", script, job_id)
        return job_id

    log.warning("sbatch error for %s (%s); checking whether the job was "
                "accepted anyway", script, result.stderr.strip())
    time.sleep(20)
    lookup = subprocess.run(["squeue", "--me", "-h", "-n", script.stem, "-o", "%i"],
                            capture_output=True, text=True)
    orphans = lookup.stdout.split()
    if lookup.returncode == 0 and len(orphans) == 1:
        log.warning("Adopted job %s for %s (the submission timed out on the client side)",
                    orphans[0], script)
        return orphans[0]
    raise RuntimeError(f"sbatch rejected {script}: {result.stderr.strip()}")


def job_state(job_id: str) -> str | None:
    """Live state from squeue, or None once the job has left the queue.

    Raises TransientSchedulerError when squeue fails for another reason.
    """
    result = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip() or None
    if "Invalid job id" in result.stderr:
        return None
    raise TransientSchedulerError(f"squeue failed: {result.stderr.strip()}")


def job_name(job_id: str) -> str | None:
    """Name of a queued or running job, or None if it is not in the queue."""
    result = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%.200j"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def cancel(job_id: str) -> None:
    """scancel, tolerating a job that already left the queue."""
    subprocess.run(["scancel", job_id], capture_output=True, text=True)
    log.warning("Cancelled job %s", job_id)


def final_state(job_id: str) -> str:
    """State from accounting (COMPLETED, FAILED, RUNNING, ... or empty)."""
    result = subprocess.run(["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
                            capture_output=True, text=True)
    return result.stdout.strip()


_BAD_STATES = ("FAILED", "OUT_OF_ME", "TIMEOUT", "CANCELLED", "NODE_FAIL",
               "PREEMPTED", "DEADLINE", "BOOT_FAIL")


def wait_for_job(job_id: str, expected_output: str | Path, poll_seconds: float = 60.0,
                 timeout: float | None = None) -> Path:
    """Block until the job completes and expected_output exists.

    The wait ends only on a corroborated terminal state: squeue says the job
    is gone AND sacct reports a terminal state (sacct can lag behind squeue).
    Raises RuntimeError if the job ended in a failed state.
    """
    expected_output = Path(expected_output)
    start = time.time()
    unknown_streak = 0

    while True:
        if timeout and time.time() - start > timeout:
            raise TimeoutError(f"Job {job_id} exceeded the wait limit of {timeout} s")
        try:
            state = job_state(job_id)
        except TransientSchedulerError as e:
            log.warning("Job %s: %s; retrying", job_id, e)
            time.sleep(poll_seconds)
            continue
        if state is not None:
            log.debug("Job %s: %s", job_id, state)
            time.sleep(poll_seconds)
            continue

        fs = final_state(job_id)
        if fs.startswith("COMPLETED"):
            break
        if any(fs.startswith(b) for b in _BAD_STATES):
            raise RuntimeError(f"Job {job_id} ended in state {fs}")
        # RUNNING or PENDING (an squeue hiccup) or empty (sacct lag): wait.
        unknown_streak += 1
        if unknown_streak > 30:
            raise RuntimeError(f"Job {job_id}: state unresolvable after {unknown_streak} "
                               f"polls (last sacct answer: {fs!r})")
        log.warning("Job %s: not in squeue but sacct says %r; retrying", job_id, fs)
        time.sleep(poll_seconds)

    # Shared file systems can show the output a little after the job ends.
    deadline = time.time() + 60
    while not expected_output.exists():
        if time.time() > deadline:
            raise FileNotFoundError(f"Job {job_id} COMPLETED but {expected_output} never appeared")
        time.sleep(2)
    return expected_output
