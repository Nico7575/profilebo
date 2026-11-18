"""Local backend: Serpent runs as a background process on this machine.

Each case is a small bash script, started in its own session so that it
keeps running if the optimizer is interrupted (Ctrl-C, a closed terminal),
like a job on a cluster. The job id is the process id; a restarted optimizer
re-attaches to a case that is still running instead of starting it again.
The script records Serpent's exit status in job_<pid>.exit next to the
input, which is how a finished run is told apart from a killed one.

Concurrency is bo.doe_parallel cases during the initial design and one
afterwards; keep doe_parallel times the cores per case (MPI ranks times
OpenMP threads) within the cores of the machine.

Uses ps, so it runs on Linux and macOS.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

SETTINGS = {"launcher", "omp_threads", "env_lines", "poll_seconds", "max_attempts"}

_SCRIPT = """\
#!/bin/bash
{env_lines}cd "{workdir}"
{command} {input_file} > {log_file} 2>&1
echo $? > "job_$$.exit"
"""

# Processes started by this optimizer, by job id. Polling them (rather than
# asking ps) also reaps them: a finished child that is never waited for stays
# in the process table as a zombie.
_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def write_job_script(workdir: str | Path, job_name: str, input_file: str, serpent_cfg: dict) -> Path:
    """Render the run script into workdir.

    serpent.local.launcher is put in front of the executable, for instance
    "mpirun -np 4" for an MPI build; leave it empty for a serial or OpenMP
    run. omp_threads adds the -omp flag.
    """
    local = serpent_cfg.get("local") or {}
    omp_threads = local.get("omp_threads")
    command = " ".join(filter(None, [
        local.get("launcher") or "",
        str(serpent_cfg.get("executable", "sss2")),
        f"-omp {int(omp_threads)}" if omp_threads else "",
    ]))
    script = _SCRIPT.format(
        env_lines="".join(f"{x}\n" for x in local.get("env_lines") or []),
        workdir=Path(workdir).resolve(),
        command=command,
        input_file=input_file,
        log_file=f"OUT_{job_name}.log",
    )
    path = Path(workdir) / f"{job_name}.sh"
    path.write_text(script)
    return path


def submit(script: str | Path) -> str:
    """Start the script in the background; returns its process id."""
    script = Path(script).resolve()
    proc = subprocess.Popen(["bash", str(script)], cwd=script.parent,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    job_id = str(proc.pid)
    with _lock:
        _procs[job_id] = proc
    log.info("Started %s as process %s", script, job_id)
    return job_id


def _ps(job_id: str, field: str) -> str | None:
    # -ww: never truncate to the terminal width (COLUMNS), or a long script
    # path would lose the job name.
    result = subprocess.run(["ps", "-ww", "-p", str(job_id), "-o", f"{field}="],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def job_state(job_id: str) -> str | None:
    """RUNNING while the process is alive, None once it has ended."""
    with _lock:
        proc = _procs.get(job_id)
    if proc is not None:
        return None if proc.poll() is not None else "RUNNING"
    stat = _ps(job_id, "stat")
    if stat is None or stat.startswith("Z"):
        return None
    return "RUNNING"


def job_name(job_id: str) -> str | None:
    """Name of the running job (its script name without .sh), or None.

    Process ids are reused, so the runner compares this name before it
    re-attaches to, or cancels, a recorded id.
    """
    if job_state(job_id) is None:
        return None
    args = _ps(job_id, "args")                    # "bash /path/to/<name>.sh"
    if not args or not args.endswith(".sh") or " " not in args:
        return None
    return Path(args.split(" ", 1)[1]).stem


def cancel(job_id: str) -> None:
    """Stop the run and everything it started (mpirun and its ranks)."""
    try:
        os.killpg(int(job_id), signal.SIGTERM)    # the script leads its own session
    except (ProcessLookupError, PermissionError):
        pass
    log.warning("Cancelled process %s", job_id)


def exit_status(job_id: str, workdir: str | Path) -> int | None:
    """Serpent's exit status, or None if the run never recorded one."""
    try:
        return int((Path(workdir) / f"job_{job_id}.exit").read_text().strip())
    except (OSError, ValueError):
        return None


def wait_for_job(job_id: str, expected_output: str | Path, poll_seconds: float = 10.0,
                 timeout: float | None = None) -> Path:
    """Block until the run ends; raise RuntimeError unless it ended cleanly
    and FileNotFoundError if expected_output is missing."""
    expected_output = Path(expected_output)
    start = time.time()
    while job_state(job_id) is not None:
        if timeout and time.time() - start > timeout:
            raise TimeoutError(f"Process {job_id} exceeded the wait limit of {timeout} s")
        time.sleep(poll_seconds)

    status = exit_status(job_id, expected_output.parent)
    if status is None:
        raise RuntimeError(f"Process {job_id} ended without an exit status (killed?)")
    if status != 0:
        raise RuntimeError(f"Process {job_id} exited with status {status}")
    if not expected_output.exists():
        raise FileNotFoundError(f"Process {job_id} finished but {expected_output} is missing")
    return expected_output
