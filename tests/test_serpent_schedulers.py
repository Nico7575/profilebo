"""Backend selection, the job scripts, and the job queries of both backends.

The local backend is exercised for real (bash processes); SLURM through a
stand-in for subprocess.run, since its commands need a cluster.
"""

import subprocess
import sys
import time
import types

import pytest

from profilebo.evaluators.serpent.schedulers import (
    backend_name,
    get_backend,
    job_config,
    local,
    slurm,
)

SLURM_CFG = {
    "executable": "/opt/serpent/sss2",
    "slurm": {
        "partition": "compute", "account": "research", "ntasks": 8, "cpus_per_task": 4,
        "mem_per_cpu": "6000M", "time": "07:00:00", "modules": ["2026", "openmpi"],
        "env_lines": ["export SERPENT_DATA=/data"], "launcher": "srun",
        "extra_lines": ["--exclusive"], "poll_seconds": 60,
    },
}


# --- backend selection ------------------------------------------------------------

def test_explicit_key_wins():
    assert backend_name({"scheduler": "local", "slurm": {}}) == "local"
    assert backend_name({"scheduler": "SLURM"}) == "slurm"


def test_backend_is_inferred_from_its_section():
    assert backend_name({"slurm": {}}) == "slurm"
    assert backend_name({"local": {}}) == "local"


@pytest.mark.parametrize("cfg", [{}, {"slurm": {}, "local": {}}, {"scheduler": "pbs"}])
def test_missing_ambiguous_or_unknown_backend_is_rejected(cfg):
    with pytest.raises(ValueError, match="scheduler"):
        backend_name(cfg)


def test_backends_expose_the_runner_api():
    for cfg in ({"slurm": {}}, {"local": {}}):
        mod = get_backend(cfg)
        for fn in ("write_job_script", "submit", "job_state", "job_name", "cancel",
                   "wait_for_job"):
            assert callable(getattr(mod, fn)), (mod.__name__, fn)


def test_job_config_returns_the_matching_section():
    assert job_config(SLURM_CFG)["poll_seconds"] == 60
    assert job_config({"scheduler": "local"}) == {}


# --- SLURM ----------------------------------------------------------------------

def test_slurm_script_carries_every_directive(tmp_path):
    text = slurm.write_job_script(tmp_path, "pbo_doe_000", "case.serp", SLURM_CFG).read_text()
    for line in ("#SBATCH --job-name=pbo_doe_000", "#SBATCH --partition=compute",
                 "#SBATCH --ntasks=8", "#SBATCH --cpus-per-task=4",
                 "#SBATCH --time=07:00:00", "#SBATCH --mem-per-cpu=6000M",
                 "#SBATCH --account=research", "#SBATCH --exclusive",
                 "module load 2026\nmodule load openmpi", "export SERPENT_DATA=/data",
                 f'cd "{tmp_path.resolve()}"',
                 # runner._serpent_error reads OUT_*.log to diagnose a rejected input
                 "srun /opt/serpent/sss2 case.serp > OUT_pbo_doe_000.log"):
        assert line in text, line
    assert "-omp" not in text


def test_slurm_omits_unset_directives_and_adds_omp_when_asked(tmp_path):
    cfg = {"slurm": {"omp_threads": 4}}
    text = slurm.write_job_script(tmp_path, "j", "case.serp", cfg).read_text()
    for option in ("partition", "time", "mem-per-cpu", "account"):
        assert f"--{option}" not in text
    assert "srun sss2 -omp 4 case.serp" in text


def _fake_run(monkeypatch, returncode, stdout="", stderr=""):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr))


def test_slurm_queued_job_reports_its_state(monkeypatch):
    _fake_run(monkeypatch, 0, "RUNNING\n")
    assert slurm.job_state("1") == "RUNNING"


def test_slurm_job_gone_from_the_queue(monkeypatch):
    _fake_run(monkeypatch, 1, stderr="slurm_load_jobs error: Invalid job id specified")
    assert slurm.job_state("1") is None


def test_slurm_controller_hiccup_is_transient_not_completion(monkeypatch):
    """The dangerous confusion: an unanswered squeue must not read as done."""
    _fake_run(monkeypatch, 1, stderr="slurm_load_jobs error: Socket timed out")
    with pytest.raises(slurm.TransientSchedulerError):
        slurm.job_state("1")


def test_slurm_failed_job_raises(monkeypatch):
    monkeypatch.setattr(slurm, "job_state", lambda j: None)
    monkeypatch.setattr(slurm, "final_state", lambda j: "TIMEOUT")
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        slurm.wait_for_job("1", "/nonexistent", poll_seconds=0)


# --- local ----------------------------------------------------------------------

def _fake_serpent(tmp_path, body):
    """An executable standing in for sss2: called as <exe> <input>."""
    exe = tmp_path / "fake_sss2"
    exe.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n")
    exe.chmod(0o755)
    return exe


def test_local_script(tmp_path):
    cfg = {"executable": "/opt/sss2",
           "local": {"launcher": "mpirun -np 4", "omp_threads": 2,
                     "env_lines": ["export SERPENT_DATA=/data"]}}
    text = local.write_job_script(tmp_path, "pbo_doe_000", "case.serp", cfg).read_text()
    assert "export SERPENT_DATA=/data\n" in text
    assert "mpirun -np 4 /opt/sss2 -omp 2 case.serp > OUT_pbo_doe_000.log 2>&1" in text
    text = local.write_job_script(tmp_path, "j", "case.serp", {"executable": "sss2"}).read_text()
    assert "\nsss2 case.serp > OUT_j.log 2>&1" in text


def test_local_run_to_completion(tmp_path):
    exe = _fake_serpent(tmp_path, "open(sys.argv[1] + '_res.m', 'w').write('ok')")
    cfg = {"executable": str(exe), "local": {}}
    job = local.submit(local.write_job_script(tmp_path, "pbo_x", "case.serp", cfg))
    out = local.wait_for_job(job, tmp_path / "case.serp_res.m", poll_seconds=0.05)
    assert out.read_text() == "ok"
    assert local.exit_status(job, tmp_path) == 0
    assert local.job_state(job) is None


def test_local_failure_is_reported(tmp_path):
    exe = _fake_serpent(tmp_path, "print('Input error in parameter'); sys.exit(255)")
    cfg = {"executable": str(exe), "local": {}}
    job = local.submit(local.write_job_script(tmp_path, "pbo_x", "case.serp", cfg))
    with pytest.raises(RuntimeError, match="status 255"):
        local.wait_for_job(job, tmp_path / "case.serp_res.m", poll_seconds=0.05)
    assert "Input error" in (tmp_path / "OUT_pbo_x.log").read_text()


def test_local_running_job_is_named_and_can_be_cancelled(tmp_path):
    exe = _fake_serpent(tmp_path, "time.sleep(60)")
    cfg = {"executable": str(exe), "local": {}}
    job = local.submit(local.write_job_script(tmp_path, "pbo_doe_003", "case.serp", cfg))
    assert local.job_state(job) == "RUNNING"
    assert local.job_name(job) == "pbo_doe_003"
    local.cancel(job)
    deadline = time.time() + 10
    while local.job_state(job) is not None and time.time() < deadline:
        time.sleep(0.05)
    assert local.job_state(job) is None
    with pytest.raises(RuntimeError, match="without an exit status|exited with status"):
        local.wait_for_job(job, tmp_path / "case.serp_res.m", poll_seconds=0.05)


def test_local_process_of_an_earlier_optimizer_is_found_with_ps(tmp_path):
    """After a restart the process is no longer our child: only ps knows it."""
    exe = _fake_serpent(tmp_path, "time.sleep(60)")
    cfg = {"executable": str(exe), "local": {}}
    job = local.submit(local.write_job_script(tmp_path, "pbo_bo_017", "case.serp", cfg))
    proc = local._procs.pop(job)                  # forget it, as a new optimizer would
    try:
        assert local.job_state(job) == "RUNNING"
        assert local.job_name(job) == "pbo_bo_017"
    finally:
        local.cancel(job)
        proc.wait(timeout=10)
    assert local.job_state(job) is None


def test_local_unknown_process_has_no_name():
    assert local.job_state("999999999") is None
    assert local.job_name("999999999") is None
