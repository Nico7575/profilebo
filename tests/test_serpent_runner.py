"""Run management: reuse, re-attach, retries, and the guards that keep a
wrong result out of the archive.

The end-to-end tests run the real local backend with a stand-in for sss2
that writes a res.m on the grid of its input; only the particle dispersion
is replaced (it needs Serpent).
"""

import sys
import types

import numpy as np
import pytest

import profilebo.evaluators.serpent.runner as runner
from profilebo.evaluators.serpent.batches import parse_batches, resolve
from profilebo.evaluators.serpent.evaluator import SerpentEvaluator, check_section
from profilebo.evaluators.serpent.inputs import INPUT_NAME, format_burnup_days
from profilebo.evaluators.serpent.schedulers import local
from profilebo.evaluators.serpent.runner import (
    _complete_result,
    _expected_points,
    _grid_matches,
    _job_is_ours,
    _serpent_error,
    _xs_data_error,
    run_case,
)

X = {"R1": 0.0205, "R2": 0.0749, "N1": 783.0, "N2": 151.0}
DAYS = [1, 5, 30, 180, 365]

# Stands in for sss2: writes <input>_res.m with one entry per burnup point
# of the input's dep daytot card, and counts its runs in the file "runs".
# With fail_first=N in the environment of the test it fails N times first.
FAKE_SSS2 = """\
import os, sys
inp = sys.argv[-1]
with open("runs", "a") as f:
    f.write("run\\n")
runs = open("runs").read().count("run")
if runs <= int(os.environ.get("FAKE_FAIL_FIRST", "0")):
    sys.exit(1)
if os.environ.get("FAKE_REJECT"):
    print("Input error in parameter \\"cell\\" on line 136\\nSimulation aborted.")
    sys.exit(255)
days = []
for line in open(inp):
    if line.strip().startswith("dep daytot"):
        days = [0.0] + [float(v) for v in line.split()[2:]]
with open(inp + "_res.m", "w") as f:
    for i, d in enumerate(days):
        k = 1.05 - 1e-5 * d
        f.write(f"ABS_KEFF (idx, [1:   2]) = [  {k:.5E} 0.00050 ];\\n")
        f.write(f"BURN_DAYS (idx, [1:   2]) = [  {d:.5E} 0.00000E+00 ];\\n")
        f.write(f"BURNUP (idx, [1:   2]) = [  {d / 10:.5E} 0.00000E+00 ];\\n")
"""


@pytest.fixture
def cfg(serpent_cfg, tmp_path, monkeypatch):
    """The example serpent section, run locally with the stand-in sss2."""
    exe = tmp_path / "fake_sss2"
    exe.write_text(f"#!{sys.executable}\n{FAKE_SSS2}")
    exe.chmod(0o755)
    serpent_cfg.update(executable=str(exe), scheduler="local", burnup_days=DAYS,
                       local={"poll_seconds": 0.05, "max_attempts": 3}, rng_seed=42)

    def fake_disperse(batches, out_file, scfg, workdir=".", seed=None):
        rows = [f"0 0 0 0.05 {scfg['fuel']['universe']}"] * 10
        for b in batches:
            rows += [f"0 0 0 {b.radius} {b.universe}"] * b.count
        path = workdir / out_file
        path.write_text("\n".join(rows) + "\n")
        return path

    monkeypatch.setattr(runner, "disperse_batches", fake_disperse)
    monkeypatch.setattr(runner, "RETRY_WAIT", 0.0)
    return serpent_cfg


def runs(case_dir) -> int:
    path = case_dir / "runs"
    return path.read_text().count("run") if path.exists() else 0


def batches_at(cfg, x=X):
    specs = parse_batches(cfg)
    return resolve(specs, x), specs


# --- a rejected input vs an infrastructure fault ------------------------------------

REJECTED_INPUT = """\
Processing cells...
OK.

Input error in parameter "cell" on line 136 in file "case.serp":

Material reflector is not defined

Simulation aborted.
"""


def test_infrastructure_fault_is_retryable(tmp_path):
    # the launcher killed the run: the log simply stops, Serpent said nothing
    (tmp_path / "OUT_pbo_bo_023.log").write_text("Reading nuclide data...\n")
    assert _serpent_error(tmp_path) is None


def test_rejected_input_is_reported(tmp_path):
    (tmp_path / "OUT_pbo_bo_023.log").write_text(REJECTED_INPUT)
    reason = _serpent_error(tmp_path)
    assert "Input error" in reason and "reflector" in reason


def test_healthy_run_and_missing_log_report_nothing(tmp_path):
    assert _serpent_error(tmp_path) is None
    (tmp_path / "OUT_pbo_bo_022.log").write_text("Transport cycle completed.\n")
    assert _serpent_error(tmp_path) is None


# --- truncated cross-section data -----------------------------------------------

# These decay warnings appear in every healthy run and are not a data fault.
BENIGN_WRN = """\
Warning message from function ReadDecayFile:

Sum of decay branching ratios is 0.450000 (ZAI = 771690)
"""
TRUNCATED_XS_WRN = BENIGN_WRN + """
Warning message from function ReadACEFile:

Error in XSS array (92235.83c)
"""
COMPLETE_RES_M = """\
ABS_KEFF                  (idx, [1:   2]) = [  9.69922E-01 0.00044 ];
BURN_DAYS                 (idx, [1:   2]) = [  0.00000E+00 0.00000E+00 ];
BURNUP                    (idx, [1:   2]) = [  0.00000E+00 0.00000E+00 ];
"""


def test_ordinary_warnings_are_not_a_data_fault(tmp_path):
    (tmp_path / "case.serp.wrn").write_text(BENIGN_WRN)
    assert _xs_data_error(tmp_path) is None


def test_truncated_cross_sections_are_detected_in_wrn_and_log(tmp_path):
    (tmp_path / "case.serp.wrn").write_text(TRUNCATED_XS_WRN)
    assert "92235.83c" in _xs_data_error(tmp_path)
    other = tmp_path / "o"
    other.mkdir()
    (other / "OUT_pbo_doe_000.log").write_text("Error in XSS array (64157.81c)\n")
    assert _xs_data_error(other) is not None


def test_a_complete_result_on_truncated_data_is_refused(tmp_path):
    res_m = tmp_path / "case.serp_res.m"
    res_m.write_text(COMPLETE_RES_M)
    assert _complete_result(res_m, expected=1) is not None
    (tmp_path / "case.serp.wrn").write_text(TRUNCATED_XS_WRN)
    assert _complete_result(res_m, expected=1) is None


def test_a_partial_result_is_refused(tmp_path):
    res_m = tmp_path / "case.serp_res.m"
    res_m.write_text(COMPLETE_RES_M)
    assert _complete_result(res_m, expected=2) is None


# --- depletion grid -------------------------------------------------------------

def _input(tmp_path, days):
    p = tmp_path / INPUT_NAME
    p.write_text(f'set title "x"\n\ndep daytot {format_burnup_days(days)}\n\nset inventory\n')
    return p


def test_expected_points_follow_the_rendered_grid(tmp_path):
    assert _expected_points(_input(tmp_path, DAYS)) == len(DAYS) + 1
    assert _expected_points(tmp_path / "missing.serp") is None


def test_a_case_on_another_grid_is_not_reusable(tmp_path):
    case = _input(tmp_path, DAYS)
    assert _grid_matches(case, {"burnup_days": DAYS})
    assert not _grid_matches(case, {"burnup_days": DAYS + [730]})
    assert not _grid_matches(tmp_path / "missing.serp", {"burnup_days": DAYS})


# --- job identity -----------------------------------------------------------------

def test_a_reused_job_id_of_someone_else_is_not_adopted(tmp_path):
    """Job and process ids are reused; the name tells our job apart."""
    case = tmp_path / "bo_016"
    other = types.SimpleNamespace(job_state=lambda j: "RUNNING", job_name=lambda j: "someone")
    ours = types.SimpleNamespace(job_state=lambda j: "RUNNING", job_name=lambda j: "pbo_bo_016")
    gone = types.SimpleNamespace(job_state=lambda j: None, job_name=lambda j: None)
    assert not _job_is_ours(other, "1", case)
    assert _job_is_ours(ours, "1", case)
    assert not _job_is_ours(gone, "1", case)


def test_a_failed_reattached_job_falls_through_to_a_fresh_run(cfg, tmp_path, monkeypatch,
                                                              write_serpent_case):
    """An optimizer restarted while its job is in flight must survive that job
    failing (a time limit): the case is run again, the campaign goes on."""
    case = tmp_path / "bo_016"
    write_serpent_case(case, cfg, X)
    (case / "job_id").write_text("111\n")
    real_wait = local.wait_for_job

    def wait(job_id, res_m, poll_seconds=60):
        if job_id == "111":
            raise RuntimeError("Job 111 ended in state TIMEOUT")
        return real_wait(job_id, res_m, poll_seconds)

    monkeypatch.setattr(local, "job_state",
                        lambda j, real=local.job_state: "RUNNING" if j == "111" else real(j))
    monkeypatch.setattr(local, "job_name", lambda j: "pbo_bo_016")
    monkeypatch.setattr(local, "wait_for_job", wait)

    traj = run_case(*batches_at(cfg), case, cfg)
    assert len(traj.days) == len(DAYS) + 1
    assert runs(case) == 1
    assert (case / "job_id").read_text().strip() != "111"


# --- end to end on the local backend ------------------------------------------------

def test_a_case_runs_and_is_then_reused(cfg, tmp_path):
    case = tmp_path / "doe_000"
    traj = run_case(*batches_at(cfg), case, cfg)
    np.testing.assert_allclose(traj.days, [0] + DAYS)
    assert "set seed" in (case / INPUT_NAME).read_text()
    assert runs(case) == 1

    again = run_case(*batches_at(cfg), case, cfg)          # a restarted campaign
    np.testing.assert_array_equal(again.k_eff, traj.k_eff)
    assert runs(case) == 1


def test_a_case_on_another_grid_is_run_again(cfg, tmp_path):
    case = tmp_path / "doe_000"
    run_case(*batches_at(cfg), case, cfg)
    cfg["burnup_days"] = DAYS + [730]
    traj = run_case(*batches_at(cfg), case, cfg)
    assert runs(case) == 2
    assert traj.days[-1] == 730


def test_an_in_flight_job_is_reattached_not_resubmitted(cfg, tmp_path, write_serpent_case):
    case = tmp_path / "bo_003"
    write_serpent_case(case, cfg, X)
    script = local.write_job_script(case, "pbo_bo_003", INPUT_NAME, cfg)
    (case / "job_id").write_text(local.submit(script) + "\n")
    local._procs.clear()                      # as seen by a new optimizer process
    traj = run_case(*batches_at(cfg), case, cfg)
    assert len(traj.days) == len(DAYS) + 1
    assert runs(case) == 1


def test_a_transient_failure_is_retried(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_FIRST", "2")
    traj = run_case(*batches_at(cfg), tmp_path / "doe_000", cfg)
    assert len(traj.days) == len(DAYS) + 1
    assert runs(tmp_path / "doe_000") == 3


def test_a_case_that_keeps_failing_stops_after_max_attempts(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_FIRST", "9")
    with pytest.raises(RuntimeError, match="failed 3 times"):
        run_case(*batches_at(cfg), tmp_path / "doe_000", cfg)
    assert runs(tmp_path / "doe_000") == 3


def test_a_rejected_input_is_not_retried(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_REJECT", "1")
    with pytest.raises(RuntimeError, match="Serpent rejected doe_000: Input error"):
        run_case(*batches_at(cfg), tmp_path / "doe_000", cfg)
    assert runs(tmp_path / "doe_000") == 1


# --- the evaluator --------------------------------------------------------------

def test_evaluator_returns_the_reactivity_trajectory(cfg, tmp_path):
    ev = SerpentEvaluator(cfg).evaluate(X, tmp_path / "doe_000")
    np.testing.assert_allclose(ev.t, [0] + DAYS)
    k = 1.05 - 1e-5 * ev.t
    np.testing.assert_allclose(ev.y, (k - 1) / k * 1e5, rtol=1e-5)
    np.testing.assert_allclose(ev.std, 0.0005 * k / k**2 * 1e5, rtol=1e-5)
    assert ev.x.keys() == X.keys()
    assert all(abs(ev.x[n] - X[n]) < 1e-9 for n in X)


def test_evaluator_reports_the_design_of_an_adopted_case(cfg, tmp_path):
    """After a restart the proposal can differ from the case on disk: the
    finished case is kept, and the design it was run for is reported."""
    evaluator = SerpentEvaluator(cfg)
    evaluator.evaluate(X, tmp_path / "bo_020")
    proposal = {**X, "R1": 0.0300, "N2": 160.0}
    ev = evaluator.evaluate(proposal, tmp_path / "bo_020")
    assert runs(tmp_path / "bo_020") == 1
    assert ev.x["R1"] == pytest.approx(X["R1"]) and ev.x["N2"] == X["N2"]


def test_a_template_with_an_unknown_placeholder_is_caught_at_start(cfg, tmp_path):
    template = tmp_path / "t.inp"
    template.write_text(cfg["template"].read_text() + "\n% {typo}\n")
    cfg["template"] = template
    with pytest.raises(ValueError, match="unknown placeholder 'typo'"):
        SerpentEvaluator(cfg)


@pytest.mark.parametrize("edit, match", [
    (lambda c: c.update(shedulr="local"), "unknown keys"),
    (lambda c: c["local"].update(ranks=4), r"serpent.local: unknown keys"),
    (lambda c: c["fuel"].pop("count"), "serpent.fuel needs"),
    (lambda c: c.pop("executable"), "executable is required"),
])
def test_section_mistakes_are_reported(cfg, edit, match):
    edit(cfg)
    with pytest.raises(ValueError, match=match):
        check_section(cfg)


# --- a whole campaign -------------------------------------------------------------

# A stand-in whose reactivity depends on the design: the absorber volume
# rendered into the material cards holds k down, and burns out over time.
DESIGN_DEPENDENT_SSS2 = """\
import math, sys
inp = sys.argv[-1]
text = open(inp).read()
vol = sum(float(line.split()[4]) for line in text.splitlines()
          if line.startswith("mat burn_abs_"))
days = [0.0]
for line in text.splitlines():
    if line.strip().startswith("dep daytot"):
        days += [float(v) for v in line.split()[2:]]
with open(inp + "_res.m", "w") as f:
    for d in days:
        k = 1.12 - 4e-5 * d - 0.02 * vol * math.exp(-d / 600)
        f.write(f"ABS_KEFF (idx, [1:   2]) = [  {k:.5E} 0.00050 ];\\n")
        f.write(f"BURN_DAYS (idx, [1:   2]) = [  {d:.5E} 0.00000E+00 ];\\n")
        f.write(f"BURNUP (idx, [1:   2]) = [  {d / 10:.5E} 0.00000E+00 ];\\n")
"""


@pytest.mark.slow
def test_a_campaign_runs_end_to_end_on_the_local_backend(cfg, tmp_path):
    import yaml

    from profilebo.config import load_config
    from profilebo.evaluators import make_evaluator
    from profilebo.optimization.bo_loop import BOLoop

    exe = tmp_path / "sss2_design"
    exe.write_text(f"#!{sys.executable}\n{DESIGN_DEPENDENT_SSS2}")
    exe.chmod(0o755)
    cfg.update(executable=str(exe), template=str(cfg["template"]))
    cfg.pop("rng_seed")
    raw = {"serpent": cfg,
           "objective": {"alpha": 100.0, "target": 1000.0, "window": [5.0, 365.0]},
           "basis": {"n_coeff": 3},
           "bo": {"n_init": 6, "budget": 8, "seed": 3, "doe_parallel": 3, "reproducible": True,
                  "num_restarts": 2, "raw_samples": 32, "mc_samples": 32},
           "paths": {"run_dir": "run"}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    config = load_config(tmp_path / "c.yaml")

    x_best, j_best = BOLoop(config, make_evaluator(config)).run()

    history = np.load(tmp_path / "run" / "history.npz")
    assert len(history["X"]) == 8
    assert set(x_best) == {"R1", "R2", "N1", "N2"} and np.isfinite(j_best)
    for tag in history["tags"]:
        case = tmp_path / "run" / "cases" / str(tag)
        assert "set seed" in (case / INPUT_NAME).read_text()     # bo.reproducible
        assert (case / f"{INPUT_NAME}_res.m").exists() and (case / "job_id").exists()
