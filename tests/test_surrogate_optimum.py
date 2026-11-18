"""Optimization on the surrogate alone: result, reproducibility, read-only,
objective overrides, the saved prediction and the command.

The port was checked bit for bit against the original research script on a
full-core campaign of the paper (three objectives, run with the same library
versions); that check needs the original code and archive, so it is not here.
"""

from dataclasses import replace

import numpy as np
import pytest
import yaml

from profilebo.analysis.report import load_campaign
from profilebo.cli import main
from profilebo.optimization import surrogate_optimum
from profilebo.optimization.surrogate_optimum import (load_prediction, optimize_on_surrogate,
                                                      with_objective)
from profilebo.parameters import Parameter

pytestmark = pytest.mark.slow          # needs a finished campaign


@pytest.fixture
def campaign(synthetic_campaign):
    return synthetic_campaign[0]


@pytest.fixture(autouse=True)
def cheap(monkeypatch):
    # These tests check bookkeeping, not the quality of the optimum.
    monkeypatch.setattr(surrogate_optimum, "NUM_RESTARTS", 2)
    monkeypatch.setattr(surrogate_optimum, "RAW_SAMPLES", 64)
    monkeypatch.setattr(surrogate_optimum, "ACQ_SAMPLES", 64)
    monkeypatch.setattr(surrogate_optimum, "PREDICT_SAMPLES", 256)


def _snapshot(run_dir):
    return {p: p.stat().st_mtime_ns for p in run_dir.rglob("*")}


def test_optimum(campaign):
    before = _snapshot(campaign.run_dir)
    opt = optimize_on_surrogate(campaign)
    assert _snapshot(campaign.run_dir) == before            # read only

    lo, hi = campaign.bounds_array
    assert np.all(opt.x >= lo) and np.all(opt.x <= hi)
    assert opt.x[2] == round(opt.x[2]) and opt.x[3] == round(opt.x[3])
    assert opt.J_p10 <= opt.J_p50 <= opt.J_p90
    assert np.all(opt.y_p10 <= opt.y) and np.all(opt.y <= opt.y_p90)

    c = load_campaign(campaign)
    assert opt.best_tag == c.tags[c.best] and opt.best_J == pytest.approx(c.J[c.best])
    X = (c.X - opt.x) / (hi - lo)
    assert opt.distance == pytest.approx(np.min(np.linalg.norm(X, axis=1)))
    assert opt.n_evals == len(c.tags)


def test_reproducible(campaign):
    a, b = optimize_on_surrogate(campaign), optimize_on_surrogate(campaign)
    assert a.seed is not None and a.seed == b.seed
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y) and a.J_p50 == b.J_p50


def test_another_objective(campaign):
    base = optimize_on_surrogate(campaign)
    other = optimize_on_surrogate(with_objective(campaign, target=1500.0, alpha=10.0))
    assert other.seed != base.seed                          # the seed follows the target
    assert other.best_J != base.best_J
    assert with_objective(campaign, target=None) is campaign


def test_target_from_a_file(campaign, tmp_path):
    t = load_campaign(campaign).t
    profile = tmp_path / "ramp.txt"
    np.savetxt(profile, np.column_stack([t, np.linspace(2000.0, 800.0, t.size)]))
    opt = optimize_on_surrogate(with_objective(campaign, target=profile))
    assert np.isfinite(opt.J_p50)


def test_box_wider_than_the_archive_is_reported(campaign):
    wide = [Parameter(p.name, p.lower, p.upper * 3, p.integer) if p.name == "N1" else p
            for p in campaign.parameters]
    flagged = lambda cfg: {w.split(":")[0] for w in optimize_on_surrogate(cfg).extrapolation}
    assert flagged(replace(campaign, parameters=wide)) == flagged(campaign) | {"N1"}
    assert "N1" not in flagged(campaign)


def test_no_surrogate_yet(synthetic_config, tmp_path):
    cfg = synthetic_config(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="no surrogate yet"):
        optimize_on_surrogate(cfg)
    assert not cfg.run_dir.exists()


def test_save_and_draw_in_the_report(campaign, tmp_path):
    opt = optimize_on_surrogate(campaign)
    opt.save(tmp_path / "p" / "pred.npz", {"note": "test"})
    pred = load_prediction(tmp_path / "p" / "pred.npz")
    assert np.array_equal(pred["x"], opt.x) and pred["J_p50"] == opt.J_p50
    assert pred["meta"] == {"note": "test", "seed": opt.seed}

    from profilebo.analysis.report import write_report
    from profilebo.evaluators import AXIS_LABELS
    labels = AXIS_LABELS["synthetic"]
    write_report(campaign, labels, tmp_path / "r", prediction=tmp_path / "p" / "pred.npz")

    np.savez(tmp_path / "other.npz", **{**{k: pred[k] for k in pred if k != "meta"},
                                        "t": pred["t"][:-1], "meta": "{}"})
    with pytest.raises(ValueError, match="another grid"):
        write_report(campaign, labels, tmp_path / "r", prediction=tmp_path / "other.npz")


def test_command(campaign, tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump({
        "synthetic": {"bounds": {p.name: [p.lower, p.upper] for p in campaign.parameters}},
        "objective": {"alpha": 100.0, "target": 1000.0, "window": [5.0, 1825.0]},
        "bo": {"seed": campaign.bo.seed, "reproducible": True},
        "paths": {"run_dir": str(campaign.run_dir)},
    }))
    main(["surrogate", str(config), "--target", "1200", "--time-weighted",
          "--save", str(tmp_path / "pred.npz")])
    out = capsys.readouterr().out
    assert "surrogate optimum, not simulated" in out and "target 1200.0" in out
    assert "time-weighted" in out and (tmp_path / "pred.npz").exists()

    with pytest.raises(SystemExit) as exc:
        main(["surrogate", str(config), "--target", "no_such_file.txt"])
    assert exc.value.code == 2
