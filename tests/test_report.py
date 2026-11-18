"""The campaign report: summary, figures, and that its J is the optimizer's J."""

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytestmark = pytest.mark.slow          # most tests share one full campaign

from profilebo.analysis.report import FIGURES, contributions, load_campaign, write_report
from profilebo.cli import main
from profilebo.config import ObjectiveConfig
from profilebo.evaluators import AXIS_LABELS
from profilebo.evaluators.synthetic import SyntheticEvaluator
from profilebo.optimization.scalar_baseline import ScalarBOLoop

LABELS = AXIS_LABELS["synthetic"]


@pytest.fixture
def campaign(synthetic_campaign):
    return synthetic_campaign


def test_j_and_contributions_match_the_optimizer(campaign):
    cfg, j_best = campaign
    c = load_campaign(cfg)
    assert c.J[c.best] == pytest.approx(j_best, rel=1e-12)
    assert np.all(c.sigma_J > 0)
    for i in range(len(c.J)):
        assert contributions(c, cfg, i).sum() == pytest.approx(c.J[i], rel=1e-10)


def test_contributions_with_time_weights(campaign):
    cfg, _ = campaign
    from dataclasses import replace
    weighted = replace(cfg, objective=ObjectiveConfig(alpha=100.0, target=1000.0,
                                                      window=(5.0, 1825.0),
                                                      time_weighted=True))
    c = load_campaign(weighted)
    assert contributions(c, weighted, 0).sum() == pytest.approx(c.J[0], rel=1e-10)


def test_report_writes_summary_and_figures(campaign, tmp_path):
    cfg, _ = campaign
    c = load_campaign(cfg)
    text, written = write_report(cfg, LABELS, tmp_path, show=[c.tags[0]], gif=True)
    names = {p.name for p in written}
    assert names == {"summary.txt", "convergence.gif", *(f"{f}.png" for f in FIGURES)}
    assert all(p.stat().st_size > 0 for p in written)
    assert f"best           {c.tags[c.best]}" in text
    assert "finished: budget reached" in text
    assert (tmp_path / "summary.txt").read_text().strip() == text


def test_report_is_read_only(campaign, tmp_path):
    cfg, _ = campaign
    before = {p: p.stat().st_mtime_ns for p in cfg.run_dir.rglob("*")}
    write_report(cfg, LABELS, tmp_path)
    assert {p: p.stat().st_mtime_ns for p in cfg.run_dir.rglob("*")} == before


def test_unknown_tag_is_refused_before_writing(campaign, tmp_path):
    cfg, _ = campaign
    with pytest.raises(ValueError, match="no evaluation 'bo_999'"):
        write_report(cfg, LABELS, tmp_path / "out", show=["bo_999"])
    assert not (tmp_path / "out").exists()


def test_no_results_yet(synthetic_config, tmp_path):
    with pytest.raises(FileNotFoundError, match="no results yet"):
        write_report(synthetic_config(tmp_path / "empty"), LABELS, tmp_path)


def test_flatness_objective_has_no_contribution_figure(campaign, tmp_path):
    cfg, _ = campaign
    from dataclasses import replace
    flat = replace(cfg, objective=ObjectiveConfig(alpha=100.0, target=1000.0,
                                                  window=(5.0, 1825.0), flatness_beta=0.5))
    _, written = write_report(flat, LABELS, tmp_path)
    assert "contributions.png" not in {p.name for p in written}


def test_report_of_a_scalar_baseline_campaign(synthetic_config, tmp_path):
    cfg = synthetic_config(tmp_path / "scalar", n_init=5, budget=7)
    ScalarBOLoop(cfg, SyntheticEvaluator()).run()
    text, written = write_report(cfg, LABELS, tmp_path / "out")
    assert "scalar baseline" in text and "log EI" in text
    assert len(written) == 1 + len(FIGURES)


def test_report_of_an_unfinished_campaign(campaign, tmp_path):
    cfg, _ = campaign
    from dataclasses import replace
    longer = replace(cfg, bo=replace(cfg.bo, budget=cfg.bo.budget + 5))
    text, _ = write_report(longer, LABELS, tmp_path)
    assert "not finished" in text


def test_command(campaign, tmp_path, capsys):
    cfg, _ = campaign
    config = tmp_path / "c.yaml"
    config.write_text(json.dumps({
        "synthetic": {"bounds": {p.name: [p.lower, p.upper] for p in cfg.parameters}},
        "objective": {"alpha": 100.0, "target": 1000.0, "window": [5.0, 1825.0]},
        "paths": {"run_dir": str(cfg.run_dir)},
    }))
    main(["report", str(config), "--out", str(tmp_path / "r"), "--format", "pdf"])
    out = capsys.readouterr().out
    assert "best" in out and (tmp_path / "r" / "convergence.pdf").exists()

    with pytest.raises(SystemExit, match="no evaluation 'nope'"):
        main(["report", str(config), "--out", str(tmp_path / "r"), "--show", "nope"])
