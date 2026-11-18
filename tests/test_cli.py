"""The profilebo command: arguments, and a campaign run, stopped and resumed
as a user would (a real process and a real Ctrl-C)."""

import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml

from profilebo import __version__
from profilebo.cli import main
from conftest import EXAMPLES


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_command_is_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def _profilebo(*args, **kw):
    return subprocess.Popen([sys.executable, "-m", "profilebo.cli", *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


@pytest.mark.slow
def test_run_stop_with_ctrl_c_and_resume(tmp_path):
    raw = yaml.safe_load((EXAMPLES / "synthetic.yaml").read_text())
    raw["bo"].update(n_init=6, budget=9, num_restarts=2, raw_samples=32, mc_samples=32)
    raw["paths"]["run_dir"] = str(tmp_path / "run")
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(raw))
    history = tmp_path / "run" / "history.npz"

    # Stop it as soon as the initial design is archived.
    proc = _profilebo("run", str(config))
    deadline = time.time() + 300
    while not (history.exists() and len(np.load(history)["tags"]) >= 6):
        assert proc.poll() is None and time.time() < deadline
        time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 130, out
    assert "Interrupted" in out

    proc = _profilebo("run", str(config))
    out, _ = proc.communicate(timeout=600)
    assert proc.returncode == 0, out
    assert "Resumed" in out and "Best design" in out
    assert len(np.load(history)["tags"]) == 9
    log = (tmp_path / "run" / "campaign.log").read_text()
    assert log.count(f"profilebo {__version__}") == 2           # both runs logged
