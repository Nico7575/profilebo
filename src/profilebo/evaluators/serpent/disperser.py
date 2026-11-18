"""Drive sss2 -disperse to generate the particle-position (pbed) file.

Serpent's disperser is interactive; it is driven here through a pty.
Particle counts are exact (grow-and-shake until all particles are placed),
and the output file is validated by line count.

Without a seed each call produces a different random particle placement,
which ends up in the observation noise of the GP. With a seed the placement
is fixed: sss2 -disperse ignores the input file, but takes its seed from the
SERPENT_RNG_SEED environment variable (read in InitData, before Disperse
runs).
"""

from __future__ import annotations

import logging
import os
import pty
import select
import subprocess
import termios
import time
from dataclasses import dataclass
from pathlib import Path

from profilebo.evaluators.serpent.batches import Batch

log = logging.getLogger(__name__)

_PROMPTS = (
    "Enter cylinder radius (cm):",
    "Enter cylinder bottom coordinate (cm):",
    "Enter cylinder top coordinate (cm):",
    "Enter number of particles (> 1) or packing fraction (< 1):",
    "Enter particle radius (cm):",
    "Enter particle universe:",
    "More particles? (y/n):",
    "Enter file name:",
    "Use grow and shake algorithm? (y/n):",
    "Enter particle shake factor",
    "Enter particle growth rate",
)


@dataclass
class ParticleSpec:
    count: int
    radius: float
    universe: int


def run_disperser(
    specs: list[ParticleSpec],
    out_file: str | Path,
    serpent_cfg: dict,
    workdir: str | Path = ".",
    timeout: float = 600.0,
    seed: int | None = None,
) -> Path:
    """Place all particle batches in the compact cylinder; return the
    validated pbed file path (absolute). seed fixes the realization."""
    workdir = Path(workdir)
    out_file = Path(out_file)
    cyl = serpent_cfg["cylinder"]
    disp = serpent_cfg.get("disperser", {})
    executable = serpent_cfg.get("executable", "sss2")

    # Build the interactive answer sequence.
    inputs: list[tuple[str, str]] = [
        ("volume type", "2"),  # cylinder
        ("cylinder radius", str(cyl["radius"])),
        ("bottom coordinate", str(cyl["z_min"])),
        ("top coordinate", str(cyl["z_max"])),
    ]
    for i, spec in enumerate(specs):
        inputs += [
            (f"batch {i} count", str(spec.count)),
            (f"batch {i} radius", str(spec.radius)),
            (f"batch {i} universe", str(spec.universe)),
            ("more particles", "y" if i < len(specs) - 1 else "n"),
        ]
    inputs += [
        ("file name", out_file.name),
        ("grow and shake", "y"),
        ("shake factor", str(disp.get("shake_factor", 0.4))),
        ("growth rate", str(disp.get("growth_rate", 0.01))),
    ]

    master, slave = pty.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~termios.ECHO
    termios.tcsetattr(slave, termios.TCSANOW, attrs)

    env = os.environ.copy()
    if seed is not None:
        env["SERPENT_RNG_SEED"] = str(int(seed))
    else:
        env.pop("SERPENT_RNG_SEED", None)

    process = None
    try:
        process = subprocess.Popen(
            [executable, "-disperse"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=workdir,
            preexec_fn=os.setsid,
            env=env,
        )

        def read_until_prompt(deadline: float) -> None:
            """Consume output until any known prompt (or the volume-type
            menu) is seen."""
            buf = ""
            menu_seen = False
            while time.time() < deadline:
                r, _, _ = select.select([master], [], [], 0.2)
                if not r:
                    if menu_seen:          # menu printed, ready for choice
                        return
                    continue
                data = os.read(master, 4096).decode(errors="replace")
                buf += data
                log.debug("disperser <<< %r", data)
                if any(p in buf for p in _PROMPTS):
                    return
                if "parallelepiped" in buf:  # last line of volume-type menu
                    menu_seen = True
            raise TimeoutError("Timed out waiting for disperser prompt")

        deadline = time.time() + timeout
        for label, value in inputs:
            read_until_prompt(deadline)
            log.debug("disperser >>> %s = %s", label, value)
            os.write(master, f"{value}\r".encode())
            time.sleep(0.1)

        # Wait for the pbed file to be complete.
        expected = sum(s.count for s in specs)
        pbed_path = workdir / out_file.name
        while time.time() < deadline:
            if pbed_path.exists():
                lines = pbed_path.read_text().splitlines()
                if len(lines) == expected and len(lines[-1].split()) == 5:
                    log.info("disperser wrote %d particles to %s", expected, pbed_path)
                    return pbed_path.resolve()
            if process.poll() is not None and not pbed_path.exists():
                raise RuntimeError("Disperser exited without producing the pbed file")
            time.sleep(2)
        raise TimeoutError(f"pbed file incomplete after {timeout:.0f}s: {pbed_path}")

    finally:
        os.close(master)
        os.close(slave)
        if process is not None and process.poll() is None:
            process.terminate()


def particle_specs(batches: list[Batch], serpent_cfg: dict) -> list[ParticleSpec]:
    """Fuel particles first, then one entry per absorber batch."""
    fuel = serpent_cfg["fuel"]
    return [ParticleSpec(int(fuel["count"]), float(fuel["radius"]), int(fuel["universe"]))] + [
        ParticleSpec(b.count, b.radius, b.universe) for b in batches]


def disperse_batches(batches: list[Batch], out_file: str | Path, serpent_cfg: dict,
                     workdir=".", seed: int | None = None) -> Path:
    """Place the fuel particles and every absorber batch in one pbed file."""
    return run_disperser(particle_specs(batches, serpent_cfg), out_file, serpent_cfg,
                         workdir=workdir, seed=seed)
