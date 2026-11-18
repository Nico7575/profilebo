"""Where Serpent runs: a batch scheduler, or the local machine.

The runner talks to a backend through six functions: write_job_script,
submit, job_state, job_name, cancel and wait_for_job. Two backends provide
them:

    slurm.py   sbatch, squeue, sacct, scancel   (HPC clusters)
    local.py   a background process per case    (workstation)

Choose with serpent.scheduler: slurm or local. If the key is absent, the
backend is the one whose section (serpent.slurm or serpent.local) the config
carries.
"""

from __future__ import annotations

from types import ModuleType

BACKENDS = ("slurm", "local")


def backend_name(serpent_cfg: dict) -> str:
    """The backend this config asks for."""
    name = serpent_cfg.get("scheduler")
    if name is None:
        present = [b for b in BACKENDS if b in serpent_cfg]
        if len(present) != 1:
            raise ValueError("set serpent.scheduler to one of "
                             f"{list(BACKENDS)} (found sections: {present})")
        name = present[0]
    name = str(name).lower()
    if name not in BACKENDS:
        raise ValueError(f"serpent.scheduler must be one of {list(BACKENDS)}, got {name!r}")
    return name


def get_backend(serpent_cfg: dict) -> ModuleType:
    """The backend module for this config."""
    if backend_name(serpent_cfg) == "local":
        from profilebo.evaluators.serpent.schedulers import local
        return local
    from profilebo.evaluators.serpent.schedulers import slurm
    return slurm


def job_config(serpent_cfg: dict) -> dict:
    """The backend's own settings (serpent.slurm or serpent.local)."""
    return serpent_cfg.get(backend_name(serpent_cfg)) or {}
