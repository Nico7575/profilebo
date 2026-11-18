#!/bin/bash
# Run the profilebo optimizer itself as a SLURM job, so that it keeps running
# after you log out. Use it where tmux and screen are not available, or where
# long processes on the login node are killed.
#
# The optimizer is almost idle: it polls the scheduler once a minute and fits
# the surrogate for a few seconds between evaluations, so one core is enough.
# The simulations it submits are separate jobs. The compute nodes must be
# allowed to run sbatch and squeue (true on most SLURM clusters).
#
# Fill in the partition, account, time limit and the paths below, then submit
# from the folder that holds the config:
#     sbatch optimizer.sbatch
# Watch:  tail -f <run_dir>/campaign.log
# Stop:   scancel <job id>   (the simulations keep running; submitting this
#                             script again resumes and re-attaches to them)
#
#SBATCH --job-name=pbo_optimizer
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=2-00:00:00
#SBATCH --output=optimizer_%j.log

set -u
cd "${SLURM_SUBMIT_DIR:-.}"

# The jobs submitted from here must not inherit this job's resources.
unset SLURM_CPUS_PER_TASK SLURM_MEM_PER_CPU SLURM_NTASKS SLURM_NPROCS

exec /path/to/profilebo/.venv/bin/profilebo run config.yaml
