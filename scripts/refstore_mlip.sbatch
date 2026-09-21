#!/bin/bash
# ---------------------------------------------------------------------------
# refstore_mlip.sbatch -- MatterSim relaxation on GPU, into <folder>/mlip/.
#
# Runs separately from the DFT because it wants a different machine (one GPU,
# one core, minutes) and because its output is the DFT's *input*: the
# MLIP-relaxed cell is a much better starting geometry than MP's own, so the
# DFT relax converges in fewer ionic steps.
#
# Chunked, not one-task-per-structure: loading the MatterSim model costs about
# 30 s, so a task relaxing a single small cell would spend its life importing.
#
# INPUT   $REFSTORE_TASKS   file with one structure folder per line
#         $REFSTORE_CHUNK   structures per array task (default 50)
#         $REFSTORE         path to refstore.py
# OUTPUT  <folder>/mlip/{CONTCAR,relaxed.cif,mlip.json}
#
# SUBMIT  python scripts/refstore.py submit mlip --apply
# ---------------------------------------------------------------------------
#SBATCH --job-name=ref-mlip
#SBATCH --partition=GPU,Nebula_GPU,Hydrus
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# THREE GPU PARTITIONS, because each carries a different per-user GPU cap and
# they overlap in hardware.  Verified live 2026-09-09 with sacctmgr/sinfo:
#
#   partition    nodes  QoS         GPUs/user  MaxTime
#   GPU            14   str_gpu        12      30 days   A100 H200 A40 L40S V100
#   Nebula_GPU      8   nebula_gpu      8       2 days   L40S H100 A100 TitanV
#   Hydrus          1   (none)     uncapped    30 days   H100 x4
#
# Hydrus adds NO new hardware: its only node, str-gpu26, is also in Nebula_GPU.
# It is named anyway because it carries no QoS, so work placed there does not
# count against the nebula_gpu cap of 8 -- the same reason Apus is named on the
# CPU side.  This job asks 08:00:00, inside Nebula_GPU's 2-day limit, and a
# bare `gpu:1` with no model, so every card above is eligible.

# No CUDA module on purpose: this cluster's driver is 570.195.03 (CUDA 12.8)
# and the cspflow env's torch ships its own runtime, so torch.cuda.is_available()
# is True with nothing loaded.  `cuda/11.8` no longer exists here, and a failed
# `module load` still exits 0 -- the job would then die at the next command
# with no useful message.

set -o pipefail
export SLURM_EXPORT_ENV=ALL
eval "$(conda shell.bash hook)"
conda activate cspflow

CHUNK=${REFSTORE_CHUNK:-50}
START=$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK ))

echo "=== task ${SLURM_ARRAY_TASK_ID:-0}: structures ${START}..$((START+CHUNK-1)) on $(hostname) ==="
python "$REFSTORE" mlip --tasks "$REFSTORE_TASKS" --start "$START" --count "$CHUNK"
