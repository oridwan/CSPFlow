#!/bin/bash
# ---------------------------------------------------------------------------
# round.sbatch -- one screening round of the loop, on a GPU node.
#
# WHY A GPU AND NOT THE LOGIN NODE
#   Both models here are torch. Measured on the login node's CPU, one 68-atom
#   Ce2Fe14B costs ~24 s in MatterSim (fmax 0.05) plus ~3 s in CHGNet; at the
#   engine's default fmax 0.01 it was ~5 MINUTES per structure. A 60-seed round
#   is therefore hours of CPU, on a shared interactive node, for work that a
#   single GPU finishes in minutes. Screening belongs on the GPU partition.
#
#   This cluster's driver is 570.195.03 (CUDA 12.8) and the cspflow env's
#   torch 2.2.1+cu118 ships its own runtime, so torch.cuda.is_available() is
#   True with NO module loaded. Do not `module load cuda` -- cuda/11.8 no longer
#   exists here, a failed `module load` still exits 0, and the job would then
#   die at the next command with no useful message.
#
# THREE GPU PARTITIONS, for the same reason refstore_mlip.sbatch names three:
#   GPU (12 GPUs/user, 30 d) | Nebula_GPU (8/user, 2 d) | Hydrus (uncapped)
#   Hydrus adds no new hardware -- str-gpu26 is also in Nebula_GPU -- but it
#   carries no QoS, so work placed there does not count against the cap.
#   A bare `gpu:1` with no model keeps every card eligible.
#
# CHUNKED, NOT ONE TASK PER STRUCTURE
#   Loading MatterSim and CHGNet costs ~30 s. A task screening one structure
#   would spend most of its life importing. LOOP_CHUNK structures per task.
#
# INPUT   $LOOP_DIR    the loop directory (holds loop.yaml and archive.csv)
#         $LOOP_ROUND  round number
#         $LOOP_SEEDS  glob of seeds to screen
#         $LOOP_CHUNK  structures per array task (default 40)
#         $LOOP_ARGS   extra flags for round.py (mask-elements, keep-dd, ...)
# OUTPUT  <loop>/round-NN/scored.<shard>.csv, pareto.<shard>.csv, digest.<shard>.md
#
# SUBMIT  python scripts/loop/orchestrate.py screen --loop <dir>
# ---------------------------------------------------------------------------
#SBATCH --job-name=csp-loop-screen
#SBATCH --partition=GPU,Nebula_GPU,Hydrus
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -o pipefail
export SLURM_EXPORT_ENV=ALL
# One thread per task: both engines are GPU-bound here, and letting torch spawn
# 4 OMP threads per process only fights the other array tasks for the same node.
export OMP_NUM_THREADS=1

eval "$(conda shell.bash hook)"
conda activate cspflow

CHUNK=${LOOP_CHUNK:-40}
TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$(( TASK * CHUNK ))
# $LOOP_SCRIPTS is set by orchestrate.py, which knows exactly where round.py
# lives. Deriving it at runtime does NOT work: SLURM copies the batch script to
# /var/spool/slurm/slurmd/job<N>/, so `dirname "$0"` points there and the job
# dies with "can't open file .../round.py". Seen live on array task 2 of 26946444.
HERE=${LOOP_SCRIPTS:-$(dirname "$0")}

echo "=== loop screen: round ${LOOP_ROUND}, task ${TASK}, seeds ${START}..$((START+CHUNK-1))"
echo "=== node $(hostname)"
python -c "import torch; print('=== cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null

python "$HERE/round.py" "$LOOP_SEEDS" \
  --archive "$LOOP_DIR" --round "$LOOP_ROUND" \
  --start "$START" --count "$CHUNK" --shard "t${TASK}" \
  $LOOP_ARGS
