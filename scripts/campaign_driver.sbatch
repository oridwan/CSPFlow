#!/bin/bash
#SBATCH --job-name=csp-driver
#SBATCH --partition=Orion,Apus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=30-00:00:00
#SBATCH --output=csp-driver-%j.log
#
# Runs a campaign's driver as its own batch job, so the pipeline keeps
# reconciling and submitting after you close your terminal.
#
# WHY THIS EXISTS
#   The Phase A slice is "--through reference".  It used to be
#   "--through calibrate"; that stage was removed (D126) and the slice is now a
#   hard error -- "unknown stage 'calibrate'" -- so a driver started with the
#   old default died in the first second.
#
#   `csp run` without --watch stops as soon as nothing is left to DO -- which
#   happens the moment everything is in flight.  Finished jobs then sit on disk
#   uncollected.  `--watch` fixes that but ties the pipeline to a login shell.
#   A nohup'd driver is worse: D111 and D112 record two of them dying with the
#   Open OnDemand session whose cgroup they were started in, once CANCELLED and
#   once TIMEOUT.  A batch job has its own allocation and its own lifetime.
#
#   AND IT EXITS WHEN THE SLICE IS FINISHED.  `--watch` used to mean forever, so
#   this job held a node for up to 7 days after its work was done -- the CeFeB
#   Phase A driver sat printing "cycle 5 ... cycle 6 ..." with every stage at
#   pending 0.  `--stop-when-done` (the csp default) ends the run after two
#   consecutive cycles with nothing pending and nothing in flight.  Pass
#   "--watch-forever" in $2 for a campaign you intend to keep feeding.
#
# WHY 8 CPUs FOR A LOOP THAT MOSTLY WAITS
#   Most of the driver's time is sleeping between cycles, but its IN-PROCESS
#   stages are real work on this node, not submitted elsewhere: stage 0's
#   duplicate detection compares structures pairwise inside each formula group,
#   which for 232 CePdGe seeds is 739 `StructureMatcher` comparisons.  Those
#   parallelise, and with --cpus-per-task=1 they could not.  The pool sizes
#   itself from `sched_getaffinity`, so it uses exactly what is allocated here
#   and nothing more.
#
# A campaign is PAUSED by creating a PAUSED file in its folder, and resumed by
# removing it.  The file's text is printed into the log so the reason travels
# with the pause.
#
# INPUTS
#   $1  campaign folder            (default: the directory you sbatch from)
#   $2  stage slice                (default: "--through reference" = Phase A)
#                                  use "--from dft" for Phase B, "" for all
#   $3  seconds between cycles     (default: 120)
#
# OUTPUTS
#   csp-driver-<jobid>.log in the submit directory
#   the campaign's own campaign.db and status.json, updated every cycle
#
# RUN
#   cd <your campaign folder>
#   sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch
#   sbatch .../campaign_driver.sbatch . "--from filter" 300   # Phase B, slower
# sbatch --time=2-00:00:00 --partition=Orion,Apus,Nebula --signal=B:TERM@300 ../../scripts/campaign_driver.sbatch . "--from filter" 30
# sbatch  --signal=B:TERM@300 ../../scripts/campaign_driver.sbatch . "--from DFT" 30

# ---------------------------------------------------------------------------
set -eo pipefail

CAMPAIGN="$(cd "${1:-$SLURM_SUBMIT_DIR}" && pwd)"
# `${2-default}` WITHOUT the colon, deliberately.
#
# With `${2:-default}` an explicitly-passed empty string is treated as unset, so
# `sbatch ... . "" 60` -- which this script's own INPUTS section documents as
# "run every stage" -- silently ran `--through reference` instead and stopped at
# Phase A. The colonless form distinguishes "not given" (use the default) from
# "given as empty" (the whole funnel), which is what both the docs and line 144's
# `${SLICE:-<every stage>}` already assumed.
SLICE="${2---through reference}"
INTERVAL="${3:-120}"

if [ ! -f "$CAMPAIGN/campaign.yaml" ]; then
    echo "no campaign.yaml in $CAMPAIGN" >&2
    exit 1
fi
if [ -f "$CAMPAIGN/PAUSED" ]; then
    echo "campaign is PAUSED -- not starting a driver."
    echo "--- $CAMPAIGN/PAUSED ---"; cat "$CAMPAIGN/PAUSED"
    echo "--- remove that file to resume ---"
    exit 0
fi

# This job's own SLURM_* variables would otherwise be inherited by every job the
# driver submits.  The generated scripts carry their own #SBATCH directives, but
# a leaked SLURM_EXPORT_ENV=NONE from a parent is what broke the first wave.
export SLURM_EXPORT_ENV=ALL
unset SLURM_NTASKS_PER_NODE SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU
unset SLURM_CPUS_PER_TASK SLURM_NNODES SLURM_NTASKS

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cspflow
# Unbuffered, so the log shows each cycle as it happens. Buffered, a slow cycle's
# report sat in an 8 KB block for hours -- nine of them on RE-magnets-CHGNet,
# while only the (unbuffered) warnings reached the log (D144).
export PYTHONUNBUFFERED=1
# `${VAR:-default}` so the shared tree is the DEFAULT, not a mandate: exporting
# a different value before `sbatch` now wins, because SLURM_EXPORT_ENV=ALL above
# carries your shell's environment into the job. That matters while the shared
# copy is still filling, and for anyone pointing at a store of their own.
export CSPFLOW_REFERENCE=${CSPFLOW_REFERENCE:-/projects/mmi/cspflow-shared/reference}

# THE STORE.  Both hulls are read out of this folder now -- the MatterSim one
# that filters and the DFT one that ranks -- so this is the single most
# important variable here.  It is also the authority on the DFT policy: a
# campaign that disagrees with $CSPFLOW_STORE/settings.yaml gets a plausible
# hull on a mixed scale, which is worse than an empty one.  `csp doctor`
# checks it and prints the hash of both sides.
export CSPFLOW_STORE=${CSPFLOW_STORE:-/projects/mmi/cspflow-shared/store}

# Only `csp reference build` and the legacy collectors read this.  It pointed at
# `mp-reference/final`, which has not existed since the store was migrated to
# `structures/` -- harmless on the campaign path, wrong everywhere else.
export CSPFLOW_STRUCTURE_DIR=${CSPFLOW_STRUCTURE_DIR:-$CSPFLOW_STORE/structures}

# campaign_db is resolved relative to the campaign root, so start there or the
# driver opens an empty database under the submit directory and reports 0 of
# everything.
cd "$CAMPAIGN"

# --- name this job after what it actually is --------------------------------
#
# `#SBATCH --job-name` is read when the job is SUBMITTED, before any of this
# script runs, so it cannot know which campaign or which slice it got -- both
# arrive as arguments.  Every driver therefore showed up in squeue as
# "csp-driver", and two of them on one campaign were indistinguishable:
#
#     26930224  Apus  csp-driv  oridwan  R  13:27:48  str-c221
#     26930205  Apus  csp-driv  oridwan  R  13:38:19  str-c221
#
# So the job renames itself now that it knows.  squeue truncates NAME to about
# eight characters by default, so the phase is a three-letter suffix and the
# campaign name leads -- `squeue -o "%.18j"` shows it in full.
case "$SLICE" in
    *"--through reference"*|*"--through dedup"*|*"--through screen"*) PHASE=pre ;;
    *"--from filter"*|*"--from dft"*)                                 PHASE=dft ;;
    *"--only "*)                                                      PHASE=$(echo "$SLICE" | sed 's/.*--only *//;s/ .*//' | cut -c1-4) ;;
    "")                                                               PHASE=all ;;
    *)                                                                PHASE=run ;;
esac
JOBNAME="$(basename "$CAMPAIGN")-$PHASE"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Never fatal: a name is a convenience, and `set -e` is on.
    scontrol update "JobId=$SLURM_JOB_ID" "JobName=$JOBNAME" 2>/dev/null || \
        echo "  (could not rename the job; it stays 'csp-driver')"
    # A log named by job id is hard to match to a phase after the fact.
    ln -sfn "csp-driver-${SLURM_JOB_ID}.log" "$CAMPAIGN/driver-${PHASE}.log" 2>/dev/null || true
fi

echo "driver '$JOBNAME' for $(basename "$CAMPAIGN") on $(hostname) at $(date)"
echo "  campaign  $CAMPAIGN"
echo "  slice     ${SLICE:-<every stage>}"
echo "  interval  ${INTERVAL}s"
echo "  job       ${SLURM_JOB_ID:-<no SLURM_JOB_ID>}  log driver-${PHASE}.log"
exec csp run -c "$CAMPAIGN/campaign.yaml" $SLICE --watch --interval "$INTERVAL"
