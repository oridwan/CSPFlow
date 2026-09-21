#!/bin/bash
#SBATCH --job-name=cspflow-store-xfer
#SBATCH --partition=DTN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=cspflow-store-xfer-%j.log
#
# Copy the cspflow reference store onto the shared group volume, on a node
# built for it.
#
# WHY A DTN AND NOT A LOGIN NODE
#     The first attempt ran from a login shell and moved 5.1 GB in 15 minutes
#     -- about 20 MB/s, which is login-node throttling rather than the
#     filesystem's limit. At that rate 381 GB takes over five hours and holds a
#     login session the whole time. The DTN partition (dtn-s1, dtn-s3) exists
#     for exactly this.
#
# WHY NOT `rsync -a`
#     `-a` implies `-p -o -g`, which preserves the SOURCE's permissions and
#     group. The source is `oridwan:oridwan`, mode 0750, so `-a` produced a
#     destination the rest of the `mmi` group could not read -- the opposite of
#     the point. `--chmod=D2775,F664` forces group-writable directories with
#     setgid, and group-writable files, whatever the source looked like.
#
# RESUMABLE
#     rsync skips files already present with the same size and mtime, so this
#     can be rerun as often as you like. It also FIXES PERMISSIONS on files it
#     skips, so a rerun repairs an earlier `-a` copy without moving the bytes
#     again.
#
# INPUTS
#     $1  source       (default: /scratch/$USER/mp-reference)
#     $2  destination  (default: /projects/mmi/cspflow-shared/store)
#
# OUTPUTS
#     the destination tree, group-writable
#     cspflow-store-xfer-<jobid>.log in the submit directory
#
# RUN
#     sbatch /projects/mmi/Ridwan/cspflow/scripts/store-maintenance/shared_store_transfer.sbatch
#     sbatch .../shared_store_transfer.sbatch /scratch/$USER/other-store /projects/mmi/cspflow-shared/store2

set -euo pipefail

SRC="${1:-/scratch/$USER/mp-reference}"
DST="${2:-/projects/mmi/cspflow-shared/store}"

echo "host        : $(hostname -s)"
echo "source      : $SRC"
echo "destination : $DST"
echo "started     : $(date)"
echo

[ -d "$SRC" ] || { echo "source does not exist: $SRC" >&2; exit 1; }
mkdir -p "$DST"

# -r recurse, -l keep symlinks as symlinks, -t keep mtimes (rsync needs them to
# decide what to skip on a rerun). Deliberately NOT -a: see the header.
#
# `--info=stats2` and NOT `progress2`: the progress bar redraws with carriage
# returns, which is right on a terminal and, in a batch job whose stdout is a
# file, produced a 13 MB log of overwrite frames with the summary buried at the
# end. Nobody watches this run; they read it afterwards.
rsync -rlt \
      --perms --chmod=D2775,F664 \
      --human-readable \
      --info=stats2 \
      "$SRC/" "$DST/"

echo
echo "setting group ownership"
chgrp -R mmi "$DST" 2>/dev/null || echo "  (chgrp skipped: not all files are yours)"

echo
echo "finished    : $(date)"
echo "source      : $(du -sh "$SRC" 2>/dev/null | cut -f1)"
echo "destination : $(du -sh "$DST" 2>/dev/null | cut -f1)"
echo
echo "Verify before deleting the source:"
echo "  diff <(cd $SRC && find . -type f | sort) <(cd $DST && find . -type f | sort) | head"
