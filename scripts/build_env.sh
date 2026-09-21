#!/usr/bin/env bash
# Build the single cspflow environment: mattergen + mattersim + cspflow.
#
# Why one env and why these pins:
#   mattergen declares `mattersim>=1.1` as a dependency, so one environment is
#   the intended arrangement, not a compromise.  Its pins are hard, though:
#   torch==2.2.1+cu118, numpy<2.0, pytorch-lightning==2.0.6.  The existing
#   `mattersim` env has torch 2.10.0+cu128 and numpy 2.2.6, so it cannot be
#   extended in place -- hence a new env rather than a modification.
#   Python 3.10 because mattersim >=1.2.4 requires 3.12 while mattergen needs
#   numpy<2; 1.2.3 is the newest that supports 3.10.
#
# torch's cu118 wheels bundle their own CUDA runtime, so they run fine against
# the cluster's newer driver (module load cuda/12.8).
# NOTE: no `set -u`.  Conda's activate.d hooks are not written to be safe under
# it -- this machine's julia_activate.sh references an unbound JULIA_DEPOT_PATH
# and aborts the whole script at `conda activate`.
set -eo pipefail

ENV_NAME="${1:-cspflow}"
# The checkout this script lives in -- not a fixed path, so a group member
# building their own env installs from whichever checkout they ran it from.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=3.10
TORCH=2.2.1
CUDA=cu118
LOG_PREFIX="[build_env]"

log() { echo "$LOG_PREFIX $*"; }

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "env '$ENV_NAME' already exists -- reusing it"
else
    log "creating env '$ENV_NAME' (python $PY)"
    conda create -y -n "$ENV_NAME" "python=$PY" pip
fi

conda activate "$ENV_NAME"
log "python: $(python -V)  at $(which python)"

log "step 1/5: torch $TORCH+$CUDA"
pip install --no-cache-dir \
    "torch==${TORCH}+${CUDA}" "torchvision==0.17.1+${CUDA}" "torchaudio==${TORCH}+${CUDA}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

log "step 2/5: torch geometric companions (prebuilt against torch ${TORCH}+${CUDA})"
pip install --no-cache-dir torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"

log "step 3/5: mattergen (pulls mattersim, numpy<2, pytorch-lightning)"
pip install --no-cache-dir mattergen

# mattergen leaves `ase` unpinned (>=3.22.1), so pip takes the newest -- but ASE
# 3.29 moved `full_3x3_to_voigt_6_stress` from ase.constraints to ase.stress,
# and mattersim 1.1.2 still imports it from the old location.  3.27.0 is the
# newest ASE that keeps the symbol where mattersim expects it.
#
# mattersim 1.1.2 rather than a newer one is itself forced: from 1.2.0 onward it
# requires numpy>=2.0 on Python >=3.10, which contradicts mattergen's numpy<2.0.
# So the two packages pin each other, and ASE has to follow mattersim.
log "step 3c/5: pin ase 3.27.0 (mattersim 1.1.2 needs the pre-3.29 location)"
pip install --no-cache-dir "ase==3.27.0"

# mattersim 1.1.2 does `import pkg_resources` in its __version__ module.
# setuptools 81 removed pkg_resources, and pip pulls the newest setuptools by
# default, so mattersim fails to import in a freshly built env. Pinning below
# 81 is the fix; the alternative (patching mattersim) is not ours to make.
log "step 3b/5: pin setuptools<81 (mattersim still imports pkg_resources)"
pip install --no-cache-dir "setuptools<81"

log "step 4/5: cspflow itself"
pip install --no-cache-dir -e "$REPO"

log "step 5/5: verification"
python - <<'PY'
import importlib, sys
ok = True
for name in ("torch", "numpy", "ase", "pymatgen", "mattersim", "mattergen", "cspflow"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:<12} {getattr(m, '__version__', 'present')}")
    except Exception as exc:
        ok = False
        print(f"  {name:<12} FAILED: {type(exc).__name__}: {exc}")
import torch
print(f"  torch.cuda    compiled={torch.version.cuda} available={torch.cuda.is_available()}")
print("  (cuda unavailable on a login node is expected)")
sys.exit(0 if ok else 1)
PY

# The lockfile belongs to the checkout. Someone building from a checkout they
# cannot write to (another member's) gets theirs in $HOME instead of a failed
# script at the very end of a successful build.
LOCK="$REPO/scripts/env.lock.txt"
[ -w "$REPO/scripts" ] || LOCK="$HOME/cspflow-env.lock.txt"
log "writing lockfile"
pip freeze > "$LOCK"
log "wrote $LOCK ($(wc -l < "$LOCK") packages)"

log "done. activate with: conda activate $ENV_NAME"
