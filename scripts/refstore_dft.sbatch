#!/bin/bash
# ---------------------------------------------------------------------------
# refstore_dft.sbatch -- relax AND static for one reference structure, in ONE
# job.  Previously these were two separate array jobs with a scheduler round
# trip and a database hand-off between them; a structure could sit for hours
# after its relax finished, waiting to be noticed.
#
# The carry between the two is the whole point:
#     dft_relax/CONTCAR  ->  dft_static/POSCAR
# and the static k-grid is re-derived from that relaxed cell (not reused from
# the relax), which is why the inputs are written here rather than at submit
# time -- the relaxed cell does not exist yet when we submit.
#
# INPUT   $REFSTORE_TASKS   file with one structure folder per line
#         $REFSTORE         path to refstore.py
#         $CSPFLOW_STORE    the store root
# OUTPUT  <folder>/dft_relax/{OUTCAR,CONTCAR,...,VASP_DONE}
#         <folder>/dft_static/{OUTCAR,...,VASP_DONE}
#
# SUBMIT  python scripts/refstore.py submit dft --apply
# ---------------------------------------------------------------------------
#SBATCH --job-name=ref-dft
#SBATCH --partition=Orion,Nebula,Apus
#SBATCH --exclude=str-c[88-97],str-abm1,str-bm5
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=36:00:00

# EXCLUDED NODES -- every machine in Orion/Nebula/Apus without AVX-512.  Our vasp_std
# carries 5,285 AVX-512 instructions, so on any of these it dies with
#     forrtl: severe (168): Program Exception - illegal instruction
# before writing an OUTCAR.  Measured live 2026-09-09 with `lscpu`, one probe
# job per architecture:
#
#     str-c[88-97]  rome        EPYC 7502        NO avx512      <- excluded
#     str-abm1      milan       EPYC (Zen 3)     no avx512      <- excluded
#     str-bm5       broadwell   Xeon             no avx512      <- excluded
#     str-c[1-26,49-66,128-167]  caslake  Xeon Gold 6248R   avx512f ... 
#     str-c[188-207]             genoa    EPYC 9654P        avx512f ...
#     str-c[67-70]               turin    EPYC 9655         avx512f ...
#     str-c[74-87]  (Nebula)     skylake  Xeon Gold 6154    avx512f ...
#     str-c[171-177,218-221] (Apus)  genoa  EPYC 9654P      avx512f ...
#
# Nebula's Skylake is Skylake-SP, which DOES have AVX-512 -- verified rather
# than assumed, because assuming is what caused this in the first place.
#
# Apus contributes 11 genoa nodes and needs NO exclusion: every one of them is
# the same EPYC 9654P already measured on Orion.  The exclude list is kept
# whole anyway, so a node moved between partitions stays covered.
#
# WHY Apus IS SAFE TO NAME HERE, when machine.yaml keeps it a separate role:
# that note is about the cspflow DRIVER, which submits many independent jobs
# and can only bound a partition by counting cores per role.  This is one
# array with `--array=...%<throttle>`, and SLURM enforces that concurrency cap
# itself no matter which partition each task lands on.  Apus has no site QoS,
# so without the throttle an array could take all 1,056 of its cores; with it,
# the ceiling is throttle x 32 cores, wherever those tasks run.
#
# 36:00:00 = 24 h relax + 12 h static, inside Nebula's 2-day cap (Orion and
# Apus both allow 30 days).  ntasks 32 <= Nebula's 36 cores, the smallest of
# the three (Orion 48, Apus 96), so this is schedulable on all three.

set -o pipefail
ulimit -s unlimited || true

# Inherited from the submitting allocation; unset or srun inherits the wrong
# geometry and silently runs on one rank.
export SLURM_EXPORT_ENV=ALL
unset SLURM_NTASKS_PER_NODE

module load intel/mkl/2024.0
module load intel/2024
module load intel-mpi/2021.11
export OMP_NUM_THREADS=1
export I_MPI_FABRICS=shm:ofi
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so.0
eval "$(conda shell.bash hook)"
conda activate cspflow

VASP=/projects/mmi/cspflow-shared/vasp/bin/vasp_std

DIR=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$REFSTORE_TASKS")
[ -n "$DIR" ] && [ -d "$DIR" ] || { echo "no such structure folder: '$DIR'"; exit 1; }
echo "=== $(basename "$DIR") on $(hostname) ==="

clean_stage () {       # $1 = stage directory; keeps only the input files
    # A retry must start from a clean directory.  Otherwise a second attempt
    # that dies early leaves half of the dead run beside half of the live one,
    # and a CHGCAR/WAVECAR written by a crashed run can poison the restart it
    # is meant to accelerate.  The inputs (INCAR KPOINTS POSCAR POTCAR) stay.
    rm -f "$1"/{OUTCAR,OSZICAR,vasprun.xml,CONTCAR,CHG,CHGCAR,WAVECAR,EIGENVAL,\
DOSCAR,PROCAR,PCDAT,XDATCAR,REPORT,IBZKPT,vasp.out,VASP_DONE,vaspout.h5,ML_LOGFILE}
}

reset_geometry () {    # $1 = structure dir -- back to the PRISTINE seed
    # Not POSCAR.seed.  That file is written only if it does not already
    # exist, so it holds whatever geometry the FIRST job to touch this folder
    # started from -- an arbitrary half-relaxed intermediate once the folder
    # has been refreshed or carried.  Measured 2026-09-10: mp-20169-Ce2O3 had
    # POSCAR.seed at volume 193.70 when its true seed was 189.38 (MLIP) and
    # its original cell 255.81.  Resetting a diverged run to that is no reset
    # at all -- it can restore the very geometry that diverged.
    #
    # refstore.py's seed chain (mlip/CONTCAR, POSCAR.orig, dft_relax/POSCAR)
    # is the authority, so ask it.  It rewrites INCAR too, which is why the
    # callers apply their tag edits AFTER calling this.
    python "$REFSTORE" inputs "$1" --stage relax
}

remaining_seconds () { # how much wall clock SLURM will still give this job
    local t d=0
    t=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | head -1 | tr -d ' ')
    case "$t" in ""|UNLIMITED|INVALID|NOT_SET) echo 0; return;; esac
    case "$t" in *-*) d=${t%%-*}; t=${t#*-};; esac
    local h=0 m=0 sec=0
    case "$t" in
        *:*:*) h=${t%%:*}; m=$(echo "$t" | cut -d: -f2); sec=${t##*:};;
        *:*)   m=${t%%:*}; sec=${t##*:};;
        *)     sec=$t;;
    esac
    echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$sec ))
}

scf_per_step () {      # $1 = stage dir -- electronic cycles per ionic step
    # The single number that separates "slow but working" from "will never
    # finish".  Healthy is 6-20.
    local o="$1/OSZICAR" scf ionic
    [ -f "$o" ] || { echo 0; return; }
    scf=$(grep -cE "^(RMM|DAV):" "$o" 2>/dev/null); scf=${scf:-0}
    ionic=$(grep -c " F= " "$o" 2>/dev/null); ionic=${ionic:-0}
    [ "$ionic" -eq 0 ] && ionic=1
    echo $(( scf / ionic ))
}

count_matches () {     # $1 = pattern, $2 = file -- always ONE integer
    local n
    n=$(grep -c "$1" "$2" 2>/dev/null)
    printf '%s' "${n:-0}"
}

relax_converged () {   # $1 = stage directory -- did the RELAX meet EDIFFG?
    # "General timing and accounting" is NOT this test.  VASP writes that on
    # any NORMAL exit, and exhausting NSW without meeting the force criterion
    # is a normal exit.  So a run that stopped at NSW=99 with fmax 0.18 eV/A
    # looks exactly like a converged one to run_vasp, and the carry below then
    # hands that unrelaxed cell to the static.  Measured 2026-09-10:
    # mp-1244911-Fe2O3 and mp-776606-Fe2O3 were both doing precisely that.
    #
    # "reached required accuracy - stopping structural energy minimisation" is
    # the only line VASP writes when EDIFFG is actually satisfied.
    grep -q "reached required accuracy" "$1/OUTCAR" 2>/dev/null
}

run_vasp () {          # $1 = stage directory
    cd "$1" || exit 1
    srun --mpi=pmi2 "$VASP" > vasp.out 2>&1
    if ! grep -q "General timing and accounting" OUTCAR 2>/dev/null; then
        echo "FAILED: $1 wrote no VASP epilogue -- see $1/vasp.out"
        return 1
    fi
    echo done > VASP_DONE
    return 0
}

# --- the retry ladder ------------------------------------------------------
# Three VASP failures are common, well understood, and fixable without changing
# the physics.  Each is retried ONCE, and what was changed is written to
# RETRY.log next to the run so the deviation is never silent.
retry_stage () {                # $1 = stage directory, $2 = "relax" | "static"
    local d="$1" kind="$2"
    run_vasp "$d" && return 0
    local note=""

    if grep -q "ZBRENT" "$d/OUTCAR" 2>/dev/null; then
        # The line minimiser lost its bracket, almost always within meV of the
        # minimum.  VASP's own advice is "copy CONTCAR to POSCAR and continue";
        # we also switch to conjugate gradient, which does not use ZBRENT.
        # IBRION chooses the PATH to the minimum, not the energy at it.
        note="ZBRENT bracketing -> IBRION=2, restarted from CONTCAR"
        [ "$kind" = relax ] && [ -s "$d/CONTCAR" ] && cp -f "$d/CONTCAR" "$d/POSCAR"
        sed -i 's/^IBRION = .*/IBRION = 2/' "$d/INCAR"

    elif grep -qE "POSMAP|RHOSYG|INVGRP|IBZKPT: not all point group" "$d/OUTCAR" 2>/dev/null; then
        # VASP detected a symmetry it can no longer map onto the positions --
        # typically the static step after the relax moved atoms just past
        # SYMPREC.  ISYM=0 switches symmetry off: the k-mesh is no longer folded
        # and the charge density is not symmetrised, so it costs more time and
        # returns the same energy.  It changes no part of the energy scale --
        # not the functional, the POTCARs, ENCUT, or the k-point DENSITY -- so
        # the result stays comparable with every other structure in the store.
        # IBZKPT added 2026-09-11.  Its message is
        #     IBZKPT: not all point group operations associated with the
        #     symmetry operators of the crystalline lattice are members of
        #     the point group of the reciprocal lattice.
        # -- the automatic mesh does not respect the cell's point group, so
        # VASP cannot fold the zone.  Same class and same remedy as the other
        # three, and it was NOT caught: mp-569451-Sb2Pd5 (3x3x1 on a 42-atom
        # cell) died on it, and that phase sits ON the hull of Ce-Pd-Sb, which
        # blocked seven chemical systems of the CePdGe campaign.
        note="symmetry error (POSMAP/RHOSYG/INVGRP/IBZKPT) -> ISYM=0"
        if grep -q "^ISYM" "$d/INCAR"; then
            sed -i 's/^ISYM = .*/ISYM = 0/' "$d/INCAR"
        else
            echo "ISYM = 0" >> "$d/INCAR"
        fi
    elif [ "$(count_matches "WARNING in EDDRMM" "$d/vasp.out")" -gt 20 ]; then
        # ALGO=Fast selects the RMM-DIIS solver, whose subspace diagonalisation
        # (LAPACK ZHEGV) can break down outright.  When it does, the
        # wavefunctions are corrupt and every later ionic step moves atoms on
        # forces computed from garbage -- mp-1215144-AsO3 ran from -71 eV to
        # +4249 eV that way, over 14 h.  ALGO=Normal is blocked Davidson:
        # slower per step, and it does not do this.
        #
        # The geometry must be RESET, not carried: the CONTCAR of a diverged
        # run is the corrupted cell, which is the one thing not worth keeping.
        note="EDDRMM/ZHEGV breakdown -> ALGO=Normal, IBRION=2, geometry reset"
        [ "$kind" = relax ] && reset_geometry "$DIR"
        if grep -q "^ALGO" "$d/INCAR"; then
            sed -i 's/^ALGO = .*/ALGO = Normal/' "$d/INCAR"
        else
            echo "ALGO = Normal" >> "$d/INCAR"
        fi
        sed -i 's/^IBRION = .*/IBRION = 2/' "$d/INCAR"

    elif [ "$(scf_per_step "$d")" -gt 40 ]; then
        # SCF-BOUND, not optimiser-bound.  A healthy ionic step converges its
        # electrons in 6-20 cycles.  Measured 2026-09-11 across the running
        # jobs, the ones that cannot finish need 51-206 cycles per step and so
        # spend 17-548 MINUTES on a single geometry step:
        #
        #   mp-685153-Fe2O3   183 SCF/step x 180 s = 548 min per ionic step
        #   mp-1245084-Fe2O3  122 SCF/step         = 109 min per ionic step
        #   mp-1215144-AsO3   206 SCF/step         = 134 min per ionic step
        #
        # These are magnetic oxides whose charge and magnetisation densities
        # converge on very different length scales, which is exactly what
        # VASP's linear-mixing parameters exist for.  ALGO=Normal alone does
        # not fix it -- mp-685153 already had it.
        #
        # Mixing changes how the SCF REACHES self-consistency, never the
        # self-consistent answer: at convergence the density satisfies the same
        # equations to the same EDIFF.  The one-policy rule is untouched.
        note="SCF-bound ($(scf_per_step "$d") cycles/ionic step) -> ALGO=Normal + linear mixing"
        if grep -q "^ALGO" "$d/INCAR"; then
            sed -i 's/^ALGO = .*/ALGO = Normal/' "$d/INCAR"
        else
            echo "ALGO = Normal" >> "$d/INCAR"
        fi
        for tag in "AMIX 0.2" "BMIX 0.0001" "AMIX_MAG 0.8" "BMIX_MAG 0.0001"; do
            k=${tag%% *}; v=${tag##* }
            if grep -q "^$k " "$d/INCAR"; then
                sed -i "s/^$k = .*/$k = $v/" "$d/INCAR"
            else
                echo "$k = $v" >> "$d/INCAR"
            fi
        done
        [ "$kind" = relax ] && [ -s "$d/CONTCAR" ] && cp -f "$d/CONTCAR" "$d/POSCAR"

    elif grep -q "subroutine BZINTS" "$d/OUTCAR" 2>/dev/null; then
        # ISMEAR=-5 needs at least 4 k-points and VASP counts the IRREDUCIBLE
        # ones.  inputs.py now substitutes Gaussian smearing at input time, so
        # this should not be reachable; it is kept because the failure is fatal
        # and the remedy costs a few meV of smearing, not a policy change.
        note="tetrahedron method fails (<4 irreducible k-points) -> ISMEAR=0"
        sed -i 's/^ISMEAR = .*/ISMEAR = 0/' "$d/INCAR"
        grep -q "^SIGMA" "$d/INCAR" || echo "SIGMA = 0.05" >> "$d/INCAR"

    elif grep -qE "^ *[0-9]+ F= *0?\.[0-9]" "$d/OSZICAR" 2>/dev/null; then
        # A POSITIVE total energy for a periodic solid is not slow convergence,
        # it is a broken one.  Same remedy, same reason to reset the geometry.
        note="positive total energy (diverged) -> ALGO=Normal, IBRION=2, geometry reset"
        [ "$kind" = relax ] && reset_geometry "$DIR"
        if grep -q "^ALGO" "$d/INCAR"; then
            sed -i 's/^ALGO = .*/ALGO = Normal/' "$d/INCAR"
        else
            echo "ALGO = Normal" >> "$d/INCAR"
        fi
        sed -i 's/^IBRION = .*/IBRION = 2/' "$d/INCAR"

    else
        return 1                # not a failure we know how to fix
    fi

    { echo "$(date -Iseconds)  $note"; } >> "$d/RETRY.log"
    echo "retrying $kind: $note"
    clean_stage "$d"          # the failed attempt's output must not survive
    run_vasp "$d"
}

# How long of this job's wall clock the RELAX may have.  The static needs its
# own share, and a relax that has eaten the whole allocation should hand over
# to the next submit rather than leave the structure with no energy at all.
RELAX_BUDGET=${REFSTORE_RELAX_BUDGET:-86400}      # 24 h of the 36 h job
MAX_CONTINUE=${REFSTORE_MAX_CONTINUE:-3}          # NSW-exhaustion restarts

# --- stage 1: relax -------------------------------------------------------
if [ ! -f "$DIR/dft_relax/VASP_DONE" ] || ! relax_converged "$DIR/dft_relax"; then
    # Only write inputs if there are none, so a resubmission continues rather
    # than resetting.  (refstore.py submit removes the marker of any stage that
    # has to be redone, so reaching here means this stage really is unfinished.)
    [ -f "$DIR/dft_relax/INCAR" ] || python "$REFSTORE" inputs "$DIR" --stage relax

    # Restart from the last geometry the previous attempt reached.  A job killed
    # at the walltime has done real work; throwing its CONTCAR away and starting
    # from the MP cell again is how a structure ends up timing out forever.
    # Carry the geometry the previous attempt reached, THEN wipe that attempt.
    # Order matters: CONTCAR is the one thing worth keeping, and it is one of
    # the files clean_stage removes.
    if [ -s "$DIR/dft_relax/CONTCAR" ]; then
        echo "continuing from the previous CONTCAR"
        cp -f "$DIR/dft_relax/CONTCAR" "$DIR/dft_relax/POSCAR"
    fi
    clean_stage "$DIR/dft_relax"
    retry_stage "$DIR/dft_relax" relax || exit 2

    # --- NSW exhaustion is not success ------------------------------------
    # VASP stopping at NSW is a normal exit, so retry_stage is satisfied and
    # VASP_DONE is written -- but the forces are still above EDIFFG and the
    # cell is not relaxed.  Continue it from its own CONTCAR, which is real
    # progress rather than a failure, for as long as the relax budget allows.
    tries=0
    attempt_start=0
    while ! relax_converged "$DIR/dft_relax"; do
        # Do not START an attempt this job cannot FINISH.
        #
        # NSW=99 is one unit of work, and for the hard cells that unit is not
        # small: measured 2026-09-11, mp-1245277-Fe2O3 runs 9.6 min per ionic
        # step, so a single 99-step attempt is 15.9 h -- and four of them, the
        # old ceiling, is 63.5 h against a 36 h wall.  The job would be killed
        # part-way through an attempt every time.  Nothing is lost when that
        # happens (CONTCAR holds the geometry) but the static never runs and
        # the structure needs yet another job.
        #
        # `RELAX_BUDGET` alone could not prevent this: it is only read HERE,
        # between attempts, so an attempt beginning at hour 20 still runs to
        # hour 36.  The real test is whether the time SLURM has left exceeds
        # what the attempt just finished actually cost.
        cost=$(( SECONDS - attempt_start ))
        left=$(remaining_seconds)
        if [ "$tries" -ge "$MAX_CONTINUE" ] || [ "$SECONDS" -ge "$RELAX_BUDGET" ] \
           || { [ "$cost" -gt 0 ] && [ "$left" -gt 0 ] && [ "$left" -lt "$cost" ]; }; then
            rm -f "$DIR/dft_relax/VASP_DONE"   # so the next submit continues it
            echo "relax NOT converged after $tries continuation(s), ${SECONDS}s used"
            [ "$cost" -gt 0 ] && [ "$left" -lt "$cost" ] && \
                echo "stopping early: last attempt took ${cost}s, only ${left}s left"
            echo "NOT running the static: it would be an energy on an unrelaxed cell"
            exit 5
        fi
        attempt_start=$SECONDS
        tries=$((tries + 1))
        echo "relax hit NSW without meeting EDIFFG -- continuation $tries/$MAX_CONTINUE"

        # Switch to conjugate gradient on the FIRST continuation.  A plain
        # restart with the same settings reproduces the same failure, and
        # measured 2026-09-10 over the 24 structures that hit NSW, the failure
        # is usually IBRION=1 oscillating rather than slow progress:
        #
        #   mp-1206082-CeCuGe2   best fmax 0.152 at step 23, then bounced
        #                        between 0.28 and 0.44 for 76 more steps --
        #                        28 of 28 direction changes in the last 30 --
        #                        and finished 0.095 eV ABOVE its own best energy.
        #   mp-1078637-Bi        same shape, finished 0.139 eV above its best.
        #
        # IBRION=1 is RMM-DIIS quasi-Newton: it trusts an approximate Hessian,
        # and with a poor one it can step uphill and never settle.  IBRION=2 is
        # conjugate gradient, which line-searches instead.  It chooses the PATH
        # to the minimum, never the energy at it, so the store's one-policy rule
        # is untouched.
        #
        # Restarting also regenerates the plane-wave basis for the current cell,
        # which is the standard remedy for ISIF=3 Pulay stress -- and two of
        # these moved their cell far enough to need it (mp-20169-Ce2O3 -24.9%,
        # mp-1147691-Fe2O5 +101.7% by volume).
        note="NSW exhausted, continued from CONTCAR (try $tries)"
        if [ "$tries" -eq 1 ] && grep -q "^IBRION = 1" "$DIR/dft_relax/INCAR"; then
            sed -i 's/^IBRION = .*/IBRION = 2/' "$DIR/dft_relax/INCAR"
            note="$note; IBRION 1 -> 2 (conjugate gradient)"
        fi
        { echo "$(date -Iseconds)  $note"; } >> "$DIR/dft_relax/RETRY.log"
        [ -s "$DIR/dft_relax/CONTCAR" ] && cp -f "$DIR/dft_relax/CONTCAR" "$DIR/dft_relax/POSCAR"
        clean_stage "$DIR/dft_relax"
        retry_stage "$DIR/dft_relax" relax || exit 2
    done
    echo "relax converged in $tries continuation(s)"
else
    echo "relax already done and converged, skipping"
fi

# --- the carry: relaxed cell becomes the static input ---------------------
# Guarded, not assumed.  Everything above can exit 5, but a resubmission that
# skips the relax block must not walk into the static on an unrelaxed cell
# either, so the condition is re-tested here rather than inferred.
if ! relax_converged "$DIR/dft_relax"; then
    echo "refusing to run the static: dft_relax never reached required accuracy"
    exit 5
fi

if [ ! -f "$DIR/dft_static/VASP_DONE" ]; then
    clean_stage "$DIR/dft_static"
    python "$REFSTORE" inputs "$DIR" --stage static || exit 3
    retry_stage "$DIR/dft_static" static || exit 4
else
    echo "static already done, skipping"
fi

echo "=== $(basename "$DIR") complete ==="
