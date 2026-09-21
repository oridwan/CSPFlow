"""How a VASP job is divided across MPI ranks: KPAR, NCORE, and how many ranks.

The campaign this was written for shipped `NCORE = 8` in all 5,451 INCARs and
never set `KPAR` at all, while the irreducible k-point count across those jobs
ranged from 1 to 232. A fixed setting cannot be right across that range: with
`KPAR = 1` VASP walks the k-points sequentially, so wall time grows roughly
linearly in `NKPTS` even though k-points are the one part of the calculation
that is embarrassingly parallel.

Three quantities, in the order they must be decided:

    ntasks   how many MPI ranks the job gets at all
    KPAR     how many independent k-point groups those ranks form
    NCORE    how many ranks inside a group cooperate on one band

with the identity

    ntasks = KPAR * NCORE * (number of band groups)

None of these changes the converged answer -- they only decide how the work is
split -- so they belong in `settings_hash` and never in `recipe_id`.


Why KPAR need not divide NKPTS
------------------------------
VASP requires `KPAR` to divide the *rank* count. It does **not** require it to
divide `NKPTS`: k-points are dealt out to the groups as evenly as possible, so
an imperfect split costs `ceil(NKPTS / KPAR)` rounds rather than failing. The
tempting rule `KPAR = gcd(NKPTS, ntasks)` is therefore far too conservative --
measured over the k-point counts this campaign actually produces, it averages a
6.0x speedup against 9.7x for "largest divisor of ntasks not exceeding NKPTS".

The gap is worst exactly where it hurts most. `NKPTS = 25` -- the 5x5x1 grid
that the Nd-Ga-Ni family and most of the repaired cells land on -- has
`gcd(25, 16) = 1`, so the gcd rule would give those structures no k-point
parallelism whatsoever, while the divisor rule gives `KPAR = 16` and a 12.5x
speedup.


Why NKPTS is computed here and not read from VASP
-------------------------------------------------
`vasp_std --dry-run` reports `NKPTS`, but that is one extra job launch per
structure -- 2,746 of them for a campaign this size. cspflow writes the KPOINTS
grid itself and already computes NBANDS, so the only missing input is the
irreducible count, which spglib supplies directly. Validated against 500
`IBZKPT` files VASP itself wrote:

    symprec 1e-5      83.4% exact
    symprec 1e-4      97.4% exact      <- used here
    symprec 1e-3      94.6% exact

Note that matching the INCAR's own `SYMPREC` scores worse (83.6%) than fixing
1e-4: VASP's symmetry tolerance and spglib's do not mean the same thing, so this
is calibration against measurement rather than a principled equivalence.

The residual ~2% split both ways. An *under*-estimate is harmless (KPAR merely
comes out smaller than it could be). An *over*-estimate can push KPAR above the
true NKPTS, which VASP rejects at startup -- so `plan()` keeps a safety margin
and the recipe's retry ladder halves KPAR if VASP complains. That is cheaper
than pessimising the 98% of jobs whose prediction is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# Matches VASP's own irreducible counts far better than 1e-5; see module docstring.
DEFAULT_SYMPREC = 1e-4

# Grids at or below this are left alone: KPAR would buy nothing and the memory
# cost of replicating the charge density per group is pure waste.
KPAR_MIN_KPOINTS = 2

# Ranks each k-group keeps, so there is something left to parallelise the bands
# with. Benchmarked: below this KPAR stops helping and starts hurting -- see
# choose_kpar. It also keeps NCORE >= 2, which VASP asks for unprompted.
MIN_RANKS_PER_KGROUP = 4


class ParallelError(Exception):
    pass


@dataclass(frozen=True)
class ParallelPlan:
    """The decision, with the evidence that produced it."""

    ntasks: int
    kpar: int
    ncore: int
    nkpts: int
    nbands: int
    reason: str = ""

    @property
    def band_groups(self) -> int:
        return self.ntasks // (self.kpar * self.ncore)

    @property
    def ranks_per_kgroup(self) -> int:
        return self.ntasks // self.kpar

    @property
    def kpoint_rounds(self) -> int:
        """How many sequential passes over k-points each group makes."""
        return math.ceil(self.nkpts / self.kpar)

    @property
    def kpar_speedup(self) -> float:
        """Ideal speedup from k-parallelism, accounting for uneven splits."""
        return self.nkpts / self.kpoint_rounds

    def incar_tags(self) -> dict[str, int]:
        """The tags to merge into the INCAR. KPAR is omitted when it is 1 --
        writing `KPAR = 1` is noise, and its absence is VASP's own default."""
        tags: dict[str, int] = {"NCORE": self.ncore}
        if self.kpar > 1:
            tags["KPAR"] = self.kpar
        return tags

    def describe(self) -> str:
        return (
            f"{self.ntasks} ranks = KPAR {self.kpar} x NCORE {self.ncore} "
            f"x {self.band_groups} band group(s); NKPTS {self.nkpts}, "
            f"NBANDS {self.nbands}; k-point speedup {self.kpar_speedup:.1f}x"
            + (f" [{self.reason}]" if self.reason else "")
        )


def divisors(n: int) -> list[int]:
    """Ascending divisors of `n`."""
    if n < 1:
        raise ParallelError(f"divisors() needs a positive integer, got {n}")
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def irreducible_kpoints(
    lattice: Sequence[Sequence[float]],
    frac_coords: Sequence[Sequence[float]],
    numbers: Sequence[int],
    grid: Sequence[int],
    symprec: float = DEFAULT_SYMPREC,
) -> int | None:
    """Irreducible k-points in a Gamma-centred `grid`, or None if spglib is absent.

    Returning None rather than raising is deliberate: the caller falls back to
    the full grid product, which is a valid upper bound and merely gives a more
    conservative KPAR. A missing optional dependency should not stop a campaign.
    """
    try:
        import numpy as np
        import spglib
    except ImportError:
        return None
    try:
        mapping, _ = spglib.get_ir_reciprocal_mesh(
            list(grid),
            (lattice, frac_coords, list(numbers)),
            is_shift=[0, 0, 0],
            symprec=symprec,
        )
        return int(len(np.unique(mapping)))
    except Exception:
        return None


def choose_kpar(nkpts: int, ntasks: int, max_kpar: int | None = None,
                min_ranks_per_kgroup: int = MIN_RANKS_PER_KGROUP) -> int:
    """Largest useful divisor of `ntasks`, not exceeding `nkpts`.

    Not `gcd(nkpts, ntasks)` -- see the module docstring -- but not the largest
    divisor either. Benchmarked on this cluster, one structure per row, every
    other setting held fixed:

        structure   ranks   KPAR=1  KPAR=2  KPAR=4  KPAR=8  KPAR=16
        t1-673          8      45s     31s     27s     28s        -
        t1-762          8      48s     34s     29s     29s        -
        t1-1314         8      38s     31s     25s     21s        -
        t2-867         16     142s    138s    139s    167s     177s

    Two lessons. The gain is real but modest -- 1.6-1.8x at best, not the
    near-linear speedup the ideal `NKPTS / ceil(NKPTS/KPAR)` predicts -- and it
    plateaus by KPAR = 4. Beyond that the larger structure got *worse*, 25%
    slower at KPAR = 16 than at KPAR = 1: once a k-group is down to one or two
    ranks, the band and FFT parallelism that was doing the real work is gone and
    replicating the charge density per group costs more than the k-splitting
    saves.

    So the cap is on ranks per group, not on KPAR itself. `min_ranks_per_kgroup`
    keeps enough ranks inside each group to parallelise the bands, which also
    keeps NCORE >= 2 as VASP asks.
    """
    if nkpts < KPAR_MIN_KPOINTS or ntasks < 2:
        return 1
    cap = nkpts if max_kpar is None else min(nkpts, max_kpar)
    if min_ranks_per_kgroup > 0:
        cap = min(cap, max(1, ntasks // min_ranks_per_kgroup))
    if cap < 1:
        return 1
    return max((d for d in divisors(ntasks) if d <= cap), default=1)


def choose_ncore(ranks_per_kgroup: int, cores_per_socket: int | None = None) -> int:
    """NCORE for a k-group of `ranks_per_kgroup` ranks.

    The starting point is sqrt(ranks) -- the usual balance between band
    parallelism and the FFT communication inside a band. It is then nudged
    toward a divisor of `cores_per_socket` where one is available at equal
    distance, because an NCORE group whose ranks straddle two sockets pays for
    every FFT it does.

    Never returns 1 when the group has ranks to spare. VASP prints, unprompted:

        For optimal performance we recommend to set
          NCORE = 2 up to number-of-cores-per-socket
        The default, NCORE=1 might be grossly inefficient on modern
        multi-core architectures or massively parallel machines.

    which it did for 154 of the 228 jobs in the first repair batch, because
    aggressive KPAR had left only one rank per k-group to divide.
    """
    if ranks_per_kgroup < 2:
        return 1
    cands = [d for d in divisors(ranks_per_kgroup) if d >= 2] or [1]
    target = math.sqrt(ranks_per_kgroup)

    def rank(d: int) -> tuple[float, int]:
        # primary: distance from sqrt; tiebreak: prefer NUMA-friendly divisors
        aligned = 0 if (cores_per_socket and cores_per_socket % d == 0) else 1
        return (abs(d - target), aligned)

    return min(cands, key=rank)


def plan(
    nkpts: int,
    nbands: int,
    ntasks: int,
    cores_per_socket: int | None = None,
    max_kpar: int | None = None,
    min_bands_per_group: int = 8,
    min_ranks_per_kgroup: int = MIN_RANKS_PER_KGROUP,
) -> ParallelPlan:
    """Decide KPAR and NCORE for a job of `ntasks` ranks.

    `min_bands_per_group` guards the far end of aggressive k-parallelism: once
    KPAR and NCORE have eaten the ranks, whatever is left forms the band groups,
    and splitting a few hundred bands across too many groups turns the run into
    communication. When that guard binds, NCORE is raised to absorb ranks rather
    than KPAR being cut, because k-parallelism scales better than band
    parallelism does.
    """
    if ntasks < 1:
        raise ParallelError(f"ntasks must be >= 1, got {ntasks}")
    if nkpts < 1:
        raise ParallelError(f"nkpts must be >= 1, got {nkpts}")

    kpar = choose_kpar(nkpts, ntasks, max_kpar=max_kpar,
                       min_ranks_per_kgroup=min_ranks_per_kgroup)
    per_group = ntasks // kpar
    ncore = choose_ncore(per_group, cores_per_socket=cores_per_socket)

    reason = ""
    groups = per_group // ncore
    if nbands and groups > 1 and nbands / groups < min_bands_per_group:
        # too few bands per group: spend the ranks on NCORE instead
        wanted = max(1, math.ceil(nbands / min_bands_per_group))
        for cand in sorted(divisors(per_group), reverse=True):
            if per_group // cand <= wanted:
                ncore = cand
                break
        reason = f"NCORE raised to keep >={min_bands_per_group} bands per group"

    return ParallelPlan(
        ntasks=ntasks, kpar=kpar, ncore=ncore,
        nkpts=nkpts, nbands=nbands, reason=reason,
    )

# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
#
# Fitted to 132 VASP runs across the CeFeB and CePdGe campaigns, using VASP's
# own "total amount of memory used by VASP MPI-rank0" line as ground truth.
# Range covered: NIONS 6-72, NBANDS 32-536, NKPTS 3-20, KPAR 2-16, and per-rank
# memory 72-589 MB.  R^2 = 0.83, and 0.75 on a held-out third.
#
# WHY THIS EXISTS.  `--mem` was one flat number in the machine profile -- 32G for
# every job, whatever it was doing.  Three CeFeB statics were OOM-killed by it
# while 129 other runs used less than half of it.  The three were not bigger:
# `dft-18-static` has FEWER bands than `dft-88-static` (496 vs 536) and the same
# 68 atoms, and needed 2.25x the memory.
#
# The difference was KPAR, and it is the dominant term below for a physical
# reason: each k-point group holds its OWN copy of the charge density, the local
# potentials and the FFT work arrays.  With 64 ranks, KPAR=8 leaves 8 ranks per
# group instead of 16, so each rank carries twice the replicated data:
#
#     dft-88-static   KPAR=4   261 MB/rank   16.3 GB/node   ran
#     dft-18-static   KPAR=8   589 MB/rank   36.8 GB/node   OOM against 32G
#
# `choose_kpar`'s own benchmark above says the speed gain plateaus by KPAR=4, so
# KPAR=8 bought nothing here and cost 2.25x the memory.  Capping KPAR is
# therefore the cheaper half of the fix; this estimator is the half that still
# works when a structure genuinely is large.
MEM_COEFFS = (21.634, -18.645, 726.761)
MEM_FLOOR_GB = 16
MEM_HEADROOM = 1.5      # 0 of 132 runs under-covered at this factor
MEM_GRANULARITY_GB = 8  # request in multiples of this; schedulers bin anyway


def estimate_memory_mb_per_rank(volume: float, encut: float, nbands: int,
                                kpar: int, ntasks: int) -> float:
    """Per-rank resident memory, in MB, before any headroom.

    `volume * encut**1.5` stands in for the plane-wave count, which is what the
    wavefunction and charge arrays are actually sized by. It predicts the fine
    FFT grid VASP ends up choosing with R^2 = 0.97, so the estimate can be made
    BEFORE the job runs -- which is the whole point, since the grid itself is
    not known until VASP prints it.
    """
    if ntasks <= 0:
        raise ValueError("ntasks must be positive")
    pw = max(volume, 1.0) * max(encut, 1.0) ** 1.5 / 1e6
    share = pw * max(kpar, 1) / ntasks
    a, b, c = MEM_COEFFS
    return a + b * share + c * nbands * share / 1e3


def estimate_memory_gb(volume: float, encut: float, nbands: int, kpar: int,
                       ntasks: int, *, headroom: float = MEM_HEADROOM,
                       floor_gb: int = MEM_FLOOR_GB) -> int:
    """What to ask Slurm for, in whole GB.

    The floor is not politeness. The fit's middle coefficient is negative -- the
    two terms are collinear over the range measured -- so at very small NBANDS
    the expression can trend toward zero or below. That never happens for a real
    campaign structure, but an estimator that can return 0 is one that will
    someday return 0 for something, and a job with no memory does not fail in a
    way anyone enjoys diagnosing.
    """
    mb = estimate_memory_mb_per_rank(volume, encut, nbands, kpar, ntasks)
    gb = mb * ntasks / 1024.0 * headroom
    step = MEM_GRANULARITY_GB
    return max(floor_gb, int(math.ceil(gb / step) * step))
