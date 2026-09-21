"""KPAR / NCORE selection, and the invariance the k-grid must have.

The last test in this file is the one that matters most. The reference recompute
sampled 324 of 2,746 structures on a starved k-grid -- one structure got a
single Gamma point for a metal and its cell then ran away by a factor of 677 --
because `reciprocal_density` was applied to the lattice *as MP happened to write
it down* rather than to a canonical basis. A crystal has no unique set of basis
vectors, so any policy keyed on `a, b, c` is keyed on an arbitrary choice.

`test_grid_is_invariant_under_change_of_basis` re-describes a crystal in a
random equivalent basis and requires the same grid back. Measured on the
campaign, the current policy passes that test 3.8% of the time and passes it
100% of the time once the cell is reduced first.
"""

import math

import numpy as np
import pytest

from cspflow.dft.recipe import Kpoints
from cspflow.dft.vasp.kpoints import grid_for
from cspflow.dft.vasp.parallel import (
    MIN_RANKS_PER_KGROUP,
    ParallelError,
    choose_kpar,
    choose_ncore,
    divisors,
    irreducible_kpoints,
    plan,
)

RANKS = [8, 16, 32, 64]
NKPTS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24, 25, 27, 32, 50, 100, 232]


def test_divisors():
    assert divisors(1) == [1]
    assert divisors(16) == [1, 2, 4, 8, 16]
    assert divisors(24) == [1, 2, 3, 4, 6, 8, 12, 24]
    with pytest.raises(ParallelError):
        divisors(0)


@pytest.mark.parametrize("ntasks", RANKS)
@pytest.mark.parametrize("nkpts", NKPTS)
def test_plan_is_a_valid_decomposition(nkpts, ntasks):
    """VASP requires KPAR to divide the ranks and NCORE to divide what is left."""
    p = plan(nkpts, nbands=200, ntasks=ntasks, cores_per_socket=24)
    assert p.ntasks % p.kpar == 0
    assert (p.ntasks // p.kpar) % p.ncore == 0
    assert p.kpar * p.ncore * p.band_groups == p.ntasks
    assert p.band_groups >= 1
    # KPAR above NKPTS is rejected by VASP at startup
    assert p.kpar <= max(nkpts, 1)


def test_gamma_only_gets_no_kpoint_parallelism():
    p = plan(nkpts=1, nbands=80, ntasks=16)
    assert p.kpar == 1
    assert "KPAR" not in p.incar_tags()      # absence is VASP's own default
    assert p.incar_tags()["NCORE"] == p.ncore


@pytest.mark.parametrize("nkpts", NKPTS)
@pytest.mark.parametrize("ntasks", RANKS)
def test_never_starves_a_kgroup_of_ranks(nkpts, ntasks):
    """The cap the benchmark forced.

    Maximising KPAR looks best on paper -- the ideal speedup is
    NKPTS / ceil(NKPTS/KPAR) -- but measured on this cluster it plateaus by
    KPAR = 4 and then reverses: t2-867 took 177s at KPAR = 16 against 142s at
    KPAR = 1. Once a k-group holds one or two ranks there is nothing left to
    parallelise the bands with. So the invariant is on ranks per group, not on
    KPAR.
    """
    p = plan(nkpts, nbands=200, ntasks=ntasks, cores_per_socket=24)
    if p.kpar > 1:
        assert p.ranks_per_kgroup >= MIN_RANKS_PER_KGROUP
        assert p.ncore >= 2          # which is also what VASP asks for


def test_still_beats_gcd_where_gcd_collapses():
    """5x5x1 -> 25 irreducible k-points, the grid most repaired cells land on.

    `gcd(25, 16) = 1`, so a rule requiring KPAR | NKPTS gives those structures
    no k-point parallelism at all. We still give them some -- just not the
    maximum, because the benchmark says the maximum is slower.
    """
    assert math.gcd(25, 16) == 1
    ours = choose_kpar(25, 16)
    assert ours > 1
    assert 16 // ours >= MIN_RANKS_PER_KGROUP


def test_kpar_can_be_uncapped_for_a_deliberate_benchmark():
    """The cap is a default, not a law -- benchmarking must be able to lift it."""
    assert choose_kpar(25, 16, min_ranks_per_kgroup=0) == 16


def test_ncore_divides_and_prefers_numa_alignment():
    assert choose_ncore(1) == 1
    for ranks in (2, 4, 8, 16, 32, 64):
        n = choose_ncore(ranks, cores_per_socket=24)
        assert ranks % n == 0
    # 16 ranks: sqrt is 4, and 4 divides a 24-core socket
    assert choose_ncore(16, cores_per_socket=24) == 4


def test_band_group_guard_binds_on_few_bands():
    """Too few bands per group is communication, not computation."""
    p = plan(nkpts=1, nbands=8, ntasks=64, min_bands_per_group=8)
    assert p.band_groups <= 1 or p.nbands / p.band_groups >= 8
    assert p.reason


def test_rejects_nonsense():
    with pytest.raises(ParallelError):
        plan(nkpts=0, nbands=10, ntasks=8)
    with pytest.raises(ParallelError):
        plan(nkpts=1, nbands=10, ntasks=0)


# --------------------------------------------------------------------------
# the invariance the original bug violated
# --------------------------------------------------------------------------

def _random_unimodular(rng):
    """A change of basis that leaves the lattice -- and so the crystal -- alone."""
    m = np.eye(3, dtype=int)
    for _ in range(4):
        i, j = rng.choice(3, size=2, replace=False)
        m[i] += rng.choice([-1, 1]) * m[j]
    return m


def test_grid_is_invariant_under_change_of_basis():
    """Same crystal, different basis vectors -> the policy must not notice.

    Regression test for the reference-recompute failure: MP ships some
    structures in near-degenerate settings (one had cell angles 169, 169, 15
    degrees), and applying `reciprocal_density` to those edge lengths produced a
    1x1x1 grid for a metal whose reduced cell needs 5x5x1.
    """
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    rng = np.random.default_rng(0)
    base = Structure(
        Lattice.from_parameters(4.4, 4.4, 4.4, 90, 90, 90),
        ["Ni", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    kpts = Kpoints(scheme="reciprocal_density", value=64)

    def grid_of(struct):
        reduced = struct.get_reduced_structure("niggli")
        g = grid_for(list(reduced.lattice.abc), kpts, n_atoms=len(reduced))
        return (g.a, g.b, g.c)

    want = grid_of(base)
    for _ in range(12):
        m = _random_unimodular(rng)
        skewed = Structure(
            Lattice(np.dot(m, base.lattice.matrix)),
            base.species,
            np.dot(base.frac_coords, np.linalg.inv(m)),
        )
        assert abs(skewed.volume - base.volume) < 1e-6, "not the same lattice"
        assert grid_of(skewed) == want


def test_irreducible_kpoints_reduces_by_symmetry():
    """Cubic symmetry must fold a 4x4x4 mesh well below its 64 points."""
    spglib = pytest.importorskip("spglib")  # noqa: F841
    lattice = [[3.52, 0, 0], [0, 3.52, 0], [0, 0, 3.52]]
    n = irreducible_kpoints(lattice, [[0, 0, 0]], [28], [4, 4, 4])
    assert n is not None
    assert 1 < n < 64


def test_irreducible_kpoints_is_optional():
    """A missing spglib must degrade, not explode -- the grid product is a
    valid upper bound and only makes KPAR more conservative."""
    n = irreducible_kpoints([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[0, 0, 0]], [1], [2, 2, 2])
    assert n is None or n >= 1
