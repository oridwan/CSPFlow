"""Magnetisation, in the four units a magnet is actually judged in.

WHAT THIS MODULE DOES
    Takes the one number VASP reports -- the cell magnetisation, in Bohr
    magnetons per cell -- together with the relaxed cell volume, and expresses
    it as the figures of merit a permanent-magnet paper quotes:

        M       mu_B per cell        what VASP printed
        V       A^3                  the relaxed cell
        M/V     mu_B / A^3           volume-normalised, comparable across cells
        mu0*M   tesla                the saturation polarisation J_s
        M/V     emu / cm^3           the same number in the CGS units
                                     the experimental literature uses

    and, separately, the per-ion table that `LORBIT = 11` writes.

WHY THE CONVERSION LIVES HERE AND NOT INLINE IN THE TEMPLATE
    mu0*M is the number a reader compares against Nd2Fe14B (1.61 T) to decide
    whether a candidate is interesting, so it is the most consequential number
    on the page, and it is one multiplication away from being silently wrong by
    10^6.  It is written once, with the derivation in the constant's comment,
    and tested.

        1 mu_B / A^3
            = 9.2740100783e-24 J/T  /  1e-30 m^3
            = 9.2740100783e6 A/m
        mu0 * that = 4*pi*1e-7 T.m/A * 9.2740100783e6 A/m
            = 11.654064 T

    The CGS factor is the same statement in the other system:
        1 mu_B / A^3 = 9.2740100783e-21 erg/G / 1e-24 cm^3 = 9274.01 emu/cm^3

WHAT IT REFUSES TO DO
    It never merges `m_dft_raw` with `m_s_reconstructed`.  The first is the
    spin density VASP integrated; the second is a Hund's-rule model layered on
    the transition-metal sublattice to stand in for a 4f shell a frozen-core
    POTCAR never put in the valence.  For a heavy rare earth they differ by
    more than a factor of two and often in sign.  Every function here takes one
    of them explicitly and labels its output with which.

INPUTS   floats already read from the database or an OUTCAR
OUTPUTS  a `MagneticSummary`, and rows for the per-site table
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..chem import RARE_EARTHS

# mu_B / A^3  ->  tesla, via mu0.  See the derivation in the module docstring.
TESLA_PER_MUB_PER_A3 = 11.654064
# mu_B / A^3  ->  emu / cm^3 (CGS), the unit the experimental literature uses.
EMU_PER_CC_PER_MUB_PER_A3 = 9274.0100783

# For scale on the page.  Not a target and not a threshold -- a reader needs
# one familiar number beside an unfamiliar one, and this is the magnet every
# rare-earth campaign is implicitly measured against.
ND2FE14B_JS_TESLA = 1.61


@dataclass(frozen=True)
class MagneticSummary:
    """One structure's magnetisation in every unit the report shows.

    `source` names which magnetisation this was built from -- "m_dft_raw" for
    the computed cell value, "m_s_reconstructed" for the Hund's-rule model --
    and travels with the numbers so a table cell can never be read as the other
    one.
    """

    source: str
    m_cell: float | None          # mu_B per cell
    volume: float | None          # A^3
    z: int | None = None          # formula units in the cell

    @property
    def m_per_volume(self) -> float | None:
        """mu_B / A^3."""
        if self.m_cell is None or not self.volume:
            return None
        return self.m_cell / self.volume

    @property
    def mu0_m(self) -> float | None:
        """Saturation polarisation mu0*M, in tesla."""
        mv = self.m_per_volume
        return None if mv is None else mv * TESLA_PER_MUB_PER_A3

    @property
    def emu_per_cc(self) -> float | None:
        """M/V in emu/cm^3, for comparison with experimental papers."""
        mv = self.m_per_volume
        return None if mv is None else mv * EMU_PER_CC_PER_MUB_PER_A3

    @property
    def m_per_formula_unit(self) -> float | None:
        if self.m_cell is None or not self.z:
            return None
        return self.m_cell / self.z

    @property
    def fraction_of_nd2fe14b(self) -> float | None:
        js = self.mu0_m
        return None if js is None else js / ND2FE14B_JS_TESLA

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "m_cell": self.m_cell,
            "volume": self.volume,
            "m_per_volume": self.m_per_volume,
            "mu0_m_tesla": self.mu0_m,
            "m_emu_per_cc": self.emu_per_cc,
            "m_per_formula_unit": self.m_per_formula_unit,
        }


def summarise(m_cell: float | None, volume: float | None, *,
              source: str = "m_dft_raw", z: int | None = None) -> MagneticSummary:
    """The four-unit summary for one structure.

    `source` is required to default to the computed value rather than the
    modelled one: a caller that forgets to say gets the number VASP measured.
    """
    return MagneticSummary(source=source, m_cell=m_cell, volume=volume, z=z)


# -- the per-ion table ------------------------------------------------------

@dataclass(frozen=True)
class SiteRow:
    """One ion's line in the per-site moment table."""

    index: int                    # 1-based, as VASP numbers ions
    element: str
    s: float
    p: float
    d: float
    f: float
    total: float
    is_rare_earth: bool

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "element": self.element, "s": self.s,
                "p": self.p, "d": self.d, "f": self.f, "total": self.total,
                "rare_earth": self.is_rare_earth}


def site_rows(sites: Iterable[Any]) -> list[SiteRow]:
    """Turn `moments.SiteMoment` objects into flat rows with fixed channels.

    The channel set VASP prints depends on the POTCARs in the run: an s-p
    system prints `s p d tot` and an f system prints `s p d f tot`.  A table
    whose columns change between structures is unreadable, so every row carries
    all four channels and a channel the run did not print is 0.0 -- which is
    what it physically is for that ion, not a missing value.
    """
    out: list[SiteRow] = []
    for site in sites:
        channels = dict(getattr(site, "channels", {}) or {})
        out.append(SiteRow(
            index=int(site.index),
            element=str(site.element),
            s=float(channels.get("s", 0.0)),
            p=float(channels.get("p", 0.0)),
            d=float(channels.get("d", 0.0)),
            f=float(channels.get("f", 0.0)),
            total=float(site.total),
            is_rare_earth=str(site.element) in RARE_EARTHS,
        ))
    return out


def by_element(rows: Iterable[SiteRow]) -> list[dict[str, Any]]:
    """Sphere-sum per element, with the spread across inequivalent sites.

    The spread is the interesting column: two Fe ions on different Wyckoff
    positions in the same cell routinely differ by 0.5 mu_B, and a single
    per-element average hides exactly the sublattice structure a substitution
    campaign is trying to see.
    """
    groups: dict[str, list[SiteRow]] = {}
    for row in rows:
        groups.setdefault(row.element, []).append(row)

    out = []
    for element, members in sorted(groups.items()):
        totals = [m.total for m in members]
        out.append({
            "element": element,
            "n": len(members),
            "sum": sum(totals),
            "mean": sum(totals) / len(totals),
            "min": min(totals),
            "max": max(totals),
            "spread": max(totals) - min(totals),
            "rare_earth": element in RARE_EARTHS,
        })
    # Largest absolute contribution first: that is the sublattice carrying the
    # magnet, and it should not be sorted alphabetically away from the top.
    out.sort(key=lambda d: -abs(d["sum"]))
    return out
