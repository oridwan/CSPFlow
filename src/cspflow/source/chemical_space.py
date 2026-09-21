"""Mode 1: element groups plus per-group ratio constraints.

Generalises `search_ternary_magnets.py` from a hardcoded ternary loop to N
groups, so binary, ternary and quaternary spaces use one code path.  Three
things change in the process, each of which was a latent bug in the original:

*   **The group is the unit of constraint, not a named role.**  The original
    takes `tm_rich_ratio=0.75` -- one named constraint that silently assumes a
    particular group is "the TM group".  Here `min_fraction`/`max_fraction` sit
    on whichever group they belong to, an absent key means unconstrained, and
    the binary case (where there is no third group to name) needs no special
    handling.

*   **Arity is explicit.**  The original expresses "two transition metals" as
    two separate groups T and T' drawn from the same list, plus a `T < TP`
    guard to suppress the duplicates that construction creates.  `pick: 2` on
    one group says the same thing with no duplicates to suppress, because
    `combinations` is unordered by construction.

*   **`t_over_t_prime_ratio` is deliberately not carried over.**  It is only
    applied `if not same_lists` -- and in every recorded run of the original,
    T and T' were the same list, so it never fired.  Worse, it is not
    well-defined once the two elements come from one unordered group: with
    `combinations` there is no "primary" and "secondary" to take the ratio of,
    so `n_T/n_T' >= 0.8` and `n_T'/n_T >= 0.8` are different constraints
    selected by nothing but iteration order.  If an intra-group balance
    constraint is wanted later it needs to be stated symmetrically (e.g. "no
    element below x of its group"), which is a new feature, not a port.

Enumeration is exhaustive but pruned: the recursion carries a running sum and
stops as soon as the remaining elements cannot fit under `max_atoms_formula`.
"""

from __future__ import annotations

from itertools import combinations, product
from math import gcd
from typing import Iterator

from ..chem import RARE_EARTHS, canonical_formula, n_rare_earth
from ..config.schema import ChemicalSpace, Source
from .base import EmittedComposition, RejectionLog, SourceError, SourceResult, expand_z


def expand_chemical_space(source: Source) -> SourceResult:
    space = source.chemical_space
    if space is None:                                   # pragma: no cover - schema guards
        raise ValueError("expand_chemical_space called on a source without a block")

    empty = [name for name, g in space.groups.items() if not g.elements]
    if empty:
        raise SourceError(
            f"source '{source.name}': chemical_space group(s) {', '.join(empty)} have no "
            f"elements yet. Fill in `elements:` for each group in campaign.yaml, "
            f"e.g. elements: [Sm, Nd, Pr]")

    result = SourceResult(name=source.name, mode=source.mode.value)
    reject = result.rejected

    group_names = list(space.groups)
    seen_formulas: dict[str, str] = {}                  # formula -> first assignment seen

    for assignment in _element_assignments(space, group_names, reject):
        elements = [el for picks in assignment for el in picks]
        spans = _group_spans(assignment)

        for vector in _stoichiometries(len(elements), space.max_atoms_formula):
            counts = dict(zip(elements, vector))
            total = sum(vector)

            reason = _fails_fractions(space, group_names, spans, vector, total)
            if reason:
                reject.add(reason)
                continue

            if space.max_rare_earth is not None:
                n_re = n_rare_earth(counts)
                if n_re > space.max_rare_earth:
                    reject.add(
                        f"more than {space.max_rare_earth} rare earth(s) in the system",
                        canonical_formula(counts),
                    )
                    continue

            formula = canonical_formula(counts)
            if formula in seen_formulas:
                # Reachable when two groups share elements in different roles.
                reject.add("duplicate of an earlier assignment", formula)
                continue
            seen_formulas[formula] = formula

            result.compositions.extend(
                expand_z(
                    counts,
                    defaults=source.defaults,
                    source_name=source.name,
                    source_mode=source.mode.value,
                    reject=reject,
                )
            )

    _warn_if_degenerate(space, result)
    return result


# --------------------------------------------------------------------------


def _element_assignments(
    space: ChemicalSpace, group_names: list[str], reject: RejectionLog
) -> Iterator[tuple[tuple[str, ...], ...]]:
    """Cartesian product over each group's element choices, minus collisions.

    An element picked by two groups would occupy two roles in one formula, which
    is meaningless -- and it is easy to write by accident, e.g. groups
    `{Fe,Co,Ni}` and `{Fe,Gd}` sharing Fe.  Rejected with a reason rather than
    silently producing a formula with a doubled element.
    """
    per_group = []
    for name in group_names:
        group = space.groups[name]
        choices: list[tuple[str, ...]] = []
        for arity in group.arities():
            choices.extend(combinations(sorted(group.elements), arity))
        per_group.append(choices)

    for assignment in product(*per_group):
        flat = [el for picks in assignment for el in picks]
        if len(set(flat)) != len(flat):
            dupes = sorted({el for el in flat if flat.count(el) > 1})
            reject.add(
                "element claimed by more than one group",
                f"{'+'.join('/'.join(p) for p in assignment)} shares {dupes}",
            )
            continue
        yield assignment


def _group_spans(assignment: tuple[tuple[str, ...], ...]) -> list[tuple[int, int]]:
    """Where each group's elements sit in the flattened stoichiometry vector."""
    spans, start = [], 0
    for picks in assignment:
        spans.append((start, start + len(picks)))
        start += len(picks)
    return spans


def _stoichiometries(k: int, max_atoms: int) -> Iterator[tuple[int, ...]]:
    """All k-tuples of positive integers with sum <= max_atoms and gcd 1.

    gcd 1 is the reduced-formula rule: without it `Fe1Co5` and `Fe2Co10` both
    enter as separate work, and MatterGen is asked to generate the same material
    twice under two names.  Filtering here rather than reducing afterwards keeps
    every emitted vector already canonical.
    """
    if k > max_atoms:
        return
    vector: list[int] = []

    def recurse(depth: int, remaining: int) -> Iterator[tuple[int, ...]]:
        left = k - depth - 1                      # elements still to place after this one
        if depth == k - 1:
            for n in range(1, remaining + 1):
                vector.append(n)
                candidate = tuple(vector)
                if _gcd_all(candidate) == 1:
                    yield candidate
                vector.pop()
            return
        for n in range(1, remaining - left + 1):
            vector.append(n)
            yield from recurse(depth + 1, remaining - n)
            vector.pop()

    yield from recurse(0, max_atoms)


def _gcd_all(values: tuple[int, ...]) -> int:
    g = 0
    for v in values:
        g = gcd(g, v)
        if g == 1:
            return 1
    return g


def _fails_fractions(
    space: ChemicalSpace,
    group_names: list[str],
    spans: list[tuple[int, int]],
    vector: tuple[int, ...],
    total: int,
) -> str:
    """Per-group fraction bounds, evaluated on the reduced formula."""
    for name, (lo, hi) in zip(group_names, spans):
        group = space.groups[name]
        if group.min_fraction is None and group.max_fraction is None:
            continue
        fraction = sum(vector[lo:hi]) / total
        if group.min_fraction is not None and fraction < group.min_fraction:
            return f"group '{name}' below min_fraction {group.min_fraction}"
        if group.max_fraction is not None and fraction > group.max_fraction:
            return f"group '{name}' above max_fraction {group.max_fraction}"
    return ""


def _warn_if_degenerate(space: ChemicalSpace, result: SourceResult) -> None:
    """Catch the two ways a chemical space silently produces nothing useful."""
    if not result.compositions:
        result.warnings.append(
            "this chemical space produced no compositions -- check min_fraction/"
            "max_fraction against max_atoms_formula, and whether `pick` leaves "
            "room for the constraint to be satisfied"
        )
        return
    if space.max_rare_earth is not None:
        groups_with_re = [
            n for n, g in space.groups.items() if set(g.elements) & RARE_EARTHS
        ]
        if not groups_with_re:
            result.warnings.append(
                f"max_rare_earth={space.max_rare_earth} is set but no group contains "
                f"a rare earth, so the guard can never fire"
            )
