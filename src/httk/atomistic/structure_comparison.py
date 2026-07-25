"""Comparing structures as crystals rather than as lists of atoms.

``Structure.__eq__`` compares the site list element by element, which is the right default
for a data object but is stricter than crystallographic identity: the same crystal written
with its atoms in a different order, or with a site represented by a different lattice
translate, is a different ``Structure`` and the same crystal.

That distinction matters for the asymmetric-unit round trip.
``Structure -> ASUStructure -> Structure`` reproduces the crystal exactly — the coordinates
are exact rationals throughout, with no numerical drift — but site ordering and the choice
of lattice representative follow the canonical expansion order rather than the original
input. :func:`same_crystal` is the predicate that guarantee is stated in.
"""

from collections import Counter
from typing import Any

from .structure_like import StructureLike

__all__ = ["same_crystal"]


def same_crystal(first: StructureLike, second: StructureLike) -> bool:
    """Whether two structures describe the same crystal.

    True when they have the same cell and the same *multiset* of occupied sites, comparing
    each site by its species and its reduced coordinate wrapped into ``[0, 1)``. Site order
    is ignored, and so is which lattice translate of a site was written down.

    The comparison is exact, not approximate: reduced coordinates are exact rationals and
    cell bases are exact, so two structures that differ by any amount at all compare
    unequal. There is deliberately no tolerance parameter — a tolerant comparison belongs
    with the recognition step that snaps a measured structure onto an idealised one, not
    here.

    Accepts anything structure-like on either side, so a
    :class:`~httk.atomistic.Structure` may be compared directly against an
    :class:`~httk.atomistic.ASUStructure` without expanding it by hand.
    """
    from .structure_simple_view import StructureSimpleView

    left = StructureSimpleView(first)
    right = StructureSimpleView(second)

    if left.cell.basis != right.cell.basis:
        return False

    return _site_multiset(left) == _site_multiset(right)


def _site_multiset(structure: Any) -> Counter[tuple[Any, ...]]:
    """Occupied sites as a multiset of ``(species, wrapped coordinate)``.

    A multiset rather than a set, so that a repeated entry counts. Collapsing duplicates
    would make a structure with a doubled atom compare equal to a correct one, which is
    exactly the sort of expansion bug this predicate exists to catch.
    """
    by_name = {item.name: item for item in structure.species}
    wrapped = structure.sites.reduced_coords.normalize()
    return Counter(
        (by_name[name], tuple(coordinate))
        for name, coordinate in zip(structure.species_at_sites, wrapped.to_fractions())
    )
