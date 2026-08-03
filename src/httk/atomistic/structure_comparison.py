"""Comparing structures as crystals rather than as lists of atoms.

``UnitcellStructure.__eq__`` compares the site list element by element, which is the right default
for a data object but is stricter than crystallographic identity: the same crystal written
with its atoms in a different order, or with a site represented by a different lattice
translate, is a different ``UnitcellStructure`` and the same crystal.

That distinction matters for the asymmetric-unit round trip.
``UnitcellStructure -> ASUStructure -> UnitcellStructure`` reproduces the crystal exactly — the coordinates
are exact rationals throughout, with no numerical drift — but site ordering and the choice
of lattice representative follow the canonical expansion order rather than the original
input. :func:`same_crystal` is the predicate that guarantee is stated in.
"""

from collections import Counter
from typing import Any

from ._periodic_wrap import wrap_periodic
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

    Periodicity takes part in two ways. Cells must agree on it, so a slab is never the same
    crystal as the bulk with the same lattice vectors. And only the periodic directions are
    wrapped, since along the others there is no lattice translate to be indifferent about —
    an atom at ``1.05`` really is somewhere else than one at ``0.05``. The comparison stays
    exact in the frame as written: two descriptions of the same slab that differ in their
    non-periodic frame vector compare unequal, because reconciling them would need an origin
    convention this function deliberately does not have.

    Accepts anything structure-like on either side, so a
    :class:`~httk.atomistic.UnitcellStructure` may be compared directly against an
    :class:`~httk.atomistic.ASUStructure` without expanding it by hand.
    """
    from .unitcell_structure_view import UnitcellStructureView

    left = UnitcellStructureView(first)
    right = UnitcellStructureView(second)

    if left.cell.periodicity != right.cell.periodicity:
        return False

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
    wrapped = wrap_periodic(structure.sites.reduced_coords, structure.cell.periodicity)
    return Counter(
        (by_name[name], tuple(coordinate))
        for name, coordinate in zip(structure.species_at_sites, wrapped.to_fractions())
    )
