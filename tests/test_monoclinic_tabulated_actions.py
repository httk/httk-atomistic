"""The tabulated actions of every monoclinic group cover every ac-``±I`` equivalent description.

The cell-from-net canonicity argument (``docs/plans/PROTONET_CELL_FROM_NET_DESIGN.md`` §2.5)
assumes that ``_tabulated_actions`` contains, up to the group and continuous shifts, every
affine map that maps the group onto itself and whose ac-linear part is ``±I``: origin shifts,
the b flip and inversion. This test enumerates those maps independently and checks the claim.

Why the ``(1/4)ℤ³`` translation grid is complete: in the standard settings of groups 3–15
every point operation ``R`` is diagonal with entries ``±1`` and every translation part lies in
``(1/2)ℤ³`` (both asserted below). A candidate ``ν = (W, t)`` with diagonal ``W`` commutes
with every ``R``, so ``ν (R, r) ν⁻¹ = (R, W r + (I − R) t)``. Normalising the group therefore
only constrains ``(I − R) t`` modulo ``(1/2)ℤ³``; ``I − R`` is diagonal with entries 0 or 2,
so each coordinate of ``t`` is either in ``(1/4)ℤ`` or unconstrained, and it is unconstrained
exactly along the polar directions, where every ``R`` is ``+1``. Those are the continuous
shifts the claim is taken modulo.

Equality modulo the group and continuous shifts is checked exactly, as affine maps: ``ν``
matches a tabulated ``τ`` when some group element ``g`` gives ``g τ`` the same linear part as
``ν`` and a translation that differs from ``ν``'s by an integer vector plus a vector along the
polar directions. As a cross-check, ``ν`` and ``τ`` must also have the same signature (the
Wyckoff letter map and the operation relabelling), which neither the group nor polar shifts change.
"""

from fractions import Fraction
from itertools import product

import pytest

from httk.atomistic import Spacegroup
from httk.atomistic.symmetry import canonical_classification as classification
from httk.atomistic.symmetry.affine_operation import AffineOperation

QUARTERS = [Fraction(k, 4) for k in range(4)]
# normalising candidates per group, of the 256 enumerated: 4 linear parts x 64 translations
NORMALISING = {3: 64, 4: 64, 5: 64, 6: 128, 7: 128, 8: 128, 9: 128, 10: 32, 11: 32, 12: 32, 13: 32, 14: 32, 15: 32}


def _candidates() -> list[AffineOperation]:
    """Return every map with linear part ``diag(s, b, s)`` and translation in ``(1/4)ℤ³`` mod ``ℤ³``."""
    return [
        AffineOperation(((s, 0, 0), (0, b, 0), (0, 0, s)), t)
        for s, b in product((1, -1), repeat=2)
        for t in product(QUARTERS, repeat=3)
    ]


def _covering_action(
    nu: AffineOperation, products: list[tuple[AffineOperation, AffineOperation]], polar: tuple[bool, ...]
) -> AffineOperation | None:
    """Return the tabulated ``τ`` whose ``g τ`` equals ``nu`` modulo ``ℤ³`` and polar shifts, or ``None``."""
    for product_, tau in products:
        if product_.matrix != nu.matrix:
            continue
        difference = [a - b for a, b in zip(nu.vector.to_fractions(), product_.vector.to_fractions(), strict=True)]
        if all(free or d.denominator == 1 for d, free in zip(difference, polar, strict=True)):
            return tau
    return None


@pytest.mark.parametrize("it_number", range(3, 16))
def test_tabulated_actions_cover_every_ac_identity_description(it_number: int) -> None:
    group = Spacegroup.standard(it_number)
    operations = group.symmetry_operations
    rotations = [op.matrix.to_fractions() for op in operations]
    # the premises of the grid argument
    assert all(r[i][j] == 0 for r in rotations for i in range(3) for j in range(3) if i != j)
    assert all((2 * v).denominator == 1 for op in operations for v in op.vector.to_fractions())
    polar = tuple(all(r[i][i] == 1 for r in rotations) for i in range(3))
    assert polar == (
        (False, True, False) if it_number <= 5 else (True, False, True) if it_number <= 9 else (False,) * 3
    )

    tabulated = classification._tabulated_actions(group)
    products = [(g * tau, tau) for g in operations for tau in tabulated]
    candidates = _candidates()
    normalising = [nu for nu in candidates if classification._signature(group, nu) is not None]
    assert (len(candidates), len(normalising)) == (256, NORMALISING[it_number])
    covering = {nu: _covering_action(nu, products, polar) for nu in normalising}
    missing = [nu.to_xyz() for nu, tau in covering.items() if tau is None]
    assert not missing, f"group {it_number}: the tabulated actions miss {missing}"
    for nu, tau in covering.items():
        assert tau is not None
        assert classification._signature(group, nu) == classification._signature(group, tau)
