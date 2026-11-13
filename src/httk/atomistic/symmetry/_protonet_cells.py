"""Cells chosen from a monoclinic bonding net (private).

The net search of :mod:`~httk.atomistic.symmetry.protonet` minimises over a finite set ``E`` of
equivalent descriptions, which is closed in signature but not under the infinitely many changes
between valid cells of groups 3 to 15. This module computes, from the net alone, a finite set
``T(D)`` of ac cell changes ``X = [[a, b], [c, d]]`` (acting as ``[[a, 0, b], [0, 1, 0], [c, 0, d]]``
on fractional coordinates) such that the least key over ``E . T(D)`` is the same for every
description of the net:

- the exact barycentric bond Gram form ``G`` of the net (its ac block transforms as ``X G Xᵀ``);
- rank 2: every cell of ``Γ`` reducing ``G``, ties kept;
- rank 1: the reduced cells modulo shears along the periodic direction by multiples of the
  anchor level ``d``;
- rank 0 with a component periodic along b: one cell per class of ``Γ`` modulo ``Γ(d)``;
- only finite components: the identity alone.

``Γ`` is the group of ac cell changes mapping the standard setting onto itself. Every set is
taken modulo ``±I``, which the tabulated actions already contain. The construction and its
proofs are in the workspace design note ``docs/plans/PROTONET_CELL_FROM_NET_DESIGN.md``.
"""

import math
from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

from httk.atomistic.symmetry.limits import _checkpoint

if TYPE_CHECKING:
    from httk.atomistic.symmetry.protonet import _Element, _GroupTables, _Row

__all__: list[str] = []

type _Matrix2 = tuple[tuple[int, int], tuple[int, int]]
type _Form = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]

_UNIT: _Matrix2 = ((1, 0), (0, 1))
# d: the anchors w/2 of the elements with ac-linear part -I lie in (1/d) Z^2 (design note, section 2.2)
_ANCHOR_LEVEL = {3: 2, 4: 2, 5: 4, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 12: 4, 13: 4, 14: 4, 15: 4}


def _in_gamma(it_number: int, X: _Matrix2) -> bool:
    """Whether an integer ac cell change maps the standard setting of a group 3 to 15 onto itself.

    :param it_number: The International Tables number.
    :param X: The ac block.
    :return: Whether ``X`` is unimodular and keeps the C centring and the c glide where present.
    """
    (a, b), (c, d) = X
    if a * d - b * c not in (1, -1):
        return False
    keep_centring, keep_glide = it_number in (5, 8, 9, 12, 15), it_number in (7, 9, 13, 14, 15)
    return not (keep_centring and c % 2) and not (keep_glide and b % 2)


def _mul(A: _Matrix2, B: _Matrix2) -> _Matrix2:
    """Multiply two 2x2 integer matrices."""
    return (
        (A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]),
        (A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]),
    )


def _inverse(A: _Matrix2) -> _Matrix2:
    """Invert a unimodular 2x2 integer matrix."""
    (a, b), (c, d) = A
    det = a * d - b * c
    return ((d * det, -b * det), (-c * det, a * det))


def _form(X: _Matrix2, G: _Form) -> _Form:
    """Return ``X G Xᵀ``."""
    XG = [[X[r][0] * G[0][c] + X[r][1] * G[1][c] for c in range(2)] for r in range(2)]
    f = [[XG[r][0] * X[c][0] + XG[r][1] * X[c][1] for c in range(2)] for r in range(2)]
    return (f[0][0], f[0][1]), (f[1][0], f[1][1])


def _form_key(F: _Form) -> tuple[Fraction, Fraction, Fraction]:
    """The order in which cells are reduced: ``(F11, F22, F12)``."""
    return F[0][0], F[1][1], F[0][1]


def _completion(t: tuple[int, int]) -> _Matrix2:
    """Return the matrix with columns ``t`` and ``w``, determinant 1, for a primitive ``t``."""
    x, y = t
    old_r, r, old_p, p, old_q, q = x, y, 1, 0, 0, 1
    while r:
        k = old_r // r
        old_r, r, old_p, p, old_q, q = r, old_r - k * r, p, old_p - k * p, q, old_q - k * q
    if old_r < 0:
        old_p, old_q = -old_p, -old_q
    return ((x, -old_q), (y, old_p))  # x p + y q = 1


# ---------------------------------------------------------------------------------------------
# the bond Gram form
# ---------------------------------------------------------------------------------------------
def _bond_gram(
    tables: "_GroupTables", sites: Sequence[tuple[object, str]], rows: Sequence["_Row"]
) -> tuple[_Form, Fraction]:
    """Return the ac block and the bb entry of the exact barycentric bond Gram form of a net.

    The unknowns are the free parameters of each site's reference atom. Row ``r`` gives the bond
    vector ``d_r``, affine in them, weighted by the number of bonds of its orbit in the cell
    (operations over the order of the bond's set stabiliser). Any solution of the normal
    equations of ``Σ m_r |d_r|²`` gives the same vectors, so free variables are set to 0.

    :param tables: The group tables.
    :param sites: ``(class id, Wyckoff letter)`` per site.
    :param rows: ``(i, j, element)`` per row.
    :return: ``(G_ac, G_bb)``.
    """
    charts = {}
    for _, letter in sites:
        if letter not in charts:
            op = tables.spacegroup.wyckoff_position(letter).representative.operation
            charts[letter] = (
                [[int(x) for x in row] for row in op.matrix.to_fractions()],
                [Fraction(x) for x in op.vector.to_fractions()],
            )
    n = 3 * len(sites)
    terms: list[tuple[int, list[dict[int, int]], list[Fraction]]] = []
    for i, j, g in rows:
        (Mi, ci), (Mj, cj) = charts[sites[i][1]], charts[sites[j][1]]
        W, w = tables.W[g[0]], tables.w[g[0]]
        A: list[dict[int, int]] = [{} for _ in range(3)]
        b = []
        for r in range(3):
            for c in range(3):
                coefficient = sum(W[r][k] * Mj[k][c] for k in range(3))
                A[r][3 * j + c] = A[r].get(3 * j + c, 0) + coefficient
                A[r][3 * i + c] = A[r].get(3 * i + c, 0) - Mi[r][c]
            b.append(sum(W[r][k] * cj[k] for k in range(3)) + w[r] + g[1 + r] - ci[r])
        terms.append((_bond_weight(tables, sites, i, j, g), [{k: v for k, v in a.items() if v} for a in A], b))
    # normal equations N u = R, sparse rows
    N: list[dict[int, Fraction]] = [{} for _ in range(n)]
    R = [Fraction(0)] * n
    for m, A, b in terms:
        for a_row, b_r in zip(A, b, strict=True):
            for p, vp in a_row.items():
                for q, vq in a_row.items():
                    N[p][q] = N[p].get(q, Fraction(0)) + m * vp * vq
                R[p] -= m * vp * b_r
    u = _particular_solution(N, R)
    gxx = gxz = gzz = gyy = Fraction(0)
    for m, A, b in terms:
        dx, dy, dz = (b[r] + sum(v * u[k] for k, v in A[r].items()) for r in range(3))
        gxx, gxz, gzz, gyy = gxx + m * dx * dx, gxz + m * dx * dz, gzz + m * dz * dz, gyy + m * dy * dy
    return ((gxx, gxz), (gxz, gzz)), gyy


def _bond_weight(tables: "_GroupTables", sites: Sequence[tuple[object, str]], i: int, j: int, g: "_Element") -> int:
    """Return the number of bonds of a row's orbit in the cell: operations over the set stabiliser.

    Each end stands for a generic point of its Wyckoff chart, so an operation fixes it exactly
    when it fixes the chart (the line through the probe ``ref``). Only the ends of a same-site
    row can be swapped: two sites are distinct orbits even when their letters, hence their probes,
    coincide. Changing the reference atom of a site by ``h`` (rows ``g -> h g`` out, ``g h⁻¹`` in)
    conjugates the stabiliser by ``h``, so the weight, like the bond vectors, is the same for
    every description of the net.
    """
    p = tables.letter(sites[i][1]).ref
    q = tables.apply(g, tables.letter(sites[j][1]).ref)
    count = 0
    for k in range(tables.nops):
        hp, hq = tables.apply((k, 0, 0, 0), p), tables.apply((k, 0, 0, 0), q)
        ends = ((hp, p, hq, q), (hp, q, hq, p)) if i == j else ((hp, p, hq, q),)  # kept, or swapped
        for x, y, z, t in ends:
            shift = [a - c for a, c in zip(x, y, strict=True)]
            if all(s.denominator == 1 for s in shift) and shift == [a - c for a, c in zip(z, t, strict=True)]:
                count += 1
                break
    return tables.nops // count


def _particular_solution(N: list[dict[int, Fraction]], R: list[Fraction]) -> list[Fraction]:
    """Solve a consistent linear system exactly by sparse elimination; free variables are 0.

    A pivot row contains no pivot column that existed when it was made, so one pass over the
    pivots in creation order reduces a new row, and back substitution runs in reverse order.
    """
    pivots: list[tuple[int, dict[int, Fraction], Fraction]] = []
    for row, rhs in zip(N, R, strict=True):
        _checkpoint()
        row = {k: v for k, v in row.items() if v}
        for col, pivot, value in pivots:
            factor = row.get(col)
            if factor:
                for k, v in pivot.items():
                    x = row.get(k, 0) - factor * v
                    if x:
                        row[k] = x
                    else:
                        del row[k]
                rhs -= factor * value
        if not row:
            if rhs:
                raise ValueError("the bond Gram system is inconsistent")
            continue
        col = min(row)
        scale = row[col]
        pivots.append((col, {k: v / scale for k, v in row.items()}, rhs / scale))
    u = [Fraction(0)] * len(R)
    for col, pivot, value in reversed(pivots):
        u[col] = value - sum(v * u[k] for k, v in pivot.items() if k != col)
    return u


# ---------------------------------------------------------------------------------------------
# cells from the form
# ---------------------------------------------------------------------------------------------
def _lagrange(G: _Form) -> _Matrix2:
    """Return ``Q`` in GL2(Z) with ``Q G Qᵀ`` Lagrange-reduced (``|2 F12| <= F11 <= F22``), for definite ``G``."""
    Q = _UNIT
    while True:
        F = _form(Q, G)
        if F[0][0] > F[1][1]:
            Q = (Q[1], Q[0])
            continue
        k = math.floor(F[0][1] / F[0][0] + Fraction(1, 2))
        if not k:
            return Q
        Q = (Q[0], (Q[1][0] - k * Q[0][0], Q[1][1] - k * Q[0][1]))


def _reduced_cells(it_number: int, G: _Form) -> list[_Matrix2]:
    """Return every cell of ``Γ`` whose form ``X G Xᵀ`` has the least key, for a definite ``G``.

    With ``X = M Q0`` and ``Q0 G Q0ᵀ`` reduced, the first row of ``M`` is the shortest vector of an
    admissible class modulo 2; every such class minimum is among ``±e1``, ``±e2`` and
    ``±(e1 - sgn(F12) e2)`` (both signs of the last when ``F12 = 0``). The second row is the
    nearest admissible completion, both when tied, with either sign.

    :param it_number: The International Tables number.
    :param G: The positive definite ac form.
    :return: The reduced cells, ties kept.
    """
    Q0 = _lagrange(G)
    F = _form(Q0, G)
    s = 1 if F[0][1] > 0 else -1
    firsts = [(1, 0), (0, 1), (1, -s)] + ([(1, s)] if F[0][1] == 0 else [])
    best: tuple[Fraction, Fraction, Fraction] | None = None
    out: list[_Matrix2] = []
    for m1 in (v for x in firsts for v in (x, (-x[0], -x[1]))):
        q1 = F[0][0] * m1[0] ** 2 + 2 * F[0][1] * m1[0] * m1[1] + F[1][1] * m1[1] ** 2
        basis = _completion(m1)
        m2 = (basis[0][1], basis[1][1])  # [m1; m2] has determinant 1
        centre = math.floor(
            -(F[0][0] * m1[0] * m2[0] + F[0][1] * (m1[0] * m2[1] + m1[1] * m2[0]) + F[1][1] * m1[1] * m2[1]) / q1
        )
        for j, sign in product(range(centre - 2, centre + 4), (1, -1)):  # every admissible j nearest the centre
            X = _mul((m1, (sign * (m2[0] + j * m1[0]), sign * (m2[1] + j * m1[1]))), Q0)
            if not _in_gamma(it_number, X):
                continue
            key = _form_key(_form(X, G))
            if best is None or key < best:
                best, out = key, []
            if key == best:
                out.append(X)
    return out


def _rank1_cells(it_number: int, t: tuple[int, int]) -> list[_Matrix2]:
    """Return the reduced cells of the form ``t tᵀ`` modulo ``±I`` and shears by the anchor level ``d``.

    The least key is reached exactly when ``X t = ±t*``, ``t*`` the first of ``e2``, ``e1`` and
    ``(1, -1)`` that ``Γ`` reaches from ``t``. Those cells are ``±Y X0`` with ``Y t* = t*``; shears
    ``Y`` that are the identity modulo ``d`` change no string.

    :param it_number: The International Tables number.
    :param t: The primitive periodic ac direction.
    :return: The cells, one per class.
    """
    Binv = _inverse(_completion(t))
    for star in ((0, 1), (1, 0), (1, -1)):
        S = _completion(star)
        for k in (0, 1):
            X0 = _mul(((star[0], S[0][1] + k * star[0]), (star[1], S[1][1] + k * star[1])), Binv)
            if _in_gamma(it_number, X0):
                Sinv = _inverse(S)
                cells = (
                    _mul(_mul(_mul(S, ((1, n), (0, e2))), Sinv), X0)
                    for e2, n in product((1, -1), range(_ANCHOR_LEVEL[it_number]))
                )
                return [X for X in cells if _in_gamma(it_number, X)]
    raise AssertionError("Γ maps every primitive direction onto e2, e1 or (1, -1)")


@cache
def _congruence_cells(it_number: int, d: int) -> tuple[_Matrix2, ...]:
    """Return the smallest cell of ``Γ`` in each class modulo ``±I`` and the level ``d``."""
    reps: dict[tuple[int, ...], _Matrix2] = {}
    for a, b, c, e in sorted(product(range(-3, 4), repeat=4), key=lambda m: (sum(map(abs, m)), m)):
        if _in_gamma(it_number, ((a, b), (c, e))):
            residue = min((a % d, b % d, c % d, e % d), (-a % d, -b % d, -c % d, -e % d))
            reps.setdefault(residue, ((a, b), (c, e)))
    return tuple(reps.values())


def _modulo_sign(cells: Sequence[_Matrix2]) -> tuple[_Matrix2, ...]:
    """Keep one cell of each pair ``±X`` of a set of reduced cells."""
    kept: dict[_Matrix2, None] = {}
    for X in cells:
        if ((-X[0][0], -X[0][1]), (-X[1][0], -X[1][1])) not in kept:
            kept.setdefault(X)
    return tuple(kept)


def _cell_candidates(
    tables: "_GroupTables", sites: Sequence[tuple[object, str]], rows: Sequence["_Row"]
) -> tuple[tuple[_Matrix2, ...], bool]:
    """Return the cells ``T(D)`` to apply before the equivalent descriptions, for a group 3 to 15.

    :param tables: The group tables.
    :param sites: ``(class id, Wyckoff letter)`` per site.
    :param rows: ``(i, j, element)`` per row.
    :return: The cells, and whether the net has rank 0 with a component periodic along b (then the
        cells already cover every linear part, and the tabulated set suffices).
    """
    it_number = tables.spacegroup.it_number
    G, gbb = _bond_gram(tables, sites, rows)
    det = G[0][0] * G[1][1] - G[0][1] * G[1][0]
    if det:
        return _modulo_sign(_reduced_cells(it_number, G)), False
    if G[0][0] or G[1][1]:
        u = (G[0][0], G[1][0]) if G[0][0] else (G[0][1], G[1][1])
        scale = math.lcm(u[0].denominator, u[1].denominator)
        x, y = int(u[0] * scale), int(u[1] * scale)
        g = math.gcd(x, y)
        return tuple(_rank1_cells(it_number, (x // g, y // g))), False
    if gbb:
        return _congruence_cells(it_number, _ANCHOR_LEVEL[it_number]), True
    return (_UNIT,), False  # only finite components (Lemma M)
