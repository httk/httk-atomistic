"""Canonical symmetric bonding nets on top of a bare prototype (private).

An *element* ``(k, n1, n2, n3)`` is operation ``k`` (0-based branch of the standard setting's
general Wyckoff position, identity ``0``) followed by the lattice translation ``n``.  A *row*
``(i, j, g)`` bonds the reference atom of site ``i`` to ``g`` applied to the reference atom of
site ``j``; the reference atom is the first-listed Wyckoff coordinate at the site's free
parameters, not reduced modulo one.  ``_canonical_net`` returns the smallest pair list and
code list over every equivalent description, numbering and gauge, together with every
realisation of it; in groups 3 to 15 the descriptions follow the cells chosen from the net
(``_protonet_cells``).  Everything is exact: integers in the search, :class:`~fractions.Fraction`
only while the per-space-group tables are built.
"""

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, cached_property, lru_cache
from itertools import combinations, permutations, product

from httk.atomistic.models.prototype.notation import class_order_key
from httk.atomistic.symmetry._protonet_cells import _cell_candidates, _Matrix2
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_classification import (
    _actions,
    _chirality_target,
    _tabulated_actions,
    _target_letter,
)
from httk.atomistic.symmetry.limits import _checkpoint
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__: list[str] = []

type _Element = tuple[int, int, int, int]
type _Point = tuple[Fraction, Fraction, Fraction]
type _IntMatrix = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
type _Row = tuple[int, int, _Element]
type _Sym = tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]  # (k, c, M): translation c + M v
type _Assign = tuple[_Sym | None, ...]
type _Constraint = tuple[list[list[int]], int, tuple[int, ...]]  # (U, rank, z): v = U[:, :rank] z + kernel
type _Rank = tuple[int, tuple[str, ...]]
type _Candidate = tuple[tuple[_Rank, ...], tuple[tuple[int, int], ...], tuple[tuple[int, int, int, int, int], ...]]

_IDENTITY: _Element = (0, 0, 0, 0)
_PROBE: _Point = (Fraction(13, 97), Fraction(29, 101), Fraction(41, 103))  # generic free-parameter values
# lattice range in which the finitely many classes of gauge moves of a special position are looked up
# (not a search range for the minimum, which is exact)
_FLAT_BOX = 2
# ponytail: b is the nearest lattice shift within +-2 of the home cell (the reference's choice); widen if it raises
_B_BOX = sorted(product(range(-2, 3), repeat=3), key=lambda d: d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
_ZERO: _Point = (Fraction(0), Fraction(0), Fraction(0))
_INVERSION = AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)), (0, 0, 0))


def _key(e: _Element) -> tuple[int, int, int, int, int]:
    """Order elements inside a double coset: shortest translation, then operation, then translation."""
    return (e[1] * e[1] + e[2] * e[2] + e[3] * e[3], e[0], e[1], e[2], e[3])


def _floor(x: Fraction) -> int:
    """Return the exact floor of a fraction."""
    return x.numerator // x.denominator


def _int_matrix(operation: AffineOperation) -> _IntMatrix | None:
    """Return an operation's matrix as integers, or ``None`` when it is not integral."""
    rows = operation.matrix.to_fractions()
    if any(Fraction(x).denominator != 1 for row in rows for x in row):
        return None
    a, b, c = (tuple(int(x) for x in row) for row in rows)
    return (a[0], a[1], a[2]), (b[0], b[1], b[2]), (c[0], c[1], c[2])


def _vector(values: Iterable[Fraction | int]) -> _Point:
    """Return three exact fractions."""
    x, y, z = (Fraction(v) for v in values)
    return x, y, z


def _mat_mul(a: _IntMatrix, b: _IntMatrix) -> _IntMatrix:
    """Multiply two integer matrices."""
    r0, r1, r2 = (
        (
            a[r][0] * b[0][0] + a[r][1] * b[1][0] + a[r][2] * b[2][0],
            a[r][0] * b[0][1] + a[r][1] * b[1][1] + a[r][2] * b[2][1],
            a[r][0] * b[0][2] + a[r][1] * b[1][2] + a[r][2] * b[2][2],
        )
        for r in range(3)
    )
    return r0, r1, r2


def _apply(matrix: _IntMatrix, vector: _Point, point: _Point) -> _Point:
    """Apply ``x -> W x + w`` exactly."""
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + vector[0],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + vector[1],
        matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + vector[2],
    )


def _column_echelon(
    rows: Sequence[Sequence[int]], ncols: int, reduce: bool = False
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Bring an integer matrix to column echelon form ``H = A U`` by integer column operations.

    :param rows: The rows of ``A``.
    :param ncols: The number of columns of ``A``.
    :param reduce: Whether to reduce the entries left of each pivot.
    :return: ``(H, U, pivots)``: ``pivots[i]`` is the row of column ``i``'s positive leading entry; the
        columns ``len(pivots)..`` of ``H`` are zero, so the same columns of ``U`` span the integer kernel.
    """
    A = [list(r) for r in rows]
    U = [[1 if i == j else 0 for j in range(ncols)] for i in range(ncols)]
    pivots: list[int] = []
    col = 0
    for row in range(len(A)):
        if col == ncols:
            break
        while True:
            live = [x for x in range(col, ncols) if A[row][x]]
            if len(live) <= 1:
                break
            p = min(live, key=lambda x: abs(A[row][x]))
            for x in live:
                q = A[row][x] // A[row][p] if x != p else 0
                if q:
                    for R in A:
                        R[x] -= q * R[p]
                    for R in U:
                        R[x] -= q * R[p]
        if not live:
            continue
        p = live[0]
        for M in (A, U):
            for R in M:
                R[p], R[col] = R[col], R[p]
        if A[row][col] < 0:
            for M in (A, U):
                for R in M:
                    R[col] = -R[col]
        if reduce:
            for x in range(col):
                q = A[row][x] // A[row][col]
                if q:
                    for R in A:
                        R[x] -= q * R[col]
                    for R in U:
                        R[x] -= q * R[col]
        pivots.append(row)
        col += 1
    return A, U, pivots


def _coset_minimum(
    H: Sequence[Sequence[int]], pivots: Sequence[int], b: Sequence[int]
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Return the smallest ``m = b + H z`` (``z`` integer) under ``(m.m, m1, m2, m3)``, with ``z``.

    ``H`` has three rows in column echelon form.  Exact: every vector not longer than the best one
    found so far is enumerated, pivot row by pivot row.

    :param H: The lattice basis in column echelon form.
    :param pivots: The pivot rows of ``H``.
    :param b: The coset offset.
    :return: ``(m.m, m, z)``.
    """
    m, z0 = list(b), []
    for i, p in enumerate(pivots):  # first candidate: round in each pivot row
        d = H[p][i]
        z = -((2 * m[p] + d) // (2 * d))
        z0.append(z)
        m = [m[x] + z * H[x][i] for x in range(3)]
    best = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2], tuple(m))
    best_z = tuple(z0)

    def descend(i: int, m: list[int], z: list[int], used: int) -> None:
        nonlocal best, best_z
        if i == len(pivots):
            candidate = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2], tuple(m))
            if candidate < best:
                best, best_z = candidate, tuple(z)
            return
        p = pivots[i]
        d = H[p][i]
        room = best[0] - used
        if room < 0:
            return
        s = math.isqrt(room)
        for zi in range(-((s + m[p]) // d), (s - m[p]) // d + 1):
            m2 = [m[x] + zi * H[x][i] for x in range(3)]
            descend(i + 1, m2, [*z, zi], used + m2[p] * m2[p])

    descend(0, list(b), [], 0)
    return best[0], best[1], best_z


@dataclass(frozen=True)
class _Letter:
    """Probe reference point, flat equations and site-symmetry group of one Wyckoff letter."""

    letter: str
    ref: _Point
    normal: tuple[tuple[int, int, int], ...]  # integer rows annihilating the chart's free columns
    offset: tuple[Fraction, ...]  # normal . point == offset on the flat
    H: tuple[_Element, ...]
    lattice: tuple[tuple[int, int, int], ...]  # basis of the lattice translations along the flat
    multiplicity: int
    free_count: int

    @property
    def general(self) -> bool:
        """Whether the site symmetry is trivial."""
        return len(self.H) == 1

    def on_flat(self, point: Sequence[Fraction]) -> bool:
        """Exact rank test: whether ``point`` lies on the first-listed coordinate's flat.

        :param point: An exact reduced coordinate.
        :return: Whether the point is on the flat.
        """
        return all(
            row[0] * point[0] + row[1] * point[1] + row[2] * point[2] == value
            for row, value in zip(self.normal, self.offset, strict=True)
        )


class _Coset:
    """One compiled equivalent description, or a chirality transfer into another group.

    The per-letter data (target letter and ``b``) is compiled on first use: a structure needs only
    its own letters, and the exact Wyckoff-action proof dominates the table cost of large domains.

    :param source: The tables of the group the action starts from.
    :param target: The tables of the group the action lands in.
    :param action: The affine action.
    :param relabel: Per source operation ``k`` the target operation and integer residue.
    """

    def __init__(
        self,
        source: "_GroupTables",
        target: "_GroupTables",
        action: AffineOperation,
        relabel: tuple[tuple[int, tuple[int, int, int]], ...],
    ) -> None:
        matrix = _int_matrix(action)
        if matrix is None:
            raise ValueError("only integer-matrix actions can be compiled")
        self.action = action
        self.relabel = relabel
        self.P = matrix[0] + matrix[1] + matrix[2]
        self._matrix, self._vector = matrix, _vector(action.vector.to_fractions())
        self._source, self._target = source, target
        self._letters: dict[str, tuple[str, _Element]] = {}

    def letter(self, letter: str) -> tuple[str, _Element]:
        """Return the target letter and the element ``b`` putting the moved reference atom on its chart.

        :param letter: The source Wyckoff letter.
        :return: ``(target letter, b)``.
        """
        hit = self._letters.get(letter)
        if hit is None:
            source, target = self._source, self._target
            name = (
                _target_letter(source.spacegroup, self.action, letter)
                if target is source
                else compile_wyckoff_action(
                    source.spacegroup, self.action, letter, target_spacegroup=target.spacegroup
                ).target_letter
            )
            q = _apply(self._matrix, self._vector, source.letter(letter).ref)
            hit = self._letters[letter] = (name, target._onto_chart(target.letter(name), q))
        return hit


class _GroupTables:
    """Exact per-space-group element algebra, Wyckoff data and equivalent descriptions.

    :param it_number: The International Tables number of the standard setting.
    """

    def __init__(self, it_number: int) -> None:
        self.spacegroup = spacegroup = Spacegroup.standard(it_number)
        general = max(spacegroup.wyckoff, key=lambda w: (w.free_count, w.multiplicity))
        self.operations = tuple(branch.operation for branch in general.branches)
        if not self.operations[0].is_identity():
            raise ValueError(f"operation 1 of space group {it_number} is not the identity")
        self.nops = len(self.operations)
        matrices = [_int_matrix(op) for op in self.operations]
        if any(m is None for m in matrices):
            raise ValueError(f"space group {it_number} has a non-integral operation matrix")
        self.W: tuple[_IntMatrix, ...] = tuple(m for m in matrices if m is not None)
        self.w: tuple[_Point, ...] = tuple(_vector(op.vector.to_fractions()) for op in self.operations)
        self.Wi = tuple(tuple(x for row in m for x in row) for m in self.W)
        scale = math.lcm(*(x.denominator for v in self.w for x in v))
        ws = [tuple(int(x * scale) for x in v) for v in self.w]
        self._lookup = {(self.Wi[k], tuple(x % 1 for x in self.w[k])): k for k in range(self.nops)}
        scaled = {(self.Wi[k], tuple(x % scale for x in ws[k])): k for k in range(self.nops)}
        if len(scaled) != self.nops:
            raise ValueError(f"space group {it_number} repeats an operation modulo the lattice")
        self.mulk: list[list[tuple[int, tuple[int, int, int]]]] = []
        for a in range(self.nops):
            Wa, wa, row = self.W[a], ws[a], []
            for b in range(self.nops):
                m, wb = _mat_mul(Wa, self.W[b]), ws[b]
                t = [Wa[r][0] * wb[0] + Wa[r][1] * wb[1] + Wa[r][2] * wb[2] + wa[r] for r in range(3)]
                k = scaled.get((m[0] + m[1] + m[2], (t[0] % scale, t[1] % scale, t[2] % scale)))
                if k is None:
                    raise ValueError(f"operation list of space group {it_number} is not closed")
                wk = ws[k]
                row.append((k, ((t[0] - wk[0]) // scale, (t[1] - wk[1]) // scale, (t[2] - wk[2]) // scale)))
            self.mulk.append(row)
        self.invk: list[tuple[int, tuple[int, int, int]]] = []
        for a in range(self.nops):
            b = next(b for b in range(self.nops) if self.mulk[a][b][0] == 0)
            r, Wb = self.mulk[a][b][1], self.Wi[b]
            s0, s1, s2 = (-(Wb[3 * i] * r[0] + Wb[3 * i + 1] * r[1] + Wb[3 * i + 2] * r[2]) for i in range(3))
            self.invk.append((b, (s0, s1, s2)))
        self._letters: dict[str, _Letter] = {}
        self._moves: dict[str, list[_Element]] = {}
        self._reps: dict[str, list[_Element]] = {}

    # --- element algebra ---------------------------------------------------------------------
    def mul(self, a: _Element, b: _Element) -> _Element:
        """Compose two elements.

        :param a: The left element.
        :param b: The right element.
        :return: ``a * b``.
        """
        k, r = self.mulk[a[0]][b[0]]
        W = self.Wi[a[0]]
        return (
            k,
            a[1] + W[0] * b[1] + W[1] * b[2] + W[2] * b[3] + r[0],
            a[2] + W[3] * b[1] + W[4] * b[2] + W[5] * b[3] + r[1],
            a[3] + W[6] * b[1] + W[7] * b[2] + W[8] * b[3] + r[2],
        )

    def inv(self, a: _Element) -> _Element:
        """Invert an element.

        :param a: The element.
        :return: ``a`` inverted.
        """
        k, s = self.invk[a[0]]
        W = self.Wi[k]
        return (
            k,
            s[0] - (W[0] * a[1] + W[1] * a[2] + W[2] * a[3]),
            s[1] - (W[3] * a[1] + W[4] * a[2] + W[5] * a[3]),
            s[2] - (W[6] * a[1] + W[7] * a[2] + W[8] * a[3]),
        )

    def apply(self, e: _Element, point: _Point) -> _Point:
        """Apply an element to an exact point.

        :param e: The element.
        :param point: The exact reduced coordinate.
        :return: The moved, unreduced coordinate.
        """
        x, y, z = _apply(self.W[e[0]], self.w[e[0]], point)
        return x + e[1], y + e[2], z + e[3]

    def canon(self, Hi: Sequence[_Element], Hj: Sequence[_Element], g: _Element, same_site: bool = False) -> _Element:
        """Return the smallest element of ``Hi g Hj``, and of its inverse for a bond inside one orbit.

        :param Hi: The site-symmetry group of the bond's first site.
        :param Hj: The site-symmetry group of the bond's second site.
        :param g: The bond element.
        :param same_site: Whether both ends are the same site.
        :return: The code of the bond orbit.
        """
        best: _Element | None = None
        for x in (g, self.inv(g)) if same_site else (g,):
            for h1 in Hi:
                y = self.mul(h1, x)
                for h2 in Hj:
                    z = self.mul(y, h2)
                    if best is None or _key(z) < _key(best):
                        best = z
        assert best is not None
        return best

    # --- per Wyckoff letter --------------------------------------------------------------------
    def letter(self, letter: str) -> _Letter:
        """Return the probe data of one Wyckoff letter (without gauge moves).

        :param letter: The Wyckoff letter.
        :return: The letter's reference point, flat and site-symmetry group.
        """
        info = self._letters.get(letter)
        if info is None:
            position = self.spacegroup.wyckoff_position(letter)
            chart = position.representative
            M = _int_matrix(chart.operation)
            if M is None:
                raise ValueError(f"Wyckoff letter {letter!r} has a non-integral chart")
            v = _vector(chart.operation.vector.to_fractions())
            ref = _apply(M, v, _PROBE)
            U = [tuple(int(Fraction(x)) for x in row) for row in chart._unimodular.to_fractions()]
            normal = tuple((row[0], row[1], row[2]) for row in U[position.free_count :])
            if any(sum(row[i] * M[i][c] for i in range(3)) for row in normal for c in range(3)):
                raise ValueError(f"Wyckoff letter {letter!r} has a chart outside its echelon form")
            offset = tuple(row[0] * v[0] + row[1] * v[1] + row[2] * v[2] for row in normal)
            H = []
            for k in range(self.nops):
                n = [a - b for a, b in zip(ref, _apply(self.W[k], self.w[k], ref), strict=True)]
                if all(x.denominator == 1 for x in n):
                    H.append((k, int(n[0]), int(n[1]), int(n[2])))
            if len(H) * position.multiplicity != self.nops:
                raise ValueError(f"site-symmetry group of Wyckoff letter {letter!r} has the wrong order")
            lattice: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            if len(H) > 1:  # lattice translations t that keep the flat: W_h t = t for every h in H
                fixed = [[self.W[h[0]][r][x] - (r == x) for x in range(3)] for h in H for r in range(3)]
                _, kernel, pivots = _column_echelon(fixed, 3)
                lattice = tuple((kernel[0][c], kernel[1][c], kernel[2][c]) for c in range(len(pivots), 3))
            info = _Letter(letter, ref, normal, offset, tuple(H), lattice, position.multiplicity, position.free_count)
            self._letters[letter] = info
        return info

    def moves(self, letter: str) -> list[_Element]:
        """Return the gauge moves of one Wyckoff letter within the lookup box, in the reference's order.

        Every letter uses the +-2 lookup; the oracle uses +-1 for general positions, where its moves
        only serve the scramble test.  The search never reads general-position moves (:meth:`reps`
        returns one element per operation there), which is why ``test_cosets_match_oracle`` compares
        moves for special positions only.

        :param letter: The Wyckoff letter.
        :return: One element per distinct moved reference point on the flat.
        """
        moves = self._moves.get(letter)
        if moves is None:
            info = self.letter(letter)
            box = range(-_FLAT_BOX, _FLAT_BOX + 1)
            moves, seen = [], set()
            for k in range(self.nops):
                base = _apply(self.W[k], self.w[k], info.ref)
                lo = [-_floor(x) for x in base]
                gap = [
                    value - (r[0] * base[0] + r[1] * base[1] + r[2] * base[2])
                    for r, value in zip(info.normal, info.offset)
                ]
                for d in product(box, repeat=3):
                    n = (lo[0] + d[0], lo[1] + d[1], lo[2] + d[2])
                    if all(r[0] * n[0] + r[1] * n[1] + r[2] * n[2] == g for r, g in zip(info.normal, gap)):
                        point = (base[0] + n[0], base[1] + n[1], base[2] + n[2])
                        if point not in seen:
                            seen.add(point)
                            moves.append((k, n[0], n[1], n[2]))
            self._moves[letter] = moves
        return moves

    def reps(self, letter: str) -> list[_Element]:
        """Return one gauge move per class modulo lattice translations along the letter's flat.

        A general position takes every operation; a special position's finitely many classes are
        collected from :meth:`moves`.

        :param letter: The Wyckoff letter.
        :return: The class representatives.
        """
        reps = self._reps.get(letter)
        if reps is None:
            info = self.letter(letter)
            if info.general:
                reps = [(k, 0, 0, 0) for k in range(self.nops)]
            else:
                basis, _, pivots = _column_echelon([[v[x] for v in info.lattice] for x in range(3)], len(info.lattice))

                def in_lattice(t: Sequence[int]) -> bool:
                    rest = list(t)
                    for i, p in enumerate(pivots):
                        if rest[p] % basis[p][i]:
                            return False
                        q = rest[p] // basis[p][i]
                        rest = [rest[x] - q * basis[x][i] for x in range(3)]
                    return not any(rest)

                reps = []
                for e in self.moves(letter):
                    if not any((d := self.mul(e, self.inv(rho)))[0] == 0 and in_lattice(d[1:]) for rho in reps):
                        reps.append(e)
            self._reps[letter] = reps
        return reps

    def _onto_chart(self, target: _Letter, q: _Point) -> _Element:
        """Return the element moving ``q`` onto ``target``'s flat: first operation, nearest shift."""
        for k in range(self.nops):
            x = _apply(self.W[k], self.w[k], q)
            gap = [value - (r[0] * x[0] + r[1] * x[1] + r[2] * x[2]) for r, value in zip(target.normal, target.offset)]
            if any(g.denominator != 1 for g in gap):
                continue
            lo = [-_floor(c) for c in x]
            for d in _B_BOX:
                n = (lo[0] + d[0], lo[1] + d[1], lo[2] + d[2])
                if all(r[0] * n[0] + r[1] * n[1] + r[2] * n[2] == g for r, g in zip(target.normal, gap)):
                    return (k, n[0], n[1], n[2])
        raise ValueError("cannot move a Wyckoff reference point onto its target chart")

    # --- equivalent descriptions ---------------------------------------------------------------
    @cached_property
    def polar_axes(self) -> tuple[int, ...]:
        """Cell axes fixed by every operation: origin shifts along them change no code."""
        return tuple(
            axis
            for axis in range(3)
            if all(W[3 * r + axis] == (1 if r == axis else 0) for W in self.Wi for r in range(3))
        )

    def _coset_form(self, P: _IntMatrix, p: _Point) -> tuple[object, ...]:
        """Normal form of an action modulo the group, the lattice and continuous polar shifts."""
        polar = self.polar_axes
        forms = []
        for W, w in zip(self.W, self.w, strict=True):
            matrix = _mat_mul(W, P)
            moved = _apply(W, w, p)
            forms.append((matrix, tuple(Fraction(0) if r in polar else moved[r] % 1 for r in range(3))))
        return min(forms)

    @cached_property
    def cosets(self) -> tuple[_Coset, ...]:
        """One compiled representative per coset of httk's finite same-group action domain."""
        kept: dict[tuple[object, ...], AffineOperation] = {}
        for action in _actions(self.spacegroup):
            _checkpoint()
            P = _int_matrix(action)
            if P is None:
                continue  # not an automorphism of this lattice basis
            kept.setdefault(self._coset_form(P, _vector(action.vector.to_fractions())), action)
        return tuple(self._compile(action, self) for action in kept.values())

    @cached_property
    def tabulated(self) -> tuple[int, ...]:
        """Indices of the cosets that are httk's tabulated actions, without the added cell changes."""
        tabulated = set(_tabulated_actions(self.spacegroup))
        return tuple(index for index, coset in enumerate(self.cosets) if coset.action in tabulated)

    def _compile(self, action: AffineOperation, target: "_GroupTables") -> _Coset:
        """Compile an action into ``target``: operation relabelling, translation matrix and ``b`` per letter."""
        P, Pinv = _int_matrix(action), _int_matrix(action.inverse())
        if P is None or Pinv is None:
            raise ValueError("only integer-matrix actions can be compiled")
        p = _vector(action.vector.to_fractions())
        relabel = []
        for k in range(self.nops):
            Wc = _mat_mul(_mat_mul(P, self.W[k]), Pinv)
            moved = _apply(Wc, _ZERO, p)
            wc = [a - b for a, b in zip(_apply(P, p, self.w[k]), moved, strict=True)]
            k2 = target._lookup.get((tuple(x for row in Wc for x in row), tuple(x % 1 for x in wc)))
            if k2 is None:
                raise ValueError("action does not normalise the group")
            c0, c1, c2 = (int(a - b) for a, b in zip(wc, target.w[k2], strict=True))
            relabel.append((k2, (c0, c1, c2)))
        return _Coset(self, target, action, tuple(relabel))

    def conjugate(self, coset: _Coset, g: _Element) -> _Element:
        """Return ``nu g nu^-1`` for the coset's action ``nu``.

        :param coset: The compiled coset (or chirality transfer).
        :param g: The element in the source numbering.
        :return: The element in the target numbering.
        """
        k, c = coset.relabel[g[0]]
        P = coset.P
        return (
            k,
            P[0] * g[1] + P[1] * g[2] + P[2] * g[3] + c[0],
            P[3] * g[1] + P[4] * g[2] + P[5] * g[3] + c[1],
            P[6] * g[1] + P[7] * g[2] + P[8] * g[3] + c[2],
        )


@cache
def _tables(it_number: int) -> _GroupTables:
    """Return the cached tables of one standard-setting space group."""
    return _GroupTables(it_number)


@cache
def _transfer(source_it: int, target_it: int) -> _Coset:
    """Compile the inversion from one group's numbering and letters into its enantiomorphic partner's."""
    return _tables(source_it)._compile(_INVERSION, _tables(target_it))


@lru_cache(maxsize=1024)
def _cell(it_number: int, X: _Matrix2) -> _Coset:
    """Compile the ac cell change ``X`` of a monoclinic group into its own tables."""
    (a, b), (c, d) = X
    tables = _tables(it_number)
    return tables._compile(AffineOperation(((a, 0, b), (0, 1, 0), (c, 0, d))), tables)


def _moved(tables: _GroupTables, coset: _Coset, sites: Sequence[tuple[Hashable, str]], row: _Row) -> _Row:
    """Map one row through a compiled coset into ``tables``' group: ``b_a (nu g nu^-1) b_b^-1``."""
    a, b, g = row
    ba, bb = coset.letter(sites[a][1])[1], coset.letter(sites[b][1])[1]
    return a, b, tables.mul(tables.mul(ba, tables.conjugate(coset, g)), tables.inv(bb))


# ---------------------------------------------------------------------------------------------
# numbering of sites: smallest list of (i, j) pairs
# ---------------------------------------------------------------------------------------------
def _minimal_numberings(
    blocks: Sequence[Sequence[int]], rows: Sequence[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], list[dict[int, int]]]:
    """Return the smallest sorted pair list and every 1-based numbering that realises it.

    :param blocks: Interchangeable sites in label order.
    :param rows: The bonded site pairs.
    :return: ``(pattern, numberings)``.
    """
    nsites = sum(len(b) for b in blocks)
    block_of: dict[int, int] = {}
    first = []
    start = 0
    for index, block in enumerate(blocks):
        first.append(start)
        for site in block:
            block_of[site] = index
        start += len(block)
    owner = [index for index, block in enumerate(blocks) for _ in block]  # number -> block
    neighbours: dict[int, Counter[int]] = {s: Counter() for b in blocks for s in b}
    for a, b in rows:
        neighbours[a][b] += 1
        if a != b:
            neighbours[b][a] += 1
    isolated = {s for s, c in neighbours.items() if not c}
    total = len(rows)
    best_pattern: tuple[tuple[int, int], ...] | None = None
    best_numberings: list[dict[int, int]] = []

    def descend(m: int, number: dict[int, int], free: list[int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal best_pattern, best_numberings
        _checkpoint()
        if m == nsites:
            if best_pattern is None or pattern < best_pattern:
                best_pattern, best_numberings = pattern, []
            if pattern == best_pattern:
                best_numberings.append(dict(number))
            return
        holder = next((s for s, n in number.items() if n == m), None)
        if holder is not None:
            candidates = [holder]
        else:
            pool = [s for s in blocks[owner[m]] if s not in number]
            connected = [s for s in pool if s not in isolated]
            candidates = connected if connected else pool[:1]  # isolated sites: any fixed order
        for site in candidates:
            number2, free2 = dict(number), list(free)
            if site not in number2:
                number2[site] = m
                free2[owner[m]] = m + 1
            pending: dict[int, list[int]] = {}
            for other in neighbours[site]:
                if other not in number2:
                    pending.setdefault(block_of[other], []).append(other)
            options = [permutations(group) for group in pending.values()]
            for choice in product(*options):
                number3, free3 = dict(number2), list(free2)
                for group in choice:
                    for other in group:  # next free numbers of that block
                        number3[other] = free3[block_of[other]]
                        free3[block_of[other]] += 1
                group_rows = sorted(
                    (m, number3[other])
                    for other, mult in neighbours[site].items()
                    if number3[other] >= m
                    for _ in range(mult)
                )
                pattern3 = pattern + tuple(group_rows)
                if best_pattern is not None and len(best_pattern) == total and pattern3 > best_pattern[: len(pattern3)]:
                    continue
                descend(m + 1, number3, free3, pattern3)

    descend(0, {}, list(first), ())
    assert best_pattern is not None
    return best_pattern, [{s: n + 1 for s, n in numbering.items()} for numbering in best_numberings]


# ---------------------------------------------------------------------------------------------
# reference atoms: smallest list of codes for one numbering, exact
#
# A reference atom is  a_s = (k_s, c_s + M_s v):  a fixed operation, a fixed lattice translation and
# a part linear in a vector v of integer unknowns.  A general position contributes three unknowns,
# a special position one per independent translation along its flat, and its finitely many other
# gauge-move classes are tried one by one.  The translation of a_i g a_j^-1 is then b + M v, so the
# smallest code of a row is the smallest vector of a lattice coset, found exactly.  Fixing it is a
# linear condition on v; it is solved, substituted, and the next row is treated the same way.  No
# search range is involved.
# ---------------------------------------------------------------------------------------------
def _sym(e: _Element, D: int) -> _Sym:
    """Lift a plain element to a symbolic one with ``D`` (absent) unknowns."""
    return (e[0], (e[1], e[2], e[3]), ((0,) * D,) * 3)


def _smul(tables: _GroupTables, a: _Sym, b: _Sym) -> _Sym:
    """Compose two symbolic elements."""
    k, r = tables.mulk[a[0]][b[0]]
    W = tables.Wi[a[0]]
    c = tuple(a[1][x] + W[3 * x] * b[1][0] + W[3 * x + 1] * b[1][1] + W[3 * x + 2] * b[1][2] + r[x] for x in range(3))
    M = tuple(
        tuple(
            a[2][x][d] + W[3 * x] * b[2][0][d] + W[3 * x + 1] * b[2][1][d] + W[3 * x + 2] * b[2][2][d]
            for d in range(len(a[2][0]))
        )
        for x in range(3)
    )
    return (k, c, M)


def _sinv(tables: _GroupTables, a: _Sym) -> _Sym:
    """Invert a symbolic element."""
    k, s = tables.invk[a[0]]
    W = tables.Wi[k]
    c = tuple(s[x] - (W[3 * x] * a[1][0] + W[3 * x + 1] * a[1][1] + W[3 * x + 2] * a[1][2]) for x in range(3))
    M = tuple(
        tuple(
            -(W[3 * x] * a[2][0][d] + W[3 * x + 1] * a[2][1][d] + W[3 * x + 2] * a[2][2][d])
            for d in range(len(a[2][0]))
        )
        for x in range(3)
    )
    return (k, c, M)


def _put(assign: _Assign, pos: int, value: _Sym) -> _Assign:
    """Return ``assign`` with one entry replaced."""
    out = list(assign)
    out[pos] = value
    return tuple(out)


def _with_new_site(
    assign: _Assign, D: int, pos: int, rep: _Element, lattice: Sequence[Sequence[int]]
) -> tuple[_Assign, int]:
    """Give the site at ``pos`` the move ``rep`` plus free translations along its lattice."""
    d = len(lattice)
    grown: list[_Sym | None] = [
        None if e is None else (e[0], e[1], tuple(row + (0,) * d for row in e[2])) for e in assign
    ]
    grown[pos] = (rep[0], (rep[1], rep[2], rep[3]), tuple((0,) * D + tuple(v[x] for v in lattice) for x in range(3)))
    return tuple(grown), D + d


def _assignments(
    tables: _GroupTables,
    info: dict[int, _Letter],
    assign: _Assign,
    D: int,
    si: int,
    sj: int,
    g: _Element,
) -> Iterator[tuple[_Assign, int]]:
    """Yield every way to give the two ends ``si <= sj`` of a row a reference atom, as far as they have none yet."""
    ai, aj = assign[si], assign[sj]
    letter_i, letter_j = info[si], info[sj]
    if si == sj:
        if ai is not None:
            yield assign, D
        else:
            for rep in tables.reps(letter_i.letter):
                yield _with_new_site(assign, D, si, rep, letter_i.lattice)
        return
    if ai is not None or (aj is not None and letter_i.general):
        starts = [(assign, D)]
    else:
        starts = [_with_new_site(assign, D, si, rep, letter_i.lattice) for rep in tables.reps(letter_i.letter)]
    for A1, D1 in starts:
        _checkpoint()
        ai, aj = A1[si], A1[sj]
        if ai is None:  # general position next to a fixed atom: a_i = h a_j g^-1
            assert aj is not None
            y = _smul(tables, aj, _sinv(tables, _sym(g, D1)))
            for h in letter_j.H:
                yield _put(A1, si, _smul(tables, _sym(h, D1), y)), D1
        elif aj is not None:
            yield A1, D1
        elif letter_j.general:  # general position next to a fixed atom: a_j = h a_i g
            y = _smul(tables, ai, _sym(g, D1))
            for h in letter_i.H:
                yield _put(A1, sj, _smul(tables, _sym(h, D1), y)), D1
        else:
            for rep in tables.reps(letter_j.letter):
                yield _with_new_site(A1, D1, sj, rep, letter_j.lattice)


def _row_minima(
    tables: _GroupTables, Hi: Sequence[_Element], Hj: Sequence[_Element], x: _Sym, D: int, same_site: bool
) -> Iterator[tuple[tuple[int, ...], _Element, _Constraint | None]]:
    """Yield ``(key, code, constraint)`` for every way to reach each candidate code of ``x``'s double coset.

    The caller keeps the minima; ``constraint`` is the linear condition on the unknowns that fixes it.
    """
    for xs in (x, _sinv(tables, x)) if same_site else (x,):
        for h1 in Hi:
            _checkpoint()
            y = _smul(tables, _sym(h1, D), xs)
            free = D > 0 and any(any(row) for row in y[2])
            H, U, pivots = _column_echelon(y[2], D) if free else ([], [], [])
            for h2 in Hj:
                z = _smul(tables, y, _sym(h2, D))
                if free:
                    n2, m, coeff = _coset_minimum(H, pivots, z[1])
                    yield (n2, z[0], *m), (z[0], m[0], m[1], m[2]), (U, len(pivots), coeff)
                else:
                    m = z[1]
                    yield (m[0] * m[0] + m[1] * m[1] + m[2] * m[2], z[0], *m), (z[0], m[0], m[1], m[2]), None


def _substitute(assign: _Assign, D: int, constraint: _Constraint | None) -> tuple[_Assign, int]:
    """Impose the linear condition that fixed a row's code: ``v = v0 + K w``."""
    if constraint is None:
        return assign, D
    U, r, coeff = constraint
    v0 = [sum(U[d][i] * coeff[i] for i in range(r)) for d in range(D)]
    out: list[_Sym | None] = []
    for e in assign:
        if e is None:
            out.append(None)
            continue
        c = tuple(e[1][x] + sum(e[2][x][d] * v0[d] for d in range(D)) for x in range(3))
        M = tuple(tuple(sum(e[2][x][d] * U[d][col] for d in range(D)) for col in range(r, D)) for x in range(3))
        out.append((e[0], c, M))
    return tuple(out), D - r


def _normalise(assign: _Assign, D: int) -> tuple[_Assign, int]:
    """Return a unique description of the same set of reference-atom choices, so equal states merge."""
    if D == 0:
        return assign, 0
    used = [p for p, e in enumerate(assign) if e is not None]
    entries = [e for e in assign if e is not None]
    stack = [list(e[2][x]) for e in entries for x in range(3)]
    const = [e[1][x] for e in entries for x in range(3)]
    H, _, pivots = _column_echelon(stack, D, reduce=True)
    for i, p in enumerate(pivots):
        q = const[p] // H[p][i]
        if q:
            const = [const[x] - q * H[x][i] for x in range(len(const))]
    out = list(assign)
    for n, (p, e) in enumerate(zip(used, entries, strict=True)):
        out[p] = (
            e[0],
            tuple(const[3 * n : 3 * n + 3]),
            tuple(tuple(H[3 * n + x][: len(pivots)]) for x in range(3)),
        )
    return tuple(out), len(pivots)


@lru_cache(maxsize=65536)
def _component_codes(
    tables: _GroupTables, letters: tuple[str, ...], rows: tuple[_Row, ...]
) -> tuple[tuple[_Element, ...], tuple[_Element | None, ...]]:
    """Run the exact state search over one connected component, in slot order (memoised).

    The component is in local form: site ``s`` has number ``s + 1`` and every row has ``a <= b``, so
    equal components met under other numberings or in other components share one search.

    :param tables: The group tables.
    :param letters: The Wyckoff letter per local site.
    :param rows: The oriented local rows, sorted.
    :return: The codes in slot order and one gauge per local site (``None`` if it took none).
    """
    info = {s: tables.letter(letter) for s, letter in enumerate(letters)}
    slots = sorted((a + 1, b + 1) for a, b, _ in rows)
    states: set[tuple[_Assign, int, frozenset[int]]] = {((None,) * len(letters), 0, frozenset())}
    codes: list[_Element] = []
    for i, j in slots:
        si, sj = i - 1, j - 1
        Hi, Hj = info[si].H, info[sj].H
        best: tuple[int, ...] | None = None
        keep: list[tuple[_Element, _Assign, int, _Constraint | None, frozenset[int]]] = []
        for assign, D, used in states:
            _checkpoint()
            for r, (a, b, g) in enumerate(rows):
                if r in used or a != si or b != sj:
                    continue
                for assign2, D2 in _assignments(tables, info, assign, D, si, sj, g):
                    ai, aj = assign2[si], assign2[sj]
                    assert ai is not None and aj is not None
                    x = _smul(tables, _smul(tables, ai, _sym(g, D2)), _sinv(tables, aj))
                    for key, code, constraint in _row_minima(tables, Hi, Hj, x, D2, si == sj):
                        if best is None or key < best:
                            best, keep = key, []
                        if key == best:
                            keep.append((code, assign2, D2, constraint, used | {r}))
        codes.append(keep[0][0])
        states = set()
        for _, assign2, D2, constraint, used2 in keep:
            states.add((*_normalise(*_substitute(assign2, D2, constraint)), used2))
    assign = next(iter(states))[0]
    return tuple(codes), tuple(None if e is None else (e[0], e[1][0], e[1][1], e[1][2]) for e in assign)


def _exact_codes(
    tables: _GroupTables, letters: dict[int, str], number: dict[int, int], rows: Sequence[_Row]
) -> tuple[list[tuple[int, int]], list[_Element], dict[int, _Element | None]]:
    """Return the slots, the smallest codes in slot order and one gauge realising them.

    Connected components of the site graph are searched separately (a row's code depends only on
    its own component's reference atoms, so the lexicographic minimum of the interleaved list is the
    per-component minimum) and merged back into slot order; their state sets never multiply.

    :param tables: The group tables.
    :param letters: Site -> Wyckoff letter.
    :param number: Site -> 1-based number.
    :param rows: The rows.
    :return: ``(slots, codes, gauge per site)``; ``None`` for a site in no row.
    """
    oriented = [(a, b, g) if number[a] <= number[b] else (b, a, tables.inv(g)) for a, b, g in rows]
    slots = sorted((number[a], number[b]) for a, b, _ in oriented)
    root = {s: s for s in letters}

    def find(s: int) -> int:
        while root[s] != s:
            root[s] = root[root[s]]
            s = root[s]
        return s

    for a, b, _ in oriented:
        root[find(a)] = find(b)
    members: dict[int, list[int]] = {}
    for s in sorted(letters, key=number.__getitem__):
        members.setdefault(find(s), []).append(s)
    site_of = {n: s for s, n in number.items()}
    codes: list[_Element] = [_IDENTITY] * len(slots)
    gauge: dict[int, _Element | None] = dict.fromkeys(letters)
    for component, sites in members.items():
        indices = [x for x, (i, _) in enumerate(slots) if find(site_of[i]) == component]
        if not indices:
            continue
        local = {s: n for n, s in enumerate(sites)}
        part, local_gauge = _component_codes(
            tables,
            tuple(letters[s] for s in sites),
            tuple(sorted((local[a], local[b], g) for a, b, g in oriented if find(a) == component)),
        )
        for x, code in zip(indices, part, strict=True):
            codes[x] = code
        for s, e in zip(sites, local_gauge, strict=True):
            gauge[s] = e
    return slots, codes, gauge


# ---------------------------------------------------------------------------------------------
# the canonical form
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Realisation:
    """One description that produces the canonical net, in terms of the input sites.

    :param coset: Index into the target group's ``_GroupTables.cosets``, applied after ``cell``.
    :param classes: Input class ids in label order.
    :param letters: Canonical Wyckoff letter per input site.
    :param number: 1-based site number per input site.
    :param gauge: Accumulated reference-atom change per input site.
    :param cell: The ac cell change chosen from the net (identity outside groups 3 to 15), applied
        first; compiled by ``_cell``.
    """

    coset: int
    classes: tuple[Hashable, ...]
    letters: tuple[str, ...]
    number: tuple[int, ...]
    gauge: tuple[_Element, ...]
    cell: _Matrix2 = ((1, 0), (0, 1))


@dataclass(frozen=True)
class _CanonicalNet:
    """The canonical bonding net and every realisation of it.

    :param it_number: The target group (differs from the input only without preserved chirality).
    :param groups: Sorted Wyckoff letters per class, in label order.
    :param slots: 1-based site pairs ``(i, j)``, ``i <= j``, sorted.
    :param codes: One element (0-based operation) per slot.
    :param realisations: The descriptions giving this net, de-duplicated by ``(classes, letters,
        number)``; a coset repeating an earlier coset's letters and bond orbits is skipped.  Other
        ``(cell, coset, gauge)`` triples of an equal realisation are therefore not kept, which
        matters only for a future geometric round trip (D9).
    """

    it_number: int
    groups: tuple[tuple[str, ...], ...]
    slots: tuple[tuple[int, int], ...]
    codes: tuple[_Element, ...]
    realisations: tuple[_Realisation, ...]


def _canonical_net(
    it_number: int,
    sites: Sequence[tuple[Hashable, str]],
    rows: Iterable[tuple[int, int, _Element]],
    *,
    preserve_chirality: bool = True,
    own_bare: bool = False,
) -> _CanonicalNet:
    """Return the canonical bonding net of one standard-setting description.

    :param it_number: The International Tables number of the standard setting.
    :param sites: ``(class id, Wyckoff letter)`` per site, in any order.
    :param rows: ``(i, j, element)`` with 0-based site indices and elements relative to the
        sites' reference atoms; any lattice translation.
    :param preserve_chirality: Whether to keep the higher enantiomorphic group.
    :param own_bare: Search only the descriptions with the input's own bare key (its sorted class
        letter groups) instead of the smallest one; the constructors' canonicalization.
    :return: The canonical net and its realisations.
    :raises ValueError: If a row index is out of range or a letter or operation is unknown.
    """
    tables = _tables(it_number)
    sites = tuple((cls, letter) for cls, letter in sites)
    known = {position.letter for position in tables.spacegroup.wyckoff}
    for _, letter in sites:
        if letter not in known:
            raise ValueError(f"space group {it_number} has no Wyckoff letter {letter!r}")
    normalised = set()
    for i, j, g in rows:
        if not (0 <= i < len(sites) and 0 <= j < len(sites)):
            raise ValueError(f"row ({i}, {j}) refers to a site outside 0..{len(sites) - 1}")
        if len(g) != 4 or not 0 <= g[0] < tables.nops:
            raise ValueError(f"element {g!r} is not an operation index plus three lattice translations")
        if i > j:
            i, j, g = j, i, tables.inv(g)
        normalised.add((i, j, tables.canon(tables.letter(sites[i][1]).H, tables.letter(sites[j][1]).H, g, i == j)))
    return _search(it_number, sites, frozenset(normalised), preserve_chirality, own_bare)


def _class_groups(
    tables: _GroupTables, sites: Sequence[tuple[Hashable, str]], letters: Mapping[int, str]
) -> tuple[dict[Hashable, tuple[str, ...]], dict[Hashable, _Rank]]:
    """Return each class's sorted letter group and its rank in the class order."""
    by_class: dict[Hashable, list[str]] = {}
    for s, (cls, _) in enumerate(sites):
        by_class.setdefault(cls, []).append(letters[s])
    group_of = {cls: tuple(sorted(ls)) for cls, ls in by_class.items()}
    rank: dict[Hashable, _Rank] = {
        cls: class_order_key(ls, sum(tables.letter(x).multiplicity for x in ls))[:2] for cls, ls in group_of.items()
    }
    return group_of, rank


@lru_cache(maxsize=4096)
def _search(
    it_number: int,
    sites: tuple[tuple[Hashable, str], ...],
    rows: frozenset[_Row],
    preserve_chirality: bool,
    own_bare: bool = False,
) -> _CanonicalNet:
    """Search every cell, coset, class order, numbering and gauge for the smallest net (memoised)."""
    tables = _tables(it_number)
    ordered = sorted(rows)
    target = _chirality_target(tables.spacegroup, preserve_chirality).it_number
    if target != it_number:
        transfer = _transfer(it_number, target)
        tables = _tables(target)
        ordered = [_moved(tables, transfer, sites, row) for row in ordered]
        sites = tuple((cls, transfer.letter(letter)[0]) for cls, letter in sites)
    connected = {sites[a][0] for a, _, _ in ordered} | {sites[b][0] for _, b, _ in ordered}
    sites_of: dict[Hashable, list[int]] = {}
    for s, (cls, _) in enumerate(sites):
        sites_of.setdefault(cls, []).append(s)
    best: _Candidate | None = None
    result: tuple[list[tuple[int, int]], list[_Element], list[tuple[str, ...]]] | None = None
    realisations: dict[tuple[object, ...], _Realisation] = {}
    seen: set[tuple[object, ...]] = set()
    # monoclinic: the cells chosen from the net come first (task 4.2); a net of rank 0 with a
    # component periodic along b gets every linear class from its cells, so the tabulated set suffices
    cells: tuple[_Matrix2, ...] = (((1, 0), (0, 1)),)
    indices: Sequence[int] = range(len(tables.cosets))
    if 3 <= target <= 15:
        cells, rank0 = _cell_candidates(tables, sites, ordered)
        if rank0:
            indices = tables.tabulated
    # filter by the bare key first: only descriptions reaching the smallest (or the own) bare key compete
    described = []
    for X in cells:
        cell_sites, cell_rows = sites, ordered
        if X != ((1, 0), (0, 1)):
            cell = _cell(target, X)
            cell_sites = tuple((cls, cell.letter(letter)[0]) for cls, letter in sites)
            cell_rows = [_moved(tables, cell, sites, row) for row in ordered]
        for index in indices:
            _checkpoint()
            coset = tables.cosets[index]
            letters = {s: coset.letter(letter)[0] for s, (_, letter) in enumerate(cell_sites)}
            group_of, rank = _class_groups(tables, sites, letters)
            described.append((X, cell_sites, cell_rows, index, letters, group_of, rank, tuple(sorted(rank.values()))))
    if own_bare:
        least = tuple(sorted(_class_groups(tables, sites, dict(enumerate(letter for _, letter in sites)))[1].values()))
    else:
        least = min(entry[-1] for entry in described)
    for X, cell_sites, cell_rows, index, letters, group_of, rank, bare_key in described:
        _checkpoint()
        if bare_key != least:
            continue
        coset = tables.cosets[index]
        moved = [_moved(tables, coset, cell_sites, row) for row in cell_rows]
        # a coset that yields the same letters and bond orbits as an earlier one adds no realisation
        seen_key = (
            tuple(letters.values()),
            frozenset(
                (a, b, tables.canon(tables.letter(letters[a]).H, tables.letter(letters[b]).H, g, a == b))
                for a, b, g in moved
            ),
        )
        if seen_key in seen:
            continue
        seen.add(seen_key)
        # classes with identical letter groups are interchangeable in the anonymous label
        tied: dict[_Rank, list[Hashable]] = {}
        for cls in group_of:
            tied.setdefault(rank[cls], []).append(cls)
        # a tie group of one-site classes is one block of interchangeable sites: the numbering search
        # orders it, instead of a factorial enumeration of class orders
        single = {
            cls
            for group in tied.values()
            if len(group) > 1 and all(len(group_of[c]) == 1 for c in group)
            for cls in group
        }
        parts = [
            [tuple(sorted(tied[key], key=str))] if tied[key][0] in single else _arrangements(tied[key], connected)
            for key in sorted(tied)
        ]
        for arrangement in product(*parts):
            _checkpoint()
            blocks = []
            for part in arrangement:
                if part[0] in single:
                    blocks.append([s for cls in part for s in sites_of[cls]])
                else:
                    blocks.extend(
                        [s for s in sites_of[cls] if letters[s] == letter]
                        for cls in part
                        for letter in sorted(set(group_of[cls]))
                    )
            pattern, numberings = _minimal_numberings(blocks, [(a, b) for a, b, _ in moved])
            if best is not None and (bare_key, pattern) > (best[0], best[1]):
                continue
            for number in numberings:
                ordered_classes = [
                    cls
                    for part in arrangement
                    for cls in (sorted(part, key=lambda c: number[sites_of[c][0]]) if part[0] in single else part)
                ]
                slots, codes, gauge = _exact_codes(tables, letters, number, moved)
                candidate = (bare_key, pattern, tuple(_key(c) for c in codes))
                if best is None or candidate < best:
                    best, result, realisations = candidate, (slots, codes, [group_of[c] for c in ordered_classes]), {}
                if candidate == best:
                    real = _Realisation(
                        index,
                        tuple(ordered_classes),
                        tuple(letters[s] for s in range(len(sites))),
                        tuple(number[s] for s in range(len(sites))),
                        tuple(gauge[s] or _IDENTITY for s in range(len(sites))),
                        X,
                    )
                    realisations.setdefault((real.classes, real.letters, real.number), real)
    assert result is not None
    slots, codes, groups = result
    return _CanonicalNet(target, tuple(groups), tuple(slots), tuple(codes), tuple(realisations.values()))


def _arrangements(group: Sequence[Hashable], connected: set[Hashable]) -> list[tuple[Hashable, ...]]:
    """Return the orders of one tie group of classes that can differ in pattern, codes or D3 minimum.

    Classes without rows are interchangeable tokens: only their positions among the connected classes
    matter, and filling those positions in sorted order gives the smallest name tuple.

    :param group: The tied classes.
    :param connected: The classes that appear in some row.
    :return: The distinct arrangements.
    """
    live = [c for c in group if c in connected]
    tokens = sorted((c for c in group if c not in connected), key=str)
    out = []
    for spots in combinations(range(len(group)), len(tokens)):
        for perm in permutations(live):
            fill, rest = iter(tokens), iter(perm)
            out.append(tuple(next(fill) if p in spots else next(rest) for p in range(len(group))))
    return out
