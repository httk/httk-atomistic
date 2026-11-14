# ruff: noqa
"""Reference implementation: canonical bonding-net descriptor on top of an httk bare prototype
(classes in Formulatype order: largest site count first, then Wyckoff letters).

Not production code.  It is the tested specification that the httk-atomistic implementation has to
reproduce (see PROTONET_IMPLEMENTATION.md).  It uses only exact integer / rational arithmetic and
httk-atomistic's own symmetry tables:

    Spacegroup.standard(n).wyckoff / .wyckoff_position(letter).representative
    canonical_classification._actions(spacegroup)       (finite same-group action domain)

Vocabulary
    element     (k, n1, n2, n3): operation k (0-based index into the general position's branch list,
                identity is 0) followed by the lattice translation n.  x -> W_k x + w_k + n.
    reference   atom of a site: the first-listed Wyckoff coordinate (WyckoffPosition.representative)
                evaluated at that site's free parameters, NOT reduced modulo one.
    row         (i, j, g): the reference atom of site i is bonded to g applied to the reference atom
                of site j.  One row stands for the whole symmetry orbit of that bond.
    gauge       re-choosing the reference atom of site s as a_s * (old reference) changes every row
                (i, j, g) into (i, j, a_i * g * a_j^-1).  a_s must keep the atom on the first-listed
                coordinate, i.e. a_s normalises the site-symmetry group.

The minimum over all reference-atom choices is found exactly (integer lattice arithmetic, no search
range).  An earlier version searched a +-1 / +-2 lattice box and could miss the minimum.

Test copy: this is the oracle for tests/test_protonet_canonical.py, copied from the workspace
docs/protonet_followup/protonet_reference.py with only ruff formatting and this note added.
Follow-up 5 (shapes) was ported by the P-b implementation agent from the design note
FU5_SHAPES_NOTE.md: bond counts from exact neighbour points, the listing order (B5) and the
comparison order (B6), and the net grammar with shapes.  The defaults table follows decision S2
(two bonds: A2), which settled after the note, and follow-up 7 (five, seven and eight bonds: SPY5,
TPRS7, DD8).
"""

import math
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product

from httk.atomistic.symmetry.canonical_classification import _actions
from httk.atomistic.symmetry.spacegroup import Spacegroup

PROBE = (Fraction(13, 97), Fraction(29, 101), Fraction(41, 103))  # generic free-parameter values
ROOT_BOX = 1  # only for the scramble test: lattice range of random general moves
FLAT_BOX = 2  # lattice range in which the finitely many classes of gauge moves of a
# special position are looked up (not a search range for the minimum)


# ---------------------------------------------------------------------------------------------
# exact helpers
# ---------------------------------------------------------------------------------------------
def _fractions(value):
    out = value.to_fractions()
    return (
        tuple(tuple(Fraction(x) for x in row) for row in out)
        if isinstance(out[0], (tuple, list))
        else tuple(Fraction(x) for x in out)
    )


def _mat_vec(M, v):
    return tuple(sum(M[r][c] * v[c] for c in range(3)) for r in range(3))


def _mat_mat(A, B):
    return tuple(tuple(sum(A[r][k] * B[k][c] for k in range(3)) for c in range(3)) for r in range(3))


def _rank(rows):
    """rank of a small rational matrix (exact Gaussian elimination)"""
    rows, rank = [list(map(Fraction, r)) for r in rows], 0
    for col in range(len(rows[0])):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(rank + 1, len(rows)):
            f = rows[r][col] / rows[rank][col]
            rows[r] = [x - f * y for x, y in zip(rows[r], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _is_int_vector(v):
    return all(Fraction(x).denominator == 1 for x in v)


def _column_echelon(rows, ncols, reduce=False):
    """Integer column operations bringing A (given by rows) to column echelon form H = A U.
    Returns (H, U, pivots): pivots[i] is the row of the positive leading entry of column i; the
    columns len(pivots).. of H are zero, so the same columns of U span the integer kernel of A."""
    A = [list(r) for r in rows]
    U = [[1 if i == j else 0 for j in range(ncols)] for i in range(ncols)]
    pivots, col = [], 0
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


def _coset_minimum(H, pivots, b):
    """The smallest vector m = b + H z (z integer) under (m.m, m1, m2, m3), with its coefficients z.
    H has three rows in column echelon form.  Exact: every vector not longer than the best one
    found so far is enumerated, pivot row by pivot row."""
    norm = lambda m: m[0] * m[0] + m[1] * m[1] + m[2] * m[2]
    m, z0 = list(b), []
    for i, p in enumerate(pivots):  # first candidate: round in each pivot row
        d = H[p][i]
        z = -((2 * m[p] + d) // (2 * d))
        z0.append(z)
        m = [m[x] + z * H[x][i] for x in range(3)]
    best = [(norm(m), tuple(m)), tuple(z0)]

    def descend(i, m, z, used):
        if i == len(pivots):
            cand = (norm(m), tuple(m))
            if cand < best[0]:
                best[0], best[1] = cand, tuple(z)
            return
        p = pivots[i]
        d = H[p][i]
        room = best[0][0] - used
        if room < 0:
            return
        s = math.isqrt(room)
        for zi in range(-((s + m[p]) // d), (s - m[p]) // d + 1):
            m2 = [m[x] + zi * H[x][i] for x in range(3)]
            descend(i + 1, m2, z + [zi], used + m2[p] * m2[p])

    descend(0, list(b), [], 0)
    return best[0][0], best[0][1], best[1]


def key(e):
    """Order of elements inside a double coset: shortest translation, then operation, then translation."""
    return (e[1] * e[1] + e[2] * e[2] + e[3] * e[3], e[0], e[1], e[2], e[3])


IDENTITY = (0, 0, 0, 0)


# ---------------------------------------------------------------------------------------------
# per-space-group tables (cache these; they do not depend on the structure)
# ---------------------------------------------------------------------------------------------
class GroupTables:
    def __init__(self, it_number):
        self.spacegroup = sg = Spacegroup.standard(it_number)
        general = max(sg.wyckoff, key=lambda w: (w.free_count, w.multiplicity))
        ops = [branch.operation for branch in general.branches]  # normative operation order
        assert ops[0].is_identity(), "operation 1 must be the identity"
        self.general_letter = general.letter
        self.nops = len(ops)
        self.W = [_fractions(op.matrix) for op in ops]
        self.w = [_fractions(op.vector) for op in ops]
        assert all(_is_int_vector(row) for W in self.W for row in W)
        self.Wi = [tuple(int(x) for row in W for x in row) for W in self.W]  # flat ints, hot path
        # multiplication and inverse tables: (k1, k2) -> (k3, integer residue)
        index = {W: [] for W in self.W}
        for k, W in enumerate(self.W):
            index[W].append(k)

        def locate(W, w):
            for k in index[W]:
                r = tuple(a - b for a, b in zip(w, self.w[k]))
                if _is_int_vector(r):
                    return k, tuple(int(x) for x in r)
            raise ValueError("operation list is not closed")

        self.mulk = [
            [
                locate(
                    _mat_mat(self.W[a], self.W[b]),
                    tuple(x + y for x, y in zip(_mat_vec(self.W[a], self.w[b]), self.w[a])),
                )
                for b in range(self.nops)
            ]
            for a in range(self.nops)
        ]
        self.invk = []
        for a in range(self.nops):
            b = next(b for b in range(self.nops) if self.mulk[a][b][0] == 0)  # W_a W_b = 1
            r = self.mulk[a][b][1]  # a*b = identity + r
            # a^-1 = b - (W_b)(r)  as a translation correction: (b, s) with  a*(b, s) = identity
            Wa = self.Wi[a]
            # a*(b,s) = (0, r + W_a s) = identity  ->  W_a s = -r  ->  s = -W_a^-1 r = -W_b r
            Wb = self.Wi[b]
            s = tuple(-(Wb[3 * i] * r[0] + Wb[3 * i + 1] * r[1] + Wb[3 * i + 2] * r[2]) for i in range(3))
            self.invk.append((b, s))
        self._letters = {}

    # --- element algebra ---------------------------------------------------------------------
    def mul(self, a, b):
        k, r = self.mulk[a[0]][b[0]]
        W = self.Wi[a[0]]
        return (
            k,
            a[1] + W[0] * b[1] + W[1] * b[2] + W[2] * b[3] + r[0],
            a[2] + W[3] * b[1] + W[4] * b[2] + W[5] * b[3] + r[1],
            a[3] + W[6] * b[1] + W[7] * b[2] + W[8] * b[3] + r[2],
        )

    def inv(self, a):
        k, s = self.invk[a[0]]
        W = self.Wi[k]
        return (
            k,
            s[0] - (W[0] * a[1] + W[1] * a[2] + W[2] * a[3]),
            s[1] - (W[3] * a[1] + W[4] * a[2] + W[5] * a[3]),
            s[2] - (W[6] * a[1] + W[7] * a[2] + W[8] * a[3]),
        )

    def apply(self, e, point):
        W, w = self.W[e[0]], self.w[e[0]]
        return tuple(sum(W[r][c] * point[c] for c in range(3)) + w[r] + e[1 + r] for r in range(3))

    # --- per Wyckoff letter --------------------------------------------------------------------
    def letter(self, letter):
        """probe reference point, chart (M, v), site-symmetry group H, gauge moves A (transversal of N(H)/H)"""
        if letter not in self._letters:
            position = self.spacegroup.wyckoff_position(letter)
            chart = position.representative.operation
            M, v = _fractions(chart.matrix), _fractions(chart.vector)
            ref = tuple(x + y for x, y in zip(_mat_vec(M, PROBE), v))
            rank = _rank([list(row) for row in M])
            on_flat = lambda p: _rank([list(M[r]) + [p[r] - ref[r]] for r in range(3)]) == rank
            H = []
            for k in range(self.nops):
                n = tuple(a - b for a, b in zip(ref, self.apply((k, 0, 0, 0), ref)))
                if _is_int_vector(n):
                    H.append((k,) + tuple(int(x) for x in n))
            assert len(H) * position.multiplicity == self.nops
            box = FLAT_BOX if len(H) > 1 else ROOT_BOX
            moves, seen = [], set()
            for k in range(self.nops):
                base = self.apply((k, 0, 0, 0), ref)
                lo = tuple(-(x.numerator // x.denominator) for x in base)  # bring into the home cell
                for d in product(range(-box, box + 1), repeat=3):
                    e = (k, lo[0] + d[0], lo[1] + d[1], lo[2] + d[2])
                    p = self.apply(e, ref)
                    if on_flat(p) and p not in seen:
                        seen.add(p)
                        moves.append(e)
            if len(H) == 1:
                lattice = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                reps = [(k, 0, 0, 0) for k in range(self.nops)]
            else:
                # lattice translations t that keep the flat: W_h t = t for every h in H
                rows = [
                    [self.Wi[h[0]][3 * r + x] - (1 if r == x else 0) for x in range(3)] for h in H for r in range(3)
                ]
                _, U, piv = _column_echelon(rows, 3)
                lattice = [tuple(U[x][col] for x in range(3)) for col in range(len(piv), 3)]
                Lh, _, Lp = _column_echelon([[v[x] for v in lattice] for x in range(3)], len(lattice))

                def in_lattice(t):
                    t = list(t)
                    for i, p in enumerate(Lp):
                        if t[p] % Lh[p][i]:
                            return False
                        q = t[p] // Lh[p][i]
                        t = [t[x] - q * Lh[x][i] for x in range(3)]
                    return not any(t)

                reps = []  # gauge moves modulo translations of the flat
                for e in moves:
                    if not any((d := self.mul(e, self.inv(rho)))[0] == 0 and in_lattice(d[1:]) for rho in reps):
                        reps.append(e)
            self._letters[letter] = dict(
                ref=ref,
                on_flat=on_flat,
                H=H,
                moves=moves,
                reps=reps,
                lattice=lattice,
                general=len(H) == 1,
                multiplicity=position.multiplicity,
                free_count=position.free_count,
            )
        return self._letters[letter]

    def canon(self, Hi, Hj, g, same_site=False):
        """smallest element of the double coset Hi g Hj (and of its inverse for a bond inside one orbit)"""
        best = None
        for x in (g, self.inv(g)) if same_site else (g,):
            for h1 in Hi:
                y = self.mul(h1, x)
                for h2 in Hj:
                    z = self.mul(y, h2)
                    if best is None or key(z) < key(best):
                        best = z
        return best

    # --- equivalent descriptions ---------------------------------------------------------------
    @property
    def cosets(self):
        """httk's finite same-group action domain, one representative per coset of the group, each
        compiled to: operation relabelling k -> (k', c), integer matrix P for translations, and per
        letter the target letter and the element b that puts the moved reference back on its chart."""
        if not hasattr(self, "_cosets"):
            kept = []
            for action in _actions(self.spacegroup):
                P = _fractions(action.matrix)
                if not all(_is_int_vector(row) for row in P):
                    continue  # not an automorphism of this lattice basis
                if any(self._in_group(other.inverse() * action) for other in kept):
                    continue  # same coset of the space group
                kept.append(action)
            self._cosets = [
                self._compile(action, _fractions(action.matrix), _fractions(action.vector)) for action in kept
            ]
        return self._cosets

    def _in_group(self, operation):
        """member of the group, up to a lattice translation and a continuous (polar) origin shift"""
        D, dv = _fractions(operation.matrix), _fractions(operation.vector)
        fixed = [axis for axis in range(3) if axis not in self.polar_axes]
        return any(self.W[k] == D and _is_int_vector([dv[a] - self.w[k][a] for a in fixed]) for k in range(self.nops))

    @property
    def polar_axes(self):
        """cell axes t with W t = t for every operation: origin shifts along them are free and commute
        with the whole group, so they never change a code"""
        if not hasattr(self, "_polar"):
            self._polar = tuple(
                axis
                for axis in range(3)
                if all(W[3 * r + axis] == (1 if r == axis else 0) for W in self.Wi for r in range(3))
            )
        return self._polar

    def _compile(self, action, P, p):
        inverse = action.inverse()
        Pinv = _fractions(inverse.matrix)
        relabel = []
        for k in range(self.nops):
            Wc = _mat_mat(_mat_mat(P, self.W[k]), Pinv)
            wc = tuple(a + b - c for a, b, c in zip(_mat_vec(P, self.w[k]), p, _mat_vec(Wc, p)))
            found = None
            for k2 in range(self.nops):
                if self.W[k2] == Wc:
                    c = tuple(a - b for a, b in zip(wc, self.w[k2]))
                    if _is_int_vector(c):
                        found = (k2, tuple(int(x) for x in c))
                        break
            if found is None:
                raise ValueError("action does not normalise the group")
            relabel.append(found)
        Pi = tuple(int(x) for row in P for x in row)
        letters = {}
        for position in self.spacegroup.wyckoff:
            src = self.letter(position.letter)
            q = tuple(a + b for a, b in zip(_mat_vec(P, src["ref"]), p))
            hit = None
            for target in self.spacegroup.wyckoff:
                if target.multiplicity != position.multiplicity or target.free_count != position.free_count:
                    continue
                tgt = self.letter(target.letter)
                for k in range(self.nops):
                    base = self.apply((k, 0, 0, 0), q)
                    lo = tuple(-(x.numerator // x.denominator) for x in base)
                    for d in sorted(product(range(-2, 3), repeat=3), key=lambda d: sum(x * x for x in d)):
                        e = (k, lo[0] + d[0], lo[1] + d[1], lo[2] + d[2])
                        if tgt["on_flat"](self.apply(e, q)):
                            hit = (target.letter, e)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit is None:
                raise ValueError(f"cannot map Wyckoff letter {position.letter}")
            letters[position.letter] = hit
        return dict(P=Pi, p=p, action=action, relabel=relabel, letters=letters)

    def conjugate(self, coset, g):
        k, c = coset["relabel"][g[0]]
        P = coset["P"]
        return (
            k,
            P[0] * g[1] + P[1] * g[2] + P[2] * g[3] + c[0],
            P[3] * g[1] + P[4] * g[2] + P[5] * g[3] + c[1],
            P[6] * g[1] + P[7] * g[2] + P[8] * g[3] + c[2],
        )


@lru_cache(maxsize=None)
def tables(it_number):
    return GroupTables(it_number)


# ---------------------------------------------------------------------------------------------
# numbering of sites: smallest list of (i, j) pairs
# ---------------------------------------------------------------------------------------------
def _minimal_numberings(blocks, rows):
    """blocks: lists of interchangeable sites in label order; rows: (site, site).  Returns the
    smallest sorted pair list and every numbering (site -> number, 1-based) that realises it."""
    nsites = sum(len(b) for b in blocks)
    block_of, first = {}, []
    start = 0
    for index, block in enumerate(blocks):
        first.append(start)
        for site in block:
            block_of[site] = index
        start += len(block)
    owner = [index for index, block in enumerate(blocks) for _ in block]  # number -> block
    neighbours = {s: Counter() for b in blocks for s in b}
    for a, b in rows:
        neighbours[a][b] += 1
        if a != b:
            neighbours[b][a] += 1
    isolated = {s for s, c in neighbours.items() if not c}
    total = len(rows)
    best = {"pattern": None, "numberings": []}

    def descend(m, number, free, pattern):
        if m == nsites:
            if best["pattern"] is None or pattern < best["pattern"]:
                best["pattern"], best["numberings"] = pattern, []
            if pattern == best["pattern"]:
                best["numberings"].append(dict(number))
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
            pending = {}
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
                if best["pattern"] is not None and len(best["pattern"]) == total:
                    head = best["pattern"][: len(pattern3)]
                    if pattern3 > head:
                        continue
                descend(m + 1, number3, free3, pattern3)

    descend(0, {}, list(first), ())
    return best["pattern"], [{s: n + 1 for s, n in numbering.items()} for numbering in best["numberings"]]


# ---------------------------------------------------------------------------------------------
# reference atoms: smallest list of codes for one numbering, exact
#
# A reference atom is  a_s = (k_s, c_s + M_s v):  a fixed operation, a fixed lattice translation and
# a part that is linear in a vector v of integer unknowns.  A general position contributes three
# unknowns (any lattice translation), a special position one unknown per independent translation
# along its own flat, and its finitely many other moves are tried one by one.  The translation of
# a_i g a_j^-1 is then  b + M v,  so the smallest code of a row is the smallest vector of a lattice
# coset, found exactly.  Fixing that vector is a linear condition on v; it is solved, substituted,
# and the next row is treated the same way.  No search range is involved.
# ---------------------------------------------------------------------------------------------
def _sym(e, D):
    return (e[0], tuple(e[1:]), ((0,) * D,) * 3)


def _smul(T, a, b):
    k, r = T.mulk[a[0]][b[0]]
    W = T.Wi[a[0]]
    c = tuple(a[1][x] + W[3 * x] * b[1][0] + W[3 * x + 1] * b[1][1] + W[3 * x + 2] * b[1][2] + r[x] for x in range(3))
    M = tuple(
        tuple(
            a[2][x][d] + W[3 * x] * b[2][0][d] + W[3 * x + 1] * b[2][1][d] + W[3 * x + 2] * b[2][2][d]
            for d in range(len(a[2][0]))
        )
        for x in range(3)
    )
    return (k, c, M)


def _sinv(T, a):
    k, s = T.invk[a[0]]
    W = T.Wi[k]
    c = tuple(s[x] - (W[3 * x] * a[1][0] + W[3 * x + 1] * a[1][1] + W[3 * x + 2] * a[1][2]) for x in range(3))
    M = tuple(
        tuple(
            -(W[3 * x] * a[2][0][d] + W[3 * x + 1] * a[2][1][d] + W[3 * x + 2] * a[2][2][d])
            for d in range(len(a[2][0]))
        )
        for x in range(3)
    )
    return (k, c, M)


def _put(assign, pos, value):
    out = list(assign)
    out[pos] = value
    return tuple(out)


def _with_new_site(assign, D, pos, rep, lattice):
    """give the site at ``pos`` the move ``rep`` plus free translations along its lattice"""
    d = len(lattice)
    grown = [None if e is None else (e[0], e[1], tuple(row + (0,) * d for row in e[2])) for e in assign]
    grown[pos] = (rep[0], tuple(rep[1:]), tuple((0,) * D + tuple(v[x] for v in lattice) for x in range(3)))
    return tuple(grown), D + d


def _assignments(T, info, assign, D, pi, pj, si, sj, g):
    """every way to give the two ends of a row a reference atom, as far as they have none yet"""
    ai, aj = assign[pi], assign[pj]
    if si == sj:
        if ai is not None:
            yield assign, D
        else:
            for rep in info[si]["reps"]:
                yield _with_new_site(assign, D, pi, rep, info[si]["lattice"])
        return
    if ai is not None or (aj is not None and info[si]["general"]):
        starts = [(assign, D)]
    else:
        starts = [_with_new_site(assign, D, pi, rep, info[si]["lattice"]) for rep in info[si]["reps"]]
    for A1, D1 in starts:
        ai, aj = A1[pi], A1[pj]
        if ai is None:  # general position next to a fixed atom: a_i = h a_j g^-1
            y = _smul(T, aj, _sinv(T, _sym(g, D1)))
            for h in info[sj]["H"]:
                yield _put(A1, pi, _smul(T, _sym(h, D1), y)), D1
        elif aj is not None:
            yield A1, D1
        elif info[sj]["general"]:  # general position next to a fixed atom: a_j = h a_i g
            y = _smul(T, ai, _sym(g, D1))
            for h in info[si]["H"]:
                yield _put(A1, pj, _smul(T, _sym(h, D1), y)), D1
        else:
            for rep in info[sj]["reps"]:
                yield _with_new_site(A1, D1, pj, rep, info[sj]["lattice"])


def _row_minima(T, Hi, Hj, x, D, same_site):
    """smallest code of the double coset of the symbolic element x over all values of the unknowns:
    yields (key, code, constraint) for every way to reach each candidate; the caller keeps the minima"""
    for xs in (x, _sinv(T, x)) if same_site else (x,):
        for h1 in Hi:
            y = _smul(T, _sym(h1, D), xs)
            free = D and any(any(row) for row in y[2])
            if free:
                H, U, pivots = _column_echelon(y[2], D)
            for h2 in Hj:
                z = _smul(T, y, _sym(h2, D))
                if free:
                    n2, m, coeff = _coset_minimum(H, pivots, z[1])
                    yield (n2, z[0]) + m, (z[0],) + m, (U, len(pivots), coeff)
                else:
                    m = z[1]
                    yield (m[0] * m[0] + m[1] * m[1] + m[2] * m[2], z[0]) + m, (z[0],) + m, None


def _substitute(assign, D, constraint):
    """impose the linear condition that fixed a row's code: v = v0 + K w"""
    if constraint is None:
        return assign, D
    U, r, coeff = constraint
    v0 = [sum(U[d][i] * coeff[i] for i in range(r)) for d in range(D)]
    out = []
    for e in assign:
        if e is None:
            out.append(None)
            continue
        c = tuple(e[1][x] + sum(e[2][x][d] * v0[d] for d in range(D)) for x in range(3))
        M = tuple(tuple(sum(e[2][x][d] * U[d][col] for d in range(D)) for col in range(r, D)) for x in range(3))
        out.append((e[0], c, M))
    return tuple(out), D - r


def _normalise(assign, D):
    """a unique description of the same set of reference-atom choices, so equal states merge"""
    if D == 0:
        return assign, 0
    used = [p for p, e in enumerate(assign) if e is not None]
    stack = [list(assign[p][2][x]) for p in used for x in range(3)]
    const = [assign[p][1][x] for p in used for x in range(3)]
    H, _, pivots = _column_echelon(stack, D, reduce=True)
    for i, p in enumerate(pivots):
        q = const[p] // H[p][i]
        if q:
            const = [const[x] - q * H[x][i] for x in range(len(const))]
    out = list(assign)
    for n, p in enumerate(used):
        out[p] = (
            assign[p][0],
            tuple(const[3 * n : 3 * n + 3]),
            tuple(tuple(H[3 * n + x][: len(pivots)]) for x in range(3)),
        )
    return tuple(out), len(pivots)


def _exact_codes(T, letters, number, rows):
    """rows: (site_a, site_b, g).  Returns (slots, codes in slot order, one gauge that realises them)."""
    info = {s: T.letter(l) for s, l in letters.items()}
    oriented = [(a, b, g) if number[a] <= number[b] else (b, a, T.inv(g)) for a, b, g in rows]
    slots = sorted((number[a], number[b]) for a, b, _ in oriented)
    site_of = {n: s for s, n in number.items()}
    order = sorted(letters, key=lambda s: number[s])
    position = {s: i for i, s in enumerate(order)}
    states = {((None,) * len(order), 0, frozenset())}
    codes = []
    for i, j in slots:
        si, sj = site_of[i], site_of[j]
        pi, pj = position[si], position[sj]
        Hi, Hj = info[si]["H"], info[sj]["H"]
        best, keep = None, []
        for assign, D, used in states:
            for r, (a, b, g) in enumerate(oriented):
                if r in used or number[a] != i or number[b] != j:
                    continue
                for assign2, D2 in _assignments(T, info, assign, D, pi, pj, si, sj, g):
                    x = _smul(T, _smul(T, assign2[pi], _sym(g, D2)), _sinv(T, assign2[pj]))
                    for k_, code, constraint in _row_minima(T, Hi, Hj, x, D2, si == sj):
                        if best is None or k_ < best:
                            best, keep = k_, []
                        if k_ == best:
                            keep.append((code, assign2, D2, constraint, used | {r}))
        codes.append(keep[0][0])
        states = set()
        for code, assign2, D2, constraint, used2 in keep:
            states.add(_normalise(*_substitute(assign2, D2, constraint)) + (used2,))
    assign = next(iter(states))[0]
    gauge = {
        s: (None if assign[position[s]] is None else (assign[position[s]][0],) + assign[position[s]][1]) for s in order
    }
    return slots, codes, gauge


# ---------------------------------------------------------------------------------------------
# the canonical form
# ---------------------------------------------------------------------------------------------
def bond_counts(T, sites, rows):
    """Number of distinct atoms bonded to the reference atom of each site, from exact neighbour points.
    An atom is (site, point): two sites at one point are two atoms."""
    refs = [T.letter(letter)["ref"] for _, letter in sites]
    H = [T.letter(letter)["H"] for _, letter in sites]
    atoms = [set() for _ in sites]
    for a, b, g in rows:
        for h in H[a]:
            atoms[a].add((b, T.apply(T.mul(h, g), refs[b])))
        for h in H[b]:
            atoms[b].add((a, T.apply(T.mul(h, T.inv(g)), refs[a])))
    return [len(x) for x in atoms]


def listing_positions(counts, pairs, number):
    """1-based listing position per site: larger count, then top sites, then smaller number (B5)."""
    top = {}
    for s in range(len(counts)):
        neighbours = [b for a, b in pairs if a == s] + [a for a, b in pairs if b == s]
        top[s] = not any(counts[t] > counts[s] for t in neighbours)
    order = sorted(range(len(counts)), key=lambda s: (-counts[s], 0 if top[s] else 1, number[s]))
    return {s: p + 1 for p, s in enumerate(order)}


def canonical_net(it_number, sites, rows, class_order="formula", hand=1, shapes=None):
    """sites: [(class_id, wyckoff_letter)] in any order; rows: [(i, j, (k, n1, n2, n3))] with 0-based
    site indices and elements relative to the sites' reference atoms.

    The net is always the anonymous one (Protonettype).  For an assigned structure pass the species
    names as class ids: 'classes' is then the species order of the Protonetstructure label, i.e. the
    class order of the canonical net, and where several realisations give the same net, the
    smallest tuple of names among them.  That order can differ from the order in the bare
    protostructure label, which is decided by names alone.

    class_order="formula" (the specification) orders the classes as Formulatype does: largest site
    count first, then sorted Wyckoff letters.  class_order="letters" is the order httk-atomistic
    used before the reorder (letters only) and is kept only to reproduce old strings.

    The hand is kept (preserve_chirality=True): in a Sohncke group only the cosets whose
    determinant has the sign of ``hand``, the handedness of the given description's cell, are
    descriptions of the same crystal; the others describe its mirror image.

    Returns a dict: 'groups' (canonical Wyckoff-letter groups of the bare prototype), 'net' (the
    string after '/'), 'classes' (class ids in label order), and 'realisations': every
    (coset, numbering, gauge) that produces the canonical string, for the species tie-break, for
    transporting parameters into the canonical gauge, and for automorphisms.
    """
    T = tables(it_number)
    sohncke = all(operation.determinant() > 0 for operation in T.spacegroup.symmetry_operations)
    shapes = [None] * len(sites) if shapes is None else list(shapes)
    counts = bond_counts(T, sites, rows)  # site invariants: the input description serves
    pairs = [(a, b) for a, b, _ in rows]
    best = None
    for coset in T.cosets:
        if sohncke and hand * coset["action"].determinant() < 0:
            continue
        letters = {s: coset["letters"][letter][0] for s, (_, letter) in enumerate(sites)}
        moved = []
        for a, b, g in rows:
            ba, bb = coset["letters"][sites[a][1]][1], coset["letters"][sites[b][1]][1]
            moved.append((a, b, T.mul(T.mul(ba, T.conjugate(coset, g)), T.inv(bb))))
        by_class = {}
        for s, (cls, _) in enumerate(sites):
            by_class.setdefault(cls, []).append(letters[s])
        group_of = {cls: tuple(sorted(ls)) for cls, ls in by_class.items()}
        if class_order == "formula":
            rank = {cls: (-sum(T.letter(l)["multiplicity"] for l in ls), ls) for cls, ls in group_of.items()}
        elif class_order == "letters":
            rank = dict(group_of)
        else:
            raise ValueError(f"unknown class order {class_order!r}")
        bare_key = tuple(sorted(rank.values()))
        if best is not None and bare_key > best["bare"]:
            continue
        # classes with identical letter groups are interchangeable in the anonymous label
        tied = {}
        for cls in group_of:
            tied.setdefault(rank[cls], []).append(cls)
        for arrangement in product(*[permutations(tied[g]) for g in sorted(tied)]):
            ordered_classes = [cls for part in arrangement for cls in part]
            blocks = []
            for cls in ordered_classes:
                for letter in sorted(set(group_of[cls])):
                    blocks.append([s for s, (c, _) in enumerate(sites) if c == cls and letters[s] == letter])
            pattern, numberings = _minimal_numberings(blocks, [(a, b) for a, b, _ in moved])
            if best is not None and (bare_key, pattern) > (best["bare"], best["pattern"]):
                continue
            for number in numberings:
                pos = listing_positions(counts, pairs, number)
                pos_slots, codes, gauge = _exact_codes(T, letters, pos, moved)  # codes minimised in label order
                number_of_pos = {pos[s]: number[s] for s in pos}
                slots = [(number_of_pos[p], number_of_pos[q]) for p, q in pos_slots]
                by_number = sorted(range(len(sites)), key=lambda s: number[s])
                cand = (bare_key, pattern, slots, [key(c) for c in codes], [shapes[s] or "" for s in by_number])
                if best is None or cand < best["cand"]:
                    best = dict(
                        cand=cand,
                        bare=bare_key,
                        pattern=pattern,
                        slots=slots,
                        codes=codes,
                        groups=[group_of[c] for c in ordered_classes],
                        shapes=[shapes[s] for s in by_number],
                        realisations=[],
                    )
                if cand == best["cand"]:
                    best["realisations"].append(
                        dict(coset=coset, number=number, gauge=gauge, classes=ordered_classes, letters=letters)
                    )
    return dict(
        cand=best["cand"],
        groups=best["groups"],
        net=render(best["slots"], best["codes"], best["shapes"]),
        slots=best["slots"],
        shapes=best["shapes"],
        codes=best["codes"],
        classes=min((r["classes"] for r in best["realisations"]), key=lambda names: [str(n) for n in names]),
        realisations=best["realisations"],
    )


# ---------------------------------------------------------------------------------------------
# notation:   8I4~1.2_9SS4~3.4_10~1.2.3.4@4
#
#   net     = group { "_" group }
#   group   = site [ shape ] [ "~" target { "." target } ]      at least one of shape and "~"
#   shape   = uppercase letters, then the bond count            e.g. OC6, TPRS9, I10
#   target  = site [ "@" operation [ "@" klm ] ]
#
#   "~"  a covalent bond of unspecified order (the only bond symbol so far)
#   "."  separates the destinations of the bonds the group's site lists
#   "@"  operation number (1-based); a second "@" adds the lattice translation as three digits,
#        each digit = translation + 5.  Operation 1 with zero translation is not written at all,
#        and a zero translation ("555") is never written.
#   A shape is written only where it differs from the default of its bond count.  Groups follow
#   the listing order; a site opens a group if it lists a bond or has a written shape.
# ---------------------------------------------------------------------------------------------
DEFAULTS = {2: "A2", 3: "TP3", 4: "T4", 5: "SPY5", 6: "OC6", 7: "TPRS7", 8: "DD8", 9: "TPRS9", 12: "COC12"}


def shape_count(symbol):
    letters = symbol.rstrip("0123456789")
    return int(symbol[len(letters) :])


def render_code(e):
    if e == IDENTITY:
        return ""
    if not any(e[1:]):
        return f"@{e[0] + 1}"
    digits = [n + 5 for n in e[1:]]
    if not all(0 <= d <= 9 for d in digits):
        raise ValueError(f"lattice translation {e[1:]} is outside the range -5..+4 of the notation")
    return f"@{e[0] + 1}@{digits[0]}{digits[1]}{digits[2]}"


def render(slots, codes, shapes=None):
    """slots in label order, each (listing site, target).  Without shapes the groups are the runs of
    equal listing sites; with shapes (per site number) the listing order also places shape-only groups."""
    targets, order = {}, []
    for (i, j), e in zip(slots, codes):
        if i not in targets:
            targets[i] = []
            order.append(i)
        targets[i].append(f"{j}{render_code(e)}")
    if shapes is None:
        return "_".join(f"{i}~" + ".".join(targets[i]) for i in order)
    in_row = {x for slot in slots for x in slot}
    counts = [shape_count(sh) if sh else int(n in in_row) for n, sh in enumerate(shapes, 1)]
    pos = listing_positions(counts, [(i - 1, j - 1) for i, j in slots], list(range(1, len(shapes) + 1)))
    out = []
    for s in sorted(pos, key=pos.get):
        n, sh = s + 1, shapes[s]
        written = sh if sh and DEFAULTS.get(shape_count(sh)) != sh else ""
        if n in targets or written:
            out.append(f"{n}{written}" + (("~" + ".".join(targets[n])) if n in targets else ""))
    return "_".join(out)


def parse(text):
    """inverse of render: ([(i, j, element)] with 1-based site numbers, {site: written shape});
    rejects non-canonical target spellings"""
    rows, written = [], {}
    if not text:
        return rows, written
    for group in text.split("_"):
        head, sep, tail = group.partition("~")
        number = head[: len(head) - len(head.lstrip("0123456789"))]
        shape = head[len(number) :]
        if not number or number[0] == "0" or (not sep and not shape) or (sep and not tail):
            raise ValueError(f"malformed group {group!r}")
        if shape and not (shape.rstrip("0123456789").isalpha() and shape.rstrip("0123456789").isupper()):
            raise ValueError(f"malformed shape {shape!r}")
        if int(number) in written or any(r[0] == int(number) for r in rows):
            raise ValueError(f"site {number} opens a second group")
        if shape:
            written[int(number)] = shape
        for target in tail.split(".") if sep else ():
            j, *code = target.split("@")
            if not j.isdigit() or len(code) > 2 or not all(part.isdigit() for part in code):
                raise ValueError(f"malformed target {target!r}")
            k = int(code[0]) - 1 if code else 0
            n = (0, 0, 0)
            if len(code) == 2:
                if len(code[1]) != 3:
                    raise ValueError(f"translation {code[1]!r} must be three digits")
                n = tuple(int(x) - 5 for x in code[1])
            element = (k, n[0], n[1], n[2])
            if target != f"{j}{render_code(element)}" or k < 0:
                raise ValueError(f"non-canonical spelling {target!r}")
            rows.append((int(number), int(j), element))
    return rows, written
