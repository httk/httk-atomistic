"""The P-1 refined cell: exact class minima of ``L/2L`` and the least admissible cell (FU4 §5).

The oracle enumerates every lattice vector up to a norm bound exactly, with the Cauchy bound
``x_k² ≤ N (G⁻¹)_kk``, and takes the lexicographically least ``(a·a, b·b, c·c)`` over the bases
whose rows keep the least discrete key.
"""

import random
from fractions import Fraction as F
from itertools import product
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import Cell, Species, WyckoffSite
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.reduction import niggli_reduce
from httk.atomistic.symmetry import canonical_classification as classification
from httk.atomistic.symmetry import canonical_protostructure as cp
from httk.atomistic.symmetry.affine_operation import AffineOperation

TRICLINIC = ((5, 0, 0), (1, 6, 0), (-1, 2, 7))
RECTANGULAR = ((4, 0, 0), (0, 5, 0), (0, 0, 6))
SQUARE = ((5, 0, 0), (0, 5, 0), (0, 0, 7))
HEXLIKE = ((6, 0, 0), (-3, 5, 0), (0, 0, 7))
SURD = SurdVector.from_radicand_map({1: [[5, 0, 0], [0, 6, 0], [-1, 0, 7]], 2: [[0, 0, 0], [3, 0, 0], [0, 1, 0]]})
SPECIES = ("Na", "Cl", "O", "K", "Mg")
GENERATORS = [
    AffineOperation(tuple(tuple(int(r == c) + s * int((r, c) == (i, j)) for c in range(3)) for r in range(3)))
    for i in range(3)
    for j in range(3)
    if i != j
    for s in (1, -1)
]


def _structure(basis: Any, letters: str) -> ASUStructure:
    sites = [
        WyckoffSite(letter, (F(1, 7 + k), F(2, 9), F(3, 11)) if letter == "i" else (), SPECIES[k])
        for k, letter in enumerate(letters)
    ]
    return ASUStructure(Cell(basis), 2, sites, [Species(name, (name,), (1,)) for name in SPECIES[: len(letters)]])


def _gram(cell: Cell) -> list[list[Any]]:
    metric = cell.metric()
    return [[metric._element((row, column)) for column in range(3)] for row in range(3)]


def _dot(gram: Any, u: tuple[int, ...], v: tuple[int, ...]) -> Any:
    return sum((gram[i][j] * (u[i] * v[j]) for i in range(3) for j in range(3) if u[i] and v[j]), start=gram[0][0] * 0)


def _det(rows: Any) -> int:
    (a, b, c), (d, e, f), (g, h, i) = rows
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _vectors_up_to(gram: Any, bound: Any) -> list[tuple[Any, tuple[int, ...]]]:
    """Return every nonzero lattice vector of norm at most ``bound`` with its norm, exactly."""
    # (G⁻¹)_kk is a principal 2x2 minor over the determinant, both exact (rational or surd)
    determinant = _det(gram)
    limits = []
    for k in range(3):
        i, j = (index for index in range(3) if index != k)
        minor = gram[i][i] * gram[j][j] - gram[i][j] * gram[j][i]
        cauchy = bound * minor / determinant
        limit = 0
        while (limit + 1) ** 2 <= cauchy:
            limit += 1
        limits.append(limit)
    found = []
    for x in product(*(range(-limit, limit + 1) for limit in limits)):
        if any(x):
            norm = _dot(gram, x, x)
            if norm <= bound:
                found.append((norm, x))
    return found


def _admissible_row_classes(structure: ASUStructure) -> set[tuple[tuple[int, ...], ...]]:
    """Return ``P mod 2`` of the cells ``P`` that keep the least discrete key (exact inverse, not F₂)."""
    letters = tuple(site.wyckoff for site in structure.wyckoff_sites)
    keyed: dict[Any, set[FracVector]] = {}
    for operation, letter_map in classification._letter_maps(structure.spacegroup).items():
        key = cp._letters_key(structure, tuple(letter_map[letter] for letter in letters))
        keyed.setdefault(key, set()).add(operation.matrix)
    return {
        tuple(tuple(int(value) % 2 for value in row) for row in matrix.T().inv().to_fractions())
        for matrix in keyed[min(keyed)]
    }


def _oracle_lengths(structure: ASUStructure) -> tuple[Any, ...]:
    """Return the least ``(a·a, b·b, c·c)`` over the proper admissible bases, by exact enumeration."""
    gram = _gram(structure.cell)
    # every class minimum is at most the norm of the class's 0/1 representative in the input basis
    bound = max(_dot(gram, x, x) for x in product((0, 1), repeat=3) if any(x))
    vectors = sorted(_vectors_up_to(gram, bound))
    admissible = _admissible_row_classes(structure)
    best: tuple[Any, ...] | None = None
    for n1, r1 in vectors:
        if best is not None and n1 > best[0]:
            break
        for n2, r2 in vectors:
            if best is not None and (n1, n2) > best[:2]:
                break
            for n3, r3 in vectors:
                if best is not None and (n1, n2, n3) >= best:
                    break
                rows = (r1, r2, r3)
                if _det(rows) == 1 and tuple(tuple(v % 2 for v in row) for row in rows) in admissible:
                    best = (n1, n2, n3)
                    break
    assert best is not None
    return best


def _stored(structure: ASUStructure) -> tuple[Any, ...]:
    representative = classification.canonical_protostructure(Protostructure(representative=structure)).representative
    assert representative is not None
    metric = representative.cell.metric()
    return (
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        tuple((s.species, s.wyckoff, tuple(s.free_params.to_fractions())) for s in representative.wyckoff_sites),
    )


def _assert_oracle_and_cell_independent(structure: ASUStructure, cells: int, seed: str) -> tuple[Any, ...]:
    stored = _stored(structure)
    assert stored[0][0::4] == _oracle_lengths(structure)
    rng = random.Random(seed)
    for _ in range(cells):
        action = AffineOperation.identity()
        for _ in range(rng.randint(1, 4)):
            action = rng.choice(GENERATORS) * action
        assert _stored(cp._apply_action(structure, action)) == stored
    return stored


def _random_unimodular(rng: random.Random, steps: int) -> list[list[int]]:
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for _ in range(steps):
        i, j = rng.sample(range(3), 2)
        sign = rng.choice((1, -1))
        matrix[i] = [a + sign * b for a, b in zip(matrix[i], matrix[j], strict=True)]
    return matrix


def _random_gram(rng: random.Random, family: str) -> list[list[F]]:
    """Return an exact Gram of a lattice family (degenerate ones included) in a random basis."""
    if family == "generic":
        while True:
            basis = [[rng.randint(-6, 6) for _ in range(3)] for _ in range(3)]
            if _det(basis):
                break
    elif family == "hexagonal":
        gram = [[F(4), F(-2), F(0)], [F(-2), F(4), F(0)], [F(0), F(0), F(rng.randint(4, 36))]]
        change = _random_unimodular(rng, rng.randint(0, 6))
        return [[_dot(gram, tuple(change[r]), tuple(change[c])) for c in range(3)] for r in range(3)]
    elif family == "boundary":
        basis = rng.choice(([[4, 0, 0], [-2, 5, 0], [1, 1, 6]], [[4, 0, 0], [2, 5, 0], [-2, 0, 7]]))
    else:
        basis = {
            "rectangular": [[rng.randint(2, 6), 0, 0], [0, rng.randint(2, 6), 0], [0, 0, rng.randint(2, 6)]],
            "cubic": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            "fcc": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "bcc": [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
            "monoclinic": [
                [rng.randint(2, 5), 0, 0],
                [0, rng.randint(2, 5), 0],
                [rng.randint(-2, 2), 0, rng.randint(2, 5)],
            ],
        }[family]
    change = _random_unimodular(rng, rng.randint(0, 6))
    rows = [[sum(change[r][k] * basis[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    return [[F(sum(rows[r][k] * rows[c][k] for k in range(3))) for c in range(3)] for r in range(3)]


FAMILIES = ("generic", "rectangular", "cubic", "hexagonal", "fcc", "bcc", "monoclinic", "boundary")


def _check_class_minima(gram: Any) -> None:
    minima = cp._p1_class_minima(tuple(tuple(row) for row in gram))
    assert len(minima) == 7
    vectors = _vectors_up_to(gram, max(norm for norm, _found in minima.values()))
    for residue, (norm, found) in minima.items():
        in_class = [(n, x) for n, x in vectors if tuple(v % 2 for v in x) == residue]
        least = min(n for n, _x in in_class)
        assert norm == least
        assert set(found) == {x for n, x in in_class if n == least}
        assert len(found) == len(set(found))


def test_class_minima_match_exact_enumeration() -> None:
    # Lemma A on about twenty fixed-seed lattices: the minima and every minimizer of each class
    rng = random.Random("p1-class-minima")
    for family in FAMILIES:
        for _ in range(2 if family in ("cubic", "fcc", "bcc") else 3):
            _check_class_minima(_random_gram(rng, family))


@pytest.mark.extended
def test_class_minima_match_exact_enumeration_extended() -> None:
    rng = random.Random("p1-class-minima-extended")
    for _ in range(400):
        _check_class_minima(_random_gram(rng, rng.choice(FAMILIES)))


def test_class_minima_of_a_surd_metric() -> None:
    gram = _gram(Cell(SURD))
    minima = cp._p1_class_minima(tuple(tuple(row) for row in gram))
    for residue, (norm, found) in minima.items():
        assert all(_dot(gram, x, x) == norm and tuple(v % 2 for v in x) == residue for x in found)
    _check_class_minima(gram)


def test_independent_class_minimizers_need_not_form_a_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    # diag(16, 25, 36): e1+e2, e1+e3 and e1-e2-e3 minimize three independent classes but have
    # determinant 3. Restricted to that row class triple, only determinant +1 triples are emitted.
    structure = _structure(RECTANGULAR, "i")
    minima = cp._p1_class_minima(tuple(tuple(row) for row in _gram(structure.cell)))
    triple = ((1, 1, 0), (1, 0, 1), (1, -1, -1))
    assert all(row in minima[tuple(v % 2 for v in row)][1] for row in triple)
    assert _det(triple) == 3
    cell = FracVector(((1, 1, 0), (1, 0, 1), (-1, -1, -1)))
    assert cell.det() == 1
    point = cell.T().inv()
    actions = {AffineOperation(point, tuple(F(v, 2) for v in shift)): {"i": "i"} for shift in product((0, 1), repeat=3)}
    monkeypatch.setattr(classification, "_letter_maps", lambda _group: actions)
    candidates = cp._p1_candidates(structure)
    assert candidates
    emitted = {candidate.basis_change for candidate in candidates}
    assert FracVector(triple) not in emitted
    for basis_change in emitted:
        assert basis_change.det() == 1
        assert tuple(tuple(int(v) % 2 for v in row) for row in basis_change.to_fractions()) == (
            (1, 1, 0),
            (1, 0, 1),
            (1, 1, 1),
        )
    assert all(candidate.reduction_key[0] == (41, 52, 77) for candidate in candidates)


def test_inversion_centres_store_the_least_admissible_cell() -> None:
    # Na on a, Cl on h of the skew cell: the stored cell is not the Niggli one (25, 37, 54)
    stored = _assert_oracle_and_cell_independent(_structure(TRICLINIC, "ah"), 0, "ah")
    assert stored[0][0::4] == (25, 37, 74)


def test_type_one_lattice_stores_its_niggli_gram() -> None:
    # all three Niggli products positive: the least-Gram rule alone would store (-, -, +)
    stored = _stored(_structure(((5, 0, 0), (1, 6, 0), (1, 1, 7)), "i"))
    assert stored[0] == (25, 5, 5, 5, 37, 7, 5, 7, 51)
    assert cp._is_niggli_gram(stored[0])


# the fourteen centre cases of the design review's brute force (four named, ten drawn at random)
CENTRE_CASES = (
    (TRICLINIC, "ah"),
    (TRICLINIC, "ad"),
    (TRICLINIC, "abcd"),
    (TRICLINIC, "bgi"),
    (SQUARE, "fgh"),
    (TRICLINIC, "dfi"),
    (RECTANGULAR, "fi"),
    (RECTANGULAR, "bei"),
    (RECTANGULAR, "eb"),
    (HEXLIKE, "bi"),
    (HEXLIKE, "ci"),
    (TRICLINIC, "d"),
    (RECTANGULAR, "ei"),
    (RECTANGULAR, "dbfi"),
)


@pytest.mark.extended
@pytest.mark.parametrize("basis,letters", CENTRE_CASES)
def test_inversion_centre_cases_match_the_exact_oracle(basis: Any, letters: str) -> None:
    stored = _assert_oracle_and_cell_independent(_structure(basis, letters), 2, letters)
    if (basis, letters) == (RECTANGULAR, "bei"):
        # the degenerate class (111) has four pairs of minimizers
        assert stored[0][0::4] == (16, 25, 77)


@pytest.mark.extended
@pytest.mark.parametrize(
    "basis",
    (((4, 0, 0), (-2, 5, 0), (1, 1, 6)), ((4, 0, 0), (2, 5, 0), (-2, 0, 7)), SURD),
    ids=("boundary-a", "boundary-b", "surd"),
)
def test_niggli_boundary_and_surd_lattices_store_a_niggli_gram(basis: Any) -> None:
    structure = _structure(basis, "i")
    stored = _stored(structure)
    assert cp._is_niggli_gram(stored[0])
    if basis is not SURD:
        niggli = niggli_reduce(structure.cell).cell.metric()
        assert stored[0] == tuple(niggli._element((row, column)) for row in range(3) for column in range(3))


@pytest.mark.extended
@pytest.mark.parametrize("letters", ("ii", "ah", "adi"))
def test_surd_metric_matches_the_exact_oracle(letters: str) -> None:
    _assert_oracle_and_cell_independent(_structure(SURD, letters), 4, letters)


def test_selling_superbase_is_obtuse() -> None:
    gram = _random_gram(random.Random("selling"), "generic")
    vectors = cp._selling_superbase(tuple(tuple(row) for row in gram))
    assert [sum(column) for column in zip(*vectors, strict=True)] == [0, 0, 0]
    assert abs(_det(vectors[1:])) == 1
    assert all(_dot(gram, vectors[i], vectors[j]) <= 0 for i in range(4) for j in range(i + 1, 4))
