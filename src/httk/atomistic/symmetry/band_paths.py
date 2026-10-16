"""Exact band-structure k-paths in the HPKOT convention.

The convention is that of Hinuma, Pizzi, Kumagai, Oba and Tanaka, "Band structure diagram
paths based on crystallography", Comput. Mater. Sci. 128, 140 (2017), as implemented by
SeeK-path 2.2.2, whose tables are vendored in ``_band_path_data``. The path is chosen
from the IT standard conventional cell after the lattice choices that the ITA symbol leaves
free (orthorhombic axis order, the monoclinic reduced cell, the triclinic reduced reciprocal
cell), and every selection rule and k-point coordinate is evaluated exactly from the rational
metric tensor.
"""

import fractions
import functools
import itertools
import re
import types
from collections.abc import Mapping
from dataclasses import dataclass

from httk.core import FracVector, SurdVector, register_citation

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import initialize_semantics
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.reduction import _niggli_reduce_metric
from httk.atomistic.symmetry.primitive import _COLUMN_CENTRING_MATRICES, primitive_cell
from httk.atomistic.symmetry.spacegroup import Spacegroup

from ._band_path_data import BAND_PATH_DATA, HpkotMetric
from ._standardization_common import _matrix_column_sum_factor, _matrix_row_sum_factor, _scaled_precision

__all__ = ["BandPathResult", "band_path"]

Fraction = fractions.Fraction
type _IntMatrix = tuple[tuple[int, ...], ...]

_IDENTITY: _IntMatrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

# SeeK-path's conventional-to-primitive matrices in column convention, (a_P, b_P, c_P) =
# (a, b, c) P, where they differ from the spglib centring matrices used by primitive_cell.
_HPKOT_COLUMN_PRIMITIVE = {
    "oA": FracVector([["0", "0", "1"], ["1/2", "1/2", "0"], ["-1/2", "1/2", "0"]]),
    "mC": FracVector([["1/2", "-1/2", "0"], ["1/2", "1/2", "0"], ["0", "0", "1"]]),
}

_FAMILIES = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}

# Hexagonal-family space groups whose HPKOT path is hP1 (SeeK-path's list; the R groups in it
# never reach the hP branch).
_HP1_GROUPS = frozenset({143, 144, 145, 146, 147, 148, 149, 151, 153, 157, 159, 160, 161, 162, 163})

# SeeK-path's aP "M2" column matrices, indexed by the position of the smallest of
# (|G*_23|, |G*_13|, |G*_12|) in the reduced reciprocal metric.
_TRICLINIC_M2 = (
    FracVector([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    FracVector([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    FracVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

_EIGHTHS = tuple(Fraction(n, 8) for n in range(8))

_POINT_TERM = re.compile(r"(|-|1-|-1\+|1/2-|1/2\+)([A-Za-z]\w*)")
_POINT_OFFSETS = {
    "": (Fraction(0), 1),
    "-": (Fraction(0), -1),
    "1-": (Fraction(1), -1),
    "-1+": (Fraction(-1), 1),
    "1/2-": (Fraction(1, 2), -1),
    "1/2+": (Fraction(1, 2), 1),
}
_POINT_CONSTANT = re.compile(r"-?\d+(/\d+)?")


def _descending(matrix: _IntMatrix) -> tuple[int, ...]:
    """Return the deterministic tie-break key: entries in descending order, row by row.

    It prefers +1 over -1 entries, so the identity comes first among equally good re-choices.
    """
    return tuple(-value for row in matrix for value in row)


def _signed_permutations() -> tuple[_IntMatrix, ...]:
    """Return the 24 det +1 signed permutation matrices in :func:`_descending` order."""
    matrices: list[_IntMatrix] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            rows = tuple(tuple(signs[i] if j == permutation[i] else 0 for j in range(3)) for i in range(3))
            if FracVector(rows).det().to_fraction() == 1:
                matrices.append(rows)
    return tuple(sorted(matrices, key=_descending))


_SIGNED_PERMUTATIONS = _signed_permutations()


@dataclass(frozen=True, slots=True)
class BandPathResult:
    """Store a recommended band-structure path in the HPKOT convention.

    ``points`` are fractional coordinates in the reciprocal basis of ``structure`` (the
    HPKOT primitive cell), keyed by SeeK-path's ASCII labels (``GAMMA``, ``X_1``, primes as
    ``X'``) in table order. ``path`` lists the segments in order. ``transform`` is the integer
    row-convention matrix with ``basis(structure) = transform * basis(primitive)``, where
    ``primitive`` is :func:`~httk.atomistic.primitive_cell` of the input; its determinant is +1,
    or -1 when the conventional basis was left-handed (the result basis is always right-handed).

    :param structure: The HPKOT primitive cell the coordinates refer to.
    :param spacegroup: The IT standard setting of the crystal's space group.
    :param bravais_lattice: The Bravais lattice symbol, such as ``"cF"``.
    :param bravais_lattice_extended: The extended HPKOT symbol, such as ``"cF2"``.
    :param points: A read-only mapping from label to exact fractional reciprocal coordinates.
    :param path: The ordered ``(start, end)`` label segments.
    :param transform: The integer row-convention basis change from the primitive cell.
    """

    structure: UnitcellStructure
    spacegroup: Spacegroup
    bravais_lattice: str
    bravais_lattice_extended: str
    points: Mapping[str, FracVector]
    path: tuple[tuple[str, str], ...]
    transform: FracVector


def _matrix_key(matrix: FracVector) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(row) for row in matrix.to_fractions())


def _preserves_setting(spacegroup: Spacegroup, rechoice: FracVector) -> bool:
    """Whether the basis re-choice keeps the standard-setting op set, up to an origin shift.

    ``rechoice`` is the row-convention ``Q`` with ``B_new = Q B``; fractional rows map as
    ``f_new = f Q^-1``, so column coordinates map by the point matrix ``A = Q^-T`` and an
    operation ``(W, w)`` becomes ``(A W A^-1, A w + (I - A W A^-1) v)`` for an origin shift
    ``v``. As in :func:`~httk.atomistic.symmetry.lift._resetting_preserves_group`, the image set
    must EQUAL the standard op set (translations wrapped into ``[0, 1)``). Origin shifts are
    searched on the 1/8 grid only, which suffices for the orthorhombic and monoclinic re-choices
    this module makes (diagonal or ac-plane linear parts, translations of at worst 1/4); it is
    not a general normalizer solver.
    """
    point = rechoice.T().inv()
    inverse = point.inv()
    operations = {
        (_matrix_key(operation.matrix), tuple(FracVector(operation.vector).normalize().to_fractions()))
        for operation in spacegroup.symmetry_operations
    }
    pieces = []
    for operation in spacegroup.symmetry_operations:
        conjugated = _matrix_key(point * operation.matrix * inverse)
        complement = tuple(tuple((1 if i == j else 0) - conjugated[i][j] for j in range(3)) for i in range(3))
        pieces.append((conjugated, complement, tuple((point * FracVector(operation.vector)).to_fractions())))
    if {conjugated for conjugated, _, _ in pieces} != {matrix for matrix, _ in operations}:
        return False
    for shift in itertools.product(_EIGHTHS, repeat=3):
        images = set()
        for conjugated, complement, translation in pieces:
            image = (
                conjugated,
                tuple(
                    (translation[i] + sum((complement[i][j] * shift[j] for j in range(3)), start=Fraction(0))) % 1
                    for i in range(3)
                ),
            )
            if image not in operations:
                break
            images.add(image)
        else:
            if images == operations:
                return True
    return False


def _transformed_metric(rechoice: FracVector, metric: FracVector) -> list[list[Fraction]]:
    return (rechoice * metric * rechoice.T()).to_fractions()


def _orthorhombic_rechoice(spacegroup: Spacegroup, metric: FracVector) -> FracVector:
    """Return the setting-preserving axis order minimizing ``(a^2, b^2, c^2)`` lexicographically."""

    def lengths(matrix: _IntMatrix) -> tuple[Fraction, ...]:
        gram = _transformed_metric(FracVector(matrix), metric)
        return (gram[0][0], gram[1][1], gram[2][2])

    # sorted() is stable, so equal lengths keep the identity-first deterministic order.
    for matrix in sorted(_SIGNED_PERMUTATIONS, key=lengths):
        if _preserves_setting(spacegroup, FracVector(matrix)):
            return FracVector(matrix)
    raise AssertionError("the identity always preserves the setting")


def _sqrt_sum_sign(p1: Fraction, q1: Fraction, p2: Fraction, q2: Fraction) -> int:
    """Return the exact sign of ``(sqrt(p1) + sqrt(q1)) - (sqrt(p2) + sqrt(q2))`` for ``p, q >= 0``."""

    def sign(value: Fraction) -> int:
        return (value > 0) - (value < 0)

    # Both sums are nonnegative, so compare their squares: r + sqrt(x) - sqrt(y).
    r = p1 + q1 - p2 - q2
    x, y = 4 * p1 * q1, 4 * p2 * q2
    radical = sign(x - y)  # sign(sqrt(x) - sqrt(y))
    if r == 0 or radical in (0, sign(r)):
        return radical or sign(r)
    # Opposite signs: compare r^2 with (sqrt(x) - sqrt(y))^2 = x + y - sqrt(4xy).
    z = r * r - x - y
    if z >= 0:
        rational_wins = 1 if (z > 0 or x * y > 0) else 0
    else:
        rational_wins = sign(4 * x * y - z * z)
    return sign(r) if rational_wins > 0 else (0 if rational_wins == 0 else radical)


def _monoclinic_rechoice(spacegroup: Spacegroup, metric: FracVector) -> FracVector:
    """Return the setting-preserving Parthé–Gelato cell (unique axis b, beta >= 90 degrees)."""
    gram = metric.to_fractions()

    def dot(x: tuple[int, int], y: tuple[int, int]) -> Fraction:
        return x[0] * y[0] * gram[0][0] + (x[0] * y[1] + x[1] * y[0]) * gram[0][2] + x[1] * y[1] * gram[2][2]

    # Exact Lagrange reduction of the (a, c) plane lattice; vectors are integer (a, c) pairs.
    u, w = (1, 0), (0, 1)
    while True:
        if dot(u, u) > dot(w, w):
            u, w = w, u
        step = round(dot(u, w) / dot(u, u))
        if not step:
            break
        w = (w[0] - step * u[0], w[1] - step * u[1])

    candidates: list[tuple[Fraction, Fraction, _IntMatrix]] = []
    for i, j, k, m in itertools.product(range(-2, 3), repeat=4):
        if abs(i * m - j * k) != 1:
            continue
        a = (i * u[0] + j * w[0], i * u[1] + j * w[1])
        c = (k * u[0] + m * w[0], k * u[1] + m * w[1])
        a2, c2 = dot(a, a), dot(c, c)
        if dot(a, c) > 0:
            continue
        # b' = (a0 c1 - a1 c0) b makes the determinant +1.
        candidates.append((a2, c2, ((a[0], 0, a[1]), (0, a[0] * c[1] - a[1] * c[0], 0), (c[0], 0, c[1]))))

    def compare(first: tuple[Fraction, Fraction, _IntMatrix], second: tuple[Fraction, Fraction, _IntMatrix]) -> int:
        tie = (first[0], first[1], _descending(first[2])), (second[0], second[1], _descending(second[2]))
        return _sqrt_sum_sign(first[0], first[1], second[0], second[1]) or (tie[0] > tie[1]) - (tie[0] < tie[1])

    # The a^2-first tie-break yields spglib's |a| <= |c| rule for SG 3, 4, 6, 10 and 11, where the
    # setting allows swapping a and c.
    for _, _, matrix in sorted(candidates, key=functools.cmp_to_key(compare)):
        if _preserves_setting(spacegroup, FracVector(matrix)):
            return FracVector(matrix)
    raise AssertionError("no setting-preserving monoclinic cell among the enumerated candidates")


def _triclinic_rechoice(metric: FracVector) -> tuple[FracVector, str]:
    """Return SeeK-path's aP re-choice (Niggli-reduced reciprocal cell, then M2) and the symbol."""
    reciprocal_transform, reduced = _niggli_reduce_metric(metric.inv())
    gram = reduced.to_fractions()
    conditions = (abs(gram[1][2]), abs(gram[0][2]), abs(gram[0][1]))
    # Reciprocal rows transform as B*' = U B*, so real rows as B' = U^-T B; M2 acts on columns.
    rechoice = _TRICLINIC_M2[conditions.index(min(conditions))].T() * reciprocal_transform.inv().T()
    # ponytail: SeeK-path's M3 sign step is the identity for a Niggli form (all-acute or all-obtuse
    # reciprocal angles) and is omitted; a mixed pattern below would mean that assumption broke.
    final = _transformed_metric(rechoice, metric)
    final_reciprocal = FracVector(final).inv().to_fractions()
    off_diagonal = (final_reciprocal[1][2], final_reciprocal[0][2], final_reciprocal[0][1])
    if all(value <= 0 for value in off_diagonal):
        return rechoice, "aP2"
    if all(value > 0 for value in off_diagonal):
        return rechoice, "aP3"
    raise AssertionError("a Niggli-reduced reciprocal metric has same-sign off-diagonal terms")


def _extended_symbol(bravais: str, it_number: int, gram: list[list[Fraction]]) -> str:
    a2, b2, c2, ac = gram[0][0], gram[1][1], gram[2][2], gram[0][2]
    match bravais:
        case "cP" | "cF":
            return bravais + ("1" if it_number <= 206 else "2")
        case "hP":
            return "hP1" if it_number in _HP1_GROUPS else "hP2"
        case "tI":
            return "tI1" if c2 <= a2 else "tI2"
        case "hR":
            return "hR1" if 3 * a2 <= 2 * c2 else "hR2"
        case "oF":
            if 1 / a2 > 1 / b2 + 1 / c2:
                return "oF1"
            return "oF2" if 1 / c2 > 1 / a2 + 1 / b2 else "oF3"
        case "oI":
            if b2 >= a2 and b2 >= c2:
                return "oI3"
            return "oI2" if a2 >= c2 else "oI1"
        case "oC":
            return "oC1" if a2 <= b2 else "oC2"
        case "oA":
            return "oA1" if b2 <= c2 else "oA2"
        case "mC":
            if b2 < a2 - ac * ac / c2:
                return "mC1"
            return "mC2" if -ac / c2 + (a2 - ac * ac / c2) / b2 <= 1 else "mC3"
        case _:
            return bravais + "1"


def _coordinate(text: str, parameters: Mapping[str, Fraction]) -> Fraction:
    """Evaluate one SeeK-path point coordinate exactly (a constant or an affine parameter term)."""
    if _POINT_CONSTANT.fullmatch(text):
        return Fraction(text)
    match = _POINT_TERM.fullmatch(text)
    if match is None or match[2] not in parameters:
        raise ValueError(f"unsupported band-path coordinate expression {text!r}")
    offset, sign = _POINT_OFFSETS[match[1]]
    return offset + sign * parameters[match[2]]


def _has_inversion(spacegroup: Spacegroup) -> bool:
    inversion = FracVector([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return any(operation.matrix == inversion for operation in spacegroup.symmetry_operations)


def band_path(
    structure: StructureLike,
    *,
    time_reversal: bool = True,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
) -> BandPathResult:
    """Return the HPKOT band-structure path of a crystal, with exact k-point coordinates.

    The space group is the one the structure holds: the declared group of an existing ASU
    (used as stored, without spglib), or the recognized group of a raw structure. The input is
    brought to its IT standard conventional cell by :func:`~httk.atomistic.primitive_cell`; the
    lattice choices the setting leaves free are then made as SeeK-path does (orthorhombic axes
    ordered by length among symmetry-allowed permutations, the monoclinic Parthé–Gelato reduced
    cell, the triclinic Niggli-reduced reciprocal cell), and the extended Bravais symbol,
    parameters and points are evaluated exactly from the rational metric. Exact extended-symbol
    ties are resolved as SeeK-path resolves them; equal-length axis-order ties and near ties within
    spglib's tolerance may give an equivalent, differently oriented cell.

    With ``time_reversal=False`` and a group without inversion, every non-``GAMMA`` point
    ``X`` gains a partner ``X'`` at ``-X`` and the path is repeated with primed endpoints.

    :param structure: The fully periodic crystal.
    :param time_reversal: Whether time-reversal symmetry relates ``k`` and ``-k``.
    :param tolerance: The Cartesian recognition tolerance, or ``None`` to derive it.
    :param limit_denominator: The maximum denominator for idealised free parameters, or
        ``None`` to retain their exact stated values.
    :return: The HPKOT primitive cell, symbols, points and path.
    :raises ImportError: If recognition is needed and the optional spglib dependency is
        unavailable.
    :raises ValueError: If the structure has site moments, its conventional metric is not
        rational, or :func:`~httk.atomistic.primitive_cell` refuses it.
    """
    if UnitcellStructureView(structure).site_moments is not None:
        raise ValueError(
            "band_path does not support magnetic structures: magnetic-group band paths are "
            "unsupported, and nuclear-symmetry paths are not valid for magnetic structures"
        )
    primitive = primitive_cell(structure, tolerance=tolerance, limit_denominator=limit_denominator)
    spacegroup = primitive.spacegroup
    conventional = primitive.conventional.structure
    surd_metric = conventional.cell.metric()
    if not surd_metric.is_rational:
        # ponytail: special-angle surd metrics (e.g. beta = 135 degrees) are refused; supporting
        # them needs exact surd comparisons throughout the selection rules and parameters.
        raise ValueError("band_path requires a rational-valued exact Gram matrix of the conventional cell")
    metric = surd_metric.coefficient(1)

    bravais = _FAMILIES[spacegroup.crystal_system] + spacegroup.centring_type
    extended: str | None = None
    if bravais[0] == "o":
        rechoice = _orthorhombic_rechoice(spacegroup, metric)
    elif bravais[0] == "m":
        rechoice = _monoclinic_rechoice(spacegroup, metric)
    elif bravais == "aP":
        rechoice, extended = _triclinic_rechoice(metric)
    else:
        rechoice = FracVector(_IDENTITY)
    if conventional.cell.basis.det().sign() < 0:
        rechoice = -rechoice
    gram = _transformed_metric(rechoice, metric)
    if extended is None:
        extended = _extended_symbol(bravais, spacegroup.it_number, gram)

    column_primitive = _HPKOT_COLUMN_PRIMITIVE.get(bravais, _COLUMN_CENTRING_MATRICES[spacegroup.centring_type])
    transform = (column_primitive.T() * rechoice * primitive.transform.inv()).simplify()
    assert transform.denom == 1 and abs(transform.det().to_fraction()) == 1

    source = primitive.structure
    inverse = transform.inv().simplify()
    cell = Cell(
        SurdVector(transform) * source.cell.unscaled_basis,
        scale=source.cell.scale,
        precision=_scaled_precision(source.cell.precision, _matrix_row_sum_factor(transform)),
        periodicity=source.cell.periodicity,
    )
    sites = Sites(
        (source.sites.reduced_coords * inverse).normalize(),
        precision=_scaled_precision(source.coordinate_precision, _matrix_column_sum_factor(inverse)),
    )
    result = UnitcellStructure(
        cell,
        sites,
        source.species,
        source.species_at_sites,
        molecular=source.molecular,
        chemical_composition=source.chemical_composition,
        chemical_formula_descriptive=source.chemical_formula_descriptive,
        chemical_formula_hill=source.chemical_formula_hill,
        optimization_type=source.optimization_type,
        charge=source.charge,
    )
    initialize_semantics(
        result,
        nsites=len(sites),
        molecular=source.molecular,
        assemblies=None,
        symmetry=None,
        chemical_composition=source.chemical_composition,
        chemical_formula_descriptive=source.chemical_formula_descriptive,
        chemical_formula_hill=source.chemical_formula_hill,
        optimization_type=source.optimization_type,
    )

    entry = BAND_PATH_DATA[extended]
    hpkot_metric = HpkotMetric(gram[0][0], gram[1][1], gram[2][2], gram[0][2])
    parameters: dict[str, Fraction] = {}
    for name, function in entry.parameters:
        parameters[name] = function(hpkot_metric, parameters)
    points = {label: FracVector([_coordinate(value, parameters) for value in values]) for label, values in entry.points}
    path = entry.path
    if not time_reversal and not _has_inversion(spacegroup):

        def primed(label: str) -> str:
            return label if label == "GAMMA" else f"{label}'"

        points |= {primed(label): -point for label, point in points.items() if label != "GAMMA"}
        path += tuple((primed(start), primed(end)) for start, end in path)

    register_citation(
        applies_to="Band-structure paths follow the HPKOT convention (tables from SeeK-path)",
        references={
            "authors": (
                {"name": "Yoyo Hinuma"},
                {"name": "Giovanni Pizzi"},
                {"name": "Yu Kumagai"},
                {"name": "Fumiyasu Oba"},
                {"name": "Isao Tanaka"},
            ),
            "title": "Band structure diagram paths based on crystallography",
            "journal": "Computational Materials Science",
            "volume": "128",
            "pages": "140-184",
            "year": "2017",
            "doi": "10.1016/j.commatsci.2016.10.015",
            "bib_type": "article",
        },
    )
    return BandPathResult(
        result,
        spacegroup,
        bravais,
        extended,
        types.MappingProxyType(points),
        path,
        transform,
    )
