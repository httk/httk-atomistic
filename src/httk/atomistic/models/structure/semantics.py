"""Shared OPTIMADE structure semantics for native atomistic representations."""

from __future__ import annotations

import ast
import datetime
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from fractions import Fraction
from functools import cache, cached_property, lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

from httk.core import MISSING as _METADATA_UNSET

from httk.atomistic.composition import (
    Assembly,
    ChemicalComposition,
    derive_structure_features,
    project_composition,
    validate_assemblies,
)
from httk.atomistic.elements import SYMBOLS
from httk.atomistic.models._vector_guards import matmul_floats
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.composition_view import CompositionView
from httk.atomistic.models.formula.formula_view import ChemicalFormulaView

if TYPE_CHECKING:
    from httk.atomistic.models.structure.backend import StructureBackend
    from httk.atomistic.models.structure.view import StructureView

OptimizationType = Literal["experimental", "hybrid", "global", "local", "none", "indeterminate", "other"]

_OPTIMIZATION_TYPES = frozenset({"experimental", "hybrid", "global", "local", "none", "indeterminate", "other"})
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
_SYMOP_COORDINATE = re.compile(r"[xyzXYZ0-9+\-*/(). ]+")
_ELEMENTS = frozenset(SYMBOLS)
_DESCRIPTIVE_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_DESCRIPTIVE_GROUPS = frozenset({"Me", "Et", "Bu", "Ph", "Bn"})
# Relative metric tolerance: float lattices (hexagonal, ~5-digit POSCARs) pass; wrong rotations distort it by O(1).
_METRIC_TOLERANCE = 1e-4


@lru_cache(maxsize=4096)
def _linear_xyz_expression(text: str) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Parse one exact affine ``xyz`` expression without evaluating source text."""

    def visit(node: ast.AST) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id.lower() in {"x", "y", "z"}:
            values = [Fraction(0)] * 4
            values[{"x": 0, "y": 1, "z": 2}[node.id.lower()]] = Fraction(1)
            return cast(tuple[Fraction, Fraction, Fraction, Fraction], tuple(values))
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return (Fraction(0), Fraction(0), Fraction(0), Fraction(str(node.value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = visit(node.operand)
            return (
                value
                if isinstance(node.op, ast.UAdd)
                else cast(tuple[Fraction, Fraction, Fraction, Fraction], tuple(-part for part in value))
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
            left, right = visit(node.left), visit(node.right)
            sign = 1 if isinstance(node.op, ast.Add) else -1
            return cast(
                tuple[Fraction, Fraction, Fraction, Fraction],
                tuple(a + sign * b for a, b in zip(left, right)),
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult | ast.Div):
            left, right = visit(node.left), visit(node.right)
            left_constant = not any(left[:3])
            right_constant = not any(right[:3])
            if isinstance(node.op, ast.Mult) and left_constant:
                return cast(tuple[Fraction, Fraction, Fraction, Fraction], tuple(left[3] * part for part in right))
            if isinstance(node.op, ast.Mult) and right_constant:
                return cast(tuple[Fraction, Fraction, Fraction, Fraction], tuple(right[3] * part for part in left))
            if isinstance(node.op, ast.Div) and right_constant and right[3] != 0:
                return cast(tuple[Fraction, Fraction, Fraction, Fraction], tuple(part / right[3] for part in left))
        raise ValueError(f"non-affine symmetry coordinate {text!r}")

    try:
        return visit(ast.parse(text, mode="eval"))
    except (SyntaxError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid symmetry coordinate {text!r}") from exc


type _Triple = tuple[Fraction, Fraction, Fraction]
type _Rotation = tuple[int, int, int, int, int, int, int, int, int]
# A wrapped operation: row-major integral rotation part and translation in [0, 1).
type _Operation = tuple[_Rotation, _Triple]
type _Vector = tuple[int, int, int]
type _CosetClass = tuple[int, int, int, int, int]
type _GroupInvariant = tuple[
    int,
    tuple[_CosetClass, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[_CosetClass, _CosetClass, bool], ...],
]

_IDENTITY_ROTATION: _Rotation = (1, 0, 0, 0, 1, 0, 0, 0, 1)
_ZERO: _Triple = (Fraction(0), Fraction(0), Fraction(0))


def _rotation_product(a: _Rotation, b: _Rotation) -> _Rotation:
    """Return the product ``a b`` of row-major 3x3 integer matrices."""
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = a
    b0, b1, b2, b3, b4, b5, b6, b7, b8 = b
    return (
        a0 * b0 + a1 * b3 + a2 * b6,
        a0 * b1 + a1 * b4 + a2 * b7,
        a0 * b2 + a1 * b5 + a2 * b8,
        a3 * b0 + a4 * b3 + a5 * b6,
        a3 * b1 + a4 * b4 + a5 * b7,
        a3 * b2 + a4 * b5 + a5 * b8,
        a6 * b0 + a7 * b3 + a8 * b6,
        a6 * b1 + a7 * b4 + a8 * b7,
        a6 * b2 + a7 * b5 + a8 * b8,
    )


def _apply(matrix: tuple[Any, ...], vector: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
    """Return ``matrix * vector`` for a row-major 3x3 matrix and a column vector."""
    x, y, z = vector
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z,
        matrix[3] * x + matrix[4] * y + matrix[5] * z,
        matrix[6] * x + matrix[7] * y + matrix[8] * z,
    )


def _determinant(m: _Rotation) -> int:
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])


def _wrap(vector: Iterable[Fraction]) -> _Triple:
    x, y, z = (value % 1 for value in vector)
    return (x, y, z)


def _plain_operation(operation: Any) -> _Operation:
    """Return an :class:`~httk.atomistic.symmetry.affine_operation.AffineOperation` in wrapped plain form."""
    rotation = cast(_Rotation, tuple(int(value) for row in operation.matrix.to_fractions() for value in row))
    return rotation, _wrap(operation.vector.to_fractions())


@cache
def _tabulated_group(hall_entry: str) -> frozenset[_Operation]:
    """Return the wrapped plain operations of one tabulated setting."""
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    return frozenset(_plain_operation(value) for value in Spacegroup.from_hall_entry(hall_entry).symmetry_operations)


def _symmetry_operation_group(operations: tuple[str, ...]) -> frozenset[_Operation]:
    """Return parsed wrapped operations after exact finite-group validation."""
    parsed = []
    for operation in operations:
        components = [_linear_xyz_expression(part) for part in operation.split(",")]
        entries = [value for component in components for value in component[:3]]
        if any(value.denominator != 1 for value in entries):
            raise ValueError("space-group symmetry operations must have unimodular rotation parts")
        rotation = cast(_Rotation, tuple(int(value) for value in entries))
        parsed.append((rotation, _wrap(component[3] for component in components)))
    group = frozenset(parsed)
    if len(group) != len(operations):
        raise ValueError("space-group symmetry operations must be unique modulo lattice translations")
    if (_IDENTITY_ROTATION, _ZERO) not in group:
        raise ValueError("space-group symmetry operations must contain identity")
    if any(abs(_determinant(rotation)) != 1 for rotation, _ in group):
        raise ValueError("space-group symmetry operations must have unimodular rotation parts")
    # Closure in integer arithmetic over the common translation denominator: grow the
    # subgroup generated by elements of the set and fail as soon as a product leaves it.
    # A finite set of invertible maps closed under composition is a group, so inverses
    # need no separate check.
    denominator = math.lcm(*(value.denominator for _, translation in group for value in translation))
    encoded = {
        (rotation, cast(_Vector, tuple(int(value * denominator) for value in translation)))
        for rotation, translation in group
    }
    generated = {(_IDENTITY_ROTATION, (0, 0, 0))}
    generators: list[tuple[_Rotation, _Vector]] = []
    for element in encoded:
        if element in generated:
            continue
        generators.append(element)
        frontier = list(generated)
        while frontier:
            found = []
            for rotation, translation in frontier:
                for left_rotation, left_translation in generators:
                    x, y, z = _apply(left_rotation, translation)
                    product = (
                        _rotation_product(left_rotation, rotation),
                        (
                            (x + left_translation[0]) % denominator,
                            (y + left_translation[1]) % denominator,
                            (z + left_translation[2]) % denominator,
                        ),
                    )
                    if product not in generated:
                        if product not in encoded:
                            raise ValueError("space-group symmetry operations are not closed under composition")
                        generated.add(product)
                        found.append(product)
            frontier = found
    return group


def _operation_group_invariant(group: Iterable[_Operation]) -> _GroupInvariant:
    """Return a basis-independent invariant of a space group given by wrapped operations.

    The operations may be expressed in any lattice basis and origin (a primitive cell of
    a centred group, a rotated or permuted setting), so the invariant only uses quantities
    that survive an affine change of basis.  ``T`` is the translation subgroup: ``Z^3``
    plus the translations of the operations whose rotation part is the identity.  For
    every distinct rotation part ``R`` (of order ``n``, with any translation ``t`` from
    its coset) the intrinsic translations ``w(t') = (1/n) sum_k R^k t'`` over
    ``t' in t + T`` form the coset ``w(t) + H`` modulo ``T``, where ``H`` is the image of
    ``T`` under ``(1/n) sum_k R^k``.  The invariant is the point-group order, the sorted
    ``(det R, trace R, n, m, |H|)`` over the distinct ``R``, where ``m`` is the smallest
    order modulo ``T`` of an intrinsic translation in that coset (``1`` when the coset
    holds a pure rotation or reflection, ``> 1`` for screw and glide classes), the sorted
    orbit sizes of the point group acting on ``(1/k)T / T`` for ``k = 2, 3, 4``, and, for
    every screw or glide coset paired with every coset ``b``, whether an intrinsic
    translation of the former is fixed modulo ``T`` by the projector ``(1/n_b) sum_k R_b^k``
    onto the axis or plane of ``b``.

    Every tabulated setting of a space-group type yields the same invariant, and the
    orbit sizes separate all arithmetic crystal classes.  Equal invariants are still
    necessary, not sufficient.  The 230 types give 213 invariants: the eleven enantiomorphic
    pairs (76/78, 91/95, 92/96, 144/145, 151/153, 152/154, 169/170, 171/172, 178/179,
    180/181, 212/213) are never separated, and neither are 23/24, 32/34, 117/118, 185/186,
    193/194 and 197/199.

    :param group: Operations wrapped modulo lattice translations, forming a finite group.
    :return: The invariant.
    """

    # Integer arithmetic over a common denominator: translations have denominators dividing
    # ``base``, and each division below is by a rotation order or by k (each dividing 12),
    # applied at most twice in succession, so 144 * base keeps every quotient integral.
    group = tuple(group)
    base = math.lcm(*(value.denominator for _, translation in group for value in translation))
    scale = 144 * base

    def add(left: _Vector, right: _Vector) -> _Vector:
        return ((left[0] + right[0]) % scale, (left[1] + right[1]) % scale, (left[2] + right[2]) % scale)

    cosets: dict[_Rotation, _Vector] = {}
    centring: set[_Vector] = set()
    for rotation, translation in group:
        x, y, z = (int(value * scale) for value in translation)
        cosets.setdefault(rotation, (x, y, z))
        if rotation == _IDENTITY_ROTATION:
            centring.add((x, y, z))

    def canonical(vector: _Vector) -> _Vector:
        return min(add(vector, translation) for translation in centring)

    def span(steps: set[_Vector]) -> set[_Vector]:
        """Close ``{0}`` under adding ``steps`` modulo ``T``."""
        found = frontier = {canonical((0, 0, 0))}
        while frontier:
            frontier = {canonical(add(value, step)) for value in frontier for step in steps} - found
            found = found | frontier
        return found

    def scaled(vector: _Vector, divisor: int) -> _Vector:
        return (vector[0] // divisor, vector[1] // divisor, vector[2] // divisor)

    generators = ((scale, 0, 0), (0, scale, 0), (0, 0, scale), *centring)
    values = []
    classes: dict[_Rotation, _CosetClass] = {}
    projections: dict[_Rotation, tuple[_Rotation, int, _Vector, list[_Vector]]] = {}
    for rotation, representative in cosets.items():
        power, total, order = rotation, _IDENTITY_ROTATION, 1
        while power != _IDENTITY_ROTATION:
            power, total, order = (
                _rotation_product(rotation, power),
                cast(_Rotation, tuple(a + b for a, b in zip(total, power))),
                order + 1,
            )
        intrinsic = scaled(_apply(total, representative), order)
        steps = [scaled(_apply(total, generator), order) for generator in generators]
        offsets = span({canonical(step) for step in steps})
        screw = min(
            next(
                multiple
                for multiple in range(1, order + 1)
                if tuple(multiple * value % scale for value in add(intrinsic, offset)) in centring
            )
            for offset in offsets
        )
        classes[rotation] = (
            _determinant(rotation),
            rotation[0] + rotation[4] + rotation[8],
            order,
            screw,
            len(offsets),
        )
        projections[rotation] = (total, order, intrinsic, steps)
        values.append(classes[rotation])

    # How screw and glide translations lie relative to the other symmetry elements (Pbam,
    # Pbcm and Pnnm agree on all values above): for a screw or glide coset ``a`` and any
    # coset ``b``, whether some intrinsic translation ``w`` of ``a`` (in ``w(t) + (1/n_a) N_a T``)
    # satisfies ``P_b w = w`` modulo ``T``, where ``P_b = (1/n_b) N_b`` projects onto the axis
    # or plane of ``b``.
    def displacement(total: _Rotation, order: int, vector: _Vector) -> _Vector:
        x, y, z = scaled(_apply(total, vector), order)
        return (x - vector[0], y - vector[1], z - vector[2])

    relations = []
    for first, (_, _, intrinsic, steps) in projections.items():
        if classes[first][3] == 1:
            continue
        for second, (total, order, _, _) in projections.items():
            moved = displacement(total, order, intrinsic)
            reachable = span({canonical(displacement(total, order, step)) for step in steps})
            relations.append(
                (classes[first], classes[second], any(add(moved, value) in centring for value in reachable))
            )
    # Orbit sizes of the point group on (1/k)T/T separate the arithmetic crystal classes
    # (for example Fm-3m from Im-3m), which the per-coset values above cannot.
    lattice_orbits = []
    for divisor in (2, 3, 4):
        points = span({canonical(scaled(generator, divisor)) for generator in generators})
        sizes = []
        while points:
            point = next(iter(points))
            orbit = {canonical(_apply(rotation, point)) for rotation in cosets}
            points -= orbit
            sizes.append(len(orbit))
        lattice_orbits.append(tuple(sorted(sizes)))
    return len(cosets), tuple(sorted(values)), tuple(lattice_orbits), tuple(sorted(relations))


@cache
def _standard_group_invariant(it_number: int) -> _GroupInvariant:
    """Return the invariant of space-group type ``it_number`` (any tabulated setting)."""
    from httk.atomistic import data

    return _operation_group_invariant(_tabulated_group(data.standard_spacegroup_setting(it_number)["hall_entry"]))


def declared_spacegroup_settings(
    *,
    it_number: int | None,
    hall: str | None,
    hermann_mauguin: str | None,
    hermann_mauguin_extended: str | None,
    operation_group: frozenset[_Operation] | None,
) -> tuple[Mapping[str, Any], ...]:
    """Return tabulated settings consistent with supplied symmetry metadata.

    Raw ``xyz`` operation strings are parsed only for validation and matching; callers
    retain the declared strings at their API boundaries.  Operations equal to a tabulated
    setting select that setting.  Operations in any other basis or origin (for example
    the 48 operations of a primitive face-centred cell) are accepted when their
    basis-independent group invariant equals that of the stated space-group type; the
    matched settings are then all settings of the stated number and symbols.  Operations
    stated without a number or symbol may be any valid group; its type is not identified
    and no setting is matched.

    :param it_number: Optional International Tables space-group number.
    :param hall: Optional Hall symbol.
    :param hermann_mauguin: Optional short Hermann–Mauguin symbol.
    :param hermann_mauguin_extended: Optional extended Hermann–Mauguin symbol.
    :param operation_group: Optional normalized operation group to match.
    :return: Matching tabulated settings.
    :raises ValueError: If supplied identifiers or operations are inconsistent.
    """
    identifiers = any(value is not None for value in (it_number, hall, hermann_mauguin, hermann_mauguin_extended))
    if not identifiers and operation_group is None:
        return ()

    from httk.atomistic import data

    records = list(data.spacegroup_settings())
    if it_number is not None:
        records = [record for record in records if record["it_number"] == it_number]
    if hall is not None:
        records = [record for record in records if record.get("hall") == hall]
    if hermann_mauguin is not None:
        records = [record for record in records if record.get("hm_short") == hermann_mauguin]
    if hermann_mauguin_extended is not None:
        records = [
            record
            for record in records
            if " ".join(str(record.get("hm_extended") or "").split()) == " ".join(hermann_mauguin_extended.split())
        ]
    if identifiers and not records:
        raise ValueError("supplied space-group number and symbols are inconsistent")
    if operation_group is not None:
        matching = [record for record in records if _tabulated_group(record["hall_entry"]) == operation_group]
        if matching:
            records = matching
        elif not identifiers:
            # Operations alone may be any valid group in any basis; its type is not identified.
            records = []
        elif _operation_group_invariant(operation_group) not in {
            _standard_group_invariant(number) for number in {record["it_number"] for record in records}
        }:
            if it_number is None or any(
                value is not None for value in (hall, hermann_mauguin, hermann_mauguin_extended)
            ):
                raise ValueError("supplied space-group operations disagree with its number or symbols")
            raise ValueError(
                f"supplied space-group operations are not compatible with space group {it_number}: they form a "
                "group of a different type (operations expressed in a supercell describe only the subgroup "
                "integral in that cell)"
            )
    return tuple(records)


@dataclass(frozen=True)
class StructureSymmetry:
    """Store optional, explicitly supplied symmetry metadata for a unit-cell structure.

    :param space_group_it_number: Optional International Tables space-group number.
    :param space_group_symbol_hall: Optional Hall symbol.
    :param space_group_symbol_hermann_mauguin: Optional short Hermann–Mauguin symbol.
    :param space_group_symbol_hermann_mauguin_extended: Optional extended Hermann–Mauguin symbol.
    :param space_group_symmetry_operations_xyz: Optional declared raw ``xyz`` operations.
    :param wyckoff_positions: Optional Wyckoff letters aligned with represented sites.
    :raises TypeError: If a symbol is not a string.
    :raises ValueError: If the metadata is invalid or mutually inconsistent.
    """

    space_group_it_number: int | None = None
    space_group_symbol_hall: str | None = None
    space_group_symbol_hermann_mauguin: str | None = None
    space_group_symbol_hermann_mauguin_extended: str | None = None
    space_group_symmetry_operations_xyz: tuple[str, ...] | None = None
    wyckoff_positions: tuple[str, ...] | None = None
    matched_settings: ClassVar[tuple[Mapping[str, Any], ...]]

    def __reduce__(self) -> tuple[type[Self], tuple[Any, ...]]:
        """Rebuild through the constructor from the declared fields only.

        ``matched_settings`` holds read-only (unpicklable) setting copies; the memoised
        validation recomputes it on unpickling.

        :return: The constructor and its positional arguments.
        """
        return type(self), tuple(getattr(self, field.name) for field in fields(self))

    def __post_init__(self) -> None:
        number = self.space_group_it_number
        if number is not None and (not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 230):
            raise ValueError("space_group_it_number must be an integer in [1, 230]")
        for name in (
            "space_group_symbol_hall",
            "space_group_symbol_hermann_mauguin",
            "space_group_symbol_hermann_mauguin_extended",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{name} must be a non-empty string or None")
        operations = self.space_group_symmetry_operations_xyz
        if operations is not None:
            operations = tuple(operations)
            if not all(isinstance(value, str) for value in operations):
                raise ValueError("space_group_symmetry_operations_xyz must contain valid operations and identity")
            object.__setattr__(self, "space_group_symmetry_operations_xyz", operations)
        positions = self.wyckoff_positions
        if positions is not None:
            positions = tuple(positions)
            if not all(isinstance(value, str) for value in positions):
                raise ValueError("wyckoff_positions must contain lowercase Wyckoff letters")
            object.__setattr__(self, "wyckoff_positions", positions)
        settings = _validated_symmetry(
            number,
            self.space_group_symbol_hall,
            self.space_group_symbol_hermann_mauguin,
            self.space_group_symbol_hermann_mauguin_extended,
            operations,
            positions,
        )
        object.__setattr__(self, "matched_settings", settings)


@lru_cache(maxsize=4096)
def _validated_symmetry(
    number: int | None,
    hall: str | None,
    hermann_mauguin: str | None,
    hermann_mauguin_extended: str | None,
    operations: tuple[str, ...] | None,
    positions: tuple[str, ...] | None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate cell-free symmetry metadata and return its matched settings.

    Memoised so that the many structures sharing one operation set validate it once; the
    returned tuple holds shared, read-only copies of the tabulated setting records.

    :param number: Optional International Tables space-group number.
    :param hall: Optional Hall symbol.
    :param hermann_mauguin: Optional short Hermann–Mauguin symbol.
    :param hermann_mauguin_extended: Optional extended Hermann–Mauguin symbol.
    :param operations: Optional declared raw ``xyz`` operations.
    :param positions: Optional Wyckoff letters.
    :return: The tabulated settings consistent with the metadata.
    :raises ValueError: If the metadata is invalid or mutually inconsistent.
    """
    operation_group = None
    if operations is not None:
        if (
            not operations
            or not all(
                len(value.split(",")) == 3 and all(_SYMOP_COORDINATE.fullmatch(part) for part in value.split(","))
                for value in operations
            )
            or "x,y,z" not in {value.replace(" ", "") for value in operations}
        ):
            raise ValueError("space_group_symmetry_operations_xyz must contain valid operations and identity")
        operation_group = _symmetry_operation_group(operations)
    if positions is not None and not all(len(value) == 1 and "a" <= value <= "z" for value in positions):
        raise ValueError("wyckoff_positions must contain lowercase Wyckoff letters")
    settings = declared_spacegroup_settings(
        it_number=number,
        hall=hall,
        hermann_mauguin=hermann_mauguin,
        hermann_mauguin_extended=hermann_mauguin_extended,
        operation_group=operation_group,
    )
    if positions is not None and settings:
        from httk.atomistic.symmetry.spacegroup import Spacegroup

        if not any(
            all(letter in {value.letter for value in Spacegroup(record).wyckoff} for letter in positions)
            for record in settings
        ):
            raise ValueError("supplied Wyckoff positions disagree with the space-group setting")
    return tuple(_frozen_setting(record["hall_entry"]) for record in settings)


def _frozen(value: Any) -> Any:
    """Return a read-only deep copy of a JSON-like value (mappings and lists)."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


@cache
def _frozen_setting(hall_entry: str) -> Mapping[str, Any]:
    """Return a read-only copy of one tabulated setting, shared by every matching symmetry."""
    from httk.atomistic import data

    return cast(Mapping[str, Any], _frozen(data.spacegroup_setting(hall_entry=hall_entry)))


def _stated_symmetry(symmetry: StructureSymmetry | None) -> StructureSymmetry | None:
    """Return ``symmetry``, or ``None`` when it states nothing (every field ``None``).

    ``None`` is the one canonical "no symmetry stated" value; an all-empty
    :class:`StructureSymmetry` must never reach a live structure or a storage record.
    """
    if symmetry is None or all(getattr(symmetry, field.name) is None for field in fields(symmetry)):
        return None
    return symmetry


def validate_optimization_type(value: str | None) -> str | None:
    """Validate an OPTIMADE optimization type.

    :param value: The optimization type to validate, or ``None``.
    :return: The validated value.
    :raises ValueError: If the value is not one of the supported optimization types.
    """
    if value is None:
        return None
    if not isinstance(value, str) or value not in _OPTIMIZATION_TYPES:
        raise ValueError(f"optimization_type must be one of {sorted(_OPTIMIZATION_TYPES)!r} or None")
    return value


def _formula_counts(formula: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(formula, str) or not formula:
        raise ValueError("chemical_formula_hill must be a non-empty formula string")
    position = 0
    values: list[tuple[str, int]] = []
    while position < len(formula):
        match = _FORMULA_TOKEN.match(formula, position)
        if match is None:
            raise ValueError("chemical_formula_hill has invalid syntax")
        element = match.group(1)
        if element not in _ELEMENTS:
            raise ValueError(f"chemical_formula_hill contains unknown element {element!r}")
        if any(existing == element for existing, _ in values):
            raise ValueError("chemical_formula_hill repeats an element")
        values.append((element, int(match.group(2) or 1)))
        position = match.end()
    elements = [element for element, _ in values]
    expected = (
        (["C"] + (["H"] if "H" in elements else []) + sorted(set(elements) - {"C", "H"}))
        if "C" in elements
        else sorted(elements)
    )
    if elements != expected:
        raise ValueError("chemical_formula_hill is not in Hill order")
    return tuple(values)


def validate_hill_formula(formula: str | None, composition: Composition | None) -> str | None:
    """Validate an explicitly assigned Hill formula without inventing its molecular scale.

    :param formula: The formula to validate, or ``None``.
    :param composition: Optional complete composition to cross-check.
    :return: The unchanged formula, or ``None``.
    :raises ValueError: If the formula syntax, order, elements, or ratios are invalid.
    """
    if formula is None:
        return None
    counts = _formula_counts(formula)
    if composition is not None and composition.complete and composition.amounts:
        stated = dict(counts)
        actual = dict(composition.amounts)
        if set(stated) != set(actual):
            raise ValueError("chemical_formula_hill elements disagree with the complete composition")
        first = next(iter(stated))
        scale = actual[first] / stated[first]
        if scale <= 0 or any(actual[element] != Fraction(count) * scale for element, count in counts):
            raise ValueError("chemical_formula_hill ratios disagree with the complete composition")
    return formula


def validate_descriptive_formula(formula: str | None) -> str | None:
    """Validate the permissive OPTIMADE descriptive-formula token and bracket grammar.

    :param formula: The formula to validate, or ``None``.
    :return: The unchanged formula, or ``None``.
    :raises ValueError: If the formula is empty, malformed, or contains unknown tokens.
    """

    if formula is None:
        return None
    if not isinstance(formula, str) or not formula:
        raise ValueError("chemical_formula_descriptive must be a non-empty formula string")
    brackets: list[str] = []
    closing = {")": "(", "]": "[", "}": "{"}
    index = 0
    while index < len(formula):
        char = formula[index]
        if char.isspace() or char in ",+-:=":
            index += 1
            continue
        if char in "([{":
            brackets.append(char)
            index += 1
            continue
        if char in closing:
            if not brackets or brackets.pop() != closing[char]:
                raise ValueError("chemical_formula_descriptive has unbalanced brackets")
            index += 1
            continue
        element = re.match(r"[A-Z][a-z]?", formula[index:])
        if element is not None:
            if element.group(0) not in _ELEMENTS and element.group(0) not in _DESCRIPTIVE_GROUPS:
                raise ValueError("chemical_formula_descriptive contains an unknown element")
            index += len(element.group(0))
            continue
        number = _DESCRIPTIVE_NUMBER.match(formula, index)
        if number is not None:
            index = number.end()
            continue
        raise ValueError("chemical_formula_descriptive contains invalid formula text")
    if brackets:
        raise ValueError("chemical_formula_descriptive has unbalanced brackets")
    return formula


def _semantic_value(
    source: Any,
    public_name: str,
    default: Any = None,
    private_name: str | None = None,
) -> Any:
    effective = getattr(source, "_effective_asu", None)
    if callable(effective):
        source = effective()
    namespace = getattr(source, "__dict__", {})
    if private_name is not None:
        if private_name in namespace:
            return namespace[private_name]
        source = namespace.get("_backend")
        if source is None:
            return default
    marker = object()
    value = getattr(source, public_name, marker)
    if value is not marker:
        return value
    unwrap = getattr(source, "unwrap", None)
    owner = unwrap() if callable(unwrap) else None
    return default if owner is None or owner is source else getattr(owner, public_name, default)


class StructureSemanticsMixin:
    """Provide semantics shared by unit-cell, fundamental-domain, and ASU structures."""

    __httk_storage_record__: ClassVar[type[Any]]
    _assemblies: tuple[Assembly, ...] | None
    _chemical_composition: ChemicalComposition | None
    _chemical_formula_descriptive: str | None
    _chemical_formula_hill: str | None
    _optimization_type: str | None
    _molecular: bool
    _immutable_id: str | None
    _last_modified: datetime.datetime | None

    @property
    def type(self) -> str:
        """Expose the logical OPTIMADE entry family.

        :return: ``"structures"``.
        """
        return "structures"

    @property
    def id(self) -> str:
        """Expose the stable content identity of this exact representation.

        :return: The content identifier.
        """
        from httk.core.storage import content_id

        return content_id(self)

    @property
    def immutable_id(self) -> str | None:
        """Expose the immutable source identifier.

        :return: The identifier, or ``None`` when it is unstated.
        """
        return _semantic_value(self, "immutable_id", private_name="_immutable_id")

    @property
    def last_modified(self) -> datetime.datetime | None:
        """Expose the source modification timestamp.

        :return: The timestamp, or ``None`` when it is unstated.
        """
        return _semantic_value(self, "last_modified", private_name="_last_modified")

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        """Expose site assemblies.

        :return: The assemblies, or ``None`` when they are unstated.
        """
        return _semantic_value(self, "assemblies", private_name="_assemblies")

    @property
    def chemical_composition(self) -> ChemicalComposition | None:
        """Expose the supplied chemical composition.

        :return: The composition, or ``None`` when it is unstated.
        """
        return _semantic_value(self, "chemical_composition", private_name="_chemical_composition")

    @cached_property
    def composition(self) -> CompositionView:
        """Present a lazy view over this structure's projected composition.

        ``isinstance(self.composition, Composition)`` holds, and projection runs
        on first data access.

        :return: The lazy composition view of this structure.
        """
        # This mixin is only mixed into structure backends/views, which the formula family accepts.
        return CompositionView(cast("StructureBackend | StructureView", self))

    @property
    def elements(self) -> tuple[str, ...]:
        """Expose the composition's element symbols.

        :return: Element symbols in composition order.
        """
        return self.composition.elements

    @property
    def nelements(self) -> int:
        """Expose the number of composition elements.

        :return: The number of distinct elements.
        """
        return self.composition.nelements

    @property
    def elements_ratios(self) -> tuple[Fraction, ...]:
        """Expose normalized composition element ratios.

        :return: Element ratios in :attr:`elements` order.
        """
        return self.composition.elements_ratios

    @property
    def chemical_formula_reduced(self) -> str | None:
        """Expose the reduced composition formula.

        :return: The reduced formula, or ``None`` when unavailable.
        """
        return self.composition.chemical_formula_reduced

    @property
    def formula(self) -> ChemicalFormulaView:
        """Present the reduced formula as a genuine ``str`` subclass view.

        ``unwrap()`` recovers this structure.

        :return: The reduced formula as a
            :class:`~httk.atomistic.models.formula.formula_view.ChemicalFormulaView`.
        :raises ValueError: If the composition is incomplete (including any ``"X"``
            species) or empty.
        """
        return ChemicalFormulaView(cast("StructureBackend | StructureView", self))

    @property
    def chemical_formula_anonymous(self) -> str | None:
        """Expose the anonymous composition formula.

        :return: The anonymous formula, or ``None`` when unavailable.
        """
        return self.composition.chemical_formula_anonymous

    @property
    def chemical_formula_descriptive(self) -> str | None:
        """Expose the descriptive chemical formula.

        :return: The formula, or ``None`` when it is unstated.
        """
        return _semantic_value(self, "chemical_formula_descriptive", private_name="_chemical_formula_descriptive")

    @property
    def chemical_formula_hill(self) -> str | None:
        """Expose the Hill chemical formula.

        :return: The formula, or ``None`` when it is unstated.
        """
        return _semantic_value(self, "chemical_formula_hill", private_name="_chemical_formula_hill")

    @property
    def optimization_type(self) -> str | None:
        """Expose the optimization provenance.

        :return: The optimization type, or ``None`` when it is unstated.
        """
        return _semantic_value(self, "optimization_type", private_name="_optimization_type")

    @property
    def dimension_types(self) -> tuple[int, int, int]:
        """Expose periodicity as OPTIMADE dimension flags.

        :return: Three ``0``/``1`` flags for the cell directions.
        """
        owner = cast(Any, self)
        return cast(tuple[int, int, int], tuple(1 if value else 0 for value in owner.periodicity))

    @property
    def lattice_vectors(self) -> list[list[float]]:
        """Expose the cell basis at the float presentation boundary.

        :return: The three lattice vectors as float rows.
        """
        return cast(Any, self).cell.basis_floats()

    @property
    def fractional_site_positions(self) -> list[list[float]]:
        """Expose reduced site positions at the float presentation boundary.

        :return: Fractional positions as float rows.
        """
        return cast(Any, self).sites.reduced_coords_floats()

    @property
    def cartesian_site_positions(self) -> list[list[float]]:
        """Expose Cartesian site positions at the float presentation boundary.

        :return: Cartesian positions as float rows.
        """
        return matmul_floats(self.fractional_site_positions, self.lattice_vectors)

    @property
    def nsites(self) -> int:
        """Expose the number of represented sites.

        :return: The site count.
        """
        return len(cast(Any, self).species_at_sites)

    @property
    def structure_features(self) -> tuple[str, ...]:
        """Expose derived OPTIMADE structure-feature flags.

        :return: Feature flags in canonical order.
        """
        return derive_structure_features(self)

    @property
    def site_coordinate_span_description(self) -> str | None:
        """Expose the optional description for a non-standard coordinate span.

        :return: The span description, or ``None`` when unavailable.
        """
        return None

    @property
    def space_group_it_number(self) -> int | None:
        """Expose the International Tables space-group number.

        :return: The number, or ``None`` when symmetry is unstated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.space_group_it_number

    @property
    def space_group_symbol_hall(self) -> str | None:
        """Expose the Hall space-group symbol.

        :return: The Hall symbol, or ``None`` when symmetry is unstated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.space_group_symbol_hall

    @property
    def space_group_symbol_hermann_mauguin(self) -> str | None:
        """Expose the short Hermann–Mauguin symbol.

        :return: The symbol, or ``None`` when symmetry is unstated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.space_group_symbol_hermann_mauguin

    @property
    def space_group_symbol_hermann_mauguin_extended(self) -> str | None:
        """Expose the extended Hermann–Mauguin symbol.

        :return: The symbol, or ``None`` when symmetry is unstated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.space_group_symbol_hermann_mauguin_extended

    @property
    def space_group_symmetry_operations_xyz(self) -> tuple[str, ...] | None:
        """Expose the declared raw ``xyz`` symmetry operations.

        :return: The operation strings, or ``None`` when no symmetry is stated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.space_group_symmetry_operations_xyz

    @property
    def wyckoff_positions(self) -> tuple[str, ...] | None:
        """Expose the site-aligned Wyckoff positions.

        :return: Wyckoff letters, or ``None`` when they are unstated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.wyckoff_positions


def _resolve_view_metadata(
    source: Any,
    *,
    immutable_id: str | None | object = _METADATA_UNSET,
    last_modified: datetime.datetime | None | object = _METADATA_UNSET,
) -> tuple[str | None, datetime.datetime | None]:
    """Inherit view metadata, allowing explicit values only when none existed."""

    inherited_immutable_id = _semantic_value(source, "immutable_id")
    inherited_last_modified = _semantic_value(source, "last_modified")
    if (
        inherited_immutable_id is not None
        and immutable_id is not _METADATA_UNSET
        and immutable_id != inherited_immutable_id
    ):
        raise ValueError("explicit immutable_id conflicts with the wrapped structure")
    if (
        inherited_last_modified is not None
        and last_modified is not _METADATA_UNSET
        and last_modified != inherited_last_modified
    ):
        raise ValueError("explicit last_modified conflicts with the wrapped structure")
    resolved = (
        inherited_immutable_id if immutable_id is _METADATA_UNSET else cast(str | None, immutable_id),
        inherited_last_modified if last_modified is _METADATA_UNSET else cast(datetime.datetime | None, last_modified),
    )
    if resolved[0] is not None and not isinstance(resolved[0], str):
        raise TypeError("immutable_id must be a string or None")
    if resolved[1] is not None:
        if not isinstance(resolved[1], datetime.datetime):
            raise TypeError("last_modified must be a datetime or None")
        if resolved[1].tzinfo is None or resolved[1].utcoffset() is None:
            raise ValueError("last_modified must include a timezone")
    return resolved


def initialize_semantics(
    owner: Any,
    *,
    nsites: int,
    molecular: bool,
    assemblies: tuple[Assembly, ...] | list[Assembly] | None,
    symmetry: StructureSymmetry | None,
    chemical_composition: ChemicalComposition | None,
    chemical_formula_descriptive: str | None,
    chemical_formula_hill: str | None,
    optimization_type: str | None,
    immutable_id: str | None = None,
    last_modified: datetime.datetime | None = None,
) -> None:
    """Validate and store shared structure semantics on an owner.

    :param owner: The structure receiving the semantic fields.
    :param nsites: The number of represented sites.
    :param molecular: Whether the structure describes a molecular unit cell.
    :param assemblies: Optional site assemblies.
    :param symmetry: Optional structure symmetry metadata.
    :param chemical_composition: Optional supplied chemical composition.
    :param chemical_formula_descriptive: Optional descriptive formula.
    :param chemical_formula_hill: Optional Hill formula.
    :param optimization_type: Optional optimization provenance.
    :param immutable_id: Optional immutable source identifier.
    :param last_modified: Optional timezone-aware source timestamp.
    :raises TypeError: If a supplied semantic value has the wrong kind.
    :raises ValueError: If supplied semantics are invalid or inconsistent.
    """
    if not isinstance(molecular, bool):
        raise TypeError("molecular must be a bool")
    if symmetry is not None and not isinstance(symmetry, StructureSymmetry):
        raise TypeError("symmetry must be a StructureSymmetry or None")
    symmetry = _stated_symmetry(symmetry)
    if chemical_composition is not None and not isinstance(chemical_composition, ChemicalComposition):
        raise TypeError("chemical_composition must be a ChemicalComposition or None")
    if immutable_id is not None and not isinstance(immutable_id, str):
        raise TypeError("immutable_id must be a string or None")
    if last_modified is not None:
        if not isinstance(last_modified, datetime.datetime):
            raise TypeError("last_modified must be a datetime or None")
        if last_modified.tzinfo is None or last_modified.utcoffset() is None:
            raise ValueError("last_modified must include a timezone")
    chemical_formula_descriptive = validate_descriptive_formula(chemical_formula_descriptive)
    normalized_assemblies = None if assemblies is None else validate_assemblies(assemblies, nsites)
    if symmetry is not None and symmetry.wyckoff_positions is not None and len(symmetry.wyckoff_positions) != nsites:
        raise ValueError("symmetry wyckoff_positions must match the number of represented sites")
    if (
        symmetry is not None
        and cast(Any, owner).cell.nperiodic_dimensions != 3
        and any(
            value is not None
            for value in (
                symmetry.space_group_it_number,
                symmetry.space_group_symbol_hall,
                symmetry.space_group_symbol_hermann_mauguin,
                symmetry.space_group_symbol_hermann_mauguin_extended,
                symmetry.wyckoff_positions,
            )
        )
    ):
        raise ValueError("space-group identifiers and Wyckoff positions require three periodic dimensions")
    if (
        symmetry is not None
        and cast(Any, owner).cell.nperiodic_dimensions == 0
        and symmetry.space_group_symmetry_operations_xyz is not None
    ):
        raise ValueError("space-group symmetry operations must be null for a nonperiodic structure")
    if (
        symmetry is not None
        and symmetry.space_group_symmetry_operations_xyz is not None
        and cast(Any, owner).cell.nperiodic_dimensions == 3
    ):
        metric = cast(Any, owner).cell.metric().to_floats()
        # Per entry, relative to |a_i| |a_j|: a global bound would let an elongated axis hide
        # a wrong rotation among the short ones.
        bound = [[_METRIC_TOLERANCE * math.sqrt(metric[i][i] * metric[j][j]) for j in range(3)] for i in range(3)]
        rotations = {
            tuple(tuple(float(value) for value in _linear_xyz_expression(part)[:3]) for part in operation.split(","))
            for operation in symmetry.space_group_symmetry_operations_xyz
        }
        for rotation in rotations:
            # (R^T G R)_ij = sum_kl R_ki G_kl R_lj for the column-convention rotation R.
            if any(
                abs(
                    sum(rotation[k][i] * metric[k][l] * rotation[l][j] for k in range(3) for l in range(3))
                    - metric[i][j]
                )
                > bound[i][j]
                for i in range(3)
                for j in range(3)
            ):
                raise ValueError("space-group symmetry operations do not preserve the cell metric")
    owner._molecular = molecular
    owner._assemblies = normalized_assemblies
    owner._symmetry = symmetry
    owner._chemical_composition = chemical_composition
    owner._chemical_formula_descriptive = chemical_formula_descriptive
    owner._optimization_type = validate_optimization_type(optimization_type)
    owner._immutable_id = immutable_id
    owner._last_modified = last_modified
    owner._chemical_formula_hill = (
        None
        if chemical_formula_hill is None
        else validate_hill_formula(chemical_formula_hill, project_composition(owner))
    )


__all__ = [
    "OptimizationType",
    "StructureSemanticsMixin",
    "StructureSymmetry",
    "initialize_semantics",
    "validate_descriptive_formula",
    "validate_hill_formula",
    "validate_optimization_type",
]
