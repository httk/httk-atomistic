"""Shared OPTIMADE structure semantics for native atomistic representations."""

from __future__ import annotations

import ast
import datetime
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from httk.atomistic.composition import (
    Assembly,
    ChemicalComposition,
    derive_structure_features,
    project_composition,
    validate_assemblies,
)
from httk.atomistic.elements import SYMBOLS
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


def _symmetry_operation_group(operations: tuple[str, ...]) -> frozenset[Any]:
    """Return parsed wrapped operations after exact finite-group validation."""

    from httk.atomistic.symmetry.affine_operation import AffineOperation

    parsed = []
    for operation in operations:
        components = [_linear_xyz_expression(part) for part in operation.split(",")]
        parsed.append(AffineOperation([part[:3] for part in components], [part[3] for part in components]).wrapped())
    group = frozenset(parsed)
    if len(group) != len(operations):
        raise ValueError("space-group symmetry operations must be unique modulo lattice translations")
    identity = AffineOperation.identity()
    if identity not in group:
        raise ValueError("space-group symmetry operations must contain identity")
    if any(abs(operation.determinant()) != 1 for operation in group):
        raise ValueError("space-group symmetry operations must have unimodular rotation parts")
    if any((left * right).wrapped() not in group for left in group for right in group):
        raise ValueError("space-group symmetry operations are not closed under composition")
    if any(operation.inverse().wrapped() not in group for operation in group):
        raise ValueError("space-group symmetry operations are not closed under inverses")
    return group


def _operation_group_signature(group: frozenset[Any]) -> tuple[tuple[Fraction, Fraction, int, int, int], ...]:
    """Integral-affine invariants used to cross-check an untabulated setting.

    The final value records the content of the lattice translation produced when
    the rotation part returns to identity.  It distinguishes, for example, a
    twofold rotation from a two-one screw even though their wrapped elements
    have the same order, trace, and determinant.
    """

    from httk.atomistic.symmetry.affine_operation import AffineOperation

    identity = AffineOperation.identity()
    values: list[tuple[Fraction, Fraction, int, int, int]] = []
    for operation in group:
        power = identity
        order = 0
        for order in range(1, len(group) + 1):
            power = (operation * power).wrapped()
            if power == identity:
                break
        else:  # pragma: no cover - closure validation makes this unreachable
            raise ValueError("space-group symmetry operation has no finite order")
        matrix = operation.matrix.to_fractions()
        trace = sum((matrix[index][index] for index in range(3)), start=Fraction(0))
        rotation_power = operation.matrix
        rotation_order = 1
        identity_matrix = identity.matrix
        while rotation_power != identity_matrix and rotation_order <= len(group):
            rotation_power = rotation_power * operation.matrix
            rotation_order += 1
        if rotation_power != identity_matrix:
            raise ValueError("space-group rotation part has no finite order")
        unwrapped_power = identity
        for _ in range(order):
            unwrapped_power = operation * unwrapped_power
        translation = unwrapped_power.vector.to_fractions()
        if any(value.denominator != 1 for value in translation):
            raise ValueError("space-group operation power is not a lattice translation")
        translation_content = math.gcd(*(abs(value.numerator) for value in translation))
        values.append((operation.determinant(), trace, order, rotation_order, translation_content))
    return tuple(sorted(values))


def declared_spacegroup_settings(
    *,
    it_number: int | None,
    hall: str | None,
    hermann_mauguin: str | None,
    hermann_mauguin_extended: str | None,
    operation_group: frozenset[Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Return tabulated settings consistent with supplied symmetry metadata.

    Raw ``xyz`` operation strings are parsed only for validation and matching; callers
    retain the declared strings at their API boundaries.

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
    from httk.atomistic.symmetry.spacegroup import Spacegroup

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
        matching = [
            record
            for record in records
            if frozenset(value.wrapped() for value in Spacegroup(record).symmetry_operations) == operation_group
        ]
        if not matching:
            standard_group = (
                frozenset(value.wrapped() for value in Spacegroup.standard(it_number).symmetry_operations)
                if it_number is not None
                else None
            )
            has_setting_symbol = any(value is not None for value in (hall, hermann_mauguin, hermann_mauguin_extended))
            if (
                has_setting_symbol
                or standard_group is None
                or len(operation_group) != len(standard_group)
                or _operation_group_signature(operation_group) != _operation_group_signature(standard_group)
            ):
                raise ValueError("supplied space-group operations disagree with its number or symbols")
        else:
            records = matching
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
    matched_settings: ClassVar[tuple[dict[str, Any], ...]]

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
        operation_group = None
        if operations is not None:
            operations = tuple(operations)
            if (
                not operations
                or not all(
                    isinstance(value, str)
                    and len(value.split(",")) == 3
                    and all(_SYMOP_COORDINATE.fullmatch(part) for part in value.split(","))
                    for value in operations
                )
                or "x,y,z" not in {value.replace(" ", "") for value in operations}
            ):
                raise ValueError("space_group_symmetry_operations_xyz must contain valid operations and identity")
            operation_group = _symmetry_operation_group(operations)
            object.__setattr__(self, "space_group_symmetry_operations_xyz", operations)
        positions = self.wyckoff_positions
        if positions is not None:
            positions = tuple(positions)
            if not all(isinstance(value, str) and len(value) == 1 and "a" <= value <= "z" for value in positions):
                raise ValueError("wyckoff_positions must contain lowercase Wyckoff letters")
            object.__setattr__(self, "wyckoff_positions", positions)

        settings = declared_spacegroup_settings(
            it_number=number,
            hall=self.space_group_symbol_hall,
            hermann_mauguin=self.space_group_symbol_hermann_mauguin,
            hermann_mauguin_extended=self.space_group_symbol_hermann_mauguin_extended,
            operation_group=operation_group,
        )
        if operations is None and settings:
            from httk.atomistic.symmetry.spacegroup import Spacegroup

            has_setting_symbol = any(
                value is not None
                for value in (
                    self.space_group_symbol_hall,
                    self.space_group_symbol_hermann_mauguin,
                    self.space_group_symbol_hermann_mauguin_extended,
                )
            )
            selected = Spacegroup(settings[0]) if has_setting_symbol or number is None else Spacegroup.standard(number)
            operations = tuple(value.wrapped().to_xyz() for value in selected.symmetry_operations)
            object.__setattr__(self, "space_group_symmetry_operations_xyz", operations)
        object.__setattr__(self, "matched_settings", settings)
        if positions is not None and settings:
            from httk.atomistic.symmetry.spacegroup import Spacegroup

            if not any(
                all(letter in {value.letter for value in Spacegroup(record).wyckoff} for letter in positions)
                for record in settings
            ):
                raise ValueError("supplied Wyckoff positions disagree with the space-group setting")


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
        return cast(Any, self).cell.basis.to_floats()

    @property
    def fractional_site_positions(self) -> list[list[float]]:
        """Expose reduced site positions at the float presentation boundary.

        :return: Fractional positions as float rows.
        """
        return cast(Any, self).sites.reduced_coords.to_floats()

    @property
    def cartesian_site_positions(self) -> list[list[float]]:
        """Expose Cartesian site positions at the float presentation boundary.

        :return: Cartesian positions as float rows.
        """
        return cast(Any, self).cartesian_sites().to_floats()

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

        :return: The operation strings, or the identity for a periodic structure without
            explicit operations.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        if symmetry is not None and symmetry.space_group_symmetry_operations_xyz is not None:
            return symmetry.space_group_symmetry_operations_xyz
        return ("x,y,z",) if cast(Any, self).nperiodic_dimensions else None

    @property
    def wyckoff_positions(self) -> tuple[str, ...] | None:
        """Expose the site-aligned Wyckoff positions.

        :return: Wyckoff letters, or ``None`` when they are unstated.
        """
        symmetry = _semantic_value(self, "symmetry", private_name="_symmetry")
        return None if symmetry is None else symmetry.wyckoff_positions


_METADATA_UNSET = object()


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
