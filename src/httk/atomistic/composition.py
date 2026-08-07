"""Exact, precision-aware chemical composition projection."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal

from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.models.species.species import Species

from ._composition_values import as_fraction, as_precision, normalization
from .elements import SYMBOLS

_ELEMENTS = frozenset(SYMBOLS)

__all__ = [
    "Assembly",
    "ChemicalComposition",
    "CompositionDiagnostic",
    "derive_structure_features",
    "project_composition",
    "validate_assemblies",
]


def _normalization_diagnostic(
    subject: str, values: tuple[Fraction, ...], precisions: tuple[Fraction | None, ...]
) -> tuple[bool, str, CompositionDiagnostic | None]:
    ok, status, total, width = normalization(values, precisions)
    if ok:
        return ok, status, None
    interval = f"[{total - (width or 0)}, {total + (width or 0)}]" if width is not None else str(total)
    return (
        False,
        status,
        CompositionDiagnostic(
            "normalization_outside_precision",
            f"{subject} sums to {total}, whose stated interval {interval} does not contain 1",
            subject,
            total,
            width,
        ),
    )


@dataclass(frozen=True)
class Assembly:
    """Represent one site-disorder assembly without normalizing its probabilities.

    :param sites_in_groups: The non-overlapping site-index groups in the assembly.
    :param group_probabilities: The probability assigned to each group.
    :param group_probabilities_precision: The precision of each group probability, if known.
    """

    sites_in_groups: tuple[tuple[int, ...], ...]
    group_probabilities: tuple[Fraction, ...]
    group_probabilities_precision: tuple[Fraction | None, ...] | None = None

    def __post_init__(self) -> None:
        groups = tuple(tuple(group) for group in self.sites_in_groups)
        if not groups or len(groups) != len(self.group_probabilities):
            raise ValueError("Assembly groups and probabilities must have matching non-empty lengths")
        seen: set[int] = set()
        for group in groups:
            if not group:
                raise ValueError("Assembly groups must be non-empty")
            for index in group:
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise ValueError("Assembly site indices must be non-negative integers")
                if index in seen:
                    raise ValueError("An Assembly cannot contain a site index more than once")
                seen.add(index)
        values: list[Fraction] = []
        inferred: list[Fraction | None] = []
        for value in self.group_probabilities:
            central, width = as_fraction(value, field="Assembly group probability")
            if not 0 <= central <= 1:
                raise ValueError("Assembly group probabilities must be in [0, 1]")
            values.append(central)
            inferred.append(width)
        stated = self.group_probabilities_precision
        if stated is None:
            precisions = tuple(inferred)
        else:
            if len(stated) != len(values):
                raise ValueError("Assembly group_probabilities_precision must match probabilities")
            precisions = tuple(as_precision(value, field="Assembly group probability precision") for value in stated)
        object.__setattr__(self, "sites_in_groups", groups)
        object.__setattr__(self, "group_probabilities", tuple(values))
        object.__setattr__(self, "group_probabilities_precision", precisions)

    @cached_property
    def _normalization(self) -> tuple[bool, str, CompositionDiagnostic | None]:
        return _normalization_diagnostic(
            "assembly probabilities", self.group_probabilities, self.group_probabilities_precision or ()
        )

    @property
    def normalized(self) -> bool:
        """Whether the group probabilities sum to one within their precision."""
        return self._normalization[0]

    @property
    def normalization_status(self) -> str:
        """Return the probability normalization status."""
        return self._normalization[1]

    @property
    def normalization_diagnostic(self) -> CompositionDiagnostic | None:
        """Return the normalization diagnostic, if the probabilities are outside precision."""
        return self._normalization[2]


@dataclass(frozen=True, init=False)
class ChemicalComposition:
    """Store explicit elemental amounts as additional or authoritative composition.

    ``implicit`` amounts supplement the site-derived composition; ``full`` amounts replace it
    while still recording a mismatch diagnostic when the two disagree.

    :param amounts: The positive amounts for named chemical elements.
    :param mode: Whether the amounts are ``"implicit"`` or authoritative ``"full"`` values.
    :param amounts_precision: The precision of the stated amounts, if known.
    """

    amounts: tuple[tuple[str, Fraction], ...]
    amounts_precision: tuple[tuple[str, Fraction | None], ...]
    mode: Literal["implicit", "full"]

    def __init__(
        self,
        amounts: Mapping[str, Any] | Iterable[tuple[str, Any]],
        mode: Literal["implicit", "full"] = "implicit",
        amounts_precision: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        if mode not in {"implicit", "full"}:
            raise ValueError("ChemicalComposition mode must be 'implicit' or 'full'")
        raw = dict(amounts)
        if not raw:
            raise ValueError("ChemicalComposition requires at least one element amount")
        supplied = {} if amounts_precision is None else dict(amounts_precision)
        if set(supplied) - set(raw):
            raise ValueError("ChemicalComposition precision keys must name stated elements")
        converted: list[tuple[str, Fraction]] = []
        precisions: list[tuple[str, Fraction | None]] = []
        for element in sorted(raw):
            if element not in _ELEMENTS:
                raise ValueError(f"ChemicalComposition amount is not a real element: {element!r}")
            central, inferred = as_fraction(raw[element], field=f"ChemicalComposition amount for {element}")
            if central <= 0:
                raise ValueError("ChemicalComposition amounts must be positive")
            converted.append((element, central))
            width = (
                as_precision(supplied[element], field=f"ChemicalComposition precision for {element}")
                if element in supplied
                else inferred
            )
            precisions.append((element, width))
        object.__setattr__(self, "amounts", tuple(converted))
        object.__setattr__(self, "amounts_precision", tuple(precisions))
        object.__setattr__(self, "mode", mode)

    @property
    def elements(self) -> tuple[str, ...]:
        """Return the element symbols in the stored amount order."""
        return tuple(element for element, _ in self.amounts)

    @property
    def amount_mapping(self) -> Mapping[str, Fraction]:
        """Return the elemental amounts as a read-only mapping."""
        return MappingProxyType(dict(self.amounts))

    @property
    def precision_mapping(self) -> Mapping[str, Fraction | None]:
        """Return the amount precisions as a read-only mapping."""
        return MappingProxyType(dict(self.amounts_precision))


def validate_assemblies(assemblies: Iterable[Assembly], nsites: int | None = None) -> tuple[Assembly, ...]:
    """Validate global assembly site ownership for a structure.

    :param assemblies: The assemblies to validate.
    :param nsites: The structure site count used to bound site indices, if supplied.
    :return: The validated assemblies in their input order.
    :raises TypeError: If an item is not an :class:`Assembly`.
    :raises ValueError: If a site index is out of bounds or occurs in multiple assemblies.
    """
    values = tuple(assemblies)
    seen: set[int] = set()
    for assembly in values:
        if not isinstance(assembly, Assembly):
            raise TypeError("assemblies must contain Assembly values")
        for group in assembly.sites_in_groups:
            for index in group:
                if nsites is not None and index >= nsites:
                    raise ValueError("Assembly site index is outside the structure")
                if index in seen:
                    raise ValueError("A site index cannot occur in more than one Assembly")
                seen.add(index)
    return values


def _site_data(structure: Any) -> tuple[tuple[str, ...], tuple[Fraction, ...], tuple[Species, ...]]:
    species = tuple(structure.species)
    if hasattr(structure, "wyckoff_sites") and hasattr(structure, "multiplicities"):
        names = tuple(site.species for site in structure.wyckoff_sites)
        factors = tuple(Fraction(value) for value in structure.multiplicities())
    else:
        names = tuple(structure.species_at_sites)
        factors = tuple(Fraction(1) for _ in names)
    return names, factors, species


def derive_structure_features(structure: Any) -> tuple[str, ...]:
    """Return the exact-composition features present on a structure.

    :param structure: The structure whose composition-related features are inspected.
    :return: The feature names in alphabetical order.
    """
    names, _, species = _site_data(structure)
    by_name = {value.name: value for value in species}
    used = tuple(by_name[name] for name in names if name in by_name)
    features: set[str] = set()
    if getattr(structure, "assemblies", None) is not None:
        features.add("assemblies")
    if any(len(value.chemical_symbols) > 1 or value.concentration != (Fraction(1),) for value in used):
        features.add("disorder")
    if any(value.attached for value in used):
        features.add("site_attachments")
    chemical = getattr(structure, "chemical_composition", None)
    if isinstance(chemical, ChemicalComposition) and chemical.mode == "implicit":
        features.add("implicit_atoms")
    if getattr(structure, "site_moments", None) is not None:
        features.add("_httk_magnetism")
    return tuple(sorted(features))


def project_composition(structure: Any) -> Composition:
    """Project a structure to exact elemental amounts without normalization.

    Site multiplicities, disorder, assemblies, attached elements, and explicit composition
    semantics are combined without silently renormalizing their stated values.

    :param structure: The unit-cell or asymmetric-unit structure to project.
    :return: The projected composition and its completeness, precision, and diagnostics.
    :raises TypeError: If the structure has an invalid explicit chemical composition.
    :raises ValueError: If a structure site refers to an unknown species or has invalid assemblies.
    """
    names, factors, species = _site_data(structure)
    by_name = {value.name: value for value in species}
    assemblies = validate_assemblies(getattr(structure, "assemblies", ()) or (), len(names))
    probability_by_site: dict[int, tuple[Fraction, Fraction | None]] = {}
    diagnostics: list[CompositionDiagnostic] = []
    statuses: list[str] = []
    for assembly in assemblies:
        statuses.append(assembly.normalization_status)
        if assembly.normalization_diagnostic is not None:
            diagnostics.append(assembly.normalization_diagnostic)
        for group, probability, width in zip(
            assembly.sites_in_groups, assembly.group_probabilities, assembly.group_probabilities_precision or ()
        ):
            for index in group:
                probability_by_site[index] = (probability, width)
    amounts: dict[str, Fraction] = {}
    widths: dict[str, Fraction | None] = {}

    def add(element: str, value: Fraction, width: Fraction | None) -> None:
        if element not in _ELEMENTS or value == 0:
            return
        amounts[element] = amounts.get(element, Fraction()) + value
        old = widths.get(element)
        widths[element] = None if old is None and width is None else (old or Fraction()) + (width or Fraction())

    for index, (name, factor) in enumerate(zip(names, factors)):
        try:
            species_value = by_name[name]
        except KeyError as exc:
            raise ValueError(f"composition site references unknown species name: {name!r}") from exc
        status = species_value.normalization_status
        statuses.append(status)
        if species_value.normalization_diagnostic is not None:
            diagnostics.append(species_value.normalization_diagnostic)
        probability, probability_width = probability_by_site.get(index, (Fraction(1), None))
        site_factor = factor * probability
        site_width = factor * (probability_width or 0)
        for element, concentration, concentration_width in zip(
            species_value.chemical_symbols,
            species_value.concentration,
            species_value.concentration_precision or (),
        ):
            value = site_factor * concentration
            width = (
                abs(site_factor) * (concentration_width or 0)
                + abs(concentration) * site_width
                + site_width * (concentration_width or 0)
            )
            add(element, value, None if concentration_width is None and probability_width is None else width)
        if species_value.attached is not None and species_value.nattached is not None:
            for element, count in zip(species_value.attached, species_value.nattached):
                add(element, site_factor * count, None if probability_width is None else site_width * count)

    site_amounts = dict(amounts)
    site_widths = dict(widths)
    chemical = getattr(structure, "chemical_composition", None)
    if chemical is not None and not isinstance(chemical, ChemicalComposition):
        raise TypeError("chemical_composition must be a ChemicalComposition")
    complete = not any("X" in by_name[name].chemical_symbols or "X" in (by_name[name].attached or ()) for name in names)
    if chemical is not None:
        stated = dict(chemical.amounts)
        stated_widths = dict(chemical.amounts_precision)
        if chemical.mode == "implicit":
            for element, value in stated.items():
                add(element, value, stated_widths[element])
        else:
            complete = True
            for element in sorted(set(site_amounts) | set(stated)):
                left, right = site_amounts.get(element, Fraction()), stated.get(element, Fraction())
                left_width, right_width = site_widths.get(element) or 0, stated_widths.get(element) or 0
                if left + left_width < right - right_width or right + right_width < left - left_width:
                    diagnostics.append(
                        CompositionDiagnostic(
                            "full_composition_mismatch", f"full composition disagrees with sites for {element}", element
                        )
                    )
            amounts, widths = stated, stated_widths
    ordered = tuple((element, amounts[element]) for element in sorted(amounts) if amounts[element])
    uncertainty = tuple((element, widths.get(element)) for element, _ in ordered)
    normalized = all(status != "outside_precision" for status in statuses)
    status = (
        "outside_precision" if not normalized else ("within_precision" if "within_precision" in statuses else "exact")
    )
    exact = all(width is None for _, width in uncertainty)
    return Composition(
        ordered,
        uncertainty,
        complete,
        exact,
        normalized,
        status,
        tuple(diagnostics),
    )
