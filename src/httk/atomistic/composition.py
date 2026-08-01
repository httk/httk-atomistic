"""Exact, precision-aware chemical composition projection."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from types import MappingProxyType
from typing import Any, Literal

from ._composition_values import as_fraction, as_precision, normalization
from .elements import SYMBOLS
from .species import Species

_ELEMENTS = frozenset(SYMBOLS)

__all__ = [
    "Assembly",
    "ChemicalComposition",
    "CompositionDiagnostic",
    "CompositionResult",
    "anonymous_symbol",
    "derive_structure_features",
    "project_composition",
    "validate_assemblies",
]


@dataclass(frozen=True)
class CompositionDiagnostic:
    """A machine-readable non-fatal issue encountered while projecting composition."""

    code: str
    message: str
    subject: str | None = None
    total: Fraction | None = None
    width: Fraction | None = None


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
    """One site-disorder assembly, without silently normalizing its probabilities."""

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

    @property
    def normalized(self) -> bool:
        return _normalization_diagnostic(
            "assembly probabilities", self.group_probabilities, self.group_probabilities_precision or ()
        )[0]

    @property
    def normalization_status(self) -> str:
        return _normalization_diagnostic(
            "assembly probabilities", self.group_probabilities, self.group_probabilities_precision or ()
        )[1]

    @property
    def normalization_diagnostic(self) -> CompositionDiagnostic | None:
        return _normalization_diagnostic(
            "assembly probabilities", self.group_probabilities, self.group_probabilities_precision or ()
        )[2]


@dataclass(frozen=True, init=False)
class ChemicalComposition:
    """Explicit elemental amounts, either additional (``implicit``) or authoritative (``full``)."""

    amounts: tuple[tuple[str, Fraction], ...]
    amounts_precision: tuple[tuple[str, Fraction | None], ...]
    mode: Literal["implicit", "full"]

    def __init__(
        self,
        amounts: Mapping[str, Any] | Iterable[tuple[str, Any]],
        mode: Literal["implicit", "full"] = "implicit",
        amounts_precision: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        *,
        precision: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        if precision is not None:
            if amounts_precision is not None:
                raise TypeError("pass either amounts_precision or precision, not both")
            amounts_precision = precision
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

    @classmethod
    def from_mapping(cls, amounts: Mapping[str, Any], **kwargs: Any) -> "ChemicalComposition":
        return cls(amounts, **kwargs)

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(element for element, _ in self.amounts)

    @property
    def amount_mapping(self) -> Mapping[str, Fraction]:
        return MappingProxyType(dict(self.amounts))

    @property
    def precision_mapping(self) -> Mapping[str, Fraction | None]:
        return MappingProxyType(dict(self.amounts_precision))


def validate_assemblies(assemblies: Iterable[Assembly], nsites: int | None = None) -> tuple[Assembly, ...]:
    """Validate global assembly site ownership for a future Structure constructor."""
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


def anonymous_symbol(index: int) -> str:
    """The unbounded OPTIMADE anonymous symbol for zero-based *index*."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("anonymous symbol index must be a non-negative integer")
    head = chr(ord("A") + index % 26)
    tail_number = index // 26
    tail: list[str] = []
    while tail_number:
        tail_number -= 1
        tail.append(chr(ord("a") + tail_number % 26))
        tail_number //= 26
    return head + "".join(reversed(tail))


@dataclass(frozen=True)
class CompositionResult:
    """The immutable projected elemental composition and formula diagnostics."""

    amounts: tuple[tuple[str, Fraction], ...]
    uncertainties: tuple[tuple[str, Fraction | None], ...]
    complete: bool
    exact: bool
    normalized: bool
    normalization_status: str
    diagnostics: tuple[CompositionDiagnostic, ...] = ()

    @property
    def elemental_amounts(self) -> tuple[tuple[str, Fraction], ...]:
        return self.amounts

    @property
    def amount_mapping(self) -> Mapping[str, Fraction]:
        return MappingProxyType(dict(self.amounts))

    @property
    def uncertainty_mapping(self) -> Mapping[str, Fraction | None]:
        return MappingProxyType(dict(self.uncertainties))

    @property
    def normalization_state(self) -> bool:
        return self.normalized

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(element for element, _ in self.amounts)

    @property
    def nelements(self) -> int:
        return len(self.amounts)

    @property
    def elements_ratios(self) -> tuple[Fraction, ...]:
        total = sum((amount for _, amount in self.amounts), Fraction())
        return () if not total else tuple(amount / total for _, amount in self.amounts)

    def _formula_coefficients(self) -> tuple[tuple[str, int], ...] | None:
        if not self.amounts:
            return None
        ratios = self.elements_ratios
        widths = dict(self.uncertainties)
        total = sum((amount for _, amount in self.amounts), Fraction())
        if self.exact:
            central = _integer_ratio(ratios)
            return (
                None
                if central is None
                else tuple((element, amount) for (element, _), amount in zip(self.amounts, central))
            )
        central = _integer_ratio(ratios)
        if central is not None and sum(central) <= 1000:
            return tuple((element, amount) for (element, _), amount in zip(self.amounts, central))
        return _measured_ratio(self.amounts, widths, total)

    @property
    def chemical_formula_reduced(self) -> str | None:
        coefficients = self._formula_coefficients()
        if coefficients is None:
            return None
        return "".join(element + (str(amount) if amount != 1 else "") for element, amount in coefficients)

    @property
    def chemical_formula_anonymous(self) -> str | None:
        coefficients = self._formula_coefficients()
        if coefficients is None:
            return None
        ordered = sorted(coefficients, key=lambda item: (-item[1], item[0]))
        return "".join(
            anonymous_symbol(index) + (str(amount) if amount != 1 else "") for index, (_, amount) in enumerate(ordered)
        )


def _integer_ratio(ratios: Sequence[Fraction]) -> tuple[int, ...] | None:
    if not ratios:
        return None
    denominator = 1
    for ratio in ratios:
        denominator = denominator * ratio.denominator // gcd(denominator, ratio.denominator)
    values = tuple(int(ratio * denominator) for ratio in ratios)
    common = reduce(gcd, values)
    return tuple(value // common for value in values)


def _ratio_intervals(
    amounts: Sequence[tuple[str, Fraction]], widths: Mapping[str, Fraction | None], total: Fraction
) -> tuple[tuple[Fraction, Fraction], ...]:
    total_width = sum((width or 0 for width in widths.values()), Fraction())
    lower_total = total - total_width
    intervals: list[tuple[Fraction, Fraction]] = []
    for element, amount in amounts:
        width = widths[element] or 0
        low = max(Fraction(), (amount - width) / (total + total_width))
        high = Fraction(1) if lower_total <= 0 else (amount + width) / lower_total
        intervals.append((low, min(Fraction(1), high)))
    return tuple(intervals)


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _measured_ratio(
    amounts: Sequence[tuple[str, Fraction]], widths: Mapping[str, Fraction | None], total: Fraction
) -> tuple[tuple[str, int], ...] | None:
    """Find the unique nearest bounded integer ratio consistent with all intervals."""
    intervals = _ratio_intervals(amounts, widths, total)
    centres = tuple(amount / total for _, amount in amounts)
    best: tuple[int, ...] | None = None
    best_score: Fraction | None = None
    ambiguous = False
    examined = 0
    max_candidates = 250_000
    count = len(amounts)

    def visit(index: int, remaining: int, values: list[int]) -> None:
        nonlocal best, best_score, ambiguous, examined
        if ambiguous and examined >= max_candidates:
            return
        if index == count - 1:
            candidate = remaining
            low, high = intervals[index]
            if candidate < 1 or not low <= Fraction(candidate, sum(values) + candidate) <= high:
                return
            vector = tuple(values + [candidate])
            common = reduce(gcd, vector)
            if common != 1:
                return
            examined += 1
            if examined > max_candidates:
                ambiguous = True
                return
            denominator = sum(vector)
            score = sum((Fraction(value, denominator) - centre) ** 2 for value, centre in zip(vector, centres))
            if best_score is None or score < best_score:
                best, best_score, ambiguous = vector, Fraction(score), False
            elif score == best_score and vector != best:
                ambiguous = True
            return
        denominator = sum(values) + remaining
        low, high = intervals[index]
        minimum = max(1, _ceil(low * denominator))
        maximum = min(
            remaining - (count - index - 1), (high * denominator).numerator // (high * denominator).denominator
        )
        for candidate in range(minimum, maximum + 1):
            visit(index + 1, remaining - candidate, values + [candidate])

    for denominator in range(count, 1001):
        visit(0, denominator, [])
        if examined > max_candidates:
            return None
    if best is None or ambiguous:
        return None
    return tuple((element, coefficient) for (element, _), coefficient in zip(amounts, best))


def _site_data(structure: Any) -> tuple[tuple[str, ...], tuple[Fraction, ...], tuple[Species, ...]]:
    species = tuple(structure.species)
    if hasattr(structure, "asu_sites") and hasattr(structure, "multiplicities"):
        names = tuple(site.species for site in structure.asu_sites)
        factors = tuple(Fraction(value) for value in structure.multiplicities())
    else:
        names = tuple(structure.species_at_sites)
        factors = tuple(Fraction(1) for _ in names)
    return names, factors, species


def derive_structure_features(structure: Any) -> tuple[str, ...]:
    """The exact-composition structure features, alphabetically ordered."""
    names, _, species = _site_data(structure)
    by_name = {value.name: value for value in species}
    used = tuple(by_name[name] for name in names if name in by_name)
    features: set[str] = set()
    if getattr(structure, "assemblies", None):
        features.add("assemblies")
    if any(len(value.chemical_symbols) > 1 or value.concentration != (Fraction(1),) for value in used):
        features.add("disorder")
    if any(value.attached for value in used):
        features.add("site_attachments")
    chemical = getattr(structure, "chemical_composition", None)
    if isinstance(chemical, ChemicalComposition) and chemical.mode == "implicit":
        features.add("implicit_atoms")
    return tuple(sorted(features))


def project_composition(structure: Any) -> CompositionResult:
    """Project a Structure or ASUStructure to exact elemental amounts without normalization."""
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
    provisional = CompositionResult(ordered, uncertainty, complete, exact, normalized, status, tuple(diagnostics))
    if ordered and not exact and provisional._formula_coefficients() is None:
        diagnostics.append(
            CompositionDiagnostic(
                "formula_ratio_unreconstructable",
                "no unique bounded integer ratio is consistent with the stated composition intervals",
            )
        )
    return CompositionResult(
        ordered,
        uncertainty,
        complete,
        exact,
        normalized,
        status,
        tuple(diagnostics),
    )
