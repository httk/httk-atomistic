"""
Species definition for httk-atomistic, mirroring the OPTIMADE ``species`` entry.
"""

import decimal
import fractions
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from httk.atomistic._composition_values import as_fraction, as_precision, normalization
from httk.atomistic.elements import SYMBOLS, symbol_of
from httk.atomistic.models.species.backend import SpeciesBackend

_ELEMENTS: frozenset[str] = frozenset(SYMBOLS)
_SPECIAL_SYMBOLS: frozenset[str] = frozenset({"X", "vacancy"})

type ExactInput = fractions.Fraction | int | float | decimal.Decimal | str
type PrecisionInput = ExactInput | None
type DecorationInput = fractions.Fraction | int | str | None


def _normalize_decoration(
    values: Sequence[DecorationInput] | None,
) -> tuple[fractions.Fraction | None, ...] | None:
    if values is None:
        return None
    return tuple(None if value is None else fractions.Fraction(value) for value in values)


@dataclass(frozen=True, eq=False, init=False)
class Species(SpeciesBackend):
    r"""
    A chemical species occupying one or more sites, mirroring the OPTIMADE ``species`` object.

    A species has a ``name`` (unique within a structure; it need not be a chemical
    symbol), a list of ``chemical_symbols`` composing it, and a matching list of
    non-negative ``concentration`` values. Concentrations on represented sites must
    lie in ``[0, 1]``; an unrepresented species may instead carry an aggregate implicit-atom
    count. Each chemical symbol is an element symbol, or one of
    the pseudo-symbols ``"X"`` (unknown) or ``"vacancy"``. The optional ``mass``,
    ``attached``, ``nattached``, and ``original_name`` fields carry the remaining
    OPTIMADE species information; ``attached`` and ``nattached`` must be given
    together and share their length.

    ``charges``, ``spins``, and ``labels`` are optional aligned decorations. An
    all-``None`` decoration is canonicalized to ``None``. Repeated chemical
    symbols are accepted only when the complete decoration distinguishes them.

    :param name: The species name.
    :param chemical_symbols: The constituent chemical symbols.
    :param concentration: The constituent occupancies.
    :param mass: The constituent masses, if stated.
    :param original_name: The source name, if stated.
    :param attached: The attached constituent symbols, if stated.
    :param nattached: The counts corresponding to ``attached``, if stated.
    :param concentration_precision: The precision of each occupancy, if stated.
    :param charges: The charge decoration, if stated.
    :param spins: The spin decoration, if stated.
    :param labels: The label decoration, if stated.
    """

    name: str = ""  # pyright: ignore[reportIncompatibleMethodOverride]
    chemical_symbols: tuple[str, ...] = ()  # pyright: ignore[reportIncompatibleMethodOverride]
    concentration: tuple[fractions.Fraction, ...] = ()  # pyright: ignore[reportIncompatibleMethodOverride]
    mass: tuple[float, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    original_name: str | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    attached: tuple[str, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    nattached: tuple[int, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    concentration_precision: tuple[fractions.Fraction | None, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    charges: tuple[fractions.Fraction | None, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    spins: tuple[fractions.Fraction | None, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    labels: tuple[str | None, ...] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(
        self,
        name: str,
        chemical_symbols: Sequence[str],
        concentration: Sequence[ExactInput],
        mass: Sequence[float | int] | None = None,
        original_name: str | None = None,
        attached: Sequence[str] | None = None,
        nattached: Sequence[int] | None = None,
        concentration_precision: Sequence[PrecisionInput] | None = None,
        charges: Sequence[DecorationInput] | None = None,
        spins: Sequence[DecorationInput] | None = None,
        labels: Sequence[str | None] | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("Species name must be a string")
        if original_name is not None and not isinstance(original_name, str):
            raise TypeError("Species original_name must be a string or None")
        if mass is not None and any(isinstance(value, bool) for value in mass):
            raise ValueError("Species mass values cannot be bool values")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "chemical_symbols", tuple(chemical_symbols))
        object.__setattr__(self, "concentration", tuple(concentration))
        object.__setattr__(self, "mass", None if mass is None else tuple(mass))
        object.__setattr__(self, "original_name", original_name)
        object.__setattr__(self, "attached", None if attached is None else tuple(attached))
        object.__setattr__(self, "nattached", None if nattached is None else tuple(nattached))
        object.__setattr__(
            self,
            "concentration_precision",
            None if concentration_precision is None else tuple(concentration_precision),
        )
        object.__setattr__(self, "charges", _normalize_decoration(charges))
        object.__setattr__(self, "spins", _normalize_decoration(spins))
        normalized_labels = None if labels is None else tuple(labels)
        object.__setattr__(self, "labels", normalized_labels)
        self.__post_init__()

    def __post_init__(self) -> None:
        symbols = tuple(self.chemical_symbols)
        if not symbols:
            raise ValueError("Species chemical_symbols must be non-empty")
        object.__setattr__(self, "chemical_symbols", symbols)
        concentration: list[fractions.Fraction] = []
        inferred_precision: list[fractions.Fraction | None] = []
        for value in self.concentration:
            central, width = as_fraction(value, field="Species concentration")
            if central < 0:
                raise ValueError("Species concentration values must be non-negative")
            concentration.append(central)
            inferred_precision.append(width)
        object.__setattr__(self, "concentration", tuple(concentration))
        if self.mass is not None:
            object.__setattr__(self, "mass", tuple(float(m) for m in self.mass))
        if self.attached is not None:
            object.__setattr__(self, "attached", tuple(self.attached))
        if self.nattached is not None:
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in self.nattached):
                raise ValueError("Species nattached values must be non-negative integers")
            object.__setattr__(self, "nattached", tuple(self.nattached))

        if len(self.concentration) != len(self.chemical_symbols):
            raise ValueError("Species concentration must have the same length as chemical_symbols")
        for symbol in self.chemical_symbols:
            if symbol not in _ELEMENTS and symbol not in _SPECIAL_SYMBOLS:
                raise ValueError(f"Species chemical symbol is not an element, 'X', or 'vacancy': {symbol!r}")
        if self.mass is not None and len(self.mass) != len(self.chemical_symbols):
            raise ValueError("Species mass must have the same length as chemical_symbols")
        if self.mass is not None:
            if any(not math.isfinite(mass) or mass < 0 for mass in self.mass):
                raise ValueError("Species mass values must be finite and non-negative")
            if any(symbol == "vacancy" and mass != 0.0 for symbol, mass in zip(self.chemical_symbols, self.mass)):
                raise ValueError("Species vacancy mass must be exactly zero")
        if (self.attached is None) != (self.nattached is None):
            raise ValueError("Species attached and nattached must be given together or not at all")
        if self.attached is not None and self.nattached is not None and len(self.attached) != len(self.nattached):
            raise ValueError("Species attached and nattached must have the same length")
        if self.attached is not None:
            if not self.attached:
                raise ValueError("Species attached cannot be empty when present")
            if any(symbol not in _ELEMENTS and symbol != "X" for symbol in self.attached):
                raise ValueError("Species attached symbols must be elements or 'X'")
        for field_name in ("charges", "spins", "labels"):
            values = getattr(self, field_name)
            if values is not None and len(values) != len(symbols):
                raise ValueError(f"Species {field_name} must have the same length as chemical_symbols")
            if values is not None and all(value is None for value in values):
                object.__setattr__(self, field_name, None)
        if self.labels is not None:
            if any(value is not None and not isinstance(value, str) for value in self.labels):
                raise ValueError("Species labels values must be strings or None")
            object.__setattr__(self, "labels", tuple(self.labels))
        decorations = tuple(
            (
                symbol,
                None if self.mass is None else self.mass[index],
                None if self.charges is None else self.charges[index],
                None if self.spins is None else self.spins[index],
                None if self.labels is None else self.labels[index],
            )
            for index, symbol in enumerate(symbols)
        )
        if len(decorations) != len(set(decorations)):
            raise ValueError(
                "Species chemical_symbols must be unique unless decorations "
                "(mass/charge/spin/label) distinguish repeated symbols; exact duplicate "
                "constituents are not allowed"
            )
        stated_precision = self.concentration_precision
        if stated_precision is None:
            precision = tuple(inferred_precision)
        else:
            if len(stated_precision) != len(self.concentration):
                raise ValueError("Species concentration_precision must have the same length as concentration")
            precision = tuple(
                as_precision(value, field="Species concentration precision") for value in stated_precision
            )
        object.__setattr__(self, "concentration_precision", precision)

    def __repr__(self) -> str:
        parts = [
            f"name={self.name!r}",
            f"chemical_symbols={self.chemical_symbols!r}",
            f"concentration={self.concentration!r}",
        ]
        for field in (
            "mass",
            "original_name",
            "attached",
            "nattached",
            "concentration_precision",
            "charges",
            "spins",
            "labels",
        ):
            value = getattr(self, field)
            if value is not None:
                parts.append(f"{field}={value!r}")
        return f"Species({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """Compare species values across the Species subclass/view family.

        :param other: The object to compare with.
        :return: Whether all species fields match.
        """
        if not isinstance(other, Species):
            return NotImplemented
        return (
            self.name,
            self.chemical_symbols,
            self.concentration,
            self.mass,
            self.original_name,
            self.attached,
            self.nattached,
            self.concentration_precision,
            self.charges,
            self.spins,
            self.labels,
        ) == (
            other.name,
            other.chemical_symbols,
            other.concentration,
            other.mass,
            other.original_name,
            other.attached,
            other.nattached,
            other.concentration_precision,
            other.charges,
            other.spins,
            other.labels,
        )

    def __hash__(self) -> int:
        """Hash the same value fields used by :meth:`__eq__`.

        :return: The species hash.
        """
        return hash(
            (
                self.name,
                self.chemical_symbols,
                self.concentration,
                self.mass,
                self.original_name,
                self.attached,
                self.nattached,
                self.concentration_precision,
                self.charges,
                self.spins,
                self.labels,
            )
        )

    @property
    def normalized(self) -> bool:
        """Whether the stated concentration interval contains one.

        :return: Whether the concentrations are normalized within their precision.
        """
        return normalization(self.concentration, self.concentration_precision or ())[0]

    @property
    def normalization_status(self) -> str:
        """Report the concentration normalization status.

        :return: ``exact``, ``within_precision``, or ``outside_precision``.
        """
        return normalization(self.concentration, self.concentration_precision or ())[1]

    @property
    def normalization_diagnostic(self) -> Any:
        """Return a structured normalization diagnostic when needed.

        :return: The diagnostic, or ``None`` when the concentrations are normalized.
        """
        if self.normalized:
            return None
        from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic

        _, _, total, width = normalization(self.concentration, self.concentration_precision or ())
        return CompositionDiagnostic(
            "normalization_outside_precision",
            f"species {self.name!r} concentrations sum to {total}, outside their stated interval around 1",
            self.name,
            total,
            width,
        )

    @property
    def is_single_element(self) -> bool:
        """
        Whether this species is a single, unattached, real chemical element.

        True only for a species composed of exactly one element symbol (not ``"X"``
        or ``"vacancy"``) with no attached particles. Such species are the ones that
        can be represented as a bare atomic number in the primitive representation.

        :return: Whether this is a single real element.
        """
        return (
            len(self.chemical_symbols) == 1
            and self.chemical_symbols[0] in _ELEMENTS
            and self.concentration == (fractions.Fraction(1),)
            and self.attached is None
        )

    def without_charges(self) -> "Species":
        """Return an EXPLICIT lossy projection that drops declared oxidation states.

        The other species fields, including spins and labels, are preserved. A species
        without declared charges is returned by identity.

        :return: A charge-free species, or this species when already charge-free.
        """
        if self.charges is None:
            return self
        return Species(
            name=self.name,
            chemical_symbols=self.chemical_symbols,
            concentration=self.concentration,
            mass=self.mass,
            original_name=self.original_name,
            attached=self.attached,
            nattached=self.nattached,
            concentration_precision=self.concentration_precision,
            spins=self.spins,
            labels=self.labels,
        )

    @classmethod
    def from_object(cls, obj: "Species | dict[str, Any] | str | int", **hints: Any) -> "Species":
        r"""
        Return a Species from an existing Species, bare symbol or atomic number, or
        OPTIMADE species dict.

        A bare element symbol, ``"X"``, or ``"vacancy"`` denotes a fully occupied
        single-symbol species. A bare atomic number denotes the corresponding element.

        :param obj: An existing species, symbol, atomic number, or species mapping.
        :param \**hints: Backend-selection hints.
        :return: The canonical species.
        :raises ValueError: If an atomic number is boolean or the input is invalid.
        """
        if isinstance(obj, Species):
            return obj
        if isinstance(obj, bool):
            raise ValueError(f"Species atomic number cannot be a bool: {obj!r}")
        if isinstance(obj, int):
            obj = symbol_of(obj)
        if isinstance(obj, str):
            return cls(name=obj, chemical_symbols=(obj,), concentration=(1.0,))
        attached = obj.get("attached")
        nattached = obj.get("nattached")
        mass = obj.get("mass")

        def parse_decoration(key: str) -> tuple[fractions.Fraction | None, ...] | None:
            raw = obj.get(key)
            if raw is None:
                return None
            return tuple(None if value is None else fractions.Fraction(str(value)) for value in raw)

        return cls(
            name=obj["name"],
            chemical_symbols=tuple(obj["chemical_symbols"]),
            concentration=tuple(obj["concentration"]),
            mass=None if mass is None else tuple(mass),
            original_name=obj.get("original_name"),
            attached=None if attached is None else tuple(attached),
            nattached=None if nattached is None else tuple(nattached),
            concentration_precision=(
                None
                if obj.get("_httk_concentration_precision") is None
                else tuple(obj["_httk_concentration_precision"])
            ),
            charges=parse_decoration("_httk_charges"),
            spins=parse_decoration("_httk_spins"),
            labels=None if obj.get("_httk_labels") is None else tuple(obj["_httk_labels"]),
        )
