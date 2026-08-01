"""
Backend wrapping a Species in the class representation.
"""

from fractions import Fraction
from typing import Any

from .species import Species
from .species_backend import SpeciesBackend


class SpeciesClass(SpeciesBackend):
    """
    Backend for a species backed by an actual ``Species`` object.

    Its accessors delegate to the wrapped Species, and ``unwrap`` returns that Species.
    """

    _species: Species

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if isinstance(obj, bool):
            raise ValueError(f"Species atomic number cannot be a bool: {obj!r}")
        if not isinstance(obj, (Species, str, int)):
            return None
        if hints and hints.get("kind", "class") != "class":
            return None
        return super().__new__(cls)

    def __init__(self, obj: Species | str | int, **hints: Any) -> None:
        self._species = Species.create(obj)

    @property
    def name(self) -> str:
        return self._species.name

    @property
    def chemical_symbols(self) -> tuple[str, ...]:
        return self._species.chemical_symbols

    @property
    def concentration(self) -> tuple[Fraction, ...]:
        return self._species.concentration

    @property
    def concentration_precision(self) -> tuple[Fraction | None, ...] | None:
        return self._species.concentration_precision

    @property
    def mass(self) -> tuple[float, ...] | None:
        return self._species.mass

    @property
    def attached(self) -> tuple[str, ...] | None:
        return self._species.attached

    @property
    def nattached(self) -> tuple[int, ...] | None:
        return self._species.nattached

    @property
    def original_name(self) -> str | None:
        return self._species.original_name

    def unwrap(self) -> Any:
        return self._species
