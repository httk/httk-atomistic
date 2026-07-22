"""
Species definition for httk-atomistic, mirroring the OPTIMADE ``species`` entry.
"""

from dataclasses import dataclass
from typing import Any

from .elements import SYMBOLS

_ELEMENTS: frozenset[str] = frozenset(SYMBOLS)
_SPECIAL_SYMBOLS: frozenset[str] = frozenset({"X", "vacancy"})


@dataclass(frozen=True)
class Species:
    """
    A chemical species occupying one or more sites, mirroring the OPTIMADE ``species`` object.

    A species has a ``name`` (unique within a structure; it need not be a chemical
    symbol), a list of ``chemical_symbols`` composing it, and a matching list of
    ``concentration`` values. Each chemical symbol is an element symbol, or one of
    the pseudo-symbols ``"X"`` (unknown) or ``"vacancy"``. The optional ``mass``,
    ``attached``, ``nattached``, and ``original_name`` fields carry the remaining
    OPTIMADE species information; ``attached`` and ``nattached`` must be given
    together and share their length.
    """

    name: str
    chemical_symbols: tuple[str, ...]
    concentration: tuple[float, ...]
    mass: tuple[float, ...] | None = None
    original_name: str | None = None
    attached: tuple[str, ...] | None = None
    nattached: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chemical_symbols", tuple(self.chemical_symbols))
        object.__setattr__(self, "concentration", tuple(float(c) for c in self.concentration))
        if self.mass is not None:
            object.__setattr__(self, "mass", tuple(float(m) for m in self.mass))
        if self.attached is not None:
            object.__setattr__(self, "attached", tuple(self.attached))
        if self.nattached is not None:
            object.__setattr__(self, "nattached", tuple(int(n) for n in self.nattached))

        if len(self.concentration) != len(self.chemical_symbols):
            raise ValueError("Species concentration must have the same length as chemical_symbols")
        for symbol in self.chemical_symbols:
            if symbol not in _ELEMENTS and symbol not in _SPECIAL_SYMBOLS:
                raise ValueError(f"Species chemical symbol is not an element, 'X', or 'vacancy': {symbol!r}")
        if self.mass is not None and len(self.mass) != len(self.chemical_symbols):
            raise ValueError("Species mass must have the same length as chemical_symbols")
        if (self.attached is None) != (self.nattached is None):
            raise ValueError("Species attached and nattached must be given together or not at all")
        if self.attached is not None and self.nattached is not None and len(self.attached) != len(self.nattached):
            raise ValueError("Species attached and nattached must have the same length")

    @property
    def is_single_element(self) -> bool:
        """
        Whether this species is a single, unattached, real chemical element.

        True only for a species composed of exactly one element symbol (not ``"X"``
        or ``"vacancy"``) with no attached particles. Such species are the ones that
        can be represented as a bare atomic number in the primitive representation.
        """
        return len(self.chemical_symbols) == 1 and self.chemical_symbols[0] in _ELEMENTS and self.attached is None

    @classmethod
    def create(cls, obj: "Species | dict[str, Any]") -> "Species":
        """
        Return a Species from either an existing Species (returned unchanged) or an OPTIMADE species dict.
        """
        if isinstance(obj, Species):
            return obj
        attached = obj.get("attached")
        nattached = obj.get("nattached")
        mass = obj.get("mass")
        return cls(
            name=obj["name"],
            chemical_symbols=tuple(obj["chemical_symbols"]),
            concentration=tuple(obj["concentration"]),
            mass=None if mass is None else tuple(mass),
            original_name=obj.get("original_name"),
            attached=None if attached is None else tuple(attached),
            nattached=None if nattached is None else tuple(nattached),
        )
