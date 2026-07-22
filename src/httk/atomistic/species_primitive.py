"""
Backend wrapping a validated OPTIMADE species dict.
"""

from typing import Any

from .species_backend import SpeciesBackend


def _is_optimade_species_dict(obj: Any) -> bool:
    """
    Conservatively check that ``obj`` is an OPTIMADE-shaped species dict.

    Only the required keys are checked, and only roughly (present and of a plausible
    type). Full validation happens when the species is converted to a ``Species``.
    """
    if not isinstance(obj, dict):
        return False
    if "name" not in obj or "chemical_symbols" not in obj or "concentration" not in obj:
        return False
    if not isinstance(obj["name"], str):
        return False
    if not isinstance(obj["chemical_symbols"], (list, tuple)):
        return False
    if not isinstance(obj["concentration"], (list, tuple)):
        return False
    return True


class SpeciesPrimitive(SpeciesBackend):
    """
    Backend for a species backed by an OPTIMADE species dict.

    The native representation is a mapping with the OPTIMADE ``species`` fields; the
    required ``name``/``chemical_symbols``/``concentration`` are validated conservatively
    on construction. The accessors read the corresponding fields (optional fields absent
    from the dict read as ``None``), and ``unwrap`` returns the original dict.
    """

    _raw: dict[str, Any]

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "primitive") != "primitive":
            return None
        if not _is_optimade_species_dict(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: dict[str, Any], **hints: Any) -> None:
        self._raw = obj

    @property
    def name(self) -> str:
        return self._raw["name"]

    @property
    def chemical_symbols(self) -> tuple[str, ...]:
        return tuple(self._raw["chemical_symbols"])

    @property
    def concentration(self) -> tuple[float, ...]:
        return tuple(float(c) for c in self._raw["concentration"])

    @property
    def mass(self) -> tuple[float, ...] | None:
        mass = self._raw.get("mass")
        return None if mass is None else tuple(float(m) for m in mass)

    @property
    def attached(self) -> tuple[str, ...] | None:
        attached = self._raw.get("attached")
        return None if attached is None else tuple(attached)

    @property
    def nattached(self) -> tuple[int, ...] | None:
        nattached = self._raw.get("nattached")
        return None if nattached is None else tuple(int(n) for n in nattached)

    @property
    def original_name(self) -> str | None:
        return self._raw.get("original_name")

    def unwrap(self) -> Any:
        return self._raw
