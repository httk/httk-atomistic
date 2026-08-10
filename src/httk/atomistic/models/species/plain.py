"""
Backend wrapping a validated OPTIMADE species dict.
"""

from fractions import Fraction
from typing import Any, Self

from httk.atomistic._composition_values import as_fraction, as_precision
from httk.atomistic.models.species.backend import SpeciesBackend


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
    return isinstance(obj["concentration"], (list, tuple))


class PlainSpecies(SpeciesBackend):
    r"""
    Backend for a species backed by an OPTIMADE species dict.

    The native representation is a mapping with the OPTIMADE ``species`` fields; the
    required ``name``/``chemical_symbols``/``concentration`` are validated conservatively
    on construction. The accessors read the corresponding fields (optional fields absent
    from the dict read as ``None``), and ``unwrap`` returns the original dict.

    :param obj: The species mapping.
    :param \**hints: Backend-selection hints.
    """

    _raw: dict[str, Any]

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt an OPTIMADE species mapping.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "plain") != "plain":
            return None
        if not _is_optimade_species_dict(obj):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: dict[str, Any], **hints: Any) -> None:
        self._raw = obj

    @property
    def name(self) -> str:
        """Return the species name.

        :return: The species name.
        """
        return self._raw["name"]

    @property
    def chemical_symbols(self) -> tuple[str, ...]:
        """Return the constituent symbols.

        :return: The chemical symbols in constituent order.
        """
        return tuple(self._raw["chemical_symbols"])

    @property
    def concentration(self) -> tuple[Fraction, ...]:
        """Return the constituent concentrations.

        :return: The concentrations in constituent order.
        """
        return tuple(as_fraction(c, field="Species concentration")[0] for c in self._raw["concentration"])

    @property
    def concentration_precision(self) -> tuple[Fraction | None, ...] | None:
        """Return the concentration precision metadata.

        :return: Per-constituent precision, or ``None`` when unavailable.
        """
        raw = self._raw.get("_httk_concentration_precision")
        if raw is None:
            return tuple(as_fraction(c, field="Species concentration")[1] for c in self._raw["concentration"])
        return tuple(as_precision(value, field="Species concentration precision") for value in raw)

    @property
    def mass(self) -> tuple[float, ...] | None:
        """Return the constituent masses, if stated.

        :return: The masses, or ``None`` when unstated.
        """
        mass = self._raw.get("mass")
        return None if mass is None else tuple(float(m) for m in mass)

    @property
    def attached(self) -> tuple[str, ...] | None:
        """Return the attached constituent symbols, if stated.

        :return: The attached symbols, or ``None`` when unstated.
        """
        attached = self._raw.get("attached")
        return None if attached is None else tuple(attached)

    @property
    def nattached(self) -> tuple[int, ...] | None:
        """Return the attached counts, if stated.

        :return: The attached counts, or ``None`` when unstated.
        """
        nattached = self._raw.get("nattached")
        return None if nattached is None else tuple(int(n) for n in nattached)

    @property
    def original_name(self) -> str | None:
        """Return the original source name, if stated.

        :return: The original name, or ``None`` when unstated.
        """
        return self._raw.get("original_name")

    @property
    def charges(self) -> tuple[Fraction | None, ...] | None:
        """Return the constituent charges, if stated.

        :return: The charges, or ``None`` when unstated.
        """
        raw = self._raw.get("_httk_charges")
        if raw is None:
            return None
        values = tuple(None if value is None else Fraction(str(value)) for value in raw)
        return None if len(values) == len(self.chemical_symbols) and all(value is None for value in values) else values

    @property
    def spins(self) -> tuple[Fraction | None, ...] | None:
        """Return the constituent spins, if stated.

        :return: The spins, or ``None`` when unstated.
        """
        raw = self._raw.get("_httk_spins")
        if raw is None:
            return None
        values = tuple(None if value is None else Fraction(str(value)) for value in raw)
        return None if len(values) == len(self.chemical_symbols) and all(value is None for value in values) else values

    @property
    def labels(self) -> tuple[str | None, ...] | None:
        """Return the constituent labels, if stated.

        :return: The labels, or ``None`` when unstated.
        """
        raw = self._raw.get("_httk_labels")
        if raw is None:
            return None
        values = tuple(raw)
        return None if len(values) == len(self.chemical_symbols) and all(value is None for value in values) else values

    def unwrap(self) -> Any:
        """Return the original species mapping.

        :return: The raw mapping.
        """
        return self._raw
