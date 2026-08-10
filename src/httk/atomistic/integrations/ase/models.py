"""ASE interoperability for the :mod:`httk.atomistic` structure family.

The protocol deliberately describes only the four ASE ``Atoms`` methods needed for
conversion. ASE is optional: duck-typed objects can be converted without installing
ASE, while :class:`~httk.atomistic.integrations.ase.view.ASEAtomsView` is available only when
ASE itself is installed.
"""

import fractions
from collections.abc import Iterable
from typing import Any, Protocol, Self, runtime_checkable

from httk.atomistic.elements import symbol_of
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure


@runtime_checkable
class ASEAtomsProtocol(Protocol):
    """Describe the minimal method surface needed to read ASE ``Atoms``.

    This is a runtime-checkable, duck-typed protocol. ASE is not required: any object
    providing these four methods qualifies for :class:`ASEAtoms`.
    """

    def get_cell(self) -> Any:
        """Return the cell vectors as rows.

        :return: The native cell rows.
        """

    def get_scaled_positions(self) -> Any:
        """Return the reduced positions.

        :return: One reduced coordinate row per site.
        """

    def get_atomic_numbers(self) -> Any:
        """Return one atomic number per site.

        :return: The atomic numbers.
        """

    def get_pbc(self) -> Any:
        """Return one periodicity flag per cell row.

        :return: The periodicity flags.
        """


def _float_rows(values: Any) -> list[list[float]]:
    """Render an iterable of vector rows without importing a numeric dependency."""
    return [[float(value) for value in row] for row in values]


def _values(value: Any) -> list[Any]:
    """Copy an ASE array-like result without importing numpy."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _exact_float(value: Any) -> fractions.Fraction:
    return fractions.Fraction(str(float(value)))


def _magnetic_moments(obj: Any) -> Any:
    if not hasattr(obj, "get_initial_magnetic_moments"):
        return None
    values = _values(obj.get_initial_magnetic_moments())
    if not values:
        return None
    if isinstance(values[0], Iterable) and not isinstance(values[0], (str, bytes)):
        rows = [list(row) for row in values]
        if all(float(item) == 0 for row in rows for item in row):
            return None
        return CartesianSiteMoments([[_exact_float(item) for item in row] for row in rows])
    if all(float(value) == 0 for value in values):
        return None
    return CollinearSiteMoments([_exact_float(value) for value in values])


def _charge_species(symbols: tuple[str, ...], obj: Any) -> tuple[tuple[Species, ...], tuple[str, ...]]:
    if not hasattr(obj, "get_initial_charges"):
        distinct_symbols = tuple(dict.fromkeys(symbols))
        return tuple(Species.create(symbol) for symbol in distinct_symbols), symbols
    values = _values(obj.get_initial_charges())
    if all(float(value) == 0 for value in values):
        distinct_symbols = tuple(dict.fromkeys(symbols))
        return tuple(Species.create(symbol) for symbol in distinct_symbols), symbols

    charges = tuple(fractions.Fraction(str(value)) for value in values)
    species_by_key: dict[tuple[str, fractions.Fraction], Species] = {}
    name_keys: dict[str, tuple[str, fractions.Fraction]] = {}
    species_values: list[Species] = []
    names: list[str] = []
    for symbol, charge in zip(symbols, charges, strict=True):
        key = (symbol, charge)
        species = species_by_key.get(key)
        if species is None:
            base = f"{symbol}{abs(charge)}{'+' if charge >= 0 else '-'}"
            name = base
            suffix = 2
            while name in name_keys and name_keys[name] != key:
                name = f"{base}_{suffix}"
                suffix += 1
            species = Species(name, (symbol,), (1,), charges=(charge,))
            species_by_key[key] = species
            name_keys[name] = key
            species_values.append(species)
        names.append(species.name)
    return tuple(species_values), tuple(names)


class ASEAtoms(StructureBackend):
    r"""Import ASE ``Atoms`` and compatible duck-typed objects.

    Conversion is eager because reading the four methods and normalizing their values is
    real work. The original object remains available through :meth:`unwrap`.

    Initial magnetic moments become site moments and nonzero initial charges become
    charged single-element species. All-zero ASE defaults remain unstated.

    :param obj: An ASE ``Atoms`` object or compatible duck-typed object.
    :param \**hints: Backend-selection hints.
    """

    _raw: Any
    _structure: UnitcellStructure

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt an ASE-compatible structure.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints.get("kind", "ase") != "ase":
            return None
        # Existing structure-family objects must remain represented by their own backend;
        # an ASEAtomsView is also an Atoms object and is intentionally round-trippable.
        if isinstance(obj, (UnitcellStructure, StructureBackend)):
            return None
        if not isinstance(obj, ASEAtomsProtocol):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: ASEAtomsProtocol, **hints: Any) -> None:
        symbols = tuple(symbol_of(int(number)) for number in obj.get_atomic_numbers())
        species, species_at_sites = _charge_species(symbols, obj)
        self._raw = obj
        self._structure = UnitcellStructure(
            Cell(
                _float_rows(obj.get_cell()),
                periodicity=tuple(bool(flag) for flag in obj.get_pbc()),
            ),
            Sites(_float_rows(obj.get_scaled_positions())),
            species,
            species_at_sites,
            site_moments=_magnetic_moments(obj),
        )

    @property
    def cell(self) -> Cell:
        """Return the exact cell converted from native cell rows."""
        return self._structure.cell

    @property
    def sites(self) -> Sites:
        """Return the exact reduced coordinates converted from native positions."""
        return self._structure.sites

    @property
    def species(self) -> tuple[Species, ...]:
        """Return distinct single-element species in first-appearance order."""
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the species name occupying each site."""
        return self._structure.species_at_sites

    @property
    def site_moments(self) -> Any:
        """Return per-site moments, or ``None`` for absent and all-zero ASE defaults."""
        return self._structure.site_moments

    def unwrap(self) -> Any:
        """Return the original ``Atoms``-like object."""
        return self._raw


try:
    from .view import ASEAtomsView  # noqa: F401
except ImportError:
    _ase_available = False
else:
    _ase_available = True


def __getattr__(name: str) -> Any:
    if name == "ASEAtomsView" and not _ase_available:
        raise ImportError("ASEAtomsView requires ASE; install the optional 'ase' package")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ASEAtoms", "ASEAtomsProtocol"]
if _ase_available:
    __all__.append("ASEAtomsView")
