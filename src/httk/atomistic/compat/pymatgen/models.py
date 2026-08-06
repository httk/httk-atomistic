"""Pymatgen interoperability without importing the optional dependency."""

import fractions
import logging
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.view import StructureView


@runtime_checkable
class PymatgenStructureProtocol(Protocol):
    """The three native attributes that identify a pymatgen structure.

    ``lattice`` supplies the cell and periodicity, ``frac_coords`` supplies the reduced
    coordinates, and ``species_and_occu`` supplies one per-site composition mapping. This
    small surface is disjoint from ASE ``Atoms`` and httk structure objects while allowing
    pymatgen-compatible duck-typed inputs without importing pymatgen.
    """

    lattice: Any
    frac_coords: Any
    species_and_occu: Any


def _to_fraction(value: Any) -> fractions.Fraction | None:
    if value is None:
        return None
    if isinstance(value, int):
        return fractions.Fraction(value)
    if isinstance(value, float):
        return fractions.Fraction(str(value))
    return fractions.Fraction(str(value))


def _is_dummy(value: Any) -> bool:
    return type(value).__name__ == "DummySpecies"


def _display_name(value: Any) -> str:
    symbol = str(value.symbol)
    if _is_dummy(value) and getattr(value, "oxi_state", None) == 0 and getattr(value, "spin", None) is None:
        return symbol
    return str(value)


def _constituent(
    value: Any, occupancy: Any
) -> tuple[str, fractions.Fraction, fractions.Fraction | None, fractions.Fraction | None, str | None]:
    symbol = str(value.symbol)
    if type(value).__name__ == "Element" or _is_dummy(value) and getattr(value, "oxi_state", None) == 0:
        charge = None
    else:
        charge = _to_fraction(getattr(value, "oxi_state", None))
    spin = _to_fraction(getattr(value, "spin", None)) if type(value).__name__ == "Species" else None
    label: str | None = None
    if _is_dummy(value):
        label = None if symbol == "X" else symbol.removeprefix("X")
        symbol = "X"
    return symbol, fractions.Fraction(str(occupancy)), charge, spin, label


def _moment_vector(value: Any) -> list[float] | None:
    if hasattr(value, "moment"):
        raw = value.global_moment if hasattr(value, "global_moment") else value.moment
        return [float(item) for item in raw]
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 3:
        return [float(item) for item in value]
    try:
        if not isinstance(value, (str, bytes)) and len(value) == 3:
            return [float(item) for item in value]
    except TypeError:
        pass
    return None


class PymatgenStructure(StructureBackend):
    """Eager pymatgen backend.

    Pymatgen ``properties``, site labels, and site properties other than ``magmom`` are
    intentionally discarded because they have no exact httk structure-family counterpart.
    Pymatgen ``DummySpecies`` values with the default zero oxidation state are imported
    with an unstated charge because pymatgen cannot distinguish that default from an
    explicitly supplied zero; nonzero dummy oxidation states remain exact charges.
    """

    kind: ClassVar[str] = "pymatgen"
    _raw: Any
    _structure: UnitcellStructure

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints.get("kind", "pymatgen") != "pymatgen":
            return None
        if isinstance(obj, (UnitcellStructure, StructureBackend, StructureView)):
            return None
        if not isinstance(obj, PymatgenStructureProtocol):
            return None
        return super().__new__(cls)

    def __init__(self, obj: PymatgenStructureProtocol, **hints: Any) -> None:
        species_by_key: dict[tuple[Any, ...], tuple[str, Species]] = {}
        names: list[str] = []
        species_values: list[Species] = []
        name_keys: dict[str, tuple[Any, ...]] = {}
        for composition in obj.species_and_occu:
            constituents = [_constituent(value, occupancy) for value, occupancy in composition.items()]
            total = sum((item[1] for item in constituents), fractions.Fraction(0))
            if total < 1:
                constituents.append(("vacancy", 1 - total, None, None, None))
            key = tuple(sorted(constituents, key=repr))
            existing = species_by_key.get(key)
            if existing is None:
                base = "".join(sorted(_display_name(value) for value in composition))
                name = base
                suffix = 2
                while name in name_keys and name_keys[name] != key:
                    name = f"{base}_{suffix}"
                    suffix += 1
                symbols = tuple(item[0] for item in key)
                concentrations = tuple(item[1] for item in key)
                charges = tuple(item[2] for item in key)
                spins = tuple(item[3] for item in key)
                labels = tuple(item[4] for item in key)
                species = Species(
                    name,
                    symbols,
                    concentrations,
                    charges=charges if any(value is not None for value in charges) else None,
                    spins=spins if any(value is not None for value in spins) else None,
                    labels=labels if any(value is not None for value in labels) else None,
                )
                existing = (name, species)
                species_by_key[key] = existing
                name_keys[name] = key
                species_values.append(species)
            names.append(existing[0])

        site_properties = getattr(obj, "site_properties", {})
        raw_moments = site_properties.get("magmom")
        site_moments: Any = None
        if raw_moments is not None:
            vector_values = [_moment_vector(value) for value in raw_moments]
            vector_mode = any(value is not None for value in vector_values)
            if any(value is None for value in raw_moments):
                logging.getLogger(__name__).warning(
                    "None pymatgen magnetic moment entries were converted to zero",
                    extra={"context": "pymatgen"},
                )
            if vector_mode:
                expanded = [
                    vector if vector is not None else [0.0, 0.0, 0.0] if value is None else [0.0, 0.0, float(value)]
                    for value, vector in zip(raw_moments, vector_values)
                ]
                site_moments = CartesianSiteMoments(expanded)
            else:
                site_moments = CollinearSiteMoments([0 if value is None else float(value) for value in raw_moments])

        charge = getattr(obj, "_charge", None)
        self._raw = obj
        self._structure = UnitcellStructure(
            Cell(obj.lattice.matrix, periodicity=tuple(bool(value) for value in obj.lattice.pbc)),
            Sites(obj.frac_coords),
            tuple(species_values),
            names,
            site_moments=site_moments,
            charge=None if charge is None else fractions.Fraction(str(charge)),
        )

    @property
    def cell(self) -> Cell:
        return self._structure.cell

    @property
    def sites(self) -> Sites:
        return self._structure.sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._structure.species_at_sites

    @property
    def site_moments(self) -> Any:
        return self._structure.site_moments

    @property
    def charge(self) -> fractions.Fraction | None:
        return self._structure.charge

    def unwrap(self) -> Any:
        return self._raw


try:
    from .view import PymatgenStructureView  # noqa: F401
except ImportError:
    _pymatgen_available = False
else:
    _pymatgen_available = True


def __getattr__(name: str) -> Any:
    if name == "PymatgenStructureView" and not _pymatgen_available:
        raise ImportError("PymatgenStructureView requires pymatgen; install the optional 'pymatgen' package")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PymatgenStructure", "PymatgenStructureProtocol"]
if _pymatgen_available:
    __all__.append("PymatgenStructureView")
