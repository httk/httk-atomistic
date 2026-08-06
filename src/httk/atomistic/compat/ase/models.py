"""ASE interoperability for the :mod:`httk.atomistic` structure family.

The protocol deliberately describes only the four ASE ``Atoms`` methods needed for
conversion. ASE is optional: duck-typed objects can be converted without installing
ASE, while :class:`~httk.atomistic.compat.ase.view.ASEAtomsView` is available only when
ASE itself is installed.
"""

from typing import Any, Protocol, runtime_checkable

from httk.atomistic.elements import symbol_of
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.view import StructureView


@runtime_checkable
class ASEAtomsProtocol(Protocol):
    """The minimal method surface needed to treat an object as ASE ``Atoms``.

    This is a runtime-checkable, duck-typed protocol. ASE is not required: any object
    providing these four methods qualifies for :class:`ASEAtoms`.
    """

    def get_cell(self) -> Any:
        """Return the cell vectors as rows."""

    def get_scaled_positions(self) -> Any:
        """Return the reduced positions."""

    def get_atomic_numbers(self) -> Any:
        """Return one atomic number per site."""

    def get_pbc(self) -> Any:
        """Return one periodicity flag per cell row."""


def _float_rows(values: Any) -> list[list[float]]:
    """Render an iterable of vector rows without importing a numeric dependency."""
    return [[float(value) for value in row] for row in values]


class ASEAtoms(StructureBackend):
    """Backend for ASE ``Atoms`` and compatible duck-typed objects.

    Conversion is eager because reading the four methods and normalizing their values is
    real work. The original object remains available through :meth:`unwrap`.
    """

    _raw: Any
    _structure: UnitcellStructure

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints.get("kind", "ase") != "ase":
            return None
        # ASEAtomsView is itself an Atoms subclass and therefore satisfies the protocol;
        # existing structure-family objects must remain represented by their own backend.
        if isinstance(obj, (UnitcellStructure, StructureBackend, StructureView)):
            return None
        if not isinstance(obj, ASEAtomsProtocol):
            return None
        return super().__new__(cls)

    def __init__(self, obj: ASEAtomsProtocol, **hints: Any) -> None:
        symbols = tuple(symbol_of(int(number)) for number in obj.get_atomic_numbers())
        distinct_symbols = tuple(dict.fromkeys(symbols))
        self._raw = obj
        self._structure = UnitcellStructure(
            Cell(
                _float_rows(obj.get_cell()),
                periodicity=tuple(bool(flag) for flag in obj.get_pbc()),
            ),
            Sites(_float_rows(obj.get_scaled_positions())),
            tuple(Species.create(symbol) for symbol in distinct_symbols),
            symbols,
        )

    @property
    def cell(self) -> Cell:
        """The exact cell converted from the native cell rows."""
        return self._structure.cell

    @property
    def sites(self) -> Sites:
        """The exact reduced coordinates converted from the native positions."""
        return self._structure.sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct single-element species in first-appearance order."""
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site."""
        return self._structure.species_at_sites

    def unwrap(self) -> Any:
        """Return the original Atoms-like object."""
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
