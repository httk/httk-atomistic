"""Backend for numeric unitcell structure presentations."""

from typing import Any

from .cell import Cell
from .numeric_cell import NumericCell
from .numeric_sites import NumericSites
from .sites import Sites
from .species import Species
from .structure import Structure
from .structure_backend import StructureBackend


class NumericUnitcellStructureBackend(StructureBackend):
    """Backend recognizing the plain-numpy structure quartet.

    The wrapped object must expose ``NumericCell``/``NumericSites`` values through
    ``cell``/``sites`` and the usual ``species``/``species_at_sites`` attributes.
    Its exact quartet is taken from an ``exact`` :class:`~httk.atomistic.structure.Structure` when available,
    then from the component presentations' ``exact`` values. If neither component
    carries an exact value, the numeric ``basis`` and ``reduced_coords`` arrays are
    passed through the vector family into ``Cell`` and ``Sites``. That last route is
    exact embedding of the floats: their binary float64 values, rather than their
    decimal spellings, become the exact values.
    """

    _object: Any
    _structure: Structure

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints.get("kind", "numeric") != "numeric":
            return None
        try:
            cell = obj.cell
            sites = obj.sites
            obj.species  # noqa: B018 - deliberate attribute access validates the required field
            obj.species_at_sites  # noqa: B018 - deliberate attribute access validates the required field
        except AttributeError:
            return None
        if not isinstance(cell, NumericCell) or not isinstance(sites, NumericSites):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._object = obj
        exact = getattr(obj, "exact", None)
        if isinstance(exact, Structure):
            self._structure = exact
            return

        cell = getattr(obj.cell, "exact", None)
        if not isinstance(cell, Cell):
            cell = Cell(
                obj.cell.basis,
                precision=getattr(obj.cell, "precision", None),
                periodicity=getattr(obj.cell, "periodicity", None),
            )

        sites = getattr(obj.sites, "exact", None)
        if not isinstance(sites, Sites):
            sites = Sites(
                obj.sites.reduced_coords,
                precision=getattr(obj.sites, "precision", None),
            )

        self._structure = Structure(cell, sites, obj.species, obj.species_at_sites)

    @property
    def cell(self) -> Cell:
        """The exact cell quartet component."""
        return self._structure.cell

    @property
    def sites(self) -> Sites:
        """The exact sites quartet component."""
        return self._structure.sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The exact structure's distinct species."""
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The exact species name for each site."""
        return self._structure.species_at_sites

    def unwrap(self) -> Any:
        """Return the wrapped numeric presentation object."""
        return self._object
