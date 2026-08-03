"""Backend for numeric unitcell structure presentations."""

from typing import Any

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.numeric import NumericCell
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell import UnitcellStructure


class NumericUnitcellStructure(StructureBackend):
    """Backend recognizing the plain-numpy structure quartet.

    The wrapped object must expose ``NumericCell``/``NumericSites`` values through
    ``cell``/``sites`` and the usual ``species``/``species_at_sites`` attributes.
    Its exact quartet is taken from an ``exact`` :class:`~httk.atomistic.models.structure.unitcell.UnitcellStructure` when available,
    then from the component presentations' ``exact`` values. If neither component
    carries an exact value, the numeric ``basis`` and ``reduced_coords`` arrays are
    passed through the vector family into ``Cell`` and ``Sites``. That last route is
    exact embedding of the floats: their binary float64 values, rather than their
    decimal spellings, become the exact values.
    """

    _object: Any
    _structure: UnitcellStructure

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
        if isinstance(exact, UnitcellStructure):
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

        self._structure = UnitcellStructure(
            cell,
            sites,
            obj.species,
            obj.species_at_sites,
            molecular=getattr(obj, "molecular", False),
            assemblies=getattr(obj, "assemblies", None),
            symmetry=getattr(obj, "symmetry", None),
            chemical_composition=getattr(obj, "chemical_composition", None),
            chemical_formula_descriptive=getattr(obj, "chemical_formula_descriptive", None),
            chemical_formula_hill=getattr(obj, "chemical_formula_hill", None),
            optimization_type=getattr(obj, "optimization_type", None),
        )

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
