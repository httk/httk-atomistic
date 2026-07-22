"""
The Simple structure representation for httk-atomistic.
"""

from collections.abc import Sequence

from .cell import Cell
from .cell_class_view import CellClassView
from .cell_like import CellLike
from .sites import Sites
from .sites_class_view import SitesClassView
from .sites_like import SitesLike
from .species import Species
from .species_class_view import SpeciesClassView
from .species_like import SpeciesLike


class Structure:
    """
    A crystal structure in the Simple representation.

    A Structure holds a ``cell`` (a ``Cell`` of 3x3 cell vectors), ``sites`` (a ``Sites``
    of Nx3 reduced coordinates), a list of ``species`` (each a ``Species``), and a
    length-N ``species_at_sites`` giving the species name occupying each site. Inputs are
    normalized on construction through the component families: the cell, sites, and each
    species are passed through their ``*Like`` unions, and every ``species_at_sites`` name
    must match one of the (uniquely named) species.

    Note: the numeric values (cell, sites) are stored behind the ``Cell`` and ``Sites``
    component objects, whose interim numerics are nested tuples of floats. They are
    intended to be replaced by the httk exact vector representation fairly soon; keep
    numeric access behind the quartet accessors so that change stays contained.
    """

    _cell: Cell
    _sites: Sites
    _species: tuple[Species, ...]
    _species_at_sites: tuple[str, ...]

    def __init__(
        self,
        cell: CellLike,
        sites: SitesLike,
        species: Sequence[SpeciesLike],
        species_at_sites: Sequence[str],
    ) -> None:
        norm_cell = cell if isinstance(cell, Cell) else CellClassView(cell)
        norm_sites = sites if isinstance(sites, Sites) else SitesClassView(sites)
        norm_species = tuple(s if isinstance(s, Species) else SpeciesClassView(s) for s in species)
        norm_species_at_sites = tuple(str(name) for name in species_at_sites)

        if len(norm_species_at_sites) != len(norm_sites):
            raise ValueError("Structure species_at_sites must have the same length as sites")

        names = [s.name for s in norm_species]
        if len(names) != len(set(names)):
            raise ValueError("Structure species names must be unique")
        known = set(names)
        for name in norm_species_at_sites:
            if name not in known:
                raise ValueError(f"Structure species_at_sites references unknown species name: {name!r}")

        self._cell = norm_cell
        self._sites = norm_sites
        self._species = norm_species
        self._species_at_sites = norm_species_at_sites

    @property
    def cell(self) -> Cell:
        """The cell (3x3 cell vectors) as a ``Cell``."""
        return self._cell

    @property
    def sites(self) -> Sites:
        """The site coordinates (Nx3 reduced coordinates) as a ``Sites``."""
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct species of this structure."""
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site, in site order."""
        return self._species_at_sites

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return NotImplemented
        return (
            self._cell == other._cell
            and self._sites == other._sites
            and self._species == other._species
            and self._species_at_sites == other._species_at_sites
        )

    def __repr__(self) -> str:
        return (
            f"Structure(cell={self._cell!r}, sites={self._sites!r}, "
            f"species={self._species!r}, species_at_sites={self._species_at_sites!r})"
        )
