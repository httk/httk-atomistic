"""
The NumericStructure presentation: a Structure exposed as plain numpy numbers.
"""

from httk.core import NumericVector, to_numeric

from ._vector_guards import require_numpy
from .numeric_cell import NumericCell
from .numeric_sites import NumericSites
from .species import Species
from .structure import Structure
from .structure_like import StructureLike
from .structure_simple_view import StructureSimpleView


class NumericStructure:
    """
    A plain-numpy presentation of a :class:`~httk.atomistic.Structure`.

    Where a ``Structure`` holds its geometry exactly (a surd ``cell`` basis, rational reduced
    coordinates, an exact Cartesian frame), a ``NumericStructure`` mirrors that interface but returns
    plain numpy numbers: its :attr:`cell` is a :class:`~httk.atomistic.NumericCell`, its :attr:`sites`
    a :class:`~httk.atomistic.NumericSites`, and :meth:`cartesian_sites` a ``float64`` numpy array.
    The ``species``/``species_at_sites`` are passed through unchanged (they are already plain data).
    It is for callers who do not need exact arithmetic and just want numpy arrays.

    The presentation is numpy-backed, so constructing a ``NumericStructure`` **requires numpy** (the
    ``httk-atomistic[numpy]`` extra) and raises :class:`ImportError` eagerly when it is unavailable.
    The exact object is always one hop away via :attr:`exact`.
    """

    _structure: Structure

    def __init__(self, structure: StructureLike) -> None:
        require_numpy()
        self._structure = structure if isinstance(structure, Structure) else StructureSimpleView(structure)

    @property
    def cell(self) -> NumericCell:
        """The cell as a :class:`~httk.atomistic.NumericCell`."""
        return NumericCell(self._structure.cell)

    @property
    def sites(self) -> NumericSites:
        """The sites as a :class:`~httk.atomistic.NumericSites`."""
        return NumericSites(self._structure.sites)

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct species, passed through unchanged."""
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site, passed through unchanged."""
        return self._structure.species_at_sites

    def cartesian_sites(self) -> NumericVector:
        """The Cartesian site positions as an ``(N, 3)`` ``float64`` numpy array."""
        return to_numeric(self._structure.cartesian_sites())

    @property
    def exact(self) -> Structure:
        """The exact :class:`~httk.atomistic.Structure` this presentation wraps."""
        return self._structure

    def __repr__(self) -> str:
        return (
            f"NumericStructure(cell={self.cell!r}, sites={self.sites!r}, "
            f"species={self._structure.species!r}, species_at_sites={self._structure.species_at_sites!r})"
        )
