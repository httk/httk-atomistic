"""
The Simple structure representation for httk-atomistic.
"""

import fractions
from collections.abc import Sequence
from typing import TYPE_CHECKING

from httk.core import SurdVector, VectorLike

from .cell import Cell
from .cell_class_view import CellClassView
from .cell_like import CellLike
from .sites import Sites
from .sites_class_view import SitesClassView
from .sites_like import SitesLike
from .species import Species
from .species_class_view import SpeciesClassView
from .species_like import SpeciesLike

if TYPE_CHECKING:
    from .numeric_unitcell_structure_view import NumericUnitcellStructureView
    from .standardization import ConventionalCellResult


class Structure:
    """
    A crystal structure in the Unitcell representation.

    A Structure holds a ``cell`` (a ``Cell`` of 3x3 cell vectors), ``sites`` (a ``Sites``
    of Nx3 reduced coordinates), a list of ``species`` (each a ``Species``), and a
    length-N ``species_at_sites`` giving the species name occupying each site. Inputs are
    normalized on construction through the component families: the cell, sites, and each
    species are passed through their ``*Like`` unions, and every ``species_at_sites`` name
    must match one of the (uniquely named) species.

    The numeric model is exact and split by purpose. The fractional frame — reduced coordinates
    and symmetry — is rational and lives in ``sites`` as a :class:`~httk.core.FracVector`. The
    Cartesian frame — where radicals such as the hexagonal ``sqrt(3)`` appear — is exact in the
    squarefree-radical field: ``cell.basis`` is a :class:`~httk.core.SurdVector` and
    :meth:`cartesian_sites` returns the exact Cartesian positions. Pure magnitudes (bond-length
    comparisons) stay rational-exact via ``cell.metric()``. Floats appear only at the presentation
    and JSON boundaries.
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

    @property
    def coordinate_precision(self) -> fractions.Fraction | None:
        """How precisely the reduced coordinates were stated, in fractional units, or ``None``.

        Read through from :attr:`sites`. Dimensionless — see :meth:`cartesian_precision`
        for the corresponding length.
        """
        return self._sites.precision

    @property
    def basis_precision(self) -> fractions.Fraction | None:
        """How precisely the cell basis was stated, as an absolute length, or ``None``.

        Read through from :attr:`cell`.
        """
        return self._cell.precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Which of the three basis rows is a genuine lattice translation.

        Read through from :attr:`cell`, where the full account lives. ``(True, True, True)``
        for an ordinary crystal, which is what a structure built without saying otherwise is.
        """
        return self._cell.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """How many of the three directions are periodic, from 0 to 3."""
        return self._cell.nperiodic_dimensions

    def cartesian_precision(self) -> fractions.Fraction | None:
        """The coordinate precision as a length, or ``None`` if it is unknown.

        This is the number a real tolerance wants — an interatomic matching tolerance or an
        spglib ``symprec`` is a distance, and a fractional precision is not. A coordinate
        good to ``1e-4`` of a cell edge means something quite different in a 3 Å cell and a
        30 Å one.

        Computed as the fractional precision times the *longest* cell edge, which is the
        conservative choice: it is the largest displacement that fractional uncertainty can
        produce along any axis. The cell's own precision is folded in as well, since a cell
        stated to ``1e-3`` cannot place an atom better than that however many digits the
        coordinates carry.
        """
        fractional = self._sites.precision
        if fractional is None:
            return None
        longest = max(length.to_float() for length in self._cell.lengths)
        cartesian = fractional * fractions.Fraction(str(longest))
        basis = self._cell.precision
        return cartesian if basis is None or basis < cartesian else basis

    def cartesian_sites(self) -> SurdVector:
        """
        The exact Cartesian site positions as an ``(N, 3)`` :class:`~httk.core.SurdVector`.

        Under the row-vector convention this is ``reduced_coords * cell.basis`` (each Cartesian
        position is the sum over lattice vectors ``sum_k reduced[k] * basis[k]``). The reduced
        coordinates are rational (a ``FracVector``), the cell basis carries the radicals (a
        ``SurdVector``), so the product is exact in the surd field — the hexagonal ``sqrt(3)``
        survives into the Cartesian positions.
        """
        return SurdVector.create(self._sites.reduced_coords) * self._cell.basis

    def numeric(self) -> "NumericUnitcellStructureView":
        """A plain-numpy presentation of this structure (requires the ``httk-atomistic[numpy]`` extra)."""
        from .numeric_unitcell_structure_view import NumericUnitcellStructureView

        return NumericUnitcellStructureView(self)

    def supercell(
        self,
        transformation: VectorLike,
        *,
        max_sites: int | None = 100_000,
    ) -> "SupercellResult":
        """Build an exact supercell from an integer 3x3 transformation."""
        from .supercell import build_supercell

        return build_supercell(self, transformation, max_sites=max_sites)

    def orthogonal_supercell(
        self,
        multiplier: int | None = None,
        *,
        tolerance: int | fractions.Fraction | str | float | None = None,
        max_multiplier: int | None = None,
        search_radius: int = 1,
        max_sites: int | None = 100_000,
    ) -> "SupercellResult":
        """Build a deterministically selected orthogonal supercell."""
        from .supercell import orthogonal_supercell

        return orthogonal_supercell(
            self,
            multiplier,
            tolerance=tolerance,
            max_multiplier=max_multiplier,
            search_radius=search_radius,
            max_sites=max_sites,
        )

    def cubic_supercell(
        self,
        multiplier: int | None = None,
        *,
        tolerance: int | fractions.Fraction | str | float | None = None,
        max_multiplier: int | None = None,
        search_radius: int = 1,
        max_sites: int | None = 100_000,
    ) -> "SupercellResult":
        """Build a deterministically selected cubic supercell."""
        from .supercell import cubic_supercell

        return cubic_supercell(
            self,
            multiplier,
            tolerance=tolerance,
            max_multiplier=max_multiplier,
            search_radius=search_radius,
            max_sites=max_sites,
        )

    def conventional_cell(
        self,
        *,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
    ) -> "ConventionalCellResult":
        """Express this structure in its space group's IT standard-setting conventional cell."""
        from .standardization import conventional_cell

        return conventional_cell(
            self,
            tolerance=tolerance,
            limit_denominator=limit_denominator,
        )

    def __eq__(self, other: object) -> bool:
        """Equality of the quartet. Stated precision does not take part.

        Precision describes how the structure was *recorded*, not what it is, so the same
        atoms read from a more carefully written file are still the same structure. The
        component classes take the same view.
        """
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


if TYPE_CHECKING:
    from .supercell import SupercellResult
