"""
The Simple structure representation for httk-atomistic.
"""

import datetime
import fractions
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from httk.core import SurdVector, VectorLike

from .cell import Cell
from .cell_like import CellLike
from .cell_view import CellView
from .sites import Sites
from .sites_like import SitesLike
from .sites_view import SitesView
from .species import Species
from .species_like import SpeciesLike
from .species_view import SpeciesView
from .structure_backend import StructureBackend
from .structure_semantics import StructureSemanticsMixin, StructureSymmetry, _semantic_value, initialize_semantics

if TYPE_CHECKING:
    from .composition import Assembly, ChemicalComposition
    from .numeric_unitcell_structure_view import NumericUnitcellStructureView
    from .standardization import ConventionalCellResult


def _norm_cell(cell: CellLike) -> Cell:
    return cell if isinstance(cell, Cell) else CellView(cell)


def _norm_sites(sites: SitesLike) -> Sites:
    return sites if isinstance(sites, Sites) else SitesView(sites)


def _norm_species(species: Sequence[SpeciesLike]) -> tuple[Species, ...]:
    return tuple(s if isinstance(s, Species) else SpeciesView(s) for s in species)


def _norm_species_at_sites(species_at_sites: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(name) for name in species_at_sites)


def _infer_species(species_at_sites: Sequence[SpeciesLike]) -> tuple[tuple[Species, ...], tuple[str, ...]]:
    """Build the distinct species table and site names from convenient values."""
    distinct: list[Species] = []
    by_name: dict[str, Species] = {}
    names: list[str] = []
    for source in species_at_sites:
        value = source if isinstance(source, Species) else SpeciesView(source)
        existing = by_name.get(value.name)
        if existing is None:
            by_name[value.name] = value
            distinct.append(value)
        elif existing != value:
            raise ValueError(
                f"UnitcellStructure species_at_sites gives conflicting definitions for species {value.name!r}"
            )
        names.append(value.name)
    return tuple(distinct), tuple(names)


def _check_species_names(species: Sequence[Species]) -> None:
    names = [s.name for s in species]
    if len(names) != len(set(names)):
        raise ValueError("UnitcellStructure species names must be unique")


def _check_species_at_sites(species_at_sites: Sequence[str], species: Sequence[Species]) -> None:
    known = {s.name for s in species}
    for name in species_at_sites:
        if name not in known:
            raise ValueError(f"UnitcellStructure species_at_sites references unknown species name: {name!r}")


def _check_sites_length(sites: Sites, species_at_sites: Sequence[str]) -> None:
    if len(species_at_sites) != len(sites):
        raise ValueError("UnitcellStructure species_at_sites must have the same length as sites")


class UnitcellStructure(StructureBackend, StructureSemanticsMixin):
    """
    A crystal structure in the Unitcell representation.

    A UnitcellStructure holds a ``cell`` (a ``Cell`` of 3x3 cell vectors), ``sites`` (a ``Sites``
    of Nx3 reduced coordinates), a list of ``species`` (each a ``Species``), and a
    length-N ``species_at_sites`` giving the species name occupying each site. Inputs are
    normalized on construction through the component families: the cell, sites, and each
    species are passed through their ``*Like`` unions, and every ``species_at_sites`` name
    must match one of the (uniquely named) species. When ``species`` is omitted,
    ``species_at_sites`` may itself contain species-like values; the distinct species table
    is then inferred in first-occurrence order.

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
    kind: ClassVar[str] = "unitcell"

    def __init__(
        self,
        cell: CellLike,
        sites: SitesLike,
        species: Sequence[SpeciesLike] | None = None,
        species_at_sites: Sequence[SpeciesLike] | None = None,
        *,
        molecular: bool = False,
        assemblies: Sequence["Assembly"] | None = None,
        symmetry: StructureSymmetry | None = None,
        chemical_composition: "ChemicalComposition | None" = None,
        chemical_formula_descriptive: str | None = None,
        chemical_formula_hill: str | None = None,
        optimization_type: str | None = None,
        immutable_id: str | None = None,
        last_modified: datetime.datetime | None = None,
    ) -> None:
        if species_at_sites is None:
            raise TypeError("UnitcellStructure species_at_sites is required")
        norm_cell = _norm_cell(cell)
        norm_sites = _norm_sites(sites)
        if species is None:
            norm_species, norm_species_at_sites = _infer_species(species_at_sites)
        else:
            norm_species = _norm_species(species)
            norm_species_at_sites = _norm_species_at_sites(species_at_sites)
        _check_sites_length(norm_sites, norm_species_at_sites)
        _check_species_names(norm_species)
        _check_species_at_sites(norm_species_at_sites, norm_species)

        self._cell = norm_cell
        self._sites = norm_sites
        self._species = norm_species
        self._species_at_sites = norm_species_at_sites
        initialize_semantics(
            self,
            nsites=len(norm_sites),
            molecular=molecular,
            assemblies=None if assemblies is None else tuple(assemblies),
            symmetry=symmetry,
            chemical_composition=chemical_composition,
            chemical_formula_descriptive=chemical_formula_descriptive,
            chemical_formula_hill=chemical_formula_hill,
            optimization_type=optimization_type,
            immutable_id=immutable_id,
            last_modified=last_modified,
        )

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

    @property
    def site_coordinate_span(self) -> str:
        """The span asserted by this representation, independent of dimensionality."""
        molecular = _semantic_value(self, "molecular", False, "_molecular")
        return "molecular_unit_cell" if molecular else "unit_cell"

    @property
    def molecular(self) -> bool:
        return bool(_semantic_value(self, "molecular", False, "_molecular"))

    @property
    def symmetry(self) -> StructureSymmetry | None:
        return _semantic_value(self, "symmetry", private_name="_symmetry")

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
        tolerance: fractions.Fraction | str | float | None = None,
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
        tolerance: fractions.Fraction | str | float | None = None,
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
        """Equality of geometry and all structural semantic assertions, including precision."""
        if not isinstance(other, UnitcellStructure):
            return NotImplemented
        return (
            self._cell == other._cell
            and self.basis_precision == other.basis_precision
            and self._sites == other._sites
            and self.coordinate_precision == other.coordinate_precision
            and self._species == other._species
            and self._species_at_sites == other._species_at_sites
            and self.molecular == other.molecular
            and self.assemblies == other.assemblies
            and self.symmetry == other.symmetry
            and self.chemical_composition == other.chemical_composition
            and self.chemical_formula_descriptive == other.chemical_formula_descriptive
            and self.chemical_formula_hill == other.chemical_formula_hill
            and self.optimization_type == other.optimization_type
        )

    def __repr__(self) -> str:
        return (
            f"UnitcellStructure(cell={self._cell!r}, sites={self._sites!r}, "
            f"species={self._species!r}, species_at_sites={self._species_at_sites!r})"
        )


if TYPE_CHECKING:
    from .supercell import SupercellResult
