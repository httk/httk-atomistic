"""
The Simple structure representation for httk-atomistic.
"""

import datetime
import fractions
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

from httk.core import SurdVector, VectorLike, unwrap
from httk.core.exactmath import best_rational_in_interval

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.moments.like import SiteMomentsLike
from httk.atomistic.models.moments.view_base import SiteMomentsViewBase
from httk.atomistic.models.sites.cartesian import CartesianSites
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view import SitesView
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view import SpeciesView
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.semantics import (
    StructureSemanticsMixin,
    StructureSymmetry,
    _semantic_value,
    initialize_semantics,
)

if TYPE_CHECKING:
    from httk.atomistic.composition import Assembly, ChemicalComposition
    from httk.atomistic.models.structure.numeric_view import NumericUnitcellStructureView
    from httk.atomistic.symmetry.standardization import ConventionalCellResult


def _norm_cell(cell: CellLike) -> Cell:
    return cell if isinstance(cell, Cell) else CellView(cell)


def _norm_sites(sites: SitesLike) -> Sites:
    return sites if isinstance(sites, Sites) else SitesView(sites)


def _cartesian_backend(sites: Sites, cell: Cell) -> CartesianSites | None:
    """The sites' :class:`CartesianSites` backend when its frame is exactly the cell basis, else ``None``."""
    backend = unwrap(sites)
    return backend if isinstance(backend, CartesianSites) and backend.basis == cell.basis else None


def _norm_species(species: Sequence[SpeciesLike]) -> tuple[Species, ...]:
    return tuple(s if isinstance(s, Species) else SpeciesView(s) for s in species)


def _norm_species_at_sites(species_at_sites: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(name) for name in species_at_sites)


def _norm_site_moments(value: SiteMomentsLike | None) -> SiteMomentsBackend | None:
    if value is None:
        return None
    if isinstance(value, SiteMomentsBackend):
        return value
    if isinstance(value, SiteMomentsViewBase):
        return cast(SiteMomentsBackend, value)
    raise TypeError(
        "UnitcellStructure site_moments must be a SiteMoments class or view; a bare array is "
        "frame-ambiguous, so construct CartesianSiteMoments, CrystalAxisSiteMoments, or "
        "CollinearSiteMoments explicitly"
    )


def _check_site_moments(value: SiteMomentsBackend | None, sites: Sites, cell: Cell) -> None:
    if value is None:
        return
    if len(value) != len(sites):
        raise ValueError("UnitcellStructure site_moments must have the same length as sites")
    if isinstance(value, CrystalAxisSiteMoments) and value.cell != cell:
        raise ValueError("UnitcellStructure site_moments has an incoherent crystal-axis frame")


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


def _check_referenced_species_concentrations(species_at_sites: Sequence[str], species: Sequence[Species]) -> None:
    referenced = set(species_at_sites)
    if any(value > 1 for item in species if item.name in referenced for value in item.concentration):
        raise ValueError("species assigned to sites must have concentration values in [0, 1]")


def _check_sites_length(sites: Sites, species_at_sites: Sequence[str]) -> None:
    if len(species_at_sites) != len(sites):
        raise ValueError("UnitcellStructure species_at_sites must have the same length as sites")


class UnitcellStructure(StructureSemanticsMixin, StructureBackend):
    """Represent a crystal structure in the Unitcell representation.

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

    :param cell: The unit-cell geometry.
    :param sites: The reduced coordinates of the sites.
    :param species: The distinct species definitions. Omit this to infer them from
        ``species_at_sites``.
    :param species_at_sites: The species name occupying each site; this value is required.
    :param site_moments: Optional magnetic moments aligned with the sites.
    :param molecular: Whether the structure describes a molecular unit cell.
    :param assemblies: Optional correlations among sites.
    :param symmetry: Optional symmetry metadata.
    :param chemical_composition: Optional chemical composition metadata.
    :param chemical_formula_descriptive: Optional descriptive chemical formula.
    :param chemical_formula_hill: Optional Hill chemical formula.
    :param optimization_type: Optional optimization provenance.
    :param immutable_id: Optional immutable source identifier.
    :param last_modified: Optional source modification timestamp.
    :param charge: An explicitly assigned charge for the cell content; it is not derived
        from the species, and an explicit zero remains distinct from an unstated charge.
    """

    _cell: Cell
    _sites: Sites
    _species: tuple[Species, ...]
    _species_at_sites: tuple[str, ...]
    _site_moments: SiteMomentsBackend | None
    _charge: fractions.Fraction | None
    kind: ClassVar[str] = "unitcell"

    def __init__(
        self,
        cell: CellLike,
        sites: SitesLike,
        species: Sequence[SpeciesLike] | None = None,
        species_at_sites: Sequence[SpeciesLike] | None = None,
        *,
        site_moments: SiteMomentsLike | None = None,
        molecular: bool = False,
        assemblies: Sequence["Assembly"] | None = None,
        symmetry: StructureSymmetry | None = None,
        chemical_composition: "ChemicalComposition | None" = None,
        chemical_formula_descriptive: str | None = None,
        chemical_formula_hill: str | None = None,
        optimization_type: str | None = None,
        immutable_id: str | None = None,
        last_modified: datetime.datetime | None = None,
        charge: fractions.Fraction | int | str | None = None,
    ) -> None:
        if species_at_sites is None:
            raise TypeError("UnitcellStructure species_at_sites is required")
        norm_cell = _norm_cell(cell)
        norm_sites = _norm_sites(sites)
        if isinstance(unwrap(norm_sites), CartesianSites) and _cartesian_backend(norm_sites, norm_cell) is None:
            raise ValueError("UnitcellStructure CartesianSites basis must equal the cell basis exactly")
        if species is None:
            norm_species, norm_species_at_sites = _infer_species(species_at_sites)
        else:
            norm_species = _norm_species(species)
            norm_species_at_sites = _norm_species_at_sites(species_at_sites)
        _check_sites_length(norm_sites, norm_species_at_sites)
        _check_species_names(norm_species)
        _check_species_at_sites(norm_species_at_sites, norm_species)
        _check_referenced_species_concentrations(norm_species_at_sites, norm_species)
        norm_site_moments = _norm_site_moments(site_moments)
        _check_site_moments(norm_site_moments, norm_sites, norm_cell)

        self._cell = norm_cell
        self._sites = norm_sites
        self._species = norm_species
        self._species_at_sites = norm_species_at_sites
        self._site_moments = norm_site_moments
        self._charge = None if charge is None else fractions.Fraction(charge)
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
        """Expose the cell geometry.

        :return: The cell in the structure's exact representation.
        """
        return self._cell

    @property
    def sites(self) -> Sites:
        """Expose the reduced site coordinates.

        :return: The sites in the structure's exact representation.
        """
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        """Expose the distinct species.

        :return: The species referenced by the structure.
        """
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose the species name occupying each site.

        :return: Site species names in site order.
        """
        return self._species_at_sites

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        """Expose optional per-site magnetic moments in ``sites`` order.

        :return: Site moments, or ``None`` when they are unstated.
        """
        return self._site_moments

    @property
    def charge(self) -> fractions.Fraction | None:
        """Expose the explicitly assigned exact charge of the cell.

        :return: The assigned charge, or ``None`` when it is unstated.
        """
        return self._charge

    @property
    def coordinate_precision(self) -> fractions.Fraction | None:
        """Expose the precision recorded for the reduced coordinates.

        Read through from :attr:`sites`. Dimensionless — see :meth:`cartesian_precision`
        for the corresponding length.

        :return: The fractional precision, or ``None`` when it is unknown.
        """
        return self._sites.precision

    @property
    def basis_precision(self) -> fractions.Fraction | None:
        """Expose the precision recorded for the cell basis.

        Read through from :attr:`cell`.

        :return: The absolute precision, or ``None`` when it is unknown.
        """
        return self._cell.precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Expose which cell directions are periodic.

        Read through from :attr:`cell`, where the full account lives. ``(True, True, True)``
        for an ordinary crystal, which is what a structure built without saying otherwise is.

        :return: The periodicity flags for the cell directions.
        """
        return self._cell.periodicity

    @property
    def site_coordinate_span(self) -> str:
        """Expose the coordinate span asserted by this representation.

        :return: ``unit_cell`` or ``molecular_unit_cell``.
        """
        molecular = _semantic_value(self, "molecular", False, "_molecular")
        return "molecular_unit_cell" if molecular else "unit_cell"

    @property
    def molecular(self) -> bool:
        """Expose whether this structure describes a molecular unit cell.

        :return: Whether molecular semantics are enabled.
        """
        return bool(_semantic_value(self, "molecular", False, "_molecular"))

    @property
    def symmetry(self) -> StructureSymmetry | None:
        """Expose the optional symmetry metadata.

        :return: The symmetry metadata, or ``None`` when it is absent.
        """
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
        coordinates carry. Sites backed by :class:`~httk.atomistic.CartesianSites` with a stated
        ``cartesian_precision`` use that length directly instead of the fractional conversion.

        :return: The conservative Cartesian precision, or ``None`` when the coordinate
            precision is unknown.
        """
        backend = _cartesian_backend(self._sites, self._cell)
        if backend is not None and backend.cartesian_precision is not None:
            cartesian = backend.cartesian_precision
        else:
            fractional = self._sites.precision
            if fractional is None:
                return None
            longest = max(length.to_float() for length in self._cell.lengths)
            cartesian = fractional * fractions.Fraction(str(longest))
        basis = self._cell.precision
        return cartesian if basis is None or basis < cartesian else basis

    def cartesian_sites(self) -> SurdVector:
        """Compute the exact Cartesian site positions.

        Under the row-vector convention this is ``reduced_coords * cell.basis`` (each Cartesian
        position is the sum over lattice vectors ``sum_k reduced[k] * basis[k]``). The reduced
        coordinates are rational (a ``FracVector``), the cell basis carries the radicals (a
        ``SurdVector``), so the product is exact in the surd field — the hexagonal ``sqrt(3)``
        survives into the Cartesian positions.

        Sites backed by :class:`~httk.atomistic.CartesianSites` return their stored coordinates
        unchanged.

        :return: The Cartesian positions in the exact surd representation.
        """
        cartesian = _cartesian_backend(self._sites, self._cell)
        if cartesian is not None:
            return cartesian.cartesian_coords
        return SurdVector(self._sites.reduced_coords) * self._cell.basis

    def numeric(self) -> "NumericUnitcellStructureView":
        """Create a plain-numpy presentation of this structure.

        :return: The numpy-backed structure view.
        :raises ImportError: If numpy is unavailable.
        """
        from httk.atomistic.models.structure.numeric_view import NumericUnitcellStructureView

        return NumericUnitcellStructureView(self)

    def rationalized(self, *, coordinate_tolerance: Any = None, basis_tolerance: Any = None) -> "UnitcellStructure":
        """Return an EXPLICIT lossy copy whose numbers are snapped to the simplest nearby values.

        Numbers are kept exactly as written by default (``"0.3333"`` is ``3333/10000``); this
        opt-in step replaces each reduced coordinate by the simplest number within
        ``coordinate_tolerance``, and each cell length within ``basis_tolerance``: the
        smallest-denominator rational or the shortest decimal in the interval, whichever has
        fewer digits (a tie goes to the decimal). So ``0.3333`` becomes ``1/3``. An omitted
        tolerance defaults to half the structure's stated precision (:attr:`coordinate_precision`
        or :attr:`basis_precision`, the coarsest of the source's numbers), capped at ``1/10000``.
        A full-precision CONTCAR read with digit-derived precision (``1e-16``) therefore keeps its
        float noise; with a realistic stated precision (``load(path, precision=5e-4)``) or explicit
        tolerances, a relaxed ``0.3549999999999969`` becomes ``0.355``. A component with neither a
        tolerance nor a stated precision is left unchanged.

        Coordinates come back as plain :class:`~httk.atomistic.Sites` (also for Cartesian-backed
        sites). A cell built from parameters keeps its exact angles and only its lengths ``a``,
        ``b``, ``c`` are snapped; a cell with a rational basis has each unscaled entry snapped
        within ``basis_tolerance / scale`` (``scale`` is kept); any other cell (an irrational
        basis not built from parameters) is left unchanged, so radicals such as the hexagonal
        ``sqrt(3)`` are never rounded away. Crystal-axis moments are re-expressed in the new cell
        so their Cartesian vectors are unchanged. Precision and all other metadata are carried
        through; ``immutable_id`` and ``last_modified`` are not, since the result is a new value.

        :param coordinate_tolerance: The fractional snapping tolerance, or ``None`` for the default.
        :param basis_tolerance: The snapping tolerance for lengths, or ``None`` for the default.
        :return: The rationalized structure.
        :raises ValueError: If a tolerance is not strictly positive.
        """
        from httk.atomistic.models._vector_guards import to_precision
        from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
        from httk.atomistic.models.moments.crystalaxis_view import CrystalAxisSiteMomentsView

        coordinate_tol = _rationalizing_tolerance(to_precision(coordinate_tolerance), self.coordinate_precision)
        basis_tol = _rationalizing_tolerance(to_precision(basis_tolerance), self.basis_precision)
        # Always plain Sites: Cartesian-backed sites would contradict a snapped cell's basis.
        coords: VectorLike = self.sites.reduced_coords
        if coordinate_tol is not None:
            coords = [[_snap(x, coordinate_tol) for x in row] for row in self.sites.reduced_coords.to_fractions()]
        sites = Sites(coords, precision=self.sites.precision)
        cell = self.cell if basis_tol is None else _rationalized_cell(self.cell, basis_tol)
        moments = self.site_moments
        if cell is not self.cell and isinstance(moments, CrystalAxisSiteMoments):
            cartesian = CartesianSiteMoments(moments.cartesian_moments, moments.precision)
            moments = CrystalAxisSiteMomentsView(cartesian, cell=cell).unview()
        return UnitcellStructure(
            cell,
            sites,
            self.species,
            self.species_at_sites,
            site_moments=moments,
            molecular=self.molecular,
            assemblies=getattr(self, "assemblies", None),
            symmetry=self.symmetry,
            chemical_composition=getattr(self, "chemical_composition", None),
            chemical_formula_descriptive=getattr(self, "chemical_formula_descriptive", None),
            chemical_formula_hill=getattr(self, "chemical_formula_hill", None),
            optimization_type=getattr(self, "optimization_type", None),
            charge=self.charge,
        )

    def supercell(
        self,
        transformation: VectorLike,
        *,
        max_sites: int | None = 100_000,
    ) -> "SupercellResult":
        """Build an exact supercell from an integer transformation.

        :param transformation: The lattice transformation to apply.
        :param max_sites: The maximum permitted number of sites, or ``None`` for no limit.
        :return: The generated supercell and transformation metadata.
        """
        from httk.atomistic.supercell import build_supercell

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
        """Build a deterministically selected orthogonal supercell.

        :param multiplier: The requested volume multiplier, or ``None`` to search.
        :param tolerance: The geometric tolerance used during the search.
        :param max_multiplier: The largest multiplier considered when searching.
        :param search_radius: The integer search radius for candidate transformations.
        :param max_sites: The maximum permitted number of sites, or ``None`` for no limit.
        :return: The generated supercell and transformation metadata.
        """
        from httk.atomistic.supercell import orthogonal_supercell

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
        """Build a deterministically selected cubic supercell.

        :param multiplier: The requested volume multiplier, or ``None`` to search.
        :param tolerance: The geometric tolerance used during the search.
        :param max_multiplier: The largest multiplier considered when searching.
        :param search_radius: The integer search radius for candidate transformations.
        :param max_sites: The maximum permitted number of sites, or ``None`` for no limit.
        :return: The generated supercell and transformation metadata.
        """
        from httk.atomistic.supercell import cubic_supercell

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
        """Express this structure in its conventional standard-setting cell.

        :param tolerance: The tolerance used when standardizing the structure.
        :param limit_denominator: The denominator limit used for rationalizing measured
            coordinates.
        :return: The standardized structure and its transformation metadata.
        """
        from httk.atomistic.symmetry.standardization import conventional_cell

        return conventional_cell(
            self,
            tolerance=tolerance,
            limit_denominator=limit_denominator,
        )

    def __eq__(self, other: object) -> bool:
        """Equality of geometry and structural assertions, including precision and moments.

        Moments are content and participate in equality; their own stated precision remains
        excluded by the SiteMoments equality contract.
        """
        if not isinstance(other, UnitcellStructure):
            return NotImplemented
        return (
            self._cell == other._cell
            and self.basis_precision == other.basis_precision
            and self._sites == other._sites
            and self.coordinate_precision == other.coordinate_precision
            and self._species == other._species
            and self._species_at_sites == other._species_at_sites
            and self._site_moments == other._site_moments
            and self.charge == other.charge
            and self.molecular == other.molecular
            and self.assemblies == other.assemblies
            and self.symmetry == other.symmetry
            and self.chemical_composition == other.chemical_composition
            and self.chemical_formula_descriptive == other.chemical_formula_descriptive
            and self.chemical_formula_hill == other.chemical_formula_hill
            and self.optimization_type == other.optimization_type
        )

    def __repr__(self) -> str:
        n = len(self._species_at_sites)
        if n <= 12:
            # Cartesian-backed sites show their backend, so eval(repr(x)) keeps the representation.
            cartesian = _cartesian_backend(self._sites, self._cell)
            sites = self._sites if cartesian is None else cartesian
            return (
                f"UnitcellStructure(cell={self._cell!r}, sites={sites!r}, "
                f"species={self._species!r}, species_at_sites={self._species_at_sites!r})"
            )
        # Large structure: abbreviate the bulky arguments and mark the elision with a trailing ...
        return (
            f"UnitcellStructure(cell={self._cell!r}, sites=Sites(... {n} sites ...), "
            f"species=(... {len(self._species)} species ...), ...)"
        )

    def __str__(self) -> str:
        names = sorted(set(self._species_at_sites))
        formula = " ".join(
            f"{name}{self._species_at_sites.count(name)}" if self._species_at_sites.count(name) > 1 else name
            for name in names
        )
        return (
            f"UnitcellStructure({formula}: {len(self._species_at_sites)} sites, volume={float(self._cell.volume):.4g})"
        )


if TYPE_CHECKING:
    from httk.atomistic.supercell import SupercellResult


# Bounds the rational approximation of an irrational cell scale; far below any length tolerance.
_SCALE_BOUND_ERROR = fractions.Fraction(1, 10**30)


def _rationalizing_tolerance(
    explicit: fractions.Fraction | None, precision: fractions.Fraction | None
) -> fractions.Fraction | None:
    """Return the explicit tolerance, else half the stated precision capped at ``1/10000``."""
    if explicit is not None or precision is None:
        return explicit
    return min(precision / 2, fractions.Fraction(1, 10000))


def _snap(value: fractions.Fraction, tolerance: fractions.Fraction) -> fractions.Fraction:
    """Return the simpler of the best rational and the shortest decimal within ``tolerance``.

    Simplicity is the number of digits written (sign excluded): numerator plus denominator for
    ``p/q``, integer part plus fractional places for a decimal (``0.001001`` is seven); a tie
    goes to the decimal.
    """
    low, high = value - tolerance, value + tolerance
    rational = best_rational_in_interval(low, high)
    places = 0
    while math.ceil(low * 10**places) > math.floor(high * 10**places):
        places += 1
    scale = 10**places
    mantissa = min(max(round(value * scale), math.ceil(low * scale)), math.floor(high * scale))
    rational_digits = len(str(abs(rational.numerator))) + (
        len(str(rational.denominator)) if rational.denominator > 1 else 0
    )
    decimal_digits = len(str(abs(mantissa) // scale)) + places
    return rational if rational_digits < decimal_digits else fractions.Fraction(mantissa, scale)


def _rationalized_cell(cell: Cell, tolerance: fractions.Fraction) -> Cell:
    """Snap a parameter cell's lengths, or a rational basis's entries; leave other cells unchanged."""
    from httk.atomistic.models.cell.params import CellParams

    backend = getattr(cell, "_backend", cell)
    if isinstance(backend, CellParams):
        a, b, c, alpha, beta, gamma = backend.params
        lengths = (_snap(a, tolerance), _snap(b, tolerance), _snap(c, tolerance))
        return CellView(CellParams((*lengths, alpha, beta, gamma), precision=backend.precision))
    unscaled = cell.unscaled_basis
    if not unscaled.is_rational:
        return cell
    scale = cell.scale
    # An upper bound on the scale keeps every scaled entry within the absolute ``tolerance``.
    scale_bound = (
        scale._rational_fraction()
        if scale.is_rational
        else scale.to_fractions_approx(_SCALE_BOUND_ERROR) + _SCALE_BOUND_ERROR
    )
    entry_tolerance = tolerance / scale_bound
    rows = [[_snap(x, entry_tolerance) for x in row] for row in unscaled.to_fractions_approx()]
    return Cell(rows, scale=scale, precision=cell.precision, periodicity=cell.periodicity)
