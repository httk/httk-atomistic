"""The NumericUnitcellStructure value and the backend adapter that gives it the exact interface."""

import datetime
from collections.abc import Sequence
from fractions import Fraction
from functools import cached_property
from typing import Any, ClassVar, Self

from httk.core.exactmath import any_to_fraction

from httk.atomistic.composition import project_composition
from httk.atomistic.models._vector_guards import matmul_floats
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.numeric import NumericCell, finite_float
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.numeric import NumericSiteMoments
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.semantics import (
    validate_descriptive_formula,
    validate_hill_formula,
    validate_optimization_type,
)
from httk.atomistic.models.structure.unitcell import (
    UnitcellStructure,
    _check_referenced_species_concentrations,
    _check_species_at_sites,
    _check_species_names,
    _infer_species,
    _norm_species,
    _norm_species_at_sites,
)

# Exact-interface content a float structure does not carry; converting a structure that has
# any of it is refused rather than silently dropped.
_UNSUPPORTED = ("assemblies", "symmetry", "chemical_composition")


def _quantized(value: Fraction | None) -> Fraction | None:
    return None if value is None else any_to_fraction(float(value))


def _quantized_species(species: Species) -> Species:
    """Return ``species`` with its exact numbers rounded through float, so storing them is lossless."""

    def each(values: tuple[Fraction | None, ...] | None) -> tuple[Fraction | None, ...] | None:
        return None if values is None else tuple(_quantized(value) for value in values)

    return Species(
        species.name,
        species.chemical_symbols,
        tuple(any_to_fraction(float(value)) for value in species.concentration),
        mass=None if species.mass is None else tuple(finite_float(value, "Species mass") for value in species.mass),
        original_name=species.original_name,
        attached=species.attached,
        nattached=species.nattached,
        concentration_precision=each(species.concentration_precision),
        charges=each(species.charges),
        spins=each(species.spins),
        labels=species.labels,
    )


def _exact_site_moments(moments: NumericSiteMoments) -> SiteMomentsBackend:
    """Return float moments as the exact moments of the same kind, each float as written."""
    precision = None if moments.precision is None else any_to_fraction(moments.precision)
    values: Any = moments._moments
    if moments.kind == "collinear":
        return CollinearSiteMoments([any_to_fraction(value) for value in values], precision)
    return CartesianSiteMoments([[any_to_fraction(value) for value in row] for row in values], precision)


class NumericUnitcellStructure:
    """A unit-cell crystal structure that owns its geometry as plain floats, an immutable value.

    The float counterpart of :class:`~httk.atomistic.UnitcellStructure`: its :attr:`cell` is a
    :class:`~httk.atomistic.models.cell.numeric.NumericCell` and its :attr:`sites` a
    :class:`~httk.atomistic.models.sites.numeric.NumericSites`. Species follow the
    ``UnitcellStructure`` rules (inferred from ``species_at_sites`` when ``species`` is omitted)
    and are rounded through float at construction: every concentration, concentration precision,
    charge and spin becomes the shortest decimal of its float (``1/3`` becomes
    ``0.3333333333333333``), so a float round trip of the value is the identity. ``-0.0`` is
    normalised to ``0.0`` and non-finite numbers are rejected.

    Construction from a single structure-like argument (anything
    :class:`~httk.atomistic.UnitcellStructureView` accepts) is the explicit lossy conversion. A
    source's site moments become :class:`~httk.atomistic.models.moments.numeric.NumericSiteMoments`
    (crystal-axis moments become Cartesian). A numeric structure carries no assemblies, symmetry
    metadata or chemical composition, so converting a structure that has any of them raises
    :class:`ValueError`. The source's ``immutable_id`` and ``last_modified`` describe the source
    and are not carried.

    ``UnitcellStructureView(numeric)`` presents the exact interface, with the floats embedded
    exactly as written. The composition properties come from that view, built lazily.
    Construction requires numpy (raising :class:`ImportError` otherwise).

    :param cell: The cell (anything ``NumericCell`` accepts), or a structure to convert.
    :param sites: The reduced coordinates (anything ``NumericSites`` accepts); omit when
        converting a structure.
    :param species: The distinct species definitions. Omit this to infer them from
        ``species_at_sites``.
    :param species_at_sites: The species name occupying each site; required unless converting.
    :param site_moments: Optional per-site magnetic moments in site order (anything
        ``NumericSiteMoments`` accepts).
    :param charge: An explicitly assigned charge for the cell content.
    :param molecular: Whether the structure describes a molecular unit cell.
    :param chemical_formula_descriptive: Optional descriptive chemical formula.
    :param chemical_formula_hill: Optional Hill chemical formula.
    :param optimization_type: Optional optimization provenance.
    :param immutable_id: Optional immutable source identifier.
    :param last_modified: Optional source modification timestamp.
    :raises ValueError: If a value is invalid, or a converted structure has content a numeric
        structure cannot carry.
    :raises TypeError: If ``species_at_sites`` is missing, a value has the wrong type, or content
        keywords are given alongside a structure to convert.
    """

    __httk_storage_record__: ClassVar[type[Any]]
    _cell: NumericCell
    _sites: NumericSites
    _species: tuple[Species, ...]
    _species_at_sites: tuple[str, ...]
    _site_moments: NumericSiteMoments | None
    _charge: float | None
    _molecular: bool
    _chemical_formula_descriptive: str | None
    _chemical_formula_hill: str | None
    _optimization_type: str | None
    _immutable_id: str | None
    _last_modified: datetime.datetime | None

    def __init__(
        self,
        cell: Any,
        sites: Any = None,
        species: Sequence[SpeciesLike] | None = None,
        species_at_sites: Sequence[SpeciesLike] | None = None,
        *,
        site_moments: Any = None,
        charge: float | None = None,
        molecular: bool = False,
        chemical_formula_descriptive: str | None = None,
        chemical_formula_hill: str | None = None,
        optimization_type: str | None = None,
        immutable_id: str | None = None,
        last_modified: datetime.datetime | None = None,
    ) -> None:
        if sites is None and species is None and species_at_sites is None:
            from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

            if (
                site_moments,
                charge,
                molecular,
                chemical_formula_descriptive,
                chemical_formula_hill,
                optimization_type,
            ) != (None, None, False, None, None, None):
                raise TypeError(
                    "converting a structure takes its content from the source; only immutable_id "
                    "and last_modified may be given"
                )
            exact = UnitcellStructureView(cell)
            present = [name for name in _UNSUPPORTED if getattr(exact, name, None) is not None]
            if present:
                # ponytail: numeric structures carry no assemblies/symmetry/composition;
                # use UnitcellStructure for those until a numeric form is needed.
                raise ValueError(
                    f"NumericUnitcellStructure cannot carry {', '.join(present)}; use UnitcellStructure instead"
                )
            cell, sites = exact.cell, exact.sites
            species, species_at_sites = exact.species, exact.species_at_sites
            site_moments = exact.site_moments
            charge = None if exact.charge is None else float(exact.charge)
            molecular = exact.molecular
            chemical_formula_descriptive = exact.chemical_formula_descriptive
            chemical_formula_hill = exact.chemical_formula_hill
            optimization_type = exact.optimization_type
        if species_at_sites is None:
            raise TypeError("NumericUnitcellStructure species_at_sites is required")
        if species is None:
            norm_species, norm_species_at_sites = _infer_species(species_at_sites)
        else:
            norm_species = _norm_species(species)
            norm_species_at_sites = _norm_species_at_sites(species_at_sites)
        norm_species = tuple(_quantized_species(value) for value in norm_species)
        self._cell = NumericCell(cell)
        self._sites = NumericSites(sites)
        if len(norm_species_at_sites) != len(self._sites):
            raise ValueError("NumericUnitcellStructure species_at_sites must have the same length as sites")
        norm_site_moments = None if site_moments is None else NumericSiteMoments(site_moments)
        if norm_site_moments is not None and len(norm_site_moments) != len(self._sites):
            raise ValueError("NumericUnitcellStructure site_moments must have the same length as sites")
        if norm_site_moments is not None and not len(norm_site_moments):
            # The exact moment classes hold at least one site, so the exact interface could not present these.
            raise ValueError("NumericUnitcellStructure site_moments of a structure without sites must be None")
        _check_species_names(norm_species)
        _check_species_at_sites(norm_species_at_sites, norm_species)
        _check_referenced_species_concentrations(norm_species_at_sites, norm_species)
        if not isinstance(molecular, bool):
            raise TypeError("molecular must be a bool")
        if immutable_id is not None and not isinstance(immutable_id, str):
            raise TypeError("immutable_id must be a string or None")
        if last_modified is not None:
            if not isinstance(last_modified, datetime.datetime):
                raise TypeError("last_modified must be a datetime or None")
            if last_modified.tzinfo is None or last_modified.utcoffset() is None:
                raise ValueError("last_modified must include a timezone")
        self._species = norm_species
        self._species_at_sites = norm_species_at_sites
        self._site_moments = norm_site_moments
        self._charge = None if charge is None else finite_float(charge, "charge")
        self._molecular = molecular
        self._chemical_formula_descriptive = validate_descriptive_formula(chemical_formula_descriptive)
        self._chemical_formula_hill = validate_hill_formula(chemical_formula_hill, project_composition(self))
        self._optimization_type = validate_optimization_type(optimization_type)
        self._immutable_id = immutable_id
        self._last_modified = last_modified

    @property
    def cell(self) -> NumericCell:
        """The cell geometry as floats.

        :return: The float-owning cell.
        """
        return self._cell

    @property
    def sites(self) -> NumericSites:
        """The reduced site coordinates as floats.

        :return: The float-owning sites.
        """
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct species, rounded through float.

        :return: The species referenced by the structure.
        """
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site.

        :return: Site species names in site order.
        """
        return self._species_at_sites

    @property
    def site_moments(self) -> NumericSiteMoments | None:
        """The per-site magnetic moments as floats.

        :return: The moments, or ``None`` when they are unstated.
        """
        return self._site_moments

    @property
    def charge(self) -> float | None:
        """The explicitly assigned charge of the cell.

        :return: The assigned charge, or ``None`` when it is unstated.
        """
        return self._charge

    @property
    def bonds(self) -> None:
        """Expose no bond network, since a float-owning value has none; converting a structure drops it.

        :return: ``None``.
        """
        return None

    @property
    def molecular(self) -> bool:
        """Whether this structure describes a molecular unit cell.

        :return: Whether molecular semantics are enabled.
        """
        return self._molecular

    @property
    def chemical_formula_descriptive(self) -> str | None:
        """The descriptive chemical formula.

        :return: The formula, or ``None`` when it is unstated.
        """
        return self._chemical_formula_descriptive

    @property
    def chemical_formula_hill(self) -> str | None:
        """The Hill chemical formula.

        :return: The formula, or ``None`` when it is unstated.
        """
        return self._chemical_formula_hill

    @property
    def optimization_type(self) -> str | None:
        """The optimization provenance.

        :return: The optimization type, or ``None`` when it is unstated.
        """
        return self._optimization_type

    @property
    def immutable_id(self) -> str | None:
        """The immutable source identifier.

        :return: The identifier, or ``None`` when it is unstated.
        """
        return self._immutable_id

    @property
    def last_modified(self) -> datetime.datetime | None:
        """The source modification timestamp.

        :return: The timestamp, or ``None`` when it is unstated.
        """
        return self._last_modified

    @property
    def type(self) -> str:
        """Expose the logical OPTIMADE entry family.

        :return: ``"structures"``.
        """
        return "structures"

    @property
    def id(self) -> str:
        """The content identity of this numeric structure.

        Numeric and exact copies of a structure are different content with different ids.

        :return: The content identifier.
        """
        from httk.core.storage import content_id

        return content_id(self)

    @property
    def nsites(self) -> int:
        """The number of sites.

        :return: The site count.
        """
        return len(self._species_at_sites)

    @cached_property
    def _exact(self) -> UnitcellStructure:
        from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

        return UnitcellStructureView(self)

    @property
    def elements(self) -> tuple[str, ...]:
        """The composition's element symbols.

        :return: Element symbols in composition order.
        """
        return self._exact.elements

    @property
    def nelements(self) -> int:
        """The number of composition elements.

        :return: The number of distinct elements.
        """
        return self._exact.nelements

    @property
    def elements_ratios(self) -> tuple[Fraction, ...]:
        """The normalized composition element ratios.

        :return: Element ratios in :attr:`elements` order.
        """
        return self._exact.elements_ratios

    @property
    def chemical_formula_reduced(self) -> str | None:
        """The reduced composition formula.

        :return: The reduced formula, or ``None`` when unavailable.
        """
        return self._exact.chemical_formula_reduced

    @property
    def chemical_formula_anonymous(self) -> str | None:
        """The anonymous composition formula.

        :return: The anonymous formula, or ``None`` when unavailable.
        """
        return self._exact.chemical_formula_anonymous

    @property
    def structure_features(self) -> tuple[str, ...]:
        """The derived OPTIMADE structure-feature flags.

        :return: Feature flags in canonical order.
        """
        return self._exact.structure_features

    @property
    def lattice_vectors(self) -> list[list[float]]:
        """The cell vectors as float rows.

        :return: The three lattice vectors.
        """
        return [list(row) for row in self._cell._basis]

    @property
    def fractional_site_positions(self) -> list[list[float]]:
        """The reduced site positions as float rows.

        :return: Fractional positions in site order.
        """
        return [list(row) for row in self._sites._reduced_coords]

    @property
    def cartesian_site_positions(self) -> list[list[float]]:
        """The Cartesian site positions as float rows.

        :return: Cartesian positions in site order.
        """
        return matmul_floats(self.fractional_site_positions, self.lattice_vectors)

    def _key(self) -> tuple[Any, ...]:
        return (
            self._cell,
            self._sites,
            self._species,
            self._species_at_sites,
            self._site_moments,
            self._charge,
            self._molecular,
            self._chemical_formula_descriptive,
            self._chemical_formula_hill,
            self._optimization_type,
        )

    def __eq__(self, other: object) -> bool:
        """Equality of content; ``immutable_id`` and ``last_modified`` do not take part.

        :param other: The object to compare with.
        :return: Whether the structures hold the same values.
        """
        if not isinstance(other, NumericUnitcellStructure):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        parts = [
            f"cell={self._cell!r}",
            f"sites={self._sites!r}",
            f"species={self._species!r}",
            f"species_at_sites={self._species_at_sites!r}",
        ]
        for name, default in (
            ("site_moments", None),
            ("charge", None),
            ("molecular", False),
            ("chemical_formula_descriptive", None),
            ("chemical_formula_hill", None),
            ("optimization_type", None),
            ("immutable_id", None),
            ("last_modified", None),
        ):
            value = getattr(self, f"_{name}")
            if value != default:
                parts.append(f"{name}={value!r}")
        return f"NumericUnitcellStructure({', '.join(parts)})"


class _NumericStructureBackend(StructureBackend):
    r"""Backend adapter giving float-owning structures the exact structure interface.

    Adopts a :class:`NumericUnitcellStructure`, its storage record (rebuilt through
    :func:`~httk.atomistic.storage.records.numeric_structure_from_record`; :meth:`unwrap` returns
    the record), or any object whose ``cell``/``sites`` are a ``NumericCell``/``NumericSites`` and
    that has ``species``/``species_at_sites``. The exact
    quartet is taken from an ``exact`` :class:`~httk.atomistic.UnitcellStructure` when the object
    has one, then from the components' ``exact`` ``Cell``/``Sites`` (the numeric views carry them),
    and otherwise from the floats embedded exactly as written: each float becomes the shortest
    decimal that round-trips it (``0.1`` is ``1/10``), not its binary value.

    :param obj: The numeric structure to wrap.
    :param \**hints: Backend-selection hints.
    """

    kind: ClassVar[str] = "numeric"
    _object: Any
    _structure: UnitcellStructure

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a numeric structure.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints.get("kind", "numeric") != "numeric":
            return None
        from httk.atomistic.storage.records import NumericUnitcellStructureRecord

        if not isinstance(obj, (NumericUnitcellStructure, NumericUnitcellStructureRecord)):
            try:
                cell = obj.cell
                sites = obj.sites
                obj.species  # noqa: B018 - deliberate attribute access validates the required field
                obj.species_at_sites  # noqa: B018 - deliberate attribute access validates the required field
            except AttributeError:
                return None
            if not isinstance(cell, NumericCell) or not isinstance(sites, NumericSites):
                return None
        return cls(obj, **hints)

    def __init__(self, obj: Any, **hints: Any) -> None:
        from httk.atomistic.storage.records import NumericUnitcellStructureRecord, numeric_structure_from_record

        self._object = obj
        if isinstance(obj, NumericUnitcellStructureRecord):
            obj = numeric_structure_from_record(obj)
        exact = getattr(obj, "exact", None)
        if isinstance(exact, UnitcellStructure):
            self._structure = exact
            return

        cell = getattr(obj.cell, "exact", None)
        if not isinstance(cell, Cell):
            cell = Cell(obj.cell.basis, precision=obj.cell.precision, periodicity=obj.cell.periodicity)
        sites = getattr(obj.sites, "exact", None)
        if not isinstance(sites, Sites):
            sites = Sites(obj.sites.reduced_coords, precision=obj.sites.precision)

        raw_charge = getattr(obj, "charge", None)
        site_moments = getattr(obj, "site_moments", None)
        if isinstance(site_moments, NumericSiteMoments):
            site_moments = _exact_site_moments(site_moments)
        self._structure = UnitcellStructure(
            cell,
            sites,
            obj.species,
            obj.species_at_sites,
            site_moments=site_moments,
            molecular=getattr(obj, "molecular", False),
            assemblies=getattr(obj, "assemblies", None),
            symmetry=getattr(obj, "symmetry", None),
            chemical_composition=getattr(obj, "chemical_composition", None),
            chemical_formula_descriptive=getattr(obj, "chemical_formula_descriptive", None),
            chemical_formula_hill=getattr(obj, "chemical_formula_hill", None),
            optimization_type=getattr(obj, "optimization_type", None),
            immutable_id=getattr(obj, "immutable_id", None),
            last_modified=getattr(obj, "last_modified", None),
            charge=None if raw_charge is None else any_to_fraction(raw_charge),
        )

    @property
    def cell(self) -> Cell:
        """Expose the exact cell quartet component."""
        return self._structure.cell

    @property
    def sites(self) -> Sites:
        """Expose the exact sites quartet component."""
        return self._structure.sites

    @property
    def species(self) -> tuple[Species, ...]:
        """Expose the exact structure's distinct species."""
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose the exact species name for each site."""
        return self._structure.species_at_sites

    @property
    def charge(self) -> Fraction | None:
        """Expose the explicitly assigned exact charge."""
        return self._structure.charge

    def unwrap(self) -> Any:
        """Return the adopted numeric object."""
        return self._object


def _delegated(name: str) -> property:
    return property(lambda self: getattr(self._structure, name), doc=f"Expose the exact structure's ``{name}``.")


# The StructureAPI defaults (no moments, no stated formulas) would otherwise hide what the
# exact quartet carries, so every stated field reads through to it.
for _name in (
    "site_moments",
    "molecular",
    "assemblies",
    "symmetry",
    "chemical_composition",
    "chemical_formula_descriptive",
    "chemical_formula_hill",
    "optimization_type",
    "immutable_id",
    "last_modified",
):
    setattr(_NumericStructureBackend, _name, _delegated(_name))
del _name
