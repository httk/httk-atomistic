"""A plain-numpy presentation of a structure backend."""

from functools import cached_property
from typing import Any, Self, cast

from httk.core import NumericVector, to_numeric, unwrap

from httk.atomistic.composition import Assembly
from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.numeric import NumericCell
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import StructureSemanticsMixin
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.view import StructureView


class NumericUnitcellStructureView(StructureSemanticsMixin, StructureView):
    """A plain-numpy presentation of a :class:`~httk.atomistic.UnitcellStructure`.

    Where a ``UnitcellStructure`` holds its geometry exactly (a surd ``cell`` basis, rational reduced
    coordinates, and an exact Cartesian frame), this view mirrors that interface but returns
    plain numpy numbers: its :attr:`cell` is a :class:`~httk.atomistic.models.cell.numeric.NumericCell`, its
    :attr:`sites` a :class:`~httk.atomistic.models.sites.numeric.NumericSites`, and :meth:`cartesian_sites` a
    ``float64`` numpy array. The ``species``/``species_at_sites`` are passed through unchanged.
    It is for callers who do not need exact arithmetic and just want numpy arrays.

    The presentation is numpy-backed, so constructing it **requires numpy** (the
    ``httk-atomistic[numpy]`` extra) and raises :class:`ImportError` eagerly when it is unavailable.
    The exact object is always one hop away via :attr:`exact`.

    This is a view, not a ``UnitcellStructure`` subclass. Its exact ``UnitcellStructure`` is built lazily on
    first access to exact geometry.
    """

    _backend: StructureBackend

    def __new__(cls, obj: StructureLike, **hints: Any) -> Self:
        require_numpy()
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **hints: Any) -> None:
        pass

    @cached_property
    def _exact(self) -> UnitcellStructure:
        from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

        return UnitcellStructureView(self._backend)

    @property
    def cell(self) -> NumericCell:
        """The cell as a :class:`~httk.atomistic.models.cell.numeric.NumericCell`."""
        return NumericCell(self._exact.cell)

    @property
    def sites(self) -> NumericSites:
        """The sites as a :class:`~httk.atomistic.models.sites.numeric.NumericSites`."""
        return NumericSites(self._exact.sites)

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct species, passed through unchanged."""
        return self._exact.species if getattr(self._backend, "resolve", None) is not None else self._backend.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site, passed through unchanged."""
        return (
            self._exact.species_at_sites
            if getattr(self._backend, "resolve", None) is not None
            else self._backend.species_at_sites
        )

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        return self._exact.assemblies

    def cartesian_sites(self) -> NumericVector:
        """The Cartesian site positions as an ``(N, 3)`` ``float64`` numpy array."""
        return to_numeric(self._exact.cartesian_sites())

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        return self._exact.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        return self._exact.nperiodic_dimensions

    @property
    def site_coordinate_span(self) -> str:
        return self._exact.site_coordinate_span

    @property
    def lattice_vectors(self) -> list[list[float]]:
        return cast(list[list[float]], cast(Any, self.cell.basis).tolist())

    @property
    def fractional_site_positions(self) -> list[list[float]]:
        return cast(list[list[float]], cast(Any, self.sites.reduced_coords).tolist())

    @property
    def cartesian_site_positions(self) -> list[list[float]]:
        return cast(list[list[float]], cast(Any, self.cartesian_sites()).tolist())

    @property
    def exact(self) -> UnitcellStructure:
        """The exact :class:`~httk.atomistic.UnitcellStructure` this view presents."""
        return self._exact

    def __repr__(self) -> str:
        return (
            f"NumericUnitcellStructureView(cell={self.cell!r}, sites={self.sites!r}, "
            f"species={self.species!r}, species_at_sites={self.species_at_sites!r})"
        )

    def unwrap(self) -> Any:
        """Return the raw object wrapped by the backend."""
        return unwrap(self._backend)
