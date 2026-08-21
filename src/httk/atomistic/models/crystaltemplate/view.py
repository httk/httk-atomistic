"""Lazy crystal-template presentation view."""

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.crystaltemplate.backend import CrystalTemplateBackend
from httk.atomistic.models.crystaltemplate.crystaltemplate import CrystalTemplate
from httk.atomistic.models.crystaltemplate.view_base import CrystalTemplateViewBase
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species

if TYPE_CHECKING:
    from httk.atomistic.models.crystaltemplate.like import CrystalTemplateLike
    from httk.atomistic.models.structure.like import StructureLike


class CrystalTemplateView(CrystalTemplateViewBase, CrystalTemplate):
    r"""Present a crystal template or ordinary structure lazily as a crystal template.

    :param obj: The crystal-template-like or structure-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: CrystalTemplateBackend

    def __new__(cls, obj: "CrystalTemplateLike | StructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def _fill_cell(self) -> None:
        object.__setattr__(self, "_cell", self._backend.cell)

    def _fill_sites(self) -> None:
        object.__setattr__(self, "_sites", self._backend.sites)

    def _fill_species(self) -> None:
        object.__setattr__(self, "_species", self._backend.species)

    def _fill_species_at_sites(self) -> None:
        object.__setattr__(self, "_species_at_sites", self._backend.species_at_sites)

    @cached_property
    def _cell(self) -> Cell:  # type: ignore[override]
        self._fill_cell()
        return self.__dict__["_cell"]

    @cached_property
    def _sites(self) -> Sites:  # type: ignore[override]
        self._fill_sites()
        return self.__dict__["_sites"]

    @cached_property
    def _species(self) -> tuple[Species, ...]:  # type: ignore[override]
        self._fill_species()
        return self.__dict__["_species"]

    @cached_property
    def _species_at_sites(self) -> tuple[str, ...]:  # type: ignore[override]
        self._fill_species_at_sites()
        return self.__dict__["_species_at_sites"]

    @property
    def periodicity(self):
        """Return the presented periodic directions."""
        return self.cell.periodicity

    @property
    def nperiodic_dimensions(self):
        """Return the number of presented periodic directions."""
        return self.cell.nperiodic_dimensions

    @property
    def nsites(self):
        """Return the number of presented sites."""
        return len(self.sites)

    def cartesian_sites(self):
        """Return the exact Cartesian presented site positions."""
        from httk.core import SurdVector

        return SurdVector(self.sites.reduced_coords) * self.cell.basis

    @property
    def coordinate_precision(self):
        """Return the presented coordinate precision."""
        return self.sites.precision

    @property
    def basis_precision(self):
        """Return the presented basis precision."""
        return self.cell.precision

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> CrystalTemplate:
        """Return the presented structure as a standalone value.

        :return: The crystal-template value.
        """
        if type(self._backend) is CrystalTemplate:
            return self._backend
        return CrystalTemplate(self.cell, self.sites, self.species, self.species_at_sites)
