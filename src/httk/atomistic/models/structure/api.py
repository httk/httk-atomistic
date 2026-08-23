"""
The minimal canonical structure interface for httk-atomistic.
"""

from abc import ABC, abstractmethod
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, cast

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species

if TYPE_CHECKING:
    from httk.atomistic.models.formula.composition import Composition
    from httk.atomistic.models.protostructure.protostructure import Protostructure
    from httk.atomistic.models.prototemplate.prototemplate import Prototemplate
    from httk.atomistic.models.structure.like import StructureLike


class StructureAPI(ABC):
    """Define the canonical structure interface.

    It declares the Unitcell quartet that every structure backend produces from its
    own native representation and every structure view builds its presentation
    from: ``cell``, ``sites``, ``species``, and ``species_at_sites``. This is the
    single interchange format; there is no pairwise conversion between backends.
    """

    @property
    @abstractmethod
    def cell(self) -> Cell:
        """Expose the structure's cell."""
        raise NotImplementedError

    @property
    @abstractmethod
    def sites(self) -> Sites:
        """Expose the structure's site coordinates."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        """Expose the structure's distinct species."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose the species occupying each site."""
        raise NotImplementedError

    @property
    def charge(self) -> Fraction | None:
        """Expose the explicitly assigned net charge of the cell content.

        ``None`` means unstated and is never derived from the species; it is distinct
        from an explicit zero.

        :return: The assigned charge, or ``None`` when it is unstated.
        """
        return None

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        """Expose optional per-site magnetic moments in ``sites`` order.

        ``None`` means "nothing stated", not "zero moments".

        :return: The site moments, or ``None`` when they are unstated.
        """
        return None

    def without_charges(self) -> "StructureAPI":
        """Return an EXPLICIT lossy projection that drops declared oxidation states.

        The canonical structure components and semantic metadata are preserved. A structure
        without species charges is returned by identity; charged structures are rebuilt in the
        canonical unit-cell family.

        :return: A charge-free structure, or this structure when already charge-free.
        """
        if not any(species.charges is not None for species in self.species):
            return self
        from httk.atomistic.models.structure.unitcell import UnitcellStructure

        return UnitcellStructure(
            self.cell,
            self.sites,
            tuple(species.without_charges() for species in self.species),
            self.species_at_sites,
            site_moments=self.site_moments,
            molecular=getattr(self, "molecular", False),
            assemblies=getattr(self, "assemblies", None),
            symmetry=getattr(self, "symmetry", None),
            chemical_composition=getattr(self, "chemical_composition", None),
            chemical_formula_descriptive=getattr(self, "chemical_formula_descriptive", None),
            chemical_formula_hill=getattr(self, "chemical_formula_hill", None),
            optimization_type=getattr(self, "optimization_type", None),
            immutable_id=getattr(self, "immutable_id", None),
            last_modified=getattr(self, "last_modified", None),
            charge=self.charge,
        )

    def canonical_protostructure(self) -> "Protostructure":
        """Return the canonical geometry-free description of this structure.

        Enantiomorphic structures are deliberately collapsed to the lower-numbered
        member of their space-group pair, so the result is independent of chirality.

        :return: A standalone canonical protostructure value.
        """
        from httk.atomistic.models.protostructure.view import ProtostructureView
        from httk.atomistic.symmetry.canonical import canonical_asu

        canonical = canonical_asu(cast("StructureLike", self), preserve_chirality=False)
        return ProtostructureView(canonical).unview()

    def canonical_prototemplate(self) -> "Prototemplate":
        """Return the canonical anonymous geometry-free template of this structure.

        Enantiomorphic structures are deliberately collapsed to the lower-numbered
        member of their space-group pair, so the result is independent of chirality.

        :return: A standalone canonical prototemplate value.
        """
        from httk.atomistic.models.prototemplate.view import PrototemplateView
        from httk.atomistic.symmetry.canonical import canonical_asu

        canonical = canonical_asu(cast("StructureLike", self), preserve_chirality=False)
        return PrototemplateView(canonical).unview()

    @cached_property
    def composition(self) -> "Composition":
        """Project the canonical components into an elemental composition."""
        from httk.atomistic.composition import project_composition

        return project_composition(self)

    @property
    def elements(self) -> tuple[str, ...] | None:
        """Expose the complete composition's element symbols, if available."""
        composition = self.composition
        return composition.elements if composition.complete else None

    @property
    def nelements(self) -> int | None:
        """Expose the complete composition's element count, if available."""
        composition = self.composition
        return composition.nelements if composition.complete else None

    @property
    def elements_ratios(self) -> tuple[Fraction, ...] | None:
        """Expose complete composition ratios, if available."""
        composition = self.composition
        return composition.elements_ratios if composition.complete else None

    @property
    def chemical_formula_reduced(self) -> str | None:
        """Expose the reduced formula derived from a complete composition."""
        return self.composition.chemical_formula_reduced

    @property
    def chemical_formula_anonymous(self) -> str | None:
        """Expose the anonymous formula derived from a complete composition."""
        return self.composition.chemical_formula_anonymous

    @property
    def chemical_formula_descriptive(self) -> str | None:
        """Expose an explicitly supplied descriptive formula, when available."""
        return None

    @property
    def chemical_formula_hill(self) -> str | None:
        """Expose an explicitly supplied Hill formula, when available."""
        return None

    @property
    def dimension_types(self) -> tuple[int, ...] | None:
        """Expose cell periodicity as OPTIMADE dimension flags."""
        return cast(tuple[int, int, int], tuple(1 if value else 0 for value in self.cell.periodicity))

    @property
    def nperiodic_dimensions(self) -> int | None:
        """Expose the number of periodic cell directions."""
        return self.cell.nperiodic_dimensions

    @property
    def nsites(self) -> int | None:
        """Expose the number of canonical site-coordinate rows."""
        return self.sites.num_sites

    def multiplicities(self) -> tuple[int, ...]:
        """Return how many unit-cell sites each represented site contributes.

        The canonical structure interface presents a unit cell, so each row contributes
        once. Symmetry-reduced representations override this with their orbit counts.

        :return: One multiplicity per represented site.
        """
        return (1,) * len(self.species_at_sites)

    @property
    def implicit_atoms(self) -> tuple[str, ...]:
        """Expose species definitions whose atoms have no represented coordinates.

        :return: Unused species names in species-definition order.
        """
        wyckoff_sites = getattr(self, "wyckoff_sites", None)
        names = tuple(site.species for site in wyckoff_sites) if wyckoff_sites is not None else self.species_at_sites
        used = set(names)
        return tuple(species.name for species in self.species if species.name not in used)

    @property
    def structure_features(self) -> tuple[str, ...] | None:
        """Expose composition-related features derived from canonical components."""
        from httk.atomistic.composition import derive_structure_features

        return derive_structure_features(self)
