"""ASE ``Atoms`` view for the :mod:`httk.atomistic` structure family.

ASE is an optional dependency. This module intentionally imports it unconditionally,
so the package-level exports can guard the import while documentation tools can see
the public :class:`ASEAtomsView` definition.
"""

from typing import TYPE_CHECKING, Any, Self, cast

import ase
from httk.core import register_citation, unwrap

register_citation(
    applies_to="Structure interchange with the Atomic Simulation Environment (ASE)",
    references={
        "authors": (
            {"name": "Ask Hjorth Larsen"},
            {"name": "Jens Jørgen Mortensen"},
            {"name": "Jakob Blomqvist"},
            {"name": "Ivano E. Castelli"},
            {"name": "Rune Christensen"},
            {"name": "Marcin Dułak"},
            {"name": "Jesper Friis"},
            {"name": "Michael N. Groves"},
            {"name": "Bjørk Hammer"},
            {"name": "Cory Hargus"},
            {"name": "and others"},
        ),
        "title": "The atomic simulation environment—a Python library for working with atoms",
        "journal": "Journal of Physics: Condensed Matter",
        "volume": "29",
        "pages": "273002",
        "year": "2017",
        "doi": "10.1088/1361-648X/aa680e",
        "bib_type": "article",
    },
)

from httk.atomistic.elements import atomic_number
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structure.view import StructureView

if TYPE_CHECKING:
    from httk.atomistic.models.structure.like import StructureLike


class ASEAtomsView(StructureView, ase.Atoms):
    r"""Present a structure-like value as ASE ``Atoms``.

    The conversion carries the structure quartet: cell, reduced positions, atomic
    numbers, and periodicity. ASE must be installed to construct this view. Mixed or
    attached species cannot be represented by ASE atomic numbers and raise
    :class:`TypeError`.

    Site moments map to ``initial_magmoms`` and species charges map to
    ``initial_charges``. Structure charge, species spins, labels, masses, assemblies,
    declared chemical composition, and non-element species are rejected rather than
    discarded; :meth:`unwrap` recovers the original value behind the backend.

    :param obj: A structure-like value to present as ``Atoms``.
    :param \**hints: Backend-selection hints.
    """

    _backend: StructureBackend

    def __new__(cls, obj: "StructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        structure = UnitcellStructureView(backend)
        if structure.assemblies is not None:
            raise TypeError("This structure cannot be represented as ASE Atoms because it has assemblies")
        if structure.chemical_composition is not None:
            raise TypeError(
                "This structure cannot be represented as ASE Atoms because it has a declared chemical composition"
            )
        source = unwrap(backend)
        charge = getattr(source, "charge", getattr(backend, "charge", None))
        if charge is not None:
            raise ValueError("This structure cannot be represented as ASE Atoms because it has a charge")
        species_values = getattr(source, "species", getattr(backend, "species", ()))
        for item in species_values:
            for field in ("spins", "labels", "mass"):
                if getattr(item, field, None) is not None:
                    raise ValueError(f"This structure cannot be represented as ASE Atoms because species has {field}")
        instance = super().__new__(cls)
        species_by_name = {species.name: species for species in structure.species}
        numbers: list[int] = []
        for name in structure.species_at_sites:
            species = species_by_name[name]
            if not species.is_single_element:
                raise TypeError(
                    "This structure cannot be represented as ASE Atoms "
                    f"(species {name!r} is not a single, unattached chemical element)"
                )
            numbers.append(atomic_number(species.chemical_symbols[0]))
        ase.Atoms.__init__(
            instance,
            cell=structure.cell.basis.to_floats(),
            scaled_positions=structure.sites.reduced_coords.to_floats(),
            numbers=numbers,
            pbc=structure.cell.periodicity,
        )
        moments = structure.site_moments
        if moments is not None:
            if isinstance(moments, CollinearSiteMoments):
                instance.set_initial_magnetic_moments([float(value) for value in moments.collinear_moments.to_floats()])
            else:
                cartesian = moments if isinstance(moments, CartesianSiteMoments) else CartesianSiteMomentsView(moments)
                instance.set_initial_magnetic_moments(
                    [[float(value) for value in row] for row in cartesian.cartesian_moments.to_floats()]
                )
        if any(species.charges is not None for species in structure.species):
            initial_charges: list[float] = []
            for name in structure.species_at_sites:
                charges = species_by_name[name].charges
                charge = None if charges is None else charges[0]
                initial_charges.append(0.0 if charge is None else float(cast(Any, charge)))
            cast(Any, instance).set_initial_charges(initial_charges)
        instance._backend = backend
        return instance

    def __init__(self, obj: "StructureLike", **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        """Return the original value represented by the underlying backend."""
        return unwrap(self._backend)

    def unview(self) -> ase.Atoms:
        """Return a base ``Atoms`` copy of this view.

        :return: A standalone ASE ``Atoms`` object.
        """
        # ase.Atoms(other) builds a base-class copy of the presented atoms.
        return ase.Atoms(self)


__all__ = ["ASEAtomsView"]
