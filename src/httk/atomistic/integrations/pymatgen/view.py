"""Pymatgen ``Structure`` view for the :mod:`httk.atomistic` structure family."""

from typing import TYPE_CHECKING, Any, Self, cast

import pymatgen.core
from httk.core import register_citation, unwrap

register_citation(
    applies_to="Structure interchange with pymatgen",
    references={
        "authors": (
            {"name": "Shyue Ping Ong"},
            {"name": "William Davidson Richards"},
            {"name": "Anubhav Jain"},
            {"name": "Geoffroy Hautier"},
            {"name": "Michael Kocher"},
            {"name": "Shreyas Cholia"},
            {"name": "Dan Gunter"},
            {"name": "Vincent Chevrier"},
            {"name": "Kristin A. Persson"},
            {"name": "Gerbrand Ceder"},
        ),
        "title": "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis",
        "journal": "Computational Materials Science",
        "volume": "68",
        "pages": "314-319",
        "year": "2013",
        "doi": "10.1016/j.commatsci.2012.10.028",
        "bib_type": "article",
    },
)

from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView

if TYPE_CHECKING:
    from httk.atomistic.models.structure.like import StructureLike


class PymatgenStructureView(StructureView, pymatgen.core.Structure):
    r"""Present a structure-like value as a pymatgen ``Structure``.

    Charges, spins, partial occupancy, dummy labels, and structure charge are
    exported, with structure charge passed as an exact ``Fraction``. Vacancy
    constituents created for occupancy shortfall are omitted because pymatgen
    represents the shortfall directly. Masses, labels on elements, attached species,
    assemblies, and declared chemical composition are rejected rather than discarded;
    :meth:`unwrap` recovers the original value behind the backend.

    :param obj: A structure-like value to present as ``Structure``.
    :param \**hints: Backend-selection hints.
    """

    _backend: StructureBackend
    _raw: Any

    def __new__(cls, obj: "StructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

        backend = cls._prepare_backend(obj, hints)
        structure = UnitcellStructureView(backend)
        for value in structure.species:
            if value.mass is not None:
                raise ValueError("pymatgen Structure cannot represent explicit constituent masses")
            for symbol, label in zip(value.chemical_symbols, value.labels or (None,) * len(value.chemical_symbols)):
                if label is not None and symbol != "X":
                    raise ValueError("pymatgen Structure cannot represent labels on elements")
                if label == "":
                    raise ValueError("pymatgen Structure cannot represent an empty label")
            if value.attached is not None:
                raise TypeError("pymatgen Structure cannot represent attached species")
            if value.nattached is not None:
                raise TypeError("pymatgen Structure cannot represent nattached species")
        if structure.assemblies is not None:
            raise TypeError("pymatgen Structure cannot represent assemblies")
        if structure.chemical_composition is not None:
            raise TypeError("pymatgen Structure cannot represent a declared chemical composition")

        species_by_name = {value.name: value for value in structure.species}
        site_species: list[dict[Any, Any]] = []
        for name in structure.species_at_sites:
            composition: dict[Any, Any] = {}
            for symbol, occupancy, charge, spin, label in zip(
                species_by_name[name].chemical_symbols,
                species_by_name[name].concentration,
                species_by_name[name].charges or (None,) * len(species_by_name[name].chemical_symbols),
                species_by_name[name].spins or (None,) * len(species_by_name[name].chemical_symbols),
                species_by_name[name].labels or (None,) * len(species_by_name[name].chemical_symbols),
            ):
                if symbol == "vacancy":
                    continue
                key: Any
                if symbol == "X":
                    dummy_symbol = "X" if label is None else "X" + label
                    key = pymatgen.core.DummySpecies(
                        dummy_symbol,
                        oxidation_state=cast(Any, charge),
                        spin=cast(Any, spin),
                    )
                    if key.symbol != dummy_symbol:
                        raise ValueError(f"pymatgen Structure cannot represent label {label!r}")
                elif charge is None and spin is None:
                    key = pymatgen.core.Element(symbol)
                else:
                    key = pymatgen.core.Species(
                        symbol,
                        oxidation_state=cast(Any, charge),
                        spin=cast(Any, spin),
                    )
                composition[key] = occupancy
            site_species.append(composition)

        site_properties: dict[str, Any] = {}
        moments = structure.site_moments
        if moments is not None:
            if isinstance(moments, CollinearSiteMoments):
                site_properties["magmom"] = [float(value) for value in moments.collinear_moments.to_floats()]
            else:
                cartesian = moments if isinstance(moments, CartesianSiteMoments) else CartesianSiteMomentsView(moments)
                site_properties["magmom"] = [
                    [float(value) for value in row] for row in cartesian.cartesian_moments.to_floats()
                ]

        instance = super().__new__(cls)
        pymatgen.core.Structure.__init__(
            instance,
            lattice=pymatgen.core.Lattice(structure.cell.basis.to_floats(), pbc=structure.cell.periodicity),
            species=site_species,
            coords=structure.sites.reduced_coords.to_floats(),
            charge=cast(Any, structure.charge),
            site_properties=site_properties or None,
        )
        instance._backend = backend
        instance._raw = unwrap(backend)
        return instance

    def __init__(self, obj: "StructureLike", **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        """Return the original value represented by the underlying backend."""
        return self._raw

    def unview(self) -> pymatgen.core.Structure:
        """Return a base ``Structure`` copy preserving exported properties.

        :return: A standalone pymatgen ``Structure`` object.
        """
        try:
            labels = self.labels
        except ValueError:
            # pymatgen 2026.3 formats unlabeled exact Fraction occupancies as floats
            # while computing labels; retain explicit labels without that lossy fallback.
            labels = [getattr(site, "_label", None) for site in self.sites]
        return pymatgen.core.Structure(
            lattice=self.lattice,
            species=self.species_and_occu,
            coords=self.frac_coords,
            charge=getattr(self, "_charge", None),
            site_properties=self.site_properties,
            labels=labels,
            properties=self.properties,
        )


__all__ = ["PymatgenStructureView"]
