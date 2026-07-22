"""
The Simple structure representation for httk-atomistic.
"""

from collections.abc import Sequence
from typing import Any

from .species import Species


class Structure:
    """
    A crystal structure in the Simple representation.

    A Structure holds a ``basis`` (3x3 cell vectors), ``sites`` (Nx3 reduced
    coordinates), a list of ``species`` (each a ``Species``), and a length-N
    ``species_at_sites`` giving the species name occupying each site. Inputs are
    normalized on construction: species entries are passed through
    ``Species.create``, and every ``species_at_sites`` name must match one of the
    (uniquely named) species.

    Note: the numeric values (basis, sites) are stored as interim nested tuples of
    floats. They are intended to be replaced by the httk exact vector representation
    fairly soon; keep numeric access behind the quartet accessors so that change
    stays contained.
    """

    _basis: tuple[tuple[float, ...], ...]
    _sites: tuple[tuple[float, ...], ...]
    _species: tuple[Species, ...]
    _species_at_sites: tuple[str, ...]

    def __init__(
        self,
        basis: Sequence[Sequence[float]],
        sites: Sequence[Sequence[float]],
        species: Sequence[Species | dict[str, Any]],
        species_at_sites: Sequence[str],
    ) -> None:
        norm_basis = tuple(tuple(float(x) for x in row) for row in basis)
        norm_sites = tuple(tuple(float(x) for x in row) for row in sites)
        norm_species = tuple(Species.create(s) for s in species)
        norm_species_at_sites = tuple(str(name) for name in species_at_sites)

        if len(norm_basis) != 3 or any(len(row) != 3 for row in norm_basis):
            raise ValueError("Structure basis must be a 3x3 sequence")
        if any(len(row) != 3 for row in norm_sites):
            raise ValueError("Structure sites must be a sequence of length-3 coordinates")
        if len(norm_species_at_sites) != len(norm_sites):
            raise ValueError("Structure species_at_sites must have the same length as sites")

        names = [s.name for s in norm_species]
        if len(names) != len(set(names)):
            raise ValueError("Structure species names must be unique")
        known = set(names)
        for name in norm_species_at_sites:
            if name not in known:
                raise ValueError(f"Structure species_at_sites references unknown species name: {name!r}")

        self._basis = norm_basis
        self._sites = norm_sites
        self._species = norm_species
        self._species_at_sites = norm_species_at_sites

    @property
    def basis(self) -> tuple[tuple[float, ...], ...]:
        """The 3x3 cell vectors as nested float tuples."""
        return self._basis

    @property
    def sites(self) -> tuple[tuple[float, ...], ...]:
        """The Nx3 reduced site coordinates as nested float tuples."""
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct species of this structure."""
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site, in site order."""
        return self._species_at_sites

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return NotImplemented
        return (
            self._basis == other._basis
            and self._sites == other._sites
            and self._species == other._species
            and self._species_at_sites == other._species_at_sites
        )

    def __repr__(self) -> str:
        return (
            f"Structure(basis={self._basis!r}, sites={self._sites!r}, "
            f"species={self._species!r}, species_at_sites={self._species_at_sites!r})"
        )
