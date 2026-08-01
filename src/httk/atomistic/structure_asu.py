"""Backend presenting an ASUStructure through the common structure interface.

Registering this makes an :class:`~httk.atomistic.ASUStructure` usable anywhere a
structure is accepted: ``UnitcellStructureView(asu)`` is the full unit cell, and the
OPTIMADE provider, the numeric layer, and everything else follow without changes.

The expansion is lazy. Building an ASUStructure, inspecting its space group, or writing it
back out never generates the cell.
For ``UnitcellStructureView(asu)``, expansion is deferred until the sites or per-site species
assignment are read; cell, species, and space-group access never expands it.
"""

from typing import Any

from .asu_structure import FundamentalDomainStructure
from .cell import Cell
from .composition import Assembly
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend


class StructureASU(StructureBackend):
    """Backend for a crystal structure carried as its asymmetric unit.

    ``cell`` and ``species`` come straight off the ASU. ``sites`` and ``species_at_sites``
    are the expansion, generated on first access and cached by the ASUStructure itself.
    """

    _asu: FundamentalDomainStructure

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if not isinstance(obj, FundamentalDomainStructure):
            return None
        if hints and hints.get("kind", "asu") != "asu":
            return None
        return super().__new__(cls)

    def __init__(self, obj: FundamentalDomainStructure, **hints: Any) -> None:
        self._asu = obj

    def _expanded_offsets(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        counts = self._asu.multiplicities()
        offsets: list[int] = []
        offset = 0
        for count in counts:
            offsets.append(offset)
            offset += count
        return counts, tuple(offsets)

    def _expanded_assemblies(self) -> tuple[Assembly, ...] | None:
        assemblies = self._asu.assemblies
        if assemblies is None or not assemblies:
            return assemblies
        counts, offsets = self._expanded_offsets()
        expanded: list[Assembly] = []
        for assembly in assemblies:
            groups: list[tuple[int, ...]] = []
            for group in assembly.sites_in_groups:
                if any(counts[index] != 1 for index in group):
                    raise ValueError(
                        "symmetry-reduced expansion cannot map assembly correlations "
                        "when a correlated domain site has multiple unit-cell images"
                    )
                groups.append(tuple(offsets[index] for index in group))
            expanded.append(
                Assembly(
                    tuple(groups),
                    assembly.group_probabilities,
                    assembly.group_probabilities_precision,
                )
            )
        return tuple(expanded)

    def _validate_expansion_semantics(self) -> None:
        # Computing the remapped assemblies is itself the exact ambiguity check.
        self._expanded_assemblies()
        if not self._asu.molecular:
            return
        counts = self._asu.multiplicities()
        if any(count != 1 for count in counts) or any(site.representative is None for site in self._asu.asu_sites):
            raise ValueError(
                "symmetry-reduced molecular expansion requires one retained representative "
                "for every one-to-one domain site"
            )

    @property
    def cell(self) -> Cell:
        return self._asu.cell

    @property
    def sites(self) -> Sites:
        self._validate_expansion_semantics()
        # In the only unambiguous molecular case the retained representatives are the
        # asserted placement; do not replace them by symmetry-snapped coordinates.
        return self._asu.sites if self._asu.molecular else self._asu.expand_sites()

    @property
    def species(self) -> tuple[Species, ...]:
        return self._asu.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        self._validate_expansion_semantics()
        return self._asu.species_at_sites if self._asu.molecular else self._asu.expand_species_at_sites()

    @property
    def molecular(self) -> bool:
        return self._asu.molecular

    @property
    def assemblies(self) -> Any:
        return self._expanded_assemblies()

    @property
    def chemical_composition(self) -> Any:
        return self._asu.chemical_composition

    @property
    def chemical_formula_descriptive(self) -> str | None:
        return self._asu.chemical_formula_descriptive

    @property
    def chemical_formula_hill(self) -> str | None:
        return self._asu.chemical_formula_hill

    @property
    def optimization_type(self) -> str | None:
        return self._asu.optimization_type

    @property
    def asu(self) -> FundamentalDomainStructure:
        """The underlying asymmetric unit, so a view can adopt it without re-deriving it."""
        return self._asu

    def unwrap(self) -> Any:
        return self._asu
