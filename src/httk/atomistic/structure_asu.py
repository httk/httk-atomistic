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

from .asu_structure import ASUStructure
from .cell import Cell
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend


class StructureASU(StructureBackend):
    """Backend for a crystal structure carried as its asymmetric unit.

    ``cell`` and ``species`` come straight off the ASU. ``sites`` and ``species_at_sites``
    are the expansion, generated on first access and cached by the ASUStructure itself.
    """

    _asu: ASUStructure

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if not isinstance(obj, ASUStructure):
            return None
        if hints and hints.get("kind", "asu") != "asu":
            return None
        return super().__new__(cls)

    def __init__(self, obj: ASUStructure, **hints: Any) -> None:
        self._asu = obj

    @property
    def cell(self) -> Cell:
        return self._asu.cell

    @property
    def sites(self) -> Sites:
        return self._asu.expand_sites()

    @property
    def species(self) -> tuple[Species, ...]:
        return self._asu.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._asu.expand_species_at_sites()

    @property
    def asu(self) -> ASUStructure:
        """The underlying asymmetric unit, so a view can adopt it without re-deriving it."""
        return self._asu

    def unwrap(self) -> Any:
        return self._asu
