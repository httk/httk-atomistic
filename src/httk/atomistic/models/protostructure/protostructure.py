"""The immutable geometry-free protostructure value."""

from collections.abc import Sequence
from typing import Any, ClassVar

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Protostructure(ProtostructureBackend):
    """Store a standard-setting space group and its occupied Wyckoff positions.

    This is a provenance-independent value: multiplicities, composition, and formula
    derivations are defined at the standard-setting conventional-cell scale, even when
    the source used to recognize it was stored in a volume-scaled setting.

    ``Protostructure`` is the assigned-species, Wyckoff-positions-only cell of the
    material-information matrix:

    ======================  ===============  ==============
    Geometrical info        Anonymous        Assigned
    ======================  ===============  ==============
    Wyckoff positions only  Prototemplate    Protostructure
    Geometrical class       Prototype        Structuretype
    Exact geometry          CrystalTemplate  Structure
    ======================  ===============  ==============

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: The occupied Wyckoff positions and their species.
    """

    _spacegroup: Spacegroup
    _occupations: tuple[WyckoffOccupation, ...]
    kind: ClassVar[str] = "protostructure"

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[WyckoffOccupation | tuple[str, Any]],
    ) -> None:
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"Protostructure records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) "
                "instead"
            )
        raw = tuple(
            occupation if isinstance(occupation, WyckoffOccupation) else WyckoffOccupation(occupation[0], occupation[1])
            for occupation in occupations
        )
        if not raw:
            raise ValueError("Protostructure occupations must be non-empty")
        species_by_name: dict[str, Any] = {}
        for occupation in raw:
            try:
                self._spacegroup.wyckoff_position(occupation.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            previous = species_by_name.get(occupation.species.name)
            if previous is not None and previous != occupation.species:
                raise ValueError(
                    f"Protostructure occupations naming species {occupation.species.name!r} must carry equal Species"
                )
            species_by_name[occupation.species.name] = occupation.species
        self._occupations = tuple(sorted(raw, key=lambda value: (value.species.name, value.wyckoff)))

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the canonical occupied Wyckoff positions."""
        return self._occupations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protostructure):
            return NotImplemented
        return (self._spacegroup, self._occupations) == (other._spacegroup, other._occupations)

    def __hash__(self) -> int:
        return hash((self._spacegroup, self._occupations))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{occupation.wyckoff}:{occupation.species.name}" for occupation in self._occupations)
        return f"Protostructure({self._spacegroup.setting!r}, {pairs})"
