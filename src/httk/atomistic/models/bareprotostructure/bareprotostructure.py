"""The immutable assigned-species Wyckoff-only classification value."""

from collections.abc import Sequence
from typing import Any, ClassVar

from httk.atomistic.models.bareprotostructure.backend import BareProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


class BareProtostructure(BareProtostructureBackend):
    """Store only standard-setting Wyckoff occupations, without geometrical refinement.

    Occupations carry no cell: they are read in a right-handed basis of the standard setting,
    so in a Sohncke group they name one hand of the crystal.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their assigned species, each stored as
        its species class (see
        :class:`~httk.atomistic.models.protostructure.occupation.WyckoffOccupation`).
    """

    kind: ClassVar[str] = "bare_protostructure"

    def __init__(
        self, spacegroup: Spacegroup | int, occupations: Sequence[WyckoffOccupation | tuple[str, Any]]
    ) -> None:
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"BareProtostructure records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) "
                "instead"
            )
        raw = tuple(
            occupation if isinstance(occupation, WyckoffOccupation) else WyckoffOccupation(occupation[0], occupation[1])
            for occupation in occupations
        )
        if not raw:
            raise ValueError("BareProtostructure occupations must be non-empty")
        for occupation in raw:
            try:
                self._spacegroup.wyckoff_position(occupation.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
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
        if not isinstance(other, BareProtostructure):
            return NotImplemented
        return (self.spacegroup, self.occupations) == (other.spacegroup, other.occupations)

    def __hash__(self) -> int:
        return hash((self.spacegroup, self.occupations))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{value.wyckoff}:{value.species.name}" for value in self.occupations)
        return f"BareProtostructure({self.spacegroup.setting!r}, {pairs})"
