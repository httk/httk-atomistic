"""The immutable assigned-species Wyckoff-only classification value."""

from collections.abc import Sequence
from typing import Any, ClassVar

from httk.atomistic.models.bareprotostructure.backend import BareProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.prototype.notation import _known, _parse_undetermined, _undetermined, _unknown
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
    _undetermined: str | None = None

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

    @classmethod
    def undetermined(cls, label: str) -> "BareProtostructure":
        """Return the undetermined bare protostructure of a label in which ``X`` marks what is not determined.

        The canonicalisers return such a value when a part did not finish within its time budget; it
        holds only its label, ``<Formulatype>_X:<names>`` or
        ``<Formulatype>_<Pearson>_<number>_X:<names>``. Two undetermined values are equal when their
        labels are; that does not mean their structures are the same.

        :param label: The undetermined label.
        :return: The undetermined value, with ``determined`` false.
        :raises ValueError: If ``label`` is not an undetermined bare protostructure label.
        """
        _parse_undetermined(label, cls.kind)
        return BareProtostructure._undetermined_of(label)

    @classmethod
    def _undetermined_of(cls, label: str) -> "BareProtostructure":
        """Build an undetermined value from a trusted label, without parsing it."""
        instance = object.__new__(BareProtostructure)
        instance._undetermined = label
        return instance

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "spacegroup", undetermined.spacegroup)
        return self._spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the canonical occupied Wyckoff positions."""
        if _undetermined(self) is not None:
            _unknown(self, "occupations")
        return self._occupations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BareProtostructure):
            return NotImplemented
        if self._undetermined is not None or other._undetermined is not None:
            return self._undetermined == other._undetermined
        return (self.spacegroup, self.occupations) == (other.spacegroup, other.occupations)

    def __hash__(self) -> int:
        if self._undetermined is not None:
            return hash(self._undetermined)
        return hash((self.spacegroup, self.occupations))

    def __repr__(self) -> str:
        if self._undetermined is not None:
            return f"BareProtostructure.undetermined({self._undetermined!r})"
        pairs = ", ".join(f"{value.wyckoff}:{value.species.name}" for value in self.occupations)
        return f"BareProtostructure({self.spacegroup.setting!r}, {pairs})"
