"""The immutable anonymous Wyckoff-only prototype value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.notation import (
    _known,
    _parse_undetermined,
    _undetermined,
    _unknown,
    canonical_label_map,
)
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


class BarePrototype(BarePrototypeBackend):
    """Store only standard-setting Wyckoff occupations, without geometrical refinement.

    Occupations carry no cell: they are read in a right-handed basis of the standard setting,
    so in a Sohncke group they name one hand of the crystal.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their anonymous labels.
    """

    kind: ClassVar[str] = "bare_prototype"
    _undetermined: str | None = None

    def __init__(
        self, spacegroup: Spacegroup | int, occupations: Sequence[PrototypeOccupation | tuple[str, str]]
    ) -> None:
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError("BarePrototype records Wyckoff data in the IT standard setting")
        raw = tuple(
            value if isinstance(value, PrototypeOccupation) else PrototypeOccupation(*value) for value in occupations
        )
        if not raw:
            raise ValueError("BarePrototype occupations must be non-empty")
        letters_by_label: dict[str, list[str]] = {}
        for value in raw:
            try:
                self._spacegroup.wyckoff_position(value.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            letters_by_label.setdefault(value.label, []).append(value.wyckoff)
        expected = {anonymous_symbol(index) for index in range(len(letters_by_label))}
        if set(letters_by_label) != expected:
            raise ValueError("BarePrototype class labels must be consecutive anonymous symbols from 'A'")
        relabel = canonical_label_map(
            self._spacegroup, {label: tuple(sorted(letters)) for label, letters in letters_by_label.items()}
        )
        self._occupations = tuple(
            sorted(
                (PrototypeOccupation(value.wyckoff, relabel[value.label]) for value in raw),
                key=lambda value: (value.label, value.wyckoff),
            )
        )

    @classmethod
    def undetermined(cls, label: str) -> "BarePrototype":
        """Return the undetermined bare prototype of a label in which ``X`` marks what is not determined.

        The canonicalisers return such a value when a part did not finish within its time budget; it
        holds only its label, ``<Formulatype>_X`` or ``<Formulatype>_<Pearson>_<number>_X``. Two
        undetermined values are equal when their labels are; that does not mean their structures are
        the same.

        :param label: The undetermined label.
        :return: The undetermined value, with ``determined`` false.
        :raises ValueError: If ``label`` is not an undetermined bare prototype label.
        """
        _parse_undetermined(label, cls.kind)
        return BarePrototype._undetermined_of(label)

    @classmethod
    def _undetermined_of(cls, label: str) -> "BarePrototype":
        """Build an undetermined value from a trusted label, without parsing it."""
        instance = object.__new__(BarePrototype)
        instance._undetermined = label
        return instance

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "spacegroup", undetermined.spacegroup)
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototypeOccupation, ...]:
        """Return canonical anonymous Wyckoff occupations."""
        if _undetermined(self) is not None:
            _unknown(self, "occupations")
        return self._occupations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BarePrototype):
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
            return f"BarePrototype.undetermined({self._undetermined!r})"
        pairs = ", ".join(f"{value.wyckoff}:{value.label}" for value in self.occupations)
        return f"BarePrototype({self.spacegroup.setting!r}, {pairs})"
