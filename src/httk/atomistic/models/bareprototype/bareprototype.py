"""The immutable anonymous Wyckoff-only prototype value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.notation import canonical_label_map
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


class BarePrototype(BarePrototypeBackend):
    """Store only standard-setting Wyckoff occupations, without geometrical refinement.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: Occupied Wyckoff positions and their anonymous labels.
    """

    kind: ClassVar[str] = "bare_prototype"

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
        relabel = canonical_label_map({label: tuple(sorted(letters)) for label, letters in letters_by_label.items()})
        self._occupations = tuple(
            sorted(
                (PrototypeOccupation(value.wyckoff, relabel[value.label]) for value in raw),
                key=lambda value: (value.label, value.wyckoff),
            )
        )

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototypeOccupation, ...]:
        """Return canonical anonymous Wyckoff occupations."""
        return self._occupations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BarePrototype):
            return NotImplemented
        return (self.spacegroup, self.occupations) == (other.spacegroup, other.occupations)

    def __hash__(self) -> int:
        return hash((self.spacegroup, self.occupations))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{value.wyckoff}:{value.label}" for value in self.occupations)
        return f"BarePrototype({self.spacegroup.setting!r}, {pairs})"
