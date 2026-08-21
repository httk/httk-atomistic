"""The immutable geometry-free, element-free protochroma value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.protochroma.backend import ProtochromaBackend
from httk.atomistic.models.protochroma.notation import canonical_label_map
from httk.atomistic.models.protochroma.occupation import ProtochromaOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Protochroma(ProtochromaBackend):
    """Store a standard-setting space group and its class-partitioned Wyckoff letters.

    A protochroma is space group plus occupied Wyckoff letters plus a partition of those
    occupations into anonymous species classes; it has no chemical elements and no
    continuous degrees of freedom. Construction re-canonicalizes the class assignment by
    the pinned group-ordering rule, so any permutation of the input class labels builds an
    equal value that renders the identical label.

    ``Protochroma`` is the anonymous-species, Wyckoff-positions-only cell of the
    material-information matrix:

    ======================  ===============  ==============
    Geometrical info        Anonymous        Assigned
    ======================  ===============  ==============
    Wyckoff positions only  Protochroma      Protostructure
    Geometrical class       Prototype        Crystallotype
    Exact geometry          Chromastructure  Structure
    ======================  ===============  ==============

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: The occupied Wyckoff positions and their anonymous class labels.
    """

    _spacegroup: Spacegroup
    _occupations: tuple[ProtochromaOccupation, ...]
    kind: ClassVar[str] = "protochroma"

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[ProtochromaOccupation | tuple[str, str]],
    ) -> None:
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"Protochroma records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) "
                "instead"
            )
        raw = tuple(
            value if isinstance(value, ProtochromaOccupation) else ProtochromaOccupation(value[0], value[1])
            for value in occupations
        )
        if not raw:
            raise ValueError("Protochroma occupations must be non-empty")
        letters_by_label: dict[str, list[str]] = {}
        for occupation in raw:
            try:
                self._spacegroup.wyckoff_position(occupation.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            letters_by_label.setdefault(occupation.label, []).append(occupation.wyckoff)
        expected = {anonymous_symbol(index) for index in range(len(letters_by_label))}
        if set(letters_by_label) != expected:
            raise ValueError("Protochroma class labels must be consecutive anonymous symbols from 'A'")
        relabel = canonical_label_map({label: tuple(sorted(letters)) for label, letters in letters_by_label.items()})
        self._occupations = tuple(
            sorted(
                (ProtochromaOccupation(occupation.wyckoff, relabel[occupation.label]) for occupation in raw),
                key=lambda value: (value.label, value.wyckoff),
            )
        )

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[ProtochromaOccupation, ...]:
        """Return the canonical class-partitioned occupied Wyckoff positions."""
        return self._occupations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protochroma):
            return NotImplemented
        return (self._spacegroup, self._occupations) == (other._spacegroup, other._occupations)

    def __hash__(self) -> int:
        return hash((self._spacegroup, self._occupations))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{occupation.wyckoff}:{occupation.label}" for occupation in self._occupations)
        return f"Protochroma({self._spacegroup.setting!r}, {pairs})"
