"""The immutable geometry-free, element-free prototemplate value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
from httk.atomistic.models.prototemplate.notation import canonical_label_map
from httk.atomistic.models.prototemplate.occupation import PrototemplateOccupation
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Prototemplate(PrototemplateBackend):
    """Store a standard-setting space group and its class-partitioned Wyckoff letters.

    A prototemplate is space group plus occupied Wyckoff letters plus a partition of those
    occupations into anonymous species classes; it has no chemical elements and no
    continuous degrees of freedom. Construction re-canonicalizes the class assignment by
    the pinned group-ordering rule, so any permutation of the input class labels builds an
    equal value that renders the identical label.

    ``Prototemplate`` is the anonymous-species, Wyckoff-positions-only cell of the
    material-information matrix:

    ======================  ===============  ==============
    Geometrical info        Anonymous        Assigned
    ======================  ===============  ==============
    Wyckoff positions only  Prototemplate    Protostructure
    Geometrical class       Prototype        Structuretype
    Exact geometry          CrystalTemplate  Structure
    ======================  ===============  ==============

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: The occupied Wyckoff positions and their anonymous class labels.
    """

    _spacegroup: Spacegroup
    _occupations: tuple[PrototemplateOccupation, ...]
    kind: ClassVar[str] = "prototemplate"

    def __init__(
        self,
        spacegroup: Spacegroup | int,
        occupations: Sequence[PrototemplateOccupation | tuple[str, str]],
    ) -> None:
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"Prototemplate records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) "
                "instead"
            )
        raw = tuple(
            value if isinstance(value, PrototemplateOccupation) else PrototemplateOccupation(value[0], value[1])
            for value in occupations
        )
        if not raw:
            raise ValueError("Prototemplate occupations must be non-empty")
        letters_by_label: dict[str, list[str]] = {}
        for occupation in raw:
            try:
                self._spacegroup.wyckoff_position(occupation.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            letters_by_label.setdefault(occupation.label, []).append(occupation.wyckoff)
        expected = {anonymous_symbol(index) for index in range(len(letters_by_label))}
        if set(letters_by_label) != expected:
            raise ValueError("Prototemplate class labels must be consecutive anonymous symbols from 'A'")
        relabel = canonical_label_map({label: tuple(sorted(letters)) for label, letters in letters_by_label.items()})
        self._occupations = tuple(
            sorted(
                (PrototemplateOccupation(occupation.wyckoff, relabel[occupation.label]) for occupation in raw),
                key=lambda value: (value.label, value.wyckoff),
            )
        )

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototemplateOccupation, ...]:
        """Return the canonical class-partitioned occupied Wyckoff positions."""
        return self._occupations

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prototemplate):
            return NotImplemented
        return (self._spacegroup, self._occupations) == (other._spacegroup, other._occupations)

    def __hash__(self) -> int:
        return hash((self._spacegroup, self._occupations))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{occupation.wyckoff}:{occupation.label}" for occupation in self._occupations)
        return f"Prototemplate({self._spacegroup.setting!r}, {pairs})"
