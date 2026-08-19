"""Store the native immutable trajectory representation."""

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.backend import TrajectoryBackend

if TYPE_CHECKING:
    from httk.atomistic.models.structure.like import StructureLike


class Trajectory(TrajectoryBackend):
    """Store an immutable trajectory in the native backend.

    A trajectory requires at least one frame and keeps one constant composition
    across all frames.

    :param frames: Unit-cell structures to coerce and store.
    :param observables: Optional per-frame observable values.
    :param reference_frames: Optional indexes of bounded reference frames.
    """

    kind: ClassVar[str] = "native"
    __httk_storage_record__: ClassVar[type[Any]]
    _frames: tuple[UnitcellStructure, ...]
    _observables: Mapping[str, tuple[Any, ...]]
    _reference_frames: tuple[int, ...] | None
    _species: tuple[Species, ...]
    _species_at_sites: tuple[str, ...]

    def __init__(
        self,
        frames: Sequence["StructureLike"],
        observables: Mapping[str, Sequence[Any]] | None = None,
        reference_frames: Sequence[int] | None = None,
    ) -> None:
        if not frames:
            raise ValueError("Trajectory requires at least one frame")
        from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

        coerced = tuple(UnitcellStructureView(frame).unview() for frame in frames)
        species = coerced[0].species
        species_at_sites = coerced[0].species_at_sites
        for index, frame in enumerate(coerced[1:], 1):
            if frame.species != species or frame.species_at_sites != species_at_sites:
                raise ValueError(f"Trajectory frame {index} has a varying composition")
        values = {} if observables is None else {name: tuple(value) for name, value in observables.items()}
        for name, value in values.items():
            if len(value) != len(coerced):
                raise ValueError(f"Trajectory observable {name!r} has length {len(value)}, expected {len(coerced)}")
        references: tuple[int, ...] | None = None
        if reference_frames is not None:
            checked: list[int] = []
            for reference in reference_frames:
                if not isinstance(reference, int) or isinstance(reference, bool):
                    raise ValueError(f"Trajectory reference frame {reference!r} is not an integer")
                if not 0 <= reference < len(coerced):
                    raise ValueError(f"Trajectory reference frame {reference!r} is out of bounds")
                checked.append(reference)
            references = tuple(sorted(set(checked)))
        self._frames = coerced
        self._observables = MappingProxyType(values)
        self._reference_frames = references
        self._species = species
        self._species_at_sites = species_at_sites

    @property
    def nframes(self) -> int:
        """Return the number of stored frames."""
        return len(self._frames)

    def frame(self, i: int) -> UnitcellStructure:
        """Return one stored frame by index.

        :param i: Frame index.
        :return: The requested unit-cell structure.
        :raises IndexError: If the index is out of range.
        """
        return self._frames[i]

    def __repr__(self) -> str:
        parts = [f"frames=(... {len(self._frames)} frame(s) ...)"]
        if self._observables:
            parts.append(f"observables={tuple(self._observables)!r}")
        if self._reference_frames is not None:
            parts.append(f"reference_frames={self._reference_frames!r}")
        return f"Trajectory({', '.join(parts)})"

    def frames(self) -> Iterator[UnitcellStructure]:
        """Iterate over the stored frames.

        :return: An iterator of unit-cell structures.
        """
        return iter(self._frames)

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        """Return the bounded reference-frame indexes, or ``None``."""
        return self._reference_frames

    @property
    def species(self) -> tuple[Species, ...]:
        """Return the constant distinct species."""
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the constant species name at each site."""
        return self._species_at_sites

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return the names of stored observables."""
        return tuple(self._observables)

    def observable(self, name: str) -> tuple[Any, ...]:
        """Return one observable's values in frame order.

        :param name: Observable name.
        :return: The observable values.
        :raises KeyError: If the observable is unavailable.
        """
        try:
            return self._observables[name]
        except KeyError:
            raise KeyError(name) from None
