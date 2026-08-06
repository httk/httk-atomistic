"""The canonical trajectory interface for httk-atomistic."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure


class TrajectoryAPI(ABC):
    """The common, frame-oriented interface for trajectory backends.

    A trajectory has one constant composition for all frames. Trajectories whose
    number or identity of sites changes are intentionally outside this interface.
    """

    @property
    @abstractmethod
    def nframes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def frame(self, i: int) -> UnitcellStructure:
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[UnitcellStructure]:
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        return None

    @property
    def observable_names(self) -> tuple[str, ...]:
        return ()

    def observable(self, name: str) -> tuple[Any, ...]:
        if name not in self.observable_names:
            raise KeyError(name)
        raise KeyError(name)
