"""Define the canonical trajectory interface for httk-atomistic."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure


class TrajectoryAPI(ABC):
    """Define the common, frame-oriented trajectory interface.

    A trajectory has one constant composition for all frames. Trajectories whose
    number or identity of sites changes are intentionally outside this interface.
    """

    @property
    @abstractmethod
    def nframes(self) -> int:
        """Return the number of frames."""
        raise NotImplementedError

    @abstractmethod
    def frame(self, i: int) -> UnitcellStructure:
        """Return one frame by index.

        :param i: Frame index.
        :return: The requested unit-cell structure.
        """
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[UnitcellStructure]:
        """Iterate over the frames.

        :return: An iterator of unit-cell structures.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        """Return the constant distinct species."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the constant species name at each site."""
        raise NotImplementedError

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        """Return stored reference-frame indexes, or ``None``."""
        return None

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return the names of available per-frame observables."""
        return ()

    def observable(self, name: str) -> tuple[Any, ...]:
        """Return one observable's values by frame.

        :param name: Observable name.
        :return: The observable values in frame order.
        :raises KeyError: If the observable is unavailable.
        """
        if name not in self.observable_names:
            raise KeyError(name)
        raise KeyError(name)
