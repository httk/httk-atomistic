"""
The minimal canonical structure interface for httk-atomistic.
"""

from abc import ABC, abstractmethod

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species


class StructureAPI(ABC):
    """
    Abstract base class for the canonical structure interface.

    It declares the Unitcell quartet that every structure backend produces from its
    own native representation and every structure view builds its presentation
    from: ``cell``, ``sites``, ``species``, and ``species_at_sites``. This is the
    single interchange format; there is no pairwise conversion between backends.
    """

    @property
    @abstractmethod
    def cell(self) -> Cell:
        raise NotImplementedError

    @property
    @abstractmethod
    def sites(self) -> Sites:
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
    def site_moments(self) -> SiteMomentsBackend | None:
        """Optional per-site magnetic moments, one entry per site in ``sites`` order.

        ``None`` means "nothing stated", not "zero moments".
        """
        return None
