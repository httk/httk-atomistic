"""
The minimal canonical structure interface for httk-atomistic.
"""

from abc import ABC, abstractmethod
from fractions import Fraction

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species


class StructureAPI(ABC):
    """Define the canonical structure interface.

    It declares the Unitcell quartet that every structure backend produces from its
    own native representation and every structure view builds its presentation
    from: ``cell``, ``sites``, ``species``, and ``species_at_sites``. This is the
    single interchange format; there is no pairwise conversion between backends.
    """

    @property
    @abstractmethod
    def cell(self) -> Cell:
        """Expose the structure's cell."""
        raise NotImplementedError

    @property
    @abstractmethod
    def sites(self) -> Sites:
        """Expose the structure's site coordinates."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        """Expose the structure's distinct species."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose the species occupying each site."""
        raise NotImplementedError

    @property
    def charge(self) -> Fraction | None:
        """Expose the explicitly assigned net charge of the cell content.

        ``None`` means unstated and is never derived from the species; it is distinct
        from an explicit zero.

        :return: The assigned charge, or ``None`` when it is unstated.
        """
        return None

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        """Expose optional per-site magnetic moments in ``sites`` order.

        ``None`` means "nothing stated", not "zero moments".

        :return: The site moments, or ``None`` when they are unstated.
        """
        return None
