"""Holder for modulated mCIF blocks not representable by standard structures."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend

__all__ = ["ModulatedStructure"]


class ModulatedStructure(StructureBackend):
    """Raw mCIF data; future httk-magnetism work will interpret the modulation."""

    kind: ClassVar[str] = "modulated-mcif"

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = MappingProxyType(dict(payload))

    @property
    def payload(self) -> Mapping[str, Any]:
        return self._payload

    @property
    def mod_dim(self) -> Any:
        return self._incomm_value("mod_dim")

    @property
    def structural_q(self) -> Any:
        return self._incomm_value("structural_q")

    @property
    def magnetic_q(self) -> Any:
        return self._incomm_value("magnetic_q")

    def _incomm_value(self, name: str) -> Any:
        incomm = self._payload.get("incomm")
        return incomm.get(name) if isinstance(incomm, Mapping) else None

    def _unavailable(self, name: str) -> ValueError:
        return ValueError(
            "an incommensurately modulated magnetic structure cannot be represented as a "
            f"{name} of a standard structure class; the raw mcif payload is available as .payload"
        )

    @property
    def cell(self) -> Cell:
        raise self._unavailable("cell")

    @property
    def sites(self) -> Sites:
        raise self._unavailable("sites")

    @property
    def species(self) -> tuple[Species, ...]:
        raise self._unavailable("species")

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        raise self._unavailable("species_at_sites")

    @property
    def site_moments(self) -> Any:
        raise self._unavailable("site_moments")
