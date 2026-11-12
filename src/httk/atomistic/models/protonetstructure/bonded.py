"""Backend labelling a fundamental domain that carries a bond network."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.symmetry.spacegroup import Spacegroup


class BondedProtonetstructure(ProtonetstructureBackend):
    r"""Present a fundamental domain with a bond network as its protonetstructure, lazily.

    The value is :meth:`Protonetstructure.from_asu <httk.atomistic.Protonetstructure.from_asu>`
    of the structure and its own network. A view whose unwrapped source is such a structure
    (``UnitcellStructureView(asu)``) is accepted too; a structure without a network is declined.

    :param obj: The fundamental-domain structure with a bond network.
    """

    kind = "protonetstructure"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a fundamental domain that carries a bond network.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        source = obj if isinstance(obj, FundamentalDomainStructure) else unwrap(obj)
        return cls(source) if isinstance(source, FundamentalDomainStructure) and source.bonds is not None else None

    def __init__(self, obj: FundamentalDomainStructure) -> None:
        self._source = obj

    @cached_property
    def _value(self) -> Protonetstructure:
        return Protonetstructure.from_asu(self._source)

    def resolve(self) -> Protonetstructure:
        """Return the protonetstructure of the structure and its network."""
        return self._value

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._value.spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the site-ordered occupations."""
        return self._value.occupations

    @property
    def net(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return the net rows."""
        return self._value.net

    @property
    def _undetermined(self) -> str | None:
        """Return the undetermined label when the net search ran out, else ``None``."""
        return self._value._undetermined

    def unwrap(self) -> Any:
        """Return the retained structure."""
        return unwrap(self._source)
