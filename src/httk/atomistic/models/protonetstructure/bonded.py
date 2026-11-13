"""Backend labelling a structure that carries a bond network."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.symmetry.spacegroup import Spacegroup


class BondedProtonetstructure(ProtonetstructureBackend):
    r"""Present a structure with a bond network as its protonetstructure, lazily.

    The value is :meth:`Protonetstructure.from_asu <httk.atomistic.Protonetstructure.from_asu>`
    of the structure and its own network. A view whose unwrapped source is such a structure
    (``UnitcellStructureView(asu)``) is accepted too, and so is a lazy structure backend whose
    ``resolve()`` gives one (a :class:`~httk.atomistic.CIFStructure`). A unit cell (or any other
    unit-cell presentation) with a network is labelled through the asymmetric unit recognition
    gives it (``ASUStructureView(cell)``), which carries the network when the recognized symmetry
    keeps it; resolving raises :class:`ValueError` when it does not. As for
    :class:`~httk.atomistic.BareProtostructureView` of a unit cell, the label is that of the
    description recognition yields, not canonicalized further. A structure without a network is
    declined.

    :param obj: The structure with a bond network.
    :raises ValueError: If ``obj`` is a unit cell that is not periodic in three dimensions, which
        has no space group to recognize.
    """

    kind = "protonetstructure"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a structure that carries a bond network.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        source = obj if isinstance(obj, FundamentalDomainStructure) else unwrap(obj)
        if isinstance(source, StructureBackend) and not isinstance(source, FundamentalDomainStructure):
            resolve = getattr(source, "resolve", None)  # a lazy source such as a CIFStructure
            source = resolve() if callable(resolve) else source
        # a unit-cell presentation's network indexes its sites, a fundamental domain's its domain sites
        return cls(source) if isinstance(source, StructureBackend) and source.bonds is not None else None

    def __init__(self, obj: StructureBackend) -> None:
        if not isinstance(obj, FundamentalDomainStructure) and obj.cell.periodicity != (True, True, True):
            raise ValueError(
                "Protonetstructure: a unit cell that is not periodic in three dimensions has no space group to "
                "recognize; label a molecular fundamental domain with its bond network instead"
            )
        self._source = obj

    @cached_property
    def _value(self) -> Protonetstructure:
        source = self._source
        if isinstance(source, FundamentalDomainStructure):
            return Protonetstructure.from_asu(source)
        from httk.atomistic.models.structure.asu_view import ASUStructureView

        asu = ASUStructureView(source).resolve()
        if asu.bonds is None:
            raise ValueError(
                f"Protonetstructure: the bond network of the {type(source).__name__} is not invariant under its "
                f"recognized symmetry IT {asu.spacegroup.it_number}"
            )
        return Protonetstructure.from_asu(asu)

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
    def shapes(self) -> tuple[str | None, ...]:
        """Return the per-site bond shapes, each supplied or else from the baseline."""
        return self._value.shapes

    @property
    def _undetermined(self) -> str | None:
        """Return the undetermined label when the net search ran out, else ``None``."""
        return self._value._undetermined

    def unwrap(self) -> Any:
        """Return the retained structure."""
        return unwrap(self._source)
