"""Anonymous erasure of protonetstructures."""

from functools import cached_property
from itertools import groupby
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonetstructure.bonded import BondedProtonetstructure
from httk.atomistic.models.protonetstructure.view_base import ProtonetstructureViewBase
from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.prototype.notation import _erased_undetermined, _erased_undetermined_label


class DerivedProtonettype(ProtonettypeBackend):
    """Retain a protonetstructure source while exposing its anonymous erasure.

    A fundamental domain with a bond network is taken through
    :class:`~httk.atomistic.models.protonetstructure.bonded.BondedProtonetstructure`.

    Class run ``c`` gets the label ``anonymous_symbol(c)`` by position, so the site
    numbering and the net are unchanged; tied classes are never reordered by name.

    :param obj: Protonetstructure source.
    """

    kind = "protonettype"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        """Adopt a protonetstructure backend or view, or a fundamental domain with a bond network."""
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        if isinstance(obj, (ProtonetstructureBackend, ProtonetstructureViewBase)):
            return cls(obj)
        bonded = BondedProtonetstructure._backend_adopt(obj)
        return None if bonded is None else cls(bonded)

    def __init__(self, obj: Any) -> None:
        self._source = obj

    @cached_property
    def _derived(self) -> Protonettype:
        if (erased := _erased_undetermined(self._source, "protonettype")) is not None:
            return erased
        runs = groupby(self._source.occupations, key=lambda o: o.species.name)
        occupations = [(o.wyckoff, anonymous_symbol(index)) for index, (_, run) in enumerate(runs) for o in run]
        # a protonetstructure's net is already canonical as an anonymous net (D3 only picks species)
        return Protonettype._of(self._source.spacegroup, occupations, self._source.net, self._source.shapes)

    def resolve(self) -> Protonettype:
        """Return the standalone anonymous protonettype."""
        return self._derived

    @property
    def spacegroup(self):
        """Return the standard-setting space group."""
        return self._derived.spacegroup

    @property
    def occupations(self):
        """Return the anonymous occupations in site order."""
        return self._derived.occupations

    @property
    def net(self):
        """Return the unchanged net rows."""
        return self._derived.net

    @property
    def shapes(self):
        """Return the unchanged per-site bond shapes."""
        return self._derived.shapes

    @property
    def _undetermined(self) -> str | None:
        """Return the source's undetermined label cut to this level, ``None`` if determined here."""
        return _erased_undetermined_label(self._source, "protonettype")

    def unwrap(self) -> Any:
        """Return the retained protonetstructure source."""
        return unwrap(self._source)
