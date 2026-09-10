"""Anonymous projection of assigned bare classifications."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.bareprotostructure.backend import BareProtostructureBackend
from httk.atomistic.models.bareprotostructure.view_base import BareProtostructureViewBase
from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.prototype.derived import _anonymous_occupations


class DerivedBarePrototype(BarePrototypeBackend):
    """Retain an assigned source while exposing anonymous occupations.

    :param obj: Assigned classification source.
    """

    kind = "bare_prototype"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        """Adopt an assigned bare or refined classification."""
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        return (
            cls(obj)
            if isinstance(
                obj,
                (BareProtostructureBackend, BareProtostructureViewBase, ProtostructureBackend, ProtostructureViewBase),
            )
            else None
        )

    def __init__(self, obj: Any) -> None:
        self._source = obj

    @cached_property
    def _derived(self) -> BarePrototype:
        return BarePrototype(self._source.spacegroup, _anonymous_occupations(self._source.occupations))

    def resolve(self) -> BarePrototype:
        """Return the standalone anonymous classification."""
        return self._derived

    @property
    def spacegroup(self):
        """Return the standard-setting space group."""
        return self._derived.spacegroup

    @property
    def occupations(self):
        """Return canonical anonymous occupations."""
        return self._derived.occupations

    def unwrap(self) -> Any:
        """Return the retained assigned source."""
        return unwrap(self._source)
