"""Projection of refined prototype values onto their bare classification."""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.view_base import PrototypeViewBase


class ProjectedBarePrototype(BarePrototypeBackend):
    """Retain a refined source while exposing its discrete classification.

    :param obj: Refined classification source.
    """

    kind = "bare_prototype"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        """Adopt a refined value without materializing its geometry."""
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        return cls(obj) if isinstance(obj, (PrototypeBackend, PrototypeViewBase)) else None

    def __init__(self, obj: Any) -> None:
        self._source = obj

    @property
    def spacegroup(self):
        """Return the standard-setting space group."""
        return self._source.spacegroup

    @property
    def occupations(self):
        """Return the canonical discrete occupations."""
        return self._source.occupations

    def unwrap(self) -> Any:
        """Return the retained refined source."""
        return unwrap(self._source)
