"""Eager protopattern-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.protopattern.backend import ProtopatternBackend
from httk.atomistic.models.protopattern.protopattern import Protopattern
from httk.atomistic.models.protopattern.view_base import ProtopatternViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.protopattern.like import ProtopatternLike


class ProtopatternLabel(ProtopatternViewBase, str):
    r"""Present a protopattern as its eager canonical httk label string.

    Any faithful render is the protopattern label; the *canonical* protopattern label is
    the one obtained from a normalizer-canonical pattern (for example one derived via
    ``canonical_asu``). This view renders the label with no affine-normalizer pass.

    :param obj: The protopattern-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtopatternBackend

    def __new__(cls, obj: "ProtopatternLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._pattern_label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "ProtopatternLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented pattern."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the class-partitioned occupations of the presented pattern."""
        return self._backend.occupations

    def unview(self) -> Protopattern:
        """Return the presented protopattern as a standalone value.

        :return: The protopattern value.
        """
        backend = self._backend
        if type(backend) is Protopattern:
            return backend
        return Protopattern(backend.spacegroup, backend.occupations)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
