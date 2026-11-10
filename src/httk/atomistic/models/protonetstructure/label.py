"""Eager protonetstructure-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protonetstructure.view_base import ProtonetstructureViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.protonetstructure.like import ProtonetstructureLike


class ProtonetstructureLabel(ProtonetstructureViewBase, str):
    r"""Present a protonetstructure as its eager httk label string.

    Any faithful render is the protonetstructure label; the *canonical* one comes from
    the canonicaliser. This view renders the given value without symmetry work.

    :param obj: The protonetstructure-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtonetstructureBackend

    def __new__(cls, obj: "ProtonetstructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "ProtonetstructureLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented protonetstructure."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the site-ordered occupations of the presented protonetstructure."""
        return self._backend.occupations

    @property
    def net(self):
        """Return the net rows of the presented protonetstructure."""
        return self._backend.net

    def unview(self) -> Protonetstructure:
        """Return the standalone protonetstructure value."""
        backend = self._backend
        if type(backend) is Protonetstructure:
            return backend
        return Protonetstructure(backend.spacegroup, backend.occupations, backend.net)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
