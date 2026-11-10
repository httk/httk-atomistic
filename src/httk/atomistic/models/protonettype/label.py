"""Eager protonettype-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.protonettype.view_base import ProtonettypeViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.protonettype.like import ProtonettypeLike


class ProtonettypeLabel(ProtonettypeViewBase, str):
    r"""Present a protonettype as its eager httk label string.

    Any faithful render is the protonettype label; the *canonical* one comes from the
    canonicaliser. This view renders the given value without symmetry work.

    :param obj: The protonettype-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtonettypeBackend

    def __new__(cls, obj: "ProtonettypeLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "ProtonettypeLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented protonettype."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the site-ordered occupations of the presented protonettype."""
        return self._backend.occupations

    @property
    def net(self):
        """Return the net rows of the presented protonettype."""
        return self._backend.net

    def unview(self) -> Protonettype:
        """Return the standalone protonettype value."""
        backend = self._backend
        if type(backend) is Protonettype:
            return backend
        return Protonettype(backend.spacegroup, backend.occupations, backend.net)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
