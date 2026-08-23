"""Eager prototype-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.view_base import PrototypeViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.prototype.like import PrototypeLike


class PrototypeLabel(PrototypeViewBase, str):
    r"""Present a prototype as its eager httk label string.

    Any faithful render is the prototype label; the *canonical* prototype label is
    the one obtained from a normalizer-canonical prototype (for example one derived via
    ``canonical_asu``). This view renders the label with no affine-normalizer pass.

    :param obj: The prototype-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: PrototypeBackend

    def __new__(cls, obj: "PrototypeLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._prototype_label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "PrototypeLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented prototype."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the class-partitioned occupations of the presented prototype."""
        return self._backend.occupations

    @property
    def representative(self):
        return self._backend.representative

    @property
    def discriminator(self):
        return self._backend.discriminator

    def unview(self) -> Prototype:
        """Return the presented prototype as a standalone value.

        :return: The prototype value.
        """
        backend = self._backend
        if type(backend) is Prototype:
            return backend
        # Labels deliberately render only the base notation, but remain views: shedding
        # the view must recover every field held by the backend, including class identity
        # that has no textual label spelling.
        return Prototype(
            backend.spacegroup,
            backend.occupations,
            representative=backend.representative,
            discriminator=backend.discriminator,
        )

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
