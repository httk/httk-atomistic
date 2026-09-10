"""Eager prototype-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.bareprototype.view_base import BarePrototypeViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.bareprototype.like import BarePrototypeLike


class BarePrototypeLabel(BarePrototypeViewBase, str):
    r"""Present a prototype as its eager httk label string.

    Any faithful render is the prototype label; the *canonical* prototype label is
    the one obtained from a normalizer-canonical prototype (for example one derived via
    ``canonical_asu``). This view renders the label with no affine-normalizer pass.

    :param obj: The prototype-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: BarePrototypeBackend

    def __new__(cls, obj: "BarePrototypeLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._prototype_label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "BarePrototypeLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented prototype."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the class-partitioned occupations of the presented prototype."""
        return self._backend.occupations

    def unview(self) -> BarePrototype:
        """Return the standalone bare classification."""
        backend = self._backend
        if type(backend) is BarePrototype:
            return backend
        return BarePrototype(backend.spacegroup, backend.occupations)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
