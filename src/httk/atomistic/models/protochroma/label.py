"""Eager protochroma-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.protochroma.backend import ProtochromaBackend
from httk.atomistic.models.protochroma.protochroma import Protochroma
from httk.atomistic.models.protochroma.view_base import ProtochromaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.protochroma.like import ProtochromaLike


class ProtochromaLabel(ProtochromaViewBase, str):
    r"""Present a protochroma as its eager httk label string.

    Any faithful render is the protochroma label; the *canonical* protochroma label is
    the one obtained from a normalizer-canonical protochroma (for example one derived via
    ``canonical_asu``). This view renders the label with no affine-normalizer pass.

    :param obj: The protochroma-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtochromaBackend

    def __new__(cls, obj: "ProtochromaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._pattern_label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "ProtochromaLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented protochroma."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the class-partitioned occupations of the presented protochroma."""
        return self._backend.occupations

    def unview(self) -> Protochroma:
        """Return the presented protochroma as a standalone value.

        :return: The protochroma value.
        """
        backend = self._backend
        if type(backend) is Protochroma:
            return backend
        return Protochroma(backend.spacegroup, backend.occupations)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
