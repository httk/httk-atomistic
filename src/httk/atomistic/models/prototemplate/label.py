"""Eager prototemplate-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
from httk.atomistic.models.prototemplate.prototemplate import Prototemplate
from httk.atomistic.models.prototemplate.view_base import PrototemplateViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.prototemplate.like import PrototemplateLike


class PrototemplateLabel(PrototemplateViewBase, str):
    r"""Present a prototemplate as its eager httk label string.

    Any faithful render is the prototemplate label; the *canonical* prototemplate label is
    the one obtained from a normalizer-canonical template (for example one derived via
    ``canonical_asu``). This view renders the label with no affine-normalizer pass.

    :param obj: The prototemplate-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: PrototemplateBackend

    def __new__(cls, obj: "PrototemplateLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = str.__new__(cls, backend._template_label_text())
        instance._backend = backend
        return instance

    def __init__(self, obj: "PrototemplateLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented template."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the class-partitioned occupations of the presented template."""
        return self._backend.occupations

    def unview(self) -> Prototemplate:
        """Return the presented prototemplate as a standalone value.

        :return: The prototemplate value.
        """
        backend = self._backend
        if type(backend) is Prototemplate:
            return backend
        return Prototemplate(backend.spacegroup, backend.occupations)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
