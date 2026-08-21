"""Eager protostructure-label presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.prototemplate.notation import render_protostructure_label

if TYPE_CHECKING:
    from httk.atomistic.models.protostructure.like import ProtostructureLike


class ProtostructureLabel(ProtostructureViewBase, str):
    r"""Present a protostructure as its eager httk label string.

    Any faithful render is the protostructure label; the *canonical* protostructure label
    is the one obtained from a normalizer-canonical protostructure (for example one derived
    via ``canonical_asu``). This view renders the label with no affine-normalizer pass.

    The unsuffixed part is the httk prototemplate label of the erased template (classes
    ordered by Wyckoff letters, not by element as AFLOW does), followed by ``:`` and the
    class species names in group order.

    :param obj: The protostructure-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtostructureBackend

    def __new__(cls, obj: "ProtostructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        text = render_protostructure_label(
            backend.spacegroup, [(occupation.wyckoff, occupation.species.name) for occupation in backend.occupations]
        )
        instance = str.__new__(cls, text)
        instance._backend = backend
        return instance

    def __init__(self, obj: "ProtostructureLike", **hints: Any) -> None:
        pass

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the presented protostructure."""
        return self._backend.spacegroup

    @property
    def occupations(self):
        """Return the occupied Wyckoff positions of the presented protostructure."""
        return self._backend.occupations

    def unview(self) -> Protostructure:
        """Return the presented protostructure as a standalone value.

        :return: The protostructure value.
        """
        backend = self._backend
        if type(backend) is Protostructure:
            return backend
        return Protostructure(backend.spacegroup, backend.occupations)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
