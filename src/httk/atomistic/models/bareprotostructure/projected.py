"""Projection of refined protostructure values onto their bare classification."""

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.bareprotostructure.backend import BareProtostructureBackend
from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.protonetstructure.view_base import ProtonetstructureViewBase
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.prototype.notation import _erased_undetermined, _erased_undetermined_label

if TYPE_CHECKING:
    from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure


class ProjectedBareProtostructure(BareProtostructureBackend):
    """Retain a refined or protonetstructure source while exposing its discrete classification.

    A refined source exposes its own occupations. A protonetstructure source exposes the
    canonical bare protostructure of its occupations: its class order follows the net, so its
    bare part is not always the canonical bare label.

    :param obj: Refined or protonetstructure classification source.
    """

    kind = "bare_protostructure"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        """Adopt a refined value or protonetstructure without materializing its geometry or net."""
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        return (
            cls(obj)
            if isinstance(
                obj,
                (ProtostructureBackend, ProtostructureViewBase, ProtonetstructureBackend, ProtonetstructureViewBase),
            )
            else None
        )

    def __init__(self, obj: Any) -> None:
        self._source = obj

    @cached_property
    def _projected(self) -> Any:
        """Return the source, or for a protonetstructure its canonical bare protostructure."""
        source = self._source
        if (erased := _erased_undetermined(source, "bare_protostructure")) is not None:
            return erased
        if not isinstance(source, (ProtonetstructureBackend, ProtonetstructureViewBase)):
            return source
        from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
        from httk.atomistic.symmetry.canonical_classification import _canonical_bare

        value = cast(ProtonetstructureBackend, source)  # the view base declares no fields
        return _canonical_bare(
            BareProtostructure(value.spacegroup, value.occupations), prototype=False, preserve_chirality=True
        )

    def resolve(self) -> "BareProtostructure":
        """Return the standalone bare classification."""
        from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure

        projected = self._projected
        if isinstance(projected, BareProtostructure):
            return projected
        return BareProtostructure(projected.spacegroup, projected.occupations)

    @property
    def spacegroup(self):
        """Return the standard-setting space group."""
        return self._projected.spacegroup

    @property
    def occupations(self):
        """Return the canonical discrete occupations."""
        return self._projected.occupations

    @property
    def _undetermined(self) -> str | None:
        """Return the source's undetermined label cut to this level, ``None`` if determined here."""
        return _erased_undetermined_label(self._source, "bare_protostructure")

    def unwrap(self) -> Any:
        """Return the retained refined source."""
        return unwrap(self._source)
