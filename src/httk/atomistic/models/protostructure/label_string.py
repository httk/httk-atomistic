"""Backend wrapping a raw canonical protostructure label string."""

from typing import Any, Self

from httk.atomistic.models.protochroma.notation import try_parse_protostructure
from httk.atomistic.models.protostructure.backend import ProtostructureBackend


class ProtostructureLabelString(ProtostructureBackend):
    r"""Wrap a canonical httk protostructure label held as a plain string.

    The string must be a canonical label carrying a ``:`` species suffix; the unsuffixed
    protochroma labels are declined (they belong to the protochroma family). Each name
    becomes ``Species(name, (name,), (1,))``. Parsing is eager.

    :param obj: The canonical protostructure label text.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "protostructure"
    _raw: str

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a canonical protostructure label string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "protostructure") != "protostructure":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_protostructure(obj) is None:
            return None
        return cls(obj, **hints)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        parsed = try_parse_protostructure(obj)
        assert parsed is not None  # _backend_adopt guarantees a canonical label
        self._value = parsed

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the parsed label."""
        return self._value.spacegroup

    @property
    def occupations(self):
        """Return the occupied Wyckoff positions of the parsed label."""
        return self._value.occupations

    def unwrap(self) -> str:
        """Return the original label text."""
        return self._raw
