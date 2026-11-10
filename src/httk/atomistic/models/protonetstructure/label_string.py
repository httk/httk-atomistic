"""Backend wrapping a raw canonical protonetstructure label string."""

from typing import Any, Self

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
from httk.atomistic.models.prototype.notation import try_parse_protonetstructure


class ProtonetstructureLabelString(ProtonetstructureBackend):
    r"""Wrap a canonical protonetstructure label held as a plain string.

    The string must be a canonical protonetstructure label (a ``__`` net segment and a
    ``:`` species suffix); anything else is declined. Each name becomes
    ``Species(name, (name,), (1,))``. Parsing is eager.

    :param obj: The canonical protonetstructure label text.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "protonetstructure"
    _raw: str

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a canonical protonetstructure label string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "protonetstructure") != "protonetstructure":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_protonetstructure(obj) is None:
            return None
        return cls(obj, **hints)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        parsed = try_parse_protonetstructure(obj)
        assert parsed is not None  # _backend_adopt guarantees a canonical label
        self._value = parsed

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the parsed label."""
        return self._value.spacegroup

    @property
    def occupations(self):
        """Return the site-ordered occupations of the parsed label."""
        return self._value.occupations

    @property
    def net(self):
        """Return the net rows of the parsed label."""
        return self._value.net

    def unwrap(self) -> str:
        """Return the original label text."""
        return self._raw
