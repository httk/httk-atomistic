"""Backend wrapping a raw canonical protonettype label string."""

from typing import Any, Self

from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.prototype.notation import parse_protonettype_label, try_parse_protonettype


class ProtonettypeLabelString(ProtonettypeBackend):
    r"""Wrap a canonical protonettype label held as a plain string.

    The string must be a canonical protonettype label (a ``__`` net segment and no ``:``
    species suffix); anything else is declined. Parsing (which canonicalizes the net once) is eager.

    :param obj: The canonical protonettype label text.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "protonettype"
    _raw: str
    _undetermined: str | None
    _value: Protonettype

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a canonical protonettype label string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "protonettype") != "protonettype":
            return None
        parsed = try_parse_protonettype(obj)
        if parsed is None:
            return None
        instance = cls.__new__(cls)
        instance._raw, instance._value = obj, parsed  # parsed once: each parse canonicalizes the net
        instance._undetermined = parsed._undetermined
        return instance

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw, self._value = obj, parse_protonettype_label(obj)
        self._undetermined = self._value._undetermined

    def resolve(self) -> Protonettype:
        """Return the parsed canonical value."""
        return self._value

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
