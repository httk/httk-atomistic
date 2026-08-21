"""Backend wrapping a raw canonical prototemplate label string."""

from typing import Any, Self

from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
from httk.atomistic.models.prototemplate.notation import try_parse_prototemplate


class PrototemplateLabelString(PrototemplateBackend):
    r"""Wrap a canonical prototemplate label held as a plain string.

    The string must be a canonical unsuffixed prototemplate label (no ``:`` species
    suffix); anything else is declined. Parsing is eager, so the wrapped value is
    validated at adoption.

    :param obj: The canonical prototemplate label text.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "prototemplate"
    _raw: str

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a canonical prototemplate label string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "prototemplate") != "prototemplate":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_prototemplate(obj) is None:
            return None
        return cls(obj, **hints)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        parsed = try_parse_prototemplate(obj)
        assert parsed is not None  # _backend_adopt guarantees a canonical label
        self._value = parsed

    @property
    def spacegroup(self):
        """Return the standard-setting space group of the parsed label."""
        return self._value.spacegroup

    @property
    def occupations(self):
        """Return the class-partitioned occupations of the parsed label."""
        return self._value.occupations

    def unwrap(self) -> str:
        """Return the original label text."""
        return self._raw
