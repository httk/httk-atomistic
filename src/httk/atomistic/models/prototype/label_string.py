"""Backend wrapping a raw canonical prototype label string."""

from typing import Any, Self

from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.notation import try_parse_prototype


class PrototypeLabelString(PrototypeBackend):
    r"""Wrap a canonical prototype label held as a plain string.

    The string must be a canonical unsuffixed prototype label (no ``:`` species
    suffix); anything else is declined. Parsing is eager, so the wrapped value is
    validated at adoption.

    :param obj: The canonical prototype label text.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "prototype"
    _raw: str

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a canonical prototype label string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "prototype") != "prototype":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_prototype(obj) is None:
            return None
        return cls(obj, **hints)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        parsed = try_parse_prototype(obj)
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

    @property
    def representative(self):
        return None

    @property
    def discriminator(self):
        return None

    def unwrap(self) -> str:
        """Return the original label text."""
        return self._raw
