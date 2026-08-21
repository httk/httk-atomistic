"""Lazy protochroma erasure and presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.protochroma.backend import ProtochromaBackend
from httk.atomistic.models.protochroma.derived import DerivedProtochroma
from httk.atomistic.models.protochroma.protochroma import Protochroma
from httk.atomistic.models.protochroma.view_base import ProtochromaViewBase


class ProtochromaView(ProtochromaViewBase, Protochroma):
    r"""Present a lazy standard-setting protochroma view.

    Sources may be an existing protochroma, a protostructure (real species erased to
    anonymous classes), or a fundamental-domain/structure source recognized and
    discretized. Erasure is deferred until the first field access.

    :param obj: The protochroma-like or structure-like source.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtochromaBackend
    _resolved_protochroma: Protochroma | None
    _DEFERRED_FIELDS = frozenset({"_spacegroup", "_occupations"})

    def __new__(cls, obj: Any = MISSING, **hints: Any) -> Self:
        if obj is MISSING:  # pickle/copy rebuild an empty instance; __setstate__ restores it
            return super().__new__(cls)
        if isinstance(obj, cls):
            if hints:
                raise ValueError("ProtochromaView rewrapping does not accept hints")
            return obj
        backend = cls._prepare_backend(obj, hints)
        if not isinstance(backend, ProtochromaBackend):
            raise TypeError(f"Cannot recognize {type(backend).__name__} as a protochroma source")
        instance = super().__new__(cls)
        instance._backend = backend
        instance._resolved_protochroma = None
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_protochroma")()
        return object.__getattribute__(self, name)

    def _effective_protochroma(self) -> Protochroma:
        cached = object.__getattribute__(self, "_resolved_protochroma")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is Protochroma:
            resolved = backend
        elif isinstance(backend, DerivedProtochroma):
            resolved = backend.resolve()
        else:
            resolved = Protochroma(backend.spacegroup, backend.occupations)
        state = dict(resolved.__dict__)
        state["_resolved_protochroma"] = resolved
        object.__getattribute__(self, "__dict__").update(state)
        return resolved

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Protochroma:
        """Return the erased protochroma as a standalone value.

        :return: The protochroma value.
        """
        return self._effective_protochroma()

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {"backend": self._backend}
        if self._resolved_protochroma is not None:
            state["resolved"] = self._resolved_protochroma
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._resolved_protochroma = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_protochroma"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
