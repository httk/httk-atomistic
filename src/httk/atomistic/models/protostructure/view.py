"""Lazy refined protostructure presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase


class ProtostructureView(ProtostructureViewBase, Protostructure):
    r"""Present an existing refined protostructure lazily.

    Raw structures and plain labels require a bare view; refinement must be constructed
    explicitly with a representative or discriminator. A bare projection view retaining
    a refined source can recover that source here.

    :param obj: Existing refined classification.
    :param \*\*hints: Reserved hints; recognition arguments are rejected.
    """

    _backend: ProtostructureBackend
    _resolved_protostructure: Protostructure | None
    _DEFERRED_FIELDS = frozenset({"_spacegroup", "_occupations", "_representative", "_discriminator"})

    def __new__(cls, obj: Any = MISSING, **hints: Any) -> Self:
        if obj is MISSING:
            return super().__new__(cls)
        if hints:
            raise ValueError("ProtostructureView does not accept recognition arguments")
        if isinstance(obj, cls):
            return obj
        instance = super().__new__(cls)
        instance._backend = cls._prepare_backend(obj, hints)
        instance._resolved_protostructure = None
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_protostructure")()
        return object.__getattribute__(self, name)

    def _effective_protostructure(self) -> Protostructure:
        cached = object.__getattribute__(self, "_resolved_protostructure")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is Protostructure:
            resolved = backend
        elif hasattr(backend, "resolve"):
            resolved = backend.resolve()
        else:
            resolved = Protostructure(
                backend.spacegroup,
                backend.occupations,
                representative=backend.representative,
                discriminator=backend.discriminator,
            )
        state = dict(resolved.__dict__)
        state["_resolved_protostructure"] = resolved
        object.__getattribute__(self, "__dict__").update(state)
        return resolved

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Protostructure:
        """Return the refined protostructure as a standalone value.

        :return: The protostructure value.
        """
        return self._effective_protostructure()

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "backend": self._backend,
        }
        if self._resolved_protostructure is not None:
            state["resolved"] = self._resolved_protostructure
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._resolved_protostructure = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_protostructure"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
