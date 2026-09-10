"""Lazy refined prototype presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.view_base import PrototypeViewBase


class PrototypeView(PrototypeViewBase, Prototype):
    r"""Present an existing refined prototype lazily.

    Raw structures and plain labels require a bare view; refinement must be constructed
    explicitly with a representative or discriminator. A bare projection view retaining
    a refined source can recover that source here.

    :param obj: Existing refined classification or assigned refined classification.
    :param \*\*hints: Reserved hints; recognition arguments are rejected.
    """

    _backend: PrototypeBackend
    _resolved_prototype: Prototype | None
    _DEFERRED_FIELDS = frozenset({"_spacegroup", "_occupations", "_representative", "_discriminator"})

    def __new__(cls, obj: Any = MISSING, **hints: Any) -> Self:
        if obj is MISSING:
            return super().__new__(cls)
        if hints:
            raise ValueError("PrototypeView does not accept recognition arguments")
        if isinstance(obj, cls):
            return obj
        instance = super().__new__(cls)
        instance._backend = cls._prepare_backend(obj, hints)
        instance._resolved_prototype = None
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_prototype")()
        return object.__getattribute__(self, name)

    def _effective_prototype(self) -> Prototype:
        cached = object.__getattribute__(self, "_resolved_prototype")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is Prototype:
            resolved = backend
        elif hasattr(backend, "resolve"):
            resolved = backend.resolve()
        else:
            # A generic backend can carry class identity even though its base
            # Wyckoff data are all this view needs for presentation.
            resolved = Prototype(
                backend.spacegroup,
                backend.occupations,
                representative=backend.representative,
                discriminator=backend.discriminator,
            )
        state = dict(resolved.__dict__)
        state["_resolved_prototype"] = resolved
        object.__getattribute__(self, "__dict__").update(state)
        return resolved

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Prototype:
        """Return the refined prototype as a standalone value.

        :return: The prototype value.
        """
        return self._effective_prototype()

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "backend": self._backend,
        }
        if self._resolved_prototype is not None:
            state["resolved"] = self._resolved_prototype
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._resolved_prototype = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_prototype"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
