"""Lazy prototype recognition and presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.bareprototype.recognized import RecognizedBarePrototype
from httk.atomistic.models.bareprototype.view_base import BarePrototypeViewBase


class BarePrototypeView(BarePrototypeViewBase, BarePrototype):
    r"""Recognize or project a lazy Wyckoff-only classification.

    Existing bare values, refined classifications, plain labels, and structure-like
    sources are accepted. Recognition hints are retained until first field access.
    Projecting a refined source preserves it for view round-trips; ``unview()``
    materializes a standalone bare value without geometrical refinement.

    :param obj: Classification or structure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: BarePrototypeBackend
    _resolved_prototype: BarePrototype | None
    _tolerance: float | None
    _limit_denominator: int | None
    _DEFERRED_FIELDS = frozenset({"_spacegroup", "_occupations"})

    def __new__(
        cls,
        obj: Any = MISSING,
        *,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
        **hints: Any,
    ) -> Self:
        if obj is MISSING:  # pickle/copy rebuild an empty instance; __setstate__ restores it
            return super().__new__(cls)
        if isinstance(obj, cls):
            if any(value is not None for value in (tolerance, limit_denominator)) or hints:
                raise ValueError("BarePrototypeView rewrapping does not accept recognition arguments")
            return obj

        backend_hints = dict(hints)
        if tolerance is not None:
            backend_hints["tolerance"] = tolerance
        if limit_denominator is not None:
            backend_hints["limit_denominator"] = limit_denominator
        backend = cls._prepare_backend(obj, backend_hints)
        if not isinstance(backend, RecognizedBarePrototype):
            if not isinstance(backend, BarePrototypeBackend):
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a prototype source")
            if any(value is not None for value in (tolerance, limit_denominator)) or hints:
                raise ValueError("BarePrototypeView recognition arguments cannot be used with a prototype")
        instance = super().__new__(cls)
        instance._backend = backend
        instance._resolved_prototype = None
        instance._tolerance = tolerance
        instance._limit_denominator = limit_denominator
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_prototype")()
        return object.__getattribute__(self, name)

    def _effective_prototype(self) -> BarePrototype:
        cached = object.__getattribute__(self, "_resolved_prototype")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is BarePrototype:
            resolved = backend
        elif isinstance(backend, RecognizedBarePrototype) or hasattr(backend, "resolve"):
            resolved = backend.resolve()
        else:
            # A generic backend can carry class identity even though its base
            # Wyckoff data are all this view needs for presentation.
            resolved = BarePrototype(
                backend.spacegroup,
                backend.occupations,
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

    def unview(self) -> BarePrototype:
        """Return the recognized prototype as a standalone value.

        :return: The prototype value.
        """
        return self._effective_prototype()

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "backend": self._backend,
            "tolerance": self._tolerance,
            "limit_denominator": self._limit_denominator,
        }
        if self._resolved_prototype is not None:
            state["resolved"] = self._resolved_prototype
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._tolerance = state["tolerance"]
        self._limit_denominator = state["limit_denominator"]
        self._resolved_prototype = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_prototype"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
