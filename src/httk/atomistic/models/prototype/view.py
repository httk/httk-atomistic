"""Lazy prototype recognition and presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.recognized import RecognizedPrototype
from httk.atomistic.models.prototype.view_base import PrototypeViewBase


class PrototypeView(PrototypeViewBase, Prototype):
    r"""Present a lazy anonymous geometrical-class prototype view.

    Sources may be an existing prototype, a :class:`~httk.atomistic.models.crystallotype.crystallotype.Crystallotype`
    (erased to its anonymous class), or a chromastructure-like/structure-like source recognized to a
    representative-carrying prototype. Recognition of a raw structure accepts optional
    ``tolerance`` and ``limit_denominator`` values; resolution is deferred until the first
    field access.

    :param obj: The prototype-like, crystallotype-like, or structure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: PrototypeBackend
    _resolved_prototype: Prototype | None
    _tolerance: float | None
    _limit_denominator: int | None
    _DEFERRED_FIELDS = frozenset({"_protochroma", "_representative", "_discriminator"})

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
                raise ValueError("PrototypeView rewrapping does not accept recognition arguments")
            return obj

        # A crystallotype erases lazily through RecognizedPrototype (adopted below); recognition
        # arguments are meaningless for it, so reject them up front like a prototype source.
        from httk.atomistic.models.crystallotype.backend import CrystallotypeBackend
        from httk.atomistic.models.crystallotype.view_base import CrystallotypeViewBase

        if isinstance(obj, (CrystallotypeBackend, CrystallotypeViewBase)) and (
            any(value is not None for value in (tolerance, limit_denominator)) or hints
        ):
            raise ValueError("PrototypeView recognition arguments cannot be used with a crystallotype")

        backend_hints = dict(hints)
        if tolerance is not None:
            backend_hints["tolerance"] = tolerance
        if limit_denominator is not None:
            backend_hints["limit_denominator"] = limit_denominator
        backend = cls._prepare_backend(obj, backend_hints)
        if not isinstance(backend, RecognizedPrototype):
            if not isinstance(backend, PrototypeBackend):
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a prototype source")
            if any(value is not None for value in (tolerance, limit_denominator)) or hints:
                raise ValueError("PrototypeView recognition arguments cannot be used with a prototype")
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

    def _effective_prototype(self) -> Prototype:
        cached = object.__getattribute__(self, "_resolved_prototype")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is Prototype:
            resolved = backend
        elif isinstance(backend, RecognizedPrototype):
            resolved = backend.resolve()
        else:
            resolved = Prototype(
                backend.protochroma,
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
