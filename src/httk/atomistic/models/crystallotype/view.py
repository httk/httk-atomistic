"""Lazy crystallotype recognition and presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.crystallotype.backend import CrystallotypeBackend
from httk.atomistic.models.crystallotype.crystallotype import Crystallotype
from httk.atomistic.models.crystallotype.recognized import RecognizedCrystallotype
from httk.atomistic.models.crystallotype.view_base import CrystallotypeViewBase


class CrystallotypeView(CrystallotypeViewBase, Crystallotype):
    r"""Present a lazy assigned geometrical-class crystallotype view.

    Sources may be an existing crystallotype or a structure-like source recognized to a
    representative-carrying crystallotype. Recognition of a raw structure accepts optional
    ``tolerance`` and ``limit_denominator`` values; resolution is deferred until the first
    field access.

    :param obj: The crystallotype-like or structure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: CrystallotypeBackend
    _resolved_crystallotype: Crystallotype | None
    _tolerance: float | None
    _limit_denominator: int | None
    _DEFERRED_FIELDS = frozenset({"_protostructure", "_representative", "_discriminator"})

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
                raise ValueError("CrystallotypeView rewrapping does not accept recognition arguments")
            return obj

        backend_hints = dict(hints)
        if tolerance is not None:
            backend_hints["tolerance"] = tolerance
        if limit_denominator is not None:
            backend_hints["limit_denominator"] = limit_denominator
        backend = cls._prepare_backend(obj, backend_hints)
        if not isinstance(backend, RecognizedCrystallotype):
            if not isinstance(backend, CrystallotypeBackend):
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a crystallotype source")
            if any(value is not None for value in (tolerance, limit_denominator)) or hints:
                raise ValueError("CrystallotypeView recognition arguments cannot be used with a crystallotype")
        instance = super().__new__(cls)
        instance._backend = backend
        instance._resolved_crystallotype = None
        instance._tolerance = tolerance
        instance._limit_denominator = limit_denominator
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_crystallotype")()
        return object.__getattribute__(self, name)

    def _effective_crystallotype(self) -> Crystallotype:
        cached = object.__getattribute__(self, "_resolved_crystallotype")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is Crystallotype:
            resolved = backend
        elif isinstance(backend, RecognizedCrystallotype):
            resolved = backend.resolve()
        else:
            resolved = Crystallotype(
                backend.protostructure,
                representative=backend.representative,
                discriminator=backend.discriminator,
            )
        state = dict(resolved.__dict__)
        state["_resolved_crystallotype"] = resolved
        object.__getattribute__(self, "__dict__").update(state)
        return resolved

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Crystallotype:
        """Return the recognized crystallotype as a standalone value.

        :return: The crystallotype value.
        """
        return self._effective_crystallotype()

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "backend": self._backend,
            "tolerance": self._tolerance,
            "limit_denominator": self._limit_denominator,
        }
        if self._resolved_crystallotype is not None:
            state["resolved"] = self._resolved_crystallotype
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._tolerance = state["tolerance"]
        self._limit_denominator = state["limit_denominator"]
        self._resolved_crystallotype = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_crystallotype"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
