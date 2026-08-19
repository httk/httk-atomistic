"""Lazy protostructure recognition and presentation view."""

import copyreg
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.recognized import RecognizedProtostructure
from httk.atomistic.models.protostructure.view_base import ProtostructureViewBase
from httk.atomistic.models.structure.asu import FundamentalDomainStructure


def _has_existing_asu(source: Any) -> bool:
    if isinstance(source, FundamentalDomainStructure):
        return True
    from httk.atomistic.models.structure.asu_view import ASUStructureView

    if isinstance(source, ASUStructureView):
        return True
    return isinstance(getattr(source, "_view", None), ASUStructureView)


class ProtostructureView(ProtostructureViewBase, Protostructure):
    r"""Recognize a lazy standard-setting protostructure view.

    Recognition accepts optional ``setting``, ``standard``, ``transform``, ``tolerance``,
    and ``limit_denominator`` values through the recognition hints.

    :param obj: The structure-like or protostructure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: ProtostructureBackend
    _resolved_protostructure: Protostructure | None
    _setting: Any
    _standard: Any
    _transform: Any
    _tolerance: float | None
    _limit_denominator: int | None
    _DEFERRED_FIELDS = frozenset({"_spacegroup", "_occupations"})

    def __new__(
        cls,
        obj: Any,
        *,
        setting: Any = None,
        standard: Any = None,
        transform: Any = None,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
        **hints: Any,
    ) -> Self:
        if isinstance(obj, cls):
            if (
                any(value is not None for value in (setting, standard, transform, tolerance, limit_denominator))
                or hints
            ):
                raise ValueError("ProtostructureView rewrapping does not accept recognition arguments")
            return obj

        # Prototype-family inputs have dummy species; report the domain mismatch before backend probing.
        from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
        from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase

        if isinstance(obj, (AnonymousStructureBackend, AnonymousStructureViewBase)):
            raise TypeError(
                "a prototype/anonymous structure carries dummy species; a protostructure needs the real ones"
            )

        recognition_values = (setting, standard, transform, tolerance, limit_denominator)
        backend_hints = dict(hints)
        for name, value in zip(
            ("setting", "standard", "transform", "tolerance", "limit_denominator"), recognition_values
        ):
            if value is not None:
                backend_hints[name] = value
        backend = cls._prepare_backend(obj, backend_hints)
        if isinstance(backend, RecognizedProtostructure):
            structure = backend._structure
            if _has_existing_asu(structure) and any(value is not None for value in recognition_values):
                raise ValueError("ProtostructureView recognition arguments cannot be used with an existing ASU")
        else:
            if not isinstance(backend, ProtostructureBackend):
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a protostructure source")
            if any(value is not None for value in recognition_values) or hints:
                raise ValueError("ProtostructureView recognition arguments cannot be used with a protostructure")
            instance = super().__new__(cls)
            instance._backend = backend
            instance._resolved_protostructure = None
            instance._setting = setting
            instance._standard = standard
            instance._transform = transform
            instance._tolerance = tolerance
            instance._limit_denominator = limit_denominator
            return instance

        instance = super().__new__(cls)
        instance._backend = backend
        instance._resolved_protostructure = None
        instance._setting = setting
        instance._standard = standard
        instance._transform = transform
        instance._tolerance = tolerance
        instance._limit_denominator = limit_denominator
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
        elif isinstance(backend, RecognizedProtostructure):
            resolved = backend.resolve()
        else:
            resolved = Protostructure(backend.spacegroup, backend.occupations)
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
        """Return the recognized protostructure as a standalone value.

        :return: The protostructure value.
        """
        return self._effective_protostructure()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        # __new__ requires an ``obj`` argument, so bypass it via the stdlib
        # reconstructor (object.__new__(cls), no __init__); state is restored
        # through __setstate__.
        return copyreg._reconstructor, (type(self), object, None), self.__getstate__()  # type: ignore[attr-defined]

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "backend": self._backend,
            "setting": self._setting,
            "standard": self._standard,
            "transform": self._transform,
            "tolerance": self._tolerance,
            "limit_denominator": self._limit_denominator,
        }
        if self._resolved_protostructure is not None:
            state["resolved"] = self._resolved_protostructure
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._setting = state["setting"]
        self._standard = state["standard"]
        self._transform = state["transform"]
        self._tolerance = state["tolerance"]
        self._limit_denominator = state["limit_denominator"]
        self._resolved_protostructure = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_protostructure"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
