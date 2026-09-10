"""Lazy protostructure recognition and presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.bareprotostructure.backend import BareProtostructureBackend
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprotostructure.recognized import RecognizedBareProtostructure
from httk.atomistic.models.bareprotostructure.view_base import BareProtostructureViewBase
from httk.atomistic.models.structure.asu import FundamentalDomainStructure


def _has_existing_asu(source: Any) -> bool:
    if isinstance(source, FundamentalDomainStructure):
        return True
    from httk.atomistic.models.structure.asu_view import ASUStructureView

    if isinstance(source, ASUStructureView):
        return True
    return isinstance(getattr(source, "_view", None), ASUStructureView)


class BareProtostructureView(BareProtostructureViewBase, BareProtostructure):
    r"""Recognize or project a lazy Wyckoff-only classification.

    Existing bare values, refined classifications, plain labels, and structure-like
    sources are accepted. Recognition hints are retained until first field access.
    Projecting a refined source preserves it for view round-trips; ``unview()``
    materializes a standalone bare value without geometrical refinement.

    :param obj: Classification or structure-like source.
    :param \*\*hints: Backend-selection and recognition hints.
    """

    _backend: BareProtostructureBackend
    _resolved_protostructure: BareProtostructure | None
    _setting: Any
    _standard: Any
    _transform: Any
    _tolerance: float | None
    _limit_denominator: int | None
    _DEFERRED_FIELDS = frozenset({"_spacegroup", "_occupations"})

    def __new__(
        cls,
        obj: Any = MISSING,
        *,
        setting: Any = None,
        standard: Any = None,
        transform: Any = None,
        tolerance: float | None = None,
        limit_denominator: int | None = None,
        **hints: Any,
    ) -> Self:
        if obj is MISSING:  # pickle/copy rebuild an empty instance; __setstate__ restores it
            return super().__new__(cls)
        if isinstance(obj, cls):
            if (
                any(value is not None for value in (setting, standard, transform, tolerance, limit_denominator))
                or hints
            ):
                raise ValueError("BareProtostructureView rewrapping does not accept recognition arguments")
            return obj

        # Prototype-family inputs have dummy species; report the domain mismatch before backend probing.
        from httk.atomistic.models.structuretype.backend import StructuretypeBackend
        from httk.atomistic.models.structuretype.view_base import StructuretypeViewBase

        if isinstance(obj, (StructuretypeBackend, StructuretypeViewBase)):
            raise TypeError("a prototype or structuretype carries dummy species; a protostructure needs the real ones")

        recognition_values = (setting, standard, transform, tolerance, limit_denominator)
        backend_hints = dict(hints)
        for name, value in zip(
            ("setting", "standard", "transform", "tolerance", "limit_denominator"), recognition_values
        ):
            if value is not None:
                backend_hints[name] = value
        backend = cls._prepare_backend(obj, backend_hints)
        if isinstance(backend, RecognizedBareProtostructure):
            structure = backend._structure
            if _has_existing_asu(structure) and any(value is not None for value in recognition_values):
                raise ValueError("BareProtostructureView recognition arguments cannot be used with an existing ASU")
        else:
            if not isinstance(backend, BareProtostructureBackend):
                raise TypeError(f"Cannot recognize {type(backend).__name__} as a protostructure source")
            if any(value is not None for value in recognition_values) or hints:
                raise ValueError("BareProtostructureView recognition arguments cannot be used with a protostructure")
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

    def _effective_protostructure(self) -> BareProtostructure:
        cached = object.__getattribute__(self, "_resolved_protostructure")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is BareProtostructure:
            resolved = backend
        elif isinstance(backend, RecognizedBareProtostructure):
            resolved = backend.resolve()
        else:
            resolved = BareProtostructure(
                backend.spacegroup,
                backend.occupations,
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

    def unview(self) -> BareProtostructure:
        """Return the recognized protostructure as a standalone value.

        :return: The protostructure value.
        """
        return self._effective_protostructure()

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
