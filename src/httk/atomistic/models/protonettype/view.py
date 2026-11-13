"""Lazy protonettype presentation view."""

from typing import Any, Self

from httk.core import MISSING, unwrap

from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.protonettype.view_base import ProtonettypeViewBase


class ProtonettypeView(ProtonettypeViewBase, Protonettype):
    r"""Present a protonettype source as a lazy protonettype value.

    Existing values, canonical label strings, protonetstructures (erased by class
    position) and fundamental-domain structures with a bond network (see
    :meth:`~httk.atomistic.Protonettype.from_asu`) are accepted; a structure without a network is not, since no
    bonds are recognized.
    ``unview()`` materializes a standalone value.

    :param obj: Protonettype-like source.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ProtonettypeBackend
    _resolved_protonettype: Protonettype | None
    _DEFERRED_FIELDS = frozenset({"_undetermined", "_spacegroup", "_occupations", "_net", "_shapes"})

    def __new__(cls, obj: Any = MISSING, **hints: Any) -> Self:
        if obj is MISSING:  # pickle/copy rebuild an empty instance; __setstate__ restores it
            return super().__new__(cls)
        if isinstance(obj, cls):
            if hints:
                raise ValueError("ProtonettypeView rewrapping does not accept hints")
            return obj
        instance = super().__new__(cls)
        instance._backend = cls._prepare_backend(obj, hints)
        instance._resolved_protonettype = None
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_protonettype")()
        return object.__getattribute__(self, name)

    def _effective_protonettype(self) -> Protonettype:
        cached = object.__getattribute__(self, "_resolved_protonettype")
        if cached is not None:
            return cached
        backend = object.__getattribute__(self, "_backend")
        if type(backend) is Protonettype:
            resolved = backend
        elif hasattr(backend, "resolve"):
            resolved = backend.resolve()
        else:
            resolved = Protonettype(backend.spacegroup, backend.occupations, backend.net, backend.shapes)
        state = dict(resolved.__dict__)
        state["_resolved_protonettype"] = resolved
        object.__getattribute__(self, "__dict__").update(state)
        return resolved

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Protonettype:
        """Return the presented protonettype as a standalone value.

        :return: The protonettype value.
        """
        return self._effective_protonettype()

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {"backend": self._backend}
        if self._resolved_protonettype is not None:
            state["resolved"] = self._resolved_protonettype
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._backend = state["backend"]
        self._resolved_protonettype = None
        resolved = state.get("resolved")
        if resolved is not None:
            state_copy = dict(resolved.__dict__)
            state_copy["_resolved_protonettype"] = resolved
            object.__getattribute__(self, "__dict__").update(state_copy)
