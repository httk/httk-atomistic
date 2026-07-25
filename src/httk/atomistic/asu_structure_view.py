"""A view presenting any structure as its asymmetric unit."""

from typing import Any, Self

from httk.core import unwrap

from .asu_recognition import DEFAULT_TOLERANCE, recognize_asu
from .asu_structure import ASUStructure
from .setting_transform import SettingTransform
from .spacegroup import Spacegroup
from .structure_backend import StructureBackend
from .structure_like import StructureLike
from .structure_view import StructureView


class ASUStructureView(StructureView, ASUStructure):
    """A view presenting an underlying structure backend as an :class:`~httk.atomistic.ASUStructure`.

    This view is a genuine ASUStructure, so it can be passed anywhere one is accepted.

    Unlike the other structure views, building it can require *work* rather than a change
    of presentation: a backend that already carries an asymmetric unit is adopted as-is,
    but a plain list of atoms has to have its symmetry recognized first. That step is
    tolerant where everything else in this package is exact — see
    :mod:`~httk.atomistic.asu_recognition` — and needs spglib unless the space group is
    supplied through ``setting`` or ``standard``/``transform``.
    """

    _backend: StructureBackend

    def __new__(
        cls,
        obj: StructureLike,
        *,
        setting: Spacegroup | None = None,
        standard: Spacegroup | None = None,
        transform: SettingTransform | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        **hints: Any,
    ) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)

        # A backend that already holds an asymmetric unit is passed straight through: no
        # recognition, no tolerance, nothing lost. The precedent is CellParamsView reading
        # a backend's native parameters when it has them.
        asu = getattr(backend, "asu", None)
        if asu is None:
            asu = recognize_asu(
                backend,
                setting=setting,
                standard=standard,
                transform=transform,
                tolerance=tolerance,
            )

        instance = super().__new__(cls)
        # ASUStructure is mutable, so its state is initialized here in __new__ (keeping
        # __init__ a no-op), so that rewrapping an existing view does not re-initialize it.
        ASUStructure.__init__(
            instance,
            asu.cell,
            asu.spacegroup,
            asu.asu_sites,
            asu.species,
            asu.transform,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **kwargs: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)
