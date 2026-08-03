"""A view presenting any structure as its asymmetric unit."""

from typing import Any, Self, cast

from httk.core import unwrap

from .asu_recognition import recognize_asu
from .asu_structure import ASUStructure, FundamentalDomainStructure
from .setting_transform import SettingTransform
from .spacegroup import Spacegroup
from .structure_backend import StructureBackend
from .structure_like import StructureLike
from .structure_semantics import _METADATA_UNSET, _resolve_view_metadata
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

    ``tolerance`` left unspecified is derived from how precisely the structure was stated.
    """

    _backend: StructureBackend

    def __new__(
        cls,
        obj: StructureLike,
        *,
        setting: Spacegroup | None = None,
        standard: Spacegroup | None = None,
        transform: SettingTransform | None = None,
        tolerance: float | None = None,
        immutable_id: str | None | object = _METADATA_UNSET,
        last_modified: Any = _METADATA_UNSET,
        **hints: Any,
    ) -> Self:
        asu: FundamentalDomainStructure | None
        if isinstance(obj, cls):
            if immutable_id is _METADATA_UNSET and last_modified is _METADATA_UNSET:
                return obj
            resolved_immutable_id, resolved_last_modified = _resolve_view_metadata(
                obj,
                immutable_id=immutable_id,
                last_modified=last_modified,
            )
            if (resolved_immutable_id, resolved_last_modified) == (obj.immutable_id, obj.last_modified):
                return obj
            backend = obj._backend
            asu = obj
        else:
            backend = cls._prepare_backend(obj, hints)
            asu = cast(FundamentalDomainStructure | None, getattr(backend, "asu", None))
            resolved_immutable_id, resolved_last_modified = _resolve_view_metadata(
                obj,
                immutable_id=immutable_id,
                last_modified=last_modified,
            )

        # A backend that already holds an asymmetric unit is passed straight through: no
        # recognition, no tolerance, nothing lost. The precedent is CellParamsView reading
        # a backend's native parameters when it has them.
        if isinstance(asu, FundamentalDomainStructure) and not isinstance(asu, ASUStructure):
            raise ValueError(
                "ASUStructureView cannot promote a fundamental domain to an asymmetric unit; "
                "construct ASUStructure explicitly to assert the stronger representation"
            )
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
            asu.coordinate_precision,
            molecular=asu.molecular,
            assemblies=asu._assemblies,
            chemical_composition=asu.chemical_composition,
            chemical_formula_descriptive=asu.chemical_formula_descriptive,
            chemical_formula_hill=asu.chemical_formula_hill,
            optimization_type=asu.optimization_type,
            immutable_id=resolved_immutable_id,
            last_modified=resolved_last_modified,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **kwargs: Any) -> None:
        pass

    @property
    def sites(self):
        return self._representative_sites()

    @property
    def species_at_sites(self):
        return self.domain_species_at_sites

    @property
    def assemblies(self):
        return self._assemblies

    def unwrap(self) -> Any:
        return unwrap(self._backend)
