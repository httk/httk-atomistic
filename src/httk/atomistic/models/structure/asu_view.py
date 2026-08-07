"""A view presenting any structure as its asymmetric unit."""

from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import _METADATA_UNSET, _resolve_view_metadata
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.symmetry.recognition import recognize_asu
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

if TYPE_CHECKING:
    from httk.atomistic.composition import Assembly
    from httk.atomistic.models.sites.sites import Sites


class ASUStructureView(StructureView, ASUStructure):
    r"""A view presenting an underlying structure backend as an :class:`~httk.atomistic.ASUStructure`.

    This view is a genuine ASUStructure, so it can be passed anywhere one is accepted.

    Unlike the other structure views, building it can require *work* rather than a change
    of presentation: a backend that already carries an asymmetric unit is adopted as-is,
    but a plain list of atoms has to have its symmetry recognized first. That step is
    tolerant where everything else in this package is exact — see
    :mod:`~httk.atomistic.symmetry.recognition` — and needs spglib unless the space group is
    supplied through ``setting`` or ``standard``/``transform``.

    ``tolerance`` left unspecified is derived from how precisely the structure was stated.

    :param obj: The structure backend or source to recognize and present.
    :param \**kwargs: Backend-selection, recognition, and metadata options passed to construction.
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
            asu.wyckoff_sites,
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
            charge=asu.charge,
        )
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **kwargs: Any) -> None:
        pass

    @property
    def sites(self) -> "Sites":
        """Expose the retained representative sites."""
        return self._representative_sites()

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose species names for the retained representative sites."""
        return self.domain_species_at_sites

    @property
    def assemblies(self) -> tuple["Assembly", ...] | None:
        """Expose correlations among the retained domain sites."""
        return self._assemblies

    def unwrap(self) -> Any:
        """Return the raw value wrapped by the backend.

        :return: The original source value.
        """
        return unwrap(self._backend)

    def unview(self) -> ASUStructure:
        """Materialize this presentation as a standalone asymmetric-unit structure.

        :return: The exact asymmetric-unit structure represented by this view.
        """
        # A genuine ASUStructure backend carrying the same metadata is exactly the presented
        # value: reuse it. Otherwise materialize a plain ASUStructure from the view's own
        # (eagerly initialized) state.
        backend = self._backend
        if type(backend) is ASUStructure and (self.immutable_id, self.last_modified) == (
            backend.immutable_id,
            backend.last_modified,
        ):
            return backend
        return ASUStructure(
            self.cell,
            self.spacegroup,
            self.wyckoff_sites,
            self.species,
            self.transform,
            self.coordinate_precision,
            molecular=self.molecular,
            assemblies=self._assemblies,
            chemical_composition=self.chemical_composition,
            chemical_formula_descriptive=self.chemical_formula_descriptive,
            chemical_formula_hill=self.chemical_formula_hill,
            optimization_type=self.optimization_type,
            immutable_id=self.immutable_id,
            last_modified=self.last_modified,
            charge=self.charge,
        )
