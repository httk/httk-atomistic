"""A lazy view presenting any structure as its asymmetric unit."""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.models.structure.semantics import _METADATA_UNSET, _resolve_view_metadata
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.storage.records import ASUStructureRecord, FundamentalDomainStructureRecord
from httk.atomistic.symmetry.recognition import recognize_asu
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup


def _validate_recognition_options(
    setting: Spacegroup | None,
    standard: Spacegroup | None,
    transform: SettingTransform | None,
) -> None:
    if setting is not None:
        if standard is not None or transform is not None:
            raise TypeError("recognize_asu() takes either 'setting' or 'standard'/'transform', not both")
    elif standard is not None or transform is not None:
        if standard is None or transform is None:
            raise TypeError("recognize_asu() needs both 'standard' and 'transform' when either is given")
        if not standard.is_standard_setting:
            raise ValueError(f"'standard' must be an IT standard setting, got {standard.setting}")


class _ASUResolverBackend(StructureBackend):
    """Carry an ASU view's deferred recognition through later structure views."""

    def __init__(self, source_backend: StructureBackend, view: Any) -> None:
        self._source_backend = source_backend
        self._view = view

    @property
    def cell(self) -> Any:
        return self.resolve().cell

    @property
    def sites(self) -> Any:
        return self.resolve().sites

    @property
    def species(self) -> Any:
        return self.resolve().species

    @property
    def species_at_sites(self) -> Any:
        return self.resolve().species_at_sites

    def resolve(self) -> ASUStructure:
        return self._view._effective_asu()

    def unwrap(self) -> Any:
        return unwrap(self._source_backend)


class ASUStructureView(StructureView, ASUStructure):
    r"""Present an underlying structure backend as a lazy :class:`~httk.atomistic.ASUStructure`.

    Resolver-backed and non-native sources are retained without recognition until the first
    asymmetric-unit access. The view then publishes the complete validated ASU state on itself,
    so its inherited API remains the genuine ASUStructure interface. Pickling retains the source
    backend and view options; once resolved, it also retains the validated derived ASU state while
    preserving that backend as the source returned by :meth:`unwrap`.

    :param obj: The structure backend or source to recognize and present.
    :param setting: The source structure's tabulated space-group setting.
    :param standard: The IT-standard space group for an untabulated setting.
    :param transform: The standard-to-source setting transform.
    :param tolerance: The Cartesian recognition tolerance.
    :param immutable_id: The optional immutable source identifier override.
    :param last_modified: The optional source modification timestamp override.
    :param \**hints: Backend-selection and reader hints.
    """

    _backend: StructureBackend
    _source_backend: StructureBackend
    _resolved_asu: ASUStructure | None
    _setting: Spacegroup | None
    _standard: Spacegroup | None
    _recognition_transform: SettingTransform | None
    _tolerance: float | None
    _deferred_immutable_id: str | None | object
    _deferred_last_modified: Any
    _pending_asu: ASUStructure | None
    _DEFERRED_FIELDS = frozenset(
        {
            "_cell",
            "_spacegroup",
            "_transform",
            "_coordinate_precision",
            "_charge",
            "_wyckoff_sites",
            "_species",
            "_molecular",
            "_assemblies",
            "_symmetry",
            "_chemical_composition",
            "_chemical_formula_descriptive",
            "_chemical_formula_hill",
            "_optimization_type",
            "_immutable_id",
            "_last_modified",
        }
    )

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
        explicit_setting = setting is not None
        explicit_standard_family = standard is not None or transform is not None
        _validate_recognition_options(setting, standard, transform)
        has_recognition_options = any(value is not None for value in (setting, standard, transform, tolerance)) or bool(
            hints
        )
        if (
            isinstance(obj, cls)
            and not has_recognition_options
            and immutable_id is _METADATA_UNSET
            and last_modified is _METADATA_UNSET
            and not hints
        ):
            return obj

        if isinstance(obj, cls):
            backend = obj._source_backend
            inherited = obj._resolved_asu
            if explicit_setting:
                standard = None
                transform = None
            elif explicit_standard_family:
                setting = None
            else:
                setting = obj._setting
                standard = obj._standard
                transform = obj._recognition_transform
            tolerance = obj._tolerance if tolerance is None else tolerance
            if immutable_id is _METADATA_UNSET:
                immutable_id = obj._deferred_immutable_id
            if last_modified is _METADATA_UNSET:
                last_modified = obj._deferred_last_modified
        else:
            backend = cls._prepare_backend(obj, hints)
            inherited = None
            if isinstance(backend, _ASUResolverBackend):
                source_view = backend._view
                if explicit_setting:
                    standard = None
                    transform = None
                elif explicit_standard_family:
                    setting = None
                else:
                    setting = source_view._setting
                    standard = source_view._standard
                    transform = source_view._recognition_transform
                tolerance = source_view._tolerance if tolerance is None else tolerance

        _validate_recognition_options(setting, standard, transform)

        instance = super().__new__(cls)
        instance._source_backend = backend
        instance._setting = setting
        instance._standard = standard
        instance._recognition_transform = transform
        instance._tolerance = tolerance
        instance._deferred_immutable_id = immutable_id
        instance._deferred_last_modified = last_modified
        instance._resolved_asu = None
        instance._pending_asu = None

        if isinstance(backend, FundamentalDomainStructure) and not isinstance(backend, ASUStructure):
            raise ValueError(
                "ASUStructureView cannot promote a fundamental domain to an asymmetric unit; "
                "construct ASUStructure explicitly to assert the stronger representation"
            )
        if (
            isinstance(backend, RecordStructure)
            and isinstance(backend._record, FundamentalDomainStructureRecord)
            and not isinstance(backend._record, ASUStructureRecord)
        ):
            raise ValueError(
                "ASUStructureView cannot promote a fundamental domain to an asymmetric unit; "
                "construct ASUStructure explicitly to assert the stronger representation"
            )

        if inherited is not None or isinstance(backend, ASUStructure):
            if inherited is not None:
                asu = inherited
            else:
                assert isinstance(backend, ASUStructure)
                asu = backend
            instance._pending_asu = asu
            instance._backend = backend
            return instance

        instance._backend = _ASUResolverBackend(backend, instance)
        return instance

    def __init__(
        self,
        obj: StructureLike,
        *,
        setting: Spacegroup | None = None,
        standard: Spacegroup | None = None,
        transform: SettingTransform | None = None,
        tolerance: float | None = None,
        immutable_id: str | None | object = _METADATA_UNSET,
        last_modified: Any = _METADATA_UNSET,
        **hints: Any,
    ) -> None:
        pass

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._DEFERRED_FIELDS:
            namespace = object.__getattribute__(self, "__dict__")
            if name not in namespace:
                object.__getattribute__(self, "_effective_asu")()
        return object.__getattribute__(self, name)

    @staticmethod
    def _copy_asu(
        asu: ASUStructure,
        immutable_id: str | None,
        last_modified: Any,
    ) -> ASUStructure:
        return ASUStructure(
            asu.cell,
            asu.spacegroup,
            asu.wyckoff_sites,
            asu.species,
            asu.transform,
            asu.coordinate_precision,
            molecular=asu.molecular,
            assemblies=asu.assemblies,
            chemical_composition=asu.chemical_composition,
            chemical_formula_descriptive=asu.chemical_formula_descriptive,
            chemical_formula_hill=asu.chemical_formula_hill,
            optimization_type=asu.optimization_type,
            immutable_id=immutable_id,
            last_modified=last_modified,
            charge=asu.charge,
        )

    def _publish_asu(self, asu: ASUStructure) -> None:
        asu._validate_expansion_semantics()
        _ = asu._expansion
        state = dict(asu.__dict__)
        state["_resolved_asu"] = asu
        object.__getattribute__(self, "__dict__").update(state)
        object.__getattribute__(self, "__dict__")["_pending_asu"] = None

    def _effective_asu(self) -> ASUStructure:
        cached = self._resolved_asu
        if cached is not None:
            return cached

        asu: Any
        resolved: Any
        pending = self._pending_asu
        if pending is not None:
            asu = pending
            resolved = pending
        else:
            backend = self._source_backend
            resolver = getattr(backend, "resolve", None)
            resolved = resolver() if resolver is not None else backend
            asu = resolved if isinstance(resolved, ASUStructure) else getattr(resolved, "asu", None)
        if isinstance(asu, FundamentalDomainStructure) and not isinstance(asu, ASUStructure):
            raise ValueError(
                "ASUStructureView cannot promote a fundamental domain to an asymmetric unit; "
                "construct ASUStructure explicitly to assert the stronger representation"
            )
        if asu is None:
            asu = recognize_asu(
                resolved,
                setting=self._setting,
                standard=self._standard,
                transform=self._recognition_transform,
                tolerance=self._tolerance,
            )
        immutable_id, last_modified = _resolve_view_metadata(
            resolved,
            immutable_id=self._deferred_immutable_id,
            last_modified=self._deferred_last_modified,
        )
        if (asu.immutable_id, asu.last_modified) == (immutable_id, last_modified):
            materialized = asu
        else:
            materialized = self._copy_asu(asu, immutable_id, last_modified)
        self._publish_asu(materialized)
        return materialized

    def _pickle_backend(self) -> StructureBackend:
        backend = self._source_backend
        while isinstance(backend, _ASUResolverBackend):
            backend = backend._source_backend
        return backend

    def resolve(self) -> ASUStructure:
        """Resolve and return the complete standalone asymmetric unit."""
        return self._effective_asu()

    def unwrap(self) -> Any:
        """Return the original source without resolving it."""
        return unwrap(self._source_backend)

    def unview(self) -> ASUStructure:
        """Return the resolved standalone asymmetric-unit structure."""
        return self._effective_asu()

    @property
    def sites(self) -> Any:
        """Expose the representative sites retained by the asymmetric-unit view."""
        return self._representative_sites()

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose species names for the representative sites retained by the view."""
        return self.domain_species_at_sites

    @property
    def assemblies(self) -> Any:
        """Expose correlations among the retained domain sites."""
        return self._assemblies

    @property
    def asu(self) -> FundamentalDomainStructure:
        """Expose this view as its own resolved fundamental domain."""
        self._effective_asu()
        return self

    def __eq__(self, other: object) -> bool:
        self._effective_asu()
        if isinstance(other, ASUStructureView):
            other._effective_asu()
        return ASUStructure.__eq__(self, other)

    @staticmethod
    def _pickle_new() -> "ASUStructureView":
        return object.__new__(ASUStructureView)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
        return type(self)._pickle_new, (), self.__getstate__()

    def __getstate__(self) -> dict[str, Any]:
        backend = self._pickle_backend()
        state = {
            "backend": backend,
            "setting": self._setting,
            "standard": self._standard,
            "transform": self._recognition_transform,
            "tolerance": self._tolerance,
            "immutable_id": None if self._deferred_immutable_id is _METADATA_UNSET else self._deferred_immutable_id,
            "immutable_id_unset": self._deferred_immutable_id is _METADATA_UNSET,
            "last_modified": None if self._deferred_last_modified is _METADATA_UNSET else self._deferred_last_modified,
            "last_modified_unset": self._deferred_last_modified is _METADATA_UNSET,
        }
        if self._resolved_asu is not None:
            state["resolved"] = self._resolved_asu
        elif self._pending_asu is not None and not isinstance(backend, ASUStructure):
            state["pending"] = self._pending_asu
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        backend = state["backend"]
        self._source_backend = backend
        self._backend = backend if isinstance(backend, ASUStructure) else _ASUResolverBackend(backend, self)
        self._resolved_asu = None
        self._pending_asu = backend if isinstance(backend, ASUStructure) else state.get("pending")
        self._setting = state["setting"]
        self._standard = state["standard"]
        self._recognition_transform = state["transform"]
        self._tolerance = state["tolerance"]
        self._deferred_immutable_id = _METADATA_UNSET if state["immutable_id_unset"] else state["immutable_id"]
        self._deferred_last_modified = _METADATA_UNSET if state["last_modified_unset"] else state["last_modified"]
        resolved = state.get("resolved")
        if resolved is not None:
            self._publish_asu(resolved)
