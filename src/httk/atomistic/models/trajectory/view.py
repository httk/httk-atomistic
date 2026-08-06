"""The lazy canonical view of a trajectory backend."""

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from httk.core import View, unwrap

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.api import TrajectoryAPI
from httk.atomistic.models.trajectory.backend import TrajectoryBackend


class TrajectoryView(View[TrajectoryBackend], TrajectoryAPI):
    """Present any trajectory backend through the canonical trajectory API."""

    _backend_base_cls: ClassVar[type[TrajectoryBackend]] = TrajectoryBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]
    __httk_storage_record__: ClassVar[type[Any]]
    _backend: TrajectoryBackend
    _effective_backend_cache: TrajectoryBackend | None

    def __new__(cls, obj: Any, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        instance = super().__new__(cls)
        instance._backend = cls._prepare_backend(obj, hints)
        instance._effective_backend_cache = None
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def _effective_backend(self) -> TrajectoryBackend:
        backend = self._effective_backend_cache
        if backend is not None:
            return backend
        resolver = getattr(self._backend, "resolve", None)
        backend = self._backend if resolver is None else resolver()
        self._effective_backend_cache = backend
        return backend

    @property
    def nframes(self) -> int:
        return self._effective_backend().nframes

    def frame(self, i: int) -> UnitcellStructure:
        return self._effective_backend().frame(i)

    def frames(self) -> Iterator[UnitcellStructure]:
        return self._effective_backend().frames()

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        return self._effective_backend().reference_frames

    @property
    def species(self) -> tuple[Species, ...]:
        return self._effective_backend().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._effective_backend().species_at_sites

    @property
    def observable_names(self) -> tuple[str, ...]:
        return self._effective_backend().observable_names

    @property
    def observable_summaries(self) -> tuple[Any, ...]:
        return getattr(self._effective_backend(), "observable_summaries", ())

    @property
    def immutable_id(self) -> str | None:
        return getattr(self._effective_backend(), "immutable_id", None)

    @property
    def last_modified(self) -> Any:
        return getattr(self._effective_backend(), "last_modified", None)

    @property
    def source_locator(self) -> str | None:
        return getattr(self._effective_backend(), "source_locator", None)

    def observable(self, name: str) -> tuple[Any, ...]:
        return self._effective_backend().observable(name)

    def unwrap(self) -> Any:
        return unwrap(self._backend)


TrajectoryView._view_base_cls = TrajectoryView
