"""Present trajectory backends through the lazy canonical view."""

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from httk.core import View, unwrap

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.api import TrajectoryAPI
from httk.atomistic.models.trajectory.backend import TrajectoryBackend


class TrajectoryView(View[TrajectoryBackend], TrajectoryAPI):
    r"""Present any trajectory backend through the canonical trajectory API.

    :param obj: A trajectory backend or another accepted trajectory value.
    :param \**hints: Backend-selection hints.
    """

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
        """Return the number of frames."""
        return self._effective_backend().nframes

    def frame(self, i: int) -> UnitcellStructure:
        """Return one frame by index.

        :param i: Frame index.
        :return: The requested unit-cell structure.
        """
        return self._effective_backend().frame(i)

    def frames(self) -> Iterator[UnitcellStructure]:
        """Iterate over the frames.

        :return: An iterator of unit-cell structures.
        """
        return self._effective_backend().frames()

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        """Return stored reference-frame indexes, or ``None``."""
        return self._effective_backend().reference_frames

    @property
    def species(self) -> tuple[Species, ...]:
        """Return the constant distinct species."""
        return self._effective_backend().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the constant species name at each site."""
        return self._effective_backend().species_at_sites

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return the names of available per-frame observables."""
        return self._effective_backend().observable_names

    @property
    def observable_summaries(self) -> tuple[Any, ...]:
        """Return backend-provided observable summaries, if any."""
        return getattr(self._effective_backend(), "observable_summaries", ())

    @property
    def immutable_id(self) -> str | None:
        """Return the backend immutable identifier, if available."""
        return getattr(self._effective_backend(), "immutable_id", None)

    @property
    def last_modified(self) -> Any:
        """Return the backend modification marker, if available."""
        return getattr(self._effective_backend(), "last_modified", None)

    @property
    def source_locator(self) -> str | None:
        """Return the source locator, if available."""
        return getattr(self._effective_backend(), "source_locator", None)

    def observable(self, name: str) -> tuple[Any, ...]:
        """Return one observable's values in frame order.

        :param name: Observable name.
        :return: The observable values.
        :raises KeyError: If the observable is unavailable.
        """
        return self._effective_backend().observable(name)

    def unwrap(self) -> Any:
        """Return the original value wrapped by the backend."""
        return unwrap(self._backend)


TrajectoryView._view_base_cls = TrajectoryView
