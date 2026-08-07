"""Define the abstract base class for trajectory backends."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.trajectory.api import TrajectoryAPI


class TrajectoryBackend(Backend["TrajectoryBackend"], TrajectoryAPI):
    """Define the backend base for all trajectory representations."""

    backend_classes: ClassVar[list[type[Backend[Any]]]] = []
