"""Accepted trajectory inputs."""

from collections.abc import Mapping
from typing import Any

from httk.atomistic.models.trajectory.backend import TrajectoryBackend
from httk.atomistic.models.trajectory.plain import PlainTrajectory
from httk.atomistic.models.trajectory.trajectory import Trajectory
from httk.atomistic.models.trajectory.view import TrajectoryView

type TrajectoryLike = TrajectoryBackend | TrajectoryView | Trajectory | PlainTrajectory | Mapping[str, Any]
