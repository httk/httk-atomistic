"""Trajectory models and views."""

from typing import TYPE_CHECKING

from .api import TrajectoryAPI
from .backend import TrajectoryBackend
from .jsonl import JsonlTrajectory
from .like import TrajectoryLike
from .plain import PlainTrajectory
from .trajectory import Trajectory
from .view import TrajectoryView

if TYPE_CHECKING:
    from .record import RecordTrajectory

__all__ = [
    "JsonlTrajectory",
    "PlainTrajectory",
    "RecordTrajectory",
    "Trajectory",
    "TrajectoryAPI",
    "TrajectoryBackend",
    "TrajectoryLike",
    "TrajectoryView",
]


def __getattr__(name: str) -> object:
    if name == "RecordTrajectory":
        from .record import RecordTrajectory

        globals()[name] = RecordTrajectory
        return RecordTrajectory
    raise AttributeError(name)
