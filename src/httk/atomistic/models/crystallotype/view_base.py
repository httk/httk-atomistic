"""The abstract crystallotype view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.crystallotype.backend import CrystallotypeBackend


class CrystallotypeViewBase(View[CrystallotypeBackend]):
    """Base class for views presenting crystallotype backends."""

    _backend_base_cls: ClassVar[type[CrystallotypeBackend]] = CrystallotypeBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


CrystallotypeViewBase._view_base_cls = CrystallotypeViewBase
