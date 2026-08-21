"""The abstract crystallotype backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.crystallotype.api import CrystallotypeAPI


class CrystallotypeBackend(Backend["CrystallotypeBackend"], CrystallotypeAPI):
    """Backend root for assigned geometrical-class crystallotypes."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
