"""The abstract prototype backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.bareprototype.api import BarePrototypeAPI


class BarePrototypeBackend(Backend["BarePrototypeBackend"], BarePrototypeAPI):
    """Backend root for anonymous Wyckoff-only prototypes."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
