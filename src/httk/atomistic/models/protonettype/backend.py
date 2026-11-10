"""The abstract protonettype backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.protonettype.api import ProtonettypeAPI


class ProtonettypeBackend(Backend["ProtonettypeBackend"], ProtonettypeAPI):
    """Backend root for anonymous Wyckoff prototypes with a covalent bonding net."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
