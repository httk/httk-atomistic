"""The abstract protonetstructure backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.protonetstructure.api import ProtonetstructureAPI


class ProtonetstructureBackend(Backend["ProtonetstructureBackend"], ProtonetstructureAPI):
    """Backend root for assigned-species Wyckoff classifications with a covalent bonding net."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
