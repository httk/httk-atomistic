"""The abstract protostructure backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.protostructure.api import ProtostructureAPI


class ProtostructureBackend(Backend["ProtostructureBackend"], ProtostructureAPI):
    """Backend root for standard-setting assigned-species classification keys."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
