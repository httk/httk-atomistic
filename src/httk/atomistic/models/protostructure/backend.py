"""The abstract protostructure backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.protostructure.api import ProtostructureAPI


class ProtostructureBackend(Backend["ProtostructureBackend"], ProtostructureAPI):
    """Backend root for standard-setting geometry-free crystal keys."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
