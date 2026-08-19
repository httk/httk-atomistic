"""The abstract protopattern backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.protopattern.api import ProtopatternAPI


class ProtopatternBackend(Backend["ProtopatternBackend"], ProtopatternAPI):
    """Backend root for standard-setting anonymous crystal keys."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
