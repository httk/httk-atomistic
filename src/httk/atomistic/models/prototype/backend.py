"""The abstract backend for anonymous structures and prototypes."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.prototype.api import AnonymousStructureAPI


class AnonymousStructureBackend(Backend["AnonymousStructureBackend"], AnonymousStructureAPI):
    """Backend root for the first PrototypeLike family."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
