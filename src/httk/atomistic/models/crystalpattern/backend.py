"""The abstract backend for crystal patterns and prototypes."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.crystalpattern.api import CrystalPatternAPI


class CrystalPatternBackend(Backend["CrystalPatternBackend"], CrystalPatternAPI):
    """Backend root for the dummy-species crystal-pattern family."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
