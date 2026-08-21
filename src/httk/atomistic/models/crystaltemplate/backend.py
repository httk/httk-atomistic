"""The abstract backend for crystal templates and prototypes."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.crystaltemplate.api import CrystalTemplateAPI


class CrystalTemplateBackend(Backend["CrystalTemplateBackend"], CrystalTemplateAPI):
    """Backend root for the dummy-species crystal-template family."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
