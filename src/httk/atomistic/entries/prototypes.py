"""Logical entry families for the geometry-free and dummy-species prototype records.

These are non-instantiable family markers used as storage-layout keys and as the
targets of :func:`~httk.core.register.register_entry_family`. OPTIMADE serving
(definitions and providers) for these families is intentionally not part of this
module; store a :class:`~httk.atomistic.storage.records.ProtostructureRecord` or
:class:`~httk.atomistic.storage.records.PrototypeRecord` (or their source values)
directly.
"""

from typing import Any, Self

__all__ = ["ProtostructureEntry", "PrototemplateEntry", "PrototypeEntry", "StructuretypeEntry"]


class ProtostructureEntry:
    """Define the non-instantiable protostructure entry family."""

    type = "protostructures"

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("ProtostructureEntry is a logical entry family; store a protostructure representation directly")


class PrototypeEntry:
    """Define the non-instantiable prototype entry family."""

    type = "prototypes"

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("PrototypeEntry is a logical entry family; store a prototype representation directly")


class PrototemplateEntry:
    """Define the non-instantiable prototemplate entry family."""

    type = "prototemplates"

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("PrototemplateEntry is a logical entry family; store a prototemplate representation directly")


class StructuretypeEntry:
    """Define the non-instantiable structuretype entry family."""

    type = "structuretypes"

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("StructuretypeEntry is a logical entry family; store a structuretype representation directly")
