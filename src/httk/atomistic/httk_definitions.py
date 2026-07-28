"""Loading the vendored property definitions published by schemas.httk.org.

OPTIMADE standardises a good deal about a structure but nothing about *which setting* it is
written in, nor about how precisely its numbers were stated. Both are served here as
database-specific properties, and rather than describing them locally the descriptions are
taken from the published definitions at `schemas.httk.org <https://schemas.httk.org>`_,
vendored verbatim in ``httk_defs/``.

The served *name* carries the ``_httk_`` prefix that OPTIMADE requires of a
database-specific property; the definition keeps its own ``$id``, so a client following the
link reaches the authoritative schema rather than a local paraphrase.
"""

import json
from collections.abc import Mapping
from importlib.resources import files

from httk.core import PropertyDefinition

__all__ = ["load_httk_definitions"]


def load_httk_definitions(names: Mapping[str, str]) -> dict[str, PropertyDefinition]:
    """Load vendored definitions, given a map of served name to definition file stem.

    Each document is loaded verbatim, ``$id`` included. Note that a definition's own
    ``x-optimade-definition.name`` stays the unprefixed name it was published under, which
    will differ from the prefixed name httk serves it as — that is correct, and rewriting a
    published document to match a local naming choice would defeat the point of pointing at
    it.
    """
    definitions: dict[str, PropertyDefinition] = {}
    for served_name, file_name in names.items():
        text = files("httk.atomistic").joinpath(f"httk_defs/{file_name}.json").read_text(encoding="utf-8")
        definitions[served_name] = PropertyDefinition.from_optimade(served_name, json.loads(text))
    return definitions
