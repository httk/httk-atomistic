"""The storage record of a :class:`~httk.atomistic.CIFStructure`.

The block is stored as one canonical JSON document, so equal blocks have equal content ids and
the stored text cannot be queried by data name. Serving never converts the block: every standard
structure property is null and ``structure_features`` is ``[]``.
"""

import datetime
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, cast

from httk.core import DecimalGap, DecimalScalar, ReferenceRecord
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique, WeakLink

from httk.atomistic.io.cif.cif_items import (
    _MAX_NESTING,
    _TOO_DEEP,
    CIFComment,
    CIFDataItem,
    CIFDialect,
    CIFItem,
    CIFLoop,
    CIFSaveFrame,
    CIFTable,
    CIFTextField,
    CIFValue,
)
from httk.atomistic.models.structure.cif import CIFStructure
from httk.atomistic.storage.records import _require_canonical_form
from httk.atomistic.storage.stored_properties import cif_structure_properties

__all__ = ["CIFStructureRecord"]

# Bump when the JSON tree or the item model changes meaning; stored records then need a rebuild.
_CIF_ITEMS_FORM_VERSION = 1


def _comment_json(comment: CIFComment) -> dict[str, Any]:
    return {"comment": comment.text, "inline": comment.inline}


def _value_json(value: CIFValue) -> dict[str, Any]:
    if isinstance(value, str):
        return {"string": value}
    if isinstance(value, CIFTextField):
        return {"text": value.text}
    if isinstance(value, DecimalScalar):
        return {"number": value.token}
    if isinstance(value, DecimalGap):
        return {"gap": value.value}
    if isinstance(value, tuple):
        return {"list": [_value_json(entry) for entry in value]}
    return {"table": [[key, _value_json(entry)] for key, entry in value.entries]}


def _item_json(item: CIFItem) -> dict[str, Any]:
    if isinstance(item, CIFComment):
        return _comment_json(item)
    placed = [[position, _comment_json(comment)] for position, comment in getattr(item, "comments", ())]
    if isinstance(item, CIFDataItem):
        return {"name": item.name, "value": _value_json(item.value), "comments": placed}
    if isinstance(item, CIFLoop):
        return {
            "loop": list(item.names),
            "rows": [[_value_json(v) for v in row] for row in item.rows],
            "comments": placed,
        }
    return {"save": item.name, "items": [_item_json(entry) for entry in item.items]}


def _content_json(structure: CIFStructure) -> str:
    """Return the canonical JSON text of a block's header, leading comments and items."""
    document = {
        "header": [_comment_json(comment) for comment in structure.header],
        "leading": [_comment_json(comment) for comment in structure.leading],
        "items": [_item_json(item) for item in structure.items],
    }
    return json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _malformed(what: str) -> ValueError:
    return ValueError(f"CIFStructureRecord content_json has a malformed {what}")


def _object(obj: object, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or obj.keys() != keys:
        raise _malformed(what)
    return obj


def _list(obj: object, what: str) -> list[Any]:
    if not isinstance(obj, list):
        raise _malformed(what)
    return obj


def _str(obj: object, what: str) -> str:
    if not isinstance(obj, str):
        raise _malformed(what)
    return obj


def _comment(obj: object) -> CIFComment:
    fields = _object(obj, {"comment", "inline"}, "comment")
    if type(fields["inline"]) is not bool:
        raise _malformed("comment")
    return CIFComment(_str(fields["comment"], "comment"), fields["inline"])


def _placed(obj: object) -> tuple[tuple[int, CIFComment], ...]:
    placed = []
    for entry in _list(obj, "comment placement"):
        pair = _list(entry, "comment placement")
        if len(pair) != 2 or type(pair[0]) is not int:
            raise _malformed("comment placement")
        placed.append((pair[0], _comment(pair[1])))
    return tuple(placed)


def _value(obj: object, depth: int) -> CIFValue:
    if not isinstance(obj, dict) or len(obj) != 1:
        raise _malformed("value")
    ((kind, data),) = obj.items()
    if kind in ("list", "table") and depth >= _MAX_NESTING:
        raise ValueError(_TOO_DEEP)
    if kind == "string":
        return _str(data, "value")
    if kind == "text":
        return CIFTextField(_str(data, "value"))
    if kind == "number":
        return DecimalScalar(_str(data, "value"))
    if kind == "gap":
        return DecimalGap(_str(data, "value"))
    if kind == "list":
        return tuple(_value(entry, depth + 1) for entry in _list(data, "value"))
    if kind == "table":
        pairs = [_list(entry, "table entry") for entry in _list(data, "value")]
        if any(len(pair) != 2 for pair in pairs):
            raise _malformed("table entry")
        return CIFTable(tuple((_str(key, "table entry"), _value(entry, depth + 1)) for key, entry in pairs))
    raise _malformed("value")


def _item(obj: object, depth: int) -> CIFItem:
    if not isinstance(obj, dict):
        raise _malformed("item")
    keys = set(obj)
    if keys == {"comment", "inline"}:
        return _comment(obj)
    if keys == {"name", "value", "comments"}:
        return CIFDataItem(_str(obj["name"], "data item"), _value(obj["value"], depth), _placed(obj["comments"]))
    if keys == {"loop", "rows", "comments"}:
        names = tuple(_str(name, "loop") for name in _list(obj["loop"], "loop"))
        rows = tuple(tuple(_value(v, depth) for v in _list(row, "loop")) for row in _list(obj["rows"], "loop"))
        return CIFLoop(names, rows, _placed(obj["comments"]))
    if keys == {"save", "items"}:
        if depth >= _MAX_NESTING:
            raise ValueError(_TOO_DEEP)
        items = tuple(_item(entry, depth + 1) for entry in _list(obj["items"], "save frame"))
        return CIFSaveFrame(_str(obj["save"], "save frame"), items)
    raise _malformed("item")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("a JSON object repeats a key")
    return result


def _cif_structure_from_record(record: "CIFStructureRecord") -> CIFStructure:
    """Rebuild the CIF structure a record stores.

    :param record: The durable record.
    :return: The CIF structure.
    :raises ValueError: If the record has another canonical form, or its content is malformed.
    """
    _require_canonical_form(record, "cif", _CIF_ITEMS_FORM_VERSION)
    try:
        document = json.loads(record.content_json, object_pairs_hook=_unique_keys)
    except (RecursionError, ValueError) as error:  # json.loads recurses on nesting
        raise _malformed("document") from error
    fields = _object(document, {"header", "leading", "items"}, "document")
    return CIFStructure(
        record.block_name,
        tuple(_item(item, 0) for item in _list(fields["items"], "document")),
        tuple(_comment(comment) for comment in _list(fields["header"], "document")),
        tuple(_comment(comment) for comment in _list(fields["leading"], "document")),
        cast(CIFDialect, record.dialect),  # checked in __post_init__
    )


@dataclass(frozen=True)
class CIFStructureRecord:
    """Store a :class:`~httk.atomistic.CIFStructure` losslessly.

    Two records are equal, and have equal content ids, exactly when their blocks are equal:
    whitespace and quoting are not stored, comments and their placement are.

    :param block_name: The block name as written.
    :param dialect: ``"cif"`` or ``"mcif"``.
    :param content_json: The block's header, leading comments and items as canonical JSON.
    :param canonical_form: The version of that JSON form the record was built with.
    :param id: The human-readable entry id shared by all revisions; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    :param last_modified: The source modification timestamp, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_cif_structure",
        identity_name="atomistic_cif_structure",
        indexes=(("block_name",), ("last_modified",)),
        links=(WeakLink("references", ReferenceRecord, exposed_relationship=True),),
    )
    __httk_canonical_source__: ClassVar = CIFStructure

    block_name: str
    dialect: str
    content_json: str
    canonical_form: int = _CIF_ITEMS_FORM_VERSION
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), str) for name in ("block_name", "dialect", "content_json")):
            raise TypeError("CIFStructureRecord block_name, dialect and content_json are str")
        if type(self.canonical_form) is not int:
            raise TypeError("CIFStructureRecord canonical_form is an int")
        if self.dialect not in ("cif", "mcif"):
            raise ValueError(f"CIFStructureRecord dialect must be 'cif' or 'mcif', not {self.dialect!r}")

    @classmethod
    def __httk_validate__(cls, record: "CIFStructureRecord") -> None:
        """Validate that the record stores a valid block in its canonical JSON.

        :param record: The record to validate.
        :raises ValueError: If the record has another canonical form, its content is malformed,
            or its JSON is not the canonical text of the block it stores.
        """
        if _content_json(_cif_structure_from_record(record)) != record.content_json:
            raise ValueError("CIFStructureRecord content_json is not canonical; store a record built from its value")

    @classmethod
    def __httk_project__(cls, structure: CIFStructure) -> Mapping[str, object]:
        """Project a CIF structure into durable fields.

        :param structure: The CIF structure to project.
        :return: The projected record fields.
        """
        return {
            "block_name": structure.name,
            "dialect": structure.dialect,
            "content_json": _content_json(structure),
            "canonical_form": _CIF_ITEMS_FORM_VERSION,
            "id": None,
            "immutable_id": None,
            "last_modified": None,
        }

    @property
    def type(self) -> str:
        """Expose the OPTIMADE entry type.

        :return: ``structures``.
        """
        return "structures"


cast(Any, CIFStructureRecord).__httk_stored_properties__ = cif_structure_properties()
