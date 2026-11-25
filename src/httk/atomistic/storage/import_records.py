"""Import records of structures: :class:`~httk.core.ImportRecord` subclasses embedding the imported structure.

Each class embeds the imported structure as a record-valued ``structure`` field. The structure is part
of the import's content identity through its own content id. Saving an import record stores the
structure once, deduplicated across imports, with its own entry id: fetchable, searchable and served
in OPTIMADE ``structures`` listings whether or not it is promoted. ``bulk.save(record,
promote=ASUStructureRecord)`` (or saving the structure itself) only marks it a main entry rather than a
dependency of the import record, which in a SQL store makes it a garbage-collection root of
``store.fsck`` and gives it a row in a multi-record family's entry dispatch; searching, fetching and serving are
otherwise unchanged.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

from httk.core import ImportRecord
from httk.core.storage import StorageInfo, stored_property_projections

from httk.atomistic.storage.cif_record import CIFStructureRecord
from httk.atomistic.storage.records import ASUStructureRecord, UnitcellStructureRecord

__all__ = ["ASUStructureImportRecord", "CIFStructureImportRecord", "UnitcellStructureImportRecord"]


def _check_structure(record: ImportRecord, structure: object, record_type: type) -> None:
    if structure is not None and not isinstance(structure, record_type):
        raise TypeError(f"Field 'structure' must be a {record_type.__name__} or None, got {type(structure).__name__}.")
    if (structure is None) == (record.status == "imported"):
        raise ValueError("Field 'structure' must be set exactly when status is 'imported'.")


@dataclass(frozen=True)
class ASUStructureImportRecord(ImportRecord):
    """The record of importing one asymmetric-unit structure.

    The other parameters are those of :class:`~httk.core.ImportRecord`.

    :param structure: The imported structure as its record, e.g. from
        :meth:`ASUStructureRecord.from_structure <httk.atomistic.FundamentalDomainStructureRecord.from_structure>`, not the
        :class:`~httk.atomistic.ASUStructure` itself (record-valued fields take records, as
        everywhere in :mod:`httk.atomistic.storage`); set exactly when ``status`` is ``"imported"``.
    :raises TypeError: If a field has the wrong type.
    :raises ValueError: If ``structure`` disagrees with ``status``, or a base field is invalid.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_asu_structure_import",
        identity_name="atomistic_asu_structure_import",
        indexes=ImportRecord.__httk_storage__.indexes,
    )

    structure: ASUStructureRecord | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_structure(self, self.structure, ASUStructureRecord)


@dataclass(frozen=True)
class UnitcellStructureImportRecord(ImportRecord):
    """The record of importing one unit-cell structure.

    The other parameters are those of :class:`~httk.core.ImportRecord`.

    :param structure: The imported structure as its record, e.g. from
        :meth:`~httk.atomistic.UnitcellStructureRecord.from_structure`; set exactly when ``status``
        is ``"imported"``.
    :raises TypeError: If a field has the wrong type.
    :raises ValueError: If ``structure`` disagrees with ``status``, or a base field is invalid.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_unitcell_structure_import",
        identity_name="atomistic_unitcell_structure_import",
        indexes=ImportRecord.__httk_storage__.indexes,
    )

    structure: UnitcellStructureRecord | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_structure(self, self.structure, UnitcellStructureRecord)


@dataclass(frozen=True)
class CIFStructureImportRecord(ImportRecord):
    """The record of importing one lossless CIF data block.

    The other parameters are those of :class:`~httk.core.ImportRecord`.

    :param structure: The imported block as its record, e.g.
        ``CIFStructureRecord(**project_storage_record(CIFStructureRecord, block))``; set exactly when
        ``status`` is ``"imported"``.
    :raises TypeError: If a field has the wrong type.
    :raises ValueError: If ``structure`` disagrees with ``status``, or a base field is invalid.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="atomistic_cif_structure_import",
        identity_name="atomistic_cif_structure_import",
        indexes=ImportRecord.__httk_storage__.indexes,
    )

    structure: CIFStructureRecord | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_structure(self, self.structure, CIFStructureRecord)


# Projections are read off the exact class, never inherited: serve the base's.
for _cls in (ASUStructureImportRecord, UnitcellStructureImportRecord, CIFStructureImportRecord):
    cast(Any, _cls).__httk_stored_properties__ = stored_property_projections(ImportRecord)
del _cls
