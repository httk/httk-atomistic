"""A lossless CIF data block that is also a structure source.

:class:`CIFStructure` keeps everything a CIF data block says (see
:class:`~httk.atomistic.io.cif.CIFBlock`) and converts to the other structure classes through
:meth:`CIFStructure.resolve`, which gives exactly what :func:`httk.core.load` gives for that block.
"""

import os
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePath
from typing import IO, Any, ClassVar, Self, cast

from httk.core.datastream.compression import split_compression_suffix

from httk.atomistic.io.cif.cif_items import (
    CIFBlock,
    CIFDataItem,
    CIFDialect,
    CIFItem,
    CIFLoop,
    CIFSaveFrame,
    CIFTable,
)
from httk.atomistic.io.cif.cif_reader import _is_consumed_tag, _is_retained_tag, _legacy_block, read_cif
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.backend import StructureBackend

__all__ = ["CIFStructure", "read_cif_structures"]


def _data(items: tuple[CIFItem, ...]) -> Iterator[CIFDataItem | CIFLoop]:
    """Yield the data items and loops of items and (nested) save frames."""
    for item in items:
        if isinstance(item, (CIFDataItem, CIFLoop)):
            yield item
        elif isinstance(item, CIFSaveFrame):
            yield from _data(item.items)


@dataclass(frozen=True)
class CIFStructure(CIFBlock, StructureBackend):
    """A lossless, structure-like CIF data block.

    It holds the block exactly as :func:`~httk.atomistic.io.cif.read_cif` reads it with
    ``lossless=True`` and is equal to another block with the same content. Any data block is
    one, also a block without atoms or a block that is not a valid structure: such errors appear
    only when it is converted. Conversion (:meth:`resolve`, or constructing a structure view
    such as ``ASUStructureView(cs)``) gives what :func:`httk.core.load` gives for the block,
    except that a data name the structure reader consumes may not occur twice. A fetched
    :class:`~httk.atomistic.CIFStructureRecord` becomes one with
    ``httk.core.coerce(record, CIFStructure)``.

    The content rule, which decides equality and the content id: the content is the block name,
    the dialect, the file's header, the leading comments, and the items in order (data names,
    values and comments in their places). Whitespace and quoting are not content (whitespace
    and line breaks outside text fields and strings, quote style, line ends, the encoding, a
    byte-order mark, trailing comment whitespace and blank lines), except that these four are
    content:

    1. a comment's line placement, inline after a token or on its own line;
    2. quotes that change a value's kind: ``'1.5'`` is a string and ``1.5`` a number, ``'?'``
       a string and ``?`` a gap;
    3. where the file header ends: a blank line before or among the leading comment lines ends
       the header, and the later comments lead the first block;
    4. runs of spaces inside a pragmatically joined value: ``P  21/c`` differs from ``P 21/c``.

    Inside a CIF2 list or table a comment's placement is not content: it moves to just after
    the value.

    :param name: The block name as written (after ``data_``).
    :param items: The block's items in written order.
    :param header: The file's header comment lines.
    :param leading: The own-line comments before the ``data_`` line.
    :param dialect: ``"cif"`` (read and converted as a ``.cif`` file) or ``"mcif"`` (as a
        ``.mcif`` file).
    :raises ValueError: If a value, name or comment placement cannot be written in the block's
        syntax.
    """

    kind: ClassVar[str] = "cif"
    __httk_storage_record__: ClassVar[type[Any]]

    @classmethod
    def from_cif(
        cls,
        source: str | os.PathLike[str] | IO[bytes] | Iterable[str],
        *,
        dialect: CIFDialect | None = None,
        pragmatic: bool = True,
        repair: bool = False,
    ) -> Self:
        """Read a CIF file that holds exactly one data block.

        :param source: A filename, open text or binary stream, or iterable of CIF lines.
        :param dialect: ``"cif"`` or ``"mcif"``; by default ``"mcif"`` for a ``.mcif`` path or
            stream name (also compressed) and ``"cif"`` otherwise.
        :param pragmatic: Accept the documented common deviations from strict CIF syntax.
        :param repair: Decode a file that is not UTF-8 as Latin-1, with a warning.
        :return: The block.
        :raises ValueError: If the CIF syntax is broken or the file does not hold exactly one
            data block.
        """
        blocks = read_cif_structures(source, dialect=dialect, pragmatic=pragmatic, repair=repair)
        if len(blocks) != 1:
            raise ValueError(f"this CIF holds {len(blocks)} data blocks; use read_cif_structures(...)")
        return cls(blocks[0].name, blocks[0].items, blocks[0].header, blocks[0].leading, blocks[0].dialect)

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a CIF structure, its storage record, or a lossless CIF block.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: The CIF structure, or ``None`` when this backend declines ``obj``.
        :raises ValueError: If ``obj`` is a record of another canonical form or with malformed
            content.
        """
        from httk.atomistic.storage.cif_record import CIFStructureRecord, _cif_structure_from_record

        if hints.get("kind", cls.kind) != cls.kind:
            return None
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, CIFStructureRecord):
            return cast(Self, _cif_structure_from_record(obj))
        if type(obj) is CIFBlock:
            try:
                return cls(obj.name, obj.items, obj.header, obj.leading, obj.dialect)
            except (TypeError, ValueError):
                return None
        return None

    def resolve(self, *, repair: bool = False, allow_large_cif_uncertainty: bool = False) -> StructureBackend:
        """Convert the block to the structure :func:`httk.core.load` gives for it.

        A ``"cif"`` block gives an :class:`~httk.atomistic.ASUStructure` (bonds included), an
        ``"mcif"`` block a :class:`~httk.atomistic.SymopsStructure` or
        :class:`~httk.atomistic.ModulatedStructure`. The default result is memoised.

        :param repair: Apply the structure reader's documented repairs, as ``load(..., repair=True)``.
        :param allow_large_cif_uncertainty: Accept large stated uncertainties, as
            ``load(..., allow_large_cif_uncertainty=True)`` (``"cif"`` blocks only).
        :return: The structure.
        :raises ValueError: If a data name the structure reader uses occurs more than once, or the
            block is not a valid structure (with the messages ``load`` gives).
        :raises TypeError: If ``allow_large_cif_uncertainty`` is given for an ``"mcif"`` block.
        """
        memoised = not repair and not allow_large_cif_uncertainty
        cached = vars(self).get("_resolved_structure")
        if memoised and cached is not None:
            return cached
        data = list(_data(self.items))
        names = [name for item in data for name in (item.names if isinstance(item, CIFLoop) else (item.name,))]
        repeated = Counter(key for name in names if _is_consumed_tag(key := name[1:].lower()))
        for key, count in repeated.items():
            if count > 1:
                raise ValueError(
                    f"CIF block {self.name!r} repeats data name {'_' + key!r}; a structure needs each data name once"
                )
        if self.dialect == "cif":  # the structure reader takes single values (mCIF reads its lists)
            values = (
                pair
                for item in data
                for pair in (
                    ((item.name, item.value),)
                    if isinstance(item, CIFDataItem)
                    else (pair for row in item.rows for pair in zip(item.names, row))
                )
            )
            for name, value in values:
                if isinstance(value, (tuple, CIFTable)) and _is_retained_tag(name[1:].lower()):
                    raise ValueError(
                        f"CIF block {self.name!r} data name {name!r} holds a CIF2 list or table; "
                        "a structure needs a single value"
                    )
        header = "".join(f"#{comment.text}\n" for comment in self.header)
        blocks = [(self.name.lower(), _legacy_block(self, structural_only=self.dialect == "cif"))]
        from httk.atomistic._loading import _structure_from_cif, _structure_from_mcif

        if self.dialect == "cif":
            from httk.atomistic.cif_structures import _cif_payload

            payload = _cif_payload(
                blocks, header, allow_large_cif_uncertainty=allow_large_cif_uncertainty, repair=repair
            )
            structure = _structure_from_cif(payload)
        else:
            if allow_large_cif_uncertainty:
                raise TypeError("allow_large_cif_uncertainty applies to 'cif' blocks, not 'mcif' ones")
            from httk.atomistic.io.cif.mcif_parser import _mcif_payload

            structure = _structure_from_mcif(_mcif_payload(blocks, header, repair=repair))
        if memoised:
            object.__setattr__(self, "_resolved_structure", structure)
        return structure

    def __getstate__(self) -> dict[str, Any]:
        # The memoised conversion is a cache, not state.
        return {key: value for key, value in vars(self).items() if key != "_resolved_structure"}

    @property
    def type(self) -> str:
        """Expose the logical OPTIMADE entry family.

        :return: ``"structures"``.
        """
        return "structures"

    @property
    def cell(self) -> Cell:
        """Expose the converted structure's cell."""
        return self.resolve().cell

    @property
    def sites(self) -> Sites:
        """Expose the converted structure's sites."""
        return self.resolve().sites

    @property
    def species(self) -> tuple[Species, ...]:
        """Expose the converted structure's distinct species."""
        return self.resolve().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Expose the species occupying each site of the converted structure."""
        return self.resolve().species_at_sites

    @property
    def site_moments(self) -> SiteMomentsBackend | None:
        """Expose the converted structure's site moments, or ``None`` when they are unstated."""
        return self.resolve().site_moments

    @property
    def charge(self) -> Fraction | None:
        """Expose the converted structure's assigned charge, or ``None`` when it is unstated."""
        return self.resolve().charge


def read_cif_structures(
    source: str | os.PathLike[str] | IO[bytes] | Iterable[str],
    *,
    dialect: CIFDialect | None = None,
    pragmatic: bool = True,
    repair: bool = False,
) -> list[CIFStructure]:
    """Read every data block of a CIF file as a :class:`CIFStructure`, in written order.

    :param source: A filename, open text or binary stream, or iterable of CIF lines.
    :param dialect: ``"cif"`` or ``"mcif"``; by default ``"mcif"`` for a ``.mcif`` path or
        stream name (also compressed) and ``"cif"`` otherwise.
    :param pragmatic: Accept the documented common deviations from strict CIF syntax.
    :param repair: Decode a file that is not UTF-8 as Latin-1, with a warning.
    :return: One block per ``data_`` block.
    :raises ValueError: If the CIF syntax is broken, or ``dialect`` is not ``"cif"`` or ``"mcif"``.
    """
    if dialect is None:
        name = os.fspath(source) if isinstance(source, (str, os.PathLike)) else getattr(source, "name", "")
        name = name if isinstance(name, str) else ""
        inner = split_compression_suffix(PurePath(name).name)[0]
        dialect = "mcif" if PurePath(inner).suffix.lower() == ".mcif" else "cif"
    elif dialect not in ("cif", "mcif"):
        raise ValueError(f"CIF dialect must be 'cif' or 'mcif', not {dialect!r}")
    blocks = read_cif(source, pragmatic, dialect == "mcif", repair=repair, lossless=True)
    return [CIFStructure(block.name, block.items, block.header, block.leading, block.dialect) for block in blocks]


def _coerce_cif_structure(value: Any, target: type) -> CIFStructure | None:
    """Coerce a CIF structure record or lossless block to a :class:`CIFStructure` (``httk.core.coerce``)."""
    return CIFStructure._backend_adopt(value)
