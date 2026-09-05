#
#    The high-throughput toolkit (httk)
#    Copyright (C) 2012-2025 The httk AUTHORS
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Write the neutral mCIF payload produced by the magnetic-structure serializer."""

import io
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from .cif_writer import _neutral_cif_block, write_cif


def _mcif_block(block: Mapping[str, object], *, exact_companions: bool) -> dict[str, object]:
    """Turn one neutral mCIF block into the low-level writer's block shape.

    The cell, ``atom_site_*`` loop, and occupancy channels reuse the nuclear-CIF conversion; the
    nuclear symmetry loop it emits is dropped because an mCIF states its symmetry in the magnetic
    operation loop. The magnetic channels (the crystal-axis moment loop, the magn-operation loop,
    and the BNS scalars) are then appended. The moment and operation tag names contain literal
    dots, which the generic writer emits verbatim.

    :param block: One neutral mCIF block from the magnetic-structure serializer.
    :param exact_companions: Whether to emit the non-standard ``_httk_*_exact`` companion columns
        for the reused structural channels.
    :return: The raw writer block mapping data names to scalars and ``loop_`` column lists.
    """
    raw = _neutral_cif_block(block, exact_companions=exact_companions)
    for key in ("loop_symops", "space_group_symop_operation_xyz"):
        raw.pop(key, None)

    magn_symops = list(cast(Iterable[str], block.get("magn_symops_xyz") or ("x,y,z,+1",)))
    raw["loop_magn_symops"] = ["space_group_symop_magn_operation.xyz"]
    raw["space_group_symop_magn_operation.xyz"] = magn_symops

    labels = block.get("moment_labels")
    if labels is not None:
        rows = list(cast(Iterable[Sequence[object]], block["moment_crystalaxis"]))
        raw["loop_moments"] = [
            "atom_site_moment.label",
            "atom_site_moment.crystalaxis_x",
            "atom_site_moment.crystalaxis_y",
            "atom_site_moment.crystalaxis_z",
        ]
        raw["atom_site_moment.label"] = list(cast(Iterable[str], labels))
        for index, tag in enumerate(
            ("atom_site_moment.crystalaxis_x", "atom_site_moment.crystalaxis_y", "atom_site_moment.crystalaxis_z")
        ):
            raw[tag] = [row[index] for row in rows]

    for source, target in (("bns_number", "space_group_magn.number_bns"), ("bns_label", "space_group_magn.name_bns")):
        value = block.get(source)
        if value is not None:
            raw[target] = value
    return raw


def _write_mcif_payload(
    destination: str | os.PathLike[str] | io.TextIOBase,
    data: Mapping[str, object],
    *,
    approximate: bool = True,
    exact_companions: bool = False,
    **kwargs: object,
) -> None:
    r"""Write the neutral mCIF payload returned by the magnetic-structure serializer.

    :param destination: Filename or open text stream receiving the mCIF text.
    :param data: The neutral mCIF payload, either a single block or a ``blocks`` sequence.
    :param approximate: Whether lossy rounding of a cell with no exact CIF form is allowed
        (default) rather than refused, mirroring :func:`_write_cif_payload`.
    :param exact_companions: Whether to emit the non-standard ``_httk_*_exact`` companion columns
        (exact rational tokens such as ``1/3``) for the reused structural channels. Off by default
        so the file carries only standard CIF columns.
    :param \**kwargs: Remaining low-level :func:`write_cif` options (``header``, ``max_line_length``).
    :raises ValueError: If a block requires approximation and ``approximate`` is ``False``.
    """
    blocks = data.get("blocks")
    if blocks is None:
        blocks = [data]
    block_list = list(cast(Iterable[Mapping[str, object]], blocks))
    if not approximate and any(block.get("approximate") for block in block_list):
        raise ValueError(
            "CIF cannot exactly represent this structure's cell parameters (an irrational or "
            "orientation-losing basis); drop approximate=False to write rounded decimals, or keep "
            "the original structure to preserve the exact basis"
        )
    options: dict[str, Any] = {"header": cast("str | None", data.get("header"))}
    options.update(kwargs)
    write_cif(
        destination,
        [("structure", _mcif_block(block, exact_companions=exact_companions)) for block in block_list],
        **options,
    )
