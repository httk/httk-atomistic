"""Build magnetic structures from neutral mCIF mappings."""

from collections.abc import Mapping
from typing import Any

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.modulated import ModulatedStructure
from httk.atomistic.models.structure.symops import SymopsStructure
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.xyz import operation_from_xyz, operation_from_xyzt

from .cif_structures import _cell_from_cif, _exact_positions, _species_name

__all__ = ["symops_structures_from_mcif"]


def symops_structures_from_mcif(payload: Mapping[str, Any]) -> list[SymopsStructure | ModulatedStructure]:
    """Build one magnetic structure per block in a neutral mCIF payload."""
    blocks = payload.get("blocks")
    if blocks is None:
        return [_structure_from_mcif_block(payload, "mcif block")]
    if not blocks:
        unparsed = payload.get("unparsed") or []
        if unparsed:
            detail = "; ".join(f"block {item['block']!r}: {item['reason']}" for item in unparsed)
            raise ValueError(f"this mCIF holds no structure blocks ({detail})")
        raise ValueError("this mCIF holds no structure blocks")
    return [_structure_from_mcif_block(block, f"mcif block {index}") for index, block in enumerate(blocks)]


def _structure_from_mcif_block(data: Mapping[str, Any], block_name: str) -> SymopsStructure | ModulatedStructure:
    if data.get("format") != "mcif":
        raise ValueError(f"{block_name} is not an 'mcif' mapping")
    if data.get("incomm") is not None:
        return ModulatedStructure(data)

    cell = _cell_from_cif(data)
    positions = _exact_positions(data)
    symbols = list(data["symbols"])
    labels = list(data.get("labels") or symbols)
    species, species_at_sites = _species(data, symbols, labels)
    site_moments = _moments(data, cell)
    return SymopsStructure(
        cell,
        Sites(positions, data.get("coordinate_precision")),
        species,
        species_at_sites,
        _symops(data, block_name),
        site_moments=site_moments,
        bns_number=data.get("bns_nbr"),
        bns_label=data.get("bns_name"),
    )


def _species(data: Mapping[str, Any], symbols: list[str], labels: list[str]) -> tuple[list[Species], list[str]]:
    occupancies = data.get("occupancies")
    occupancies_exact = data.get("occupancies_exact")
    occupancy_precisions = data.get("occupancy_precisions")
    by_name: dict[str, Species] = {}
    species_at_sites: list[str] = []
    for index, (symbol, label) in enumerate(zip(symbols, labels)):
        if occupancies_exact is not None and occupancies_exact[index] is not None:
            occupancy = occupancies_exact[index]
        elif occupancies is None:
            occupancy = 1
        elif occupancies[index] is None:
            raise ValueError(f"mCIF occupancy is missing for site {label!r}")
        else:
            occupancy = occupancies[index]
        name = _species_name(symbol, label, occupancy)
        if name not in by_name:
            precision = None if occupancy_precisions is None else occupancy_precisions[index]
            by_name[name] = Species(
                name=name,
                chemical_symbols=(symbol,),
                concentration=(occupancy,),
                original_name=None if label == symbol else label,
                concentration_precision=(precision,) if occupancy_precisions is not None else None,
            )
        species_at_sites.append(name)
    return list(by_name.values()), species_at_sites


def _moments(data: Mapping[str, Any], cell: Cell) -> Any:
    basis = data.get("moment_basis")
    if basis is None:
        return None
    rows = data.get("magmoms_exact")
    if rows is None:
        raise ValueError("mCIF declares a moment basis but has no exact magnetic moments")
    precision = data.get("magmom_precision")
    if basis == "crystalaxis":
        return CrystalAxisSiteMoments(rows, cell, precision=precision)
    if basis == "cartesian":
        return CartesianSiteMoments(rows, precision=precision)
    raise ValueError(f"unsupported mCIF moment basis {basis!r}")


def _parse_operation(value: str) -> tuple[AffineOperation, int]:
    try:
        if len(value.split(",")) == 3:
            return operation_from_xyz(value), 1
        return operation_from_xyzt(value)
    except ValueError as error:
        raise ValueError(f"cannot parse mCIF symmetry operation {value!r}: {error}") from error


def _symops(data: Mapping[str, Any], block_name: str) -> tuple[tuple[AffineOperation, int], ...]:
    raw_base = data.get("symops_xyz")
    if not raw_base:
        raise ValueError(f"{block_name} has no magnetic symmetry operations (symops_xyz is empty)")
    raw_centerings = data.get("centerings_xyz") or ("x,y,z,+1",)
    base = tuple(_parse_operation(value) for value in raw_base)
    centerings = tuple(_parse_operation(value) for value in raw_centerings)
    identity = AffineOperation.identity().matrix
    for raw, (operation, _) in zip(raw_centerings, centerings):
        if operation.matrix != identity:
            raise ValueError(f"mCIF centering must be a pure translation: {raw!r}")
    return tuple(
        (centering_operation * base_operation, base_time * centering_time)
        for base_operation, base_time in base
        for centering_operation, centering_time in centerings
    )
