"""Private exact neutral-payload serializers used by :func:`httk.core.save`."""

import fractions
from collections.abc import Mapping
from typing import Any

from httk.core import FracVector

from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup


def _finite_decimal(value: fractions.Fraction) -> str:
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return str(value)
    places = max(twos, fives)
    integer = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    sign = "-" if integer < 0 else ""
    digits = str(abs(integer)).rjust(places + 1, "0")
    if places == 0:
        return sign + digits
    return f"{sign}{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")


def _exact_value(value: Any, *, field: str) -> fractions.Fraction:
    if isinstance(value, fractions.Fraction):
        return value
    if hasattr(value, "is_rational") and value.is_rational:
        return value._rational_fraction()
    raise ValueError(f"CIF serializer cannot represent {field} exactly as a rational CIF number")


def _render(value: Any, *, field: str) -> str:
    return _finite_decimal(_exact_value(value, field=field))


def _cell_parameters(cell: Any) -> tuple[str, ...]:
    params = tuple(_exact_value(value, field="cell parameter") for value in (*cell.lengths, *cell.angles))
    rendered = tuple(_finite_decimal(value) for value in params)
    from httk.atomistic.models.cell.params import CellParams

    if CellParams(params).basis != cell.basis:
        raise ValueError("CIF cell parameters would change the exact cell basis")
    return rendered


def _species_fields(species: Any) -> tuple[str, str]:
    if len(species.chemical_symbols) != 1 or len(species.concentration) != 1:
        raise ValueError(f"CIF serializer cannot represent mixed species {species.name!r} exactly")
    occupancy = _render(species.concentration[0], field="occupancy")
    precision = species.concentration_precision[0]
    if precision is not None and "." not in occupancy and precision.denominator in (10, 100, 1000, 10000):
        occupancy += "." + "0" * (len(str(precision.denominator)) - 1)
    return species.chemical_symbols[0], occupancy


def _block(
    structure: Any,
    positions: list[FracVector],
    symops: tuple[AffineOperation, ...],
    *,
    spacegroup: Spacegroup,
    labels: list[str],
) -> dict[str, object]:
    by_name = {species.name: species for species in structure.species}
    symbols: list[str] = []
    occupancies: list[str] = []
    written_labels = []
    has_nondefault_occupancy = False
    for label in labels:
        species = by_name[label]
        symbol, occupancy = _species_fields(species)
        symbols.append(symbol)
        occupancies.append(occupancy)
        has_nondefault_occupancy |= species.concentration[0] != 1 or species.concentration_precision[0] is not None
        written_labels.append(species.original_name or label)
    return {
        "format": "cif",
        "cell_parameters_exact": _cell_parameters(structure.cell),
        "positions_exact": [
            tuple(_render(value, field="fractional coordinate") for value in row.to_fractions()) for row in positions
        ],
        "symops_xyz": tuple(operation.wrapped().to_xyz() for operation in symops),
        "occupancies_exact": tuple(occupancies) if has_nondefault_occupancy else None,
        "symbols": tuple(symbols),
        "labels": tuple(written_labels),
        "space_group_nbr": str(spacegroup.it_number),
        "space_group_name_hm": spacegroup.hermann_mauguin,
        "space_group_name_hall": spacegroup.hall_symbol,
    }


def _cif_payload_from_structure(obj: Any) -> Mapping[str, object]:
    """Serialize a structure to CIF's exact neutral channels.

    Exact channels retain rational tokens (including ``1/3``); the CIF writer emits
    valid standard decimals and adds ``_httk_*_exact`` companions when those decimals
    are lossy. Lossy standard values use 16 significant digits; finite coordinate
    values are padded to 16 decimal places to keep recognition precision stable.
    Irrational cell parameters are rejected: CIF has no exact syntax for them, so this
    serializer never silently turns an exact structure into a float.
    """
    if isinstance(obj, FundamentalDomainStructure):
        setting = obj.setting()
        if setting is None:
            raise ValueError(
                "CIF cannot preserve an ASUStructure with an unregistered setting transform; "
                "use a tabulated setting or save a unit-cell view"
            )
        operations = tuple(
            obj.transform.symop_to_setting(operation) for operation in obj.spacegroup.symmetry_operations
        )
        positions = [
            site.representative.normalize()
            if site.representative is not None
            else obj._representatives_for_site(site)[0]
            for site in obj.wyckoff_sites
        ]
        return {
            "format": "cif",
            "blocks": [
                _block(
                    obj, positions, operations, spacegroup=setting, labels=[site.species for site in obj.wyckoff_sites]
                )
            ],
        }

    structure = UnitcellStructureView(obj)
    identity = AffineOperation.identity()
    labels = list(structure.species_at_sites)
    return {
        "format": "cif",
        "blocks": [
            _block(
                structure,
                list(structure.sites.reduced_coords),
                (identity,),
                spacegroup=Spacegroup.standard(1),
                labels=labels,
            )
        ],
    }
