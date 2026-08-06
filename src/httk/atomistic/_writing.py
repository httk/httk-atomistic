"""Private exact neutral-payload serializers used by :func:`httk.core.save`."""

import fractions
from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from httk.core import FracVector

from httk.atomistic._atomic_projection import require_bare_atomic_projection
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup

_POSCAR_DECIMAL_DIGITS = 16


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


def _poscar_token(value: Any) -> str:
    if isinstance(value, fractions.Fraction):
        rational = value
    elif getattr(value, "is_rational", False):
        rational = value._rational_fraction()
    else:
        with localcontext() as context:
            context.prec = _POSCAR_DECIMAL_DIGITS
            return format(value.to_decimal(digits=_POSCAR_DECIMAL_DIGITS), ".16g")

    denominator = rational.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator == 1:
        return _finite_decimal(rational)
    with localcontext() as context:
        context.prec = _POSCAR_DECIMAL_DIGITS
        return format(Decimal(rational.numerator) / Decimal(rational.denominator), ".16g")


def _pad_poscar_decimal(value: str) -> str:
    if "e" in value.lower():
        return value
    whole, dot, fraction = value.partition(".")
    if not dot:
        return whole + "." + "0" * _POSCAR_DECIMAL_DIGITS
    return whole + "." + fraction.ljust(_POSCAR_DECIMAL_DIGITS, "0")


def _poscar_padded_token(value: Any) -> str:
    return _pad_poscar_decimal(_poscar_token(value))


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
        require_bare_atomic_projection(obj, "CIF")
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
    require_bare_atomic_projection(structure, "CIF")
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


def _poscar_payload_from_structure(obj: Any) -> Mapping[str, object]:
    """Serialize a structure to POSCAR's neutral payload."""
    structure = UnitcellStructureView(obj)
    require_bare_atomic_projection(structure, "POSCAR")
    species_at_sites = tuple(structure.species_at_sites)
    if not species_at_sites:
        raise ValueError("POSCAR cannot represent an empty structure")

    by_name = {species.name: species for species in structure.species}
    names: list[str] = []
    symbol_for_name: dict[str, str] = {}
    for name in species_at_sites:
        if name in symbol_for_name:
            continue
        species = by_name[name]
        if not species.is_single_element:
            raise ValueError(
                f"POSCAR cannot represent disorder/partial occupancy for species {name!r}; "
                "each occupied site must have one fully occupied chemical symbol"
            )
        symbol_for_name[name] = species.chemical_symbols[0]
        names.append(name)

    groups: dict[str, list[int]] = {name: [] for name in names}
    for index, name in enumerate(species_at_sites):
        groups[name].append(index)
    order = [index for name in names for index in groups[name]]
    symbols = [symbol_for_name[name] for name in names]
    rows = list(structure.sites.reduced_coords)
    coords = [[_poscar_padded_token(value) for value in rows[index].to_fractions()] for index in order]
    cell = [
        [_poscar_padded_token(structure.cell.unscaled_basis._element((row, column))) for column in range(3)]
        for row in range(3)
    ]
    formula = structure.chemical_formula_hill or structure.chemical_formula_reduced or "structure"
    return {
        "format": "vasp-poscar",
        "comment": f"{formula} {structure.id}",
        "scale": _poscar_token(structure.cell.scale),
        "volume": None,
        "cell": cell,
        "symbols": symbols,
        "counts": [len(groups[name]) for name in names],
        "cartesian": False,
        "coords": coords,
        "selective_dynamics": None,
    }
