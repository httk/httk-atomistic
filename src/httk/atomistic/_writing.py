"""Private exact neutral-payload serializers used by :func:`httk.core.save`."""

import fractions
from collections.abc import Mapping
from decimal import Decimal, localcontext
from typing import Any

from httk.core import FracVector, unwrap

from httk.atomistic._atomic_projection import require_bare_atomic_projection
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.models.trajectory.backend import TrajectoryBackend
from httk.atomistic.models.trajectory.view import TrajectoryView
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup

_POSCAR_DECIMAL_DIGITS = 16

#: Significant digits used when the approximate (lossy) CIF mode renders a cell parameter that has
#: no exact CIF syntax. Twelve is the CIF-community-typical width and well beyond crystallographic
#: relevance; it applies only to the opt-in ``approximate=True`` path, never to the exact default.
_APPROX_CIF_SIG_DIGITS = 12


def _approx_decimal(value: Any) -> str:
    """Render a cell parameter as a rounded decimal for the lossy approximate CIF mode.

    :param value: An exact rational or surd cell parameter.
    :return: A standard decimal string with :data:`_APPROX_CIF_SIG_DIGITS` significant digits.
    """
    if isinstance(value, fractions.Fraction):
        with localcontext() as context:
            context.prec = _APPROX_CIF_SIG_DIGITS
            return format(Decimal(value.numerator) / Decimal(value.denominator), f".{_APPROX_CIF_SIG_DIGITS}g")
    return format(value.to_decimal(digits=_APPROX_CIF_SIG_DIGITS), f".{_APPROX_CIF_SIG_DIGITS}g")


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


def _cell_parameters(cell: Any) -> tuple[tuple[str, ...], bool]:
    """Render the six CIF cell parameters, reporting whether the rendering is lossy.

    The exact default keeps rational parameters as standard decimals; a parameter with no exact
    CIF syntax (an irrational length or angle) or a rational set whose six values would rebuild a
    different oriented basis cannot round-trip, so it is rounded and flagged. The writer refuses a
    flagged block unless the caller opted in to the approximate mode.

    :param cell: The cell whose lengths and angles are serialized.
    :return: The rendered parameters and whether the rendering is lossy (approximate).
    """
    raw = (*cell.lengths, *cell.angles)
    try:
        params = tuple(_exact_value(value, field="cell parameter") for value in raw)
    except ValueError:
        # An irrational length or angle has no exact CIF number; only the approximate mode can write it.
        return tuple(_approx_decimal(value) for value in raw), True
    from httk.atomistic.models.cell.params import CellParams

    rendered = tuple(_finite_decimal(value) for value in params)
    # Exact numbers can still reconstruct a different oriented basis, which is lossy in its own right.
    return rendered, CellParams(params).basis != cell.basis


def _occupancy(species: Any, index: int) -> str:
    occupancy = _render(species.concentration[index], field="occupancy")
    precision = species.concentration_precision[index]
    if precision is not None and "." not in occupancy and precision.denominator in (10, 100, 1000, 10000):
        occupancy += "." + "0" * (len(str(precision.denominator)) - 1)
    return occupancy


def _type_symbol(species: Any, index: int) -> str:
    """Return the CIF atom-type spelling for one OPTIMADE species constituent."""
    symbol = species.chemical_symbols[index]
    label = None if species.labels is None else species.labels[index]
    if symbol == "H" and label in {"D", "T"}:
        base = label
    elif symbol == "X":
        base = label or "X"
    elif symbol == "vacancy":
        base = "vacancy"
    else:
        base = symbol
    charge = None if species.charges is None else species.charges[index]
    if charge is None:
        return base

    from httk.atomistic.cif_structures import _decode_type_symbol

    # A read-derived, single-constituent species commonly retains the source atom-type
    # spelling as its name. Reuse it when it decodes to this exact constituent so forms
    # such as ``P+5`` survive a read/write/read cycle instead of becoming ``P5+``.
    source_candidate = species.name
    source_decoded = _decode_type_symbol(source_candidate, None)
    if (source_decoded.chemical_symbol, source_decoded.charge, source_decoded.species_label) == (
        symbol,
        charge,
        label,
    ):
        return source_candidate

    if charge.denominator != 1:
        raise ValueError(f"CIF serializer cannot represent species charges: fractional {charge} on atom type {base!r}")
    magnitude = abs(charge.numerator)
    if charge == 0:
        candidate = f"{base}0"
    elif symbol == "X":
        candidate = f"{base}{'+' if charge > 0 else '-'}{magnitude}"
    else:
        candidate = f"{base}{magnitude}{'+' if charge > 0 else '-'}"

    decoded = _decode_type_symbol(candidate, None)
    if (decoded.chemical_symbol, decoded.charge, decoded.species_label) != (symbol, charge, label):
        raise ValueError(f"CIF serializer cannot represent species charges: state {charge} on constituent {base!r}")
    return candidate


def _source_labels(species: Any, indices: list[int], fallback: str) -> list[str]:
    """Recover source row labels for constituents of a read-derived mixed species."""
    if len(indices) == 1:
        return [species.original_name or fallback]
    name_parts = species.name.split("/")
    if len(name_parts) == len(indices):
        return name_parts
    labels = []
    for offset, index in enumerate(indices, start=1):
        label = None if species.labels is None else species.labels[index]
        labels.append(label or f"{fallback}_{offset}")
    return labels


def _attached_hydrogen_count(species: Any) -> int:
    """Return the exactly CIF-representable attached-hydrogen count."""
    if species.attached is None:
        return 0
    if species.attached != ("H",) or species.nattached is None or len(species.nattached) != 1:
        raise ValueError("CIF serializer can represent only one attached hydrogen kind per species")
    count = species.nattached[0]
    if count == 0:
        raise ValueError(
            "CIF serializer cannot distinguish an explicit zero attached-hydrogen count from no attachment"
        )
    return count


def _require_cif_projection(backend: Any) -> None:
    """Reject structure state for which this CIF serializer has no exact channel."""
    source = unwrap(backend)
    assemblies = getattr(source, "assemblies", getattr(backend, "assemblies", None))
    if assemblies is not None:
        raise TypeError("This structure cannot be represented as CIF because it has assemblies")
    composition = getattr(source, "chemical_composition", getattr(backend, "chemical_composition", None))
    if composition is not None:
        raise TypeError("This structure cannot be represented as CIF because it has a declared chemical composition")
    charge = getattr(source, "charge", getattr(backend, "charge", None))
    if charge is not None:
        raise ValueError("This structure cannot be represented as CIF because it has a charge")
    for species in getattr(source, "species", getattr(backend, "species", ())):
        if getattr(species, "spins", None) is not None:
            raise ValueError("This structure cannot be represented as CIF because species has spins")
        _attached_hydrogen_count(species)


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
    written_labels: list[str] = []
    written_positions: list[FracVector] = []
    attached_hydrogens: list[int] = []
    calc_flags: list[str] = []
    atom_type_masses: dict[str, float] = {}
    has_nondefault_occupancy = False

    def append_row(species: Any, index: int, source_label: str, position: FracVector, calc_flag: str) -> None:
        nonlocal has_nondefault_occupancy
        symbol = _type_symbol(species, index)
        occupancy = _occupancy(species, index)
        symbols.append(symbol)
        occupancies.append(occupancy)
        written_labels.append(source_label)
        written_positions.append(position)
        attached_hydrogens.append(_attached_hydrogen_count(species))
        calc_flags.append(calc_flag)
        has_nondefault_occupancy |= (
            species.concentration[index] != 1 or species.concentration_precision[index] is not None
        )
        if species.mass is not None:
            mass = species.mass[index]
            previous = atom_type_masses.get(symbol)
            if previous is not None and previous != mass:
                raise ValueError(f"CIF serializer cannot represent two masses for atom type {symbol!r}")
            atom_type_masses[symbol] = mass

    for position, label in zip(positions, labels):
        species = by_name[label]
        nonvacancy = [index for index, symbol in enumerate(species.chemical_symbols) if symbol != "vacancy"]
        indices = nonvacancy or [0]
        if len(indices) == 1 and species.labels is not None:
            index = indices[0]
            constituent_label = species.labels[index]
            symbol = species.chemical_symbols[index]
            if constituent_label is not None and not (
                symbol == "X" or (symbol == "H" and constituent_label in {"D", "T"})
            ):
                raise ValueError("CIF serializer cannot represent this single-constituent species label exactly")
        source_labels = _source_labels(species, indices, label)
        for index, source_label in zip(indices, source_labels):
            append_row(species, index, source_label, position, "d")

    for label in structure.implicit_atoms:
        species = by_name[label]
        if len(species.chemical_symbols) != 1 or species.chemical_symbols[0] == "vacancy":
            raise ValueError("CIF serializer requires implicit-atom species to have one non-vacancy constituent")
        append_row(species, 0, species.original_name or label, FracVector((-1, -1, -1)), "dum")

    cell_parameters, cell_is_approximate = _cell_parameters(structure.cell)
    return {
        "format": "cif",
        "approximate": cell_is_approximate,
        "cell_parameters_exact": cell_parameters,
        "positions_exact": [
            tuple(_render(value, field="fractional coordinate") for value in row.to_fractions())
            for row in written_positions
        ],
        "symops_xyz": tuple(operation.wrapped().to_xyz() for operation in symops),
        "occupancies_exact": tuple(occupancies) if has_nondefault_occupancy else None,
        "symbols": tuple(symbols),
        "labels": tuple(written_labels),
        "attached_hydrogens": tuple(attached_hydrogens) if any(attached_hydrogens) else None,
        "calc_flags": tuple(calc_flags) if structure.implicit_atoms else None,
        "atom_type_symbols": tuple(atom_type_masses),
        "atom_type_masses": tuple(atom_type_masses.values()),
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
    A cell with no exact CIF form (an irrational length or angle, or a rational set whose six
    values would rebuild a different oriented basis) is rounded and the block is flagged
    ``approximate``; the CIF writer refuses a flagged block unless the caller opts in with
    ``approximate=True``, so this serializer never silently turns an exact structure into a float.
    """
    if isinstance(obj, FundamentalDomainStructure):
        _require_cif_projection(obj)
        setting = obj.setting()
        if setting is None:
            raise ValueError(
                "CIF cannot preserve an ASUStructure with an unregistered setting transform; "
                "use a tabulated setting or save a unit-cell view"
            )
        operations = tuple(
            obj.transform.symop_to_setting(operation) for operation in obj.spacegroup.symmetry_operations
        )
        # Serialize the exact Wyckoff representative reconstructed from the ASU state.
        # A retained source coordinate may only have been close enough to snap onto that
        # position; writing the rounded source coordinate with fresh high precision can make
        # the next reader miss the special position and change its Wyckoff orbit.
        positions = [obj._representatives_for_site(site)[0] for site in obj.wyckoff_sites]
        return {
            "format": "cif",
            "blocks": [
                _block(
                    obj, positions, operations, spacegroup=setting, labels=[site.species for site in obj.wyckoff_sites]
                )
            ],
        }

    structure = UnitcellStructureView(obj)
    _require_cif_projection(structure)
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
    from httk.atomistic.integrations.vasp import VASPStructure

    backend = obj._backend if isinstance(obj, StructureView) else unwrap(obj)
    if isinstance(obj, VASPStructure):
        return obj.payload
    if isinstance(backend, VASPStructure):
        return backend.payload

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


def _trajectory_jsonl_payload(obj: Any) -> Mapping[str, object]:
    """Adapt a trajectory to a streaming neutral JSONL writer payload."""
    trajectory = TrajectoryBackend._select_backend(obj) if isinstance(obj, Mapping) else obj
    if not isinstance(trajectory, (TrajectoryBackend, TrajectoryView)):
        raise TypeError(f"cannot save {type(obj).__name__} as a trajectory JSONL file")

    from httk.atomistic.models.species.plain_view import PlainSpeciesView

    first_cell: list[list[float]] | None = None
    constant = True
    for frame in trajectory.frames():
        cell = frame.cell.basis.to_floats()
        if first_cell is None:
            first_cell = cell
        elif cell != first_cell:
            constant = False
    if first_cell is None:
        raise ValueError("cannot save an empty trajectory")

    observable_values = {name: iter(trajectory.observable(name)) for name in trajectory.observable_names}

    def frame_lines() -> Any:
        for index, frame in enumerate(trajectory.frames()):
            values = {name: next(observable_values[name]) for name in observable_values}
            line: dict[str, Any] = {
                "index": index,
                "fractional_site_positions": frame.sites.reduced_coords.to_floats(),
                "observables": values,
            }
            if not constant:
                line["lattice_vectors"] = frame.cell.basis.to_floats()
            yield line

    return {
        "format": "httk-trajectory-jsonl",
        "header": {
            "species": [dict(PlainSpeciesView(species)) for species in trajectory.species],
            "species_at_sites": list(trajectory.species_at_sites),
            "constant_cell": first_cell if constant else None,
            "nframes": trajectory.nframes,
            "observable_names": list(trajectory.observable_names),
            "reference_frames": (None if trajectory.reference_frames is None else list(trajectory.reference_frames)),
        },
        "frames": frame_lines(),
    }
