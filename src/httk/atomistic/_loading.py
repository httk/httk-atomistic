"""Private neutral-payload adapters used by the core loading registry."""

import fractions
import math
from collections.abc import Callable, Mapping
from typing import Any

from httk.core import SurdScalar, SurdVector, exactmath
from httk.core.optimade import OptimadeResource

from httk.atomistic.models._vector_guards import to_surdscalar
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.optimade import OptimadeStructure
from httk.atomistic.models.structure.unitcell import UnitcellStructure

# Cube roots leave the exact squarefree-radical field, so the volume-scaled cell's
# overall scale factor is a *deterministic rational approximation* at this precision
# (exactmath's default accuracy, 1e-10). Everything else — the basis rows and the
# reduced coordinates — remains exact.
_FALLBACK_PREC = exactmath.default_accuracy


def _to_fraction(scalar: SurdScalar) -> fractions.Fraction:
    """An exact :class:`~fractions.Fraction` for a rational scalar, else a deterministic approximation."""
    return fractions.Fraction(scalar.to_fractions_approx(_FALLBACK_PREC))


def _cube_root(value: fractions.Fraction) -> fractions.Fraction:
    """A deterministic rational approximation of the cube root of a positive rational."""
    ln = fractions.Fraction(exactmath.log(value, prec=_FALLBACK_PREC, limit=True))
    return fractions.Fraction(exactmath.exp(ln / 3, prec=_FALLBACK_PREC, limit=True))


def _structure_from_poscar(data: Mapping[str, Any]) -> UnitcellStructure:
    """Build an exact :class:`~httk.atomistic.UnitcellStructure` from a neutral POSCAR mapping."""
    fmt = data.get("format")
    if fmt != "vasp-poscar":
        raise ValueError(f"_structure_from_poscar expected a 'vasp-poscar' mapping, got format={fmt!r}.")

    cell_rows = data["cell"]
    raw_basis = SurdVector.create(cell_rows)

    scale_str = data.get("scale")
    volume_str = data.get("volume")
    if scale_str is not None:
        scale: Any = scale_str
    elif volume_str is not None:
        abs_det = abs(_to_fraction(raw_basis.det()))
        if abs_det == 0:
            raise ValueError("Cannot volume-scale a degenerate cell (zero determinant).")
        target_volume = _to_fraction(SurdVector.create(volume_str)._as_scalar())
        scale = _cube_root(target_volume / abs_det)
    else:
        scale = 1

    cell = Cell(cell_rows, scale, _basis_precision(data, raw_basis, scale))

    if data["cartesian"]:
        # reduced = cart * basis^-1 (row-vector convention); the universal scale cancels.
        reduced: Any = SurdVector.create(data["coords"]) * raw_basis.inv()
    else:
        reduced = data["coords"]
    sites = Sites(reduced, _coordinate_precision(data, raw_basis))

    symbols = data.get("symbols")
    if symbols is None:
        raise ValueError(
            "_structure_from_poscar cannot build species for a VASP-4 POSCAR (no species symbols); "
            "provide a VASP-5 file with an explicit species line."
        )
    counts = data["counts"]

    species: list[Species] = []
    seen: set[str] = set()
    species_at_sites: list[str] = []
    for symbol, count in zip(symbols, counts):
        species_at_sites.extend([symbol] * count)
        if symbol not in seen:
            seen.add(symbol)
            species.append(Species(name=symbol, chemical_symbols=(symbol,), concentration=(1.0,)))

    return UnitcellStructure(cell, sites, species, species_at_sites)


def _structure_from_optimade_payload(data: Mapping[str, Any]) -> OptimadeStructure:
    """Build an :class:`~httk.atomistic.OptimadeStructure` from a tagged payload."""
    if not isinstance(data, Mapping):
        raise ValueError(f"_structure_from_optimade_payload expected a mapping, got {type(data).__name__}.")
    fmt = data.get("format")
    if fmt != "optimade-entry":
        raise ValueError(f"_structure_from_optimade_payload expected an 'optimade-entry' mapping, got format={fmt!r}.")
    resource = data.get("resource")
    if not isinstance(resource, OptimadeResource):
        raise ValueError(
            "_structure_from_optimade_payload expected an OptimadeResource in 'resource', "
            f"got {type(resource).__name__}."
        )
    return OptimadeStructure(resource)


def _structure_from_cif(data: Mapping[str, Any]) -> Any:
    """Build the native ASUStructure of one loaded CIF."""
    from .cif_structures import asu_structures_from_cif

    structures = asu_structures_from_cif(data)
    if len(structures) != 1:
        raise ValueError(
            f"this CIF holds {len(structures)} structures; load() builds one, so use "
            f"httk.atomistic.asu_structures_from_cif(httk.core.load(path, raw=True)) to get them all"
        )
    return structures[0]


def _structure_from_mcif(data: Mapping[str, Any]) -> Any:
    """Build the native SymopsStructure or ModulatedStructure of one loaded mCIF."""
    from .mcif_structures import symops_structures_from_mcif

    structures = symops_structures_from_mcif(data)
    if len(structures) != 1:
        raise ValueError(
            f"this mCIF holds {len(structures)} structures; load() builds one, so use "
            f"symops_structures_from_mcif(httk.core.load(path, raw=True)) to get them all"
        )
    return structures[0]


def _basis_precision(data: Mapping[str, Any], raw_basis: SurdVector, scale: Any) -> fractions.Fraction | None:
    """How precisely the POSCAR states its cell, as an absolute length."""
    cell_precision = data.get("cell_precision")
    if cell_precision is None:
        return None
    scale_value = abs(_to_fraction(to_surdscalar(scale)))
    return scale_value * fractions.Fraction(cell_precision)


def _coordinate_precision(data: Mapping[str, Any], raw_basis: SurdVector) -> fractions.Fraction | None:
    """How precisely the POSCAR states its coordinates, in fractional units."""
    precision = data.get("coordinate_precision")
    if precision is None:
        return None
    if not data["cartesian"]:
        return fractions.Fraction(precision)

    shortest = min(math.dist((0.0, 0.0, 0.0), row) for row in raw_basis.to_floats())
    if shortest <= 0:
        return None
    return fractions.Fraction(precision) / fractions.Fraction(str(shortest)).limit_denominator(10**12)


_STRUCTURE_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "vasp-poscar": _structure_from_poscar,
    "cif": _structure_from_cif,
    "mcif": _structure_from_mcif,
}


def _structure_from_payload(data: Mapping[str, Any]) -> Any:
    """Build a structure from a tagged neutral payload."""
    if not isinstance(data, Mapping):
        raise ValueError(f"_structure_from_payload expected a mapping, got {type(data).__name__}.")
    fmt = data.get("format")
    adapter = _STRUCTURE_ADAPTERS.get(fmt) if isinstance(fmt, str) else None
    if adapter is None:
        raise ValueError(
            f"unrecognized payload format tag {fmt!r}. "
            f"Known structure formats: {', '.join(sorted(_STRUCTURE_ADAPTERS))}."
        )
    return adapter(data)
