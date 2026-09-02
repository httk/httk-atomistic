"""Private neutral-payload adapters used by the core loading registry."""

import fractions
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
    raw_basis = SurdVector(cell_rows)

    scale_str = data.get("scale")
    volume_str = data.get("volume")
    if scale_str is not None:
        scale: Any = scale_str
    elif volume_str is not None:
        abs_det = abs(_to_fraction(raw_basis.det()))
        if abs_det == 0:
            raise ValueError("Cannot volume-scale a degenerate cell (zero determinant).")
        target_volume = _to_fraction(SurdVector(volume_str)._as_scalar())
        scale = _cube_root(target_volume / abs_det)
    else:
        scale = 1

    override = data.get("precision_override")
    if override is not None:
        # A realistic Cartesian precision (Å) supplied by the caller replaces the
        # digit-derived one; relaxed CONTCAR digits otherwise imply ~machine-epsilon.
        # Cell precision is an absolute length; Sites precision is fractional, so scale
        # the length down by the longest cell edge exactly as cartesian_precision() folds
        # it back up, making structure.cartesian_precision() return prec.
        prec = fractions.Fraction(str(override))
        cell = Cell(cell_rows, scale, prec)
        longest = max(length.to_float() for length in cell.lengths)
        coordinate_precision: fractions.Fraction | None = prec / fractions.Fraction(str(longest))
    else:
        cell = Cell(cell_rows, scale, _basis_precision(data, raw_basis, scale))
        coordinate_precision = _coordinate_precision(data, raw_basis)

    if data["cartesian"]:
        # reduced = cart * basis^-1 (row-vector convention); the universal scale cancels.
        reduced: Any = SurdVector(data["coords"]) * raw_basis.inv()
    else:
        reduced = data["coords"]
    sites = Sites(reduced, coordinate_precision)

    symbols = data.get("symbols")
    if symbols is None:
        raise ValueError(
            "_structure_from_poscar cannot build species for a VASP-4 POSCAR (no species symbols); "
            "provide a VASP-5 file with an explicit species line."
        )
    counts = data["counts"]

    species: list[Species] = []
    group_totals: dict[str, int] = {}
    group_numbers: dict[str, int] = {}
    for symbol in symbols:
        group_totals[symbol] = group_totals.get(symbol, 0) + 1
    species_at_sites: list[str] = []
    for symbol, count in zip(symbols, counts):
        group_numbers[symbol] = group_numbers.get(symbol, 0) + 1
        name = symbol if group_totals[symbol] == 1 else f"{symbol}{group_numbers[symbol]}"
        species_at_sites.extend([name] * count)
        species.append(Species(name=name, chemical_symbols=(symbol,), concentration=(1.0,)))

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

    # A Cartesian row vector transforms as reduced = Cartesian * basis^-1. If each
    # Cartesian component carries the same absolute bound p, reduced component j is
    # bounded by p * sum_i |basis^-1[i, j]|. The largest column sum is therefore the
    # conservative scalar precision needed by Sites. This is exact for a POSCAR basis,
    # whose decimal tokens are rational, and remains safe for skewed cells where dividing
    # by the shortest lattice-vector length can severely underestimate the uncertainty.
    inverse_surd = raw_basis.inv()
    if not inverse_surd.is_rational:
        raise RuntimeError("a POSCAR decimal basis unexpectedly produced an irrational inverse")
    inverse = inverse_surd.coefficient(1).to_fractions()
    factor = max(
        sum((abs(inverse[row][column]) for row in range(3)), start=fractions.Fraction()) for column in range(3)
    )
    return fractions.Fraction(precision) * factor


_STRUCTURE_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "vasp-poscar": _structure_from_poscar,
    "cif": _structure_from_cif,
    "mcif": _structure_from_mcif,
}


def _trajectory_from_payload(data: Mapping[str, Any]) -> Any:
    """Build a lazy VASPTrajectory; OUTCAR-only files are one-frame trajectories.

    The raw payload remains available through ``load(..., raw=True)``, including
    its ``final_energies`` object.
    """
    if not isinstance(data, Mapping) or data.get("format") not in (
        "vasp-outcar",
        "vasp-xdatcar",
        "httk-trajectory-jsonl",
    ):
        fmt = data.get("format") if isinstance(data, Mapping) else None
        raise ValueError(f"_trajectory_from_payload expected a VASP trajectory payload, got format={fmt!r}.")
    if data.get("format") == "httk-trajectory-jsonl":
        from httk.atomistic.models.trajectory.jsonl import JsonlTrajectory

        return JsonlTrajectory(data)

    from httk.atomistic.integrations.vasp.trajectory import VASPTrajectory

    return VASPTrajectory(data)


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
