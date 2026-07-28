"""Build an exact :class:`~httk.atomistic.Structure` from a neutral POSCAR mapping.

:func:`structure_from_poscar` consumes the plain, string-preserving mapping
produced by ``httk.io.read_poscar`` (format tag ``"vasp-poscar"``) and turns it
into an exact :class:`~httk.atomistic.Structure`. It imports nothing from
*httk-io* — it only understands the neutral mapping shape — keeping the parsing
capability (*httk-io*) and the domain model (*httk-atomistic*) decoupled.

:func:`structure_from_payload` exposes the payload-to-domain conversion used by
the format-adapter registry. :func:`load_structure` is the explicit convenience
entry point for callers that want to name the domain operation themselves.
"""

import fractions
import math
from collections.abc import Mapping
from typing import Any, Callable

from httk.core import SurdScalar, SurdVector, exactmath, load

from ._vector_guards import to_surdscalar
from .cell import Cell
from .sites import Sites
from .species import Species
from .structure import Structure

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


def structure_from_poscar(data: Mapping[str, Any]) -> Structure:
    """Build an exact :class:`~httk.atomistic.Structure` from a neutral POSCAR mapping.

    ``data`` is the mapping returned by ``httk.io.read_poscar`` (its ``format``
    must be ``"vasp-poscar"``). The cell basis is taken exactly from the file's
    string rows. For a positive universal scaling factor the ``scale`` string is
    used directly as the cell's exact scale; for a negative scaling factor (a
    target **volume** ``V``) the scale is the cube root of ``V / |det(basis)|`` —
    a value outside the exact surd field, so it is a deterministic rational
    approximation (the basis rows themselves stay exact).

    Direct coordinates become reduced coordinates directly (exact strings).
    Cartesian coordinates are converted exactly as ``cart * basis.inv()`` under
    the row-vector convention; because VASP scales *both* the lattice vectors and
    the Cartesian positions by the universal scaling factor, that factor cancels
    and the reduced coordinates are exact regardless of the scale/volume case.

    Species come from the VASP-5 species line (one single-element, unattached
    :class:`~httk.atomistic.Species` of concentration 1.0 per distinct symbol);
    a VASP-4 file (no species symbols) raises a :class:`ValueError`. Selective
    dynamics flags, if present, are ignored.
    """
    fmt = data.get("format")
    if fmt != "vasp-poscar":
        raise ValueError(f"structure_from_poscar expected a 'vasp-poscar' mapping, got format={fmt!r}.")

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
            "structure_from_poscar cannot build species for a VASP-4 POSCAR (no species symbols); "
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

    return Structure(cell, sites, species, species_at_sites)


def _structure_from_cif(data: Mapping[str, Any]) -> Structure:
    """The full unit cell of a loaded CIF, expanded from its asymmetric unit."""
    from .cif_structures import asu_structures_from_cif
    from .unitcell_structure_view import UnitcellStructureView

    structures = asu_structures_from_cif(data)
    if len(structures) != 1:
        raise ValueError(
            f"this CIF holds {len(structures)} structures; load_structure() builds one, so use "
            f"httk.atomistic.asu_structures_from_cif(httk.core.load(path, raw=True)) to get them all"
        )
    return UnitcellStructureView(structures[0])


def _basis_precision(data: Mapping[str, Any], raw_basis: SurdVector, scale: Any) -> fractions.Fraction | None:
    """How precisely the POSCAR states its cell, as an absolute length.

    The reader reports the precision of the cell-vector tokens *as written*, before the
    universal scaling factor multiplies them, so the conversion happens here where the
    scale is known: ``scale * cell_precision``.

    The scale's **own** written precision is deliberately not folded in, even though the
    reader reports it. The scaling factor is a definition of units — a choice about whether
    to carry the lattice constant in the factor or in the rows — and not a quantity measured
    independently of them. Charging its digits as well would double-count the same
    measurement, and would give a nonsensical answer in the ordinary case: a POSCAR that
    writes ``1.0`` would be charged 0.1 of an uncertainty it plainly does not have, and the
    cell of a 5.64 A structure would be declared good to only half an angstrom.

    For the negative, volume-scaling form the scale is derived from a target volume rather
    than written; the volume's precision is likewise not propagated through the cube root.
    """
    cell_precision = data.get("cell_precision")
    if cell_precision is None:
        return None
    scale_value = abs(_to_fraction(to_surdscalar(scale)))
    return scale_value * fractions.Fraction(cell_precision)


def _coordinate_precision(data: Mapping[str, Any], raw_basis: SurdVector) -> fractions.Fraction | None:
    """How precisely the POSCAR states its coordinates, in fractional units.

    Direct coordinates are already fractional and pass straight through — and the universal
    scaling factor cancels for them, exactly as it does for the coordinates themselves.

    Cartesian coordinates are a length, so they are divided by the *shortest* cell edge to
    become fractional. Shortest because that yields the largest fractional uncertainty,
    which is the conservative choice; it is an approximation for a strongly oblique cell,
    where a rigorous bound would need the norm of the inverse basis. The scale cancels here
    too, since it multiplies the coordinates and the edge lengths alike.
    """
    precision = data.get("coordinate_precision")
    if precision is None:
        return None
    if not data["cartesian"]:
        return fractions.Fraction(precision)

    shortest = min(math.dist((0.0, 0.0, 0.0), row) for row in raw_basis.to_floats())
    if shortest <= 0:
        return None
    return fractions.Fraction(precision) / fractions.Fraction(str(shortest)).limit_denominator(10**12)


#: Maps a loaded payload's ``"format"`` tag to the structure builder for it.
_STRUCTURE_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Structure]] = {
    "vasp-poscar": structure_from_poscar,
    "cif": _structure_from_cif,
}


def structure_from_payload(data: Mapping[str, Any]) -> Structure:
    """Build a :class:`~httk.atomistic.Structure` from a tagged neutral payload.

    The ``"format"`` tag selects the structure builder. Currently supported
    payloads are ``"vasp-poscar"`` and single-block ``"cif"`` mappings. A
    multi-block CIF retains the same :class:`ValueError` as
    :func:`load_structure`; use :func:`~httk.atomistic.cif_structures.asu_structures_from_cif` for all blocks.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"structure_from_payload expected a mapping, got {type(data).__name__}.")
    fmt = data.get("format")
    adapter = _STRUCTURE_ADAPTERS.get(fmt) if isinstance(fmt, str) else None
    if adapter is None:
        raise ValueError(
            f"unrecognized payload format tag {fmt!r}. "
            f"Known structure formats: {', '.join(sorted(_STRUCTURE_ADAPTERS))}."
        )
    return adapter(data)


def load_structure(path: str) -> Structure:
    """Load a file and build a :class:`~httk.atomistic.Structure` from it.

    ``httk.core.load(path, raw=True)`` selects the reader by file type
    (transparently decompressing ``.bz2`` / ``.gz`` files); the payload's
    ``"format"`` tag then selects the matching structure builder. A payload
    without a recognized ``"format"`` tag raises a clear :class:`ValueError`.

    For a CIF this expands the asymmetric unit into the full cell. Use
    :func:`load_asu_structure` instead to keep the asymmetric-unit form, which is both
    smaller and richer — it carries the space group, the Wyckoff assignment of every
    site, and the setting the file was written in.
    """
    payload = load(path, raw=True)
    try:
        return structure_from_payload(payload)
    except ValueError as exc:
        raise ValueError(f"Cannot build a Structure from {path!r}: {exc}") from exc


def load_asu_structure(path: str, **options: Any) -> Any:
    """Load a file and build an :class:`~httk.atomistic.ASUStructure` from it.

    Currently CIF only, since it is the format that states its own symmetry. Unlike
    :func:`load_structure` this keeps the asymmetric-unit form rather than expanding it,
    so the space group, the Wyckoff position of each site, and the file's setting all
    survive. Expand it at any time with ``UnitcellStructureView(asu)``.

    ``options`` are passed to :func:`~httk.atomistic.asu_structure_from_cif` — notably
    ``tolerance`` and ``trust_declared_symmetry``, the latter being the way to load a file
    whose declared Hall symbol or space-group number contradicts its own symmetry
    operations.
    """
    from .cif_structures import asu_structures_from_cif

    payload = load(path, raw=True)
    fmt = payload.get("format") if isinstance(payload, Mapping) else None
    if fmt != "cif":
        raise ValueError(
            f"Cannot build an ASUStructure from {path!r}: only CIF files state the symmetry an "
            f"asymmetric unit needs, but this payload has format tag {fmt!r}."
        )
    structures = asu_structures_from_cif(payload, **options)
    if len(structures) != 1:
        raise ValueError(
            f"this CIF holds {len(structures)} structures; use "
            f"httk.atomistic.asu_structures_from_cif(httk.core.load(path, raw=True)) to get them all"
        )
    return structures[0]
