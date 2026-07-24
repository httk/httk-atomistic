"""Build an exact :class:`~httk.atomistic.Structure` from a neutral POSCAR mapping.

:func:`structure_from_poscar` consumes the plain, string-preserving mapping
produced by ``httk.io.read_poscar`` (format tag ``"vasp-poscar"``) and turns it
into an exact :class:`~httk.atomistic.Structure`. It imports nothing from
*httk-io* — it only understands the neutral mapping shape — keeping the parsing
capability (*httk-io*) and the domain model (*httk-atomistic*) decoupled.

:func:`load_structure` is the convenience end-to-end entry point:
``httk.core.load`` picks the reader by file type, and a small adapter table maps
the payload's ``"format"`` tag to the matching structure builder.
"""

import fractions
from collections.abc import Mapping
from typing import Any, Callable

from httk.core import SurdScalar, SurdVector, load
from httk.core.vectors import exactmath

from .cell import Cell
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
        cell = Cell(cell_rows, scale_str)
    elif volume_str is not None:
        abs_det = abs(_to_fraction(raw_basis.det()))
        if abs_det == 0:
            raise ValueError("Cannot volume-scale a degenerate cell (zero determinant).")
        target_volume = _to_fraction(SurdVector.create(volume_str)._as_scalar())
        cell = Cell(cell_rows, _cube_root(target_volume / abs_det))
    else:
        cell = Cell(cell_rows)

    if data["cartesian"]:
        # reduced = cart * basis^-1 (row-vector convention); the universal scale cancels.
        reduced: Any = SurdVector.create(data["coords"]) * raw_basis.inv()
    else:
        reduced = data["coords"]

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

    return Structure(cell, reduced, species, species_at_sites)


#: Maps a loaded payload's ``"format"`` tag to the structure builder for it.
_STRUCTURE_ADAPTERS: dict[str, Callable[[Mapping[str, Any]], Structure]] = {
    "vasp-poscar": structure_from_poscar,
}


def load_structure(path: str) -> Structure:
    """Load a file and build a :class:`~httk.atomistic.Structure` from it.

    ``httk.core.load(path)`` selects the reader by file type (transparently
    decompressing ``.bz2`` / ``.gz`` files); the payload's ``"format"`` tag then
    selects the matching structure builder. A payload without a recognized
    ``"format"`` tag (for example a CIF, which returns a different shape) raises a
    clear :class:`ValueError`.
    """
    payload = load(path)
    fmt = payload.get("format") if isinstance(payload, Mapping) else None
    adapter = _STRUCTURE_ADAPTERS.get(fmt) if isinstance(fmt, str) else None
    if adapter is None:
        raise ValueError(
            f"Cannot build a Structure from {path!r}: unrecognized payload format tag {fmt!r}. "
            f"Known structure formats: {', '.join(sorted(_STRUCTURE_ADAPTERS))}."
        )
    return adapter(payload)
