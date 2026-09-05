"""Finding the magnetic symmetry of a full structure with collinear moments.

This is the magnetic counterpart of :func:`~httk.atomistic.recognize_asu`: the one place
that runs a symmetry search, here spglib's *magnetic* search, and turns its tolerant
floating-point answer into an exact :class:`~httk.atomistic.SymopsStructure`. The listed
sites are one representative per orbit, snapped exactly onto their site-stabilizer's fixed
subspace so that :class:`~httk.atomistic.SymopsStructure`'s exact, dedup-collapsing expansion regenerates
the input cell — same sites, species, and per-site moments — within the tolerance.

Only collinear moments aligned with ``z`` are handled. spglib's collinear magnetic search
takes a scalar per site; a non-collinear moment (any ``x`` or ``y`` component beyond the
tolerance) is a different problem and is rejected rather than silently projected.

Moments are handed to spglib as vectors, invoking its axial (non-collinear) magnetic search,
so the dataset is already the axial magnetic space group that :class:`~httk.atomistic.SymopsStructure` expands
with: the operations are used exactly as returned and the BNS number identifies that same
group. spglib's collinear *scalar* search reports a different, larger spin-decoupled group
(an index-2 supergroup for a simple antiferromagnet) whose BNS number would not match the
operations that actually keep an axial moment invariant, so it is deliberately not used.

Nothing else in the package searches for magnetic symmetry: the mCIF writer and a plain
:class:`~httk.atomistic.UnitcellStructure` stay detection-free, and this is the single
bridge across which a tolerance is spent.
"""

import fractions
from typing import Any

from httk.core import FracVector, SurdVector, register_citation

from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.symops import SymopsStructure
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry.affine_operation import AffineOperation

__all__ = ["find_magnetic_symmetry"]

#: Crystallographic change-of-origin fractions are halves, thirds, quarters, sixths, and
#: eighths; the same denominator bound spglib translations are snapped to in recognition.
_SPGLIB_TRANSLATION_MAX_DENOMINATOR = 48

#: Denominator bounds tried, smallest first, when idealising a stabilizer-projected
#: representative onto an exact special position. The exact orbit centroid is already a
#: fixed point, so these only sharpen a measured coordinate; whichever preserves exact
#: stabilizer invariance is kept, falling back to the unsnapped centroid.
_REP_SNAP_DENOMINATORS = (48, 10000)


def find_magnetic_symmetry(structure: StructureLike, tolerance: float = 1e-3) -> SymopsStructure:
    """Find the magnetic symmetry of a collinear structure and return it as symops.

    Runs spglib's magnetic-symmetry search on the full cell, then rebuilds the result as a
    :class:`~httk.atomistic.SymopsStructure` carrying, in the *input* cell:

    * every magnetic symmetry operation spglib found, each with its ``+1``/``-1``
      time-reversal flag;
    * one representative site per orbit, snapped exactly onto its site-stabilizer's fixed
      subspace so expansion regenerates the orbit and collapses the repeats;
    * the representatives' Cartesian moments, carried through unchanged;
    * the Belov–Neronova–Smirnova number of the magnetic space group.

    Expanding the returned structure back to a full cell reproduces ``structure`` — same
    sites, species, and per-site moments — to within ``tolerance`` and up to reordering.

    Only collinear moments along ``z`` are accepted: every site must have ``x`` and ``y``
    moment components no larger than ``tolerance``. The moments must be
    :class:`~httk.atomistic.CartesianSiteMoments`.

    :param structure: The full structure whose magnetic symmetry is sought.
    :param tolerance: The spglib search symprec, a Cartesian distance in the cell's units.
    :return: The magnetic symmetry as listed representatives plus operations.
    :raises ImportError: If the optional spglib dependency is unavailable.
    :raises ValueError: If the structure is not fully periodic, carries no Cartesian
        moments, has a non-collinear moment, or spglib finds no magnetic dataset.
    """
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    view = UnitcellStructureView(structure)
    require_full_periodicity(view.cell, "find_magnetic_symmetry")

    site_moments = view.site_moments
    if not isinstance(site_moments, CartesianSiteMoments):
        raise ValueError("find_magnetic_symmetry requires CartesianSiteMoments (collinear moments along z)")
    cartesian = site_moments.cartesian_moments.to_floats()
    if any(abs(row[0]) > tolerance or abs(row[1]) > tolerance for row in cartesian):
        raise ValueError("find_magnetic_symmetry requires collinear moments along z (zero x/y components)")
    # Vector moments invoke spglib's axial magnetic search, which returns the axial magnetic
    # space group directly -- the group SymopsStructure expands with, so no scalar-group filtering.
    # Project onto z (the collinear contract permits dropping the sub-tolerance in-plane noise):
    # a real relaxed-DFT moment carries a tiny x/y component that spglib 2.7.0 can SEGFAULT on
    # (an uncatchable process crash), and that also violates a mirror stabilizer at expansion.
    magmoms = [[0.0, 0.0, row[2]] for row in cartesian]

    species_at_sites = tuple(view.species_at_sites)
    names = sorted(set(species_at_sites))
    cell = view.cell
    lattice = cell.basis.to_floats()
    positions = view.sites.reduced_coords.to_floats()
    numbers = [names.index(name) + 1 for name in species_at_sites]

    dataset = _magnetic_dataset(lattice, positions, numbers, magmoms, tolerance)

    ops = _operations(dataset)
    reps, rep_species, rep_moment_rows = _representatives(
        view.sites.reduced_coords,
        species_at_sites,
        site_moments.cartesian_moments,
        ops,
        cell,
        tolerance,
    )

    reps_sites = Sites(FracVector([list(coord.to_fractions()) for coord in reps]), view.sites.precision)
    rep_moments = CartesianSiteMoments(
        SurdVector._from_scalar_grid(rep_moment_rows, (len(rep_moment_rows), 3)),
        precision=site_moments.precision,
    )
    return SymopsStructure(
        cell,
        reps_sites,
        view.species,
        rep_species,
        ops,
        site_moments=rep_moments,
        bns_number=_bns_number(dataset),
        bns_label=None,
        chemical_composition=view.chemical_composition,
        chemical_formula_descriptive=view.chemical_formula_descriptive,
        chemical_formula_hill=view.chemical_formula_hill,
        charge=view.charge,
    )


def _magnetic_dataset(
    lattice: Any,
    positions: Any,
    numbers: Any,
    magmoms: Any,
    tolerance: float,
) -> Any:
    """Run spglib's axial (vector) magnetic-symmetry search, or explain why it cannot."""
    try:
        import spglib
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "finding the magnetic symmetry of a structure requires spglib; "
            "install it with `pip install httk-atomistic[default]`"
        ) from error
    register_citation(
        applies_to="Magnetic symmetry detection uses spglib",
        references={
            "authors": (
                {"name": "Kohei Shinohara"},
                {"name": "Atsushi Togo"},
                {"name": "Isao Tanaka"},
            ),
            "title": "Algorithms for magnetic symmetry operation search and identification "
            "of magnetic space group from magnetic crystal structure",
            "journal": "Acta Crystallographica Section A",
            "volume": "79",
            "number": "5",
            "pages": "390-398",
            "year": "2023",
            "doi": "10.1107/S2053273323005016",
            "bib_type": "article",
        },
    )
    dataset = spglib.get_magnetic_symmetry_dataset((lattice, positions, numbers, magmoms), symprec=tolerance)
    if dataset is None:
        raise ValueError(f"spglib found no magnetic symmetry dataset at symprec={tolerance}")
    return dataset


def _operations(dataset: Any) -> list[tuple[AffineOperation, int]]:
    """Turn spglib's rotations, translations, and time reversals into exact httk operations.

    Rotations are exact integers in the same column-vector convention as
    :class:`~httk.atomistic.AffineOperation`. Translations are floats snapped to
    crystallographic fractions, and each time-reversal ``0``/``1`` becomes httk's ``+1``/``-1``.
    """
    operations: list[tuple[AffineOperation, int]] = []
    for rotation, translation, time_reversal in zip(dataset.rotations, dataset.translations, dataset.time_reversals):
        matrix = [[int(entry) for entry in row] for row in rotation]
        vector = [
            fractions.Fraction(float(component)).limit_denominator(_SPGLIB_TRANSLATION_MAX_DENOMINATOR)
            for component in translation
        ]
        operations.append((AffineOperation(matrix, vector), 1 - 2 * int(time_reversal)))
    return operations


def _representatives(
    coords: FracVector,
    species_at_sites: tuple[str, ...],
    cartesian_moments: SurdVector,
    ops: list[tuple[AffineOperation, int]],
    cell: Any,
    tolerance: float,
) -> tuple[list[FracVector], tuple[str, ...], list[list[Any]]]:
    """Pick one representative per orbit and project its coordinate onto its fixed subspace.

    Orbits are recomputed from the magnetic operations' spatial parts (equivalently to spglib's
    ``equivalent_atoms`` for the axial group) so that a shared representative expands back to the
    whole orbit.
    """
    spatial = [operation for operation, _ in ops]
    orbit_of = _orbits(coords, spatial, cell, tolerance)
    reps: list[FracVector] = []
    rep_species: list[str] = []
    rep_moment_rows: list[list[Any]] = []
    seen: set[int] = set()
    for index in range(len(coords)):
        orbit = orbit_of[index]
        if orbit in seen:
            continue
        seen.add(orbit)
        reps.append(_project_onto_stabilizer(coords[index], spatial, cell, tolerance))
        rep_species.append(species_at_sites[index])
        # Store the representative moment projected onto z, matching the z-only magmoms handed
        # to spglib: keeping a sub-tolerance in-plane component would break the axial expansion
        # against a mirror stabilizer that the detected group (found for a z-moment) contains.
        # ``zero`` is an exact scalar of the moments' own surd type (the grid rejects a bare int).
        z_component = cartesian_moments._element((index, 2))
        zero = z_component - z_component
        rep_moment_rows.append([zero, zero, z_component])
    return reps, tuple(rep_species), rep_moment_rows


def _orbits(
    coords: FracVector,
    spatial: list[AffineOperation],
    cell: Any,
    tolerance: float,
) -> list[int]:
    """Group sites into orbits under the operations, returning each site's orbit root.

    A union–find over sites: two sites share an orbit when some operation maps one onto the
    other within ``tolerance`` (compared as the shortest Cartesian distance in the real cell,
    as elsewhere in the symmetry package).
    """
    count = len(coords)
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for operation in spatial:
        for source in range(count):
            image = operation.apply(coords[source])
            for target in range(count):
                if _same_site(image, coords[target], cell, tolerance):
                    parent[find(source)] = find(target)
    return [find(index) for index in range(count)]


def _same_site(left: FracVector, right: FracVector, cell: Any, tolerance: float) -> bool:
    """Whether two fractional coordinates coincide modulo the lattice within ``tolerance``."""
    difference = left - right
    shortest = SurdVector(difference.normalize_half()) * cell.basis
    return float(shortest.lengthsqr().to_float()) <= tolerance * tolerance


def _project_onto_stabilizer(
    rep_coord: FracVector,
    spatial: list[AffineOperation],
    cell: Any,
    tolerance: float,
) -> FracVector:
    """The exact fixed point of a representative's site stabilizer, snapped to a special position.

    The stabilizer is the set of spatial operations that map the representative back onto
    itself within ``tolerance``. Its orbit centroid is, exactly, a point fixed by every one
    of those operations (each permutes the orbit, so it permutes the summands). Averaging
    the images — each unwrapped to lie nearest the representative first, so a ``0.99``/``0.01``
    split does not average to the cell centre — lands exactly on the fixed subspace.

    The centroid already round-trips, so snapping is only to recover an exact special
    position from a measured coordinate: the smallest denominator bound whose snapped point
    is still exactly stabilizer-invariant is kept, falling back to the unsnapped centroid.
    """
    stabilizer = [
        operation for operation in spatial if _same_site(operation.apply(rep_coord), rep_coord, cell, tolerance)
    ]
    reference = rep_coord.to_fractions()
    accumulated = [fractions.Fraction(0), fractions.Fraction(0), fractions.Fraction(0)]
    for operation in stabilizer:
        image = operation.apply(rep_coord).to_fractions()
        for axis in range(3):
            shift = _nearest_integer(image[axis] - reference[axis])
            accumulated[axis] += image[axis] - shift
    centroid = FracVector([value / len(stabilizer) for value in accumulated]).normalize()

    for bound in _REP_SNAP_DENOMINATORS:
        snapped = FracVector([value.limit_denominator(bound) for value in centroid.to_fractions()]).normalize()
        if all(operation.apply_wrapped(snapped) == snapped for operation in stabilizer):
            return snapped
    return centroid


def _nearest_integer(value: fractions.Fraction) -> fractions.Fraction:
    """The integer nearest an exact rational, rounding halves towards positive infinity."""
    from math import floor

    return fractions.Fraction(floor(value + fractions.Fraction(1, 2)))


def _bns_number(dataset: Any) -> str | None:
    """spglib's Belov–Neronova–Smirnova number for the detected group, as a string.

    This identifies the same axial magnetic space group whose operations are used (the vector
    search's ``uni_number``), so the number and the operation set are mutually consistent. There
    is no BNS symbol in spglib 2.7.0, so the label stays ``None`` at the call site. The value is
    already a string in spglib 2.7.0; it is passed through verbatim rather than via ``float`` so
    a number like ``4.10`` never loses its trailing digit.
    """
    import spglib

    magnetic_type = spglib.get_magnetic_spacegroup_type(int(dataset.uni_number))
    if magnetic_type is None:
        return None
    return magnetic_type.bns_number if isinstance(magnetic_type.bns_number, str) else str(magnetic_type.bns_number)
