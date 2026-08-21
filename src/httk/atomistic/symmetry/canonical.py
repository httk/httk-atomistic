"""A one-liner canonical form for noisy input: tolerant recognition composed with exact lifting.

:func:`canonical_asu` bridges the two layers.  The tolerant layer (:func:`~httk.atomistic.recognize_asu`,
backed by spglib) snaps a measured structure onto a space group within a Cartesian tolerance.  Because
a measured structure can sit just inside or just outside a tolerance boundary, recognition is swept
over a few symprec multiples from loosest to tightest, and the first member that fits within the
*base* tolerance wins -- so a looser member rescues a boundary flip without ever accepting more than
the claimed noise.  The exact layer then fixes that winner's representation deterministically: by
default (``lift=False``) it returns the canonical representative *within the recognized group*
(setting, origin, orbit representatives, basis orientation all fixed) without searching upward;
``lift=True`` additionally runs :func:`~httk.atomistic.canonicalize` to hunt higher pseudosymmetry the
recognition missed.

This module is deliberately outside ``lift.py``: ``lift`` imports ``recognition`` (for its tolerance
helpers) and stays spglib-free, while this composer imports both.
"""

from collections import Counter
from fractions import Fraction

from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.lift import _canonical_without_bfs, canonicalize
from httk.atomistic.symmetry.recognition import recognize_asu, structure_tolerance

__all__ = ["canonical_asu"]


def _fits_within(view: UnitcellStructureView, recognized: ASUStructure, tolerance: float) -> bool:
    """Whether the recognized model reproduces every input site within ``tolerance`` (Cartesian).

    The recognized ASU lives in the input's own frame, so its expansion shares the input cell.  The
    match is *injective*: each input site is paired with a distinct same-species model site, greedily
    over pairs sorted by distance, so two input sites cannot both claim one model site and leave a
    third model site orphaned.  Every input site must find such a partner within ``tolerance`` and the
    per-species counts must match, so a model that drops, adds, or merges atoms is rejected.

    This is the tolerant layer, so distances are computed in plain floats (a per-component
    minimum-image wrap -- an upper bound on the true minimum-image distance, exact for the
    near-coincident pairs that matter, conservative i.e. over-estimating otherwise).  Near-boundary
    acceptance is therefore float-determined, as the whole recognition stage already is.
    """
    expanded = UnitcellStructureView(recognized)
    input_coords = [[float(value) for value in row] for row in view.sites.reduced_coords.to_fractions()]
    input_species = list(view.species_at_sites)
    model_coords = [[float(value) for value in row] for row in expanded.sites.reduced_coords.to_fractions()]
    model_species = list(expanded.species_at_sites)
    if Counter(input_species) != Counter(model_species):
        return False
    basis = view.cell.basis.to_floats()
    limit = tolerance * tolerance
    pairs: list[tuple[float, int, int]] = []
    for i, (coordinate, species) in enumerate(zip(input_coords, input_species)):
        for m, (other, other_species) in enumerate(zip(model_coords, model_species)):
            if other_species != species:
                continue
            wrapped = [coordinate[k] - other[k] - round(coordinate[k] - other[k]) for k in range(3)]
            cartesian = [sum(wrapped[k] * basis[k][axis] for k in range(3)) for axis in range(3)]
            pairs.append((sum(component * component for component in cartesian), i, m))
    pairs.sort()
    claimed_inputs: set[int] = set()
    claimed_models: set[int] = set()
    for distance, i, m in pairs:
        if distance > limit:
            break  # pairs are sorted, so no unclaimed input can still be matched within tolerance
        if i in claimed_inputs or m in claimed_models:
            continue
        claimed_inputs.add(i)
        claimed_models.add(m)
    return len(claimed_inputs) == len(input_coords)


def canonical_asu(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    factors: tuple[Fraction | float | int, ...] = (Fraction(1, 5), 1, 5),
    lift: bool = False,
    preserve_chirality: bool = False,
) -> ASUStructure:
    """Return the canonical :class:`~httk.atomistic.ASUStructure` of a measured structure's symmetry.

    This is the noisy-input counterpart to :func:`~httk.atomistic.canonicalize`: it recognizes the
    symmetry of a measured structure with spglib and then canonicalizes the result exactly.

    An :class:`~httk.atomistic.ASUStructure` input is expanded to its unit cell first and the symmetry
    is re-recognized from the actual coordinates -- always from the geometry, never the declared
    label.  Re-recognition can raise a declared symmetry (a hand-written low-symmetry cell whose
    coordinates in fact support more) and can also lower it (a declared symmetry the coordinates do
    not support at the derived tolerance).

    Recognition is swept over the ``base * factor`` symprecs from loosest to tightest.  A member is
    accepted only when its recognized model reproduces every input site within the **base** tolerance
    (never the swept one), by an injective same-species match, and matches the per-species site counts.
    The first accepted member wins -- by the same operation-count monotonicity the loosest fitting
    member is the highest-symmetry one -- so recognition (and the expensive stage) runs once in the
    common case, and a looser member still rescues a tolerance-boundary flip a tighter one fails.

    ``lift`` selects the expensive stage applied to that winner:

    * ``lift=False`` (default): it is mapped to the deterministic canonical representative *within its
      recognized group* -- the exact terminal representation (setting, origin, orbit representatives,
      basis orientation all fixed), returned *without* searching upward.  The result is the canonical
      form of the recognized symmetry; no pseudosymmetry above it is sought.  Per-structure cost is
      essentially recognition-bound.
    * ``lift=True``: it is additionally run through the exact upward search
      (:func:`~httk.atomistic.canonicalize`) to find higher pseudosymmetry the recognition missed.
      This is exact but can be expensive -- minutes and beyond for low-symmetry, many-atom cells.

    Tolerance bound: the recognition stage is held to the base tolerance -- every returned atom sits
    within ``base`` of the input.  Under ``lift=False`` that is the whole bound (no further hops).
    Under ``lift=True`` each lift hop can move coordinates and snap the metric by up to another
    ``base`` and the residual/path are not re-checked here, so the returned structure's distance from
    the input is bounded roughly by ``base * (1 + hops)``.

    Determinism: the recognition stage is floating-point/spglib-based, so its outcome is reproducible
    on one platform but may differ across floating-point architectures or spglib builds.  The exact
    stage is platform-independent and erases spglib's representational freedom for ordinary rational
    crystallographic Gram matrices, so cross-platform variation is confined to *which* symmetry is
    accepted near a tolerance boundary, never to *how* an accepted symmetry is represented.  An exact
    non-rational Gram whose canonical Cartesian factor requires nested radicals outside the supported
    surd field remains idempotent but can retain its input's global Cartesian rotation.  Free-parameter
    values are least-squares fits of the measured coordinates, so two noisy measurements of the same
    crystal reach the same Wyckoff choices but slightly different rational parameter values.

    :param structure: The measured structure, ``UnitcellStructure`` or ``ASUStructure``.
    :param tolerance: The base Cartesian tolerance, or ``None`` to derive it from the structure's
        stated precision (:func:`~httk.atomistic.symmetry.recognition.structure_tolerance`).
    :param factors: Multipliers for the recognition symprec sweep; each candidate symprec is
        ``base * factor``.
    :param lift: Whether to search upward for pseudosymmetry above the recognized group (default
        ``False``: return the canonical representative of the recognized symmetry).
    :param preserve_chirality: How enantiomorphic space groups are canonicalized.  By default
        (``False``) a result landing in the higher member of one of the 11 enantiomorphic pairs
        (76/78, 91/95, 92/96, 144/145, 151/153, 152/154, 169/170, 171/172, 178/179, 180/181, 212/213)
        is mapped to the LOWER-numbered member by an exact chirality-flipping transformation
        (fractional coordinates ``f -> (-f) mod 1`` with the cell basis unchanged -- the Cartesian
        inversion ``r -> -r`` -- and the group swapped to its partner), so an enantiomorphic pair shares
        one canonical representative and the canonical labels of the two partners coincide.  Set
        ``True`` to keep the recognized group and retain the distinction.  A structure carrying site
        moments is never flipped (axial vectors are out of scope under improper maps) and is left in
        its own group regardless of this flag.
    :return: The canonical asymmetric unit.
    :raises ImportError: If spglib is unavailable when symmetry must be searched (the error names the
        ``httk-atomistic[default]`` extra).
    :raises ValueError: If recognition fails or is rejected at every swept tolerance.
    """
    view = UnitcellStructureView(structure)
    base = structure_tolerance(view) if tolerance is None else float(tolerance)

    # Loosest symprec first: by the spglib op-count monotonicity assumption a looser symprec never
    # recognizes fewer operations, so the FIRST member that both recognizes and fits within the base
    # tolerance is the highest-symmetry one -- take it and stop, running recognition (and the expensive
    # canonicalization stage) once in the common case.
    # ponytail: if monotonicity ever fails, this (like the earlier tier prune) can settle for the rare
    # lower-symmetry loosest member where an all-members scan would have found a higher tighter one.
    failures: list[str] = []
    winner: ASUStructure | None = None
    for factor in sorted(factors, key=lambda value: -float(value)):
        symprec = base * float(factor)
        try:
            recognized = recognize_asu(view, tolerance=symprec)
        except ValueError as error:
            failures.append(f"{symprec:g}: {error}")
            continue
        if not _fits_within(view, recognized, base):
            failures.append(f"{symprec:g}: recognized model exceeds the base tolerance")
            continue
        winner = recognized
        break

    if winner is None:
        raise ValueError(f"no symmetry fit the structure within tolerance {base:g}; tried [{', '.join(failures)}]")

    if lift:
        return canonicalize(winner, tolerance=base, preserve_chirality=preserve_chirality).asu
    return _canonical_without_bfs(winner, preserve_chirality=preserve_chirality)
