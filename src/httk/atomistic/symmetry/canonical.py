"""A one-liner canonical form for noisy input: tolerant recognition composed with exact lifting.

:func:`canonical_asu` bridges the two layers.  The tolerant layer (:func:`~httk.atomistic.recognize_asu`, backed by
spglib) snaps a measured structure onto a space group within a Cartesian tolerance; the exact layer
(:func:`~httk.atomistic.canonicalize`) then removes spglib's representational freedom -- setting,
origin, orbit representatives, basis orientation -- and returns the deterministic canonical
representative.  Because a measured structure can sit just inside or just outside a tolerance
boundary, recognition is swept over a few symprec multiples and each candidate is re-checked against
the *base* tolerance, so a looser sweep member can rescue a boundary flip without ever accepting more
than the claimed noise.

This module is deliberately outside ``lift.py``: ``lift`` imports ``recognition`` (for its tolerance
helpers) and stays spglib-free, while this composer imports both.
"""

from collections import Counter
from fractions import Fraction

from httk.core import FracVector

from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.lift import LiftResult, canonicalize
from httk.atomistic.symmetry.lift import _canonical_result_key as _canonical_key
from httk.atomistic.symmetry.recognition import (
    _cartesian_distance_squared,
    recognize_asu,
    structure_tolerance,
)

__all__ = ["canonical_asu"]


def _fits_within(view: UnitcellStructureView, recognized: ASUStructure, tolerance: float) -> bool:
    """Whether the recognized model reproduces every input site within ``tolerance`` (Cartesian).

    The recognized ASU lives in the input's own frame, so its expansion shares the input cell.  The
    match is *injective*: each input site is paired with a distinct same-species model site, greedily
    over pairs sorted by distance, so two input sites cannot both claim one model site and leave a
    third model site orphaned.  Every input site must find such a partner within ``tolerance`` and the
    per-species counts must match, so a model that drops, adds, or merges atoms is rejected.

    Distances use :func:`_cartesian_distance_squared`, a per-component minimum-image wrap -- an upper
    bound on the true minimum-image distance (exact for the near-coincident pairs that matter here,
    conservative -- i.e. it can only over-estimate -- otherwise).
    """
    expanded = UnitcellStructureView(recognized)
    input_coords = view.sites.reduced_coords.to_fractions()
    input_species = list(view.species_at_sites)
    model_coords = expanded.sites.reduced_coords.to_fractions()
    model_species = list(expanded.species_at_sites)
    if Counter(input_species) != Counter(model_species):
        return False
    cell = view.cell
    limit = tolerance * tolerance
    pairs = sorted(
        (_cartesian_distance_squared(FracVector(input_coords[i]) - FracVector(model_coords[m]), cell), i, m)
        for i in range(len(input_coords))
        for m in range(len(model_coords))
        if model_species[m] == input_species[i]
    )
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
) -> ASUStructure:
    """Return the canonical :class:`~httk.atomistic.ASUStructure` of the highest symmetry that fits.

    This is the noisy-input counterpart to :func:`~httk.atomistic.canonicalize`: it recognizes the
    symmetry of a measured structure with spglib and then canonicalizes the result exactly.

    An :class:`~httk.atomistic.ASUStructure` input is expanded to its unit cell first and the symmetry
    is re-recognized from the actual coordinates -- always from the geometry, never the declared
    label.  Re-recognition can raise a declared symmetry (a hand-written low-symmetry cell whose
    coordinates in fact support more) and can also lower it (a declared symmetry the coordinates do
    not support at the derived tolerance).

    Recognition is run at each ``base * factor`` symprec.  A member is accepted only when its
    recognized model reproduces every input site within the **base** tolerance (never the swept one),
    by an injective same-species match, and matches the per-species site counts; this lets a looser
    member rescue a tolerance-boundary flip without accepting more than the claimed noise.  Accepted
    members are canonicalized at the base tolerance, deduplicated by their exact canonical key, and
    the best is selected -- most symmetry operations, then lowest IT number, then smallest residual,
    then the exact canonical key -- mirroring :func:`~httk.atomistic.highest_symmetry`.

    Tolerance bound: only the recognition stage is held to the base tolerance.  Each exact lift hop
    can move coordinates and snap the metric by up to another ``base``, and the lift residual and path
    are not re-checked against the input here, so the returned structure's distance from the input is
    bounded roughly by ``base * (1 + hops)``, not by ``base`` alone.  A post-hoc re-check is not
    offered because the canonical structure is generally expressed in a different cell.

    Determinism: the recognition stage is floating-point/spglib-based, so its outcome is reproducible
    on one platform but may differ across floating-point architectures or spglib builds.  The exact
    canonicalization stage is platform-independent and erases spglib's representational freedom
    (setting, origin, orbit representatives, basis orientation), so cross-platform variation is
    confined to *which* symmetry is accepted near a tolerance boundary, never to *how* an accepted
    symmetry is represented.  Free-parameter values are least-squares fits of the measured
    coordinates, so two noisy measurements of the same crystal canonicalize to the same Wyckoff
    choices but slightly different rational parameter values.

    :param structure: The measured structure, ``UnitcellStructure`` or ``ASUStructure``.
    :param tolerance: The base Cartesian tolerance, or ``None`` to derive it from the structure's
        stated precision (:func:`~httk.atomistic.symmetry.recognition.structure_tolerance`).
    :param factors: Multipliers for the recognition symprec sweep; each candidate symprec is
        ``base * factor``.
    :return: The canonical asymmetric unit of the accepted highest-symmetry group.
    :raises ImportError: If spglib is unavailable when symmetry must be searched (the error names the
        ``httk-atomistic[default]`` extra).
    :raises ValueError: If recognition fails or is rejected at every swept tolerance.
    """
    view = UnitcellStructureView(structure)
    base = structure_tolerance(view) if tolerance is None else float(tolerance)

    accepted: list[tuple[int, ASUStructure]] = []
    failures: list[str] = []
    for factor in factors:
        symprec = base * float(factor)
        try:
            recognized = recognize_asu(view, tolerance=symprec)
        except ValueError as error:
            failures.append(f"{symprec:g}: {error}")
            continue
        if not _fits_within(view, recognized, base):
            failures.append(f"{symprec:g}: recognized model exceeds the base tolerance")
            continue
        accepted.append((len(recognized.spacegroup.symmetry_operations), recognized))

    if not accepted:
        raise ValueError(f"no symmetry fit the structure within tolerance {base:g}; tried [{', '.join(failures)}]")

    # Canonicalize only the most-recognized-symmetry members (there is usually one), which also
    # avoids the slow many-site BFS of an under-recognized (tight-symprec) snapping.  canonicalize()
    # does lift symmetry (e.g. a recognized IT 99 can canonicalize to IT 129), so the prune is not
    # "canonicalize only refines": it is sound iff spglib's recognized operation count is monotone in
    # symprec -- a looser symprec never recognizes *fewer* operations than a tighter one -- so the
    # top-recognized tier already contains the input at its highest recognizable symmetry.
    # ponytail: monotonicity held in every reviewer probe but is a spglib property, not a theorem;
    # widen to all accepted members if a counterexample ever appears.
    accepted.sort(key=lambda item: -item[0])
    top_operations = accepted[0][0]
    candidates: dict[tuple, LiftResult] = {}
    for recognized_operations, recognized in accepted:
        if recognized_operations < top_operations:
            break
        result = canonicalize(recognized, tolerance=base)
        candidates.setdefault(_canonical_key(result), result)

    best = min(
        candidates.values(),
        key=lambda result: (
            -len(result.asu.spacegroup.symmetry_operations),
            result.asu.spacegroup.it_number,
            result.residual,
            _canonical_key(result),
        ),
    )
    return best.asu
