"""A one-liner canonical form for noisy input: tolerant recognition composed with exact lifting.

:func:`canonical_asu` bridges the two layers. Before tolerant recognition, the exact unit-cell geometry
is normalized as P1, giving spglib one deterministic basis, origin, and site order for every exact
re-expression of the same measured structure. The tolerant layer (:func:`~httk.atomistic.recognize_asu`,
backed by spglib) then snaps that stable input onto a space group within a Cartesian tolerance. Because
a measured structure can sit just inside or just outside a tolerance boundary, recognition is swept
over a few symprec multiples from loosest to tightest, and the first member that fits within the
*base* tolerance wins. The exact layer finally fixes that winner's representation deterministically:
by default (``lift=False``) it returns the canonical representative *within the recognized group*;
``lift=True`` additionally runs :func:`~httk.atomistic.canonicalize` to hunt higher pseudosymmetry.

This module is deliberately outside ``lift.py``: ``lift`` imports ``recognition`` (for its tolerance
helpers) and stays spglib-free, while this composer imports both.
"""

import logging
from collections import Counter
from fractions import Fraction

from httk.core import FracVector

from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.lift import _canonical_without_bfs, canonicalize
from httk.atomistic.symmetry.recognition import recognize_asu, structure_tolerance

__all__ = ["canonical_asu"]


def _exact_p1(view: UnitcellStructureView) -> ASUStructure:
    """Return the exact unit-cell geometry as a P1 asymmetric unit."""
    sites = [
        WyckoffSite("a", FracVector(coordinate).normalize(), species)
        for coordinate, species in zip(view.sites.reduced_coords.to_fractions(), view.species_at_sites)
    ]
    return ASUStructure(
        view.cell,
        1,
        sites,
        view.species,
        coordinate_precision=view.sites.precision,
        charge=view.charge,
    )


def _p1_fallback(view: UnitcellStructureView, failures: list[str]) -> ASUStructure:
    """Report failed recognition and return the exact unit-cell geometry as P1.

    Exhausting spglib's tolerance sweep proves only that no higher symmetry could be
    recognized reliably. P1 needs no recognition or coordinate snapping: every input row
    is already one general-position orbit representative. Invalid coincident rows remain
    invalid and are rejected by ``ASUStructure`` when the fallback is expanded downstream.
    """
    if view.site_moments is not None:
        raise ValueError(
            "no symmetry fit the structure and exact P1 fallback does not yet support site moments; "
            f"tried [{', '.join(failures)}]"
        )
    logging.getLogger(__name__).warning(
        "no symmetry fit the structure within the requested tolerance sweep; using its exact P1 geometry",
        extra={"context": "symmetry", "attempts": tuple(failures)},
    )
    return _exact_p1(view)


def _reversed_p1_frame(structure: ASUStructure) -> ASUStructure:
    """Return a fixed secondary spglib frame with the canonical P1 site order reversed.

    Spglib can be sensitive to which atom is encountered first even when basis, origin, and geometry
    are identical. The primary sorted P1 frame remains the common path; this deterministic reverse is
    tried only when that frame finds no symmetry above P1. Unlike retrying the caller's original
    frame, it is identical for every exact re-expression of the measured structure.
    """
    if structure.spacegroup.it_number != 1:
        raise ValueError("secondary recognition frame requires a P1 structure")
    return ASUStructure(
        structure.cell,
        structure.spacegroup,
        tuple(reversed(structure.wyckoff_sites)),
        structure.species,
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _fits_within(view: UnitcellStructureView, recognized: ASUStructure, tolerance: float) -> bool:
    """Whether the recognized model reproduces every input site within ``tolerance`` (Cartesian).

    The recognized ASU lives in the input's own frame, so its expansion shares the input cell.  The
    match is *bijective*: an augmenting-path bipartite matcher pairs every input site with a distinct
    same-species model site within ``tolerance``. A greedy nearest-pair matcher is insufficient here:
    assigning a flexible input to the closest model can strand another input that had only that model
    available, and the outcome then depends on site ordering. Per-species counts must also match, so a
    model that drops, adds, or merges atoms is rejected.

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
    candidates: list[list[tuple[float, int]]] = [[] for _ in input_coords]
    for i, (coordinate, species) in enumerate(zip(input_coords, input_species)):
        for m, (other, other_species) in enumerate(zip(model_coords, model_species)):
            if other_species != species:
                continue
            wrapped = [coordinate[k] - other[k] - round(coordinate[k] - other[k]) for k in range(3)]
            cartesian = [sum(wrapped[k] * basis[k][axis] for k in range(3)) for axis in range(3)]
            distance = sum(component * component for component in cartesian)
            if distance <= limit:
                candidates[i].append((distance, m))
    if any(not choices for choices in candidates):
        return False
    for choices in candidates:
        choices.sort()

    # Deterministic Kuhn-style augmenting paths, implemented iteratively so large cells do not risk
    # Python's recursion limit. Start with the most constrained inputs to keep paths short.
    model_for_input: dict[int, int] = {}
    input_for_model: dict[int, int] = {}
    for root in sorted(range(len(input_coords)), key=lambda index: (len(candidates[index]), index)):
        queue = [root]
        seen_inputs = {root}
        seen_models: set[int] = set()
        previous_input: dict[int, int] = {}
        terminal: int | None = None
        while queue and terminal is None:
            current = queue.pop(0)
            for _distance, model in candidates[current]:
                if model in seen_models:
                    continue
                seen_models.add(model)
                previous_input[model] = current
                matched = input_for_model.get(model)
                if matched is None:
                    terminal = model
                    break
                if matched not in seen_inputs:
                    seen_inputs.add(matched)
                    queue.append(matched)
        if terminal is None:
            return False
        model = terminal
        while True:
            current = previous_input[model]
            previous_model = model_for_input.get(current)
            model_for_input[current] = model
            input_for_model[model] = current
            if previous_model is None:
                break
            model = previous_model
    return True


def _recognition_sweep(
    view: UnitcellStructureView,
    base: float,
    factors: tuple[Fraction | float | int, ...],
) -> tuple[ASUStructure | None, list[str]]:
    """Return the loosest fitting recognized model and diagnostics for failed members."""
    failures: list[str] = []
    for factor in sorted(factors, key=lambda value: -float(value)):
        symprec = base * float(factor)
        try:
            recognized = recognize_asu(view, tolerance=symprec, _retain_found_transform=True)
        except ValueError as error:
            failures.append(f"{symprec:g}: {error}")
            continue
        if not _fits_within(view, recognized, base):
            failures.append(f"{symprec:g}: recognized model exceeds the base tolerance")
            continue
        return recognized, failures
    return None, failures


def canonical_asu(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    factors: tuple[Fraction | float | int, ...] = (Fraction(1, 5), 1, 5),
    lift: bool = False,
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Return the canonical :class:`~httk.atomistic.ASUStructure` of a measured structure's symmetry.

    This is the noisy-input counterpart to :func:`~httk.atomistic.canonicalize`: it first normalizes
    the exact measured geometry in P1, recognizes its symmetry with spglib, and then canonicalizes the
    recognized result exactly. P1 preconditioning prevents spglib's tolerance-boundary result from
    depending on an equivalent input shear, origin shift, or site ordering.

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
      form of the recognized symmetry; no pseudosymmetry above it is sought.
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
    :param preserve_chirality: How enantiomorphic space groups are canonicalized.  The default
        (``True``) is to keep the recognized group. When ``False`` a result landing in the higher
        member of one of the 11 enantiomorphic pairs (76/78, 91/95, 92/96, 144/145, 151/153, 152/154,
        169/170, 171/172, 178/179, 180/181, 212/213) is mapped to the LOWER-numbered member by an
        exact chirality-flipping transformation (fractional coordinates ``f -> (-f) mod 1`` with
        the cell basis unchanged -- the Cartesian inversion ``r -> -r`` -- and the group swapped to
        its partner), so an enantiomorphic pair shares one canonical representative and the canonical
        labels of the two partners coincide.  A structure carrying site moments is never flipped
        (axial vectors are out of scope under improper maps) and is left in its own group regardless
        of this flag.
    :return: The canonical asymmetric unit.
    :raises ImportError: If spglib is unavailable when symmetry must be searched (the error names the
        ``httk-atomistic[default]`` extra).
    If recognition fails or is rejected at every swept tolerance, the exact unit-cell
    geometry is canonicalized as P1 instead. This fallback performs no tolerance-level
    snapping and therefore cannot invent symmetry that spglib did not establish.
    """
    source_view = UnitcellStructureView(structure)
    base = structure_tolerance(source_view) if tolerance is None else float(tolerance)

    # spglib is not representation-invariant near a tolerance boundary: equivalent lattice shears
    # can make it return different snapped models. Canonicalize the exact, unsnapped geometry as P1
    # first so every such description reaches spglib in the same frame. Magnetic normal forms are not
    # yet supported by the exact P1 path, so retain the established direct-recognition behavior there.
    normalized_p1: ASUStructure | None = None
    if source_view.site_moments is None:
        normalized_p1 = _canonical_without_bfs(_exact_p1(source_view), preserve_chirality=True)
        view = UnitcellStructureView(normalized_p1)
    else:
        view = source_view

    # Loosest symprec first: by the spglib op-count monotonicity assumption a looser symprec never
    # recognizes fewer operations, so the FIRST member that both recognizes and fits within the base
    # tolerance is the highest-symmetry one -- take it and stop, running recognition (and the expensive
    # canonicalization stage) once in the common case.
    # ponytail: if monotonicity ever fails, this (like the earlier tier prune) can settle for the rare
    # lower-symmetry loosest member where an all-members scan would have found a higher tighter one.
    winner, failures = _recognition_sweep(view, base, factors)

    # Some spglib paths depend on which atom is encountered first (notably a chiral cubic primitive
    # frame). If the primary canonical order finds only P1, retry the SAME canonical geometry with a
    # fixed reversed order. Retrying the caller's original frame here would reintroduce exactly the
    # shear/origin/order dependence that P1 preconditioning removes.
    if normalized_p1 is not None and (winner is None or winner.spacegroup.it_number == 1):
        alternate_view = UnitcellStructureView(_reversed_p1_frame(normalized_p1))
        alternate_winner, alternate_failures = _recognition_sweep(alternate_view, base, factors)
        failures.extend(f"reversed canonical frame {failure}" for failure in alternate_failures)
        if alternate_winner is not None and (
            winner is None
            or len(alternate_winner.spacegroup.symmetry_operations) > len(winner.spacegroup.symmetry_operations)
        ):
            winner = alternate_winner

    if winner is None:
        if normalized_p1 is None:
            winner = _p1_fallback(view, failures)
        else:
            logging.getLogger(__name__).warning(
                "no symmetry fit the structure within the requested tolerance sweep; using its exact P1 geometry",
                extra={"context": "symmetry", "attempts": tuple(failures)},
            )
            winner = normalized_p1

    if lift:
        return canonicalize(winner, tolerance=base, preserve_chirality=preserve_chirality).asu
    return _canonical_without_bfs(
        winner,
        preserve_chirality=preserve_chirality,
    )
