"""Default bond networks from interatomic distances.

:func:`distance_bonds` gives a structure that lacks a bond network a simple default one. It uses
the radius-sum form of OpenBabel and Jmol (their form, not their exact radii or floor): two atoms
are bonded when their distance lies between 0.5 Å and the sum of their covalent radii plus a
tolerance (0.45 Å by default). The radii are the single-bond covalent radii of Cordero et al.
[Cordero2008]_ (the low-spin values for Mn, Fe and Co, sp3 for C). On top of it, two metals are
not bonded in a structure that contains a non-metal (H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar,
Ge, As, Se, Br, Kr, Sb, Te, I, Xe, At, Rn: the metalloids count as non-metals), so the generous
radius sums of cations do not bond the cations of ionic solids to each other.

Finally, a metal does not bond to a non-metal across a non-metal ligand they share: such a bond
longer than the sum of the two radii is dropped when a non-metal bonded to both is more
electronegative than either (Pauling electronegativities as compiled by WebElements
[WebElementsEN]_, mostly Allred's revised values [Allred1961]_), so the large cations of
oxo-salts do not bond to the oxo-anion centres.

The network it returns names its rule as its ``source`` (software ``httk_basic``, the installed
*httk-atomistic* version), and by default it also assigns the baseline shapes;
:func:`assign_baseline_shapes` adds those to a network that has none.

:func:`distance_bonds` never recognises symmetry itself (an ``ASUStructureView`` input resolves
its own recognition): a unit cell gets a unit-cell network and an asymmetric unit a network of
bond orbits. On a measured cell whose metric keeps the group only
approximately the two routes can label one crystal differently: the unit-cell route assigns each
atom its own shape and keeps contacts at their measured, float-noisy distances, which recognition
may then find not invariant and drop, while the asymmetric-unit route
(``distance_bonds(ASUStructureView(cell, tolerance=...))``) idealises first.

This is a heuristic, not a chemical model: it ignores metal-metal bonds in compounds (Re-Re in
Re2Cl8(2-) salts, Hg2(2+)), judges anion-anion contacts by distance alone, and the radii fit
covalent bonds better than ionic ones.  It takes no account of occupancy, so on a disordered
structure the partially occupied atoms that overlap or sit a few tenths of an angstrom apart are
all bonded: per-site bond counts above about 60 come from disorder, not coordination (the
largest measured were 1522 on ICSD, a mixed O/Ti site of ICSD 162568, and 203 on carbon-free COD).

.. [Cordero2008] B. Cordero, V. Gómez, A. E. Platero-Prats, M. Revés, J. Echeverría,
   E. Cremades, F. Barragán, S. Alvarez, "Covalent radii revisited", *Dalton Trans.* 2008,
   2832-2838, doi:10.1039/b801115j, Table 2.
.. [Allred1961] A. L. Allred, "Electronegativity values from thermochemical data",
   *J. Inorg. Nucl. Chem.* 17 (1961) 215-221, doi:10.1016/0022-1902(61)80142-5.
.. [WebElementsEN] M. Winter, WebElements, "Pauling electronegativity",
   https://www.webelements.com/periodicity/eneg_pauling/ (values from Allred 1961, else
   L. Pauling, *The Nature of the Chemical Bond*, 3rd ed., 1960, and for the noble gases and
   Am-Cm J. E. Huheey, E. A. Keiter, R. L. Keiter, *Inorganic Chemistry*, 4th ed., 1993).
"""

import importlib.metadata
import itertools
import math
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from functools import cache

from httk.core import register_citation

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import (
    FundamentalDomainStructure,
    _affine,
    _apply,
    _group,
    _Op,
    _Point,
    _reference,
)
from httk.atomistic.models.structure.asu_view import ASUStructureView
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.network import BondNetwork, BondSource, SiteBond, _cell_pairs
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.symmetry.shapes import _site_shapes

__all__ = ["assign_baseline_shapes", "distance_bonds"]

_MIN_DISTANCE = 0.5  # Å: closer pairs are split or disordered positions, not bonds
_HTTK_BASIC = "httk_basic"
_IDENTITY: _Op = (tuple(Fraction(int(r == c)) for r in range(3) for c in range(3)), (Fraction(0),) * 3)

# Å, Cordero et al. 2008, Table 2: C sp3, Mn/Fe/Co low spin
_COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21,
    "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}  # fmt: skip

# Pauling electronegativities as compiled by WebElements (Allred 1961, else Pauling 1960, Huheey
# 1993); WebElements has no value for He, Ne, Ar, Rn, Pm, Eu, Tb and Yb
_ELECTRONEGATIVITIES = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82,
    "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16, "Tc": 1.9,
    "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05,
    "Te": 2.1, "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13,
    "Nd": 1.14, "Sm": 1.17, "Gd": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Lu": 1.27,
    "Hf": 1.3, "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.20, "Pt": 2.28, "Au": 2.54,
    "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0, "At": 2.2, "Fr": 0.7, "Ra": 0.9,
    "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.3, "Cm": 1.3,
}  # fmt: skip

# non-metals for the metal-metal rule, the metalloids included (so Zintl pairs such as Na-Si stay bonded)
_NON_METALS = frozenset(
    {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl"}
    | {"Ar", "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn"}
)

_BASELINE_DESCRIPTION = (
    "Baseline bond shapes: the ideal shape whose sorted bond angles fit best (IUPAC and SHAPE symbols), "
    "I (irregular) above 15 degrees RMS misfit, L2 from 165 degrees."
)


@cache
def _distribution_version() -> str | None:
    """Return the installed *httk-atomistic* version, ``None`` when it is not installed as a distribution."""
    try:
        return importlib.metadata.version("httk-atomistic")
    except importlib.metadata.PackageNotFoundError:
        return None


def _distance_description(tolerance: float) -> str:
    """Return the one-sentence description of the distance rule at ``tolerance``."""
    return (
        f"Distance bonds: 0.5 Å <= d <= r(a) + r(b) + {tolerance:g} Å (Cordero et al. 2008 covalent radii), "
        "no metal-metal bonds where a non-metal is present, and no metal to non-metal bond longer than "
        "r(a) + r(b) across a shared, more electronegative non-metal."
    )


def _presented(structure: StructureLike) -> FundamentalDomainStructure | UnitcellStructureView:
    """Return a structure as the kind it is: a fundamental domain, else its unit-cell presentation.

    Raw inputs (paths, streams, records, foreign objects) are adopted into a backend first. An
    :class:`ASUStructureView` resolves its own recognition, a fundamental-domain record gives its
    domain, and lazy backends (CIF blocks, datastreams) are resolved; nothing else is recognised.
    """
    if not isinstance(structure, (StructureBackend, StructureView)):
        structure = StructureBackend._select_backend(structure)
    if isinstance(structure, ASUStructureView):
        return structure.resolve()
    if isinstance(structure, RecordStructure) and not structure._is_unitcell:
        return structure._native
    resolve = getattr(structure, "resolve", None) if isinstance(structure, StructureBackend) else None
    resolved: StructureLike = structure if resolve is None else resolve()
    if isinstance(resolved, FundamentalDomainStructure):
        return resolved
    return UnitcellStructureView(resolved)


def _site_radii(
    species: Sequence[Species], names: Sequence[str]
) -> tuple[list[float | None], list[bool | None], list[float | None], list[float | None]]:
    """Return per site its radius, whether it is a metal, and its electronegativity as bond end and as ligand.

    The constituents are the symbols with a positive concentration other than ``vacancy``
    (attached atoms are ignored); the radius is the largest of theirs, and the site is a metal
    when none of them is a non-metal. A site's electronegativity is the largest of its
    constituents' as a bond end and the smallest as a shared ligand. A metal end ignores
    constituents without a value, and is ``-inf`` (below every value) when none has one; any
    other end or ligand is ``None`` when a constituent has no value, and a ligand also unless
    every constituent is a non-metal. All are ``None`` for a site without an element.
    """
    constituents = {
        item.name: [
            symbol
            for symbol, amount in zip(item.chemical_symbols, item.concentration, strict=True)
            if amount > 0 and symbol != "vacancy"
        ]
        for item in species
    }
    radii: list[float | None] = []
    metals: list[bool | None] = []
    ends: list[float | None] = []
    ligands: list[float | None] = []
    for number, name in enumerate(names):
        elements = constituents[name]
        missing = [symbol for symbol in elements if symbol not in _COVALENT_RADII]
        if missing:
            raise ValueError(f"no covalent radius for {', '.join(map(repr, missing))} (site {number})")
        radii.append(max((_COVALENT_RADII[symbol] for symbol in elements), default=None))
        metals.append(not _NON_METALS.intersection(elements) if elements else None)
        values = [_ELECTRONEGATIVITIES.get(symbol) for symbol in elements]
        known = [value for value in values if value is not None]
        complete = bool(elements) and len(known) == len(values)
        if elements and not _NON_METALS.intersection(elements):  # a metal end: valueless constituents rank lowest
            ends.append(max(known, default=-math.inf))
        else:
            ends.append(max(known) if complete else None)
        ligands.append(min(known) if complete and _NON_METALS.issuperset(elements) else None)
    return radii, metals, ends, ligands


def distance_bonds(
    structure: StructureLike, *, tolerance: float = 0.45, shapes: bool = True
) -> FundamentalDomainStructure | UnitcellStructure:
    """Return a structure with the bond network found from interatomic distances, of the kind it was given.

    The result keeps the input's sites and replaces its ``bonds``. It never recognises symmetry
    itself; an :class:`~httk.atomistic.ASUStructureView` input resolves its own recognition. A
    :class:`~httk.atomistic.FundamentalDomainStructure` or :class:`~httk.atomistic.ASUStructure`
    (an :class:`~httk.atomistic.ASUStructureView` resolved) gives the same class with a network
    of bond orbits. A unit cell or unit-cell view gives
    ``UnitcellStructureView(structure).unview()`` with explicit pairs between its sites. Raw
    inputs (a path, a text stream, a storage record) are first adopted as structures; lazy
    sources (a CIF block or file, a datastream) are resolved and a fundamental-domain record gives
    its domain, and then these two rules apply. Any other structure gives
    ``UnitcellStructureView(structure).unview()``, so a
    :class:`~httk.atomistic.NumericUnitcellStructure` gives a
    :class:`~httk.atomistic.UnitcellStructure`. For the network of the recognised asymmetric
    unit, recognise first: ``distance_bonds(ASUStructureView(structure, tolerance=...))``. On a
    measured cell whose metric keeps the group only approximately the two routes can label one
    crystal differently: the unit-cell route assigns each atom its own shape and keeps contacts
    at their measured distances, the asymmetric-unit route idealises first.

    Two atoms *a* and *b* are bonded iff ``0.5 <= d(a, b) <= r(a) + r(b) + tolerance`` (Å), with
    *r* the covalent radii of Cordero et al. (2008). A mixed-occupancy site takes the largest
    radius of its constituents (positive concentration, vacancies and attached atoms ignored); a
    site with no constituent (only vacancies) gets no bonds. Pairs closer than 0.5 Å (split or
    disordered positions) are never bonded, so coincident sites are never bonded to each other.
    In a structure with at least one non-metal (H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Ge,
    As, Se, Br, Kr, Sb, Te, I, Xe, At, Rn), two metal sites are never bonded; a mixed site is a
    metal when all its constituents are, so even a dilute non-metal on a mixed site switches this
    rule on for the whole structure. Without a non-metal (intermetallics, elements) the distance
    rule alone applies. Along a non-periodic direction of a unit cell, atoms are not repeated.

    The bonds so found form the tentative network. A metal does not bond to a non-metal across a
    non-metal ligand they share: a tentative bond between a metal *a* and a non-metal *b* is
    dropped iff ``d(a, b) > r(a) + r(b)`` (it exists only through the tolerance) and some non-metal
    *c* tentatively bonded to both is more electronegative than both. All drops are decided on the
    tentative network in one pass. Metal-metal and non-metal-non-metal bonds are never dropped.
    Electronegativities are Pauling's as compiled by WebElements (Allred 1961, else Pauling 1960
    or Huheey et al. 1993). A mixed site is a metal end when all its constituents are metals and a
    non-metal ligand when all are non-metals; it counts with its constituents' largest value as
    *a* or *b* and their smallest as *c*, so a bond is dropped only when the condition holds for
    every constituent with a value. A metal end without a value (Pm, Eu, Tb, Yb) counts as less
    electronegative than any non-metal with one: a mixed metal end ignores its constituents
    without a value when taking the largest, and counts so when none has one. Any other site with
    a constituent without a value (He, Ne, Ar, Rn, or one of those metals beside a non-metal on a
    mixed site) never acts as *b* or *c*. No values are invented. This parts the large cations of
    oxo-salts from the anion centres (in diopside CaMgSi2O6, Ca from Si: Ca 8 and Si 4 bonds, not
    12 and 6).

    A heuristic: it misses metal-metal bonds in compounds (Re-Re in Re2Cl8(2-) salts, Hg2(2+)) and
    judges anion-anion contacts by distance only. The shared-ligand rule cannot part a cation from
    an anion centre closer than their radius sum (Cs, Rb and Ba radii are large) or sharing no
    more electronegative non-metal; and it drops real metal-ligand bonds longer than the radius
    sum that share a cage or ring atom (C, N or S) more electronegative than both, as in
    metallacarboranes (Fe-B 2.23-2.38 Å across cage C), thiaboranes (Ir-B across S) and
    eta-bonded heteroarenes (Fe-C across N).

    Distances are floats computed from the exact structure: the network is the same for every
    exact re-description of one crystal, except that a pair within float noise of its cutoff or
    of its radius sum may fall on either side.

    The network's ``source`` names the rule: software ``httk_basic``, the installed
    *httk-atomistic* version and a one-sentence description with ``tolerance``. With ``shapes``
    (the default) every site of two or more bonds also gets its baseline shape (see
    :func:`assign_baseline_shapes`), with ``shapes_source`` ``None`` (the same source).

    Example::

        >>> from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite, distance_bonds
        >>> nacl = ASUStructure(
        ...     Cell([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]), 225,
        ...     [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
        ...     [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
        ... )
        >>> bonded = distance_bonds(nacl)  # one bond orbit: Na-Cl at 2.82 Å
        >>> len(bonded.bonds.pairs), bonded.bonds.shapes
        (1, ('OC6', 'OC6'))

    :param structure: Any structure.
    :param tolerance: Å added to the sum of the radii.
    :param shapes: Whether to assign the baseline shapes as well.
    :return: The structure, of the kind described above, with its distance-based network.
    :raises ValueError: If a site holds an element without a radius (named in the message),
        ``tolerance`` is not a finite number, or (with ``shapes``) the baseline cannot assign a
        shape, as for a fundamental-domain site at a special point of its own Wyckoff letter; that
        message says to pass ``shapes=False``.
    """
    return _distance_bonds(structure, tolerance, True, shapes)


def _distance_bonds(
    structure: StructureLike, tolerance: float, shared_ligand: bool, shapes: bool = False
) -> FundamentalDomainStructure | UnitcellStructure:
    """Return :func:`distance_bonds`, or with ``shared_ligand`` false its tentative network (no source)."""
    if isinstance(tolerance, bool) or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite number, got {tolerance!r}")
    target = _presented(structure)
    basis = target.cell.basis.to_floats()
    own = [[math.fsum(basis[i][k] * basis[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    if isinstance(target, FundamentalDomainStructure):
        spacegroup = target.spacegroup
        references = [_reference(spacegroup, site) for site in target.wyckoff_sites]
        names = [site.species for site in target.wyckoff_sites]
        group = _group(spacegroup)[0]
        # the Cartesian metric of the stored coordinates (own = M stored + v, cell rows in Å)
        M = target.transform.matrix.to_floats()
        G = [
            [math.fsum(M[a][i] * own[a][b] * M[b][j] for a in range(3) for b in range(3)) for j in range(3)]
            for i in range(3)
        ]
    else:  # the stored coordinates, unwrapped, and no symmetry
        references = [tuple(Fraction(x) for x in row) for row in target.sites.reduced_coords.to_fractions()]
        names, group, G = list(target.species_at_sites), (_IDENTITY,), own
    radii, metals, ends, ligands = _site_radii(target.species, names)
    found = _found_bonds(references, group, G, target.cell.periodicity, radii, metals, tolerance)
    if not (shared_ligand and True in metals and False in metals):  # no metal to non-metal bond to drop
        bonds = [SiteBond(site, partner, _affine(op)) for site, partner, op, _distance in found]
    else:
        bonds = _without_shared_ligand_bonds(found, references, radii, metals, ends, ligands)
    _register_citations(shared_ligand)
    source = (
        BondSource(_distance_description(tolerance), _HTTK_BASIC, _distribution_version()) if shared_ligand else None
    )
    if isinstance(target, FundamentalDomainStructure):  # a domain normalizes its pairs on first access
        network = BondNetwork(tuple(bonds), source=source)
    else:
        network = BondNetwork(_cell_pairs(bonds, len(references), target.cell.periodicity), source=source)
    result = target._with_bonds(network)
    if not shapes:
        return result
    try:
        assigned = _site_shapes(result)
    except ValueError as error:
        raise ValueError(f"{error}; pass shapes=False to leave the shapes unassigned") from error
    return result._with_bonds(replace(network, shapes=assigned))


def _found_bonds(
    references: Sequence[_Point],
    group: Sequence[_Op],
    G: Sequence[Sequence[float]],
    periodicity: Sequence[bool],
    radii: Sequence[float | None],
    metals: Sequence[bool | None],
    tolerance: float,
) -> list[tuple[int, int, _Op, float]]:
    """Return the tentative network: ``(site, partner, operation, distance)`` per bond found from a reference atom.

    :param references: Each site's reference atom, exact, in the coordinates the operations act on.
    :param group: The group operations of those coordinates (the identity alone for a unit cell).
    :param G: The float Cartesian metric of those coordinates.
    :param periodicity: The periodic axes; along the others coordinates are not wrapped and no
        translation is tried.
    :param radii: Per site its radius, ``None`` for no bonds.
    :param metals: Per site whether it is a metal.
    :param tolerance: Å added to the sum of the radii.
    :return: Each bond from a site's reference atom to ``operation`` applied to the partner's.
    """
    has_non_metal = False in metals  # then metal-metal pairs are not bonded
    Ginv = _inverse([list(row) for row in G])
    # Completeness: a displacement x (stored coordinates) of Cartesian length at most R has
    # |x_k| = |<G^-1 e_k, x>_G| <= |G^-1 e_k|_G R = sqrt((G^-1)_kk) R by Cauchy-Schwarz, so the
    # lattice translations n with |q + n - p|_G <= R lie in p_k - q_k - R s_k <= n_k <= p_k - q_k + R s_k,
    # s_k = sqrt((G^-1)_kk) (widened by 1e-9 against float rounding): every such n is visited.
    reach = [math.sqrt(Ginv[k][k]) * (1 + 1e-9) for k in range(3)]
    # every atom of the cell (periodic coordinates modulo 1) with the group element taking its site's reference atom there
    atoms: list[tuple[int, _Op, tuple[float, ...]]] = []
    for number, reference in enumerate(references):
        if radii[number] is None:
            continue
        seen = set()
        for W, w in group:
            image = _apply((W, w), reference)
            wrapped = tuple(x % 1 if periodic else x for x, periodic in zip(image, periodicity, strict=True))
            if wrapped not in seen:
                seen.add(wrapped)
                shift = tuple(t + a - b for t, a, b in zip(w, wrapped, image, strict=True))
                atoms.append((number, (W, shift), tuple(map(float, wrapped))))
    largest = max((r for r in radii if r is not None), default=0.0)
    R = 2 * largest + tolerance
    found: list[tuple[int, int, _Op, float]] = []
    if R < _MIN_DISTANCE:
        return found
    # Bins per axis, of width at least R s_k (with a 1e-6 margin for the float floor): on a periodic
    # axis g_k bins of width 1/g_k, numbered modulo g_k; on a non-periodic axis unbounded bins of width
    # R s_k. Two atoms within R of each other have |dx_k| <= R s_k, so floor(x_k / width) differs by
    # at most one and their bins are equal or adjacent (modulo g_k on a periodic axis).
    widths = [R * s * (1 + 1e-6) for s in reach]
    moduli = [max(1, int(1 / w)) if periodic else None for w, periodic in zip(widths, periodicity, strict=True)]

    def binned(x: float, k: int, d: int = 0) -> int:
        m = moduli[k]
        return (math.floor(x * m) + d) % m if m is not None else math.floor(x / widths[k]) + d

    bins: dict[tuple[int, ...], list[int]] = {}
    for index, (_number, _element, q) in enumerate(atoms):
        bins.setdefault(tuple(binned(x, k) for k, x in enumerate(q)), []).append(index)
    for site, reference in enumerate(references):
        r_site = radii[site]
        if r_site is None:
            continue
        p = tuple(map(float, reference))
        near = [{binned(x, k, d) for d in (-1, 0, 1)} for k, x in enumerate(p)]
        for key in itertools.product(*near):
            for index in bins.get(key, ()):
                partner, (W, w), q = atoms[index]
                r_partner = radii[partner]
                assert r_partner is not None
                cutoff = r_site + r_partner + tolerance
                if cutoff < _MIN_DISTANCE or (has_non_metal and metals[site] and metals[partner]):
                    continue
                delta = [a - b for a, b in zip(q, p, strict=True)]
                ranges = [
                    range(math.ceil(-d - cutoff * s), math.floor(-d + cutoff * s) + 1) if periodic else (0,)
                    for d, s, periodic in zip(delta, reach, periodicity, strict=True)
                ]
                for n in itertools.product(*ranges):
                    x = [d + t for d, t in zip(delta, n, strict=True)]
                    distance = math.sqrt(max(0.0, sum(x[a] * G[a][b] * x[b] for a in range(3) for b in range(3))))
                    if _MIN_DISTANCE <= distance <= cutoff:
                        found.append((site, partner, (W, tuple(t + k for t, k in zip(w, n, strict=True))), distance))
    return found


def assign_baseline_shapes(structure: StructureLike) -> FundamentalDomainStructure | UnitcellStructure:
    """Return a structure whose bond network has no shapes with its baseline shapes assigned.

    The structure is taken as :func:`distance_bonds` takes it (a fundamental domain stays one,
    anything else is presented as a unit cell, and symmetry is never recognised by the function itself). Every site of two or more
    bonds gets the shape whose ideal sorted bond angles fit its own best (IUPAC and SHAPE
    symbols, ``I`` (irregular) above 15 degrees RMS misfit, ``L2`` from 165 degrees; see
    :mod:`httk.atomistic.symmetry.shapes`), and ``shapes_source`` names the baseline (software
    ``httk_basic``, the installed *httk-atomistic* version); the pairs and ``source`` are kept.
    When no site gets a shape (no site has two bonds) the structure is returned with its network
    unchanged.

    :param structure: A structure with a bond network and no shapes.
    :return: The structure, of the kind :func:`distance_bonds` returns, with the shapes assigned.
    :raises ValueError: If the structure has no network, its network already has shapes, or the
        baseline cannot assign a shape (a fundamental-domain site at a special point of its own
        Wyckoff letter).
    """
    target = _presented(structure)
    result = target.unview() if isinstance(target, UnitcellStructureView) else target
    network = result.bonds
    if network is None:
        raise ValueError("a structure without a bond network has no shapes to assign")
    if network.shapes is not None:
        raise ValueError("the bond network already has shapes; baseline shapes are assigned only to a network without")
    assigned = _site_shapes(result)
    if all(symbol is None for symbol in assigned):
        return result
    source = BondSource(_BASELINE_DESCRIPTION, _HTTK_BASIC, _distribution_version())
    return result._with_bonds(replace(network, shapes=assigned, shapes_source=source))


def _register_citations(shared_ligand: bool) -> None:
    """Register the radii, and with ``shared_ligand`` the electronegativities, as used."""
    register_citation(
        applies_to="Default bond networks use the covalent radii of Cordero et al.",
        references={
            "authors": tuple(
                {"name": name}
                for name in (
                    "Beatriz Cordero",
                    "Verónica Gómez",
                    "Ana E. Platero-Prats",
                    "Marc Revés",
                    "Jorge Echeverría",
                    "Eduard Cremades",
                    "Flavia Barragán",
                    "Santiago Alvarez",
                )
            ),
            "title": "Covalent radii revisited",
            "journal": "Dalton Transactions",
            "pages": "2832-2838",
            "year": "2008",
            "doi": "10.1039/b801115j",
            "bib_type": "article",
        },
    )
    if shared_ligand:
        register_citation(
            applies_to="Default bond networks use Pauling electronegativities, mostly the revised values of Allred",
            references={
                "authors": ({"name": "A. L. Allred"},),
                "title": "Electronegativity values from thermochemical data",
                "journal": "Journal of Inorganic and Nuclear Chemistry",
                "volume": "17",
                "pages": "215-221",
                "year": "1961",
                "doi": "10.1016/0022-1902(61)80142-5",
                "bib_type": "article",
            },
        )


def _without_shared_ligand_bonds(
    found: list[tuple[int, int, _Op, float]],
    references: list[_Point],
    radii: list[float | None],
    metals: list[bool | None],
    ends: list[float | None],
    ligands: list[float | None],
) -> list[SiteBond]:
    """Return the tentative bonds less those of a metal and a non-metal across a shared ligand.

    A bond is found from its first site's reference atom a to b = operation(reference of
    partner). The atoms bonded to b are the operation applied to those bonded to the partner's
    reference atom, since the operation is a symmetry of the network, so the common neighbours of
    a and b are compared exactly. A group element h maps the network onto itself, so the pair
    (h a, h b) has the images of these common neighbours (same sites, same electronegativities and
    metal classes) and the same distance: the decision is the same for every member of a bond
    orbit, and the conditions are symmetric in a and b, so the result stays a union of orbits. All
    decisions are made on the tentative network, in one pass.
    """
    neighbours: list[set[tuple[int, _Point]]] = [set() for _ in references]
    for site, partner, op, _distance in found:
        neighbours[site].add((partner, _apply(op, references[partner])))

    def dropped(site: int, partner: int, op: _Op, distance: float) -> bool:
        r_site, r_partner, end, other = radii[site], radii[partner], ends[site], ends[partner]
        assert r_site is not None and r_partner is not None
        if metals[site] == metals[partner] or distance <= r_site + r_partner or end is None or other is None:
            return False
        limit = max(end, other)
        return any(
            (value := ligands[ligand]) is not None and value > limit and (ligand, _apply(op, y)) in neighbours[site]
            for ligand, y in neighbours[partner]
        )

    return [
        SiteBond(site, partner, _affine(op))
        for site, partner, op, distance in found
        if not dropped(site, partner, op, distance)
    ]


def _inverse(G: list[list[float]]) -> list[list[float]]:
    """Return the inverse of a 3x3 matrix (the adjugate over the determinant)."""
    (a, b, c), (d, e, f), (g, h, i) = G
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    adjugate = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    return [[value / det for value in row] for row in adjugate]
