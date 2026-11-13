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

This is a heuristic, not a chemical model: it ignores metal-metal bonds in compounds (Re-Re in
Re2Cl8(2-) salts, Hg2(2+)), judges anion-anion contacts by distance alone, and the radii fit
covalent bonds better than ionic ones.

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

import itertools
import math

from httk.core import register_citation

from httk.atomistic.models.structure.api import _without_network
from httk.atomistic.models.structure.asu import (
    ASUStructure,
    _affine,
    _apply,
    _group,
    _Op,
    _Point,
    _reference,
)
from httk.atomistic.models.structure.asu_view import ASUStructureView
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.network import BondNetwork, SiteBond

__all__ = ["distance_bonds"]

_MIN_DISTANCE = 0.5  # Å: closer pairs are split or disordered positions, not bonds

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


def _site_radii(
    asu: ASUStructure,
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
        species.name: [
            symbol
            for symbol, amount in zip(species.chemical_symbols, species.concentration, strict=True)
            if amount > 0 and symbol != "vacancy"
        ]
        for species in asu.species
    }
    radii: list[float | None] = []
    metals: list[bool | None] = []
    ends: list[float | None] = []
    ligands: list[float | None] = []
    for number, site in enumerate(asu.wyckoff_sites):
        elements = constituents[site.species]
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


def distance_bonds(structure: StructureLike, *, tolerance: float = 0.45) -> ASUStructure:
    """Return a structure's asymmetric unit with the bond network found from interatomic distances.

    The result is ``ASUStructureView(structure).resolve()`` with ``bonds`` replaced by the network
    found and no supplied shapes. For an asymmetric-unit input that is the input itself, sites
    unchanged; any other input (a unit cell, a P1, primitive or supercell description) is first
    recognised, and the recognised structure is idealised within the recognition tolerance, so a
    contact within that tolerance of its cutoff can fall on either side. For control, recognise
    first: ``distance_bonds(ASUStructureView(structure, tolerance=...))``.

    Two atoms *a* and *b* are bonded iff ``0.5 <= d(a, b) <= r(a) + r(b) + tolerance`` (Å), with
    *r* the covalent radii of Cordero et al. (2008). A mixed-occupancy site takes the largest
    radius of its constituents (positive concentration, vacancies and attached atoms ignored); a
    site with no constituent (only vacancies) gets no bonds. Pairs closer than 0.5 Å (split or
    disordered positions) are never bonded. In a structure with at least one non-metal (H, He, B,
    C, N, O, F, Ne, Si, P, S, Cl, Ar, Ge, As, Se, Br, Kr, Sb, Te, I, Xe, At, Rn), two metal sites
    are never bonded; a mixed site is a metal when all its constituents are, so even a dilute
    non-metal on a mixed site switches this rule on for the whole structure. Without a non-metal
    (intermetallics, elements) the distance rule alone applies.

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
    exact asymmetric-unit re-description of one crystal, except that a pair within float noise of
    its cutoff or of its radius sum may fall on either side.

    Example::

        >>> from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite, distance_bonds
        >>> nacl = ASUStructure(
        ...     Cell([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]), 225,
        ...     [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
        ...     [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
        ... )
        >>> len(distance_bonds(nacl).bonds.pairs)  # one bond orbit: Na-Cl at 2.82 Å
        1

    :param structure: Any structure an :class:`~httk.atomistic.ASUStructureView` accepts.
    :param tolerance: Å added to the sum of the radii.
    :return: The asymmetric unit with its distance-based network.
    :raises ValueError: If a site holds an element without a radius (named in the message), or
        ``tolerance`` is not a finite number.
    """
    return _distance_bonds(structure, tolerance, True)


def _distance_bonds(structure: StructureLike, tolerance: float, shared_ligand: bool) -> ASUStructure:
    """Return :func:`distance_bonds`, or with ``shared_ligand`` false its tentative network."""
    if isinstance(tolerance, bool) or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite number, got {tolerance!r}")
    asu = ASUStructureView(_without_network(structure)).resolve()  # its network is replaced
    radii, metals, ends, ligands = _site_radii(asu)
    has_non_metal = False in metals  # then metal-metal pairs are not bonded
    spacegroup, sites = asu.spacegroup, asu.wyckoff_sites
    # the Cartesian metric of the stored coordinates (own = M stored + v, cell rows in Å)
    basis = asu.cell.basis.to_floats()
    M = asu.transform.matrix.to_floats()
    own = [[math.fsum(basis[i][k] * basis[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    G = [
        [math.fsum(M[a][i] * own[a][b] * M[b][j] for a in range(3) for b in range(3)) for j in range(3)]
        for i in range(3)
    ]
    Ginv = _inverse(G)
    # Completeness: a displacement x (stored coordinates) of Cartesian length at most R has
    # |x_k| = |<G^-1 e_k, x>_G| <= |G^-1 e_k|_G R = sqrt((G^-1)_kk) R by Cauchy-Schwarz, so the
    # lattice translations n with |q + n - p|_G <= R lie in p_k - q_k - R s_k <= n_k <= p_k - q_k + R s_k,
    # s_k = sqrt((G^-1)_kk) (widened by 1e-9 against float rounding): every such n is visited.
    reach = [math.sqrt(Ginv[k][k]) * (1 + 1e-9) for k in range(3)]
    references = [_reference(spacegroup, site) for site in sites]
    # every atom of the cell (stored coordinates modulo 1) with the group element taking its site's reference atom there
    atoms: list[tuple[int, _Op, tuple[float, ...]]] = []
    for number, reference in enumerate(references):
        if radii[number] is None:
            continue
        seen = set()
        for W, w in _group(spacegroup)[0]:
            image = _apply((W, w), reference)
            wrapped = tuple(x % 1 for x in image)
            if wrapped not in seen:
                seen.add(wrapped)
                shift = tuple(t + a - b for t, a, b in zip(w, wrapped, image, strict=True))
                atoms.append((number, (W, shift), tuple(map(float, wrapped))))
    largest = max((r for r in radii if r is not None), default=0.0)
    R = 2 * largest + tolerance
    found: list[tuple[int, int, _Op, float]] = []  # the tentative network: site, partner, operation, distance
    if R >= _MIN_DISTANCE:
        # Bins of width 1/g_k >= R s_k (with a 1e-6 margin for the float floor) per axis: an atom
        # within R of a reference atom (any translate) lies in its bin or a neighbouring one, mod g_k.
        g = [max(1, int(1 / (R * s * (1 + 1e-6)))) for s in reach]
        bins: dict[tuple[int, ...], list[int]] = {}
        for index, (_number, _element, q) in enumerate(atoms):
            bins.setdefault(tuple(int(x * n) % n for x, n in zip(q, g, strict=True)), []).append(index)
        for site, reference in enumerate(references):
            r_site = radii[site]
            if r_site is None:
                continue
            p = tuple(map(float, reference))
            near = [{(int(x % 1 * n) + d) % n for d in (-1, 0, 1)} for x, n in zip(p, g, strict=True)]
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
                        range(math.ceil(-d - cutoff * s), math.floor(-d + cutoff * s) + 1)
                        for d, s in zip(delta, reach, strict=True)
                    ]
                    for n in itertools.product(*ranges):
                        x = [d + t for d, t in zip(delta, n, strict=True)]
                        distance = math.sqrt(max(0.0, sum(x[a] * G[a][b] * x[b] for a in range(3) for b in range(3))))
                        if _MIN_DISTANCE <= distance <= cutoff:
                            found.append(
                                (site, partner, (W, tuple(t + k for t, k in zip(w, n, strict=True))), distance)
                            )
    if not (shared_ligand and True in metals and False in metals):  # no metal to non-metal bond to drop
        bonds = [SiteBond(site, partner, _affine(op)) for site, partner, op, _distance in found]
    else:
        bonds = _without_shared_ligand_bonds(found, references, radii, metals, ends, ligands)
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
    return asu._with_bonds(BondNetwork(tuple(bonds)))  # normalized on first access


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
