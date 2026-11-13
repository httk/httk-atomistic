"""Default bond networks from interatomic distances.

:func:`distance_bonds` gives a structure that lacks a bond network a simple default one. It uses
the radius-sum form of OpenBabel and Jmol (their form, not their exact radii or floor): two atoms
are bonded when their distance lies between 0.5 Å and the sum of their covalent radii plus a
tolerance (0.45 Å by default). The radii are the single-bond covalent radii of Cordero et al.
[Cordero2008]_ (the low-spin values for Mn, Fe and Co, sp3 for C). On top of it, two metals are
not bonded in a structure that contains a non-metal (H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar,
Ge, As, Se, Br, Kr, Sb, Te, I, Xe, At, Rn: the metalloids count as non-metals), so the generous
radius sums of cations do not bond the cations of ionic solids to each other.

This is a heuristic, not a chemical model: it ignores metal-metal bonds in compounds (Re-Re in
Re2Cl8(2-) salts, Hg2(2+)), judges anion-anion contacts by distance alone, and the radii fit
covalent bonds better than ionic ones.

.. [Cordero2008] B. Cordero, V. Gómez, A. E. Platero-Prats, M. Revés, J. Echeverría,
   E. Cremades, F. Barragán, S. Alvarez, "Covalent radii revisited", *Dalton Trans.* 2008,
   2832-2838, doi:10.1039/b801115j, Table 2.
"""

import itertools
import math

from httk.core import register_citation

from httk.atomistic.models.structure.asu import (
    ASUStructure,
    SiteBond,
    _affine,
    _apply,
    _group,
    _Op,
    _reference,
)
from httk.atomistic.models.structure.asu_view import ASUStructureView
from httk.atomistic.models.structure.like import StructureLike

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

# non-metals for the metal-metal rule, the metalloids included (so Zintl pairs such as Na-Si stay bonded)
_NON_METALS = frozenset(
    {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl"}
    | {"Ar", "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn"}
)


def _site_radii(asu: ASUStructure) -> tuple[list[float | None], list[bool | None]]:
    """Return per site its radius and whether it is a metal, ``None`` for a site without an element.

    The constituents are the symbols with a positive concentration other than ``vacancy``
    (attached atoms are ignored); the radius is the largest of theirs, and the site is a metal
    when none of them is a non-metal.
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
    for number, site in enumerate(asu.wyckoff_sites):
        elements = constituents[site.species]
        missing = [symbol for symbol in elements if symbol not in _COVALENT_RADII]
        if missing:
            raise ValueError(f"no covalent radius for {', '.join(map(repr, missing))} (site {number})")
        radii.append(max((_COVALENT_RADII[symbol] for symbol in elements), default=None))
        metals.append(not _NON_METALS.intersection(elements) if elements else None)
    return radii, metals


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
    (intermetallics, elements) the distance rule alone applies. A heuristic: it misses
    metal-metal bonds in compounds (Re-Re in Re2Cl8(2-) salts, Hg2(2+)) and judges anion-anion
    contacts by distance only.

    Distances are floats computed from the exact structure: the network is the same for every
    exact asymmetric-unit re-description of one crystal, except that a pair within float noise of its cutoff may
    fall on either side.

    Example::

        >>> from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite, distance_bonds
        >>> nacl = ASUStructure(
        ...     Cell([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]), 225,
        ...     [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
        ...     [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
        ... )
        >>> len(distance_bonds(nacl).bonds)  # one bond orbit: Na-Cl at 2.82 Å
        1

    :param structure: Any structure an :class:`~httk.atomistic.ASUStructureView` accepts.
    :param tolerance: Å added to the sum of the radii.
    :return: The asymmetric unit with its distance-based network.
    :raises ValueError: If a site holds an element without a radius (named in the message), or
        ``tolerance`` is not a finite number.
    """
    if isinstance(tolerance, bool) or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite number, got {tolerance!r}")
    asu = ASUStructureView(structure).resolve()
    radii, metals = _site_radii(asu)
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
    bonds = []
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
                            operation = _affine((W, tuple(t + k for t, k in zip(w, n, strict=True))))
                            bonds.append(SiteBond(site, partner, operation))
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
    return asu._with_bonds(bonds, shapes=None)  # normalized on first access


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
