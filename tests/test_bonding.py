"""Default bond networks from distances (``distance_bonds``)."""

import itertools
import math
from collections import Counter
from fractions import Fraction as F
from typing import Any

import pytest
from httk.core import FracVector
from test_protonet_molecular import MOLECULAR, _described

from httk.atomistic import (
    AffineOperation,
    ASUStructure,
    SettingTransform,
    SiteBond,
    Spacegroup,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    canonical_protonetstructure,
    distance_bonds,
)
from httk.atomistic.bonding import _COVALENT_RADII, _NON_METALS
from httk.atomistic.models.structure.asu import _apply, _op, _reference
from httk.atomistic.symmetry.protonet import _tables
from httk.atomistic.symmetry.shapes import _structure_bond_counts

NA = Species("Na", ("Na",), (1,))
CL = Species("Cl", ("Cl",), (1,))


def _counts(structure: ASUStructure) -> tuple[int, ...]:
    assert structure.bonds is not None
    return _structure_bond_counts(structure.spacegroup, structure.wyckoff_sites, structure.bonds)


def _pairs(structure: ASUStructure) -> Counter[tuple[str, ...]]:
    """Bond orbits by their (sorted) site species."""
    assert structure.bonds is not None
    names = [site.species for site in structure.wyckoff_sites]
    return Counter(tuple(sorted((names[b.site], names[b.partner]))) for b in structure.bonds)


def _rocksalt(a: object, cation: Species = NA) -> ASUStructure:
    return ASUStructure(
        [[a, 0, 0], [0, a, 0], [0, 0, a]],
        225,
        [WyckoffSite("a", (), cation.name), WyckoffSite("b", (), "Cl")],
        [cation, CL],
    )


def _diamond(setting: str) -> ASUStructure:
    basis = [[F(357, 100), 0, 0], [0, F(357, 100), 0], [0, 0, F(357, 100)]]
    return ASUStructure(
        basis, Spacegroup.from_setting(setting), [WyckoffSite("a", (), "C")], [Species("C", ("C",), (1,))]
    )


def test_rocksalt_bonds_each_ion_to_six_counterions_only() -> None:
    nacl = distance_bonds(_rocksalt(F(564, 100)))  # Na-Cl 2.82, Na-Na and Cl-Cl 3.99 Å
    assert _pairs(nacl) == Counter({("Cl", "Na"): 1})
    assert _counts(nacl) == (6, 6)
    assert nacl.shapes is None
    assert nacl == _rocksalt(F(564, 100))._with_bonds(nacl.bonds, shapes=None)


def test_diamond_bonds_each_carbon_to_four() -> None:
    diamond = distance_bonds(_diamond("227:2"))
    assert len(diamond.bonds or ()) == 1
    assert _counts(diamond) == (4,)


def test_rutile_bonds_titanium_to_six_oxygens() -> None:
    rutile = ASUStructure(
        [[F(4594, 1000), 0, 0], [0, F(4594, 1000), 0], [0, 0, F(2959, 1000)]],
        136,
        [WyckoffSite("a", (), "Ti"), WyckoffSite("f", (F(3048, 10000),), "O")],
        [Species("Ti", ("Ti",), (1,)), Species("O", ("O",), (1,))],
    )
    # Ti-O 1.95 and 1.98 Å (cutoff 2.71); Ti-Ti at 2.96 and 3.57 Å lies within its 3.65 Å cutoff
    # but is a metal-metal pair in a structure with a non-metal
    found = distance_bonds(rutile)
    assert _pairs(found) == Counter({("O", "Ti"): 2})
    assert _counts(found) == (6, 3)
    assert _counts(distance_bonds(rutile, tolerance=-0.3)) == (4, 2)  # Ti-O cutoff 1.96 Å: the short pair only


def test_metal_metal_pairs_are_not_bonded_beside_a_non_metal() -> None:
    nacl = distance_bonds(_rocksalt(F(564, 100)), tolerance=10.0)
    assert ("Na", "Na") not in _pairs(nacl)
    assert ("Cl", "Cl") in _pairs(nacl)  # anion-anion contacts go by distance alone


def test_an_intermetallic_bonds_by_distance() -> None:
    cu3au = ASUStructure(
        [[F(375, 100), 0, 0], [0, F(375, 100), 0], [0, 0, F(375, 100)]],
        221,
        [WyckoffSite("a", (), "Au"), WyckoffSite("c", (), "Cu")],
        [Species("Au", ("Au",), (1,)), Species("Cu", ("Cu",), (1,))],
    )
    found = distance_bonds(cu3au)  # nearest neighbours 2.65 Å, the next 3.75 Å
    assert _pairs(found) == Counter({("Au", "Cu"): 1, ("Cu", "Cu"): 1})
    assert _counts(found) == (12, 12)


def test_unitcell_and_asu_descriptions_give_one_network() -> None:
    nacl = _rocksalt(F(564, 100))
    from_cell = distance_bonds(UnitcellStructureView(nacl))
    assert canonical_protonetstructure(from_cell) == canonical_protonetstructure(distance_bonds(nacl))


def test_a_setting_change_gives_the_same_network() -> None:
    first, second = distance_bonds(_diamond("227:1")), distance_bonds(_diamond("227:2"))
    assert first.spacegroup != second.spacegroup
    assert canonical_protonetstructure(first) == canonical_protonetstructure(second)
    # a non-standard monoclinic setting
    p21n = Spacegroup.from_hm_entry("P 1 21/n 1")
    sites = [WyckoffSite("e", (F(1, 8), F(2, 7), F(3, 11)), "Si"), WyckoffSite("a", (), "O")]
    species = [Species("Si", ("Si",), (1,)), Species("O", ("O",), (1,))]
    source = ASUStructure([[F(5, 2), 0, 0], [0, 3, 0], [F(-1, 2), 0, F(7, 2)]], p21n, sites, species)
    transform = p21n.transform_from_standard
    standard = p21n.standard_setting()
    moved = []
    for site in sites:
        own = p21n.wyckoff_position(site.wyckoff).representative.coordinate(site.free_params)
        identified = standard.identify_wyckoff(transform.to_standard(own).normalize())
        assert identified is not None
        moved.append(WyckoffSite(identified[0].letter, identified[1], site.species))
    target = ASUStructure(transform.basis_to_standard(source.cell.basis), standard, moved, species)
    left, right = distance_bonds(source), distance_bonds(target)
    assert left.bonds and _counts(left) == _counts(right)
    assert canonical_protonetstructure(left) == canonical_protonetstructure(right)


def test_a_mixed_site_takes_its_largest_radius() -> None:
    # Na-Cl 3.3 Å: beyond the Na-Cl cutoff 3.13, within the K-Cl cutoff 3.50
    assert distance_bonds(_rocksalt(F(66, 10))).bonds == ()
    mixed = Species("M", ("Na", "K", "vacancy"), (F(1, 2), F(1, 4), F(1, 4)))
    assert _counts(distance_bonds(_rocksalt(F(66, 10), mixed))) == (6, 6)


def test_split_positions_closer_than_half_an_angstrom_are_not_bonded() -> None:
    p1 = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        1,
        [WyckoffSite("a", (0, 0, 0), "Na"), WyckoffSite("a", (F(1, 20), 0, 0), "Na")],  # 0.25 Å apart
        [NA],
    )
    assert distance_bonds(p1).bonds == ()


def test_an_element_without_a_radius_is_named() -> None:
    with pytest.raises(ValueError, match="'Bk'"):
        distance_bonds(_rocksalt(5, Species("Bk", ("Bk",), (1,))))
    for bad in (float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError, match="finite"):
            distance_bonds(_rocksalt(5), tolerance=bad)


def test_a_vacancy_only_site_has_no_bonds_and_no_say_in_the_metal_rule() -> None:
    structure = ASUStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        221,
        [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "V")],
        [NA, Species("V", ("vacancy",), (1,))],
    )
    assert _counts(distance_bonds(structure)) == (6, 0)  # Na-Na 3 Å, bonded: no non-metal present


def test_a_dilute_non_metal_on_a_mixed_site_switches_the_metal_rule_on() -> None:
    def cu3au(species: Species) -> ASUStructure:
        return ASUStructure(
            [[F(375, 100), 0, 0], [0, F(375, 100), 0], [0, 0, F(375, 100)]],
            221,
            [WyckoffSite("a", (), species.name), WyckoffSite("c", (), "Cu")],
            [species, Species("Cu", ("Cu",), (1,))],
        )

    assert _counts(distance_bonds(cu3au(Species("M", ("Au", "O"), (F(9, 10), F(1, 10)))))) == (12, 4)
    # a zero concentration is no constituent
    assert _counts(distance_bonds(cu3au(Species("M", ("Au", "O"), (1, 0))))) == (12, 12)


def test_a_site_bonds_to_its_own_translates() -> None:
    found = distance_bonds(
        ASUStructure([[F(5, 2), 0, 0], [0, 3, 0], [0, 0, 4]], 1, [WyckoffSite("a", (0, 0, 0), "Na")], [NA])
    )
    assert _counts(found) == (4,)  # 2.5 and 3 Å along a and b; 4 Å along c is beyond the 3.77 Å cutoff
    assert all(bond.site == bond.partner == 0 for bond in found.bonds or ())


def _brute(structure: ASUStructure, tolerance: float = 0.45) -> list[Counter[tuple[Any, ...]]]:
    """Per site, the (partner, Cartesian vector) of every bonded atom, by enumerating the expanded cell."""
    coords, _, counts = structure._expansion
    owner = [site for site, count in enumerate(counts) for _ in range(count)]
    basis = structure.cell.basis.to_floats()
    M, v = structure.transform.matrix.to_floats(), structure.transform.vector.to_floats()
    elements = {s.name: [e for e, c in zip(s.chemical_symbols, s.concentration) if c > 0] for s in structure.species}
    sites = structure.wyckoff_sites
    radius = [max(_COVALENT_RADII[e] for e in elements[site.species]) for site in sites]
    metal = [not _NON_METALS.intersection(elements[site.species]) for site in sites]
    rule = not all(metal)
    found: list[Counter[tuple[Any, ...]]] = [Counter() for _ in sites]
    for i, site in enumerate(sites):
        p = [float(x) for x in _reference(structure.spacegroup, site)]
        own = [sum(M[r][c] * p[c] for c in range(3)) + v[r] for r in range(3)]
        for q, j in zip(coords.to_floats(), owner, strict=True):
            if rule and metal[i] and metal[j]:
                continue
            for n in itertools.product(range(-4, 5), repeat=3):
                f = [q[k] + n[k] - own[k] for k in range(3)]
                x = [sum(f[k] * basis[k][c] for k in range(3)) for c in range(3)]
                if 0.5 <= math.hypot(*x) <= radius[i] + radius[j] + tolerance:
                    found[i][(j, *(round(c, 4) for c in x))] += 1
    return found


def _listed(structure: ASUStructure) -> list[Counter[tuple[Any, ...]]]:
    """Per site, the (partner, Cartesian vector) of every bond as found, before normalization."""
    basis = structure.cell.basis.to_floats()
    M = structure.transform.matrix.to_floats()
    sites, group = structure.wyckoff_sites, structure.spacegroup
    found: list[Counter[tuple[Any, ...]]] = [Counter() for _ in sites]
    for bond in structure._bonds or ():
        p, q = _reference(group, sites[bond.site]), _apply(_op(bond.operation), _reference(group, sites[bond.partner]))
        f = [sum(M[r][c] * float(q[c] - p[c]) for c in range(3)) for r in range(3)]
        found[bond.site][(bond.partner, *(round(sum(f[k] * basis[k][c] for k in range(3)), 4) for c in range(3)))] += 1
    return found


def _skewed_p1() -> ASUStructure:
    sites = [
        WyckoffSite("a", (F(1, 9), F(2, 7), F(3, 11)), "Si"),
        WyckoffSite("a", (F(5, 8), F(1, 3), F(7, 9)), "O"),
        WyckoffSite("a", (F(2, 5), F(5, 6), F(1, 2)), "Na"),
        WyckoffSite("a", (F(7, 8), F(5, 7), F(1, 13)), "Fe"),
    ]
    species = [Species(e, (e,), (1,)) for e in ("Si", "O", "Na", "Fe")]
    return ASUStructure(
        [[F(41, 10), 0, 0], [F(13, 10), F(37, 10), 0], [F(-9, 10), F(7, 10), F(39, 10)]], 1, sites, species
    )


def _c2m() -> ASUStructure:
    sites = [
        WyckoffSite("j", (F(1, 7), F(1, 5), F(2, 9)), "Si"),
        WyckoffSite("a", (), "O"),
        WyckoffSite("i", (F(3, 8), F(1, 6)), "Mg"),
    ]
    species = [Species(e, (e,), (1,)) for e in ("Si", "O", "Mg")]
    return ASUStructure([[F(61, 10), 0, 0], [0, F(43, 10), 0], [F(-12, 10), 0, F(52, 10)]], 12, sites, species)


def _rhombohedral() -> ASUStructure:
    sites = [WyckoffSite("c", (F(1, 7),), "Si"), WyckoffSite("a", (), "O")]
    species = [Species(e, (e,), (1,)) for e in ("Si", "O")]
    return ASUStructure([[0, 3, 3], [3, 0, 3], [3, 3, 0]], Spacegroup.from_setting("166:R"), sites, species)


def _transformed() -> tuple[ASUStructure, ASUStructure]:
    """Rock salt in its standard cell and in a sheared, shifted own cell (non-identity transform)."""
    plain = _rocksalt(F(564, 100))
    transform = SettingTransform([[1, 1, 0], [0, 1, 0], [0, 0, 1]], (F(1, 8), F(1, 4), 0))
    sheared = ASUStructure(
        transform.basis_to_setting(plain.cell.basis), 225, plain.wyckoff_sites, plain.species, transform
    )
    return plain, sheared


@pytest.mark.parametrize("make", [_skewed_p1, _c2m, _rhombohedral, lambda: _transformed()[1]])
def test_the_neighbour_search_matches_a_cartesian_enumeration(make: Any) -> None:
    structure = make()
    found = distance_bonds(structure)
    assert _listed(found) == _brute(structure)
    assert _counts(found) == tuple(sum(counter.values()) for counter in _brute(structure))


def test_a_non_identity_setting_transform_gives_the_same_network() -> None:
    plain, sheared = _transformed()
    assert not sheared.transform.is_identity()
    assert canonical_protonetstructure(distance_bonds(sheared)) == canonical_protonetstructure(distance_bonds(plain))


@pytest.mark.parametrize("entry", list(MOLECULAR))
def test_molecular_networks_equal_the_pair_cutoff_ones(entry: str) -> None:
    asu, _, rows = _described(entry)
    tables = _tables(asu.space_group_it_number)
    reference = ASUStructure(
        asu.cell,
        asu.spacegroup,
        asu.wyckoff_sites,
        asu.species,
        bonds=[
            SiteBond(i, j, AffineOperation(op.matrix, op.vector + FracVector(g[1:])))
            for i, j, g in rows
            for op in (tables.operations[g[0]],)
        ],
    )
    assert distance_bonds(asu).bonds == reference.bonds
