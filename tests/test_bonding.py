"""Default bond networks from distances (``distance_bonds``)."""

import itertools
import math
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import FracVector
from test_protonet_molecular import MOLECULAR, _described

from httk.atomistic import (
    AffineOperation,
    ASUStructure,
    ASUStructureView,
    SettingTransform,
    SiteBond,
    Spacegroup,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    canonical_protonetstructure,
    distance_bonds,
)
from httk.atomistic.bonding import _COVALENT_RADII, _ELECTRONEGATIVITIES, _NON_METALS, _distance_bonds
from httk.atomistic.models.structure.asu import _apply, _op, _reference
from httk.atomistic.symmetry.protonet import _tables
from httk.atomistic.symmetry.shapes import _structure_bond_counts

FIXTURES = Path(__file__).parent / "fixtures"
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
    # nearest neighbours 2.65 Å, the next 3.75 Å. Cu-Cu is longer than its 2.64 Å radius sum and
    # shares the more electronegative Au, but the shared-ligand rule never parts two metals
    for found in (_distance_bonds(cu3au, 0.45, False), distance_bonds(cu3au)):
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
    """Per site, the (partner, Cartesian vector) of every bonded atom, by enumerating the expanded cell.

    The shared-ligand rule is applied in Cartesian form: the bond from site i along x to a site j atom,
    one of them a metal and the other not, is dropped when it is longer than the radius sum and a
    non-metal atom of site k bonded to i at y is also bonded to that atom (distance |y - x|) with a
    larger electronegativity than both ends.
    """
    coords, _, counts = structure._expansion
    owner = [site for site, count in enumerate(counts) for _ in range(count)]
    basis = structure.cell.basis.to_floats()
    M, v = structure.transform.matrix.to_floats(), structure.transform.vector.to_floats()
    elements = {s.name: [e for e, c in zip(s.chemical_symbols, s.concentration) if c > 0] for s in structure.species}
    sites = structure.wyckoff_sites
    radius = [max(_COVALENT_RADII[e] for e in elements[site.species]) for site in sites]
    metal = [not _NON_METALS.intersection(elements[site.species]) for site in sites]
    rule = not all(metal)
    en = [[_ELECTRONEGATIVITIES.get(e) for e in elements[site.species]] for site in sites]
    end = [None if None in values else max(values) for values in en]
    ligand = [
        None if None in values or not _NON_METALS.issuperset(elements[site.species]) else min(values)
        for values, site in zip(en, sites, strict=True)
    ]
    raw: list[list[tuple[int, list[float]]]] = [[] for _ in sites]
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
                    raw[i].append((j, x))

    def shared(i: int, j: int, x: list[float]) -> bool:
        first, second = end[i], end[j]
        if metal[i] == metal[j] or math.hypot(*x) <= radius[i] + radius[j] or first is None or second is None:
            return False
        return any(
            (value := ligand[k]) is not None
            and value > max(first, second)
            and 0.5 <= math.dist(y, x) <= radius[k] + radius[j] + tolerance
            and not (rule and metal[k] and metal[j])
            for k, y in raw[i]
        )

    found: list[Counter[tuple[Any, ...]]] = [Counter() for _ in sites]
    for i, bonded in enumerate(raw):
        for j, x in bonded:
            if not shared(i, j, x):
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


def _fixture(name: str) -> Any:
    return httk.core.load(str(FIXTURES / f"{name}.cif"))


@pytest.mark.parametrize(
    "make", [_skewed_p1, _c2m, _rhombohedral, lambda: _transformed()[1], lambda: _fixture("cod/1000008")]
)
def test_the_neighbour_search_matches_a_cartesian_enumeration(make: Any) -> None:
    structure = ASUStructureView(make()).resolve()
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


# per Wyckoff site, as "element count" in site order: the shared-ligand rule's reference results
SHARED_LIGAND = {
    "cod/1000008": "Mg6 Ca8 Si4 O4 O3 O4",  # diopside: Ca 12 and Si 6 before
    "structreading/159": "K9 Al4 Si4 O5 O4",  # KAlSiO4: K 12 and Si 7 before
    "structreading/188": "Ba6 Ti6 Si4 O3 O2",  # BaTiSi3O9: Ba 12 and Si 6 before
    "structreading/45": "Tm8 O3 O3 O3 O3 O3 Cu6 B4 B3 B3",  # TmCuB5O10: Tm 10 and B 6 before
    # unchanged by the rule
    "structreading/192": "Al6 Be4 Si4 O2 O3",
    "structreading/196": "S4 O2 Zn4",
    "structreading/168": "Al4 Al4 P4 P4 O2 O2 O2 O2 O2 O2 O2 O2",
    "structreading/28": "Hg1 Te3 O1 O2",
    "structreading/76": "B3 B4 O2 O2 O2 O3 Tl1 Tl1",
    "structreading/103": "Se3 Se3 V6 V6 V6 O2 O2 O2 O2 O2 O2 O2 O2",
    "structreading/145": "B4 B4 O3 O3 O3 O3 Bi5 F1",
    "structreading/122": "Si4 O2",
    "structreading/92": "Te4 O2",
    "cod/1509425": "S6 S4 S4 Ag4 K6 S6 K8 Ag4 K4 Sb4",
    "cod/1509688": "O3 Ag2 Ag4",
    "cod/1535669": "Cl2 Cl2 Al6",
    "cod/1539945": "Cl2 Cu4",
    "cod/4002923": "Fe6 F/O2 O0",
    "cod/1001844": "Np9 Np8 Np9 Np8 Np9 Np8" + " F2" * 23 + " F3 O1 O1",
    "cod/7219102": "N4 H1 H1 C3 O1 C4 H1 H1 O1 C4 H1 H1 H1",
    "cod/7218683": "Br1 Cu4 O2 N3 C3 C3 H1 C3 H1 C3 H1 C3 H1 C3 C3 H1 C4 H1 H1 C4 H1 H1",
    "cod/7217843": "F1 N3 O1 O1 O2 C3 C3 H1 C3 H1 C3 C3 C3 H1 C3 H1 C3 H1 C3 C3 C3 C3 C4 H1 H1 C4 H1 H1 H1 C3 C3 H1"
    " C3 H1 C3 C3 H1 C3 H1",
    "cod/7217003": "S4 N3 N2 N2 O1 C3 O1 C3 C3 H1 C3 H1 C3 C3 C3 C3 H1 C3 C3 H1 C3 H1 C4 H1 H1 H1 C3 H1 C3 H1 C3 H1"
    " N3 C4 H1 H1 H1",
    "cod/4131686": "O1 N3 H1 C2 C2 C3 C3 H1 C3 H1 C3 H1 C3 C3 H1 C3 C4 H1 H1 H1",
}


def _described_counts(structure: ASUStructure) -> str:
    names = {s.name: "/".join(e for e in s.chemical_symbols if e != "vacancy") for s in structure.species}
    return " ".join(f"{names[w.species]}{n}" for w, n in zip(structure.wyckoff_sites, _counts(structure), strict=True))


@pytest.mark.parametrize("name", list(SHARED_LIGAND))
def test_fixture_bond_counts_under_the_shared_ligand_rule(name: str) -> None:
    assert _described_counts(distance_bonds(_fixture(name))) == SHARED_LIGAND[name]


def _cluster(*atoms: tuple[Species | str, tuple[float, float, float]]) -> ASUStructure:
    """Isolated atoms (Cartesian Å) in a 12 Å P1 cube, one site each."""
    species = {}
    for kind, _ in atoms:
        found = kind if isinstance(kind, Species) else Species(kind, (kind,), (1,))
        species[found.name] = found
    sites = [
        WyckoffSite("a", tuple(F(str(round(x, 6))) / 12 for x in xyz), kind if isinstance(kind, str) else kind.name)
        for kind, xyz in atoms
    ]
    return ASUStructure([[12, 0, 0], [0, 12, 0], [0, 0, 12]], 1, sites, list(species.values()))


def _triangle(a: Species | str, b: Species | str, c: Species | str, ab: float, ac: float, bc: float) -> ASUStructure:
    """Atoms a, b and c with the given distances (Å) in one plane."""
    x = (ac**2 - bc**2 + ab**2) / (2 * ab)
    return _cluster((a, (0, 0, 0)), (b, (ab, 0, 0)), (c, (x, math.sqrt(ac**2 - x**2), 0)))


def _casio(a: Species | str = "Ca", c: Species | str = "O", b: Species | str = "Si", ab: float = 3.1) -> ASUStructure:
    """A cation a and an anion centre b ``ab`` Å apart, sharing c at 2.4 and 1.6 Å."""
    return _triangle(a, b, c, ab, 2.4, 1.6)


def test_a_cation_does_not_bond_to_an_anion_centre_across_a_shared_ligand() -> None:
    # Ca-Si 3.1 Å, beyond the 2.87 Å radius sum, within the 3.32 Å cutoff; both bonded to O
    assert _counts(_distance_bonds(_casio(), 0.45, False)) == (2, 2, 2)
    assert _counts(distance_bonds(_casio())) == (1, 1, 2)
    # the shared atom must be a non-metal: a shared Na (bonded to Si only, by the metal rule) parts nothing
    assert _counts(distance_bonds(_casio(c="Na"))) == (1, 2, 1)
    # a shared He has no electronegativity
    assert _counts(distance_bonds(_casio(c="He"))) == (2, 2, 2)


def test_non_metal_pairs_and_metal_pairs_are_never_parted() -> None:
    # Si-Si 2.6 Å and C-C 1.6 Å, both beyond their radius sums, across a shared O
    assert _counts(distance_bonds(_triangle("Si", "Si", "O", 2.6, 1.48, 1.48))) == (2, 2, 2)
    assert _counts(distance_bonds(_triangle("C", "C", "O", 1.6, 1.43, 1.43))) == (2, 2, 2)
    # metal pairs: see test_an_intermetallic_bonds_by_distance (Cu3Au keeps Cu-Cu beside Au)


def test_all_drops_are_decided_on_the_tentative_network() -> None:
    # Ca-O 2.6 Å is dropped across F (Ca-F 2.18, O-F 1.30 Å); O still parts Ca from Si (3.1 Å)
    ca, si = (0.0, 0.0, 0.0), (3.1, 0.0, 0.0)
    x = (2.6**2 - 1.6**2 + 3.1**2) / (2 * 3.1)
    o = (x, math.sqrt(2.6**2 - x**2), 0.0)
    structure = _cluster(("Ca", ca), ("Si", si), ("O", o), ("F", (1.4, o[1], 1.0)))
    assert _counts(_distance_bonds(structure, 0.45, False)) == (3, 2, 3, 2)
    assert _pairs(distance_bonds(structure)) == Counter({("O", "Si"): 1, ("F", "O"): 1, ("Ca", "F"): 1})


def test_the_length_condition_keeps_a_cation_closer_than_its_radius_sum() -> None:
    # Cs-Si 3.3 Å, shorter than the 3.55 Å radius sum; Cs-O 3.0 Å and Si-O 1.6 Å
    cssio = _triangle("Cs", "Si", "O", 3.3, 3.0, 1.6)
    assert _counts(distance_bonds(cssio)) == (2, 2, 2)  # the length condition keeps Cs-Si


def test_a_mixed_site_is_a_metal_end_or_a_non_metal_ligand_only_when_all_constituents_are() -> None:
    def mixed(*symbols: str) -> Species:
        return Species("M", symbols, tuple(F(1, len(symbols)) for _ in symbols))

    assert _counts(distance_bonds(_casio(a=mixed("Ca", "Mg")))) == (1, 1, 2)  # a metal end
    assert _counts(distance_bonds(_casio(a=mixed("Ca", "Si")))) == (2, 2, 2)  # Si makes it a non-metal end
    assert _counts(distance_bonds(_casio(c=mixed("O", "S")))) == (1, 1, 2)  # S 2.58 > Si 1.90 as well
    assert _counts(distance_bonds(_casio(c=mixed("O", "Na")))) == (2, 2, 2)  # not a non-metal ligand
    # an end counts with its largest value: H (2.20) parts Ca (1.00) from Si (1.90), not Ca/Au (Au 2.54)
    assert _counts(distance_bonds(_casio(c="H"))) == (1, 1, 2)
    assert _counts(distance_bonds(_casio(a=mixed("Ca", "Au"), c="H"))) == (2, 2, 2)


def test_an_element_without_an_electronegativity_never_loses_a_bond() -> None:
    assert not {"He", "Ne", "Ar", "Rn", "Pm", "Eu", "Tb", "Yb"} & set(_ELECTRONEGATIVITIES)
    # Eu-Si 3.3 Å, beyond the 3.09 Å radius sum; Ca at the same distance is parted
    assert _counts(distance_bonds(_casio(a="Eu", ab=3.3))) == (2, 2, 2)
    assert _counts(distance_bonds(_casio(ab=3.3))) == (1, 1, 2)


def test_a_dropped_bond_orbit_goes_from_both_ends() -> None:
    diopside = ASUStructureView(_fixture("cod/1000008")).resolve()
    tentative, found = _distance_bonds(diopside, 0.45, False), distance_bonds(diopside)
    names = [site.species for site in diopside.wyckoff_sites]

    def ca_si(structure: ASUStructure) -> list[SiteBond]:
        return [b for b in structure._bonds or () if {names[b.site], names[b.partner]} == {"Ca", "Si"}]

    # the tentative network finds each Ca-Si orbit from the Ca and from the Si reference atom
    assert {names[b.site] for b in ca_si(tentative)} == {"Ca", "Si"}
    assert ca_si(found) == []  # dropped as seen from either end, every member (see the Cartesian check)


def test_a_known_limitation_metallacarborane_loses_its_metal_boron_bonds() -> None:
    # COD 4120336: the file's own loop bonds Fe to 6 C and 6 B (Fe-B 2.23-2.38 Å, beyond the 2.16 Å
    # low-spin radius sum); every Fe-B shares a cage C (2.55) more electronegative than Fe and B
    loaded = _fixture("cod/4120336")
    assert {k: v for k, v in _pairs(loaded).items() if "Fe" in k} == {("C", "Fe"): 6, ("B", "Fe"): 6}
    tentative, found = _distance_bonds(loaded, 0.45, False), distance_bonds(loaded)
    assert {k: v for k, v in _pairs(tentative).items() if "Fe" in k} == {("C", "Fe"): 6, ("B", "Fe"): 6}
    assert {k: v for k, v in _pairs(found).items() if "Fe" in k} == {("C", "Fe"): 6}
    assert _counts(found)[0] == 6
