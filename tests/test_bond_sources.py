"""Bond sources: ``distance_bonds`` by kind, ``assign_baseline_shapes`` and the CIF loop source."""

import importlib.metadata
import itertools
import math
import random
from fractions import Fraction as F
from pathlib import Path

import httk.core
import pytest
from test_bonding import _rocksalt

from httk import atomistic
from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    BondNetwork,
    BondSource,
    Cell,
    DatastreamStructure,
    FundamentalDomainStructure,
    RecordStructure,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    assign_baseline_shapes,
    distance_bonds,
    read_cif_structures,
)
from httk.atomistic.bonding import _COVALENT_RADII, _distance_bonds
from httk.atomistic.models.structure import asu_view
from httk.atomistic.storage.records import _domain_structure_record_from_value, _unitcell_record_from_structure

FIXTURES = Path(__file__).parent / "fixtures"
VERSION = importlib.metadata.version("httk-atomistic")
DISTANCE = BondSource(
    "Distance bonds: 0.5 Å <= d <= r(a) + r(b) + 0.45 Å (Cordero et al. 2008 covalent radii), no metal-metal "
    "bonds where a non-metal is present, and no metal to non-metal bond longer than r(a) + r(b) across a "
    "shared, more electronegative non-metal.",
    "httk_basic",
    VERSION,
)
BASELINE = BondSource(
    "Baseline bond shapes: the ideal shape whose sorted bond angles fit best (IUPAC and SHAPE symbols), I<n> "
    "above 15 degrees RMS misfit, L2 from 165 degrees.",
    "httk_basic",
    VERSION,
)


def _rutile() -> ASUStructure:
    return ASUStructure(
        [[F(4594, 1000), 0, 0], [0, F(4594, 1000), 0], [0, 0, F(2959, 1000)]],
        136,
        [WyckoffSite("a", (), "Ti"), WyckoffSite("f", (F(3048, 10000),), "O")],
        [Species("Ti", ("Ti",), (1,)), Species("O", ("O",), (1,))],
    )


def _diopside() -> ASUStructure:
    return ASUStructureView(httk.core.load(str(FIXTURES / "cod" / "1000008.cif"))).resolve()


@pytest.fixture
def no_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("distance_bonds recognised symmetry")

    monkeypatch.setattr(asu_view, "recognize_asu", refuse)


@pytest.mark.usefixtures("no_recognition")
def test_distance_bonds_returns_the_kind_it_was_given() -> None:
    nacl = _rocksalt(F(564, 100))
    domain = FundamentalDomainStructure(nacl.cell, nacl.spacegroup, nacl.wyckoff_sites, nacl.species)
    cell = UnitcellStructureView(nacl).unview()
    assert type(distance_bonds(nacl)) is ASUStructure
    assert type(distance_bonds(ASUStructureView(nacl))) is ASUStructure
    assert type(distance_bonds(domain)) is FundamentalDomainStructure
    for unit in (cell, UnitcellStructureView(nacl), UnitcellStructureView(cell)):
        found = distance_bonds(unit)
        assert type(found) is UnitcellStructure and found.sites == cell.sites
        assert found.bonds is not None and len(found.bonds.pairs) == 24  # 4 Na x 6 Cl, each bond once
    # lazy sources and paths resolve to their domain
    path = str(FIXTURES / "cif_bonds" / "ordered.cif")
    (block,) = read_cif_structures(path)
    for lazy in (block, DatastreamStructure(path), path, Path(path), httk.core.TextstreamFileView(path)):
        found = distance_bonds(lazy)
        assert type(found) is ASUStructure and found.wyckoff_sites == block.resolve().wyckoff_sites  # type: ignore[attr-defined]
        assert found.bonds is not None and found.bonds.source == DISTANCE  # the CIF loop's network replaced
    # records: a domain record keeps its domain kind, a unit-cell record gives a unit cell
    record = _domain_structure_record_from_value(nacl)
    for stored in (record, RecordStructure(record)):
        found = distance_bonds(stored)
        assert type(found) is ASUStructure and found.wyckoff_sites == nacl.wyckoff_sites
    for stored in (_unitcell_record_from_structure(cell), RecordStructure(_unitcell_record_from_structure(cell))):
        found = distance_bonds(stored)
        assert type(found) is UnitcellStructure and found.sites == cell.sites
    pytest.importorskip("numpy")
    numeric = distance_bonds(atomistic.NumericUnitcellStructure(cell))
    assert type(numeric) is UnitcellStructure and numeric.bonds is not None and len(numeric.bonds.pairs) == 24


def test_an_asu_view_of_a_unit_cell_is_the_recognised_route() -> None:
    cell = UnitcellStructureView(_rocksalt(F(564, 100))).unview()
    found = distance_bonds(ASUStructureView(cell))
    assert type(found) is ASUStructure and found.bonds is not None and len(found.bonds.pairs) == 1


@pytest.mark.parametrize("make", [lambda: _rocksalt(F(564, 100)), _rutile, _diopside])
def test_the_unit_cell_and_domain_routes_give_one_network(make: object) -> None:
    domain = make()  # type: ignore[operator]
    from_domain = UnitcellStructureView(distance_bonds(domain)).unview()  # the domain's network, expanded
    from_cell = distance_bonds(UnitcellStructureView(domain).unview())
    assert from_domain.sites == from_cell.sites and from_domain.species_at_sites == from_cell.species_at_sites
    assert from_domain.bonds is not None and from_domain.bonds.shapes is not None
    assert from_domain.bonds == from_cell.bonds  # pairs and shapes
    assert from_domain.bonds.source == from_cell.bonds.source == DISTANCE  # type: ignore[union-attr]


def test_the_sources_and_shapes_false() -> None:
    nacl = _rocksalt(F(564, 100))
    for structure in (nacl, UnitcellStructureView(nacl).unview()):
        found, bare = distance_bonds(structure).bonds, distance_bonds(structure, shapes=False).bonds
        assert found is not None and bare is not None
        assert found.shapes is not None and set(found.shapes) == {"OC6"} and bare.shapes is None
        assert found.pairs == bare.pairs
        assert found.source == bare.source == DISTANCE and found.shapes_source is None is bare.shapes_source
    wide = distance_bonds(nacl, tolerance=-0.25).bonds
    assert wide is not None and wide.source is not None and wide.source.description is not None
    assert "r(a) + r(b) + -0.25 Å" in wide.source.description


def test_a_site_at_a_special_point_of_its_own_letter_raises_with_the_shapes_hint() -> None:
    # P-1 general letter i at the inversion centre 0,0,0
    structure = ASUStructure(
        [[F(564, 100), 0, 0], [0, F(564, 100), 0], [0, 0, F(564, 100)]],
        2,
        [WyckoffSite("i", (0, 0, 0), "Na"), WyckoffSite("h", (), "Cl")],
        [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
    )
    with pytest.raises(ValueError, match=r"special point of its Wyckoff letter 'i'.*pass shapes=False"):
        distance_bonds(structure)
    assert distance_bonds(structure, shapes=False).bonds is not None


def test_assign_baseline_shapes_fills_the_shapes_and_keeps_the_source() -> None:
    nacl = _rocksalt(F(564, 100))
    for bare in (distance_bonds(nacl, shapes=False), distance_bonds(UnitcellStructureView(nacl), shapes=False)):
        shaped = assign_baseline_shapes(bare)
        assert type(shaped) is type(bare) and shaped.bonds is not None and bare.bonds is not None
        assert shaped.bonds.pairs == bare.bonds.pairs and shaped.bonds.source == DISTANCE
        assert shaped.bonds.shapes == distance_bonds(bare).bonds.shapes  # type: ignore[union-attr]
        assert shaped.bonds.shapes_source == BASELINE
    # a hand-built network keeps its (absent) source; the view input gives the kind distance_bonds gives
    hand = UnitcellStructureView(nacl._with_bonds(BondNetwork(distance_bonds(nacl).bonds.pairs)))  # type: ignore[union-attr]
    shaped = assign_baseline_shapes(hand)
    assert type(shaped) is UnitcellStructure and shaped.bonds is not None and shaped.bonds.source is None
    assert shaped.bonds.shapes_source == BASELINE


def test_assign_baseline_shapes_refusals_and_no_op() -> None:
    nacl = _rocksalt(F(564, 100))
    with pytest.raises(ValueError, match="without a bond network"):
        assign_baseline_shapes(nacl)
    with pytest.raises(ValueError, match="already has shapes"):
        assign_baseline_shapes(distance_bonds(nacl))
    lonely = nacl._with_bonds(BondNetwork((), source=BondSource("by hand")))
    assert assign_baseline_shapes(lonely) is lonely  # no site has two bonds: unchanged
    far = distance_bonds(_rocksalt(F(66, 10)), shapes=False)  # Na-Cl 3.3 Å: no bonds
    unchanged = assign_baseline_shapes(far)
    assert unchanged is far and unchanged.bonds == BondNetwork(()) and unchanged.bonds.shapes_source is None
    # a diatomic: every site has one bond, so no site gets a shape
    diatomic = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], periodicity=(False, False, False)),
        [[0, 0, 0], [F(121, 100), 0, 0]],
        [Species("O", ("O",), (1,))],
        ["O", "O"],
    )
    bonded = distance_bonds(diatomic, shapes=False)
    assert bonded.bonds is not None and len(bonded.bonds.pairs) == 1
    kept = assign_baseline_shapes(bonded)
    assert kept is bonded and kept.bonds is not None and kept.bonds.shapes is kept.bonds.shapes_source is None


def test_the_cif_bond_loop_names_its_source() -> None:
    loaded = httk.core.load(str(FIXTURES / "cif_bonds" / "ordered.cif"))
    assert loaded.bonds is not None and loaded.bonds.shapes is None and loaded.bonds.shapes_source is None
    assert loaded.bonds.source == BondSource(description="Bonds listed in the CIF _geom_bond_ loop")
    assert (loaded.bonds.source.software, loaded.bonds.source.version, loaded.bonds.source.uri) == (None, None, None)


def test_a_molecule_in_a_non_periodic_cell_gets_bonds() -> None:
    # methane about the origin in a 1 Å frame: coordinates are lengths, mostly negative or above 1
    d = F(629, 1000)  # C-H 1.089 Å; H-H 1.779 Å, beyond the 1.07 Å cutoff
    hydrogens = [[d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]]
    methane = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], periodicity=(False, False, False)),
        [[0, 0, 0], *hydrogens],
        [Species("C", ("C",), (1,)), Species("H", ("H",), (1,))],
        ["C", "H", "H", "H", "H"],
    )
    found = distance_bonds(methane)
    assert type(found) is UnitcellStructure and found.bonds is not None
    assert [(b.site, b.partner, b.operation.to_xyz()) for b in found.bonds.pairs] == [
        (0, h, "x,y,z") for h in range(1, 5)
    ]
    assert found.bonds.shapes == ("T4", None, None, None, None)
    # a slab: translations only along its periodic directions
    slab = UnitcellStructure(
        Cell([[3, 0, 0], [0, 3, 0], [0, 0, 1]], periodicity=(True, True, False)),
        [[0, 0, -2]],
        [Species("Na", ("Na",), (1,))],
        ["Na"],
    )
    pairs = distance_bonds(slab).bonds.pairs  # type: ignore[union-attr]
    assert sorted(b.operation.to_xyz() for b in pairs) == ["x,y-1,z", "x-1,y,z"]


def test_the_neighbour_search_matches_an_exhaustive_search_with_mixed_periodicity() -> None:
    # random skewed cells, coordinates outside [0, 1) (negative too) on every axis and spread over many
    # bins along the non-periodic ones; tentative networks against every translation in a safe range
    rng = random.Random(3)
    elements = ["C", "Si", "O", "S"]
    species = [Species(e, (e,), (1,)) for e in elements]
    total = 0
    for _ in range(25):
        basis = [[F(rng.randint(-15, 15), 10) + (F(rng.randint(50, 90), 10) if r == c else 0) for c in range(3)]
                 for r in range(3)]  # fmt: skip
        periodicity = (rng.random() < 0.6, rng.random() < 0.6, False)
        # a random walk of steps up to 0.25 per axis, so bonded neighbours straddle integer coordinates
        coords = [[F(rng.randint(-300, 300), 100) if not p else F(rng.randint(-150, 250), 100) for p in periodicity]]
        for _ in range(rng.randint(3, 7)):
            coords.append([x + F(rng.randint(-25, 25), 100) for x in rng.choice(coords)])
        names = [rng.choice(elements) for _ in coords]
        structure = UnitcellStructure(Cell(basis, periodicity=periodicity), coords, species, names)
        network = _distance_bonds(structure, 0.45, False).bonds
        assert network is not None
        found = {(b.site, b.partner, tuple(int(x) for x in b.operation.vector.to_fractions())) for b in network.pairs}
        rows = [[float(x) for x in row] for row in basis]
        expected = set()
        for i, j in itertools.product(range(len(coords)), repeat=2):
            cutoff = _COVALENT_RADII[names[i]] + _COVALENT_RADII[names[j]] + 0.45
            for n in itertools.product(*(range(-9, 10) if p else (0,) for p in periodicity)):
                d = [float(coords[j][k] - coords[i][k]) + n[k] for k in range(3)]
                if 0.5 <= math.hypot(*(sum(d[k] * rows[k][c] for k in range(3)) for c in range(3))) <= cutoff:
                    a, b, t = (
                        (j, i, tuple(-x for x in n)) if i > j else (i, j, min(n, tuple(-x for x in n)) if i == j else n)
                    )
                    expected.add((a, b, t))
        assert found == expected
        total += len(found)
    assert total > 100  # many bonds, not an empty comparison
