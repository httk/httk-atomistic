"""The bond network as one value on every structure (protonet follow-up 7, part B, packet P1).

Values and validation, the unit-cell normal form and shapes, the bridge from unit cells to the
fundamental-domain bond algebra, and ``bonds`` on every structure class and view.
"""

import itertools
import logging
import math
import pickle
import random
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import FracVector, SurdVector
from test_bonding import _rocksalt, _transformed
from test_optimade_structure import _resource
from test_structure_bonds import _diamond, _same_network
from test_structure_shapes import B8, _b8

from httk import atomistic
from httk.atomistic import (
    AffineOperation,
    ASEAtoms,
    ASUStructure,
    ASUStructureView,
    BondNetwork,
    BondSource,
    Cell,
    DatastreamStructure,
    FundamentalDomainStructure,
    ModulatedStructure,
    OptimadeStructure,
    PlainStructure,
    PlainStructureView,
    PymatgenStructure,
    RecordStructure,
    SiteBond,
    Sites,
    Spacegroup,
    Species,
    SymopsStructure,
    Trajectory,
    UnitcellStructure,
    UnitcellStructureView,
    VASPStructure,
    WyckoffSite,
    distance_bonds,
    niggli_reduced,
    read_cif_structures,
    recognize_asu,
)
from httk.atomistic.models.structure import bonds as bonds_module
from httk.atomistic.models.structure.asu import _normalized_bonds
from httk.atomistic.models.structure.bonds import (
    _carried,
    _cell_carried,
    _cell_domain,
    _CoincidentAtoms,
    _expanded_network,
    _matched,
    _pair_key,
)
from httk.atomistic.storage.records import _domain_structure_record_from_value, _unitcell_record_from_structure
from httk.atomistic.symmetry import shapes
from httk.atomistic.symmetry.shapes import (
    _baseline_shapes,
    _cell_baseline_shapes,
    _cell_bond_counts,
    _site_shapes,
    _structure_bond_counts,
)

FIXTURES = Path(__file__).parent / "fixtures"
C = Species("C", ("C",), (1,))
FE, MN, O = (Species(name, (name,), (1,)) for name in ("Fe", "Mn", "O"))
SOURCE = BondSource(description="by hand", software="test", version="1")
SHAPES_SOURCE = BondSource(software="shapes")
LOGGER = "httk.atomistic.models.structure.bonds"


def T(*n: Any) -> AffineOperation:
    """The lattice translation ``n``."""
    return AffineOperation(FracVector.eye((3, 3)), n)


# a chain along a through C at 0 and C at 1/2 of a 6 x 4 x 5 cell
CHAIN = (SiteBond(0, 1, T(0, 0, 0)), SiteBond(1, 0, T(1, 0, 0)))
NORMAL_CHAIN = (SiteBond(0, 1, T(0, 0, 0)), SiteBond(0, 1, T(-1, 0, 0)))


def _pair(bonds: Any = None, cell: Any = None) -> UnitcellStructure:
    sites = [[0, 0, 0], [F(1, 2), 0, 0]]
    return UnitcellStructure(cell or [[6, 0, 0], [0, 4, 0], [0, 0, 5]], sites, species_at_sites=[C, C], bonds=bonds)


# ---------------------------------------------------------------------------------------------
# values and construction
# ---------------------------------------------------------------------------------------------
def test_bonds_are_given_as_a_network_a_mapping_or_a_plain_sequence() -> None:
    network = BondNetwork(CHAIN, ("L2", "L2"), SOURCE)
    source = {"description": "by hand", "software": "test", "version": "1"}
    mapping = {"pairs": CHAIN, "shapes": ["L2", "L2"], "source": source}
    for make in (_pair, lambda bonds: ASUStructure([[6, 0, 0], [0, 4, 0], [0, 0, 5]], 1, _P1_SITES, [C], bonds=bonds)):
        from_network, from_mapping, plain = make(network), make(mapping), make(list(CHAIN))
        assert from_network == from_mapping and from_network.bonds == from_mapping.bonds
        assert from_mapping.bonds.source == SOURCE and from_mapping.bonds.shapes == ("L2", "L2")
        assert plain.bonds is not None and plain.bonds.pairs == NORMAL_CHAIN
        assert plain.bonds.shapes is None and plain.bonds.source is None and plain.bonds.shapes_source is None
        assert make(None).bonds is None and make([]).bonds == BondNetwork(())


_P1_SITES = [WyckoffSite("a", (0, 0, 0), "C"), WyckoffSite("a", (F(1, 2), 0, 0), "C")]


@pytest.mark.parametrize(
    ("bonds", "error", "message"),
    [
        ("x", TypeError, "a sequence of SiteBond"),
        (3, TypeError, "a sequence of SiteBond"),
        ({"pairs": (), "extra": 1}, ValueError, "unknown keys"),
        ({"shapes": None}, ValueError, "needs the key 'pairs'"),
        ([(0, 1)], TypeError, "SiteBond values"),
        ({"pairs": (), "source": {"description": ""}}, ValueError, "description must not be empty"),
        ({"pairs": (), "source": {"uri": " \t"}}, ValueError, "uri must not be empty or blank"),
        ({"pairs": (), "source": {"version": 2}}, TypeError, "version must be a str or None"),
        ({"pairs": (), "source": {"name": "x"}}, ValueError, "source has unknown keys"),
        ({"pairs": (), "source": 1}, TypeError, "a BondSource, a mapping or None"),
        ({"pairs": CHAIN, "shapes_source": SHAPES_SOURCE}, ValueError, "shapes_source needs shapes"),
        ({"pairs": CHAIN, "shapes": (None, None), "shapes_source": SHAPES_SOURCE}, ValueError, "needs shapes"),
        ({"pairs": CHAIN, "shapes": "L2"}, TypeError, "not a str"),
        ({"pairs": CHAIN, "shapes": (2, None)}, TypeError, "str or None"),
        ([SiteBond(0, 1, AffineOperation("-x,y,z"))], ValueError, "must be a lattice translation"),
        ([SiteBond(0, 1, T(F(1, 2), 0, 0))], ValueError, "must be a lattice translation"),
        ([SiteBond(0, 2, T(0, 0, 0))], ValueError, r"outside 0\.\.1"),
        ([SiteBond(1, 1, T(0, 0, 0))], ValueError, "bonds an atom to itself"),
    ],
)
def test_unit_cell_networks_are_validated(bonds: Any, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        _pair(bonds)


def test_a_unit_cell_bond_never_translates_along_a_non_periodic_axis() -> None:
    slab = Cell([[6, 0, 0], [0, 4, 0], [0, 0, 5]], periodicity=(True, True, False))
    assert _pair([SiteBond(0, 1, T(-1, 1, 0))], slab).bonds is not None
    with pytest.raises(ValueError, match="non-periodic axis"):
        _pair([SiteBond(0, 1, T(0, 0, 1))], slab)


def test_unit_cell_pairs_have_one_normal_form() -> None:
    spellings = [CHAIN, CHAIN[::-1], (SiteBond(1, 0, T(0, 0, 0)), SiteBond(0, 1, T(-1, 0, 0)), *CHAIN)]
    assert {_pair(list(pairs)).bonds.pairs for pairs in spellings} == {NORMAL_CHAIN}  # type: ignore[union-attr]
    # a bond of a site to its own translate: n and -n are one bond, the smaller kept
    single = UnitcellStructure([[3, 0, 0], [0, 4, 0], [0, 0, 5]], [[0, 0, 0]], species_at_sites=[C])
    for n in ((1, 0, 0), (-1, 0, 0)):
        assert UnitcellStructure(
            single.cell, single.sites, species_at_sites=[C], bonds=[SiteBond(0, 0, T(*n))]
        ).bonds == (BondNetwork((SiteBond(0, 0, T(-1, 0, 0)),)))
    # a larger network, shuffled and with half its bonds reversed, is the P1 domain's normal form
    cell = UnitcellStructureView(distance_bonds(_rocksalt(F(564, 100)))).unview()
    assert cell.bonds is not None and len(cell.bonds.pairs) == 24
    spelled = [
        SiteBond(b.partner, b.site, b.operation.inverse()) if index % 2 else b
        for index, b in enumerate(cell.bonds.pairs)
    ]
    random.Random(7).shuffle(spelled)
    again = UnitcellStructure(cell.cell, cell.sites, cell.species, cell.species_at_sites, bonds=spelled)
    assert again.bonds == cell.bonds
    domain = _cell_domain(cell)
    assert _normalized_bonds(Spacegroup.standard(1), domain.wyckoff_sites, spelled) == cell.bonds.pairs
    assert domain.bonds is not None and domain.bonds.pairs == cell.bonds.pairs


def test_equality_and_hashing_ignore_the_sources_and_repr_round_trips() -> None:
    sourced = BondNetwork(CHAIN, ("L2", "L2"), SOURCE, SHAPES_SOURCE)
    bare = BondNetwork(list(CHAIN), ["L2", "L2"])
    assert sourced == bare and hash(sourced) == hash(bare)
    assert sourced != BondNetwork(CHAIN) != BondNetwork(CHAIN[:1])
    names = {"F": F, "Fraction": F, **vars(atomistic)}
    again = eval(repr(sourced), names)
    assert again == sourced and (again.source, again.shapes_source) == (SOURCE, SHAPES_SOURCE)
    # all-None shapes and an all-None source are no shapes and no source
    empty = BondNetwork((), (None, None), BondSource(), {})
    assert (empty.shapes, empty.source, empty.shapes_source) == (None, None, None)
    # structures: sources are outside equality; None, no pairs and pairs differ
    cell = _pair(sourced)
    assert cell == _pair(bare) and cell.bonds is not None and cell.bonds.source == SOURCE
    assert _pair() != _pair([]) != _pair(list(CHAIN)) != _pair()
    rebuilt = eval(repr(cell), names)
    assert rebuilt == cell and rebuilt.bonds.shapes_source == SHAPES_SOURCE
    assert "bonds" not in repr(_pair())
    domain = ASUStructure(cell.cell, 1, _P1_SITES, [C], bonds=sourced)
    assert domain == ASUStructure(cell.cell, 1, _P1_SITES, [C], bonds=bare)
    assert "bonds=BondNetwork(pairs=" in repr(domain)


# ---------------------------------------------------------------------------------------------
# shapes on unit cells
# ---------------------------------------------------------------------------------------------
def test_unit_cell_shapes_are_validated_against_the_bond_counts() -> None:
    assert _pair(BondNetwork(CHAIN, ("L2", None))).bonds == BondNetwork(NORMAL_CHAIN, ("L2", None))
    dimer = [SiteBond(0, 1, T(0, 0, 0))]
    for bonds, message in (
        (BondNetwork(CHAIN, ("L2",)), "one symbol or None per site: 1 for 2 sites"),
        (BondNetwork(CHAIN, ("T4", None)), r"shapes\[0\]: shape 'T4' is for 4 bonds, but the site has 2"),
        (BondNetwork(CHAIN, ("XX2", None)), "unknown shape symbol 'XX2'"),
        (BondNetwork(dimer, (None, "L2")), r"shapes\[1\]: shape 'L2' given for a site with 1 bond"),
    ):
        with pytest.raises(ValueError, match=message):
            _pair(bonds)


@pytest.mark.parametrize("name", ["AgO", "diopside"])
def test_cell_bond_counts_count_the_distinct_neighbour_atoms(name: str) -> None:
    cell = UnitcellStructureView(_b8(name)).unview()
    assert cell.bonds is not None
    pairs, n = cell.bonds.pairs, len(cell.species_at_sites)
    # every atom within the fixture's element-pair cutoff, counted from Cartesian distances
    cutoffs = {frozenset(pair): value for pair, value in B8[name][1].items()}
    element = {species.name: species.chemical_symbols[0] for species in cell.species}
    names = [element[name] for name in cell.species_at_sites]
    basis, rows = cell.cell.basis.to_floats(), cell.sites.reduced_coords.to_floats()

    def cartesian(x: Any) -> list[float]:
        return [sum(x[k] * basis[k][c] for k in range(3)) for c in range(3)]

    brute = []
    for i in range(n):
        here = cartesian(rows[i])
        brute.append(
            sum(
                1
                for j in range(n)
                if frozenset((names[i], names[j])) in cutoffs
                for t in itertools.product(range(-2, 3), repeat=3)
                if 1e-6
                < math.dist(here, cartesian([rows[j][k] + t[k] for k in range(3)]))
                < cutoffs[frozenset((names[i], names[j]))]
            )
        )
    assert _cell_bond_counts(n, pairs) == tuple(brute)
    domain = _cell_domain(cell)  # the follow-up 5 coset count in P1
    assert _structure_bond_counts(domain.spacegroup, domain.wyckoff_sites, pairs) == tuple(brute)
    assert _cell_bond_counts(1, [SiteBond(0, 0, T(1, 0, 0))]) == (2,)  # an atom and its translate: two bonds


def _cell_angles(structure: Any) -> list[tuple[float, ...]]:
    """Per site, the sorted angles between its bonds, as the baseline reads them."""
    points = [[F(x) for x in row] for row in structure.sites.reduced_coords.to_fractions()]
    metric = shapes._exact_metric(structure)
    angles = []
    for s, atoms in enumerate(shapes._cell_neighbours(len(points), structure.bonds.pairs)):
        vectors = [tuple(p + t - q for p, t, q in zip(points[j], n, points[s], strict=True)) for j, n in atoms]
        angles.append(tuple(sorted(shapes._angle(metric, u, v) for u, v in itertools.combinations(vectors, 2))))
    return angles


def _moved(cell: UnitcellStructure, W: Any, p: Any = (0, 0, 0), order: Any = None) -> UnitcellStructure:
    """``cell`` exactly re-described by ``x' = W x + p`` (``W`` integer, det +-1), sites unwrapped and in ``order``."""
    matrix = FracVector(W).to_fractions()
    order = list(range(len(cell.species_at_sites))) if order is None else order
    rows = cell.sites.reduced_coords.to_fractions()
    position = {old: new for new, old in enumerate(order)}

    def linear(x: Any) -> list[F]:
        return [sum((F(matrix[r][c]) * F(x[c]) for c in range(3)), F(0)) for r in range(3)]

    assert cell.bonds is not None
    # a bond vector x_j + n - x_i becomes W (x_j + n - x_i): the translation becomes W n
    pairs = [
        SiteBond(position[b.site], position[b.partner], T(*linear(b.operation.vector.to_fractions())))
        for b in cell.bonds.pairs
    ]
    return UnitcellStructure(
        SurdVector(FracVector(W).inv().T()) * cell.cell.basis,
        [[a + F(q) for a, q in zip(linear(rows[k]), p, strict=True)] for k in order],
        cell.species,
        [cell.species_at_sites[k] for k in order],
        bonds=pairs,
    )


def _doubled(cell: UnitcellStructure) -> tuple[UnitcellStructure, list[int]]:
    """The cell doubled along a, its network carried by the bridge; and each new site's original site."""
    rows = cell.sites.reduced_coords.to_fractions()
    n = len(rows)
    target = UnitcellStructure(
        SurdVector(FracVector(((2, 0, 0), (0, 1, 0), (0, 0, 1)))) * cell.cell.basis,
        [[(F(x[0]) + k) / 2, x[1], x[2]] for k in (0, 1) for x in rows],
        cell.species,
        [cell.species_at_sites[i] for _ in (0, 1) for i in range(n)],
    )
    halve = AffineOperation(((F(1, 2), 0, 0), (0, 1, 0), (0, 0, 1)))
    return _cell_carried(cell, target, halve), [i for _ in (0, 1) for i in range(n)]


@pytest.mark.parametrize("name", ["AgO", "CuCl2 2.6", "diopside"])
def test_the_unit_cell_baseline_is_bit_identical_over_exact_redescriptions(name: str) -> None:
    domain = _b8(name)
    cell = UnitcellStructureView(domain).unview()
    symbols = _cell_baseline_shapes(cell)
    # the domain baseline on the expansion: these metrics keep the group exactly
    expanded = [
        symbol for symbol, count in zip(_baseline_shapes(domain), domain.multiplicities()) for _ in range(count)
    ]
    assert list(symbols) == expanded == list(_site_shapes(cell))
    niggli = niggli_reduced(cell)
    carried = _cell_carried(cell, niggli.structure, AffineOperation(niggli.transform.inv().T()))
    _assert_redescriptions_agree(cell, [(carried, list(range(len(symbols))))])


def _assert_redescriptions_agree(cell: UnitcellStructure, more: list[tuple[UnitcellStructure, list[int]]]) -> None:
    """Exact re-descriptions of ``cell`` (and ``more``) give bit-identical angles and the same symbols per atom."""
    angles, symbols = _cell_angles(cell), _cell_baseline_shapes(cell)
    n = len(symbols)
    order = list(range(n))
    random.Random(n).shuffle(order)
    descriptions = [
        (_moved(cell, ((1, 1, 0), (0, 1, 0), (0, 0, 1))), list(range(n))),  # a cell change
        (_moved(cell, ((1, 0, 0), (0, 1, 0), (1, -1, 1)), (F(1, 7), F(-1, 3), F(5, 2))), list(range(n))),  # and origin
        (_moved(cell, ((-1, 0, 0), (0, 1, 0), (0, 0, 1))), list(range(n))),  # the mirror-image description
        (_moved(cell, ((0, 1, 0), (1, 0, 0), (0, 0, 1)), order=order), order),  # permuted sites and axes
        _doubled(cell),  # a supercell
        *more,
    ]
    for described, origin in descriptions:
        assert described.bonds is not None
        assert _cell_angles(described) == [angles[k] for k in origin]
        assert _cell_baseline_shapes(described) == tuple(symbols[k] for k in origin)


def test_the_unit_cell_baseline_is_bit_identical_with_a_surd_metric() -> None:
    basis = SurdVector.from_radicand_map(
        {
            1: [[3, 0, 0], [0, 3, 0], [0, 1, 4]],
            2: [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
            3: [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
        }
    )
    rows = [[0, 0, 0], [F(1, 3), F(1, 4), F(1, 5)], [F(-1, 3), F(1, 5), F(1, 7)], [F(1, 6), F(-1, 4), F(2, 5)]]
    bonds = [
        *(SiteBond(0, j, T(0, 0, 0)) for j in (1, 2, 3)),
        SiteBond(0, 1, T(-1, 0, 0)),
        SiteBond(1, 2, T(0, 1, 0)),
        SiteBond(2, 3, T(0, 0, -1)),
    ]
    oxygen = Species("O", ("O",), (1,))
    cell = UnitcellStructure(Cell(basis), rows, species_at_sites=[C, oxygen, oxygen, oxygen], bonds=bonds)
    assert not cell.cell.metric().is_rational
    _assert_redescriptions_agree(cell, [])


def test_the_unit_cell_baseline_refuses_a_zero_length_bond() -> None:
    stacked = UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [0, 0, 0], [F(1, 2), 0, 0]],
        species_at_sites=[FE, MN, O],
        bonds=[SiteBond(0, 1, T(0, 0, 0)), SiteBond(0, 2, T(0, 0, 0))],
    )
    with pytest.raises(ValueError, match="site 0: a zero-length bond has no direction; supply its shape"):
        _site_shapes(stacked)
    assert _site_shapes(stacked._with_bonds(BondNetwork(stacked.bonds.pairs, ("I2", None, None))))[0] == "I2"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------------------------
# a domain's network on its unit cell, and back through recognition
# ---------------------------------------------------------------------------------------------
def _assigned(domain: Any) -> Any:
    """``domain`` with its baseline shapes assigned and both sources."""
    domain = ASUStructureView(domain).resolve() if not isinstance(domain, FundamentalDomainStructure) else domain
    symbols = _site_shapes(domain)
    shapes_source = SHAPES_SOURCE if any(symbols) else None
    return domain._with_bonds(BondNetwork(domain.bonds.pairs, symbols, SOURCE, shapes_source))


ROUND_TRIPS = {
    "NaCl": lambda: distance_bonds(_rocksalt(F(564, 100))),
    "NaCl, sheared own cell with an origin shift": lambda: distance_bonds(_transformed()[1]),
    "diamond, non-standard origin choice": lambda: _diamond("227:1"),
    "diopside": lambda: distance_bonds(httk.core.load(str(FIXTURES / "cod" / "1000008.cif"))),
    "CIF site served as another orbit member": lambda: httk.core.load(
        str(FIXTURES / "cif_bonds" / "merged_other_member.cif")
    ),
}


def _shape_map(cell: Any) -> dict[tuple[Any, ...], Any]:
    rows = cell.sites.reduced_coords.to_fractions()
    symbols = cell.bonds.shapes or (None,) * len(rows)
    return {
        (*(F(x) % 1 for x in row), name): symbol
        for row, name, symbol in zip(rows, cell.species_at_sites, symbols, strict=True)
    }


def _atom_pairs(cell: Any) -> set[tuple[object, ...]]:
    rows = [[F(x) for x in row] for row in cell.sites.reduced_coords.to_fractions()]
    names = cell.species_at_sites
    return {
        _pair_key(
            tuple(rows[b.site]),
            tuple(q + F(t) for q, t in zip(rows[b.partner], b.operation.vector.to_fractions(), strict=True)),
            (names[b.site], names[b.partner]),
        )
        for b in cell.bonds.pairs
    }


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_a_domain_network_goes_to_its_unit_cell_and_back_through_recognition(name: str) -> None:
    domain = _assigned(ROUND_TRIPS[name]())
    view = UnitcellStructureView(domain)
    cell = view.unview()
    assert cell.bonds == view.bonds and cell.bonds is not None and view.bonds is not None
    assert cell.bonds.pairs == view.bonds.pairs  # the view's expansion is already in normal form
    sources = (SOURCE, domain.bonds.shapes_source)
    assert (cell.bonds.source, cell.bonds.shapes_source) == sources
    # every atom has its site's bond count and shape
    counts = _structure_bond_counts(domain.spacegroup, domain.wyckoff_sites, domain.bonds.pairs)
    multiplicities = domain.multiplicities()
    assert _cell_bond_counts(len(cell.species_at_sites), cell.bonds.pairs) == tuple(
        count for count, times in zip(counts, multiplicities, strict=True) for _ in range(times)
    )
    symbols = domain.bonds.shapes or (None,) * len(multiplicities)
    expanded = tuple(symbol for symbol, times in zip(symbols, multiplicities, strict=True) for _ in range(times))
    assert cell.bonds.shapes == (None if domain.bonds.shapes is None else expanded)
    # back: matched onto the recognized domain (the recognition carry), then compared as atoms
    recognized = recognize_asu(cell)
    assert recognized.cell == cell.cell
    carried = _matched(_cell_domain(cell), recognized, recognized.transform.operation.inverse())
    assert carried is not None and (carried.source, carried.shapes_source) == sources
    back = recognized._with_bonds(carried)
    assert _same_network(domain, back)
    again = UnitcellStructureView(back).unview()
    assert _atom_pairs(again) == _atom_pairs(cell)
    assert _shape_map(again) == _shape_map(cell)


def test_a_molecular_representative_off_its_orbit_drops_the_network(caplog: pytest.LogCaptureFixture) -> None:
    def molecule(representative: Any) -> FundamentalDomainStructure:
        site = WyckoffSite("a", (0, 0, 0), "C", representative=representative)
        cell = [[6, 0, 0], [0, 4, 0], [0, 0, 5]]
        return FundamentalDomainStructure(cell, 1, [site], [C], coordinate_precision="0.001", molecular=True, bonds=[])

    assert _expanded_network(molecule((0, 0, 0))) == BondNetwork(())
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert _expanded_network(molecule((F(1, 1000), 0, 0))) is None
    assert any("not an exact orbit point" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------------------------
# coincident sites on the bridge (review M4)
# ---------------------------------------------------------------------------------------------
def _coincident(second: Species = MN) -> UnitcellStructure:
    """Fe and ``second`` at the origin of a 4 Angstrom cube, O at (1/2, 0, 0), each metal bonded to O on both sides."""
    bonds = [SiteBond(site, 2, T(n, 0, 0)) for site in (0, 1) for n in (0, -1)]
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [0, 0, 0], [F(1, 2), 0, 0]],
        species_at_sites=[FE, second, O],
        bonds=BondNetwork(bonds, source=SOURCE),
    )


def _never_expanded(self: Any) -> Any:
    pytest.fail("a bridge domain was expanded")


def test_coincident_sites_of_two_species_carry_through_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FundamentalDomainStructure, "_expansion", property(_never_expanded))
    doubled, origin = _doubled(_coincident())
    assert doubled.bonds is not None and doubled.bonds.source == SOURCE
    assert origin == [0, 1, 2, 0, 1, 2]
    assert _cell_bond_counts(6, doubled.bonds.pairs) == (2, 2, 4, 2, 2, 4)
    names = doubled.species_at_sites
    assert {tuple(sorted((names[b.site], names[b.partner]))) for b in doubled.bonds.pairs} == {("Fe", "O"), ("Mn", "O")}


def test_coincident_sites_that_cannot_be_told_apart_drop_the_network(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(FundamentalDomainStructure, "_expansion", property(_never_expanded))
    stacked = _coincident(FE)  # two Fe sites at one point
    shift = AffineOperation(FracVector.eye((3, 3)), (0, F(1, 2), 0))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        doubled, _origin = _doubled(stacked)
        assert doubled.bonds is None
        # the coincident Fe and Mn renamed (and so changed) on the way
        source = _coincident()
        renamed = [Species("Fe", ("Fe",), (1,), charges=(2,)), Species("Mn", ("Mn",), (1,), charges=(3,)), O]
        rows = [[F(x) for x in row] for row in source.sites.reduced_coords.to_fractions()]
        target = UnitcellStructure(source.cell, [[a, b + F(1, 2), c] for a, b, c in rows], species_at_sites=renamed)
        assert _cell_carried(source, target, shift).bonds is None
        # with the species kept, the same carry is fine
        kept = UnitcellStructure(target.cell, target.sites, source.species, source.species_at_sites)
        assert _cell_carried(source, kept, shift).bonds is not None
    drops = [record for record in caplog.records if "coincident atoms" in record.getMessage()]
    assert len(drops) == 2 and all(getattr(record, "context", None) == "bonds" for record in drops)
    # the bond algebra itself raises; only the bridge paths drop
    with pytest.raises(_CoincidentAtoms):
        _carried(_cell_domain(source), _cell_domain(target), shift)


# ---------------------------------------------------------------------------------------------
# every structure exposes bonds
# ---------------------------------------------------------------------------------------------
def test_every_structure_class_and_view_exposes_bonds() -> None:
    domain = _assigned(distance_bonds(_rocksalt(F(564, 100))))
    cell = UnitcellStructureView(domain).unview()
    expected = cell.bonds
    assert expected is not None and domain.bonds is not None
    assert ASUStructureView(domain).bonds == domain.bonds
    assert ASUStructureView(domain, immutable_id="x").bonds == domain.bonds
    assert UnitcellStructureView(domain).bonds == UnitcellStructureView(cell).bonds == expected
    record = _domain_structure_record_from_value(domain)
    assert RecordStructure(record).bonds == UnitcellStructureView(RecordStructure(record)).bonds == expected
    stored = RecordStructure(_unitcell_record_from_structure(cell)).bonds
    assert stored == expected and (stored.source, stored.shapes_source) == (expected.source, expected.shapes_source)
    assert PlainStructureView(cell).bonds is None
    pytest.importorskip("numpy")
    assert atomistic.NumericUnitcellStructureView(cell).bonds == expected
    assert atomistic.NumericUnitcellStructure(cell).bonds is None
    # CIF reading: the presented (expanded) sites
    path = str(FIXTURES / "cif_bonds" / "ordered.cif")
    (block,) = read_cif_structures(path)
    loaded = UnitcellStructureView(httk.core.load(path)).bonds
    assert loaded is not None and block.bonds == loaded == DatastreamStructure(path).bonds
    assert UnitcellStructureView(block).bonds == loaded
    # pickled views keep the network, read or not
    for view in (UnitcellStructureView(domain), ASUStructureView(domain)):
        assert pickle.loads(pickle.dumps(view)).bonds == view.bonds
        assert pickle.loads(pickle.dumps(view)).bonds == view.bonds  # now read on the original
    for module, view_name in (("pymatgen", "PymatgenStructureView"), ("ase", "ASEAtomsView")):
        pytest.importorskip(module)
        assert getattr(atomistic, view_name)(cell).bonds is None


def test_the_debug_log_names_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        bonds_module._dropped("a reason")
    (record,) = caplog.records
    assert record.getMessage() == "bond network dropped: a reason" and getattr(record, "context", None) == "bonds"


def test_a_bridge_carry_never_keeps_the_target_s_own_network(caplog: pytest.LogCaptureFixture) -> None:
    halve = AffineOperation(((2, 0, 0), (0, 1, 0), (0, 0, 1)))  # onto the half cell
    stale = BondNetwork((SiteBond(0, 0, T(0, 1, 0)),), source=SOURCE)
    half = UnitcellStructure([[3, 0, 0], [0, 4, 0], [0, 0, 5]], [[0, 0, 0]], species_at_sites=[C], bonds=stale)
    # carried: the chain's own network replaces the target's
    assert _cell_carried(_pair(list(CHAIN)), half, halve).bonds == BondNetwork((SiteBond(0, 0, T(-1, 0, 0)),))
    # dropped: a dimer is not invariant under the half cell's translation, and no network is no network
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert _cell_carried(_pair([SiteBond(0, 1, T(0, 0, 0))]), half, halve).bonds is None
    assert any("not invariant" in record.getMessage() for record in caplog.records)
    assert _cell_carried(_pair(), half, halve).bonds is None
    # dropped on coincident atoms
    source = _coincident(FE)
    shift = AffineOperation(FracVector.eye((3, 3)), (0, F(1, 2), 0))
    moved = [[F(a), F(b) + F(1, 2), F(c)] for a, b, c in source.sites.reduced_coords.to_fractions()]
    target = UnitcellStructure(source.cell, moved, source.species, source.species_at_sites, bonds=stale)
    assert _cell_carried(source, target, shift).bonds is None


def test_trajectory_frames_and_replaced_networks_never_expand_a_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    domain = _assigned(distance_bonds(_rocksalt(F(564, 100))))
    monkeypatch.setattr(bonds_module, "_expanded_network", _never_expanded)
    trajectory = Trajectory([domain, domain])
    assert all(frame.bonds is None for frame in trajectory.frames())
    assert UnitcellStructureView(domain)._with_bonds(None).bonds is None
    cell = UnitcellStructure(domain.cell, [[0, 0, 0]], species_at_sites=[C], bonds=[SiteBond(0, 0, T(1, 0, 0))])
    assert Trajectory([cell]).frame(0).bonds is None


def test_structures_without_bond_data_have_no_network() -> None:
    cell = UnitcellStructure(
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], species_at_sites=[Species("He", ("He",), (1,))]
    )
    assert VASPStructure(cell).bonds is None
    assert OptimadeStructure(_resource()).bonds is None
    silicon = [Species("Si", ("Si",), (1,))]
    assert SymopsStructure(cell.cell, Sites([[0, 0, 0]]), silicon, ("Si",), ("x,y,z",)).bonds is None
    assert (
        ModulatedStructure({"incomm": {"mod_dim": 1, "structural_q": ((F(1, 3), 0, 0),), "magnetic_q": None}}).bonds
        is None
    )
    assert PlainStructure(([[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[0, 0, 0]], [2])).bonds is None
    pymatgen = pytest.importorskip("pymatgen.core")
    assert PymatgenStructure(pymatgen.Structure([[2, 0, 0], [0, 2, 0], [0, 0, 2]], ["He"], [[0, 0, 0]])).bonds is None
    ase = pytest.importorskip("ase")
    assert ASEAtoms(ase.Atoms("He", cell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], pbc=True)).bonds is None
