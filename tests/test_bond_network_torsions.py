"""Torsion statements on the bond network (Stage 2A): values, normal forms, the geometric sign and carrying.

Every carried statement is checked against the geometry of the structure that holds it: its signed
torsion, recomputed from that structure's own exact coordinates, cell and setting transform, is the
carried angle. The fixtures cover a centrosymmetric group, a Sohncke group (sarcosine), a recognized
cell whose setting transform has determinant -1, a mirror image in a left-handed basis, a
non-standard setting and a non-identity own cell.
"""

import logging
from dataclasses import replace
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import FracVector
from test_bond_network import LOGGER, ROUND_TRIPS, SOURCE
from test_structure_bonds import _diamond

from httk.atomistic import (
    AffineOperation,
    ASUStructure,
    ASUStructureView,
    BondNetwork,
    BondSource,
    Cell,
    FundamentalDomainStructure,
    SiteBond,
    SiteTorsion,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
    canonical_asu_protostructure,
    canonicalize,
    conventional_cell,
    niggli_reduced,
    primitive_cell,
    recognize_asu,
)
from httk.atomistic.models.structure.asu import _affine, _apply, _inv, _mul, _op, _reference, _standard_elements
from httk.atomistic.models.structure.bonds import _bonds_key, _cell_carried, _cell_domain, _matched, _signed_torsions
from httk.atomistic.models.structure.network import _bond_network
from httk.atomistic.symmetry.canonical_protostructure import _mirrored, _right_handed
from httk.atomistic.symmetry.paths import represent_like
from httk.atomistic.symmetry.subgroups import subgroup_representation

FIXTURES = Path(__file__).parent / "fixtures"
C = Species("C", ("C",), (1,))
ONE = AffineOperation.identity()


def T(*n: Any) -> AffineOperation:
    """The lattice translation ``n``."""
    return AffineOperation(FracVector.eye((3, 3)), n)


# ---------------------------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"sites": (0, 1, 2)}, ValueError, "four values"),
        ({"sites": "0123"}, TypeError, "sequence of four"),
        ({"sites": (0, 1, -2, 3)}, ValueError, "non-negative integers"),
        ({"sites": (0, True, 2, 3)}, ValueError, "non-negative integers"),
        ({"operations": (ONE, ONE, ONE, "x,y,z")}, TypeError, "AffineOperation"),
        ({"symbol": "x"}, ValueError, "symbol must be one of"),
        ({"angle": float("nan")}, ValueError, "finite"),
        ({"angle": float("inf")}, ValueError, "finite"),
        ({"angle": 180.5}, ValueError, r"\[-180, 180\]"),
        ({"angle": "60"}, TypeError, "float or None"),
        ({"angle": None, "symbol": "t", "uncertainty": 1.0}, ValueError, "needs an angle"),
        ({"uncertainty": 0.0}, ValueError, "positive"),
        ({"angle": None, "symbol": None}, ValueError, "a symbol, an angle or both"),
        ({"symbol": "+sc", "angle": -60.0}, ValueError, "disagree in sign"),
        ({"symbol": "-ac", "angle": 120.0}, ValueError, "disagree in sign"),
    ],
)
def test_a_torsion_statement_is_validated(changes: dict[str, Any], error: type[Exception], message: str) -> None:
    values: dict[str, Any] = {"sites": (0, 1, 2, 3), "operations": (ONE,) * 4, "angle": 60.0, **changes}
    with pytest.raises(error, match=message):
        SiteTorsion(**values)


def test_a_torsion_statement_normalizes_its_angle() -> None:
    def angle(value: Any, symbol: str | None = None) -> float | None:
        return SiteTorsion([0, 1, 2, 3], [ONE] * 4, symbol, value).angle

    assert angle(-180) == 180.0 and angle(-0.0) == 0.0 and str(angle(-0.0)) == "0.0" and angle(60) == 60.0
    # 0 and 180 agree with both signs; only the sign of a signed symbol is checked, not its range
    assert angle(0, "-sc") == 0.0 and angle(-180, "+ac") == 180.0 and angle(10, "+ac") == 10.0
    assert SiteTorsion((0, 1, 2, 3), (ONE,) * 4, "none").angle is None


def test_the_network_holds_torsions_and_their_source() -> None:
    row = SiteTorsion((0, 1, 0, 1), (ONE, ONE, T(1, 0, 0), T(1, 0, 0)), "t", 180.0)
    network = BondNetwork((), torsions=[row], source=SOURCE, torsions_source=SOURCE)
    assert network.torsions == (row,) and network.torsions_source is None  # equal to the source: one spelling
    assert BondNetwork((), torsions=()).torsions is None
    other = BondSource(software="torsions")
    assert BondNetwork((), torsions=[row], torsions_source={"software": "torsions"}).torsions_source == other
    # torsions are content, their source provenance
    assert network == BondNetwork((), torsions=[row], torsions_source=other) != BondNetwork(())
    mapping = {"pairs": [], "torsions": [row], "torsions_source": {"software": "torsions"}}
    assert _bond_network(mapping) == network and _bond_network(mapping).torsions_source == other  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="torsions_source needs torsions"):
        BondNetwork((), torsions_source=other)
    with pytest.raises(TypeError, match="SiteTorsion values"):
        BondNetwork((), torsions=[SiteBond(0, 1, ONE)])
    with pytest.raises(TypeError, match="sequence of SiteTorsion"):
        BondNetwork((), torsions=row)  # type: ignore[arg-type]
    # a network without torsions reads as before them; one with torsions round-trips
    assert "torsions" not in repr(BondNetwork((), ("L2",), SOURCE))
    namespace = {"BondNetwork": BondNetwork, "BondSource": BondSource, "SiteTorsion": SiteTorsion}
    assert eval(repr(network), {**namespace, "AffineOperation": AffineOperation}) == network


# ---------------------------------------------------------------------------------------------
# geometry: the IUPAC sign
# ---------------------------------------------------------------------------------------------
def _staggered(basis: Any) -> UnitcellStructure:
    """B at the origin's column, C above it along c; A along a from B, D at 60 degrees from a (hexagonal) at C."""
    sites = [[F(1, 5), 0, F(2, 5)], [0, 0, F(2, 5)], [0, 0, F(1, 2)], [F(1, 5), F(1, 5), F(1, 2)]]
    row = SiteTorsion((0, 1, 2, 3), (ONE,) * 4, angle=60.0)
    return UnitcellStructure(basis, sites, species_at_sites=[C] * 4, bonds=BondNetwork((), torsions=[row]))


def test_the_signed_torsion_follows_iupac() -> None:
    # viewed along B -> C (+z), A-B along +x turns clockwise by 60 degrees onto C-D: +60
    right = _staggered(Cell([5, 5, 10, 90, 90, 120]))
    assert right.cell.basis.det().sign() > 0
    assert right.bonds is not None and right.bonds.torsions is not None
    (angle,) = _signed_torsions(right, right.bonds.torsions)
    assert angle == pytest.approx(60.0, abs=1e-9)
    # the same coordinates in the negated basis are the mirror image
    left = _staggered(Cell(-right.cell.basis))
    assert _signed_torsions(left, right.bonds.torsions)[0] == pytest.approx(-60.0, abs=1e-9)
    # the reversed statement has the same angle; a straight end has none
    reversed_row = SiteTorsion((3, 2, 1, 0), (ONE,) * 4, angle=60.0)
    assert _signed_torsions(right, [reversed_row])[0] == pytest.approx(60.0, abs=1e-9)
    flat = UnitcellStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 10]],
        [[0, 0, 0], [F(1, 5), 0, 0], [F(2, 5), 0, 0], [F(2, 5), F(1, 5), 0]],
        species_at_sites=[C] * 4,
    )
    assert _signed_torsions(flat, [SiteTorsion((0, 1, 2, 3), (ONE,) * 4, angle=0.0)]) == [None]


# ---------------------------------------------------------------------------------------------
# statements built from a fixture's bonds, with their angles from its geometry
# ---------------------------------------------------------------------------------------------
def _chains(domain: FundamentalDomainStructure, per: int) -> list[SiteTorsion]:
    """Up to ``per`` statements A-B-C-D with distinct angles per bond orbit B-C, along the network's bonds."""
    assert domain.bonds is not None
    group, sites = domain.spacegroup, domain.wyckoff_sites
    elements = _standard_elements(group)
    references = [_reference(group, site) for site in sites]
    stabilizers = [[elements.operation(e) for e in elements.stabilizer(r)] for r in references]
    neighbours: dict[int, dict[tuple[Any, ...], Any]] = {i: {} for i in range(len(sites))}
    for bond in domain.bonds.pairs:
        g = _op(bond.operation)
        for i, j, op in ((bond.site, bond.partner, g), (bond.partner, bond.site, _inv(g))):
            for h in stabilizers[i]:
                moved = _mul(h, op)
                neighbours[i].setdefault((j, _apply(moved, references[j])), moved)
    rows = []
    for bond in domain.bonds.pairs:
        i, j, g = bond.site, bond.partner, _op(bond.operation)
        B, Cpoint = references[i], _apply(g, references[j])
        found = []
        for (a, A), first in neighbours[i].items():
            for (d, _point), last in neighbours[j].items():
                op = _mul(g, last)
                D = _apply(op, references[d])
                if A != Cpoint and D not in (B, A):
                    found.append(SiteTorsion((a, i, j, d), (_affine(first), ONE, bond.operation, _affine(op)), "t"))
        chosen: dict[float, SiteTorsion] = {}
        for row, angle in zip(found, _signed_torsions(domain, found), strict=True):
            if angle is not None and len(chosen) < per:
                chosen.setdefault(round(angle, 3), replace(row, symbol=None, angle=angle))
        rows.extend(chosen.values())
    return rows


def _with_torsions(domain: Any, per: int = 2) -> FundamentalDomainStructure:
    domain = domain if isinstance(domain, FundamentalDomainStructure) else ASUStructureView(domain).resolve()
    assert domain.bonds is not None
    rows = _chains(domain, per)
    network = replace(domain.bonds, torsions=rows, torsions_source=BondSource(software="torsions"))
    return type(domain)(
        domain.cell, domain.spacegroup, domain.wyckoff_sites, domain.species, domain.transform, bonds=network
    )


def _cell(structure: Any) -> UnitcellStructure:
    return UnitcellStructureView(structure).unview()


@cache
def _sarcosine() -> FundamentalDomainStructure:
    return _with_torsions(httk.core.load(str(FIXTURES / "cod" / "7219102.cif")), per=3)


SWAP = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]


@cache
def _swapped() -> FundamentalDomainStructure:
    """Sarcosine with a and b swapped (a left-handed cell), recognized: its setting transform has determinant -1."""
    swapped = build_supercell(_cell(_sarcosine()), SWAP).structure
    assert swapped.cell.basis.det().sign() < 0
    recognized = recognize_asu(swapped)
    assert recognized.transform.determinant() == -1 and recognized.bonds is not None
    return recognized


FIXTURE_DOMAINS = {
    "sarcosine (Sohncke)": _sarcosine,
    "sarcosine with a and b swapped (transform det -1)": _swapped,
    "sarcosine mirrored (left-handed basis)": cache(lambda: _mirrored(_sarcosine())),  # type: ignore[arg-type]
    "NaCl (centrosymmetric, special positions)": cache(lambda: _with_torsions(ROUND_TRIPS["NaCl"]())),
    "NaCl in a sheared own cell": cache(
        lambda: _with_torsions(ROUND_TRIPS["NaCl, sheared own cell with an origin shift"]())
    ),
    "diamond (non-standard origin choice)": cache(
        lambda: _with_torsions(ROUND_TRIPS["diamond, non-standard origin choice"]())
    ),
    "diopside (C2/c)": cache(lambda: _with_torsions(ROUND_TRIPS["diopside"]())),
}


def _rows(structure: Any) -> tuple[SiteTorsion, ...]:
    assert structure.bonds is not None and structure.bonds.torsions is not None
    return structure.bonds.torsions


def _assert_geometric(structure: Any, source: Any) -> None:
    """Every statement's angle is its signed torsion in ``structure``'s geometry; the magnitudes are the source's."""
    if not isinstance(structure, FundamentalDomainStructure | UnitcellStructure):
        structure = structure.unview()
    rows = _rows(structure)
    assert structure.bonds.torsions_source == BondSource(software="torsions")
    computed = _signed_torsions(structure, rows)
    for row, angle in zip(rows, computed, strict=True):
        assert row.angle is not None and angle is not None
        assert abs(row.angle - angle) < 1e-6 or abs(abs(row.angle - angle) - 360) < 1e-6, (row, angle)
    assert {round(abs(row.angle or 0), 4) for row in rows} == {round(abs(row.angle or 0), 4) for row in _rows(source)}


def test_the_fixtures_hold_their_geometric_torsions() -> None:
    for make in FIXTURE_DOMAINS.values():
        domain = make()
        _assert_geometric(domain, domain)
    # both signs occur, and the left-handed basis really is left-handed
    assert {row.angle > 0 for row in _rows(_sarcosine()) if row.angle} == {True, False}
    assert FIXTURE_DOMAINS["sarcosine mirrored (left-handed basis)"]().cell.basis.det().sign() < 0


# ---------------------------------------------------------------------------------------------
# the normal forms
# ---------------------------------------------------------------------------------------------
def test_the_unit_cell_normal_form() -> None:
    def normal(*rows: SiteTorsion) -> Any:
        sites = [[0, 0, 0], [F(1, 5), 0, 0], [F(1, 5), F(1, 5), 0], [F(1, 5), F(1, 5), F(1, 5)]]
        bonds = {"pairs": [], "torsions": rows}
        return UnitcellStructure([[5, 0, 0], [0, 5, 0], [0, 0, 5]], sites, species_at_sites=[C] * 4, bonds=bonds).bonds

    forward = SiteTorsion((2, 1, 0, 3), (T(0, 1, 0), T(0, 1, 0), T(1, 1, 0), T(0, 0, 0)), "+sc", 70.0)
    (row,) = normal(forward).torsions
    assert row.operations[1] == T(0, 0, 0) and row.sites == (2, 1, 0, 3) and row.angle == 70.0
    backward = SiteTorsion((3, 0, 1, 2), (T(0, 0, 0), T(1, 1, 0), T(0, 1, 0), T(0, 1, 0)), "+sc", 70.0)
    moved = SiteTorsion(
        forward.sites,
        tuple(T(*(op.vector + FracVector((2, -1, 3))).to_fractions()) for op in forward.operations),
        "+sc",
        70.0,
    )
    assert normal(backward) == normal(moved) == normal(forward, backward, moved) == normal(row)
    # the smaller orientation: sites (2, 1, 0, 3) before (3, 0, 1, 2)
    assert normal(backward).torsions[0].sites == (2, 1, 0, 3)
    with pytest.raises(ValueError, match="conflicting torsion statements"):
        normal(forward, replace(backward, angle=71.0, symbol=None))
    for bad, message in (
        (SiteTorsion((0, 1, 2, 9), (ONE,) * 4, "t"), "outside"),
        (SiteTorsion((0, 1, 0, 3), (ONE,) * 4, "t"), "names one atom twice"),
        (SiteTorsion((0, 1, 2, 1), (ONE,) * 4, "t"), "names one atom twice"),
        (
            SiteTorsion((0, 1, 2, 3), (ONE, ONE, ONE, AffineOperation(((0, 1, 0), (1, 0, 0), (0, 0, 1)))), "t"),
            "lattice translation",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            normal(bad)
    assert (
        normal(SiteTorsion((0, 1, 0, 3), (T(1, 0, 0), ONE, ONE, ONE), "t")).torsions is not None
    )  # A is C's translate


def _variants(domain: FundamentalDomainStructure, row: SiteTorsion) -> list[SiteTorsion]:
    """``row`` moved by every group element (mirrored by the improper ones), reversed, and with other coset members."""
    group, sites = domain.spacegroup, domain.wyckoff_sites
    elements = _standard_elements(group)
    stabilizers = [[elements.operation(e) for e in elements.stabilizer(_reference(group, site))] for site in sites]
    variants = []
    for operation in group.symmetry_operations:
        g = _op(AffineOperation(operation.matrix, operation.vector + FracVector((1, 0, -1))))
        ops = [
            _mul(_mul(g, _op(op)), stabilizers[site][-1]) for site, op in zip(row.sites, row.operations, strict=True)
        ]
        assert row.symbol is None and row.angle is not None
        angle = row.angle if operation.determinant() > 0 else -row.angle  # the image under an improper element
        moved = replace(row, operations=tuple(_affine(op) for op in ops), angle=angle)
        variants += [moved, replace(moved, sites=moved.sites[::-1], operations=moved.operations[::-1])]
    return variants


@pytest.mark.parametrize(
    "name", ["NaCl (centrosymmetric, special positions)", "diopside (C2/c)", "sarcosine (Sohncke)"]
)
def test_every_member_of_a_torsion_orbit_has_one_normal_form(name: str) -> None:
    domain = FIXTURE_DOMAINS[name]()
    network = domain.bonds
    assert network is not None and network.torsions is not None
    for row in network.torsions[:3]:
        for variant in _variants(domain, row)[::7]:  # a sample one by one; all of them together below
            rebuilt = type(domain)(
                domain.cell,
                domain.spacegroup,
                domain.wyckoff_sites,
                domain.species,
                bonds=replace(network, torsions=[variant]),
            )
            assert rebuilt.bonds is not None and rebuilt.bonds.torsions == (row,)
    # all members together collapse to the statements
    every = [variant for row in network.torsions for variant in _variants(domain, row)]
    rebuilt = type(domain)(
        domain.cell, domain.spacegroup, domain.wyckoff_sites, domain.species, bonds=replace(network, torsions=every)
    )
    assert rebuilt.bonds == network
    with pytest.raises(ValueError, match="conflicting torsion statements"):
        type(domain)(
            domain.cell,
            domain.spacegroup,
            domain.wyckoff_sites,
            domain.species,
            bonds=replace(network, torsions=[*network.torsions, replace(every[-1], angle=1.0, symbol=None)]),
        )


def test_a_statement_that_is_its_own_mirror_image_is_unsigned() -> None:
    nacl = FIXTURE_DOMAINS["NaCl (centrosymmetric, special positions)"]()
    rows = {row.angle: row for row in _rows(nacl)}
    assert set(rows) == {90.0, 180.0}
    domain = type(nacl)(nacl.cell, nacl.spacegroup, nacl.wyckoff_sites, nacl.species)

    def normal(row: SiteTorsion) -> SiteTorsion:
        (normalized,) = _rows(
            type(domain)(
                domain.cell,
                domain.spacegroup,
                domain.wyckoff_sites,
                domain.species,
                bonds={"pairs": [], "torsions": [row]},
            )
        )
        return normalized

    # the planar zigzag lies in a mirror plane: any stated sign is not a property of the orbit
    planar = rows[180.0]
    assert normal(replace(planar, symbol="+sc", angle=60.0)) == replace(planar, symbol="sc", angle=60.0)
    assert normal(replace(planar, symbol="-ac", angle=-100.0)) == replace(planar, symbol="ac", angle=100.0)
    # the 90-degree statement has no improper self-map: it keeps a sign
    turned = normal(replace(rows[90.0], symbol="+sc", angle=60.0))
    assert (turned.symbol, turned.angle) in {("+sc", 60.0), ("-sc", -60.0)}


def test_a_unit_cell_and_its_p1_bridge_share_one_normal_form() -> None:
    for name in ("sarcosine (Sohncke)", "NaCl (centrosymmetric, special positions)"):
        cell = _cell(FIXTURE_DOMAINS[name]())
        bridge = _cell_domain(cell)
        assert bridge.bonds is not None and bridge.bonds.torsions == _rows(cell)


def test_a_domain_statement_is_checked_against_its_group() -> None:
    nacl = FIXTURE_DOMAINS["NaCl (centrosymmetric, special positions)"]()
    row = _rows(nacl)[0]

    def build(value: SiteTorsion) -> Any:
        return type(nacl)(
            nacl.cell, nacl.spacegroup, nacl.wyckoff_sites, nacl.species, bonds={"pairs": [], "torsions": [value]}
        )

    with pytest.raises(ValueError, match="outside"):
        build(replace(row, sites=(0, 1, 0, 5)))
    with pytest.raises(ValueError, match="not an element"):
        build(replace(row, operations=(*row.operations[:3], T(F(1, 3), 0, 0))))
    with pytest.raises(ValueError, match="names one atom twice"):
        build(replace(row, sites=row.sites[:3] + (row.sites[1],), operations=row.operations[:3] + (row.operations[1],)))


# ---------------------------------------------------------------------------------------------
# networks without torsions are unchanged; torsions order structures after the shapes
# ---------------------------------------------------------------------------------------------
def test_torsions_take_part_in_equality_and_the_ordering_key_only_when_present() -> None:
    domain = _sarcosine()
    network = domain.bonds
    assert network is not None
    plain = domain._with_bonds(replace(network, torsions=None, torsions_source=None))
    key = _bonds_key(plain)
    assert all(entry[0] != -2 for entry in key)  # type: ignore[index]
    assert _bonds_key(domain)[: len(key)] == key and _bonds_key(domain)[-1][0] == -2  # type: ignore[index]
    assert plain != domain and "torsions" not in repr(plain)
    assert domain._with_bonds(replace(network, torsions_source=None)) == domain  # provenance only
    other = replace(network, torsions=(replace(network.torsions[0], uncertainty=0.5),))  # type: ignore[index]
    assert domain._with_bonds(other) != domain and _bonds_key(domain._with_bonds(other)) != _bonds_key(domain)


# ---------------------------------------------------------------------------------------------
# carrying: the sign suite
# ---------------------------------------------------------------------------------------------
SKEW = [[1, 1, 0], [0, 1, 0], [1, 0, 2]]
SUBGROUPS = {"NaCl (centrosymmetric, special positions)": 123, "diopside (C2/c)": 2}


def _cells(domain: Any) -> list[Any]:
    cell = _cell(domain)
    supercell = build_supercell(cell, SKEW)
    back = AffineOperation(supercell.transformation.T(), (0, 0, 0))
    sheared = build_supercell(cell, [[1, 2, 0], [0, 1, 0], [1, 1, 1]]).structure
    reduced = _cell_carried(supercell.structure, cell._with_bonds(None), back)
    assert reduced == cell
    return [
        cell,
        supercell.structure,
        reduced,
        niggli_reduced(sheared).structure,
        build_supercell(domain, SWAP).structure,
    ]


def _recognized(domain: Any) -> list[Any]:
    cell = _cell(domain)
    conventional = conventional_cell(cell)
    return [
        recognize_asu(cell),
        ASUStructureView(cell).resolve(),
        conventional.asu,
        conventional.structure,
        primitive_cell(cell).structure,
    ]


def _handed(domain: Any) -> list[Any]:
    mirrored = _mirrored(domain)
    return [mirrored, _right_handed(mirrored), _right_handed(domain), _mirrored(mirrored)]


def _subgroups(domain: Any, name: str) -> list[Any]:
    results = [represent_like(domain, _diamond("227:2"))] if name.startswith("diamond") else []
    if name in SUBGROUPS:
        results.append(subgroup_representation(domain, SUBGROUPS[name]).asu)
    return results


def _canonical(domain: Any) -> list[Any]:
    return [canonicalize(_cell(domain)), canonicalize(domain), canonical_asu_protostructure(domain)]


ROUTES = {
    "unit cells, supercells, reductions": lambda domain, name: _cells(domain),
    "recognition, conventional and primitive cells": lambda domain, name: _recognized(domain),
    "mirror images and right-handed re-descriptions": lambda domain, name: _handed(domain),
    "settings and subgroups": _subgroups,
    "canonicalizers": lambda domain, name: _canonical(domain),
}
# the canonicalizers are the slow route: by default only on the Sohncke and the monoclinic fixtures
SLOW = {("canonicalizers", name) for name in FIXTURE_DOMAINS if name not in ("sarcosine (Sohncke)", "diopside (C2/c)")}


@pytest.mark.parametrize(
    ("name", "route"),
    [
        pytest.param(name, route, marks=[pytest.mark.extended] if (route, name) in SLOW else [])
        for name in FIXTURE_DOMAINS
        for route in ROUTES
    ],
)
def test_carried_torsions_keep_their_geometric_sign(name: str, route: str) -> None:
    domain = FIXTURE_DOMAINS[name]()
    for result in ROUTES[route](domain, name):
        _assert_geometric(result, domain)


def test_the_snap_route_keeps_signs() -> None:
    """Recognition matches moved atoms to the recognized ones (``_matched``): the signs and classes survive."""
    cell = _cell(_sarcosine())
    rows = cell.sites.reduced_coords.to_fractions()
    moved = [[F(x) + F((-1) ** (n + k), 20000) for k, x in enumerate(row)] for n, row in enumerate(rows)]
    shaken = UnitcellStructure(cell.cell, moved, cell.species, cell.species_at_sites, bonds=cell.bonds)
    recognized = recognize_asu(shaken, tolerance=0.01)
    assert recognized.spacegroup.it_number == 19 and recognized.cell == cell.cell
    carried = _rows(recognized)
    for row, angle in zip(carried, _signed_torsions(recognized, carried), strict=True):
        assert row.angle is not None and angle is not None
        assert (row.angle > 0) == (angle > 0) and abs(row.angle - angle) < 0.5
    assert {round(abs(row.angle or 0), 4) for row in carried} == {
        round(abs(row.angle or 0), 4) for row in _rows(_sarcosine())
    }


def test_a_torsion_conflict_on_carrying_drops_only_the_torsions(caplog: pytest.LogCaptureFixture) -> None:
    cell = _cell(FIXTURE_DOMAINS["diopside (C2/c)"]())
    network = cell.bonds
    assert network is not None and network.torsions is not None and network.shapes is not None
    changed = network.torsions[0]
    other = 1.0 if changed.angle != 1.0 else 2.0
    altered = replace(network, torsions=(replace(changed, angle=other, symbol=None), *network.torsions[1:]))
    broken = cell._with_bonds(altered)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        recognized = recognize_asu(broken)
        assert recognized.bonds is not None and recognized.bonds.torsions is None
        assert recognized.bonds.torsions_source is None and recognized.bonds.shapes is not None
        assert recognized.bonds.pairs == recognize_asu(cell).bonds.pairs  # type: ignore[union-attr]
    assert any(
        "torsions dropped: torsion conflict" in record.getMessage() and getattr(record, "context", None) == "bonds"
        for record in caplog.records
    )


def test_a_statement_need_not_name_a_bond() -> None:
    sites = [WyckoffSite("a", (0, 0, 0), "C"), WyckoffSite("a", (F(1, 5), 0, 0), "C")]
    row = SiteTorsion((0, 1, 0, 1), (ONE, ONE, T(0, 1, 0), T(0, 1, 0)), "t", 180.0)
    domain = ASUStructure([[6, 0, 0], [0, 4, 0], [0, 0, 5]], 1, sites, [C], bonds=BondNetwork((), torsions=[row]))
    assert _rows(domain) == (row,) and _signed_torsions(domain, [row]) == [180.0]
    assert _rows(_cell(domain)) == (row,)


def _coincident_chain(metals: tuple[Species, Species]) -> UnitcellStructure:
    """Two unbonded metals at the origin, a bonded C chain beside them, and a torsion statement through one metal."""
    sites = [[0, 0, 0], [0, 0, 0], [F(1, 4), 0, 0], [F(1, 4), F(1, 4), 0], [F(1, 4), F(1, 4), F(1, 4)]]
    pairs = (SiteBond(2, 3, ONE), SiteBond(3, 4, ONE))
    row = SiteTorsion((0, 2, 3, 4), (ONE,) * 4, "t")
    network = BondNetwork(
        pairs, (None, None, None, "A2", None), SOURCE, torsions=[row], torsions_source=BondSource(software="t")
    )
    return UnitcellStructure(
        [[8, 0, 0], [0, 8, 0], [0, 0, 8]], sites, species_at_sites=[*metals, C, C, C], bonds=network
    )


def test_a_torsion_atom_on_an_unbonded_coincident_site_drops_only_the_torsions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Plan amendment R12: torsion atoms may be coincident sites no bond or shape names (pairs and shapes stay)."""
    fe, mn = Species("Fe", ("Fe",), (1,)), Species("Mn", ("Mn",), (1,))
    source = _coincident_chain((fe, mn))
    assert source.bonds is not None
    # different species, renamed on the way (as the protonet class structure does): the metals cannot be told apart
    renamed = (Species("A", ("Fe",), (1,)), Species("B", ("Mn",), (1,)))
    target = _cell_domain(_coincident_chain(renamed), network=False)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        network = _matched(_cell_domain(source), target, ONE)
        # the public supercell route: two coincident atoms of one species
        doubled = build_supercell(_coincident_chain((fe, fe)), [[2, 0, 0], [0, 1, 0], [0, 0, 1]]).structure
    for carried in (network, doubled.bonds):
        assert carried is not None and carried.torsions is None and carried.torsions_source is None
        assert carried.pairs and carried.shapes is not None and carried.source == SOURCE
    assert network is not None and network.pairs == source.bonds.pairs and network.shapes == source.bonds.shapes
    messages = [r.getMessage() for r in caplog.records if getattr(r, "context", None) == "bonds"]
    assert sum("torsions dropped: coincident atoms" in message for message in messages) == 2
    assert not any(message.startswith("bond network dropped") for message in messages)
