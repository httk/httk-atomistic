"""Carrying the bond network through operations (protonet follow-up 7, part B, packet P2).

Every new carry row of the carry table: a unit cell's network through recognition and the
canonicalizers, supercells, cell re-descriptions and reductions, ``rationalized`` and
``without_charges``, and subgroup descent with the paths built on it; the drops that recognition
and the bridges log; and the protonet labels of a unit cell with a network. Networks are compared
through an exact site mapping (or an isometry), never by index alone.
"""

import logging
from fractions import Fraction as F
from typing import Any

import pytest
from httk.core import FracVector
from test_bond_network import (
    LOGGER,
    ROUND_TRIPS,
    SHAPES_SOURCE,
    SOURCE,
    _assigned,
    _atom_pairs,
    _coincident,
    _moved,
    _pair,
    _shape_map,
)
from test_bonding import CL, NA, _rocksalt
from test_structure_bonds import _closure, _diamond, _isometries, _moved_pairs, _own_pairs, _same_network
from test_structure_shapes import _b8

from httk.atomistic import (
    AffineOperation,
    ASUStructureView,
    BondNetwork,
    ProtonetstructureView,
    SiteBond,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    build_supercell,
    canonical_asu_protostructure,
    canonical_asu_protostructure_assignments,
    canonical_protonetstructure,
    canonicalize,
    conventional_cell,
    cubic_supercell,
    distance_bonds,
    niggli_reduced,
    orthogonal_supercell,
    primitive_cell,
    recognize_asu,
)
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.structure.asu import _op
from httk.atomistic.models.structure.bonds import _cell_carried, _cosets, _pair_key
from httk.atomistic.symmetry.canonical_protostructure import _bonded_domain
from httk.atomistic.symmetry.lift import rerepresent
from httk.atomistic.symmetry.paths import (
    canonicalize_full,
    common_subgroup_representation,
    interpolate_structures,
    list_representations,
    represent_like,
)
from httk.atomistic.symmetry.subgroups import subgroup_representation

CASES = {
    "NaCl": lambda: _assigned(ROUND_TRIPS["NaCl"]()),
    "NaCl, sheared own cell with an origin shift": lambda: _assigned(
        ROUND_TRIPS["NaCl, sheared own cell with an origin shift"]()
    ),
    "diopside": lambda: _assigned(_b8("diopside")),
}


def _cell(domain: Any) -> UnitcellStructure:
    return UnitcellStructureView(domain).unview()


def _sources(structure: Any) -> tuple[Any, Any]:
    return structure.bonds.source, structure.bonds.shapes_source


def _assert_cell_carried(source: Any, result: Any, back: AffineOperation) -> None:
    """``result``'s network is ``source``'s, atom for atom through ``back`` (result -> source coordinates).

    Both atoms of every result bond and its lattice translates, mapped back, are a bond of the
    source (modulo its lattice), every source bond is reached, and the result has one bond per
    source bond and copy of the cell; every atom has its source atom's shape; the sources are kept.
    """
    assert result.bonds is not None and _sources(result) == _sources(source)
    names = result.species_at_sites
    own = [FracVector(row) for row in result.sites.reduced_coords.to_fractions()]
    rows = [back.apply(row).to_fractions() for row in own]
    shifts = _cosets(_op(back))  # the result's lattice translations that the source lattice lacks
    pairs = {
        _pair_key(
            tuple(x + c for x, c in zip(rows[b.site], shift, strict=True)),
            tuple(
                x + c for x, c in zip(back.apply(b.operation.apply(own[b.partner])).to_fractions(), shift, strict=True)
            ),
            (names[b.site], names[b.partner]),
        )
        for b in result.bonds.pairs
        for shift in shifts
    }
    assert pairs == _atom_pairs(source)
    copies = F(len(names), len(source.species_at_sites))
    assert len(result.bonds.pairs) == len(source.bonds.pairs) * copies
    symbols = result.bonds.shapes or (None,) * len(names)
    shapes = _shape_map(source)
    assert all(
        shapes[(*(F(x) % 1 for x in row), name)] == symbol
        for row, name, symbol in zip(rows, names, symbols, strict=True)
    )


def _atom_shapes(structure: Any) -> dict[tuple[Any, ...], str | None]:
    """Every atom of a domain's unit cell (chemistry, own coordinates modulo one) with its assigned shape."""
    view = UnitcellStructureView(structure)
    assert view.bonds is not None
    chemistry = {species.name: species.chemical_symbols for species in view.species}
    symbols = view.bonds.shapes or (None,) * len(view.species_at_sites)
    rows = view.sites.reduced_coords.to_fractions()
    return {
        (chemistry[name], *(F(x) % 1 for x in row)): symbol
        for name, row, symbol in zip(view.species_at_sites, rows, symbols, strict=True)
    }


def _moved_shapes(shapes: dict[tuple[Any, ...], Any], A: FracVector, t: tuple[F, ...]) -> dict[tuple[Any, ...], Any]:
    """Map atoms by ``f -> A f + t`` (with the translations the other lattice adds), keyed modulo one."""
    cosets = _closure([tuple(A.T()[k].to_fractions()) for k in range(3)])
    moved = {}
    for (chemistry, *point), symbol in shapes.items():
        image = (FracVector(point) * A.T() + FracVector(t)).to_fractions()
        for c in cosets:
            moved[(chemistry, *((x + y) % 1 for x, y in zip(image, c, strict=True)))] = symbol
    return moved


def _maps_onto(source: Any, result: Any) -> bool:
    """Whether one isometry from ``source`` onto ``result`` maps the bond pairs both ways and every atom's shape."""
    source_pairs, result_pairs = _own_pairs(source), _own_pairs(result)
    source_shapes, result_shapes = _atom_shapes(source), _atom_shapes(result)
    for A, t in _isometries(source, result):
        back = tuple((-(FracVector(t) * A.inv().T())).to_fractions())
        if (
            _moved_pairs(source_pairs, A, t) == result_pairs
            and _moved_pairs(result_pairs, A.inv(), back) == source_pairs
            and _moved_shapes(source_shapes, A, t) == result_shapes
        ):
            return True
    return False


def _assert_domain_carried(domain: Any, result: Any) -> None:
    """``result`` (a fundamental domain) has ``domain``'s network, atom for atom through an isometry, and its sources.

    The isometry search runs from the larger cell to the smaller one, so both directions are tried.
    """
    assert result.bonds is not None and _sources(result) == _sources(domain)
    assert _maps_onto(domain, result) or _maps_onto(result, domain)


# ---------------------------------------------------------------------------------------------
# unit cell -> domain by recognition
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CASES))
def test_recognition_carries_a_unit_cell_network(name: str) -> None:
    domain = CASES[name]()
    cell = _cell(domain)
    for recognized in (recognize_asu(cell), ASUStructureView(cell).resolve(), ASUStructureView(cell)):
        _assert_domain_carried(domain, recognized)
        # recognition keeps the input frame: the same atoms, the same pairs, the same shapes
        again = _cell(recognized)
        assert again.cell == cell.cell
        assert _atom_pairs(again) == _atom_pairs(cell) and _shape_map(again) == _shape_map(cell)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_standard_and_primitive_cells_of_a_unit_cell_carry_its_network(name: str) -> None:
    domain = CASES[name]()
    cell = _cell(domain)
    conventional = conventional_cell(cell)
    _assert_domain_carried(domain, conventional.asu)
    _assert_cell_carried(cell, conventional.structure, conventional.transform.operation)
    primitive = primitive_cell(cell)
    _assert_cell_carried(cell, primitive.structure, _primitive_back(primitive))


def _primitive_back(primitive: Any) -> AffineOperation:
    """The map from a primitive cell's coordinates to its input's (recognition keeps the input frame)."""
    return primitive.conventional.transform.operation * AffineOperation(primitive.transform.T(), (0, 0, 0))


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_canonicalizers_carry_a_unit_cell_network(name: str) -> None:
    domain = CASES[name]()
    cell = _cell(domain)
    canonical = canonicalize(cell)
    _assert_domain_carried(domain, canonical)
    # the unit cell and its domain canonicalize to one value, network and sources included
    assert canonical == canonicalize(domain) and _sources(canonical) == _sources(canonicalize(domain))
    protostructure = canonical_asu_protostructure(cell)
    assert protostructure == canonical_asu_protostructure(domain) and _sources(protostructure) == _sources(domain)
    for assignment in canonical_asu_protostructure_assignments(cell):
        assert assignment.bonds is not None and _sources(assignment) == _sources(domain)


def test_a_unit_cell_with_coincident_sites_carries_through_recognition_but_not_the_canonical_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cell = _coincident()  # Fe and Mn at one point: told apart by their species
    recognized = recognize_asu(cell, tolerance=0.01)
    assert recognized.bonds is not None and recognized.bonds.source == SOURCE
    names = [site.species for site in recognized.wyckoff_sites]
    bonded = {tuple(sorted((names[b.site], names[b.partner]))) for b in recognized.bonds.pairs}
    assert bonded == {("Fe", "O"), ("Mn", "O")}  # (the ASU itself cannot be expanded: its sites coincide)
    # the canonical frames rename species, so they cannot tell the two apart: dropped, never raised
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert _bonded_domain(cell) is None
    assert any("the canonical frames rename species" in record.getMessage() for record in caplog.records)
    assert _bonded_domain(_cell(CASES["NaCl"]())) is not None


# ---------------------------------------------------------------------------------------------
# recognition drops what its symmetry does not keep
# ---------------------------------------------------------------------------------------------
def _nacl_cell() -> UnitcellStructure:
    return _cell(CASES["NaCl"]())


def _drop_logged(caplog: pytest.LogCaptureFixture, reason: str) -> bool:
    return any(
        reason in record.getMessage() and getattr(record, "context", None) == "bonds" for record in caplog.records
    )


def test_recognition_drops_a_network_that_is_not_invariant(caplog: pytest.LogCaptureFixture) -> None:
    cell = _nacl_cell()
    assert cell.bonds is not None
    removed = cell._with_bonds(BondNetwork(cell.bonds.pairs[1:], source=SOURCE))  # one atom's bond fewer
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert recognize_asu(removed).bonds is None
        assert ASUStructureView(removed).bonds is None
        assert canonicalize(removed).bonds is None
    assert _drop_logged(caplog, "not invariant under the recognized symmetry")
    with pytest.raises(ValueError, match="is not invariant under its recognized symmetry"):
        ProtonetstructureView(removed).unview()


def test_recognition_drops_shapes_that_differ_within_an_orbit(caplog: pytest.LogCaptureFixture) -> None:
    cell = _nacl_cell()
    assert cell.bonds is not None and cell.bonds.shapes is not None
    differing = ("TPR6", *cell.bonds.shapes[1:])  # one Na atom a trigonal prism, its orbit octahedra
    changed = UnitcellStructure(
        cell.cell,
        cell.sites,
        cell.species,
        cell.species_at_sites,
        bonds=BondNetwork(cell.bonds.pairs, differing, SOURCE, SHAPES_SOURCE),
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert recognize_asu(changed).bonds is None
        assert canonicalize(changed).bonds is None
    assert _drop_logged(caplog, "shapes 'OC6' and 'TPR6'") or _drop_logged(caplog, "shapes 'TPR6' and 'OC6'")


# ---------------------------------------------------------------------------------------------
# supercells, cell re-descriptions and reductions of a unit cell
# ---------------------------------------------------------------------------------------------
SKEW = [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


@pytest.mark.parametrize("name", sorted(CASES))
def test_supercells_carry_the_network(name: str) -> None:
    domain = CASES[name]()
    cell = _cell(domain)
    for source in (cell, domain):  # any input is presented as a unit cell first
        result = build_supercell(source, SKEW)
        back = AffineOperation(result.transformation.T(), (0, 0, 0))
        _assert_cell_carried(cell, result.structure, back)


def test_shape_searched_supercells_carry_the_network() -> None:
    cell = primitive_cell(_nacl_cell()).structure
    for result in (orthogonal_supercell(cell, 4), cubic_supercell(cell, 4)):
        _assert_cell_carried(cell, result.structure, AffineOperation(result.transformation.T(), (0, 0, 0)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_supercell_and_its_reduction_round_trip_exactly(name: str) -> None:
    cell = _cell(CASES[name]())
    supercell = build_supercell(cell, SKEW)
    back = AffineOperation(supercell.transformation.T(), (0, 0, 0))
    reduced = _cell_carried(supercell.structure, cell._with_bonds(None), back)
    assert reduced == cell and _sources(reduced) == _sources(cell)
    # the public reduction: the primitive cell of the supercell carries the cell's network
    primitive = primitive_cell(supercell.structure)
    _assert_cell_carried(cell, primitive.structure, back * _primitive_back(primitive))


def test_a_reduction_that_gives_up_a_translation_the_bonds_lack_drops_them(caplog: pytest.LogCaptureFixture) -> None:
    # a chain bonded only within each doubled cell: halving the cell does not keep that network
    pair = _pair()
    doubled = build_supercell(pair, [[2, 0, 0], [0, 1, 0], [0, 0, 1]]).structure
    network = BondNetwork((SiteBond(0, 2, AffineOperation(FracVector.eye((3, 3)), (0, 0, 0))),), source=SOURCE)
    broken = doubled._with_bonds(network)
    halve = AffineOperation(((2, 0, 0), (0, 1, 0), (0, 0, 1)))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert _cell_carried(broken, pair, halve).bonds is None
    assert _drop_logged(caplog, "not invariant")


@pytest.mark.parametrize("name", sorted(CASES))
def test_niggli_reduction_carries_the_network(name: str) -> None:
    cell = _cell(CASES[name]())
    shear = build_supercell(cell, [[1, 2, 0], [0, 1, 0], [1, 1, 1]])  # the same lattice, sheared
    result = niggli_reduced(shear.structure)
    back = AffineOperation(shear.transformation.T(), (0, 0, 0)) * AffineOperation(result.transform.T(), (0, 0, 0))
    _assert_cell_carried(cell, result.structure, back)


def test_rationalized_carries_the_network_unchanged() -> None:
    pairs = tuple(SiteBond(0, 1, AffineOperation(FracVector.eye((3, 3)), (n, 0, 0))) for n in (0, -1))
    rough = UnitcellStructure(
        [[6, 0, 0], [0, 4, 0], [0, 0, 5]],
        [[0, 0, 0], [F(4999, 10000), 0, 0]],
        species_at_sites=[NA, CL],
        bonds=BondNetwork(pairs, ("L2", None), SOURCE, SHAPES_SOURCE),
    )
    result = rough.rationalized(coordinate_tolerance=F(1, 1000))
    assert result.sites.reduced_coords.to_fractions()[1][0] == F(1, 2)
    assert result.bonds == rough.bonds and _sources(result) == _sources(rough)


def test_without_charges_carries_the_network() -> None:
    charged = [Species("Na", ("Na",), (1,), charges=(1,)), Species("Cl", ("Cl",), (1,), charges=(-1,))]
    domain = _rocksalt(F(564, 100))
    domain = type(domain)(domain.cell, domain.spacegroup, domain.wyckoff_sites, charged, bonds=CASES["NaCl"]().bonds)
    cell = _cell(domain)
    for structure in (cell, domain):
        result = structure.without_charges()
        assert all(species.charges is None for species in result.species)
        # the same atoms in the same order, so the network compares index for index
        assert result.cell == cell.cell and result.sites == cell.sites
        assert result.species_at_sites == cell.species_at_sites
        assert result.bonds == cell.bonds and _sources(result) == _sources(cell)


# ---------------------------------------------------------------------------------------------
# subgroup descent and the paths built on it
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("name", "target"), [("NaCl", 123), ("NaCl", 166), ("NaCl", 1), ("diopside", 2)])
def test_subgroup_descent_carries_the_network(name: str, target: int) -> None:
    domain = CASES[name]()
    for result in (
        subgroup_representation(domain, target).asu,
        rerepresent(domain, target),
        list_representations(domain, target)[0],
        canonicalize_full(domain, target),
    ):
        assert result.spacegroup.it_number == target
        _assert_domain_carried(domain, result)


def test_represent_like_carries_into_a_reference_setting() -> None:
    first, second = _diamond("227:1"), _diamond("227:2")  # origin choices 1 (not standard) and 2
    for structure, reference in ((second, first), (first, second)):
        result = represent_like(structure, reference)
        assert result.spacegroup == reference.spacegroup
        assert result.bonds is not None and _same_network(structure, result)
    nacl, other = CASES["NaCl"](), _assigned(ROUND_TRIPS["NaCl, sheared own cell with an origin shift"]())
    _assert_domain_carried(nacl, represent_like(nacl, other))
    common = common_subgroup_representation(nacl, _assigned(ROUND_TRIPS["NaCl"]()))
    _assert_domain_carried(nacl, common.first)
    _assert_domain_carried(nacl, common.second)


def test_lifts_and_interpolated_frames_still_drop_the_network() -> None:
    nacl = CASES["NaCl"]()
    child = subgroup_representation(nacl, 123).asu
    assert child.bonds is not None
    assert rerepresent(child, 225).bonds is None  # a lift: the documented deviation
    larger = _assigned(distance_bonds(_rocksalt(F(600, 100))))
    frames = interpolate_structures(nacl, larger, steps=3).frames
    assert [frame.bonds is not None for frame in frames] == [True, False, True]


# ---------------------------------------------------------------------------------------------
# protonet labels of a unit cell
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["NaCl", "diopside", "K2AgSbS4", "AgO"])
def test_a_unit_cell_with_a_network_has_its_domain_s_protonet_label(name: str) -> None:
    domain = ROUND_TRIPS["NaCl"]() if name == "NaCl" else _b8(name)
    cell = _cell(domain)
    label = ProtonetstructureView(cell).unview()
    assert label == ProtonetstructureView(domain).unview()
    assert ProtonetstructureView(cell).unview().label == label.label


def test_a_unit_cell_without_a_network_has_no_protonet_label() -> None:
    plain = _nacl_cell()._with_bonds(None)
    with pytest.raises((TypeError, ValueError)):
        ProtonetstructureView(plain).unview()


# ---------------------------------------------------------------------------------------------
# exact re-descriptions of a unit cell, and the moments branch of conventional_cell
# ---------------------------------------------------------------------------------------------
def _variants(cell: UnitcellStructure) -> dict[str, UnitcellStructure]:
    """``cell`` as given, sheared, origin-shifted and with its sites permuted, its shapes and sources kept.

    Every variant is rebuilt the same way, so they share their (unstated) precision.
    """
    assert cell.bonds is not None
    n = len(cell.species_at_sites)
    order = [*range(1, n, 2), *range(0, n, 2)]
    moves = {
        "as given": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 0, 0), None),
        "sheared": ([[1, 1, 0], [0, 1, 0], [0, 1, 1]], (0, 0, 0), None),
        "origin-shifted": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (F(1, 8), F(1, 4), F(1, 3)), None),
        "permuted": ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 0, 0), order),
    }
    variants = {}
    for name, (W, p, permutation) in moves.items():
        moved = _moved(cell, W, p, permutation)
        assert moved.bonds is not None
        sequence = list(range(n)) if permutation is None else permutation
        shapes = None if cell.bonds.shapes is None else tuple(cell.bonds.shapes[k] for k in sequence)
        network = BondNetwork(moved.bonds.pairs, shapes, cell.bonds.source, cell.bonds.shapes_source)
        variants[name] = moved._with_bonds(network)
    return variants


@pytest.mark.parametrize("name", ["NaCl", "AgO", "CuCl2 2.6"])
def test_re_described_unit_cells_carry_through_recognition_and_the_canonicalizers(name: str) -> None:
    domain = CASES["NaCl"]() if name == "NaCl" else _assigned(_b8(name))
    variants = _variants(_cell(domain))
    canonical = canonicalize(variants["as given"])
    label = canonical_protonetstructure(ProtonetstructureView(domain))
    for variant in variants.values():
        _assert_domain_carried(domain, recognize_asu(variant))
        result = canonicalize(variant)
        assert result == canonical and _sources(result) == _sources(canonical)
        # the view labels the description recognition yields; the canonical label is the crystal's
        assert canonical_protonetstructure(ProtonetstructureView(variant)) == label


def test_conventional_cell_carries_the_network_with_site_moments() -> None:
    cell = _nacl_cell()
    moments = CollinearSiteMoments([1 if name == "Na" else 0 for name in cell.species_at_sites])
    magnetic = UnitcellStructure(
        cell.cell, cell.sites, cell.species, cell.species_at_sites, site_moments=moments, bonds=cell.bonds
    )
    sheared = build_supercell(magnetic, [[1, 2, 0], [0, 1, 0], [1, 1, 1]])
    result = conventional_cell(sheared.structure)
    assert result.structure.site_moments is not None and result.asu.bonds is not None
    back = AffineOperation(sheared.transformation.T(), (0, 0, 0)) * result.transform.operation
    _assert_cell_carried(cell, result.structure, back)


# ---------------------------------------------------------------------------------------------
# recognition takes the identity frame only when the metric admits the group (review M1)
# ---------------------------------------------------------------------------------------------
def test_a_sheared_cell_is_recognized_in_a_frame_its_metric_admits() -> None:
    cell = _nacl_cell()
    sheared = build_supercell(cell, [[1, 2, 0], [0, 1, 0], [1, 1, 1]]).structure  # a cubic lattice, sheared basis
    recognized = recognize_asu(sheared)
    assert recognized.spacegroup.it_number == 225 and not recognized.transform.operation.is_identity()
    _assert_domain_carried(CASES["NaCl"](), recognized)
    assert conventional_cell(sheared).structure.cell.metric() == cell.cell.metric()
    assert ProtonetstructureView(sheared).unview() == ProtonetstructureView(cell).unview()


def test_a_unit_cell_that_is_not_periodic_in_three_dimensions_has_no_protonet_label() -> None:
    from httk.atomistic import Cell

    flat = _pair([SiteBond(0, 1, AffineOperation(FracVector.eye((3, 3)), (0, 0, 0)))])
    slab = UnitcellStructure(
        Cell(flat.cell.basis, periodicity=(True, True, False)),
        flat.sites,
        flat.species,
        flat.species_at_sites,
        bonds=flat.bonds,
    )
    with pytest.raises(ValueError, match="label a molecular fundamental domain"):
        ProtonetstructureView(slab).unview()
