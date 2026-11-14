"""Supplied and baseline bond shapes on fundamental domains: counts, validation, carrying, storage."""

import contextlib
import math
import os
import random
import uuid
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import pytest
from _protonet_support import load_structure, raw_rows
from httk.core import FracVector, SurdVector
from httk.core.storage import content_id
from test_protonet_molecular import COVALENT, MOLECULAR
from test_structure_bonds import (
    CARRIERS,
    ONE,
    SI,
    C,
    _chain,
    _diamond,
    _in_standard_setting,
    _nearest_bonds,
    _p1_pair,
    _p21n,
    _p42nbc,
    _primitive_diamond,
    _rhombohedral,
    _same_network,
)

from httk.atomistic import (
    AffineOperation,
    ASUStructure,
    ASUStructureRecord,
    ASUStructureView,
    BondNetwork,
    Cell,
    FundamentalDomainStructure,
    FundamentalDomainStructureRecord,
    Protostructure,
    SiteBond,
    Spacegroup,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    canonical_asu_protostructure,
    recognize_asu,
    subgroup_representation,
)
from httk.atomistic import data as atomistic_data
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.models.structure import asu as asu_module
from httk.atomistic.models.structure.bonds import _bonds_key, _carried, _CoincidentAtoms, _matched, _ShapeConflict
from httk.atomistic.storage.records import _domain_structure_from_record, _domain_structure_record_from_value
from httk.atomistic.symmetry import protonet, shapes
from httk.atomistic.symmetry.canonical_protostructure import _p1bar_primitive_entry
from httk.atomistic.symmetry.shapes import (
    _IDEAL,
    _baseline_shapes,
    _bond_counts,
    _detect,
    _detect_angles,
    _site_shapes,
    _structure_bond_counts,
)

FIXTURES = Path(__file__).parent / "fixtures"
# name -> (COD entry, cutoffs): the B8 cases of protonet follow-up 5
B8 = {
    "AlCl3": ("1535669", {("Al", "Cl"): 2.6}),
    "AgO": ("1509688", {("Ag", "O"): 2.4}),
    "CuCl2 2.6": ("1539945", {("Cu", "Cl"): 2.6}),
    "CuCl2 3.1": ("1539945", {("Cu", "Cl"): 3.1}),
    "K2AgSbS4": ("1509425", {("Ag", "S"): 3.0, ("Sb", "S"): 3.0}),
    "diopside": ("1000008", {("Si", "O"): 2.0, ("Mg", "O"): 2.5, ("Ca", "O"): 2.9}),
}
# the shapes the follow-up 5 brief lists for the B8 cases, by species name
B8_SHAPES = {
    "AlCl3": {"Al+3": {"OC6"}, "Cl-1": {"A2"}},
    "AgO": {"Ag+3": {"SP4"}, "Ag+1": {"L2"}, "O-2": {"TPY3"}},
    "CuCl2 2.6": {"Cu+2": {"SP4"}, "Cl-1": {"A2"}},
    "CuCl2 3.1": {"Cu+2": {"OC6"}, "Cl-1": {"I3"}},
    "K2AgSbS4": {"S-2": {"A2"}, "Ag+1": {"I4", "SS4"}, "Sb+5": {"T4"}, "K+1": {None}},
    "diopside": {"Ca": {"DD8"}, "Mg": {"OC6"}, "Si": {"T4"}, "O": {"T4", "I3", "I4"}},
}
# monoclinic cell changes of the B8 groups (their invariance probe)
CELLS = {
    "1509688": [((-1, 0), (-2, 1)), ((-1, 2), (0, -1)), ((-1, 0), (1, 1))],
    "1000008": [((1, 2), (0, 1)), ((-1, 2), (2, -5)), ((1, 0), (2, 1))],
    "1539945": [((1, 1), (0, 1)), ((1, 0), (2, 1)), ((-1, 1), (2, -3))],
    "1535669": [((1, 1), (0, 1)), ((1, 0), (2, 1)), ((-1, 1), (2, -3))],
}


# ---------------------------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------------------------
def _bonds_from_rows(spacegroup: Spacegroup, rows: list[Any]) -> list[SiteBond]:
    elements = asu_module._standard_elements(spacegroup)
    return [SiteBond(i, j, asu_module._affine(elements.operation(g))) for i, j, g in rows]


@cache
def _cutoff_structure(entry: str, cutoffs: tuple[tuple[tuple[str, str], float], ...] | None) -> ASUStructure:
    """A COD fixture with bonds by element-pair cutoffs (``None``: the ``COVALENT`` table)."""
    path = str(FIXTURES / "cod" / f"{entry}.cif")
    loaded = load_structure(path)
    assert loaded is not None
    it, sites, refs, lattice = loaded
    table = COVALENT if cutoffs is None else {frozenset(pair): value for pair, value in cutoffs}
    rows, _ = raw_rows(protonet._tables(it), sites, refs, lattice, table, reach=3)
    cif = httk.core.load(path)
    return ASUStructure(
        cif.cell,
        cif.spacegroup,
        [WyckoffSite(w.wyckoff, w.free_params, w.species) for w in cif.wyckoff_sites],
        cif.species,
        cif.transform,
        cif.coordinate_precision,
        bonds=_bonds_from_rows(cif.spacegroup, rows),
    )


def _b8(name: str) -> ASUStructure:
    entry, cutoffs = B8[name]
    return _cutoff_structure(entry, tuple(sorted(cutoffs.items())))


def _loop_fixtures() -> list[str]:
    paths = sorted(FIXTURES.rglob("*.cif"))
    return [str(path.relative_to(FIXTURES)) for path in paths if b"_geom_bond_" in path.read_bytes()]


@cache
def _loaded(name: str) -> Any:
    try:
        structure = httk.core.load(str(FIXTURES / name))
    except ValueError:
        return None
    return structure if getattr(structure, "bonds", None) is not None else None


def _with(structure: Any, shapes: Any = None, **changes: Any) -> ASUStructure:
    """The structure rebuilt with other ``shapes`` (and other constructor values; ``bonds`` as pairs)."""
    values = {
        "cell": structure.cell,
        "spacegroup": structure.spacegroup,
        "wyckoff_sites": structure.wyckoff_sites,
        "species": structure.species,
        "transform": structure.transform,
    }
    pairs = changes.pop("bonds", structure.bonds.pairs)
    values.update(changes)
    return ASUStructure(**values, bonds=BondNetwork(pairs, shapes))


def _shapes(structure: Any) -> Any:
    """A structure's assigned shapes, ``None`` without a network or shapes."""
    return None if structure.bonds is None else structure.bonds.shapes


def _irregular(structure: Any) -> tuple[str | None, ...]:
    """A valid supplied symbol for every shaped site: ``I<n>``, which the baseline never gives a regular site.

    Two bonds have no irregular symbol (``I2`` is invalid), so they get ``A2``, which the straight chains do not detect.
    """
    counts = _structure_bond_counts(structure.spacegroup, structure.wyckoff_sites, structure.bonds.pairs)
    symbols = tuple(None if count < 2 else "A2" if count == 2 else f"I{count}" for count in counts)
    return None if all(symbol is None for symbol in symbols) else symbols


# ---------------------------------------------------------------------------------------------
# test-only geometry: neighbours by the stored group's operations (no cosets, no element tables)
# ---------------------------------------------------------------------------------------------
def _neighbour_atoms(structure: Any, s: int) -> set[tuple[int, tuple[F, ...]]]:
    """The atoms ``(site, exact point)`` bonded to the reference atom of site ``s``, stored coordinates."""
    group = structure.spacegroup

    def reference(index: int) -> tuple[F, ...]:
        return asu_module._reference(group, structure.wyckoff_sites[index])

    target = reference(s)
    atoms = set()
    for bond in structure.bonds.pairs:
        ends = [
            (bond.site, reference(bond.site)),
            (bond.partner, asu_module._apply(asu_module._op(bond.operation), reference(bond.partner))),
        ]
        for (site, x), (other, y) in (ends, ends[::-1]):
            if site != s:
                continue
            for g in asu_module._group(group)[0]:
                shift = [a - b for a, b in zip(asu_module._apply(g, x), target, strict=True)]
                if all(value.denominator == 1 for value in shift):
                    atoms.add((other, tuple(p - t for p, t in zip(asu_module._apply(g, y), shift, strict=True))))
    return atoms


def _cartesian(structure: Any, s: int) -> list[list[float]]:
    """Float Cartesian bond vectors of site ``s`` (structures in their own, standard setting)."""
    assert structure.is_standard_setting
    basis = structure.cell.basis.to_floats()
    origin = asu_module._reference(structure.spacegroup, structure.wyckoff_sites[s])
    vectors = []
    for _site, point in _neighbour_atoms(structure, s):
        d = [float(p - q) for p, q in zip(point, origin, strict=True)]
        vectors.append([sum(d[k] * basis[k][axis] for k in range(3)) for axis in range(3)])
    return vectors


@contextlib.contextmanager
def _recorded() -> Iterator[list[tuple[int, tuple[float, ...], tuple[str | None, float | None]]]]:
    """Record every ``(count, sorted angles, result)`` the baseline hands the detector."""
    calls: list[tuple[int, tuple[float, ...], tuple[str | None, float | None]]] = []
    original = shapes._detect_angles

    def record(count: int, angles: Any) -> tuple[str | None, float | None]:
        result = original(count, angles)
        calls.append((count, tuple(angles), result))
        return result

    shapes._detect_angles = record  # type: ignore[assignment]
    try:
        yield calls
    finally:
        shapes._detect_angles = original  # type: ignore[assignment]


def _stars(structure: Any) -> list[tuple[str, int, tuple[float, ...], str | None]]:
    """The sorted multiset of (species, bond count, exact-path angles, symbol) over the measured sites."""
    counts = _structure_bond_counts(structure.spacegroup, structure.wyckoff_sites, structure.bonds.pairs)
    with _recorded() as calls:
        symbols = _baseline_shapes(structure)
    measured = [s for s, count in enumerate(counts) if count >= 2]
    assert len(calls) == len(measured)
    return sorted(
        (structure.wyckoff_sites[s].species, count, angles, symbols[s])
        for s, (count, angles, _result) in zip(measured, calls, strict=True)
    )


# ---------------------------------------------------------------------------------------------
# exact re-descriptions
# ---------------------------------------------------------------------------------------------
def _redescribed(source: Any, action: AffineOperation, *, basis: Any = None, setting: Any = None, seed: int = 0) -> Any:
    """The same crystal under ``action`` (coordinates; basis ``P^-T B`` unless given), sites shuffled."""
    group = source.spacegroup if setting is None else setting
    sites = []
    for site in source.wyckoff_sites:
        point = action.apply(FracVector(asu_module._reference(source.spacegroup, site)))
        found = group.identify_wyckoff(point.normalize())
        assert found is not None
        sites.append(WyckoffSite(found[0].letter, found[1], site.species))
    random.Random(seed).shuffle(sites)
    if basis is None:
        basis = SurdVector(action.matrix.inv().T()) * source.cell.basis
    result = _carried(source, ASUStructure(basis, group, sites, source.species), action)
    assert result.bonds is not None
    return result


def _redescriptions(name: str) -> Iterator[Any]:
    source = _b8(name)
    it = source.spacegroup.it_number
    tables = protonet._tables(it)
    for index, coset in enumerate(tables.cosets[1:6]):  # origin shifts and other normalizer actions
        yield _redescribed(source, coset.action, seed=index)
    for index, X in enumerate(CELLS[B8[name][0]]):  # cells of the monoclinic group
        yield _redescribed(source, protonet._cell(it, X).action, seed=10 + index)
    negate = AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
    yield _redescribed(source, negate, basis=source.cell.basis, seed=20)  # the mirror image
    setting = next(
        Spacegroup(record)
        for record in atomistic_data.spacegroup_settings()
        if record["it_number"] == it and not Spacegroup(record).is_standard_setting
    )
    transform = setting.transform_from_standard
    yield _redescribed(
        source, transform.operation, basis=transform.basis_to_setting(source.cell.basis), setting=setting
    )
    # the standard setting stored with an explicit transform to the non-standard cell
    yield _with(source, cell=transform.basis_to_setting(source.cell.basis), transform=transform)


# ---------------------------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------------------------
def test_the_angle_detector_is_the_reference_detector_after_its_angles() -> None:
    for count, ideal in _IDEAL.items():
        for symbol, vectors in ideal.items():
            angles = shapes._sorted_angles(vectors)
            assert _detect_angles(count, angles) == _detect(vectors) == (symbol, 0.0)
    assert _detect_angles(0, []) == _detect_angles(1, []) == (None, None)
    assert _detect_angles(2, [165.0]) == ("L2", None)
    assert _detect_angles(2, [164.9999996]) == ("L2", None)  # rounded to 1e-6 first
    assert _detect_angles(2, [164.9999994]) == ("A2", None)
    assert _detect_angles(10, [60.0] * 45) == ("I10", None)
    assert _detect_angles(4, [60.0] * 6)[0] == "I4"


# ---------------------------------------------------------------------------------------------
# bond counts
# ---------------------------------------------------------------------------------------------
def _counts(structure: Any) -> tuple[int, ...]:
    return _structure_bond_counts(structure.spacegroup, structure.wyckoff_sites, structure.bonds.pairs)


def test_bond_counts_on_special_positions() -> None:
    def by_species(structure: Any) -> dict[str, set[int]]:
        result: dict[str, set[int]] = {}
        for site, count in zip(structure.wyckoff_sites, _counts(structure), strict=True):
            result.setdefault(site.species, set()).add(count)
        return result

    assert by_species(_b8("CuCl2 2.6"))["Cu+2"] == {4}
    assert by_species(_b8("CuCl2 3.1"))["Cu+2"] == {6}
    assert by_species(_b8("AlCl3"))["Al+3"] == {6}


def test_a_same_site_bond_counts_its_inverse_unless_it_is_its_own() -> None:
    cell = [[5, 0, 0], [0, 6, 0], [0, 0, 7]]
    site = [WyckoffSite("i", (F(1, 8), F(1, 5), F(1, 7)), "C")]
    inversion = AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)), (1, 0, 0))  # g = g^-1
    assert _counts(ASUStructure(cell, 2, site, [C], bonds=[SiteBond(0, 0, inversion)])) == (1,)
    translation = AffineOperation(ONE.matrix, (1, 0, 0))  # g^-1 is another atom
    assert _counts(ASUStructure(cell, 2, site, [C], bonds=[SiteBond(0, 0, translation)])) == (2,)
    assert _counts(
        ASUStructure(cell, 1, [WyckoffSite("a", (0, 0, 0), "C")], [C], bonds=[SiteBond(0, 0, translation)])
    ) == (2,)


def _networks() -> list[Any]:
    loaded = [_loaded(name) for name in _loop_fixtures()]
    cutoff = [_b8(name) for name in B8] + [_cutoff_structure(entry, None) for entry in MOLECULAR]
    return [structure for structure in loaded if structure is not None] + cutoff


def test_bond_counts_equal_the_exact_neighbour_atoms_on_every_fixture_network() -> None:
    networks = _networks()
    assert len(networks) >= 35
    for structure in networks:
        expected = tuple(len(_neighbour_atoms(structure, s)) for s in range(len(structure.wyckoff_sites)))
        assert _counts(structure) == expected


def test_bond_counts_do_not_depend_on_the_stored_setting() -> None:
    for structure in (_p21n(), _rhombohedral(), _diamond("227:1"), _p42nbc("133:1")):
        assert not structure.spacegroup.is_standard_setting
        assert _counts(structure) == _counts(_in_standard_setting(structure))


def test_the_generic_counting_helper_reads_stabilisers_and_rows() -> None:
    tables = protonet._tables(1)
    # one P1 site bonded along a and along b: four atoms
    assert _bond_counts(tables, [[(0, 0, 0, 0)]], [(0, 0, (0, 1, 0, 0)), (0, 0, (0, 0, 1, 0))]) == (4,)
    # the same row twice, once reversed, counts each atom once
    assert _bond_counts(tables, [[(0, 0, 0, 0)]], [(0, 0, (0, 1, 0, 0)), (0, 0, (0, -1, 0, 0))]) == (2,)


# ---------------------------------------------------------------------------------------------
# supplied shapes: validation
# ---------------------------------------------------------------------------------------------
def test_supplied_shapes_are_validated_against_the_bond_counts() -> None:
    structure = _p21n()  # Si: three bonds, O: two
    assert _counts(structure) == (3, 2)
    ok = _with(structure, ("TPY3", "A2"))
    assert _shapes(ok) == ("TPY3", "A2")
    assert _shapes(_with(structure, ("I3", None))) == ("I3", None)
    with pytest.raises(ValueError, match="one symbol or None per Wyckoff site"):
        _with(structure, ("TPY3",))
    with pytest.raises(ValueError, match=r"shapes\[0\]: unknown shape symbol 'XX3'"):
        _with(structure, ("XX3", None))
    with pytest.raises(ValueError, match=r"shapes\[1\]: shape 'T4' is for 4 bonds, but the site has 2"):
        _with(structure, (None, "T4"))
    with pytest.raises(ValueError, match="unknown shape symbol 'I1'"):
        _with(structure, (None, "I1"))
    with pytest.raises(TypeError, match="str or None"):
        _with(structure, (3, None))
    with pytest.raises(TypeError, match="not a str"):
        _with(structure, "A2")
    # a symbol on a site with fewer than two bonds
    chain = _p1_pair(chain=False)  # a dimer: one bond per atom
    assert _counts(chain) == (1, 1)
    with pytest.raises(ValueError, match=r"shapes\[0\]: shape 'L2' given for a site with 1 bond"):
        _with(chain, ("L2", None))
    with pytest.raises(ValueError, match="with 0 bond"):
        _with(chain, ("L2", None), bonds=[])


def test_all_none_shapes_are_no_shapes() -> None:
    structure = _p21n()
    plain = _with(structure, (None, None))
    assert _shapes(plain) is None
    assert plain == structure == _with(structure, None)
    assert "shapes=None" in repr(plain)
    supplied = _with(structure, ("TPY3", None))
    assert supplied != structure
    assert "shapes=('TPY3', None)" in repr(supplied)
    assert _shapes(structure._with_bonds(BondNetwork(structure.bonds.pairs, (None, None)))) is None


# ---------------------------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------------------------
def _coincident(*, bonded: bool, shapes: Any = None) -> ASUStructure:
    """Fe and Mn both on 1a of P m -3 m and O on (1/2, 0, 0); Fe-O, Mn-O, and Fe-Mn when ``bonded``."""
    group = Spacegroup.standard(221)
    found = group.identify_wyckoff(FracVector((F(1, 2), 0, 0)))
    assert found is not None
    sites = [WyckoffSite("a", (), "Fe"), WyckoffSite("a", (), "Mn"), WyckoffSite(found[0].letter, found[1], "O")]
    species = [Species("Fe", ("Fe",), (1,)), Species("Mn", ("Mn",), (1,)), Species("O", ("O",), (1,))]
    oxygen = FracVector(asu_module._reference(group, sites[2]))
    to_oxygen = next(  # the element moving O's reference atom onto (1/2, 0, 0), Fe's neighbour
        AffineOperation(g.matrix, g.vector + (FracVector((F(1, 2), 0, 0)) - g.apply(oxygen)))
        for g in group.symmetry_operations
        if all((x - y).denominator == 1 for x, y in zip(g.apply(oxygen).to_fractions(), (F(1, 2), 0, 0), strict=True))
    )
    bonds = [SiteBond(0, 2, to_oxygen), SiteBond(1, 2, to_oxygen)] + ([SiteBond(0, 1, ONE)] if bonded else [])
    return ASUStructure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], group, sites, species, bonds=BondNetwork(bonds, shapes))


def test_coincident_sites_are_two_atoms() -> None:
    structure = _coincident(bonded=False)
    assert _counts(structure) == (6, 6, 4)  # O sees two Fe and two Mn
    assert _site_shapes(structure)[:2] == ("OC6", "OC6")
    bonded = _coincident(bonded=True)
    assert _counts(bonded) == (7, 7, 4)
    with pytest.raises(ValueError, match="site 0: a zero-length bond has no direction; supply its shape"):
        _site_shapes(bonded)
    supplied = _coincident(bonded=True, shapes=("I7", "I7", None))
    assert _site_shapes(supplied)[:2] == ("I7", "I7")


def test_a_site_at_a_special_point_of_its_letter_is_refused() -> None:
    cell = [[5, 0, 0], [0, 6, 0], [-1, 0, 7]]
    sites = [WyckoffSite("e", (0, 0, 0), "C")]
    bonds = [SiteBond(0, 0, AffineOperation(ONE.matrix, (1, 0, 0)))]
    structure = ASUStructure(cell, 14, sites, [C], bonds=bonds)  # constructible: no shapes asked for
    message = "site 0 lies at a special point of its Wyckoff letter 'e'"
    with pytest.raises(ValueError, match=message):
        _counts(structure)
    with pytest.raises(ValueError, match=message):
        _site_shapes(structure)
    with pytest.raises(ValueError, match=message):
        ASUStructure(cell, 14, sites, [C], bonds=BondNetwork(bonds, ["A2"]))


def test_a_structure_without_a_network_has_no_shapes() -> None:
    diamond = _diamond("227:2")
    with pytest.raises(ValueError, match="without a bond network"):
        _site_shapes(diamond._with_bonds(None))


# ---------------------------------------------------------------------------------------------
# the baseline
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(B8))
def test_the_baseline_gives_the_shapes_of_the_brief(name: str) -> None:
    structure = _b8(name)
    found: dict[str, set[str | None]] = {}
    for site, symbol in zip(structure.wyckoff_sites, _site_shapes(structure), strict=True):
        found.setdefault(site.species, set()).add(symbol)
    assert found == B8_SHAPES[name]


@pytest.mark.parametrize("name", list(B8))
def test_the_exact_baseline_agrees_with_the_reference_detector_on_cartesian_vectors(name: str) -> None:
    structure = _b8(name)
    counts = _counts(structure)
    with _recorded() as calls:
        _baseline_shapes(structure)
    measured = [s for s, count in enumerate(counts) if count >= 2]
    for s, (_count, _angles, (symbol, misfit)) in zip(measured, calls, strict=True):
        reference_symbol, reference_misfit = _detect(_cartesian(structure, s))
        assert symbol == reference_symbol
        if misfit is None:
            assert reference_misfit is None
        else:
            assert reference_misfit is not None and abs(misfit - reference_misfit) <= 1e-6


def test_supplied_shapes_skip_the_baseline_for_their_sites() -> None:
    structure = _b8("CuCl2 3.1")
    supplied = tuple("I6" if site.species == "Cu+2" else None for site in structure.wyckoff_sites)
    result = _site_shapes(_with(structure, supplied))
    assert result == tuple("I6" if s == "I6" else b for s, b in zip(supplied, _site_shapes(structure), strict=True))


@pytest.mark.parametrize("name", ["AgO", "CuCl2 3.1", "diopside", "AlCl3"])
def test_the_baseline_is_bit_identical_over_exact_re_descriptions(name: str) -> None:
    reference = _stars(_b8(name))
    seen = 0
    for described in _redescriptions(name):
        assert _stars(described) == reference
        seen += 1
    assert seen >= 9


def test_the_baseline_of_a_left_handed_description_is_that_of_the_right_handed_one() -> None:
    right, left = _chain(), _chain(left=True)
    assert _same_network(left, right)
    assert _stars(left) == _stars(right)


def test_the_baseline_of_a_true_mirror_image_is_bit_identical() -> None:
    # the P2_1 helix and its enantiomorph: y -> -y in the same basis is a Cartesian reflection
    chain = _chain()
    mirror = _redescribed(chain, AffineOperation(((1, 0, 0), (0, -1, 0), (0, 0, 1))), basis=chain.cell.basis, seed=1)
    assert _stars(mirror) == _stars(chain)
    # P3_1 into P3_2 by z -> -z, in a hexagonal basis with a surd component
    root3 = SurdVector.sqrt_of(3)
    basis = SurdVector([[4, 0, 0], [-2, 0, 0], [0, 0, 5]]) + SurdVector([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) * root3
    group = Spacegroup.standard(144)
    sites = [WyckoffSite("a", (F(1, 7), F(2, 9), F(1, 11)), "C"), WyckoffSite("a", (F(3, 7), F(1, 9), F(2, 11)), "C")]
    source = ASUStructure(basis, group, sites, [C], bonds=_nearest_bonds(group, basis, sites, [(0, 0), (0, 1), (1, 1)]))
    flip = AffineOperation(((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    image = _redescribed(source, flip, basis=basis, setting=Spacegroup.standard(145), seed=2)
    assert image.spacegroup.it_number == 145
    assert _stars(image) == _stars(source)


def test_a_surd_metric_is_converted_once_per_pair() -> None:
    a = SurdVector.sqrt_of(2) + 1
    basis = SurdVector([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * a + SurdVector([[0, 0, 0], [0, 2, 0], [0, 0, 3]])
    bonds = [
        SiteBond(0, 0, AffineOperation(ONE.matrix, (1, 0, 0))),
        SiteBond(0, 0, AffineOperation(ONE.matrix, (1, 1, 0))),
    ]
    structure = ASUStructure(Cell(basis), 1, [WyckoffSite("a", (0, 0, 0), "C")], [C], bonds=bonds)
    assert not structure.cell.metric().is_rational
    assert _counts(structure) == (4,)
    with _recorded() as calls:
        (symbol,) = _baseline_shapes(structure)
    ((_count, angles, _result),) = calls
    floats = [[float(x) for x in row] for row in basis.to_floats()]
    vectors = [
        [s * sum(n[k] * floats[k][axis] for k in range(3)) for axis in range(3)]
        for n in ((1, 0, 0), (1, 1, 0))
        for s in (1, -1)
    ]
    # the float path loses digits near 180 degrees, where acos is ill-conditioned; the exact one gives 180
    assert all(abs(x - y) < 1e-5 for x, y in zip(angles, shapes._sorted_angles(vectors), strict=True))
    assert angles[-2:] == (180.0, 180.0)
    assert math.isclose(angles[0], math.degrees(math.atan2(2, float(a))), abs_tol=1e-9)
    assert symbol == _detect(vectors)[0]


# ---------------------------------------------------------------------------------------------
# carrying
# ---------------------------------------------------------------------------------------------
SOURCES = {
    "diamond 227:1": lambda: _diamond("227:1"),
    "chain": _chain,
    "left chain": lambda: _chain(left=True),
    "P 1 21/n 1": _p21n,
    "P 4_2/n b c 133:1": lambda: _p42nbc("133:1"),
    "R -3 m 166:R": _rhombohedral,
    "P1 supercell chain": lambda: _p1_pair(chain=True),
    "P1 primitive diamond": _primitive_diamond,
}


@pytest.mark.parametrize("carrier", sorted(CARRIERS))
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_every_canonicaliser_carries_supplied_shapes(name: str, carrier: str) -> None:
    plain = SOURCES[name]()
    source = _with(plain, _irregular(plain))
    result = CARRIERS[carrier](source)
    assert result.bonds is not None
    # every shaped site of the result is an image of a shaped source site, with its symbol
    assert _shapes(result) == _irregular(result)
    assert result.bonds.pairs == CARRIERS[carrier](plain).bonds.pairs


def _twins(spacegroup: Spacegroup, second: str) -> ASUStructure:
    """Two general sites, a half b apart, each in a chain of its own: equal counts (2), L2 and A2 supplied."""
    basis = [[5, 0, 0], [0, 6, 0], [-1, 0, 7]]
    p = (F(1, 8), F(2, 7), F(3, 11))
    sites = [WyckoffSite("e", p, "Si"), WyckoffSite("e", (p[0], p[1] + F(1, 2), p[2]), second)]
    species = [SI] if second == "Si" else [SI, Species(second, (second,), (1,))]
    bonds = _nearest_bonds(spacegroup, basis, sites, [(0, 0), (1, 1)])
    return ASUStructure(basis, spacegroup, sites, species, bonds=BondNetwork(bonds, ("L2", "A2")))


@pytest.mark.parametrize("carrier", sorted(CARRIERS))
def test_shapes_of_sites_with_equal_counts_land_on_their_own_sites(carrier: str) -> None:
    source = _twins(Spacegroup.from_hm_entry("P 1 21/n 1"), "Ge")
    assert _counts(source) == (2, 2)
    result = CARRIERS[carrier](source)
    assert _shapes(result) is not None
    assert {site.species: shape for site, shape in zip(result.wyckoff_sites, _shapes(result), strict=True)} == {
        "Si": "L2",
        "Ge": "A2",
    }


def test_shapes_of_one_species_follow_their_atoms() -> None:
    source = _twins(Spacegroup.standard(14), "Si")
    group = source.spacegroup
    for index, coset in enumerate(protonet._tables(14).cosets[1:6]):
        result = _redescribed(source, coset.action, seed=index)
        assert _shapes(result) is not None
        for site, shape in zip(source.wyckoff_sites, _shapes(source) or (), strict=True):
            image = coset.action.apply(FracVector(asu_module._reference(group, site))).normalize().to_fractions()
            (number,) = [
                n
                for n, target in enumerate(result.wyckoff_sites)
                if any(
                    tuple(g.apply(FracVector(asu_module._reference(group, target))).normalize().to_fractions())
                    == tuple(image)
                    for g in group.symmetry_operations
                )
            ]
            assert _shapes(result)[number] == shape


def test_coincident_sites_carry_their_bonds_and_shapes() -> None:
    source = _coincident(bonded=False, shapes=("OC6", "I6", None))
    expected = {"Fe": (6, "OC6"), "Mn": (6, "I6"), "O": (4, None)}
    for index, coset in enumerate(protonet._tables(221).cosets[1:4]):
        result = _redescribed(source, coset.action, seed=index)
        assert result.bonds is not None and len(result.bonds.pairs) == 2  # Fe-O and Mn-O
        assert _shapes(result) is not None
        found = {
            site.species: (count, shape)
            for site, count, shape in zip(result.wyckoff_sites, _counts(result), _shapes(result), strict=True)
        }
        assert found == expected


def _coincident_target(source: ASUStructure, action: AffineOperation, names: list[str], species: Any) -> ASUStructure:
    group = source.spacegroup
    sites = []
    for site, name in zip(source.wyckoff_sites, names, strict=True):
        found = group.identify_wyckoff(action.apply(FracVector(asu_module._reference(group, site))).normalize())
        assert found is not None
        sites.append(WyckoffSite(found[0].letter, found[1], name))
    return ASUStructure(SurdVector(action.matrix.inv().T()) * source.cell.basis, group, sites, species)


def test_carries_that_rename_coincident_sites_or_stack_one_species_fail_loudly() -> None:
    action = protonet._tables(221).cosets[1].action
    source = _coincident(bonded=False, shapes=(None, "I6", None))
    oxygen = source.species[2]
    # C: the coincident Fe and Mn renamed (and so changed) on the way
    renamed = [Species("Mn", ("Mn",), (1,), charges=(2,)), Species("Fe", ("Fe",), (1,), charges=(3,)), oxygen]
    with pytest.raises(_CoincidentAtoms, match="not one atom"):
        _carried(source, _coincident_target(source, action, ["Mn", "Fe", "O"], renamed), action)
    # the same carry with the species kept is fine, whichever site lists which species
    swapped = _carried(source, _coincident_target(source, action, ["Mn", "Fe", "O"], source.species), action)
    assert swapped.bonds is not None and len(swapped.bonds.pairs) == 2
    # E: two sites of one species at one point
    fe = source.species[0]
    stacked = _with(
        source,
        (None, "I6", None),
        wyckoff_sites=(*[WyckoffSite("a", (), "Fe")] * 2, source.wyckoff_sites[2]),
        species=[fe, oxygen],
    )
    with pytest.raises(_CoincidentAtoms, match="not one atom"):
        _carried(stacked, _coincident_target(stacked, action, ["Fe", "Fe", "O"], [fe, oxygen]), action)


def test_views_keep_supplied_shapes() -> None:
    source = _with(_p21n(), ("TPY3", None))
    assert _shapes(ASUStructureView(source)) == ("TPY3", None)
    assert _shapes(ASUStructureView(source, immutable_id="x")) == ("TPY3", None)


def test_operations_carry_or_drop_the_shapes_with_the_network() -> None:
    diamond = _diamond("227:2")
    source = _with(diamond, ("SS4",))
    # subgroup descent and recognition carry the network (follow-up 7 part B, packet P2)
    assert _shapes(subgroup_representation(source, 141).asu) == ("SS4",)
    assert _shapes(recognize_asu(UnitcellStructureView(source))) == ("SS4",)
    refined = Protostructure(representative=source)
    assert refined.representative is not None and _shapes(refined.representative) is None
    assert refined == Protostructure(representative=diamond._with_bonds(None))


def _merging_pair(shapes: Any) -> ASUStructure:
    """The P1 supercell chain (two C atoms, one per half cell) with ``shapes``."""
    return _with(_p1_pair(chain=True), shapes)


def _half_cell() -> ASUStructure:
    return ASUStructure([[3, 0, 0], [0, 4, 0], [0, 0, 5]], 1, [WyckoffSite("a", (0, 0, 0), "C")], [C])


HALVE = AffineOperation(((2, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_carrying_merges_shapes_and_drops_the_network_on_a_conflict() -> None:
    # _carried serves lift, the anonymous frames, subgroups and standardization alike
    assert _shapes(_carried(_merging_pair(("L2", "L2")), _half_cell(), HALVE)) == ("L2",)
    assert _shapes(_carried(_merging_pair(("A2", None)), _half_cell(), HALVE)) == ("A2",)  # None gives way
    assert _shapes(_carried(_merging_pair(None), _half_cell(), HALVE)) is None
    dropped = _carried(_merging_pair(("L2", "A2")), _half_cell(), HALVE)
    assert dropped.bonds is None and _shapes(dropped) is None


def test_recognition_drops_the_network_on_a_shape_conflict() -> None:
    carried = canonical_asu_protostructure(_merging_pair(("L2", None)))
    assert carried.bonds is not None and _shapes(carried) == ("L2",)
    dropped = canonical_asu_protostructure(_merging_pair(("L2", "A2")))
    assert dropped.bonds is None and _shapes(dropped) is None
    # the recognition carry itself
    source = _merging_pair(("L2", "A2"))
    target = recognize_asu(UnitcellStructureView(source))
    assert len(target.wyckoff_sites) == 1
    with pytest.raises(_ShapeConflict):
        _matched(source, target, target.transform.operation.inverse())
    agreeing = _merging_pair(("L2", "L2"))
    network = _matched(agreeing, target, target.transform.operation.inverse())
    assert network is not None and network.shapes == ("L2",)
    unshaped = _matched(_p1_pair(chain=True), target, target.transform.operation.inverse())
    assert unshaped is not None and unshaped.shapes is None


def test_the_p1bar_collapse_keeps_the_uncollapsed_structure_on_a_shape_conflict() -> None:
    cell = [[8, 0, 0], [0, 5, 0], [0, 0, 6]]
    p = (F(1, 8), F(1, 5), F(1, 7))
    sites = [WyckoffSite("i", p, "C"), WyckoffSite("i", (p[0] + F(1, 2), p[1], p[2]), "C")]
    bonds = [SiteBond(0, 1, ONE), SiteBond(1, 0, AffineOperation(ONE.matrix, (1, 0, 0)))]
    agreeing = ASUStructure(cell, 2, sites, [C], bonds=BondNetwork(bonds, ("L2", "L2")))
    collapsed = _p1bar_primitive_entry(agreeing)
    assert len(collapsed.wyckoff_sites) == 1 and _shapes(collapsed) == ("L2",)
    conflicting = ASUStructure(cell, 2, sites, [C], bonds=BondNetwork(bonds, ("L2", "A2")))
    assert _p1bar_primitive_entry(conflicting) is conflicting


def test_the_tie_break_key_tells_supplied_shapes_apart() -> None:
    structure = _p21n()
    assert _bonds_key(structure) == tuple(
        (b.site, b.partner, *asu_module._op(b.operation)) for b in structure.bonds.pairs
    )
    keys = [_bonds_key(_with(structure, value)) for value in (("TPY3", None), ("I3", None), ("TPY3", "A2"))]
    assert keys[0][:-1] == keys[1][:-1] == _bonds_key(structure)
    assert keys[0][-1] == (-1, ("TPY3", "")) and keys[2][-1] == (-1, ("TPY3", "A2"))
    # a key with shapes against one with more bonds compares numbers, never a string with a number
    longer = (*keys[0][:-1], (5, 5))
    assert keys[0] < longer
    assert sorted(keys) == [keys[1], keys[0], keys[2]]


# ---------------------------------------------------------------------------------------------
# storage and identity
# ---------------------------------------------------------------------------------------------
def test_content_identity_is_unchanged_without_shapes() -> None:
    # computed on c951156, before shapes existed
    diamond = _diamond("227:2")
    assert content_id(diamond) == "6e31b0fd1d501c7ef1205e1611edbff5a2b47c3a5f67a5feb096d9bfc803ad56"
    domain = FundamentalDomainStructure(
        diamond.cell, diamond.spacegroup, diamond.wyckoff_sites, [C], bonds=diamond.bonds
    )
    assert content_id(domain) == "cd04538802d02202ae8db4fff0de94789044c9b6b40715cb63638c6751792856"
    assert content_id(_p21n()) == "6c112ea63c002aea268d751c1451ee741c84cad2cf1bee6c668c440e163f0ff6"
    assert content_id(_with(_p21n(), (None, None))) == content_id(_p21n())
    assert content_id(_with(_p21n(), ("TPY3", None))) != content_id(_p21n())


@pytest.mark.parametrize("kind", [ASUStructure, FundamentalDomainStructure])
def test_records_round_trip_supplied_shapes(kind: type[Any]) -> None:
    plain = _p21n()
    args = (plain.cell, plain.spacegroup, plain.wyckoff_sites, plain.species)
    values = (
        kind(*args, bonds=plain.bonds),
        kind(*args, bonds=BondNetwork(plain.bonds.pairs, ("TPY3", None))),
        kind(*args, bonds=BondNetwork(plain.bonds.pairs, ("TPY3", "A2"))),
    )
    records = [_domain_structure_record_from_value(value) for value in values]
    record_type = ASUStructureRecord if kind is ASUStructure else FundamentalDomainStructureRecord
    assert [record.shapes for record in records] == [None, ("TPY3", ""), ("TPY3", "A2")]
    assert len({content_id(record) for record in records}) == 3
    for value, record in zip(values, records, strict=True):
        assert content_id(record) == content_id(value)
        assert _domain_structure_from_record(record) == value
        record_type.__httk_validate__(record)
    with pytest.raises(ValueError, match="shapes are not in their normal form"):
        record_type.__httk_validate__(replace(records[0], shapes=("", "")))
    with pytest.raises(ValueError, match="shapes need bonds"):
        replace(records[1], bonds=None)
    with pytest.raises(TypeError, match="str values"):
        replace(records[1], shapes=("TPY3", None))


@pytest.fixture(params=["sqlite", "mongo"])
def store(request: pytest.FixtureRequest) -> Iterator[Any]:
    pytest.importorskip("httk.store")
    from httk.store import EntryIdScheme

    records = {StructureEntry: ASUStructureRecord}
    ids = ("httk.test", "1")
    if request.param == "sqlite":
        pytest.importorskip("sqlalchemy")
        from httk.store import Backend, SqlStore

        with Backend.sqlite() as database:
            yield SqlStore(database, entry_ids=EntryIdScheme(*ids), entry_records=records)
        return
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    pymongo = pytest.importorskip("pymongo")
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=1000)
    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        client.admin.command("ping")
    except Exception as error:
        client.close()
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        yield MongoStore(MongoDatabase(client, name), entry_ids=EntryIdScheme(*ids), entry_records=records)
    finally:
        client.drop_database(name)
        client.close()


def test_a_store_round_trips_supplied_shapes(store: Any) -> None:
    plain = _p21n()
    values = (plain, _with(plain, ("TPY3", None)), _with(plain, ("TPY3", "A2")))
    fetched = [store.fetch(ASUStructureRecord, store.save(value), eager=True) for value in values]
    assert [record.shapes for record in fetched] == [None, ("TPY3", ""), ("TPY3", "A2")]
    for value, record in zip(values, fetched, strict=True):
        assert content_id(record) == content_id(value)
        assert _domain_structure_from_record(record) == value
