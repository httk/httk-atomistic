"""Baseline torsion statements and protonet torsion letters from a network's statements (torsions phase 2B)."""

import importlib.metadata
import logging
from dataclasses import replace
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import pytest
from test_protonet_molecular import MOLECULAR
from test_structure_shapes import _redescribed
from test_torsions import BRANCHED, C_, CHECK, HEXAGONAL, ONE, SHAPES, SPECIES, A, B, D, P, Q, X, _branched, _chain

from httk.atomistic import (
    ASUStructure,
    BondSource,
    Cell,
    FundamentalDomainStructure,
    Protonetstructure,
    SiteTorsion,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    assign_baseline_shapes,
    assign_baseline_torsions,
    distance_bonds,
)
from httk.atomistic.models.protonetstructure.protonetstructure import _protonet_asu
from httk.atomistic.models.structure import bonds as bonds_module
from httk.atomistic.models.structure.asu import _apply, _reference, _standard_elements
from httk.atomistic.models.structure.bonds import _signed_torsions
from httk.atomistic.symmetry.canonical_protostructure import _mirrored
from httk.atomistic.symmetry.shapes import _site_shapes
from httk.atomistic.symmetry.torsions import _baseline_torsions, _bond_torsions, _provided, _site_pair_torsions

FIXTURES = Path(__file__).parent / "fixtures" / "cod"
BASELINE = BondSource(
    "Baseline torsions: the protonet torsion letter (c, g or t) of every bond the letter rules qualify, with the "
    "signed IUPAC torsion angle of the reference atoms A-B-C-D rounded to 0.01 degrees.",
    "httk_basic",
    importlib.metadata.version("httk-atomistic"),
)


@cache
def _bare(name: str) -> Any:
    """A COD fixture with distance bonds and shapes but no torsions."""
    return distance_bonds(httk.core.load(str(FIXTURES / f"{name}.cif")), torsions=False)


def _labelled(structure: Any) -> Protonetstructure:
    return Protonetstructure.from_asu(structure)


def _with(structure: Any, *rows: SiteTorsion) -> Any:
    """The structure with its network's torsions replaced by ``rows``."""
    return structure._with_bonds(replace(structure.bonds, torsions=rows or None, torsions_source=None))


def _letters(value: Protonetstructure) -> list[str]:
    return sorted(filter(None, value.torsions))


# ---------------------------------------------------------------------------------------------
# baseline assignment
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(MOLECULAR) | set(CHECK)))
def test_the_baseline_statements_keep_every_label(name: str) -> None:
    bare = _bare(name)
    found = distance_bonds(httk.core.load(str(FIXTURES / f"{name}.cif")))
    assert found.bonds is not None and found.bonds.pairs == bare.bonds.pairs
    # one statement per lettered bond: the ad-hoc letter and its angle (rounded) from the geometry
    letters = [x for x in _bond_torsions(bare, _site_shapes(bare)) if x[0]]
    rows = found.bonds.torsions or ()
    assert sorted((row.symbol, round(abs(row.angle or 0))) for row in rows) == sorted(
        (letter, round(tau)) for letter, tau, _reason in letters
    )
    assert found.bonds.torsions_source == (BASELINE if rows else None)  # the baseline names itself
    for row, angle in zip(rows, _signed_torsions(found, rows), strict=True):
        assert angle is not None and abs(row.angle - angle) <= 0.005  # type: ignore[operator]
    value, before = _labelled(found), _labelled(bare)
    assert value == before and str(value.label) == str(before.label)


def test_assign_baseline_torsions_stamps_its_source_and_keeps_the_rest() -> None:
    bare = _bare("7219102")
    assigned = assign_baseline_torsions(bare)
    assert type(assigned) is type(bare) and assigned.bonds is not None and bare.bonds is not None
    assert assigned.bonds.torsions == distance_bonds(bare).bonds.torsions  # type: ignore[union-attr]
    assert assigned.bonds.torsions_source == BASELINE == distance_bonds(bare).bonds.torsions_source  # type: ignore[union-attr]
    assert (assigned.bonds.pairs, assigned.bonds.shapes) == (bare.bonds.pairs, bare.bonds.shapes)
    assert (assigned.bonds.source, assigned.bonds.shapes_source) == (bare.bonds.source, bare.bonds.shapes_source)
    # without shapes the rules read transient baseline shapes, which are not stored
    unshaped = distance_bonds(bare, shapes=False)
    assert unshaped.bonds is not None and unshaped.bonds.shapes is None
    assert unshaped.bonds.torsions == assigned.bonds.torsions


def test_a_unit_cell_gets_the_statements_of_its_p1_domain() -> None:
    cell = UnitcellStructureView(_bare("7219102")).unview()
    assigned = assign_baseline_torsions(cell._with_bonds(replace(cell.bonds, shapes=None, shapes_source=None)))
    assert type(assigned) is UnitcellStructure and assigned.bonds is not None
    rows = assigned.bonds.torsions
    assert rows is not None and len(rows) == 4  # Z = 4: the one lettered bond of each molecule
    assert {row.symbol for row in rows} == {"t"} and assigned.bonds.torsions_source == BASELINE
    assert all(op.matrix == ONE.matrix for row in rows for op in row.operations)  # lattice translations
    for row, angle in zip(rows, _signed_torsions(assigned, rows), strict=True):
        assert angle is not None and abs(row.angle - angle) <= 0.005  # type: ignore[operator]
    assert distance_bonds(cell).bonds.torsions == rows  # type: ignore[union-attr]


def test_assign_baseline_torsions_refusals_and_no_op() -> None:
    bare = _bare("7219102")
    with pytest.raises(ValueError, match="without a bond network"):
        assign_baseline_torsions(bare._with_bonds(None))
    with pytest.raises(ValueError, match="already has torsions"):
        assign_baseline_torsions(distance_bonds(bare))
    diopside = _bare("1000008")  # no molecular piece: no letter, the structure comes back unchanged
    assert assign_baseline_torsions(diopside) is diopside
    # a structure that is not periodic in three directions: refused here, skipped by distance_bonds
    cell = UnitcellStructureView(bare).unview()
    loose = UnitcellStructure(
        Cell(cell.cell.basis, periodicity=(False, False, False)),
        cell.sites.reduced_coords,
        cell.species,
        cell.species_at_sites,
    )
    found = distance_bonds(loose)
    assert found.bonds is not None and found.bonds.pairs and found.bonds.torsions is None
    with pytest.raises(ValueError, match="periodic in three directions"):
        assign_baseline_torsions(found)
    shaped = assign_baseline_shapes(distance_bonds(bare, shapes=False, torsions=False))
    assert shaped.bonds is not None and shaped.bonds.torsions is None


def test_shapes_false_skips_torsions_where_the_baseline_shapes_are_undefined(caplog: pytest.LogCaptureFixture) -> None:
    # P-1 general letter i at the inversion centre: no bond count, so no shapes and no torsions
    structure = ASUStructure(
        [[F(564, 100), 0, 0], [0, F(564, 100), 0], [0, 0, F(564, 100)]],
        2,
        [WyckoffSite("i", (0, 0, 0), "Na"), WyckoffSite("h", (), "Cl")],
        [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
    )
    with caplog.at_level(logging.DEBUG, logger="httk.atomistic.bonding"):
        found = distance_bonds(structure, shapes=False)
    assert found.bonds is not None and found.bonds.shapes is None and found.bonds.torsions is None
    assert "baseline torsions skipped" in caplog.text
    with pytest.raises(ValueError, match=r"special point of its Wyckoff letter 'i'"):
        assign_baseline_torsions(found)


@pytest.mark.parametrize(
    ("basis", "letter", "angle", "stored"),
    [
        (None, "g", -119.996, -119.99),  # g below the 120 degree edge: rounding to 120 would cross it
        (None, "g", 119.994, 119.99),
        (HEXAGONAL, "t", 119.996, 120.0),
        (HEXAGONAL, "t", 119.994, 120.0),  # t from the edge: rounding below it would cross it
        (HEXAGONAL, "t", -179.996, 180.0),
    ],
)
def test_the_stored_angle_never_crosses_the_edge(
    monkeypatch: pytest.MonkeyPatch, basis: Any, letter: str, angle: float, stored: float
) -> None:
    # the geometry gives 90 (g) or 120 (t) degrees; the signed angle is replaced to probe the rounding
    structure = _chain() if basis is None else _chain(basis)
    monkeypatch.setattr(bonds_module, "_signed_torsions", lambda _structure, rows: [angle] * len(rows))
    (row,) = _baseline_torsions(structure, SHAPES)
    assert (row.sites, row.symbol, row.angle) == ((A, B, C_, D), letter, stored)


# ---------------------------------------------------------------------------------------------
# letters from statements, on the chain X-A-B-C-D-Y (T4-T4: edge 120, the rule gives g at 90)
# ---------------------------------------------------------------------------------------------
def _row(sites: tuple[int, int, int, int], symbol: str | None = None, angle: float | None = None) -> SiteTorsion:
    return SiteTorsion(sites, (ONE,) * 4, symbol, angle)


def _letter(structure: Any, *rows: SiteTorsion, shapes: Any = SHAPES) -> str | None:
    full = tuple(shapes) + (None,) * (len(structure.wyckoff_sites) - len(shapes))
    return _site_pair_torsions(_with(structure, *rows), full).get((B, C_))


REFERENCE = (A, B, C_, D)


@pytest.mark.parametrize(
    ("symbol", "angle", "letter"),
    [
        ("t", None, "t"),  # a letter the pair allows
        ("g", 170.0, "g"),  # the symbol wins over the angle
        ("ap", None, "t"),  # Klyne-Prelog, sign ignored
        ("sp", None, "g"),
        ("-sc", None, "g"),
        (None, -150.0, "t"),  # the angle alone, either sign
        (None, 119.9999996, "t"),  # rounded to 1e-6: on the edge, the upper class
        (None, 60.0, "g"),
        ("ac", 170.0, "t"),  # ac at the 120 degree edge decides nothing: the angle does
        ("+ac", None, "g"),  # ... or, without one, the rule
        ("c", 170.0, "t"),  # c is not a letter of this pair
        ("c", None, "g"),
    ],
)
def test_a_statement_on_the_reference_atoms_gives_the_letter(symbol: str, angle: float, letter: str) -> None:
    assert _letter(_chain()) == "g"
    assert _letter(_chain(), _row(REFERENCE, symbol, angle)) == letter


@pytest.mark.parametrize(
    ("symbol", "letter"), [("c", "c"), ("t", "t"), ("g", "t"), ("sp", "c"), ("sc", "c"), ("ac", "t"), ("ap", "t")]
)
def test_the_symbols_at_the_90_degree_edge(symbol: str, letter: str) -> None:
    planar = (None, "A2", "TP3", "TP3", "A2", None)  # edge 90 (c/t): the rule gives t at 90 degrees
    angle = 120.0 if symbol == "g" else None  # g is not a letter of this pair: its angle decides
    assert _letter(_chain(), shapes=planar) == "t"
    assert _letter(_chain(), _row(REFERENCE, symbol, angle), shapes=planar) == letter


def test_none_suppresses_the_letter_and_conflicts_with_a_determining_statement() -> None:
    assert _letter(_chain(), _row(REFERENCE, "none")) is None
    reason = [x for x in _bond_torsions(_with(_chain(), _row(REFERENCE, "none")), SHAPES) if x[2] == "none statement"]
    assert len(reason) == 1
    # on a branched B any A and D count for none: none on P-B-C-D suppresses, with t on A-B-C-D it raises
    branched = _branched(1)
    assert _letter(branched, shapes=BRANCHED) == "g"
    assert _letter(branched, _row((P, B, C_, D), "none"), shapes=BRANCHED) is None
    with pytest.raises(ValueError, match=r"torsion statements on the bond between C2 on a \(site 2\) and C3"):
        _letter(branched, _row((P, B, C_, D), "none"), _row(REFERENCE, "t"), shapes=BRANCHED)


def test_disagreeing_statements_raise() -> None:
    # statements of one orbit are merged by the structure (a conflict drops them); the guard is direct
    structure = _chain()
    stated = [_row(REFERENCE, "t"), _row(REFERENCE, None, 60.0)]
    assert structure.bonds is not None
    elements = _standard_elements(structure.spacegroup)
    refs = [_apply(elements.to_standard, _reference(structure.spacegroup, s)) for s in structure.wyckoff_sites]
    points = [refs[s] for s in REFERENCE]
    statements = [(row, row.sites, [(0, 0, 0, 0)] * 4, points) for row in stated]
    ends = (A, refs[A], refs[B], refs[C_], D, refs[D])
    assert _provided(elements.tables, [(0, 0, 0, 0)], statements, (B, C_), ends, (120.0, "g")) == (False, {"t", "g"})
    # the label refuses them: a structure built with both raises, a carried pair of them is dropped
    with pytest.raises(ValueError, match="conflicting torsion statements"):
        FundamentalDomainStructure(
            structure.cell,
            structure.spacegroup,
            structure.wyckoff_sites,
            structure.species,
            bonds=replace(structure.bonds, torsions=(_row(REFERENCE, "t"), _row(REFERENCE[::-1], "g"))),
        )


def test_the_letter_of_a_statement_ignores_the_collinear_cutoff() -> None:
    straight = _chain(a=(F(51, 104), 2 * Q, F(1, 8)))  # A-B-C 2.94 degrees from straight
    assert _letter(straight) is None
    assert _letter(straight, _row(REFERENCE, "t")) == "t"


def test_statements_on_other_atoms_or_bonds_are_ignored() -> None:
    branched = _branched(1)  # A (two bonds) is the reference neighbour at B, not P (one bond)
    assert _letter(branched, _row((P, B, C_, D), "t"), shapes=BRANCHED) == "g"
    # a statement on an end-group bond gives it no letter
    rows = _with(_chain(), _row((X, A, B, C_), "t"))
    assert [x[0] for x in _bond_torsions(rows, SHAPES)].count(None) == 4
    assert _site_pair_torsions(rows, SHAPES) == {(B, C_): "g"}


def test_a_reversed_statement_is_read_in_both_orientations() -> None:
    # sites shuffled so that the stored (normalized) statement runs D-C-B-A against the bond's B-C
    for seed in range(20):
        moved = _redescribed(_chain(), ONE, basis=_chain().cell.basis, seed=seed)
        index = {site.species: number for number, site in enumerate(moved.wyckoff_sites)}
        sites = tuple(index[SPECIES[k]] for k in REFERENCE)
        if sites[0] > sites[3] and sites[1] < sites[2]:  # stored reversed, judged forward
            break
    else:
        raise AssertionError("no shuffle stores the statement reversed")
    shapes = [dict(zip(SPECIES, SHAPES, strict=True))[site.species] for site in moved.wyckoff_sites]
    stated = _with(moved, _row(sites, "t"))
    assert stated.bonds.torsions[0].sites == sites[::-1]
    assert sorted(_site_pair_torsions(stated, shapes).values()) == ["t"]
    assert sorted(_site_pair_torsions(moved, shapes).values()) == ["g"]


# ---------------------------------------------------------------------------------------------
# letters from statements through Protonetstructure.from_asu (sarcosine: one bond with t)
# ---------------------------------------------------------------------------------------------
@cache
def _sarcosine() -> Any:
    found = distance_bonds(_bare("7219102"))
    assert found.bonds is not None and found.bonds.torsions is not None and len(found.bonds.torsions) == 1
    return found


def _restated(structure: Any, **changes: Any) -> Any:
    (row,) = structure.bonds.torsions
    return _with(structure, replace(row, **changes))


@pytest.mark.parametrize(
    "changes", [{"symbol": "g"}, {"symbol": "sc"}, {"symbol": None, "angle": 60.0}, {"symbol": "+sc", "angle": 60.0}]
)
def test_a_provided_letter_is_the_labels(changes: dict[str, Any]) -> None:
    assert _letters(_labelled(_sarcosine())) == ["t"]
    value = _labelled(_restated(_sarcosine(), **changes))
    assert _letters(value) == ["g"] and value != _labelled(_sarcosine())


def test_a_none_statement_removes_the_labels_letter() -> None:
    assert _letters(_labelled(_restated(_sarcosine(), symbol="none", angle=None))) == []


def test_a_statement_given_through_another_orbit_member() -> None:
    structure = _sarcosine()
    (row,) = structure.bonds.torsions
    screw = structure.spacegroup.symmetry_operations[1]  # P2_12_12_1 is proper: the image keeps its sign
    image = SiteTorsion(row.sites, tuple(screw * op for op in row.operations), "g")
    assert image.operations != row.operations
    assert _letters(_labelled(_with(structure, image))) == ["g"]


def test_letters_survive_the_re_description_inside_from_asu() -> None:
    # the mirror image in a left-handed basis of a Sohncke group: from_asu re-describes it right-handed
    mirrored = _mirrored(_restated(_sarcosine(), symbol="g"))
    assert _protonet_asu(mirrored) is not mirrored
    assert mirrored.bonds is not None and mirrored.bonds.torsions is not None
    assert _letters(Protonetstructure.from_asu(mirrored)) == ["g"]


def test_partial_statements_give_their_letters_and_the_rule_the_rest() -> None:
    found = distance_bonds(_bare("4134228"))  # four C-C bonds with t
    rows = found.bonds.torsions  # type: ignore[union-attr]
    assert rows is not None and len(rows) == 4
    value = _labelled(_with(found, replace(rows[0], symbol="g")))
    assert _letters(value) == ["g", "t", "t", "t"]
    assert _letters(_labelled(_with(found))) == ["t", "t", "t", "t"]
