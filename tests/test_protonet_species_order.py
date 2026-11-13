"""Species order of protonetstructure labels (D3) and how the levels project onto each other."""

from functools import cache

import httk.core
import pytest
from _protonet_support import default_shapes, generic_cutoffs, load_structure, raw_rows
from test_protonet_canonical import FIXTURE_EXTENDED, FIXTURE_NORMAL, FIXTURES, _profile
from test_protonet_canonical_api import CASES, _asu, _bonds, _value

from httk.atomistic import BareProtostructureView, Species
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.symmetry.canonical_classification import (
    canonical_bare_protostructure,
    canonical_bare_prototype,
    canonical_protonetstructure,
    canonical_protonettype,
)
from httk.atomistic.symmetry.protonet import _tables


def _species(name: str) -> Species:
    return Species(name, (name,), (1,))


# Group 14: operation 1 is the identity, operation 2 the inversion -x,-y,-z. Occupations are in
# site order (count, letters, then name); bonds are 1-based (i, j, k, n1, n2, n3).
# name -> (occupations, bonds, protonetstructure label, canonical bare protostructure label)
ROWS = {
    "n-self-and-h": (
        [("e", "H"), ("e", "N")],
        [(2, 2, 2, 0, 0, 0), (1, 2, 1, 0, 0, 0)],
        "AB_mP8_14_e_e__1~1@2.2:N-H",
        "AB_mP8_14_e_e:H-N",
    ),
    "c-self-and-h": (
        [("e", "C"), ("e", "H")],
        [(1, 1, 2, 0, 0, 0), (1, 2, 1, 0, 0, 0)],
        "AB_mP8_14_e_e__1~1@2.2:C-H",
        "AB_mP8_14_e_e:C-H",
    ),
    "n-h-two-bonds": (
        [("e", "H"), ("e", "N")],
        [(1, 2, 1, 0, 0, 0), (1, 2, 2, 0, 0, 0)],
        "AB_mP8_14_e_e__1~2.2@2:H-N",
        "AB_mP8_14_e_e:H-N",
    ),
    "o-cu": (
        [("e", "O"), ("a", "Cu"), ("b", "Zn")],
        [(1, 2, 1, 0, 0, 0)],
        "A2BC_mP8_14_e_a_b__2L2~1:O-Cu-Zn",
        "A2BC_mP8_14_e_a_b:O-Cu-Zn",
    ),
    "o-zn": (
        [("e", "O"), ("a", "Cu"), ("b", "Zn")],
        [(1, 3, 1, 0, 0, 0)],
        "A2BC_mP8_14_e_a_b__2L2~1:O-Zn-Cu",
        "A2BC_mP8_14_e_a_b:O-Cu-Zn",
    ),
}
SELF_BONDED_PROTONETTYPE = "AB_mP8_14_e_e__1~1@2.2"
# the default shapes, except that two bonds through an inversion centre are linear
LINEAR = {"o-cu": (None, "L2", None), "o-zn": (None, None, "L2")}


@pytest.mark.parametrize("name", list(ROWS))
def test_species_order_rows(name: str) -> None:
    occupations, bonds, label, bare_label = ROWS[name]
    shapes = LINEAR.get(name) or default_shapes(14, [letter for letter, _ in occupations], bonds)
    value = Protonetstructure(14, [(letter, _species(species)) for letter, species in occupations], bonds, shapes)
    canonical = canonical_protonetstructure(value)
    bare = canonical_bare_protostructure(value)
    assert str(canonical.label) == label
    assert str(bare.label) == bare_label
    assert BareProtostructureView(canonical).unview() == bare
    assert str(BareProtostructureView(canonical).label) == bare_label
    if name.endswith("self-and-h"):
        assert str(canonical_protonettype(value.protonettype).label) == SELF_BONDED_PROTONETTYPE


@cache
def _input(fixture: str) -> Protonetstructure:
    """The uncanonicalized value of a handmade API case or a CIF fixture with generic cutoffs."""
    if fixture in CASES:
        return _value(fixture)
    path = str(FIXTURES / fixture)
    loaded = load_structure(path)
    assert loaded is not None, "fixture must be an ordered element-pure standard-setting structure"
    it, sites, refs, lattice = loaded
    cif = httk.core.load(path)
    # rebuild with element-named species and no retained representatives, as from_asu requires
    spec = [(cls, ws.wyckoff, tuple(ws.free_params.to_fractions())) for (cls, _), ws in zip(sites, cif.wyckoff_sites)]
    tables = _tables(it)
    rows, _ = raw_rows(tables, sites, refs, lattice, generic_cutoffs(tables, sites, refs, lattice))
    return Protonetstructure.from_asu(_asu(it, spec, cif.cell), _bonds(rows))


@pytest.mark.parametrize("fixture", [*CASES, *_profile(FIXTURE_NORMAL, FIXTURE_EXTENDED)])
def test_levels_project_onto_each_other(fixture: str) -> None:
    try:
        value = _input(fixture)
    except ValueError as exc:
        if "-5..+4" not in str(exc):
            raise
        pytest.skip(f"from_asu cannot express this fixture: {exc}")
    canonical = canonical_protonetstructure(value)
    protonettype = canonical_protonettype(value.protonettype)
    # protonetstructure -> protonettype: cut the species suffix
    assert str(canonical.label).rpartition(":")[0] == str(protonettype.label)
    # protonettype -> bare prototype: cut the net
    assert str(protonettype.label).partition("__")[0] == str(canonical_bare_prototype(value).label)
    # protonetstructure -> bare protostructure: not a string operation, the projection re-canonicalizes
    bare = canonical_bare_protostructure(value)
    projected = BareProtostructureView(canonical)
    assert projected.unview() == bare
    assert str(projected.label) == str(bare.label)
