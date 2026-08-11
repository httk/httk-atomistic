import fractions

import pytest
from httk.core import SurdVector

from httk.atomistic import (
    ASUStructureView,
    CartesianSiteMoments,
    Cell,
    CellParams,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    CrystalAxisSiteMomentsView,
    ModulatedStructure,
    Species,
    SymopsStructure,
    UnitcellStructure,
    UnitcellStructureView,
    same_crystal,
)

F = fractions.Fraction


def _cell() -> Cell:
    return Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _species() -> tuple[Species]:
    return (Species("Fe", ("Fe",), (1,)),)


def _symops(sites, operations, *, moments=None, cell=None) -> SymopsStructure:
    return SymopsStructure(cell or _cell(), sites, _species(), ("Fe",) * len(sites), operations, site_moments=moments)


def _moment_rows(moments):
    return tuple(
        tuple(moments.crystalaxis_moments._element((row, column)) for column in range(3)) for row in range(len(moments))
    )


def test_p_minus_one_expands_exactly_and_preserves_precision() -> None:
    structure = _symops([[F(1, 4), 0, 0]], ("x,y,z", "-x,-y,-z"))

    assert tuple(tuple(row) for row in structure.sites.reduced_coords.to_fractions()) == (
        (F(1, 4), F(0), F(0)),
        (F(3, 4), F(0), F(0)),
    )
    assert structure.sites.precision is None
    assert structure.symops[0][1] == 1


def test_magnetic_centering_transforms_crystalaxis_moments_exactly() -> None:
    cell = _cell()
    structure = _symops(
        [[0, 0, 0]],
        ("x,y,z,+1", "x+1/2,y+1/2,z+1/2,-1"),
        moments=CrystalAxisSiteMoments([[1, 0, 0]], cell),
        cell=cell,
    )

    assert _moment_rows(structure.site_moments) == ((1, 0, 0), (-1, 0, 0))
    assert structure.site_moments.kind == "crystalaxis"


def test_axial_inversion_keeps_a_moment_unchanged() -> None:
    structure = _symops(
        [[F(1, 4), 0, 0]],
        ("x,y,z,+1", "-x,-y,-z,+1"),
        moments=CrystalAxisSiteMoments([[1, 2, 3]], _cell()),
    )

    assert _moment_rows(structure.site_moments) == ((1, 2, 3), (1, 2, 3))


def test_hexagonal_lattice_frame_moment_math_is_exact() -> None:
    cell = Cell(CellParams((3, 3, 4, 90, 90, 120)).basis)
    structure = _symops(
        [[F(1, 7), F(2, 7), 0]],
        ("x,y,z", "-y,x-y,z", "y-x,-x,z"),
        moments=CrystalAxisSiteMoments([[1, 2, 3]], cell),
        cell=cell,
    )
    rows = _moment_rows(structure.site_moments)
    expected = {tuple(SurdVector(value)._as_scalar() for value in row) for row in ((1, 2, 3), (-2, -1, 3), (1, -1, 3))}

    assert set(rows) == expected
    assert (1, 2, 3) in rows


def test_cartesian_source_stays_cartesian_and_round_trips_exactly() -> None:
    cell = Cell(CellParams((3, 3, 4, 90, 90, 120)).basis)
    crystal = _symops(
        [[F(1, 7), F(2, 7), 0]],
        ("x,y,z", "-y,x-y,z", "y-x,-x,z"),
        moments=CrystalAxisSiteMoments([[1, 2, 3]], cell),
        cell=cell,
    )
    cartesian = _symops(
        [[F(1, 7), F(2, 7), 0]],
        ("x,y,z", "-y,x-y,z", "y-x,-x,z"),
        moments=CartesianSiteMoments(crystal.listed_site_moments.cartesian_moments),
        cell=cell,
    )

    assert cartesian.site_moments.kind == "cartesian"
    assert {
        tuple(
            CrystalAxisSiteMomentsView(cartesian.site_moments, cell=cell).crystalaxis_moments._element((i, j))
            for j in range(3)
        )
        for i in range(len(cartesian.site_moments))
    } == set(_moment_rows(crystal.site_moments))


def test_collinear_moments_use_time_and_determinant_signs() -> None:
    structure = _symops(
        [[0, 0, 0]],
        ("x,y,z,+1", "x+1/2,y,z,-1", "-x+1/3,y,z,+1", "-x+2/3,y,z,-1"),
        moments=CollinearSiteMoments([2]),
    )

    assert set(structure.site_moments.collinear_moments.to_fractions()) == {F(2), F(-2)}
    assert len(structure.site_moments) == 4


def test_duplicate_position_with_different_moment_is_rejected() -> None:
    structure = _symops(
        [[0, 0, 0]],
        ("x,y,z,+1", "x,y,z,-1"),
        moments=CollinearSiteMoments([1]),
    )

    with pytest.raises(ValueError, match="different species/moment"):
        _ = structure.sites


def test_views_expand_symops_and_asu_recognition_checks_moment_uniformity() -> None:
    structure = _symops([[F(1, 4), 0, 0]], ("x,y,z", "-x,-y,-z"))
    hand_built = UnitcellStructure(_cell(), [[F(1, 4), 0, 0], [F(3, 4), 0, 0]], _species(), ("Fe", "Fe"))

    assert same_crystal(UnitcellStructureView(structure), hand_built)
    assert len(ASUStructureView(structure).wyckoff_sites) == 1

    afm = _symops(
        [[0, 0, 0]],
        ("x,y,z,+1", "x+1/2,y+1/2,z+1/2,-1"),
        moments=CrystalAxisSiteMoments([[1, 0, 0]], _cell()),
    )
    with pytest.raises(ValueError, match="non-uniform site moments"):
        ASUStructureView(afm)


def test_modulated_structure_holds_payload_and_rejects_standard_properties() -> None:
    payload = {"incomm": {"mod_dim": 1, "structural_q": ((F(1, 3), 0, 0),), "magnetic_q": None}}
    structure = ModulatedStructure(payload)

    assert structure.payload == payload
    assert structure.mod_dim == 1
    assert structure.structural_q == payload["incomm"]["structural_q"]
    assert structure.magnetic_q is None
    message = "cannot be represented as a"
    for name in ("cell", "sites", "species", "species_at_sites", "site_moments"):
        with pytest.raises(ValueError, match=message):
            getattr(structure, name)
    with pytest.raises(ValueError, match=message):
        _ = UnitcellStructureView(structure).cell
