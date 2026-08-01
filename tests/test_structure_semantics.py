from fractions import Fraction as F

import pytest

from httk.atomistic import (
    Assembly,
    ASUSite,
    ASUStructure,
    ASUStructureView,
    Cell,
    ChemicalComposition,
    FundamentalDomainStructure,
    Sites,
    Spacegroup,
    Species,
    Structure,
    StructureSymmetry,
)
from httk.atomistic.affine_operation import AffineOperation


def _structure(**kwargs):
    return Structure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]]),
        (Species("B", ("B",), (1,)), Species("TlO", ("Tl", "O"), (F(1, 3), F(2, 3)))),
        ("B", "TlO"),
        **kwargs,
    )


def test_structure_exposes_common_optimade_semantics() -> None:
    structure = _structure(optimization_type="experimental")
    assert structure.formula == structure.chemical_formula_reduced == "B3O2Tl"
    assert structure.chemical_formula_descriptive is None
    assert structure.chemical_formula_hill is None
    assert structure.elements == ("B", "O", "Tl")
    assert structure.nelements == 3
    assert structure.nsites == 2
    assert structure.dimension_types == (1, 1, 1)
    assert structure.site_coordinate_span == "unit_cell"
    assert structure.space_group_symmetry_operations_xyz == ("x,y,z",)
    assert structure.optimization_type == "experimental"
    assert structure.structure_features == ("disorder",)


def test_span_depends_on_representation_not_dimensionality() -> None:
    zero_dimensional = Structure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], periodicity=(False, False, False)),
        Sites([[0, 0, 0]]),
        (Species("C", ("C",), (1,)),),
        ("C",),
    )
    assert zero_dimensional.site_coordinate_span == "unit_cell"
    assert zero_dimensional.space_group_symmetry_operations_xyz is None
    molecular = Structure(
        zero_dimensional.cell,
        zero_dimensional.sites,
        zero_dimensional.species,
        zero_dimensional.species_at_sites,
        molecular=True,
    )
    assert molecular.site_coordinate_span == "molecular_unit_cell"


def test_native_annotations_are_explicit_structural_values() -> None:
    symmetry = StructureSymmetry(
        space_group_it_number=1,
        space_group_symbol_hall="P 1",
        space_group_symbol_hermann_mauguin="P 1",
        space_group_symmetry_operations_xyz=("x,y,z",),
        wyckoff_positions=("a", "a"),
    )
    annotated = _structure(
        assemblies=(Assembly(((0,), (1,)), (F(1, 2), F(1, 2))),),
        symmetry=symmetry,
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
        chemical_formula_descriptive="B3H6O2Tl",
        optimization_type="local",
    )
    assert annotated.space_group_it_number == 1
    assert annotated.wyckoff_positions == ("a", "a")
    assert annotated.assemblies is not None
    assert annotated.structure_features == ("assemblies", "disorder", "implicit_atoms")
    assert annotated != _structure()
    assert _structure().numeric().chemical_formula_reduced == "B3O2Tl"
    assert _structure().numeric().site_coordinate_span == "unit_cell"


def test_explicit_hill_validation_and_roundtrip() -> None:
    methane = Structure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        Sites([[0, 0, 0]]),
        (Species("methane", ("C",), (1,), attached=("H",), nattached=(4,)),),
        ("methane",),
        chemical_formula_hill="CH4",
        chemical_formula_descriptive="CH4",
    )
    assert methane.chemical_formula_hill == "CH4"
    assert methane.chemical_formula_descriptive == "CH4"
    with pytest.raises(ValueError, match="Hill order"):
        Structure(methane.cell, methane.sites, methane.species, methane.species_at_sites, chemical_formula_hill="H4C")
    with pytest.raises(ValueError, match="ratios disagree"):
        Structure(methane.cell, methane.sites, methane.species, methane.species_at_sites, chemical_formula_hill="CH3")


def test_fundamental_domain_and_asu_spans_and_representatives() -> None:
    cell = Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    group = Spacegroup.standard(221)
    site = ASUSite("a", (), "Cs", representative=(0, 0, 0))
    species = (Species("Cs", ("Cs",), (1,)),)
    domain = FundamentalDomainStructure(cell, group, (site,), species)
    asu = ASUStructure(cell, group, (site,), species, molecular=True)
    assert domain.site_coordinate_span == "fundamental_domain"
    assert asu.site_coordinate_span == "molecular_asymmetric_unit"
    assert isinstance(asu, FundamentalDomainStructure)
    assert domain.fractional_site_positions == [[0.0, 0.0, 0.0]]
    assert domain.wyckoff_positions == ("a",)
    assert domain != asu
    assert domain != FundamentalDomainStructure(cell, group, (site,), species, coordinate_precision=F(1, 100))
    with pytest.raises(ValueError, match="cannot promote"):
        ASUStructureView(domain)
    with pytest.raises(ValueError, match="representative coordinate"):
        FundamentalDomainStructure(
            cell,
            group,
            (ASUSite("a", (), "Cs", representative=(F(1, 4), 0, 0)),),
            species,
        )


def test_semantic_input_validation() -> None:
    with pytest.raises(ValueError, match="optimization_type"):
        _structure(optimization_type="invented")
    with pytest.raises(ValueError, match="outside the structure"):
        _structure(assemblies=(Assembly(((2,),), (1,)),))
    with pytest.raises(ValueError, match="wyckoff_positions"):
        _structure(symmetry=StructureSymmetry(wyckoff_positions=("a",)))
    with pytest.raises(ValueError, match="inconsistent"):
        StructureSymmetry(
            space_group_it_number=2,
            space_group_symbol_hall="P 1",
            space_group_symmetry_operations_xyz=("x,y,z",),
        )
    with pytest.raises(ValueError, match="disagree"):
        StructureSymmetry(
            space_group_it_number=3,
            space_group_symmetry_operations_xyz=("x,y,z", "-x,y+1/2,-z"),
        )
    with pytest.raises(ValueError, match="Wyckoff"):
        StructureSymmetry(wyckoff_positions=("bogus",))
    with pytest.raises(ValueError, match="three periodic"):
        Structure(
            Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], periodicity=(True, True, False)),
            Sites([[0, 0, 0]]),
            (Species("B", ("B",), (1,)),),
            ("B",),
            symmetry=StructureSymmetry(space_group_it_number=1),
        )
    with pytest.raises(ValueError, match="must be null"):
        Structure(
            Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], periodicity=(False, False, False)),
            Sites([[0, 0, 0]]),
            (Species("B", ("B",), (1,)),),
            ("B",),
            symmetry=StructureSymmetry(space_group_symmetry_operations_xyz=("x,y,z",)),
        )

    setting = Spacegroup.for_setting("15:c1")
    metadata = StructureSymmetry(
        space_group_it_number=15,
        space_group_symbol_hall=setting.hall_symbol,
        space_group_symbol_hermann_mauguin=setting.hermann_mauguin,
    )
    assert metadata.space_group_symmetry_operations_xyz == tuple(
        operation.wrapped().to_xyz() for operation in setting.symmetry_operations
    )

    centered = Spacegroup.standard(5)
    origin_shift = AffineOperation.identity()
    origin_shift = AffineOperation(origin_shift.matrix, (F(1, 7), F(2, 11), 0))
    shifted_operations = tuple(
        operation.conjugated_by(origin_shift).wrapped().to_xyz() for operation in centered.symmetry_operations
    )
    assert (
        StructureSymmetry(
            space_group_it_number=5,
            space_group_symmetry_operations_xyz=shifted_operations,
        ).space_group_symmetry_operations_xyz
        == shifted_operations
    )


def test_present_empty_assemblies_and_precision_are_structural() -> None:
    empty = _structure(assemblies=())
    assert empty.assemblies == ()
    assert empty.structure_features == ("assemblies", "disorder")
    assert empty != _structure()
    precise = Structure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], precision=F(1, 100)),
        Sites([[0, 0, 0]], precision=F(1, 100)),
        (Species("B", ("B",), (1,)),),
        ("B",),
    )
    vague = Structure(precise.cell.basis, precise.sites.reduced_coords, precise.species, precise.species_at_sites)
    assert precise != vague
