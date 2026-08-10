"""Exact subgroup descent and round-trip checks."""

from collections import Counter
from fractions import Fraction

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    Assembly,
    ASUStructure,
    Cell,
    SettingTransform,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.symmetry.subgroups import subgroup_transforms

F = Fraction
NO_PARAMETERS = FracVector(())


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _site_multiset(structure: object) -> Counter[tuple[str, tuple[F, ...]]]:
    view = UnitcellStructureView(structure)
    return Counter(
        (name, tuple(coordinate.normalize().to_fractions()))
        for name, coordinate in zip(view.species_at_sites, view.sites.reduced_coords, strict=True)
    )


def test_221_to_123_is_exact_and_uses_the_first_table_entry() -> None:
    parent = ASUStructure(
        Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
        221,
        [
            WyckoffSite("a", NO_PARAMETERS, "Sr"),
            WyckoffSite("b", NO_PARAMETERS, "Ti"),
            WyckoffSite("c", NO_PARAMETERS, "O"),
        ],
        _species("Sr", "Ti", "O"),
    )

    result = subgroup_representation(parent, 123)

    assert result.spacegroup is result.asu.spacegroup
    assert result.asu.transform.is_identity()
    assert result.multiplier == F(1)
    assert result.path == (subgroup_transforms(221, 123)[0],)
    assert tuple(site.wyckoff for site in result.asu.wyckoff_sites) == ("a", "d", "e", "c")
    assert same_crystal(parent, result.asu)


def test_entry_affine_direction_is_child_to_parent() -> None:
    transform = subgroup_transforms(15, 5)[0]
    parent_point = transform.parent.wyckoff_position("a").representative.coordinate(())
    child_point = transform.splittings["a"][0].operation.apply_wrapped(parent_point)

    assert transform.operation.apply_wrapped(child_point) == parent_point


def test_15_to_2_maps_free_parameters_exactly_in_a_smaller_cell() -> None:
    parent = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [WyckoffSite("e", FracVector([F(1, 3)]), "Si")],
        _species("Si"),
    )

    result = subgroup_representation(parent, 2)
    matrix = result.path[0].operation.matrix.T()

    assert result.asu.wyckoff_sites[0].wyckoff == "i"
    assert result.asu.wyckoff_sites[0].free_params.to_fractions() == [F(2, 3), F(1, 3), F(1, 4)]
    assert result.multiplier == F(1, 2)
    assert same_crystal(build_supercell(result.asu, matrix.inv()).structure, parent)


def test_3_to_4_is_the_index_two_enlarged_cell_and_matches_supercell() -> None:
    parent = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        3,
        [WyckoffSite("a", FracVector([F(1, 3)]), "Si")],
        _species("Si"),
    )

    result = subgroup_representation(parent, 4)
    expected = build_supercell(parent, [[1, 0, 0], [0, 2, 0], [0, 0, 1]]).structure

    assert result.multiplier == F(2)
    assert same_crystal(result.asu, expected)


def test_two_level_descent_is_deterministic_and_composes() -> None:
    parent = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        221,
        [WyckoffSite("a", NO_PARAMETERS, "Sr")],
        _species("Sr"),
    )

    direct = subgroup_representation(parent, 148)
    first = subgroup_representation(parent, 166)
    composed = subgroup_representation(first.asu, 148)

    assert len(direct.path) == 2
    assert same_crystal(direct.asu, composed.asu)
    assert direct.multiplier == first.multiplier * composed.multiplier
    assert direct == subgroup_representation(parent, 148)


def test_166_to_148_rhombohedral_descent_is_exact() -> None:
    parent = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 12]],
        166,
        [WyckoffSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
    )

    result = subgroup_representation(parent, 148)

    assert result.multiplier == F(1)
    assert same_crystal(parent, result.asu)


def test_charge_and_precision_follow_the_exact_k_hop_factors() -> None:
    parent = ASUStructure(
        Cell([[5, 0, 0], [0, 6, 0], [0, 0, 7]], precision=F(1, 10)),
        3,
        [WyckoffSite("a", FracVector([F(1, 3)]), "Si")],
        _species("Si"),
        coordinate_precision=F(1, 100),
        charge=F(2),
    )

    result = subgroup_representation(parent, 4)

    assert result.asu.charge == F(4)
    assert result.asu.cell.precision == F(1, 5)
    assert result.asu.coordinate_precision == F(1, 100)


def test_rejections_and_identity() -> None:
    parent = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        15,
        [WyckoffSite("e", FracVector([F(1, 3)]), "Si")],
        _species("Si"),
    )
    with pytest.raises(ValueError, match="221.*15"):
        subgroup_representation(parent, 221)

    moments = ASUStructure(
        parent.cell,
        15,
        [WyckoffSite("e", FracVector([F(1, 3)]), "Si", moment=CollinearSiteMoments([1]))],
        _species("Si"),
    )
    with pytest.raises(ValueError, match="site moments"):
        subgroup_representation(moments, 2)

    with pytest.raises(ValueError, match="assemblies"):
        subgroup_representation(
            ASUStructure(
                parent.cell,
                15,
                parent.wyckoff_sites,
                parent.species,
                assemblies=(Assembly(((0,),), (1,)),),
            ),
            2,
        )
    with pytest.raises(ValueError, match="molecular"):
        subgroup_representation(
            ASUStructure(parent.cell, 15, parent.wyckoff_sites, parent.species, molecular=True),
            2,
        )

    identity = subgroup_representation(parent, 15)
    assert identity.path == ()
    assert identity.multiplier == F(1)
    assert identity.asu.transform.is_identity()
    assert same_crystal(identity.asu, parent)


def test_nonstandard_input_cell_is_normalized_without_changing_sites() -> None:
    transform = SettingTransform([[1, 2, 0], [0, 1, 1], [0, 0, 1]])
    parent = ASUStructure(
        Cell([[5, 0, 0], [-10, 5, 0], [10, -5, 5]], precision=F(1, 50)),
        221,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        transform=transform,
        coordinate_precision=F(1, 1000),
    )

    result = subgroup_representation(parent, 221)

    assert result.asu.transform.is_identity()
    assert result.asu.cell.basis == SurdVector([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    assert result.asu.cell.precision == F(3, 50)
    assert result.asu.coordinate_precision == F(1, 200)


def test_special_parameter_collapse_uses_the_identified_more_special_letter() -> None:
    parent = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        221,
        [WyckoffSite("e", FracVector([0]), "Si")],
        _species("Si"),
    )
    result = subgroup_representation(parent, 123)

    assert tuple(site.wyckoff for site in result.asu.wyckoff_sites) == ("a",)
    assert result.multiplier == F(1)
    assert same_crystal(parent, result.asu)
