"""Tests for subgroup graph navigation and exact tabulated transformations."""

from httk.atomistic import (
    Spacegroup,
    data,
    maximal_subgroups,
    minimal_supergroups,
    subgroup_closure,
    supergroup_closure,
)
from httk.atomistic.symmetry import AffineOperation, WyckoffPosition, subgroup_transforms


def _generic(position: WyckoffPosition):
    parameters = ("1/7", "2/11", "3/13")
    return position.representative.coordinate(parameters[: position.free_count])


def _transform(parent: int, child: int, subgroup_type: str):
    return next(item for item in subgroup_transforms(parent, child) if item.subgroup_type == subgroup_type)


def test_graph_facts_and_verified_cubic_closure() -> None:
    assert maximal_subgroups(2) == (1,)
    assert 1 not in maximal_subgroups(1)
    assert subgroup_transforms(1, 1)
    assert subgroup_closure(2) == (1,)
    assert subgroup_closure(2, include_self=True) == (1, 2)
    assert maximal_subgroups(221)
    assert all(1 <= target <= 230 for target in maximal_subgroups(221))
    assert 221 not in maximal_subgroups(221)
    assert 123 in subgroup_closure(221)
    assert 2 in subgroup_closure(221)


def test_closure_duality_on_deterministic_sample_of_actual_edges() -> None:
    sampled: list[tuple[int, int]] = []
    for parent in range(1, 231):
        for child in maximal_subgroups(parent):
            sampled.append((parent, child))
            if len(sampled) == 20:
                break
        if len(sampled) == 20:
            break
    assert len(sampled) == 20
    for parent, child in sampled:
        assert (child in subgroup_closure(parent)) == (parent in supergroup_closure(child))


def test_minimal_supergroups_contains_every_inverted_maximal_edge() -> None:
    for parent in range(1, 231):
        for child in maximal_subgroups(parent):
            assert parent in minimal_supergroups(child)


def test_split_affines_pin_parent_to_child_direction() -> None:
    samples = ((221, 166, "t"), (15, 2, "t"), (3, 4, "k"), (166, 148, "t"))
    for parent_number, child_number, subgroup_type in samples:
        transform = _transform(parent_number, child_number, subgroup_type)
        parent = Spacegroup.standard(parent_number)
        child = Spacegroup.standard(child_number)
        for parent_letter, pieces in transform.splittings.items():
            point = _generic(parent.wyckoff_position(parent_letter))
            for piece in pieces:
                identified = child.identify_wyckoff(piece.operation.apply_wrapped(point))
                assert identified is not None
                assert identified[0].letter == piece.letter


def test_subgroup_transform_structure_is_exact_and_uses_valid_letters() -> None:
    checked = 0
    for parent_number in range(1, 231):
        for subgroup in data.spacegroup_subgroup_record(parent_number)["baernighausen"]:
            child_number = subgroup["target_it_number"]
            for transform in subgroup_transforms(parent_number, child_number):
                assert transform.operation.determinant() != 0
                assert transform.subgroup_type in {"t", "k"}
                assert (transform.k_subtype is None) is (transform.subgroup_type == "t")
                parent_letters = {position.letter for position in transform.parent.wyckoff}
                child_letters = {position.letter for position in transform.subgroup.wyckoff}
                assert set(transform.splittings) <= parent_letters
                assert all(
                    piece.letter in child_letters for pieces in transform.splittings.values() for piece in pieces
                )
                checked += 1
    assert checked > 0


def test_unknown_pairs_and_bad_it_numbers() -> None:
    assert subgroup_transforms(1, 230) == ()
    for query in (0, 231):
        try:
            maximal_subgroups(query)
        except KeyError:
            pass
        else:
            raise AssertionError(f"IT number {query} did not raise KeyError")


def test_models_keep_exact_affines_and_immutable_splittings() -> None:
    transform = subgroup_transforms(15, 2)[0]
    assert isinstance(transform.operation, AffineOperation)
    assert transform.operation.determinant() != 0
    try:
        transform.splittings["new"] = ()  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("splittings mapping is mutable")
