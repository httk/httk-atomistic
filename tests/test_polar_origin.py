"""Regression checks for the exact polar continuous-origin fast path."""

from fractions import Fraction

import pytest
from httk.core import FracVector
from httk.core.storage import content_id

from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite
from httk.atomistic.symmetry import lift, paths
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

F = Fraction

_POLAR_GROUPS = (4, 29, 33)


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _position(spacegroup: int):
    position = next(candidate for candidate in Spacegroup.standard(spacegroup).wyckoff if candidate.free_count == 3)
    assert position.free == (0, 1, 2)
    assert position.representative.operation.matrix == FracVector.eye((3, 3))
    assert position.representative.operation.vector == FracVector((0, 0, 0))
    return position


def _structure_for_group(group: Spacegroup, rows: tuple[tuple[str, tuple[F, F, F]], ...]) -> ASUStructure:
    position = next(candidate for candidate in group.wyckoff if candidate.free_count == 3)
    names = tuple(dict.fromkeys(row[0] for row in rows))
    return ASUStructure(
        Cell(
            ((F(5), 0, 0), (F(1, 3), F(7), 0), (F(2, 5), F(1, 4), F(9))),
            precision=F(1, 100),
        ),
        group,
        tuple(WyckoffSite(position.letter, FracVector(parameters), species) for species, parameters in rows),
        _species(*names),
        coordinate_precision=F(1, 1000),
        charge=F(3, 2),
    )


def _structure(spacegroup: int, rows: tuple[tuple[str, tuple[F, F, F]], ...]) -> ASUStructure:
    return _structure_for_group(Spacegroup.standard(spacegroup), rows)


def _alternative_orbit_parameters(spacegroup: int, parameters: tuple[F, F, F]) -> tuple[F, F, F]:
    position = _position(spacegroup)
    point = position.branches[1].coordinate(FracVector(parameters))
    recovered = position.parameters_of(point)
    assert recovered is not None
    return tuple(recovered.to_fractions())


def _axis(spacegroup: int) -> int:
    return 1 if spacegroup == 4 else 2


def _shifted(structure: ASUStructure, shift: F) -> ASUStructure:
    translation = [F(0), F(0), F(0)]
    translation[_axis(structure.spacegroup.it_number)] = shift
    result = lift._apply_normalizer_operation(
        structure, AffineOperation(FracVector.eye((3, 3)), FracVector(translation)), trusted=True
    )
    assert result is not None
    return result


def _optimized_and_generic(
    monkeypatch: pytest.MonkeyPatch, structure: ASUStructure
) -> tuple[ASUStructure, ASUStructure]:
    original = lift._polar_translation_normal_form
    calls: list[ASUStructure | None] = []

    def checked(value: ASUStructure, axes: list[int]) -> ASUStructure | None:
        result = original(value, axes)
        calls.append(result)
        return result

    with monkeypatch.context() as context:
        context.setattr(lift, "_polar_translation_normal_form", checked)
        optimized = lift._translation_normal_form(structure)
    assert calls and any(result is not None for result in calls)

    with monkeypatch.context() as context:
        context.setattr(lift, "_polar_translation_normal_form", lambda _value, _axes: None)
        generic = lift._translation_normal_form(structure)
    return optimized, generic


def _assert_same_structure(first: ASUStructure, second: ASUStructure) -> None:
    assert first == second
    assert first.cell == second.cell
    assert first.cell.basis == second.cell.basis
    assert first.cell.precision == second.cell.precision
    assert first.spacegroup == second.spacegroup
    assert first.wyckoff_sites == second.wyckoff_sites
    assert tuple((site.wyckoff, site.free_params, site.species) for site in first.wyckoff_sites) == tuple(
        (site.wyckoff, site.free_params, site.species) for site in second.wyckoff_sites
    )
    assert first.species == second.species
    assert first.transform == second.transform
    assert first.coordinate_precision == second.coordinate_precision
    assert first.charge == second.charge
    assert lift._site_key(first) == lift._site_key(second)
    assert content_id(first) == content_id(second)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
@pytest.mark.parametrize(
    "parameters",
    (
        (F(-7, 5), F(14, 9), F(53, 11)),
        (F(0), F(1, 2), F(0)),
        (F(17, 13), F(-19, 7), F(101, 12)),
    ),
)
def test_polar_fast_path_matches_generic_for_boundary_and_out_of_range_parameters(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int, parameters: tuple[F, F, F]
) -> None:
    optimized, generic = _optimized_and_generic(monkeypatch, _structure(spacegroup, (("C", parameters),)))
    _assert_same_structure(optimized, generic)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_preserves_site_order_species_and_arbitrary_axis_shifts(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    rows = (
        ("C", (F(1, 7), F(2, 11), F(3, 13))),
        ("O", (F(5, 17), F(7, 19), F(11, 23))),
        ("N", (F(-13, 29), F(31, 37), F(41, 43))),
    )
    base = _structure(spacegroup, rows)
    sources = (
        base,
        _shifted(base, F(37, 101)),
        _structure(spacegroup, rows[::-1]),
        _structure(spacegroup, (rows[1], rows[2], rows[0])),
    )
    for source in sources:
        optimized, generic = _optimized_and_generic(monkeypatch, source)
        _assert_same_structure(optimized, generic)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_accepts_an_alternative_orbit_representative(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    parameters = (F(1, 7), F(2, 11), F(3, 13))
    alternate = _alternative_orbit_parameters(spacegroup, parameters)
    source = _structure(
        spacegroup,
        (("C", alternate), ("O", (F(5, 17), F(7, 19), F(11, 23)))),
    )
    optimized, generic = _optimized_and_generic(monkeypatch, source)
    _assert_same_structure(optimized, generic)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_keeps_tied_prefix_and_full_origin_candidates(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    # Both sites share the coordinates preceding the polar axis while remaining distinct
    # general-position orbits. The orbit itself supplies tied origin candidates as well.
    if spacegroup == 4:
        rows = (
            ("C", (F(1, 7), F(2, 11), F(3, 13))),
            ("C", (F(1, 7), F(5, 11), F(3, 13))),
        )
    else:
        rows = (
            ("C", (F(1, 7), F(2, 11), F(3, 13))),
            ("C", (F(1, 7), F(2, 11), F(5, 13))),
        )
    optimized, generic = _optimized_and_generic(monkeypatch, _structure(spacegroup, rows))
    _assert_same_structure(optimized, generic)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_keeps_identity_first_tie_for_full_keys(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    source = _structure(spacegroup, (("C", (F(0), F(0), F(0))),))
    position = _position(spacegroup)
    axis = _axis(spacegroup)
    candidate_shifts = {F(0)}
    for point in position.coordinates(FracVector((F(0), F(0), F(0)))).to_fractions():
        candidate_shifts.add((-point[axis]) % 1)
    keys = {}
    for shift in candidate_shifts:
        image = _shifted(source, shift) if shift else source
        keys[shift] = lift._site_key(image)
    minimum = min(keys.values())
    tied = sorted(shift for shift, key in keys.items() if key == minimum)
    assert len(tied) > 1 and tied[0] == 0
    optimized, generic = _optimized_and_generic(monkeypatch, source)
    _assert_same_structure(optimized, generic)
    assert lift._site_key(optimized) == minimum


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_keeps_smallest_nonzero_origin_on_a_full_key_tie(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    # With the polar coordinate at 1/7, the two best images are nonzero and have
    # equal keys. The generic loop's sorted candidates choose the smaller shift.
    parameters = (F(0), F(1, 7), F(0)) if spacegroup == 4 else (F(0), F(0), F(1, 7))
    source = _structure(spacegroup, (("C", parameters),))
    position = _position(spacegroup)
    axis = _axis(spacegroup)
    shifts = {F(0)}
    shifts.update((-point[axis]) % 1 for point in position.coordinates(FracVector(parameters)).to_fractions())
    images = {shift: _shifted(source, shift) if shift else source for shift in shifts}
    keys = {shift: lift._site_key(image) for shift, image in images.items()}
    minimum = min(keys.values())
    tied = sorted(shift for shift, key in keys.items() if key == minimum)
    assert len(tied) == 2
    assert all(shift > 0 for shift in tied)

    optimized, generic = _optimized_and_generic(monkeypatch, source)
    expected = images[tied[0]]
    _assert_same_structure(generic, expected)
    _assert_same_structure(optimized, expected)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_retains_many_tied_anchors_with_mixed_species(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    # Several C orbits and an O orbit share the coordinates before the polar
    # axis. This leaves many anchors in the least-prefix class and checks that
    # pruning does not assume that one site or one species supplies the winner.
    if spacegroup == 4:
        rows = tuple(
            (species, (F(0), F(index, 7), F(index + 1, 13)))
            for species, index in (("C", 1), ("C", 2), ("C", 3), ("O", 4))
        )
        shared_prefix = (F(0),)
    else:
        rows = tuple(
            (species, (F(0), F(0), F(index, 7))) for species, index in (("C", 1), ("C", 2), ("C", 3), ("O", 4))
        )
        shared_prefix = (F(0), F(0))
    source = _structure(spacegroup, rows)
    position = _position(spacegroup)
    axis = _axis(spacegroup)
    expanded = [
        (species, tuple(point.to_fractions()))
        for species, parameters in rows
        for point in position.coordinates(FracVector(parameters))
    ]
    assert min(point[:axis] for _species_name, point in expanded) == shared_prefix
    assert {species for species, point in expanded if point[:axis] == shared_prefix} == {"C", "O"}
    prefix = ("C", _position(spacegroup).letter, shared_prefix)
    anchors = {(-point[axis]) % 1 for species, point in expanded if (species, position.letter, point[:axis]) == prefix}
    assert len(anchors) >= 6

    optimized, generic = _optimized_and_generic(monkeypatch, source)
    _assert_same_structure(optimized, generic)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_matches_generic_across_normalizer_images(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    source = _structure(
        spacegroup,
        (("C", (F(-7, 5), F(14, 9), F(53, 11))), ("O", (F(5, 17), F(7, 19), F(11, 23)))),
    )
    original = lift._polar_translation_normal_form
    optimized_calls: list[ASUStructure | None] = []

    def checked(value: ASUStructure, axes: list[int]) -> ASUStructure | None:
        result = original(value, axes)
        optimized_calls.append(result)
        return result

    monkeypatch.setattr(lift, "_polar_translation_normal_form", checked)
    optimized = paths._representation_orbit(source)
    assert optimized_calls and any(result is not None for result in optimized_calls)
    with monkeypatch.context() as context:
        context.setattr(lift, "_polar_translation_normal_form", lambda _value, _axes: None)
        generic = paths._representation_orbit(source)

    assert len(optimized) == len(generic)
    for optimized_value, generic_value in zip(optimized, generic, strict=True):
        _assert_same_structure(optimized_value, generic_value)
        assert lift._site_key(optimized_value) == lift._site_key(generic_value)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_materializes_only_the_changed_winner(monkeypatch: pytest.MonkeyPatch, spacegroup: int) -> None:
    source = _structure(spacegroup, (("C", (F(1, 7), F(2, 11), F(3, 13))),))
    original = lift._apply_normalizer_operation
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(lift, "_apply_normalizer_operation", counted)
    result = lift._translation_normal_form(source)
    assert calls == 1
    assert result.wyckoff_sites != source.wyckoff_sites


@pytest.mark.parametrize("spacegroup", (1, 2, 7))
def test_unsupported_groups_use_the_generic_fallback(monkeypatch: pytest.MonkeyPatch, spacegroup: int) -> None:
    source = _structure(spacegroup, (("C", (F(1, 7), F(2, 11), F(3, 13))),))
    outcomes: list[ASUStructure | None] = []
    original = lift._polar_translation_normal_form

    def checked(value: ASUStructure, axes: list[int]) -> ASUStructure | None:
        result = original(value, axes)
        outcomes.append(result)
        return result

    monkeypatch.setattr(lift, "_polar_translation_normal_form", checked)
    result = lift._translation_normal_form(source)
    if spacegroup == 2:
        assert outcomes == []
    else:
        assert outcomes and all(outcome is None for outcome in outcomes)
    assert result == lift._translation_normal_form(source)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_falls_back_for_nonstandard_spacegroup_setting(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    setting = {4: "4:a", 29: "29:cab", 33: "33:cab"}[spacegroup]
    group = Spacegroup.from_setting(setting)
    source = _structure_for_group(group, (("C", (F(1, 7), F(2, 11), F(3, 13))),))
    assert not group.is_standard_setting

    original = lift._polar_translation_normal_form
    outcomes: list[ASUStructure | None] = []

    def checked(value: ASUStructure, axes: list[int]) -> ASUStructure | None:
        result = original(value, axes)
        outcomes.append(result)
        return result

    with monkeypatch.context() as context:
        context.setattr(lift, "_polar_translation_normal_form", checked)
        optimized = lift._translation_normal_form(source)
    assert outcomes and all(outcome is None for outcome in outcomes)

    with monkeypatch.context() as context:
        context.setattr(lift, "_polar_translation_normal_form", lambda _value, _axes: None)
        generic = lift._translation_normal_form(source)
    _assert_same_structure(optimized, generic)


@pytest.mark.parametrize("spacegroup", _POLAR_GROUPS)
def test_polar_fast_path_preserves_generic_result_with_nonidentity_own_cell_transform(
    monkeypatch: pytest.MonkeyPatch, spacegroup: int
) -> None:
    standard = _structure(spacegroup, (("C", (F(1, 7), F(2, 11), F(3, 13))),))
    source = ASUStructure(
        standard.cell,
        standard.spacegroup,
        standard.wyckoff_sites,
        standard.species,
        transform=SettingTransform(FracVector.eye((3, 3)), (F(1, 7), F(0), F(0))),
        coordinate_precision=standard.coordinate_precision,
        charge=standard.charge,
    )

    optimized, generic = _optimized_and_generic(monkeypatch, source)
    _assert_same_structure(optimized, generic)


def test_polar_empty_domain_returns_the_original_object() -> None:
    source = ASUStructure(Cell(((1, 0, 0), (0, 1, 0), (0, 0, 1))), 33, (), ())
    assert lift._translation_normal_form(source) is source
