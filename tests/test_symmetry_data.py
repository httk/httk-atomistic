"""Tests for the vendored symmetry datasets and their accessors.

These assert the structural invariants that the ASU machinery relies on. They are not
decoration: the expansion algorithm is tolerance-free *because* orbits are pre-expanded
and deduplicated, and the free-parameter handling needs no string parsing *because*
``hasfreedom`` agrees with the orbit matrix rank. If a data refresh broke either, the
code above would fail in subtle ways rather than loudly, so it is checked here.
"""

import fractions

import pytest

from httk.atomistic import data
from httk.atomistic.symmetry.symop_key import symop_key_v1

F = fractions.Fraction

# The seven rhombohedral-axes settings, the only ones whose transform is not unimodular.
RHOMBOHEDRAL_IT_NUMBERS = (146, 148, 155, 160, 161, 167, 166)

# Space groups where the IT standard setting differs from spglib's default setting.
TWO_ORIGIN_IT_NUMBERS = (
    48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130,
    133, 134, 137, 138, 141, 142, 201, 203, 222, 224, 227, 228,
)  # fmt: skip


def _rational_matrix(rows: list[list[str]]) -> list[list[F]]:
    return [[F(entry) for entry in row] for row in rows]


def _det3(m: list[list[F]]) -> F:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _rank3(m: list[list[F]]) -> int:
    """The rank of a 3x3 exact rational matrix, by Gaussian elimination over the rationals."""
    rows = [list(row) for row in m]
    rank = 0
    for column in range(3):
        pivot = next((r for r in range(rank, 3) if rows[r][column] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][column]
        for r in range(3):
            if r != rank and rows[r][column] != 0:
                factor = rows[r][column] / scale
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


# --- dataset shape ---


def test_spacegroup_subgroup_records_cover_all_it_numbers() -> None:
    records = [data.spacegroup_subgroup_record(it_number) for it_number in range(1, 231)]
    assert len(records) == 230
    assert all(set(record) == {"it_number", "baernighausen", "continuous_normalizer"} for record in records)
    assert [record["it_number"] for record in records] == list(range(1, 231))
    with pytest.raises(KeyError):
        data.spacegroup_subgroup_record(0)
    with pytest.raises(KeyError):
        data.spacegroup_subgroup_record(231)


def test_baernighausen_references_valid_it_numbers_and_subgroup_types() -> None:
    for it_number in range(1, 231):
        for subgroup in data.spacegroup_subgroup_record(it_number)["baernighausen"]:
            assert 1 <= subgroup["target_it_number"] <= 230
            for transform in subgroup["transforms"]:
                assert transform["subgroup_type"] in {"t", "k"}
                if transform["subgroup_type"] == "k":
                    assert transform["k_subtype"]


def test_subgroup_affine_data_is_exact_and_nonsingular() -> None:
    for it_number in range(1, 231):
        record = data.spacegroup_subgroup_record(it_number)
        for subgroup in record["baernighausen"]:
            for transform in subgroup["transforms"]:
                for splitting in transform["wyckoff_splitting"]:
                    for split in splitting["splits"]:
                        affine = split["affine"]
                        assert len(affine) == 3
                        assert all(len(row) == 4 for row in affine)
                        assert all(isinstance(value, str) for row in affine for value in row)
                        for row in affine:
                            [F(value) for value in row]

                affine = transform["affine_transformation"]
                matrix = affine["matrix"]
                vector = affine["vector"]
                assert len(matrix) == 3
                assert all(len(row) == 3 for row in matrix)
                assert len(vector) == 3
                assert all(isinstance(value, str) for row in matrix for value in row)
                assert all(isinstance(value, str) for value in vector)
                assert _det3(_rational_matrix(matrix)) != 0
                [F(value) for value in vector]


def test_baernighausen_letters_match_standard_settings() -> None:
    for it_number in range(1, 231):
        parent_letters = {entry["letter"] for entry in data.standard_spacegroup_setting(it_number)["wyckoff"]}
        for subgroup in data.spacegroup_subgroup_record(it_number)["baernighausen"]:
            child_letters = {
                entry["letter"] for entry in data.standard_spacegroup_setting(subgroup["target_it_number"])["wyckoff"]
            }
            for transform in subgroup["transforms"]:
                for splitting in transform["wyckoff_splitting"]:
                    assert {splitting["parent"]} <= parent_letters
                    assert {split["letter"] for split in splitting["splits"]} <= child_letters


def test_continuous_normalizers_are_exact_fractional_bases() -> None:
    for it_number in range(1, 231):
        normalizer = data.spacegroup_subgroup_record(it_number)["continuous_normalizer"]
        assert normalizer["dimension"] in {0, 1, 2, 3}
        assert normalizer["coordinate_system"] == "fractional"
        assert len(normalizer["basis_vectors"]) == normalizer["dimension"]
        for vector in normalizer["basis_vectors"]:
            assert len(vector) == 3
            assert all(isinstance(value, str) for value in vector)
            [F(value) for value in vector]


def test_affine_normalizer_cosets_cover_reference_settings() -> None:
    hall_entries = [data.standard_spacegroup_setting(it_number)["hall_entry"] for it_number in range(1, 231)]
    assert len(set(hall_entries)) == 230
    for hall_entry in hall_entries:
        assert data.affine_normalizer_coset_record(hall_entry)["hall_entry"] == hall_entry

    systems = {"triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"}
    sample = data.affine_normalizer_coset_record(hall_entries[0])
    cosets = sample["orthogonal_affine_normalizer_cosets"] + sample["affine_normalizer_cosets"]
    assert {system for coset in cosets for system in coset["compatible_systems"]} <= systems
    with pytest.raises(KeyError):
        data.affine_normalizer_coset_record("not_a_hall_symbol")


def test_settings_and_reference_settings_are_complete() -> None:
    settings = data.spacegroup_settings()
    assert len(settings) == 527
    assert sum(1 for record in settings if record["is_reference_setting"]) == 230
    assert data.standard_setting_it_numbers() == list(range(1, 231))
    assert len(data.point_groups()) == 32


def test_every_it_number_has_a_reference_setting_that_agrees_with_the_flag() -> None:
    for it_number in range(1, 231):
        record = data.standard_spacegroup_setting(it_number)
        assert record["it_number"] == it_number
        assert record["is_reference_setting"] is True


def test_lookup_keys_agree_with_each_other() -> None:
    record = data.spacegroup_setting(setting_it_nc="15:c1")
    assert record["it_number"] == 15
    assert record["hall_entry"] == "-a_2a"
    assert data.spacegroup_setting(hall_entry="-a_2a") is record
    assert data.spacegroup_setting(hm_entry=record["hm_entry"]) is record


def test_symop_key_index_matches_the_v1_canonical_recipe() -> None:
    """Canary for drift from data-generators/generate_basics_hall.py:2957."""
    settings = data.spacegroup_settings()
    computed = {symop_key_v1(record["symops"]): position for position, record in enumerate(settings)}
    vendored = data._lookup_index(data._basics(), "symmetry_basics", "index_symop_key_to_spacegroups")

    assert computed == vendored
    assert len(computed) == 527
    assert next(iter(computed)) == "63dbcdb54bd5d8c35ce8ae32cb34369717b95ee5d3c49dba36f5bbf9bc800048"
    assert data.spacegroup_setting_by_symop_key(next(iter(computed))) is settings[0]


def test_lookup_rejects_unknown_keys_and_ambiguous_calls() -> None:
    with pytest.raises(KeyError):
        data.spacegroup_setting(hall_entry="not_a_hall_symbol")
    with pytest.raises(KeyError):
        data.standard_spacegroup_setting(231)
    with pytest.raises(TypeError):
        data.spacegroup_setting()
    with pytest.raises(TypeError):
        data.spacegroup_setting(hall_entry="p_1", setting_it_nc="1")


def test_settings_are_expressed_in_their_own_coordinates() -> None:
    """The same Wyckoff letter reads differently in different settings of one group.

    This is the property that forces the standard-setting-plus-transform design: a
    setting's Wyckoff table cannot be used interchangeably with another's.
    """

    def first_orbit_of(setting_it_nc: str, letter: str) -> str:
        record = data.spacegroup_setting(setting_it_nc=setting_it_nc)
        return next(w["first_orbit"] for w in record["wyckoff"] if w["letter"] == letter)

    assert first_orbit_of("15:b1", "e") == "0,y,1/4"
    assert first_orbit_of("15:c1", "e") == "1/4,0,z"


# --- invariants the ASU machinery depends on ---


def test_orbits_are_pre_expanded_and_complete() -> None:
    """``len(orbit) == multiplicity`` everywhere, so expansion needs no deduplication."""
    checked = 0
    for record in data.spacegroup_settings():
        for wyckoff in record["wyckoff"]:
            assert len(wyckoff["orbit"]) == wyckoff["multiplicity"]
            checked += 1
    assert checked == 3440


def test_hasfreedom_selects_the_free_parameters_of_every_orbit_member() -> None:
    """``sum(hasfreedom) == rank(orbit[k].matrix)``, with non-free columns identically zero.

    This is why the ``first_orbit`` strings never need parsing: the free-parameter count
    and placement are already carried by ``hasfreedom`` plus the tabulated affine maps.
    """
    items = 0
    for record in data.spacegroup_settings():
        for wyckoff in record["wyckoff"]:
            hasfreedom = wyckoff["hasfreedom"]
            expected_rank = sum(1 for free in hasfreedom if free)
            for item in wyckoff["orbit"]:
                matrix = _rational_matrix(item["matrix"])
                assert _rank3(matrix) == expected_rank
                for column, free in enumerate(hasfreedom):
                    if not free:
                        assert all(matrix[row][column] == 0 for row in range(3))
                items += 1
    assert items == 24079


def test_fixed_coordinates_and_translations_have_small_denominators() -> None:
    """Orbit translations divide 12 and symop translations divide 6.

    A cheap corroborating invariant: a transform bug that corrupts coordinates shows up
    immediately as a denominator outside these sets.
    """
    orbit_denominators: set[int] = set()
    symop_denominators: set[int] = set()
    for record in data.spacegroup_settings():
        for symop in record["symops"]:
            for entry in symop["affine_transformation"]["vector"]:
                symop_denominators.add(F(entry).denominator)
        for wyckoff in record["wyckoff"]:
            for item in wyckoff["orbit"]:
                for entry in item["vector"]:
                    orbit_denominators.add(F(entry).denominator)
    assert orbit_denominators <= {1, 2, 3, 4, 6, 8, 12}
    assert symop_denominators <= {1, 2, 3, 4, 6}


def test_symop_counts_match_the_declared_totals() -> None:
    for record in data.spacegroup_settings():
        assert len(record["symops"]) == record["n_symops"]
        assert len(record["centering_translations"]) == record["n_centering_translations"]


# --- setting transforms ---


def test_every_setting_has_a_transform_to_its_standard_setting() -> None:
    for record in data.spacegroup_settings():
        transform = data.setting_transform(record["hall_entry"])
        assert transform["it_number"] == record["it_number"]
        standard = data.standard_spacegroup_setting(record["it_number"])
        assert transform["to_hall_entry"] == standard["hall_entry"]


def test_reference_settings_transform_by_the_identity() -> None:
    identity = [[F(1), F(0), F(0)], [F(0), F(1), F(0)], [F(0), F(0), F(1)]]
    for it_number in range(1, 231):
        record = data.standard_spacegroup_setting(it_number)
        affine = data.setting_transform(record["hall_entry"])["affine_transformation"]
        assert _rational_matrix(affine["matrix"]) == identity
        assert [F(entry) for entry in affine["vector"]] == [F(0), F(0), F(0)]


def test_only_the_rhombohedral_settings_change_the_cell_volume() -> None:
    non_unimodular: dict[str, F] = {}
    for record in data.spacegroup_settings():
        affine = data.setting_transform(record["hall_entry"])["affine_transformation"]
        determinant = _det3(_rational_matrix(affine["matrix"]))
        assert determinant != 0
        if abs(determinant) != 1:
            non_unimodular[record["setting_it_nc"]] = determinant

    assert set(non_unimodular.values()) == {F(3)}
    assert sorted(int(key.split(":")[0]) for key in non_unimodular) == sorted(RHOMBOHEDRAL_IT_NUMBERS)


def test_transform_with_an_unknown_hall_entry_raises() -> None:
    with pytest.raises(KeyError):
        data.setting_transform("not_a_hall_symbol")


def test_standard_setting_differs_from_spglib_default_exactly_for_two_origin_groups() -> None:
    """Documented so no interop path quietly assumes the two coincide."""
    differing = [
        it_number
        for it_number in range(1, 231)
        if data.standard_spacegroup_setting(it_number)["hall_entry"]
        != data.spglib_default_spacegroup_setting(it_number)["hall_entry"]
    ]
    assert differing == sorted(TWO_ORIGIN_IT_NUMBERS)


def test_the_conventional_hall_spelling_normalizes_to_the_indexed_one() -> None:
    """``hall`` lower-cased with spaces as underscores is exactly ``hall_entry``.

    A CIF writes the conventional spelling (``-C 2yc``) while the tables are indexed by the
    normalized one (``-c_2yc``), so reading a declared Hall symbol depends on this holding.
    Both spellings are also unique across the settings, so neither can name two of them.
    """
    settings = data.spacegroup_settings()
    for record in settings:
        assert record["hall"].lower().replace(" ", "_") == record["hall_entry"], record["setting_it_nc"]

    assert len({record["hall"] for record in settings}) == len(settings)
    assert len({record["hall_entry"] for record in settings}) == len(settings)
