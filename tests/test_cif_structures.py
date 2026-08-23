"""Tests for building asymmetric-unit structures from CIF files.

A CIF is the natural source for an ASU: it lists one site per orbit and states the
operations that generate the rest, so no symmetry search is needed and spglib is not
involved. What has to be right is the *setting* — a file written in a non-standard setting
must be recognized as such rather than reinterpreted — and the fidelity of the numbers.

CIF text is generated from the vendored tables so that a fixture can be produced for any
setting; the coordinates and occupancies in it are written by hand.
"""

import fractions
import logging
from pathlib import Path
from typing import Any

import pytest
from httk.core import FracVector, decimal_precision, load, save
from httk.core.report import collect_reports

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    SettingTransform,
    Spacegroup,
    Species,
    UnitcellStructureView,
    asu_structure_from_cif,
    asu_structures_from_cif,
    cif_setting,
    data,
)
from httk.atomistic.cif_structures import (
    _cell_from_cif,
    _definitely_general,
    _general_position_screen,
    _has_rounded_orbit_overlap,
    _hm_it_numbers,
    _normalized_hm,
    _parse_type_symbol,
    _site_declaration,
    _site_uncertainty,
    _snap,
)
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.structure.asu import FundamentalDomainStructure, WyckoffSite, _ValidatedASUProof
from httk.atomistic.symmetry.wyckoff import WyckoffBranch

F = fractions.Fraction


def _write_cif(
    path: Path,
    setting: str,
    parameters: tuple[float, ...],
    sites: list[tuple[str, str, tuple[str, str, str], str]],
    *,
    name: str = "test",
    declare_number: bool = True,
    wyckoff_labels: list[str] | None = None,
    symmetry_multiplicities: list[str] | None = None,
    site_symmetry_orders: list[str] | None = None,
    deprecated_symmetry_multiplicities: list[str] | None = None,
    attached_hydrogens: list[str] | None = None,
    calc_flags: list[str] | None = None,
) -> Path:
    """A CIF for one setting, with its complete symmetry-operation list."""
    spacegroup = Spacegroup.from_setting(setting)
    a, b, c, alpha, beta, gamma = parameters
    lines = [
        f"data_{name}",
        f"_cell_length_a {a}",
        f"_cell_length_b {b}",
        f"_cell_length_c {c}",
        f"_cell_angle_alpha {alpha}",
        f"_cell_angle_beta {beta}",
        f"_cell_angle_gamma {gamma}",
    ]
    if declare_number:
        lines.append(f"_space_group_IT_number {spacegroup.it_number}")
        lines.append(f"_space_group_name_H-M_alt '{spacegroup.hermann_mauguin}'")
    lines += ["loop_", "_space_group_symop_operation_xyz"]
    lines += [f"'{operation.wrapped().to_xyz()}'" for operation in spacegroup.symmetry_operations]
    lines += [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        *(["_atom_site_Wyckoff_label"] if wyckoff_labels is not None else []),
        *(["_atom_site_site_symmetry_multiplicity"] if symmetry_multiplicities is not None else []),
        *(["_atom_site_site_symmetry_order"] if site_symmetry_orders is not None else []),
        *(["_atom_site_symmetry_multiplicity"] if deprecated_symmetry_multiplicities is not None else []),
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        *(["_atom_site_attached_hydrogens"] if attached_hydrogens is not None else []),
        *(["_atom_site_calc_flag"] if calc_flags is not None else []),
    ]
    for index, (label, symbol, (x, y, z), occupancy) in enumerate(sites):
        declarations = [
            *([wyckoff_labels[index]] if wyckoff_labels is not None else []),
            *([symmetry_multiplicities[index]] if symmetry_multiplicities is not None else []),
            *([site_symmetry_orders[index]] if site_symmetry_orders is not None else []),
            *([deprecated_symmetry_multiplicities[index]] if deprecated_symmetry_multiplicities is not None else []),
        ]
        trailing = [
            *([attached_hydrogens[index]] if attached_hydrogens is not None else []),
            *([calc_flags[index]] if calc_flags is not None else []),
        ]
        lines.append(f"{label} {symbol} {' '.join(declarations)} {x} {y} {z} {occupancy} {' '.join(trailing)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rocksalt_cif(tmp_path: Path) -> Path:
    return _write_cif(
        tmp_path / "nacl.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0.0", "0.0", "0.0"), "1.0"), ("Cl1", "Cl", ("0.5", "0.5", "0.5"), "1.0")],
        name="NaCl",
    )


def _rocksalt_integer_cif(tmp_path: Path) -> Path:
    return _write_cif(
        tmp_path / "nacl-integer.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0", "0", "0"), "1"), ("Cl1", "Cl", ("0.5", "0.5", "0.5"), "1")],
        name="NaClInteger",
    )


def test_cif_retains_validated_expansion_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    structure = load(str(_rocksalt_cif(tmp_path)))
    assert "_precomputed_expansion" in structure.__dict__
    assert "_expansion" not in structure.__dict__

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("the validated CIF expansion was recomputed")

    monkeypatch.setattr(Spacegroup, "wyckoff_position", fail)
    assert structure.multiplicities() == (4, 4)
    assert "_precomputed_expansion" not in structure.__dict__
    assert "_expansion" in structure.__dict__


def test_cif_dummy_sites_become_implicit_species_and_attached_hydrogens(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "implicit.cif",
        "1",
        (5, 5, 5, 90, 90, 90),
        [
            ("C1", "C", ("0.123", "0.234", "0.345"), "1"),
            ("O1", "O", (".", ".", "."), "0.5"),
            ("H1", "H", ("-1", "-1", "-1"), "5"),
        ],
        attached_hydrogens=["3", "0", "0"],
        calc_flags=["d", "dum", "dum"],
    )

    structure = load(path)
    by_name = {species.name: species for species in structure.species}

    assert structure.domain_species_at_sites == ("C1",)
    assert structure.implicit_atoms == ("O1", "H1")
    assert structure.structure_features == ("implicit_atoms", "site_attachments")
    assert by_name["C1"].attached == ("H",)
    assert by_name["C1"].nattached == (3,)
    assert by_name["O1"].chemical_symbols == ("O",)
    assert by_name["O1"].concentration == (F(1, 2),)
    assert by_name["H1"].concentration == (F(5),)

    destination = tmp_path / "implicit-roundtrip.cif"
    save(structure, destination)
    raw = load(destination, raw=True)["blocks"][0]
    restored = load(destination)

    assert raw["attached_hydrogens"] == [3, 0, 0]
    assert raw["calc_flags"] == ["d", "dum", "dum"]
    assert restored.species == structure.species
    assert restored.implicit_atoms == ("O1", "H1")
    assert restored.structure_features == ("implicit_atoms", "site_attachments")


def test_strict_undeclared_cif_does_not_build_orbit_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    screens: list[object] = []
    original_snap = _snap

    def capture_snap(*args: Any, **kwargs: Any) -> Any:
        screens.append(kwargs["orbit_screen"])
        return original_snap(*args, **kwargs)

    monkeypatch.setattr("httk.atomistic.cif_structures._snap", capture_snap)
    load(str(_rocksalt_cif(tmp_path)))

    assert screens
    assert all(screen is None for screen in screens)


def test_generic_cif_matching_skips_wyckoff_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_cif(
        tmp_path / "generic.cif",
        "2",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", ("0.142857", "0.285714", "0.428571"), "1")],
        declare_number=False,
    )
    calls: dict[int, int] = {}
    original = WyckoffBranch.nearest_parameters_float

    def counted(self: WyckoffBranch, coordinate: tuple[float, ...]) -> tuple[float, ...]:
        calls[id(self)] = calls.get(id(self), 0) + 1
        return original(self, coordinate)

    monkeypatch.setattr(WyckoffBranch, "nearest_parameters_float", counted)
    asu_structure_from_cif(load(str(path), raw=True)["blocks"][0], tolerance=1e-6)

    assert calls == {}


@pytest.mark.parametrize(
    ("number", "point"),
    [(99, (F(1, 7), F(1, 7), F(2, 7))), (149, (F(1, 7), F(6, 7), F(0)))],
)
def test_general_screen_keeps_equal_and_opposite_coordinate_positions(
    number: int, point: tuple[fractions.Fraction, ...]
) -> None:
    standard = Spacegroup.standard(number)
    screen = _general_position_screen(standard, Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1e-8)

    assert screen is not None
    assert not _definitely_general(FracVector(point), screen)


@pytest.mark.parametrize(("setting_name", "letter"), [("15:c1", "e"), ("166:R", "c"), ("224:1", "e")])
def test_general_screen_uses_setting_local_wyckoff_rules(setting_name: str, letter: str) -> None:
    setting = Spacegroup.from_setting(setting_name)
    position = setting.wyckoff_position(letter)
    point = position.representative.coordinate([F(1, 7)] * position.free_count)
    screen = _general_position_screen(setting, Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1e-8)

    assert screen is not None
    assert not _definitely_general(point, screen)


@pytest.mark.extended
def test_general_screen_keeps_every_tabulated_setting_special_branch() -> None:
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    parameters = (F(1, 7), F(2, 11), F(3, 13))
    for record in data.spacegroup_settings():
        setting = Spacegroup(record)
        screen = _general_position_screen(setting, cell, 1e-8)
        assert screen is not None
        for position in setting.wyckoff:
            if position.free_count == 3:
                continue
            for branch in position.branches:
                point = branch.coordinate(parameters[: position.free_count]).normalize()
                assert not _definitely_general(point, screen), (setting.setting, position.letter, branch.operation)


def test_cif_proof_reuses_deduplicated_orbits_for_representatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("CIF construction regenerated a representative orbit")

    monkeypatch.setattr(FundamentalDomainStructure, "_representatives_for_site", fail)
    structure = load(str(_rocksalt_cif(tmp_path)))

    assert structure.multiplicities() == (4, 4)


def test_cif_proof_checks_representatives_against_exact_expansion() -> None:
    site = WyckoffSite("a", FracVector(()), "Na", representative=FracVector((F(1, 4), 0, 0)))
    expansion = (FracVector([[0, 0, 0]]), ("Na",), (1,))

    with pytest.raises(ValueError, match="representative coordinate"):
        _ValidatedASUProof._issue_from_cif_deduplication(
            Spacegroup.standard(221), SettingTransform.identity(), (site,), expansion, None
        )


@pytest.mark.parametrize(
    ("coordinates", "species_at_sites", "counts", "message"),
    [
        (FracVector([[0, 0]]), ("Cs",), (1,), "shape"),
        (FracVector([[0, 0, 0], [F(1, 2), 0, 0]]), ("Cs", "O"), (2,), "species"),
    ],
)
def test_cif_proof_rejects_malformed_expansion(
    coordinates: FracVector,
    species_at_sites: tuple[str, ...],
    counts: tuple[int, ...],
    message: str,
) -> None:
    site = WyckoffSite("a", FracVector(()), "Cs", representative=FracVector((0, 0, 0)))

    with pytest.raises(ValueError, match=message):
        _ValidatedASUProof._issue_from_cif_deduplication(
            Spacegroup.standard(221),
            SettingTransform.identity(),
            (site,),
            (coordinates, species_at_sites, counts),
            None,
        )


def test_cif_proof_is_not_directly_constructible_and_binds_context() -> None:
    group = Spacegroup.standard(221)
    transform = SettingTransform.identity()
    site = WyckoffSite("a", FracVector(()), "Cs", representative=FracVector((0, 0, 0)))
    proof = _ValidatedASUProof._issue_from_cif_deduplication(
        group, transform, (site,), (FracVector([[0, 0, 0]]), ("Cs",), (1,)), None
    )
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    species = (Species("Cs", ("Cs",), (1,)),)

    with pytest.raises(TypeError):
        _ValidatedASUProof(group, transform, (site,), proof.expansion, None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Wyckoff sites"):
        ASUStructure(
            cell,
            group,
            (WyckoffSite("a", FracVector(()), "Cs", representative=FracVector((F(1, 4), 0, 0))),),
            species,
            _validated_proof=proof,
        )
    with pytest.raises(ValueError, match="structure context"):
        ASUStructure(cell, Spacegroup.standard(222), (site,), species, transform, None, _validated_proof=proof)
    with pytest.raises(ValueError, match="structure context"):
        ASUStructure(
            cell,
            group,
            (site,),
            species,
            SettingTransform(FracVector.eye((3, 3)), (F(1, 8), F(1, 8), F(1, 8))),
            None,
            _validated_proof=proof,
        )


def test_rational_uncertainty_metric_is_exactly_equivalent(tmp_path: Path) -> None:
    block = load(str(_rocksalt_cif(tmp_path)), raw=True)["blocks"][0]
    metric = _cell_from_cif(block).metric()
    assert metric.is_rational
    rational = metric.coefficient(1)

    for index in range(len(block["positions_exact"])):
        assert _site_uncertainty(block, index, rational) == _site_uncertainty(block, index, metric)


@pytest.mark.parametrize(
    ("raw", "symbol", "charge"),
    [
        ("Ca2+", "Ca", F(2)),
        ("O2-", "O", F(-2)),
        ("Cu+", "Cu", F(1)),
        ("Ti0", "Ti", F(0)),
        ("Ti", "Ti", None),
        ("D0", "H", F(0)),
        ("O-2", "O", F(-2)),
        ("Na+1", "Na", F(1)),
        ("P+5", "P", F(5)),
    ],
)
def test_cif_type_symbol_parsing(raw: str, symbol: str, charge: fractions.Fraction | None) -> None:
    assert _parse_type_symbol(raw) == (symbol, charge)


def test_decorated_cif_symbols_load_as_species_charges() -> None:
    structure = load(str(Path(__file__).with_name("fixtures") / "oxidation_states.cif"))

    assert {species.name: species.charges for species in structure.species} == {
        "Ca2+": (F(2),),
        "O1-": (F(-1),),
        "Cu1+": (F(1),),
        "Ti0": (F(0),),
    }


def test_plain_cif_symbols_leave_charges_unstated(tmp_path: Path) -> None:
    structure = load(str(_rocksalt_cif(tmp_path)))
    assert {species.name: species.charges for species in structure.species} == {"Na": None, "Cl": None}


def test_redundant_identical_cif_sites_are_deduplicated() -> None:
    structure = load(str(Path(__file__).with_name("fixtures") / "redundant_cif_sites.cif"))
    assert len(structure.sites) == 1


def test_coincident_cif_sites_with_different_species_are_rejected(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "conflict.cif",
        Spacegroup.standard(1).setting,
        (1, 1, 1, 90, 90, 90),
        [("Ca1", "Ca2+", ("0", "0", "0"), "1"), ("O1", "O1-", ("0", "0", "0"), "1")],
        name="Conflict",
    )
    with pytest.raises(ValueError, match=r"CIF block 'conflict'.*co-located sites.*occupancies sum to 2"):
        _ = load(str(path)).sites


def test_repair_never_drops_an_overoccupied_coincident_site(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "conflict.cif",
        Spacegroup.standard(1).setting,
        (1, 1, 1, 90, 90, 90),
        [("Ca1", "Ca2+", ("0", "0", "0"), "1"), ("O1", "O1-", ("0", "0", "0"), "1")],
        name="Conflict",
    )

    with (
        caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"),
        pytest.raises(ValueError, match=r"co-located sites.*occupancies sum to 2"),
    ):
        load(str(path), repair=True)

    assert caplog.records == []


def test_repair_clamps_an_individual_refined_occupancy(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "refined.cif",
        Spacegroup.standard(1).setting,
        (1, 1, 1, 90, 90, 90),
        [("O1", "O", ("0", "0", "0"), "1.013")],
        name="Refined",
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load(path)
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        repaired = load(path, repair=True)

    assert repaired.species[0].concentration == (1,)
    assert any("clamped site 'O1' occupancy" in record.getMessage() for record in caplog.records)


def test_coincident_partial_sites_form_one_mixed_species(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "mixed.cif",
        Spacegroup.standard(1).setting,
        (1, 1, 1, 90, 90, 90),
        [("Ca1", "Ca2+", ("0", "0", "0"), ".25"), ("O1", "O1-", ("0", "0", "0"), ".75")],
        name="Mixed",
    )

    asu = load(str(path))

    assert [(site.wyckoff, site.species) for site in asu.wyckoff_sites] == [("a", "Ca1/O1")]
    assert len(asu.species) == 1
    species = asu.species[0]
    assert species.chemical_symbols == ("Ca", "O")
    assert species.concentration == (F(1, 4), F(3, 4))
    assert species.charges == (F(2), F(-1))
    assert species.labels == ("Ca1", "O1")
    assert species.normalized


def test_disordered_217_is_read_without_losing_chemistry() -> None:
    asu = load(str(Path(__file__).with_name("fixtures") / "disorder" / "217.cif"))

    assert [(site.wyckoff, site.species) for site in asu.wyckoff_sites] == [("c", "B1"), ("c", "B2/Li1")]
    assert [(species.name, species.chemical_symbols, species.concentration) for species in asu.species] == [
        ("B1", ("B", "vacancy"), (F(1, 8), F(7, 8))),
        ("B2/Li1", ("B", "Li"), (F(3, 8), F(5, 8))),
    ]
    assert asu.chemical_formula_reduced == "B4Li5"
    assert len(UnitcellStructureView(asu).sites) == 16


def test_disordered_217_read_write_read_preserves_species_and_orbits(tmp_path: Path) -> None:
    source = load(str(Path(__file__).with_name("fixtures") / "disorder" / "217.cif"))
    destination = tmp_path / "217.cif"

    save(source, destination)
    restored = load(destination)

    assert restored.species == source.species
    assert restored.wyckoff_sites == source.wyckoff_sites
    assert restored.spacegroup == source.spacegroup
    assert restored.cell == source.cell


def test_conditional_attached_hydrogens_become_an_assembly(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "fluoride-hydroxide.cif",
        "1",
        (5, 5, 5, 90, 90, 90),
        [
            ("F1", "F", ("0.1", "0.2", "0.3"), "0.75"),
            ("O1", "O", ("0.1", "0.2", "0.3"), "0.25"),
        ],
        wyckoff_labels=["a", "a"],
        attached_hydrogens=["0", "1"],
    )

    structure = load(path)
    viewed = ASUStructureView(path).unview()
    by_name = {species.name: species for species in structure.species}

    assert viewed.assemblies == structure.assemblies
    assert structure.domain_species_at_sites == ("F1", "O1")
    assert by_name["F1"].concentration == by_name["O1"].concentration == (F(1),)
    assert by_name["F1"].attached is None
    assert (by_name["O1"].attached, by_name["O1"].nattached) == (("H",), (1,))
    assert structure.assemblies is not None
    assert structure.assemblies[0].sites_in_groups == ((0,), (1,))
    assert structure.assemblies[0].group_probabilities == (F(3, 4), F(1, 4))
    assert structure.composition.amounts == (("F", F(3, 4)), ("H", F(1, 4)), ("O", F(1, 4)))
    assert structure.structure_features == ("assemblies", "site_attachments")

    with pytest.raises(TypeError, match="cannot be represented as CIF because it has assemblies"):
        save(structure, tmp_path / "unsupported.cif")


# --- reading ---


def test_cif_expands_to_the_full_cell(tmp_path: Path) -> None:
    asu = load(str(_rocksalt_cif(tmp_path)))
    assert asu.spacegroup.it_number == 225
    assert [(site.wyckoff, site.species) for site in asu.wyckoff_sites] == [("a", "Na"), ("b", "Cl")]

    structure = UnitcellStructureView(asu)
    assert len(structure.sites) == 8
    assert sorted(structure.species_at_sites) == ["Cl"] * 4 + ["Na"] * 4
    # Exact, not approximate: expansion never leaves the rationals.
    assert {tuple(row) for row in structure.sites.reduced_coords.to_fractions()} == {
        (F(0), F(0), F(0)),
        (F(0), F(1, 2), F(1, 2)),
        (F(1, 2), F(0), F(1, 2)),
        (F(1, 2), F(1, 2), F(0)),
        (F(0), F(0), F(1, 2)),
        (F(0), F(1, 2), F(0)),
        (F(1, 2), F(0), F(0)),
        (F(1, 2), F(1, 2), F(1, 2)),
    }


@pytest.mark.skipif(
    decimal_precision("0") is not None,
    reason="requires httk-core integer-literals-are-exact (unreleased)",
)
def test_integer_coordinate_tokens_do_not_swallow_the_second_rocksalt_orbit(tmp_path: Path) -> None:
    asu = load(str(_rocksalt_integer_cif(tmp_path)))
    assert [(site.wyckoff, site.species) for site in asu.wyckoff_sites] == [("a", "Na"), ("b", "Cl")]
    assert len(UnitcellStructureView(asu).sites) == 8


def test_loading_fidelity_oracle(tmp_path: Path) -> None:
    path = str(_sg15_cif(tmp_path, declaration="_space_group_IT_number 15\n"))

    asu = load(path)
    full = UnitcellStructureView(load(path))
    assert asu.spacegroup.it_number == 15
    assert asu.spacegroup.setting == "15:b1"
    assert [(site.wyckoff, site.free_params.to_fractions()) for site in asu.wyckoff_sites] == [("e", [F(3333, 10000)])]
    assert full.cell.basis.to_floats() == [[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]]
    assert full.sites.reduced_coords.to_fractions() == [
        [F(0), F(3333, 10000), F(1, 4)],
        [F(0), F(6667, 10000), F(3, 4)],
        [F(1, 2), F(1667, 10000), F(3, 4)],
        [F(1, 2), F(8333, 10000), F(1, 4)],
    ]
    assert [(species.name, species.chemical_symbols, species.concentration) for species in full.species] == [
        ("Si", ("Si",), (F(1),))
    ]
    assert full.species_at_sites == ("Si", "Si", "Si", "Si")


def test_core_load_adapts_single_cif_and_raw_keeps_payload(tmp_path: Path) -> None:
    path = _rocksalt_cif(tmp_path)
    structure = load(str(path))
    assert isinstance(structure, ASUStructure)
    payload = load(str(path), raw=True)
    assert payload["format"] == "cif"


def test_clean_cif_is_unchanged_when_repair_is_disabled(tmp_path: Path) -> None:
    path = _rocksalt_cif(tmp_path)
    assert load(str(path), raw=True) == load(str(path), raw=True, repair=False)


def test_clean_cif_load_is_unchanged_by_repair(tmp_path: Path) -> None:
    path = _rocksalt_cif(tmp_path)
    assert load(str(path), repair=True) == load(str(path))


def test_the_cell_is_exact_not_the_files_rounded_basis(tmp_path: Path) -> None:
    """Built from a, b, c and the angles, so a cubic cell keeps exact right angles.

    The mapping also carries a pre-multiplied floating-point basis whose off-diagonal
    entries are ~3e-16 rather than zero; using it would put that noise into every
    structure.
    """
    asu = load(str(_rocksalt_cif(tmp_path)))
    assert asu.cell.angles == (F(90), F(90), F(90))
    assert asu.cell.lengths[0] == asu.cell.lengths[1] == asu.cell.lengths[2]


# --- settings ---


def test_a_non_standard_setting_is_recognized_as_itself(tmp_path: Path) -> None:
    """Identified from the symmetry operations, so the file is not silently reinterpreted."""
    path = _write_cif(
        tmp_path / "sg15.cif",
        "15:c1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.25", "0.0", "0.3333"), "1.0")],
        name="SG15c1",
    )
    asu = load(str(path))
    setting = asu.setting()
    assert setting is not None
    assert setting.setting == "15:c1"
    assert not asu.is_standard_setting
    assert asu.spacegroup.setting == "15:c1"
    assert asu.transform.is_identity()


def test_the_setting_is_found_even_when_the_file_declares_nothing(tmp_path: Path) -> None:
    """With no symbol to narrow the search, the operations alone still identify it."""
    path = _write_cif(
        tmp_path / "bare.cif",
        "15:c1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.25", "0.0", "0.3333"), "1.0")],
        declare_number=False,
    )
    assert cif_setting(load(str(path), raw=True)["blocks"][0]).setting == "15:c1"


@pytest.mark.extended
def test_an_unidentifiable_setting_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """A transform cannot be derived; infinitely many are valid and they differ."""
    path = _rocksalt_cif(tmp_path)
    block = dict(load(str(path), raw=True)["blocks"][0])
    # Drop most of the operations, so the set matches no tabulated group, and drop the
    # declaration too so the failure is about the operations rather than a contradiction.
    block["symops_xyz"] = block["symops_xyz"][:3]
    block["space_group_nbr"] = None
    block["space_group_name_hall"] = None
    block["space_group_name_hm"] = None
    with pytest.raises(ValueError, match="no tabulated space-group setting"):
        asu_structure_from_cif(block)

    # Ignoring the declaration does not rescue it: there is genuinely no such setting.
    with pytest.raises(ValueError, match="no tabulated space-group setting"):
        asu_structure_from_cif(block, trust_declared_symmetry=False)


def test_a_block_with_no_symmetry_operations_is_refused(tmp_path: Path) -> None:
    path = _rocksalt_cif(tmp_path)
    block = dict(load(str(path), raw=True)["blocks"][0])
    block["symops_xyz"] = []
    with pytest.raises(ValueError, match="no symmetry operations"):
        asu_structure_from_cif(block)


# --- fidelity ---


def test_occupancies_survive_into_the_structure(tmp_path: Path) -> None:
    """They were parsed and then dropped before; a half-occupied site is not a full one."""
    path = _write_cif(
        tmp_path / "partial.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0.0", "0.0", "0.0"), "0.5"), ("Cl1", "Cl", ("0.5", "0.5", "0.5"), "1.0")],
        name="Partial",
    )
    asu = load(str(path))
    species = {value.name: value for value in asu.species}
    assert species["Na1"].chemical_symbols == ("Na", "vacancy")
    assert species["Na1"].concentration == (F(1, 2), F(1, 2))
    assert species["Cl"].chemical_symbols == ("Cl",)
    assert species["Cl"].concentration == (F(1),)
    # A partially occupied site is named for its CIF label, since two sites of one element
    # can carry different occupancies.
    assert [site.species for site in asu.wyckoff_sites] == ["Na1", "Cl"]


def test_neutral_exact_occupancies_preserve_central_values_and_precision(tmp_path: Path) -> None:
    """The atomistic adapter consumes the neutral exact fields from the CIF reader payload."""
    payload = dict(load(str(_rocksalt_cif(tmp_path)), raw=True)["blocks"][0])
    payload["occupancies"] = [0.5, 1 / 3]
    payload["occupancies_exact"] = ["0.5000", "1/3"]
    payload["occupancy_precisions"] = [F(7, 10000), None]

    asu = asu_structure_from_cif(payload)
    concentrations = {species.name: species for species in asu.species}
    assert concentrations["Na1"].chemical_symbols == ("Na", "vacancy")
    assert concentrations["Na1"].concentration == (F(1, 2), F(1, 2))
    assert concentrations["Na1"].concentration_precision == (F(7, 10000), F(7, 10000))
    assert concentrations["Cl1"].chemical_symbols == ("Cl", "vacancy")
    assert concentrations["Cl1"].concentration == (F(1, 3), F(2, 3))
    assert concentrations["Cl1"].concentration_precision == (None, None)


def test_neutral_missing_occupancy_is_not_treated_as_full_occupancy(tmp_path: Path) -> None:
    payload = dict(load(str(_rocksalt_cif(tmp_path)), raw=True)["blocks"][0])
    payload["occupancies"] = [None, 1.0]
    payload["occupancies_exact"] = [None, "1"]
    payload["occupancy_precisions"] = [None, F(1)]
    with pytest.raises(ValueError, match="occupancy is missing.*Na1"):
        asu_structure_from_cif(payload)

    payload.pop("occupancies")
    payload.pop("occupancies_exact")
    payload.pop("occupancy_precisions")
    asu = asu_structure_from_cif(payload)
    assert all(species.concentration == (F(1),) for species in asu.species)


def test_coordinates_embed_as_the_decimal_the_file_wrote(tmp_path: Path) -> None:
    """``0.3333`` is 3333/10000, not the binary value of ``float("0.3333")``.

    The free parameter keeps what the file said; only the position's fixed components are
    replaced by their exact values.
    """
    path = _write_cif(
        tmp_path / "sg15.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.0", "0.3333", "0.25"), "1.0")],
    )
    asu = load(str(path))
    assert asu.wyckoff_sites[0].wyckoff == "e"
    assert asu.wyckoff_sites[0].free_params.to_fractions() == [F(3333, 10000)]


def test_uncertainties_are_stripped_from_coordinates(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "esd.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.0", "0.3333(7)", "0.25"), "1.0")],
    )
    assert load(str(path)).wyckoff_sites[0].free_params.to_fractions() == [F(3333, 10000)]


def test_a_site_on_no_special_position_falls_back_to_the_general_one(tmp_path: Path) -> None:
    """A CIF site can always be placed, because the general position accepts any point.

    That is worth stating rather than assuming: it means reading a CIF never fails for want
    of a matching Wyckoff position, and a site that is not on any special position simply
    generates the full orbit of the general position — which is exactly what the file's own
    symmetry operations would generate from it.
    """
    path = _write_cif(
        tmp_path / "general.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0.3", "0.11", "0.07"), "1.0")],
    )
    asu = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0], tolerance=0.0)
    general = Spacegroup.standard(225).wyckoff[-1]
    assert general.free_count == 3
    assert asu.wyckoff_sites[0].wyckoff == general.letter
    assert len(UnitcellStructureView(asu).sites) == general.multiplicity


# --- payload handling ---


def test_asu_structures_from_cif_reports_why_a_file_yielded_nothing(tmp_path: Path) -> None:
    """An empty result must not read as "this file contained no structures"."""
    path = tmp_path / "incomplete.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0\nloop_\n_atom_site_label\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNa 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    payload = load(str(path), raw=True)
    assert payload["blocks"] == []
    with pytest.raises(ValueError, match="CIF block 'x', CIF block has no unit cell"):
        asu_structures_from_cif(payload)


def test_a_multi_block_cif_yields_one_structure_per_block(tmp_path: Path) -> None:
    first = _rocksalt_cif(tmp_path).read_text(encoding="utf-8")
    second = _write_cif(
        tmp_path / "second.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.0", "0.3333", "0.25"), "1.0")],
        name="second",
    ).read_text(encoding="utf-8")
    combined = tmp_path / "both.cif"
    combined.write_text(first + second, encoding="utf-8")

    structures = asu_structures_from_cif(load(str(combined), raw=True))
    assert [structure.spacegroup.it_number for structure in structures] == [225, 15]

    with pytest.raises(ValueError, match="holds 2 structures"):
        load(str(combined))


def test_precision_stays_with_valid_block_after_unparsed_block(tmp_path: Path) -> None:
    invalid = """data_invalid
_cell_length_a 0.5
_cell_length_b 0.5
_cell_length_c 0.5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si_bad Si 0.9(9) 0.9 0.9
"""
    valid = _write_cif(
        tmp_path / "valid.cif",
        Spacegroup.standard(1).setting,
        (0.5, 0.5, 0.5, 90, 90, 90),
        [("Si1", "Si", ("1.1", "1/3", "1/3"), "1")],
        name="valid",
    ).read_text(encoding="utf-8")
    path = tmp_path / "mixed.cif"
    path.write_text(invalid + valid, encoding="utf-8")

    with collect_reports(level="debug") as collection:
        structure = load(str(path))

    assert len(structure.sites) == 1
    assert not any(record.levelno >= logging.WARNING for record in collection.records)
    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.DEBUG
    assert "maximum is 0.1 Å" in collection.records[0].getMessage()


def test_a_non_cif_payload_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected a 'cif' mapping"):
        asu_structure_from_cif({"format": "vasp-poscar"})


# --- what the file declares is checked, not merely used as a hint ---


def _sg15_cif(tmp_path: Path, *, declaration: str) -> Path:
    """SG 15 operations, with whatever space-group declaration is passed."""
    spacegroup = Spacegroup.from_setting("15:b1")
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "declared.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0\n_cell_length_b 6.0\n_cell_length_c 7.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        + declaration
        + f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si 0.0 0.3333 0.25\n",
        encoding="utf-8",
    )
    return path


def test_a_conventionally_spelled_hall_symbol_is_recognized(tmp_path: Path) -> None:
    """CIFs write ``-C 2yc``; the tables key it ``-c_2yc``.

    Without normalizing, every correctly declared Hall symbol looks unknown — which used to
    be survivable only because the miss was silent, and would now be an error.
    """
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_name_Hall '-C 2yc'\n")), raw=True)["blocks"][0]
    assert cif_setting(block).setting == "15:b1"


def test_normalized_hermann_mauguin_lookup() -> None:
    assert [_normalized_hm(symbol) for symbol in ("P4_2cm", "P 42 c m", "P42cm")] == ["p42cm"] * 3
    assert _hm_it_numbers()["p42cm"] == 101


@pytest.mark.parametrize(("filename", "it_number"), [("93.cif", 93), ("101.cif", 101)])
def test_materials_project_hm_declarations_load_without_repair(filename: str, it_number: int) -> None:
    path = Path(__file__).parent / "fixtures" / "structreading" / filename
    raw = load(str(path), raw=True)
    assert raw["blocks"][0]["space_group_nbr"] == str(it_number)
    assert load(str(path)).spacegroup.it_number == it_number


def test_setting_tables_are_shared_by_immutable_setting_identity() -> None:
    first = Spacegroup.standard(93)
    second = Spacegroup.standard(93)
    assert first.wyckoff is second.wyckoff
    assert first.symmetry_operations is second.symmetry_operations


def test_multiplicity_only_site_loads_and_is_identified_from_coordinates(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "multiplicity-only.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.000000", "0.333300", "0.250000"), "1")],
        declare_number=False,
        symmetry_multiplicities=["4"],
    )
    structure = load(str(path))
    assert structure.setting().setting == "15:b1"
    assert [(site.wyckoff, site.species) for site in structure.wyckoff_sites] == [("e", "Si")]


def test_multiplicity_only_filters_nearby_special_position(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "multiplicity-filter.cif",
        "2",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", ("0.075", "0", "0"), "1")],
        declare_number=False,
        symmetry_multiplicities=["2"],
    )
    structure = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0], tolerance=0.1)
    assert [(site.wyckoff, site.species) for site in structure.wyckoff_sites] == [("i", "X")]
    assert len(UnitcellStructureView(structure).sites) == 2


def test_multiplicity_only_with_no_matching_position_is_invalid(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "multiplicity-invalid.cif",
        "2",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", ("0.1", "0.2", "0.3"), "1")],
        declare_number=False,
        symmetry_multiplicities=["3"],
    )
    with pytest.raises(ValueError, match="unknown setting-local multiplicity '3'"):
        load(str(path))


def test_multiplicity_only_filter_mismatch_is_a_declaration_error(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "multiplicity-mismatch.cif",
        "2",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", ("0.100000", "0.200000", "0.300000"), "1")],
        declare_number=False,
        symmetry_multiplicities=["1"],
    )
    with pytest.raises(ValueError, match=r"invalid declaration .*multiplicity '1'"):
        load(str(path))


@pytest.mark.parametrize("repair", (False, True))
def test_deprecated_multiplicity_is_ignored_with_one_debug_note(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, repair: bool
) -> None:
    path = _write_cif(
        tmp_path / "deprecated-only.cif",
        "14:b1",
        (5, 6, 7, 90, 90, 90),
        [
            ("X1", "X", ("0", "0", "0"), "1"),
            ("X2", "X", ("0.123", "0.234", "0.345"), "1"),
        ],
        name="DeprecatedOnly",
        deprecated_symmetry_multiplicities=["1", "1"],
    )
    with caplog.at_level(logging.DEBUG, logger="httk.atomistic.cif_structures"):
        structure = load(str(path), repair=repair)

    assert [site.wyckoff for site in structure.wyckoff_sites] == ["a", "e"]
    records = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(records) == 1
    assert records[0].context == "cif"
    assert "_atom_site_symmetry_multiplicity" in records[0].getMessage()
    assert "deprecatedonly" in records[0].getMessage()


def test_deprecated_multiplicity_is_silent_when_a_modern_tag_is_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_cif(
        tmp_path / "deprecated-with-modern.cif",
        "14:b1",
        (5, 6, 7, 90, 90, 90),
        [("X1", "X", ("0", "0", "0"), "1")],
        name="DeprecatedWithModern",
        symmetry_multiplicities=["2"],
        deprecated_symmetry_multiplicities=["1"],
    )
    with caplog.at_level(logging.DEBUG, logger="httk.atomistic.cif_structures"):
        structure = load(str(path))

    assert [site.wyckoff for site in structure.wyckoff_sites] == ["a"]
    assert not any("_atom_site_symmetry_multiplicity" in record.getMessage() for record in caplog.records)


def test_site_symmetry_order_only_selects_its_wyckoff_stratum(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "order-only.cif",
        "2",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", ("0.075", "0", "0"), "1")],
        declare_number=False,
        site_symmetry_orders=["1"],
    )
    structure = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0], tolerance=0.1)
    assert [(site.wyckoff, site.species) for site in structure.wyckoff_sites] == [("i", "X")]
    assert len(UnitcellStructureView(structure).sites) == 2


def test_impossible_site_symmetry_order_is_invalid_or_repaired(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_cif(
        tmp_path / "order-invalid.cif",
        "14:b1",
        (5, 6, 7, 90, 90, 90),
        [("X1", "X", ("0.123", "0.234", "0.345"), "1")],
        site_symmetry_orders=["3"],
    )
    with pytest.raises(ValueError, match=r"invalid setting-local site-symmetry order '3'.*Remedy: load"):
        load(str(path))
    with caplog.at_level(logging.WARNING, logger="httk.atomistic.cif_structures"):
        structure = load(str(path), repair=True)
    assert [site.wyckoff for site in structure.wyckoff_sites] == ["e"]
    assert "ignored declared Wyckoff data" in caplog.records[0].getMessage()
    assert "site-symmetry order '3'" in caplog.records[0].getMessage()


def test_site_symmetry_order_and_label_conflict_is_invalid(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "order-label-conflict.cif",
        "14:b1",
        (5, 6, 7, 90, 90, 90),
        [("X1", "X", ("0", "0", "0"), "1")],
        wyckoff_labels=["a"],
        site_symmetry_orders=["1"],
    )
    with pytest.raises(ValueError, match="the declared letter and site-symmetry order identify different positions"):
        load(str(path))


def test_site_symmetry_order_and_multiplicity_conflict_is_invalid(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "order-multiplicity-conflict.cif",
        "14:b1",
        (5, 6, 7, 90, 90, 90),
        [("X1", "X", ("0", "0", "0"), "1")],
        symmetry_multiplicities=["4"],
        site_symmetry_orders=["2"],
    )
    with pytest.raises(
        ValueError, match="the declared multiplicity and site-symmetry order identify different positions"
    ):
        load(str(path))


def test_repair_drops_mismatching_multiplicity_filter(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "multiplicity-mismatch-repair.cif",
        "2",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", ("0.100000", "0.200000", "0.300000"), "1")],
        declare_number=False,
        symmetry_multiplicities=["1"],
    )
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        structure = load(str(path), repair=True)
    assert [(site.wyckoff, site.species) for site in structure.wyckoff_sites] == [("i", "X")]
    assert len(UnitcellStructureView(structure).sites) == 2
    assert "ignored declared Wyckoff data" in caplog.records[0].getMessage()
    assert "multiplicity '1'" in caplog.records[0].getMessage()


def test_hm_only_recognized_declaration_loads_strictly(tmp_path: Path) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_name_H-M_alt 'C 1 2/c 1'\n")
    block = load(str(path), raw=True)["blocks"][0]
    assert cif_setting(block).setting == "15:b1"
    assert load(str(path)).setting().setting == "15:b1"


def test_hm_only_contradiction_is_rejected(tmp_path: Path) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_name_H-M_alt 'P 1'\n")
    block = load(str(path), raw=True)["blocks"][0]
    with pytest.raises(ValueError, match="declares Hermann-Mauguin symbol"):
        cif_setting(block)


def test_hm_only_unknown_declaration_is_ignored(tmp_path: Path) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_name_H-M_alt 'F m -3 m:1'\n")
    assert load(str(path)).setting().setting == "15:b1"


def test_a_hall_symbol_naming_no_setting_is_an_error(tmp_path: Path) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_name_Hall 'Not A Symbol'\n")), raw=True)["blocks"][0]
    with pytest.raises(ValueError, match=r"names no known space-group setting.*Remedy: load\(\.\.\., repair=True\)"):
        cif_setting(block)


def test_repair_ignores_an_unrecognized_declared_setting(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_name_Hall 'Not A Symbol'\n")

    with pytest.raises(ValueError, match=r"Remedy: load\(\.\.\., repair=True\)"):
        load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        asu = load(str(path), repair=True)

    assert asu.setting() is not None
    assert asu.setting().setting == "15:b1"
    assert len(UnitcellStructureView(asu).sites) == 4
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings == [
        (
            "CIF block 'x': ignored declared symmetry Hall symbol 'Not A Symbol' and identified setting '15:b1' "
            "from its symmetry operations"
        )
    ]


def test_unrecognized_declaration_identifies_sg_228_from_operations(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "sg228.cif",
        "228:2",
        (10, 10, 10, 90, 90, 90),
        [("X1", "X", ("0", "0", "0"), "1")],
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "loop_\n_space_group_symop_operation_xyz",
            "_space_group_name_Hall 'Not A Symbol'\nloop_\n_space_group_symop_operation_xyz",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="names no known space-group setting"):
        load(str(path))
    assert load(str(path), repair=True).setting().setting == "228:2"


def test_repair_stamp_preserves_strict_precision_snap(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "rounded.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.33333", "0.66667", "0.50000"), "1")],
        name="Rounded",
    )
    assert len(UnitcellStructureView(load(str(path))).sites) == 1

    payload = load(str(path), raw=True, repair=True)
    assert payload["repair"] is True
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        asu = asu_structures_from_cif(payload)[0]

    assert [(site.wyckoff, site.species) for site in asu.wyckoff_sites] == [("d", "N")]
    assert len(UnitcellStructureView(asu).sites) == 1
    assert caplog.records == []


def test_partial_occupancy_uses_the_same_precision_snap(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "split.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.33333", "0.66667", "0.50000"), "0.5")],
        name="Split",
    )
    strict = load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert len(UnitcellStructureView(strict).sites) == len(UnitcellStructureView(corrected).sites) == 1
    assert caplog.records == []


def test_declared_special_position_is_assigned_strictly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "declared-special.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.33333", "0.66667", "0.50000"), "0.5")],
        name="DeclaredSpecial",
        wyckoff_labels=["d"],
        symmetry_multiplicities=["1"],
    )

    strict = load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert len(UnitcellStructureView(strict).sites) == len(UnitcellStructureView(corrected).sites) == 1
    assert caplog.records == []


def test_repair_keeps_a_declared_general_orbit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "declared-general.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.20000", "0.80000", "0.50000"), "1")],
        name="DeclaredGeneral",
        wyckoff_labels=["k"],
        symmetry_multiplicities=["3"],
    )
    strict = load(str(path))
    payload = load(str(path), raw=True, repair=True)
    assert payload["blocks"][0]["_httk_atomistic_wyckoff_labels"] == ["k"]
    assert payload["blocks"][0]["_httk_atomistic_symmetry_multiplicities"] == ["3"]
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = asu_structures_from_cif(payload)[0]

    assert len(UnitcellStructureView(strict).sites) == len(UnitcellStructureView(corrected).sites) == 3
    assert caplog.records == []


def test_invalid_declared_position_is_an_integrity_error_or_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_cif(
        tmp_path / "bad-declaration.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.10000", "0.20000", "0.30000"), "1")],
        name="BadDeclaration",
        wyckoff_labels=["d"],
        symmetry_multiplicities=["1"],
    )

    with pytest.raises(
        ValueError, match=r"N1.*invalid declaration.*measured distance.*Remedy: load\(\.\.\., repair=True\)"
    ):
        load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert len(UnitcellStructureView(corrected).sites) == 6
    assert len(caplog.records) == 1
    assert "ignored declared Wyckoff data" in caplog.records[0].getMessage()


def test_invalid_declaration_fallback_uses_the_undeclared_rounded_site_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_cif(
        tmp_path / "unknown-declaration.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.33333", "0.66667", "0.50000"), "1")],
        name="UnknownDeclaration",
        wyckoff_labels=["z"],
    )

    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert len(UnitcellStructureView(corrected).sites) == 1
    warnings = [record.getMessage() for record in caplog.records]
    assert any("Wyckoff label 'z'" in warning and "Wyckoff position 'd'" in warning for warning in warnings)
    assert len(warnings) == 1


@pytest.mark.parametrize("value", ("", "?"))
def test_empty_wyckoff_declaration_entries_are_absent(value: str) -> None:
    assert _site_declaration([value], 0) is None


def test_declared_containing_position_is_an_integrity_error_or_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_cif(
        tmp_path / "containing-position.cif",
        Spacegroup.standard(149).setting,
        (4.7241, 4.7241, 4.3862, 90, 90, 120),
        [("N1", "N", ("0.333333", "0.666667", "0.500000"), "1")],
        name="ContainingPosition",
        wyckoff_labels=["k"],
        symmetry_multiplicities=["3"],
    )

    with pytest.raises(ValueError, match=r"declares Wyckoff position 'k'.*more-specific Wyckoff position 'd'"):
        load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert len(UnitcellStructureView(corrected).sites) == 1
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert "more-specific Wyckoff position 'd'" in warnings[0]


def test_declared_position_is_kept_when_no_more_specific_position_matches(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "near-special.cif",
        "28:a-cb",
        (5.332, 11.13, 5.455, 90, 90, 90),
        [("Ca2", "Ca", ("0.24", "0.183", "0.7"), "0.5")],
        wyckoff_labels=["d"],
    )

    assert [site.wyckoff for site in load(str(path)).wyckoff_sites] == ["d"]


@pytest.mark.parametrize("setting", ("48:1", "50:1", "50:1bca", "50:1cab", "73:ba-c", "126:1", "142:1", "222:1"))
def test_zero_tolerance_float_screen_keeps_exact_matches(setting: str) -> None:
    spacegroup = Spacegroup.from_setting(setting)
    standard = spacegroup.standard_setting()
    position = standard.wyckoff[-1]
    parameters = FracVector([F(1, 7), F(2, 7), F(3, 7)])
    point = position.representative.coordinate(parameters)
    own = spacegroup.transform_from_standard.to_setting(point).normalize()

    assert _snap(
        standard, point, own, Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), spacegroup.transform_from_standard, 0
    ) == (
        position.letter,
        parameters,
    )


def test_exact_special_position_beats_an_earlier_nearby_position() -> None:
    spacegroup = Spacegroup.standard(149)
    parameters = FracVector([F(49, 100)])
    point = spacegroup.wyckoff_position("h").representative.coordinate(parameters)
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert _snap(spacegroup, point, point, cell, spacegroup.transform_from_standard, 0.02) == ("h", parameters)
    matched_letters: set[str] = set()
    assert _snap(
        spacegroup,
        point,
        point,
        cell,
        spacegroup.transform_from_standard,
        0.02,
        matched_letters=matched_letters,
    ) == ("h", parameters)
    assert matched_letters == {"d", "h", "k", "l"}


def test_snap_collection_keeps_the_first_branch_for_a_letter() -> None:
    spacegroup = Spacegroup.standard(10)
    point = FracVector([F(7, 10), F(19, 20), F(1, 2)])
    matched_letters: set[str] = set()

    match = _snap(
        spacegroup,
        point,
        point,
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        spacegroup.transform_from_standard,
        0.2,
        matched_letters=matched_letters,
    )

    assert match == ("l", FracVector([F(19, 20)]))
    assert matched_letters == {"l", "m", "o"}


def test_unwrapped_large_coordinate_falls_through_the_float_screen(tmp_path: Path) -> None:
    """A binary float loses the fractional part, but exact P1 matching must retain it."""
    path = _write_cif(
        tmp_path / "unwrapped-large-coordinate.cif",
        "1",
        (1, 1, 1, 90, 90, 90),
        [("X1", "X", (f"{2**54}.25", "0", "0"), "1")],
    )

    structure = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0], tolerance=0)

    assert [(site.wyckoff, tuple(site.free_params.to_fractions())) for site in structure.wyckoff_sites] == [
        ("a", (F(1, 4), F(0), F(0)))
    ]


def test_orbit_float_screen_keeps_an_exact_tolerance_boundary_match() -> None:
    """A P -1 pair lies one exact fraction below the float tolerance boundary.

    The two general-position branches are fractionally below the boundary exactly, but the
    double squared distance rounds above it. A screen without its slack would omit exact
    confirmation, so this must still detect the overlap.
    """
    spacegroup = Spacegroup.standard(2)
    position = spacegroup.wyckoff[-1]
    tolerance = 1e-6
    point = FracVector([F(60655832901, 200000000000000000), F(6211242099, 15625000000000000), 0])
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    orbit_screen: list[tuple[tuple[float, float, float], object, FracVector]] = []
    match = _snap(
        spacegroup,
        point,
        point,
        cell,
        spacegroup.transform_from_standard,
        tolerance,
        positions=(position,),
        orbit_screen=orbit_screen,
    )

    assert match is not None
    assert sum((2 * value.to_float()) ** 2 for value in point) > tolerance**2
    assert _has_rounded_orbit_overlap(
        spacegroup, spacegroup.transform_from_standard, match, cell, tolerance, orbit_screen
    )


def test_orbit_float_screen_uses_the_magnitude_of_a_negative_tolerance() -> None:
    """The exact orbit gap is below ``abs(tolerance)`` but above the old negative screen."""
    spacegroup = Spacegroup.standard(2)
    position = spacegroup.wyckoff[-1]
    tolerance = -1e-6
    point = FracVector([F(499999, 10**12), 0, 0])
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    orbit_screen: list[tuple[tuple[float, float, float], object, FracVector]] = []
    match = _snap(
        spacegroup,
        point,
        point,
        cell,
        spacegroup.transform_from_standard,
        tolerance,
        positions=(position,),
        orbit_screen=orbit_screen,
    )

    assert match is not None
    assert _has_rounded_orbit_overlap(
        spacegroup, spacegroup.transform_from_standard, match, cell, tolerance, orbit_screen
    )


def test_repair_compares_declared_letters_in_the_cif_setting(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_cif(
        tmp_path / "setting-local-letter.cif",
        "224:1",
        (10, 10, 10, 90, 90, 120),
        [("X1", "X", ("0.75000", "0.39286", "0.10714"), "1")],
        name="SettingLocalLetter",
        wyckoff_labels=["i"],
    )

    strict = load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert [(site.wyckoff, site.species) for site in strict.wyckoff_sites] == [("i", "X")]
    assert strict.setting().setting == "224:1"
    assert corrected == strict
    assert len(UnitcellStructureView(corrected).sites) == 24
    assert caplog.records == []


def test_repair_compares_declared_multiplicities_in_the_cif_setting(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_cif(
        tmp_path / "setting-local-multiplicity.cif",
        "160:R",
        (10, 10, 10, 90, 90, 120),
        [("X1", "X", ("0.00002", "0.00002", "0.99999"), "0.5")],
        name="SettingLocalMultiplicity",
        wyckoff_labels=["b"],
        symmetry_multiplicities=["3"],
    )

    strict = load(str(path))
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        corrected = load(str(path), repair=True)

    assert [(site.wyckoff, site.species) for site in strict.wyckoff_sites] == [("b", "X1")]
    assert strict.setting().setting == "160:R"
    assert corrected == strict
    assert len(UnitcellStructureView(corrected).sites) == 3
    assert caplog.records == []


def test_a_hall_symbol_naming_the_wrong_group_is_an_error(tmp_path: Path) -> None:
    """SG 14's Hall symbol on a file whose operations are SG 15's: the file contradicts itself."""
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_name_Hall '-P 2ybc'\n")), raw=True)["blocks"][0]
    with pytest.raises(ValueError, match="contradicts itself"):
        cif_setting(block)


def test_repair_does_not_override_a_contradictory_known_declaration(tmp_path: Path) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_name_Hall 'P 1'\n")
    errors = []
    for repair in (False, True):
        with pytest.raises(ValueError) as caught:
            load(str(path), repair=repair)
        errors.append(str(caught.value))

    assert errors[0] == errors[1]
    assert "contradicts itself" in errors[0]
    assert "Remedy:" not in errors[0]


def test_a_wrong_it_number_is_an_error(tmp_path: Path) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_IT_number 14\n")), raw=True)["blocks"][0]
    with pytest.raises(ValueError, match="contradicts itself"):
        cif_setting(block)


@pytest.mark.parametrize(
    ("declaration", "message"),
    [("_space_group_IT_number 999\n", "outside the range"), ("_space_group_IT_number banana\n", "not a\nnumber")],
    ids=["out-of-range", "not-a-number"],
)
def test_an_unusable_it_number_is_an_error(tmp_path: Path, declaration: str, message: str) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration=declaration)), raw=True)["blocks"][0]
    with pytest.raises(ValueError, match=message.replace("\n", " ")):
        cif_setting(block)


@pytest.mark.parametrize(
    "declaration",
    [
        "_space_group_name_Hall 'Not A Symbol'\n",
        "_space_group_name_Hall '-P 2ybc'\n",
        "_space_group_IT_number 14\n",
        "_space_group_IT_number 999\n",
    ],
    ids=["unknown-hall", "wrong-hall", "wrong-number", "out-of-range-number"],
)
def test_the_declaration_can_be_ignored_on_request(tmp_path: Path, declaration: str) -> None:
    """The escape hatch: when the operations are the trustworthy half of the file."""
    block = load(str(_sg15_cif(tmp_path, declaration=declaration)), raw=True)["blocks"][0]
    assert cif_setting(block, trust_declared_symmetry=False).setting == "15:b1"

    asu = asu_structure_from_cif(block, trust_declared_symmetry=False)
    assert asu.spacegroup.it_number == 15
    assert asu.wyckoff_sites[0].wyckoff == "e"


def test_the_escape_hatch_uses_the_raw_load_path(tmp_path: Path) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_IT_number 14\n")
    with pytest.raises(ValueError, match="contradicts itself"):
        load(str(path))
    assert (
        asu_structures_from_cif(load(str(path), raw=True), trust_declared_symmetry=False)[0].spacegroup.it_number == 15
    )


def test_a_file_with_no_declaration_searches_every_setting(tmp_path: Path) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration="")), raw=True)["blocks"][0]
    assert cif_setting(block).setting == "15:b1"
