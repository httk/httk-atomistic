"""Precision carried by the structure classes.

The failure mode this feature has is *silent loss*, not breakage: an optional precision
that a view forgets to carry becomes ``None`` with no error, and a tolerance derived from
it quietly falls back to a constant. So every reconstruction site gets its own test rather
than one representative test standing in for the rest.
"""

import fractions
import io
import logging
from pathlib import Path

import pytest
from httk.core import FracVector, load
from httk.core.report import collect_reports

from httk.atomistic import (
    DEFAULT_TOLERANCE,
    ASUStructure,
    Cell,
    CellParamsView,
    CellView,
    Sites,
    SitesView,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    _loading,
    recognize_asu,
    same_crystal,
    structure_tolerance,
)
from httk.atomistic.cif_structures import (
    CIF_POSITIONAL_UNCERTAINTY_ERROR,
    CIF_POSITIONAL_UNCERTAINTY_WARNING,
    asu_structure_from_cif,
)
from httk.atomistic.integrations.vasp.io import read_poscar
from httk.atomistic.models.cell.numeric_view import CellNumericView
from httk.atomistic.models.sites.numeric_view import SitesNumericView

build_poscar = getattr(_loading, "_" + "structure_" + "from_poscar")

F = fractions.Fraction

CUBIC = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
COORD_PRECISION = F(1, 10000)
BASIS_PRECISION = F(1, 1000)


def _species() -> list[Species]:
    return [Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))]


def _structure() -> UnitcellStructure:
    return UnitcellStructure(
        Cell(CUBIC, 1, BASIS_PRECISION),
        Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], COORD_PRECISION),
        _species(),
        ["Na", "Na"],
    )


def _cif_with_sites(tmp_path: Path, cell_length: str, sites: list[tuple[str, ...]]) -> Path:
    path = tmp_path / f"coarse-{cell_length.replace('.', '_')}.cif"
    rows = "".join(" ".join(site) + "\n" for site in sites)
    path.write_text(
        "data_x\n"
        f"_cell_length_a {cell_length}\n_cell_length_b {cell_length}\n_cell_length_c {cell_length}\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 1\n"
        "loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n" + rows,
        encoding="utf-8",
    )
    return path


def _coarse_cif(tmp_path: Path, cell_length: str) -> Path:
    return _cif_with_sites(tmp_path, cell_length, [("Si1", "Si", "0.3", "0.11", "0.07")])


def _coarse_sg2_special_cif(tmp_path: Path) -> Path:
    spacegroup = Spacegroup.standard(2)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "coarse-special.cif"
    path.write_text(
        "data_x\n"
        "_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 2\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si 0.0(9) 0.0 0.0\n",
        encoding="utf-8",
    )
    return path


# --- storing it ---


@pytest.mark.parametrize(
    ("given", "expected"),
    [(1e-4, F(1, 10000)), ("1/10000", F(1, 10000)), (F(1, 10000), F(1, 10000)), (0.0003, F(3, 10000)), (None, None)],
    ids=["float", "rational-string", "fraction", "esd-like-float", "unknown"],
)
def test_a_precision_is_stored_exactly(given: object, expected: F | None) -> None:
    """A float goes through its decimal spelling, so 1e-4 is 1/10000.

    Embedding it the way a float literally holds it would record a number nobody stated —
    the same trap that makes a CIF's ``0.3333`` become 6004199023210345/18014398509481984.
    """
    assert Cell(CUBIC, 1, given).precision == expected
    assert Sites([[0, 0, 0]], given).precision == expected


def test_precision_defaults_to_unknown() -> None:
    """Unknown is a real answer, and not the same as claiming exactness."""
    assert Cell(CUBIC).precision is None
    assert Sites([[0, 0, 0]]).precision is None
    assert UnitcellStructure(CUBIC, [[0, 0, 0]], _species(), ["Na"]).coordinate_precision is None


@pytest.mark.parametrize("bad", [0, -1, "-1/2"])
def test_a_non_positive_precision_is_rejected(bad: object) -> None:
    """Zero would claim an exactness no measurement has; use None to say unknown."""
    with pytest.raises(ValueError, match="strictly positive"):
        Cell(CUBIC, 1, bad)
    with pytest.raises(ValueError, match="strictly positive"):
        Sites([[0, 0, 0]], bad)


# --- carrying it through every reconstruction site ---


def test_cell_view_carries_precision() -> None:
    assert CellView(Cell(CUBIC, 1, BASIS_PRECISION)).precision == BASIS_PRECISION


def test_cell_params_view_carries_precision() -> None:
    """This view rebuilds a reference Cell internally; it must rebuild it with the precision."""
    view = CellParamsView(Cell(CUBIC, 1, BASIS_PRECISION))
    assert view.a == pytest.approx(5.0)
    assert view._backend.precision == BASIS_PRECISION


def test_cell_numeric_view_carries_precision_as_a_float() -> None:
    pytest.importorskip("numpy")
    assert CellNumericView(Cell(CUBIC, 1, BASIS_PRECISION)).precision == pytest.approx(1e-3)


def test_sites_view_carries_precision() -> None:
    assert SitesView(Sites([[0, 0, 0]], COORD_PRECISION)).precision == COORD_PRECISION


def test_sites_numeric_view_carries_precision_as_a_float() -> None:
    pytest.importorskip("numpy")
    assert SitesNumericView(Sites([[0, 0, 0]], COORD_PRECISION)).precision == pytest.approx(1e-4)


def test_structure_view_carries_both_precisions() -> None:
    view = UnitcellStructureView(_structure())
    assert view.coordinate_precision == COORD_PRECISION
    assert view.basis_precision == BASIS_PRECISION


def test_precision_survives_repeated_rewrapping() -> None:
    """Views are meant to be applied freely, so a round trip must not erode anything."""
    cell = Cell(CUBIC, 1, BASIS_PRECISION)
    assert CellView(CellView(CellView(cell))).precision == BASIS_PRECISION

    sites = Sites([[0, 0, 0]], COORD_PRECISION)
    assert SitesView(SitesView(sites)).precision == COORD_PRECISION


def test_numeric_layer_round_trips_back_to_the_exact_value() -> None:
    pytest.importorskip("numpy")
    structure = _structure()
    assert structure.numeric().cell.precision == pytest.approx(1e-3)
    assert structure.numeric().exact.basis_precision == BASIS_PRECISION


# --- a backend with no source of precision ---


def test_a_backend_that_knows_no_precision_reports_unknown() -> None:
    """`CellParams` and the primitive backends have nothing to derive one from.

    They inherit the concrete `None` from the API rather than being forced to invent a
    value, which is also what keeps out-of-tree backends working unchanged.
    """
    from httk.atomistic import CellBackend, CellParams, SitesBackend

    assert CellParams((5, 5, 5, 90, 90, 90)).precision is None
    assert CellBackend._select_backend([[5, 0, 0], [0, 5, 0], [0, 0, 5]]).precision is None
    assert SitesBackend._select_backend([[0, 0, 0]]).precision is None


# --- the derived Cartesian value ---


def test_cartesian_precision_scales_with_the_cell() -> None:
    """A tolerance is a distance, and a fractional precision is not."""
    small = UnitcellStructure(Cell(CUBIC), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"])
    large = UnitcellStructure(
        Cell([[30, 0, 0], [0, 30, 0], [0, 0, 30]]), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"]
    )
    assert float(small.cartesian_precision()) == pytest.approx(5e-4)
    assert float(large.cartesian_precision()) == pytest.approx(3e-3)


def test_cartesian_precision_uses_the_longest_edge() -> None:
    """The conservative choice: the largest displacement the uncertainty can produce."""
    oblong = UnitcellStructure(
        Cell([[2, 0, 0], [0, 5, 0], [0, 0, 20]]), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"]
    )
    assert float(oblong.cartesian_precision()) == pytest.approx(2e-3)


def test_a_coarse_cell_precision_dominates() -> None:
    """A cell known only to 1e-3 cannot place an atom better than that."""
    structure = _structure()
    assert float(structure.cartesian_precision()) == pytest.approx(1e-3)

    sharper_cell = UnitcellStructure(
        Cell(CUBIC, 1, F(1, 1000000)), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"]
    )
    assert float(sharper_cell.cartesian_precision()) == pytest.approx(5e-4)


def test_cartesian_precision_is_unknown_when_the_coordinates_are() -> None:
    structure = UnitcellStructure(Cell(CUBIC, 1, BASIS_PRECISION), Sites([[0, 0, 0]]), _species(), ["Na"])
    assert structure.cartesian_precision() is None


# --- component equality and structure identity ---


def test_precision_is_structural_metadata_but_not_component_geometry() -> None:
    assert Cell(CUBIC, 1, F(1, 10)) == Cell(CUBIC, 1, F(1, 1000000)) == Cell(CUBIC)
    assert Sites([[0, 0, 0]], F(1, 10)) == Sites([[0, 0, 0]])

    precise = _structure()
    vague = UnitcellStructure(Cell(CUBIC), Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]]), _species(), ["Na", "Na"])
    assert precise != vague
    assert same_crystal(precise, vague)


# --- precision arriving from a file ---


def _poscar(*, scale: str = "1.0", mode: str = "Direct", coords: str = "0.0000 0.0000 0.0000") -> str:
    return f"NaCl\n{scale}\n5.6400 0.0000 0.0000\n0.0000 5.6400 0.0000\n0.0000 0.0000 5.6400\nNa\n1\n{mode}\n{coords}\n"


def test_poscar_direct_coordinates_pass_their_precision_through() -> None:
    """Direct coordinates are already fractional, and the scale cancels for them."""
    structure = build_poscar(read_poscar(io.StringIO(_poscar(coords="0.5000 0.5000 0.5000"))))
    assert structure.coordinate_precision == F(1, 10000)
    assert structure.basis_precision == F(1, 10000)


def test_poscar_cartesian_coordinates_are_converted_to_fractional() -> None:
    """An orthogonal Cartesian precision transforms through the inverse basis."""
    structure = build_poscar(read_poscar(io.StringIO(_poscar(mode="Cartesian", coords="2.8200 2.8200 2.8200"))))
    assert float(structure.coordinate_precision) == pytest.approx(1e-4 / 5.64)


def test_poscar_cartesian_precision_uses_a_conservative_skew_cell_bound() -> None:
    source = _poscar(mode="Cartesian", coords="0.1234 0.1234 0.1234").replace(
        "5.6400 0.0000 0.0000\n0.0000 5.6400 0.0000\n0.0000 0.0000 5.6400",
        "1.0000 0.0000 0.0000\n1.0000 1.0000 0.0000\n0.0000 0.0000 1.0000",
    )

    structure = build_poscar(read_poscar(io.StringIO(source)))

    # For this row-vector basis, max_j sum_i |basis^-1[i,j]| is exactly 2.
    assert structure.coordinate_precision == F(1, 5000)


def test_the_poscar_scale_multiplies_the_basis_precision() -> None:
    """The cell entries are scaled, so their absolute precision scales with them."""
    assert build_poscar(read_poscar(io.StringIO(_poscar(scale="1.0")))).basis_precision == F(1, 10000)
    assert build_poscar(read_poscar(io.StringIO(_poscar(scale="2.0")))).basis_precision == F(1, 5000)


def test_the_scales_own_digits_are_not_charged_as_uncertainty() -> None:
    """The scaling factor defines units; it is not measured apart from the rows it scales.

    Charging its digits would double-count the same measurement, and would declare the cell
    of a 5.64 A structure good to only half an angstrom because the file wrote ``1.0``.
    """
    structure = build_poscar(read_poscar(io.StringIO(_poscar(scale="1.0"))))
    assert float(structure.basis_precision) == pytest.approx(1e-4)


def test_an_asu_structure_carries_and_propagates_its_precision() -> None:
    """Recorded in the structure's own setting, so it needs no transforming on the way out."""
    asu = ASUStructure(
        Cell(CUBIC, 1, BASIS_PRECISION),
        225,
        [WyckoffSite("a", FracVector(()), "Na")],
        _species(),
        coordinate_precision=COORD_PRECISION,
    )
    assert asu.coordinate_precision == COORD_PRECISION
    assert asu.expand_sites().precision == COORD_PRECISION

    expanded = UnitcellStructureView(asu)
    assert expanded.coordinate_precision == COORD_PRECISION
    assert expanded.basis_precision == BASIS_PRECISION


def test_recognition_carries_the_precision_onto_the_asu() -> None:
    """Nothing about recognizing symmetry sharpens the data, so the claim is inherited."""
    structure = UnitcellStructure(
        Cell(CUBIC, 1, BASIS_PRECISION),
        Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], COORD_PRECISION),
        _species() + [Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))],
        ["Na", "Cl"],
    )
    recovered = recognize_asu(structure, setting=Spacegroup.standard(221))
    assert recovered.coordinate_precision == COORD_PRECISION
    assert recovered.cell.precision == BASIS_PRECISION


# --- deriving a tolerance from the precision ---


def _two_site(cell: object, precision: object) -> UnitcellStructure:
    chlorine = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))
    sites = Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], precision)
    return UnitcellStructure(Cell(cell), sites, _species() + [chlorine], ["Na", "Cl"])


def test_the_tolerance_follows_the_stated_precision() -> None:
    """Four decimals in a 5 A cell reproduces the constant this replaces, which is a good sign."""
    assert structure_tolerance(_two_site(CUBIC, F(1, 10000))) == pytest.approx(1e-3)
    assert structure_tolerance(_two_site(CUBIC, F(1, 100))) == pytest.approx(1e-1)


def test_the_tolerance_scales_with_the_cell() -> None:
    big = [[30, 0, 0], [0, 30, 0], [0, 0, 30]]
    assert structure_tolerance(_two_site(big, F(1, 10000))) == pytest.approx(6e-3)


def test_an_unknown_precision_falls_back_to_the_constant() -> None:
    assert structure_tolerance(_two_site(CUBIC, None)) == DEFAULT_TOLERANCE
    assert structure_tolerance(_two_site(CUBIC, None), fallback=0.05) == 0.05


def test_the_tolerance_is_capped_below_half_the_closest_approach() -> None:
    """Otherwise coarse data could give a tolerance that merges genuinely distinct atoms."""
    chlorine = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))
    close = UnitcellStructure(
        Cell(CUBIC),
        Sites([[0, 0, 0], [F(1, 10), 0, 0]], F(1, 10)),  # 0.5 A apart
        _species() + [chlorine],
        ["Na", "Cl"],
    )
    # Uncapped this would be 0.1 * 5 * 2 = 1.0 A, far enough to merge the two sites.
    tolerance = structure_tolerance(close)
    assert tolerance == pytest.approx(0.25)
    assert tolerance < 0.25


def test_the_cap_does_not_engage_for_well_separated_atoms() -> None:
    assert structure_tolerance(_two_site(CUBIC, F(1, 10))) == pytest.approx(1.0)


def test_a_single_site_structure_has_no_separation_to_cap_against() -> None:
    lone = UnitcellStructure(Cell(CUBIC), Sites([[0, 0, 0]], F(1, 10)), _species(), ["Na"])
    assert structure_tolerance(lone) == pytest.approx(1.0)


def test_recognition_uses_the_derived_tolerance_by_default() -> None:
    """And an explicit value still overrides it."""
    structure = _two_site(CUBIC, F(1, 10000))
    assert recognize_asu(structure, setting=Spacegroup.standard(221)).coordinate_precision == F(1, 10000)
    assert recognize_asu(structure, setting=Spacegroup.standard(221), tolerance=1e-6) is not None


def test_a_coarsely_written_file_is_matched_at_the_precision_it_claims(tmp_path: Path) -> None:
    """The measured payoff, not a claimed one.

    This site is meant to sit on Wyckoff ``4e`` of SG 15 (``0, y, 1/4``) and explicitly states
    one-last-digit ESDs in x and z. Judged against a fixed 1e-3 Cartesian tolerance it misses
    the special position, lands on the general position, and generates eight atoms where the
    structure has four — a materially wrong answer, silently. The ESDs justify the component
    corrections, at which it is recognized correctly.
    """
    spacegroup = Spacegroup.from_setting("15:b1")
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "coarse.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.000\n_cell_length_b 6.000\n_cell_length_c 7.000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 15\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si 0.001(1) 0.333 0.251(1)\n",
        encoding="utf-8",
    )
    block = load(str(path), raw=True)["blocks"][0]
    assert block["coordinate_precision"] == F(1, 1000)

    with_fixed = asu_structure_from_cif(block, tolerance=1e-3)
    assert with_fixed.wyckoff_sites[0].wyckoff == "f"
    assert len(UnitcellStructureView(with_fixed).sites) == 8

    derived = asu_structure_from_cif(block)
    assert derived.wyckoff_sites[0].wyckoff == "e"
    assert len(UnitcellStructureView(derived).sites) == 4


def test_cif_site_precision_controls_default_wyckoff_tolerance(tmp_path: Path) -> None:
    spacegroup = Spacegroup.standard(2)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "site-precision.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0000\n_cell_length_b 5.0000\n_cell_length_c 5.0000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 2\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si 0.49(2) 0 0\nGe1 Ge 0 0.499999 0\n",
        encoding="utf-8",
    )
    block = load(str(path), raw=True)["blocks"][0]
    assert block["position_snap_bounds"] == [(F(1, 50), None, None), (None, F(1, 2_000_000), None)]

    derived = asu_structure_from_cif(block)
    assert derived.coordinate_precision == F(1, 50)
    assert [site.wyckoff for site in derived.wyckoff_sites] == ["d", "i"]

    overridden = asu_structure_from_cif(block, tolerance=0.2)
    assert [site.wyckoff for site in overridden.wyckoff_sites] == ["d", "c"]


def test_cif_aggregate_precision_fallback_does_not_use_three_axis_uncertainty(tmp_path: Path) -> None:
    spacegroup = Spacegroup.standard(2)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "aggregate-precision.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0000\n_cell_length_b 5.0000\n_cell_length_c 5.0000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 2\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "X1 X 0.45(2) 0.5000 0.5000\n",
        encoding="utf-8",
    )
    block = dict(load(str(path), raw=True)["blocks"][0])
    assert block["coordinate_precision"] == F(1, 50)
    block.pop("position_precisions")

    derived = asu_structure_from_cif(block)
    assert derived.wyckoff_sites[0].wyckoff == "i"


def test_cif_snap_prefers_lower_multiplicity_then_earlier_letter(tmp_path: Path) -> None:
    spacegroup = Spacegroup.standard(2)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "ambiguous-snap.cif"
    path.write_text(
        "data_x\n_cell_length_a 1.0000\n_cell_length_b 1.0000\n_cell_length_c 1.0000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 2\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si 0.25(25) 0.25(25) 0.25(25)\n",
        encoding="utf-8",
    )

    with collect_reports(level="warning") as collection:
        structure = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0])
    assert structure.wyckoff_sites[0].wyckoff == "a"
    assert spacegroup.wyckoff_position("a").multiplicity < spacegroup.wyckoff_position("i").multiplicity
    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.WARNING
    assert "ambiguous Wyckoff matches" in collection.records[0].getMessage()


@pytest.mark.parametrize("wyckoff_tag", ("_atom_site_Wyckoff_label", "_atom_site_Wyckoff_symbol"))
def test_cif_declared_wyckoff_makes_coarse_ambiguity_debug_only(tmp_path: Path, wyckoff_tag: str) -> None:
    spacegroup = Spacegroup.standard(2)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "declared-ambiguous.cif"
    path.write_text(
        "data_x\n"
        "_cell_length_a 1.0000\n_cell_length_b 1.0000\n_cell_length_c 1.0000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_space_group_IT_number 2\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        f"loop_\n_atom_site_label\n_atom_site_type_symbol\n{wyckoff_tag}\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si i 0.25(25) 0.25(25) 0.25(25)\n",
        encoding="utf-8",
    )
    with collect_reports(level="debug") as read_debug:
        block = dict(load(str(path), raw=True)["blocks"][0])

    alias_messages = [
        record for record in read_debug.records if "interpreted _atom_site_Wyckoff_symbol" in record.getMessage()
    ]
    assert len(alias_messages) == (1 if wyckoff_tag.endswith("symbol") else 0)
    assert all(record.levelno == logging.DEBUG for record in alias_messages)

    with collect_reports(level="warning") as warnings:
        structure = asu_structure_from_cif(block)
    with collect_reports(level="debug") as debug:
        asu_structure_from_cif(block)

    assert structure.wyckoff_sites[0].wyckoff == "i"
    assert warnings.records == []
    assert len(debug.records) == 1
    assert debug.records[0].levelno == logging.DEBUG

    invalid = dict(block)
    invalid["_httk_atomistic_wyckoff_labels"] = ["z"]
    with collect_reports(level="warning") as repaired:
        asu_structure_from_cif(invalid, repair=True)
    assert any(
        record.levelno == logging.WARNING and "ambiguous Wyckoff matches" in record.getMessage()
        for record in repaired.records
    )


def test_cif_snap_prefers_approximate_higher_symmetry_over_exact_lower_symmetry(tmp_path: Path) -> None:
    spacegroup = Spacegroup.standard(194)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "special-before-exact.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0000\n_cell_length_b 5.0000\n_cell_length_c 8.0000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        "_space_group_IT_number 194\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Zr1 Zr 0.3333 0.6667 0.4323(9)\n",
        encoding="utf-8",
    )

    structure = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0])
    assert structure.wyckoff_sites[0].wyckoff == "f"
    assert spacegroup.wyckoff_position("f").multiplicity < spacegroup.wyckoff_position("k").multiplicity


def test_cif_snap_finds_a_coupled_wyckoff_point_inside_the_rounding_box(tmp_path: Path) -> None:
    spacegroup = Spacegroup.standard(177)
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "coupled-rounding.cif"
    path.write_text(
        "data_x\n_cell_length_a 1.00000000\n_cell_length_b 1.00000000\n_cell_length_c 1.00000000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        "_space_group_IT_number 177\n"
        f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Se1 Se 0.11178605 0.55589303 0.50000000\n",
        encoding="utf-8",
    )

    structure = asu_structure_from_cif(load(str(path), raw=True)["blocks"][0])
    assert structure.wyckoff_sites[0].wyckoff == "m"


def test_cif_positional_uncertainty_is_debug_at_the_warning_threshold(tmp_path: Path) -> None:
    path = _coarse_cif(tmp_path, "0.5")
    with collect_reports(level="warning") as collection:
        structure = load(str(path))
    assert collection.records == []

    with collect_reports(level="debug") as collection:
        load(str(path))

    assert len(structure.sites) == 1
    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.DEBUG
    assert collection.records[0].context == "cif"
    assert "1 site(s)" in collection.records[0].getMessage()
    assert "maximum is" in collection.records[0].getMessage()
    assert CIF_POSITIONAL_UNCERTAINTY_WARNING == F(1, 10)


def test_cif_positional_uncertainty_below_warning_threshold_is_silent(tmp_path: Path) -> None:
    with collect_reports(level="warning") as collection:
        load(str(_coarse_cif(tmp_path, "0.49")))

    assert collection.records == []


def test_cif_positional_uncertainty_raises_at_error_threshold(tmp_path: Path) -> None:
    path = _coarse_cif(tmp_path, "5.0")
    with pytest.raises(ValueError, match=r"token '0\.3'.*1\.00995 Å.*allow_large_cif_uncertainty=True"):
        load(str(path))

    with collect_reports(level="debug") as collection:
        overridden = load(str(path), allow_large_cif_uncertainty=True)
    assert len(overridden.sites) == 1
    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.WARNING
    assert CIF_POSITIONAL_UNCERTAINTY_ERROR == F(1)


def test_cif_large_uncertainty_exact_special_stays_silent(tmp_path: Path) -> None:
    path = _coarse_sg2_special_cif(tmp_path)
    with collect_reports(level="debug") as collection:
        load(str(path))
    assert collection.records == []

    with collect_reports(level="debug") as collection:
        load(str(path), allow_large_cif_uncertainty=True)
    assert collection.records == []


def test_cif_esd_precision_is_preserved_and_projected(tmp_path: Path) -> None:
    path = _cif_with_sites(tmp_path, "0.5", [("Si1", "Si", "0.3000(2000)", "0.3000", "0.3000")])
    with collect_reports(level="debug") as collection:
        load(str(path))

    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.DEBUG
    assert "0.2 Å" in collection.records[0].getMessage()

    path = _cif_with_sites(tmp_path, "5", [("Si1", "Si", "0.3000(2000)", "0.3000", "0.3000")])
    with pytest.raises(ValueError, match=r"token '0\.3000\(2000\)'.*2 Å"):
        load(str(path))


def test_cif_uncertainty_uses_the_cubic_corner_norm(tmp_path: Path) -> None:
    path = _cif_with_sites(tmp_path, "0.5", [("Si1", "Si", "0.1", "0.1", "0.1")])
    with collect_reports(level="debug") as collection:
        load(str(path))

    assert collection.records[0].levelno == logging.DEBUG
    assert "0.173205" in collection.records[0].getMessage()


def test_cif_uncertainty_debug_is_aggregated_per_block(tmp_path: Path) -> None:
    path = _cif_with_sites(
        tmp_path,
        "0.5",
        [(f"Si{index}", "Si", "0.3", f"0.{index + 1}", "0.07") for index in range(1, 4)],
    )
    with collect_reports(level="debug") as collection:
        load(str(path))

    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.DEBUG
    assert "3 site(s)" in collection.records[0].getMessage()


def test_cif_uncertainty_thresholds_are_inclusive(tmp_path: Path) -> None:
    warning_path = _cif_with_sites(tmp_path, "0.5", [("Si1", "Si", "0.1", "1/3", "1/3")])
    with collect_reports(level="debug") as collection:
        load(str(warning_path))
    assert len(collection.records) == 1
    assert collection.records[0].levelno == logging.DEBUG

    error_path = _cif_with_sites(tmp_path, "5", [("Si1", "Si", "0.1", "1/3", "1/3")])
    with pytest.raises(ValueError, match=r"projected positional uncertainty of 1 Å"):
        load(str(error_path))
