"""The lossless ``httk-structure`` (``.httkstruct``) format round-trips structures exactly."""

import datetime
import gzip
import io
import json
import time
from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import SurdVector, load, save
from httk.core.loading import load_source
from httk.core.storage import content_id

from httk.atomistic import Cell, Species, UnitcellStructure, UnitcellStructureView
from httk.atomistic.composition import Assembly, ChemicalComposition
from httk.atomistic.io.httk_structure import read_httk_structure
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.structure.semantics import StructureSymmetry

FIXTURES = Path(__file__).with_name("fixtures")

DIAMOND_POSCAR = """\
C2 diamond (Fd-3m, a = 3.567 A), primitive fcc cell
3.567
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
C
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
"""

HALF_SQRT3 = SurdVector.from_radicand_map({3: Fraction(1, 2)})


def _round_trip(structure, tmp_path, name="structure.httkstruct"):
    path = tmp_path / name
    save(structure, path)
    recovered = load(path)
    assert type(recovered) is UnitcellStructure
    assert recovered == structure
    assert content_id(recovered) == content_id(structure)
    assert recovered.cell.scale == structure.cell.scale
    assert recovered.cell.unscaled_basis == structure.cell.unscaled_basis
    assert recovered.basis_precision == structure.basis_precision
    assert recovered.coordinate_precision == structure.coordinate_precision
    assert recovered.immutable_id == structure.immutable_id
    assert recovered.last_modified == structure.last_modified
    if structure.site_moments is not None:
        assert recovered.site_moments.precision == structure.site_moments.precision
    return recovered, path


def _everything() -> UnitcellStructure:
    cell = Cell(Cell([1, 1, Fraction(8, 5), 90, 90, 120]).basis, scale=SurdVector.sqrt_of(Fraction(25, 2)))
    cell = Cell(cell, precision=Fraction(1, 1000))
    species = (
        Species(
            "FeNi",
            ["Fe", "Ni"],
            [Fraction(1, 3), Fraction(2, 3)],
            mass=[55.845, 58.6934],
            charges=[2, None],
            spins=[Fraction(5, 2), None],
            labels=["Fe1", None],
            concentration_precision=[Fraction(1, 100), None],
        ),
        Species("OH", ["O"], [1], original_name="O_h", attached=["H"], nattached=[1], charges=[-1]),
    )
    moments = SurdVector.from_radicand_map(
        {1: [[0, 0, 0], [0, 0, 0], [1, 0, 0]], 3: [[0, 0, Fraction(1, 2)], [0, 0, 0], [0, 0, 0]]}
    )
    return UnitcellStructure(
        cell,
        ([0, 0, 0], [Fraction(1, 3), Fraction(2, 3), Fraction(1, 2)], [Fraction(2, 3), Fraction(1, 3), Fraction(1, 2)]),
        species,
        ["FeNi", "OH", "OH"],
        site_moments=CrystalAxisSiteMoments(moments, cell, precision=Fraction(1, 10)),
        assemblies=[Assembly(((1,), (2,)), (Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 100), None))],
        symmetry=StructureSymmetry(space_group_it_number=1, wyckoff_positions=("a", "a", "a")),
        chemical_composition=ChemicalComposition(
            {"Fe": Fraction(1, 3), "Ni": Fraction(2, 3), "O": 1, "H": 1}, "full", {"O": Fraction(1, 10)}
        ),
        chemical_formula_descriptive="FeNi(OH)",
        optimization_type="experimental",
        immutable_id="urn:example:1",
        last_modified=datetime.datetime(
            2026, 10, 20, 12, 30, 1, 123456, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
        ),
        charge=Fraction(1, 2),
    )


def test_diamond_poscar_round_trips_exactly(tmp_path):
    poscar = tmp_path / "diamond.POSCAR"
    poscar.write_text(DIAMOND_POSCAR, encoding="utf-8")
    structure = load(poscar)
    expected_scale = Fraction(3567, 1000)
    assert structure.cell.scale == SurdVector(expected_scale)
    assert structure.basis_precision == Fraction(3567, 10000)

    _, path = _round_trip(structure, tmp_path)

    assert json.loads(path.read_text(encoding="ascii"))["cell"]["scale"] == str(expected_scale)


def test_surd_basis_from_cod_cif_survives(tmp_path):
    structure = UnitcellStructureView(load(FIXTURES / "cod" / "1008801.cif"))
    assert 3 in structure.cell.basis.radicands

    recovered, path = _round_trip(structure, tmp_path)

    assert 3 in recovered.cell.basis.radicands
    assert "sqrt(3)" in path.read_text(encoding="ascii")


@pytest.mark.parametrize(
    "fixture",
    ["oxidation_states.cif", "disorder/217.cif", "magnetic_cartesian_hexagonal.mcif", "magnetic_cartesian.mcif"],
)
def test_fixture_structures_round_trip(tmp_path, fixture):
    _round_trip(UnitcellStructureView(load(FIXTURES / fixture)), tmp_path)


def test_every_structure_feature_round_trips(tmp_path):
    structure = _everything()

    recovered, _ = _round_trip(structure, tmp_path)

    assert recovered.species == structure.species
    assert recovered.species[0].mass == (55.845, 58.6934)
    assert recovered.site_moments == structure.site_moments
    assert recovered.assemblies == structure.assemblies
    assert recovered.symmetry == structure.symmetry
    assert recovered.chemical_composition == structure.chemical_composition
    assert recovered.charge == Fraction(1, 2)


def test_moment_kinds_periodicity_and_molecular_flag_round_trip(tmp_path):
    cartesian = SurdVector.from_radicand_map({1: [[0, 0, 0], [0, 0, -1]], 2: [[0, 0, 1], [0, 0, 0]]})
    molecule = UnitcellStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        [[0, 0, 0], [Fraction(1, 5), 0, 0]],
        species_at_sites=["H", "H"],
        molecular=True,
        site_moments=CartesianSiteMoments(cartesian),
    )
    slab = UnitcellStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]], periodicity=(True, True, False)),
        [[0, 0, Fraction(-7, 2)]],
        species_at_sites=["Si"],
        site_moments=CollinearSiteMoments([Fraction(-3, 2)], precision=Fraction(1, 4)),
    )
    assert _round_trip(molecule, tmp_path, "molecule.httkstruct")[0].molecular
    assert _round_trip(slab, tmp_path, "slab.httkstruct")[0].periodicity == (True, True, False)


def test_output_is_byte_deterministic(tmp_path):
    structure = _everything()
    first, second, third = (tmp_path / f"{name}.httkstruct" for name in ("a", "b", "c"))
    save(structure, first)
    save(_everything(), second)
    save(load(first), third)

    assert first.read_bytes() == second.read_bytes() == third.read_bytes()
    assert first.read_bytes().isascii()


def test_any_filename_through_format_and_compression(tmp_path):
    structure = _everything()
    plain = tmp_path / "input.httk.json"
    save(structure, plain, format="httk-structure")
    assert load(plain, format="httk-structure") == structure

    compressed = tmp_path / "input.httkstruct.gz"
    save(structure, compressed)
    assert gzip.decompress(compressed.read_bytes()) == plain.read_bytes()
    assert load(compressed) == structure
    assert load(compressed, raw=True)["format"] == "httk-structure"


def test_writer_refuses_values_that_are_not_httk_structures(tmp_path):
    with pytest.raises(ValueError, match="UnitcellStructureView"):
        save(([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0]], [14]), tmp_path / "x.httkstruct")


def test_writer_refuses_numbers_past_the_bound(tmp_path):
    structure = UnitcellStructure(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[Fraction(1, 3**3000), 0, 0]], species_at_sites=["Si"]
    )
    with pytest.raises(ValueError, match="1024 characters"):
        save(structure, tmp_path / "x.httkstruct")
    assert not (tmp_path / "x.httkstruct").exists()


def _document(tmp_path):
    path = tmp_path / "valid.httkstruct"
    save(_everything(), path)
    return json.loads(path.read_text(encoding="ascii"))


def _cell(document):
    return document["cell"]["unscaled_basis"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(format="cif"), "format"),
        (lambda d: d.update(version=2), "version"),
        (lambda d: d.update(version=True), "version"),
        (lambda d: d.update(extra=1), "unknown members"),
        (lambda d: d.pop("charge"), "missing members"),
        (lambda d: d["species"][0].update(colour="red"), "unknown members"),
        (lambda d: d.update(charge="1/0"), "rational"),
        (lambda d: d.update(charge="1.5"), "rational"),
        (lambda d: d.update(charge="2/4"), "lowest terms"),
        (lambda d: d.update(charge="-0"), "lowest terms"),
        (lambda d: d.update(charge="4/2"), "lowest terms"),
        (lambda d: d.update(charge=0.5), "number string"),
        (lambda d: d.update(charge="1" * 1025), "1024 characters"),
        (lambda d: _cell(d)[0].__setitem__(0, "sqrt(-2)"), "surd"),
        (lambda d: _cell(d)[0].__setitem__(0, "1*sqrt(4)"), "canonical"),
        (lambda d: _cell(d)[0].__setitem__(0, "1*sqrt(3)+1"), "canonical"),
        (lambda d: _cell(d)[0].__setitem__(0, "1*sqrt(2)+1*sqrt(2)"), "canonical"),
        (lambda d: _cell(d)[0].__setitem__(0, "0*sqrt(2)"), "canonical"),
        (lambda d: _cell(d)[0].__setitem__(0, "+1"), "canonical"),
        (lambda d: _cell(d)[0].__setitem__(0, "1*sqrt(4294967311)"), "surd"),
        (lambda d: _cell(d)[0].__setitem__(0, "1*sqrt(3)*sqrt(2)"), "surd"),
        (lambda d: _cell(d).pop(), "3 entries"),
        (lambda d: d["sites"]["reduced_coords"][0].append("0"), "3 entries"),
        (lambda d: d["species_at_sites"].pop(), "3 entries"),
        (lambda d: d["cell"].update(periodicity=[1, 1, 1]), "boolean"),
        (lambda d: d["species"][0].update(mass=[55, 58.6934]), "floating-point"),
        (lambda d: d["site_moments"].update(kind="spiral"), "kind"),
        (lambda d: d.update(last_modified="2026-10-20T12:30:01"), "UTC offset"),
        (lambda d: d["chemical_composition"].update(mode="partial"), "mode"),
        (lambda d: d.update(species={}), "array"),
    ],
)
def test_reader_rejects_malformed_documents(tmp_path, mutate, message):
    document = _document(tmp_path)
    mutate(document)
    path = tmp_path / "bad.httkstruct"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_httk_structure(path)
    with pytest.raises(ValueError, match=message):
        load(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[]", "JSON object"),
        ("not json", "not valid JSON"),
        ('{"format": "httk-structure", "format": "httk-structure"}', "repeats"),
        ('{"charge": NaN}', "non-JSON constant"),
        ("[" * 100_000 + "]" * 100_000, "nested too deeply|not valid JSON"),
    ],
)
def test_reader_rejects_non_documents(tmp_path, text, message):
    path = tmp_path / "bad.httkstruct"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_httk_structure(path)


def test_reader_rejects_overflowing_float_mass(tmp_path):
    text = json.dumps(_document(tmp_path)).replace("55.845", "1e400")
    path = tmp_path / "bad.httkstruct"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_httk_structure(path)


def test_semantically_inconsistent_document_fails_in_the_constructors(tmp_path):
    document = _document(tmp_path)
    document["species_at_sites"][0] = "Unknown"
    path = tmp_path / "bad.httkstruct"
    path.write_text(json.dumps(document), encoding="utf-8")
    read_httk_structure(path)  # well-formed
    with pytest.raises(ValueError, match="unknown species"):
        load(path)


# --------------------------------------------------------------------------- hardening


def _primes(below: int, count: int) -> list[int]:
    found: list[int] = []
    candidate = below
    while len(found) < count:
        candidate -= 1
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1)):
            found.append(candidate)
    return found


def _pell_pair(digits: int) -> tuple[int, int]:
    """Return ``p, q`` with ``p*p - 2*q*q == 1``, so ``p - q*sqrt(2)`` is about ``1/(3q)``."""
    p, q = 3, 2
    while len(str(p)) < digits:
        p, q = 3 * p + 4 * q, 2 * p + 3 * q
    return p, q


def _load_quickly(tmp_path, document):
    path = tmp_path / "adversarial.httkstruct"
    path.write_text(json.dumps(document), encoding="utf-8")
    start = time.perf_counter()
    try:
        return load(path)
    finally:
        assert time.perf_counter() - start < 2


_BIG_PRIMES = (4294967291, 4294967279, 4294967231)
_P, _Q = _pell_pair(100)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        # D1: radicands, terms and cancellation that would make the exact cell checks slow.
        (
            lambda d: d["cell"].update(
                unscaled_basis=[
                    [f"1*sqrt({p})" if i == j else "0" for j in range(3)] for i, p in enumerate(_BIG_PRIMES)
                ]
            ),
            "surd",
        ),
        (lambda d: _cell(d)[0].__setitem__(0, "1*sqrt(1009)"), "canonical"),
        (lambda d: _cell(d)[0].__setitem__(0, "1+1*sqrt(2)+1*sqrt(3)+1*sqrt(5)+1*sqrt(7)"), "surd terms"),
        (lambda d: _cell(d)[0].__setitem__(0, "1/" + "3" * 255), "256 characters"),
        (lambda d: d["cell"].update(scale=f"{_P}-{_Q}*sqrt(2)"), "cancels"),
        (
            lambda d: d["cell"].update(
                unscaled_basis=[[str(_P), f"{_Q}*sqrt(2)", "0"], ["1", "1", "0"], ["0", "0", "1"]]
            ),
            "cancels",
        ),
        # D2: too many symmetry operations, rejected before any is parsed.
        (
            lambda d: d["symmetry"].update(
                space_group_symmetry_operations_xyz=["x,y,z"] + [f"x+{k}/800,y,z" for k in range(1, 800)]
            ),
            "more than 192 entries",
        ),
        (lambda d: d["symmetry"].update(space_group_symmetry_operations_xyz=["x,y,z"] * 100_000), "more than 192"),
        # D3: operation strings that would exhaust the expression parser.
        (
            lambda d: d["symmetry"].update(space_group_symmetry_operations_xyz=["x" + "+0" * 50_000 + ",y,z"]),
            "operation",
        ),
        (lambda d: d["symmetry"].update(space_group_symmetry_operations_xyz=["-" * 50_000 + "x,y,z"]), "operation"),
        (
            lambda d: d["symmetry"].update(space_group_symmetry_operations_xyz=["x" + "*1" * 50_000 + ",y,z"]),
            "operation",
        ),
        (lambda d: d["species"][0].update(name="N" * 1025), "1024 characters"),
        (lambda d: d["species"][0]["labels"].__setitem__(0, "L" * 1025), "1024 characters"),
        # D4: empty space-group symbols.
        (lambda d: d["symmetry"].update(space_group_symbol_hall=""), "non-empty"),
        (lambda d: d["symmetry"].update(space_group_symbol_hermann_mauguin=""), "non-empty"),
        (lambda d: d["symmetry"].update(space_group_symbol_hermann_mauguin_extended=""), "non-empty"),
        (lambda d: d["symmetry"].update(wyckoff_positions=["", "a", "a"]), "non-empty"),
    ],
)
def test_adversarial_documents_are_rejected_quickly(tmp_path, mutate, message):
    document = _document(tmp_path)
    mutate(document)
    with pytest.raises(ValueError, match=message):
        _load_quickly(tmp_path, document)


def test_worst_accepted_cell_loads_quickly(tmp_path):
    # Four irrational terms per number with distinct prime radicands just below the cap
    # maximize the radicand factoring in the determinant (measured: about 0.1 s per load).
    radicands = iter(sorted(_primes(1000, 40)))

    def number() -> str:
        return "+".join(f"{index + 1}*sqrt({next(radicands)})" for index in range(4))

    document = _document(tmp_path)
    document["cell"]["unscaled_basis"] = [[number() for _ in range(3)] for _ in range(3)]
    document["cell"]["scale"] = number()
    structure = _load_quickly(tmp_path, document)
    path = tmp_path / "again.httkstruct"
    save(structure, path)
    assert load(path) == structure


def test_writer_refuses_cells_outside_the_bounds(tmp_path):
    corner = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for radicands, message in (((1009,), "radicand above"), ((2, 3, 5, 7), "surd terms")):
        basis = SurdVector.from_radicand_map({1: identity, **dict.fromkeys(radicands, corner)})
        structure = UnitcellStructure(Cell(basis), [[0, 0, 0]], species_at_sites=["Si"])
        with pytest.raises(ValueError, match=message):
            save(structure, tmp_path / "x.httkstruct")
        assert not (tmp_path / "x.httkstruct").exists()


def test_binary_stream_source(tmp_path):
    path = tmp_path / "valid.httkstruct"
    save(_everything(), path)
    assert load_source(io.BytesIO(path.read_bytes()), "x.httkstruct") == _everything()
    with pytest.raises(ValueError):
        read_httk_structure(io.BytesIO(b"\xff" + path.read_bytes()))


def test_space_group_with_192_operations_round_trips(tmp_path):
    structure = UnitcellStructure(
        [[Fraction(3567, 1000), 0, 0], [0, Fraction(3567, 1000), 0], [0, 0, Fraction(3567, 1000)]],
        [[0, 0, 0], [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)]],
        species_at_sites=["C", "C"],
        symmetry=StructureSymmetry(space_group_it_number=227),
    )
    assert len(structure.symmetry.space_group_symmetry_operations_xyz) == 192
    _round_trip(structure, tmp_path)
