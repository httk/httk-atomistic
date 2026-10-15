"""Quantum ESPRESSO pw.x input (read/write) and output (read) through httk.core load/save."""

import gzip
import io
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import SurdVector

from httk.atomistic import Trajectory, UnitcellStructure
from httk.atomistic.integrations.qe.io import BOHR_ANGSTROM, read_pw_input, read_pw_output

FIXTURES = Path(__file__).parent / "fixtures" / "qe"
BOHR = Fraction(BOHR_ANGSTROM)
SPECIES = "ATOMIC_SPECIES\n Si 28.0855 Si.upf\n"
ONE_ATOM = "ATOMIC_POSITIONS crystal\n Si 0 0 0\n"


def pw(system: str, cards: str = SPECIES + ONE_ATOM, nat: int = 1) -> list[str]:
    """A minimal pw.x input as lines."""
    return f"&system\n  nat = {nat}, ntyp = 1\n  {system}\n/\n{cards}".splitlines(keepends=True)


def metric(cell: list[list[Any]]) -> list[list[Any]]:
    """The exact metric tensor of payload cell rows."""
    return [[sum((x * y for x, y in zip(u, v, strict=True)), Fraction(0)) for v in cell] for u in cell]


def flat(rows: Any) -> list[float]:
    return [float(x) for row in rows for x in row]


def determinant(cell: list[list[Any]]) -> float:
    (a, b, c), (d, e, f), (g, h, i) = ([float(x) for x in row] for row in cell)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def fractions_of(rows: Any) -> list[list[Fraction]]:
    """Rows of an exact rational vector (FracVector, or SurdVector without radicals)."""
    if isinstance(rows, SurdVector):
        assert rows.is_rational
        rows = rows.coefficient(1)
    return [[Fraction(x) for x in row] for row in rows.to_fractions()]


# alat = 10 bohr, b = 15, c = 20 (celldm(2) = 3/2, celldm(3) = 2); the cosines below.
# Expected metric tensors G_ij = v_i . v_j in bohr^2, derived by hand from each lattice's
# conventional lengths and angles (e.g. fcc: |v|^2 = a^2/2, 60 degrees).
IBRAV_CASES = {
    1: ("", [[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
    2: ("", [[50, 25, 25], [25, 50, 25], [25, 25, 50]]),
    3: ("", [[75, 25, -25], [25, 75, 25], [-25, 25, 75]]),
    -3: ("", [[75, -25, -25], [-25, 75, -25], [-25, -25, 75]]),
    4: ("celldm(3) = 2", [[100, -50, 0], [-50, 100, 0], [0, 0, 400]]),
    5: ("celldm(4) = 0.1", [[100, 10, 10], [10, 100, 10], [10, 10, 100]]),
    -5: ("celldm(4) = 0.1", [[100, 10, 10], [10, 100, 10], [10, 10, 100]]),
    6: ("celldm(3) = 2", [[100, 0, 0], [0, 100, 0], [0, 0, 400]]),
    7: ("celldm(3) = 2", [[150, 100, 100], [100, 150, 50], [100, 50, 150]]),
    8: ("celldm(2) = 1.5, celldm(3) = 2", [[100, 0, 0], [0, 225, 0], [0, 0, 400]]),
    9: ("celldm(2) = 1.5, celldm(3) = 2", [["325/4", "125/4", 0], ["125/4", "325/4", 0], [0, 0, 400]]),
    -9: ("celldm(2) = 1.5, celldm(3) = 2", [["325/4", "-125/4", 0], ["-125/4", "325/4", 0], [0, 0, 400]]),
    91: ("celldm(2) = 1.5, celldm(3) = 2", [[100, 0, 0], [0, "625/4", "-175/4"], [0, "-175/4", "625/4"]]),
    10: ("celldm(2) = 1.5, celldm(3) = 2", [[125, 25, 100], [25, "325/4", "225/4"], [100, "225/4", "625/4"]]),
    11: (
        "celldm(2) = 1.5, celldm(3) = 2",
        [["725/4", "525/4", "75/4"], ["525/4", "725/4", "275/4"], ["75/4", "275/4", "725/4"]],
    ),
    12: ("celldm(2) = 1.5, celldm(3) = 2, celldm(4) = 0.1", [[100, 15, 0], [15, 225, 0], [0, 0, 400]]),
    -12: ("celldm(2) = 1.5, celldm(3) = 2, celldm(5) = 0.2", [[100, 0, 40], [0, 225, 0], [40, 0, 400]]),
    13: (
        "celldm(2) = 1.5, celldm(3) = 2, celldm(4) = 0.1",
        [[125, "15/2", -75], ["15/2", 225, "15/2"], [-75, "15/2", 125]],
    ),
    -13: (
        "celldm(2) = 1.5, celldm(3) = 2, celldm(5) = 0.2",
        [["325/4", "125/4", 20], ["125/4", "325/4", -20], [20, -20, 400]],
    ),
    14: (
        "celldm(2) = 1.5, celldm(3) = 2, celldm(4) = 0.1, celldm(5) = 0.2, celldm(6) = 0.3",
        [[100, 45, 40], [45, 225, 30], [40, 30, 400]],
    ),
}


@pytest.mark.parametrize("ibrav", sorted(IBRAV_CASES))
def test_ibrav_lattices_match_hand_derived_metric(ibrav: int) -> None:
    extra, expected = IBRAV_CASES[ibrav]
    payload = read_pw_input(pw(f"ibrav = {ibrav}, celldm(1) = 10.0, {extra}".rstrip(", ")))
    assert payload["cell_unit"] == "bohr"
    assert metric(payload["cell"]) == [[Fraction(x) for x in row] for row in expected]
    assert determinant(payload["cell"]) > 0
    structure = httk.core.load_source(pw(f"ibrav = {ibrav}, celldm(1) = 10.0, {extra}".rstrip(", ")), "x.pwi")
    assert isinstance(structure, UnitcellStructure)


def test_ibrav_values_are_exact_surds() -> None:
    cell = read_pw_input(pw("ibrav = 4, celldm(1) = 2, celldm(3) = 1.5"))["cell"]
    assert cell[1][1] * cell[1][1] == 3  # 2 * sqrt(3)/2, squared exactly
    assert cell[2] == [0, 0, 3]


def test_abc_parameters_map_like_qe() -> None:
    hexagonal = read_pw_input(pw("ibrav = 4, A = 3.0, C = 5.0"))
    assert hexagonal["cell_unit"] == "angstrom"
    assert metric(hexagonal["cell"]) == [[9, Fraction(-9, 2), 0], [Fraction(-9, 2), 9, 0], [0, 0, 25]]
    triclinic = read_pw_input(pw("ibrav = 14, A = 2, B = 3, C = 4, cosBC = 0.1, cosAC = 0.2, cosAB = 0.3"))
    assert metric(triclinic["cell"]) == [
        [4, Fraction(9, 5), Fraction(8, 5)],
        [Fraction(9, 5), 9, Fraction(6, 5)],
        [Fraction(8, 5), Fraction(6, 5), 16],
    ]
    unique_b = read_pw_input(pw("ibrav = -12, A = 2, B = 3, C = 4, cosAC = 0.5"))
    assert metric(unique_b["cell"])[0][2] == 4  # 2 * 4 * 0.5
    trigonal = read_pw_input(pw("ibrav = 5, A = 2, cosAB = 0.25"))
    assert metric(trigonal["cell"])[0][1] == 1


def test_si_fixture_cell_and_positions_are_exact() -> None:
    structure = httk.core.load(str(FIXTURES / "si.in"), format="qe-pw-input")
    half = Fraction(513, 100) * BOHR  # celldm(1)/2 in Å
    assert fractions_of(structure.cell.basis) == [[-half, 0, half], [0, half, half], [-half, half, 0]]
    assert fractions_of(structure.sites.reduced_coords) == [[0, 0, 0], [Fraction(1, 4)] * 3]
    assert [item.name for item in structure.species] == ["Si"]
    assert structure.species[0].mass == (28.085,)


@pytest.mark.parametrize(
    ("system", "cards", "fractional"),
    [
        ("ibrav = 1, celldm(1) = 10", "ATOMIC_POSITIONS alat\n Si 0.5 0 0\n", Fraction(1, 2)),
        ("ibrav = 1, celldm(1) = 10", "ATOMIC_POSITIONS\n Si 0.5 0 0\n", Fraction(1, 2)),
        ("ibrav = 1, celldm(1) = 10", "ATOMIC_POSITIONS {bohr}\n Si 2.5 0 0\n", Fraction(1, 4)),
        ("ibrav = 1, A = 10", "ATOMIC_POSITIONS (angstrom)\n Si 2.5 0 0 0 1 1\n", Fraction(1, 4)),
        ("ibrav = 1, A = 10", "ATOMIC_POSITIONS crystal\n Si 0.75 0 0\n", Fraction(3, 4)),
        ("ibrav = 1, celldm(1) = 10", f"ATOMIC_POSITIONS angstrom\n Si {BOHR_ANGSTROM} 0 0\n", Fraction(1, 10)),
        # bohr/angstrom vectors: alat is |a1| = 5
        ("ibrav = 0", "CELL_PARAMETERS bohr\n 3 4 0\n -4 3 0\n 0 0 5\nATOMIC_POSITIONS alat\n Si 0.6 0.8 0\n", 1),
        (
            "ibrav = 0",
            "CELL_PARAMETERS angstrom\n 5 0 0\n 0 5 0\n 0 0 5\nATOMIC_POSITIONS alat\n Si 0.5 0 0\n",
            Fraction(1, 2),
        ),
        # unit-less CELL_PARAMETERS: alat with a lattice parameter, bohr without
        (
            "ibrav = 0, celldm(1) = 2",
            "CELL_PARAMETERS\n 2 0 0\n 0 2 0\n 0 0 2\nATOMIC_POSITIONS bohr\n Si 1 0 0\n",
            Fraction(1, 4),
        ),
        ("ibrav = 0", "CELL_PARAMETERS\n 2 0 0\n 0 2 0\n 0 0 2\nATOMIC_POSITIONS bohr\n Si 1 0 0\n", Fraction(1, 2)),
        (
            "ibrav = 0, A = 2",
            "CELL_PARAMETERS alat\n 2 0 0\n 0 2 0\n 0 0 2\nATOMIC_POSITIONS angstrom\n Si 1 0 0\n",
            Fraction(1, 4),
        ),
    ],
)
def test_position_and_cell_units(system: str, cards: str, fractional: Fraction) -> None:
    structure = httk.core.load_source(pw(system, SPECIES + cards), "x.pwi")
    assert fractions_of(structure.sites.reduced_coords)[0][0] == fractional


def test_namelist_syntax_comments_case_and_d_exponents() -> None:
    text = """! leading comment
 &CONTROL
   calculation='scf', prefix = 'it''s ! no comment', ! a comment
   pseudo_dir = "./pp/", tprnfor = .TRUE.
 /
&SYSTEM
  IBRAV= 0, CellDM( 1 ) = 1.0D1 ,NAT=2
  nTyp = 2, ecutwfc = 3.0d1, starting_magnetization(2) = 0.5
  starting_ns_eigenvalue(1,2,1) = 0.0, Hubbard_U(1) = 1.0d-1
/
&electrons
/
# a card-section comment
cell_parameters {Alat}
  1.0d0 0 0   ! trailing
  0 1.0D0 0
  0 0 1.0
Atomic_Species
  Fe1 55.845 Fe.pbe-spn-rrkjus_psl.1.0.0.UPF
  O_h 15.999d0 O.upf
ATOMIC_POSITIONS {Crystal}
  Fe1 0.0 0.0 0.0
  O_h 0.5d0 0.5 0.5
K_POINTS gamma
"""
    payload = read_pw_input(io.StringIO(text))
    assert payload["raw"] == text
    assert payload["species"] == [
        {"name": "Fe1", "element": "Fe", "mass": "55.845"},
        {"name": "O_h", "element": "O", "mass": "15.999d0"},
    ]
    structure = httk.core.load_source(io.StringIO(text), "x.pwi")
    assert fractions_of(structure.cell.basis)[0] == [10 * BOHR, 0, 0]
    assert fractions_of(structure.sites.reduced_coords)[1] == [Fraction(1, 2)] * 3
    assert [item.chemical_symbols for item in structure.species] == [("Fe",), ("O",)]


def test_celldm_list_and_repeat_assignments() -> None:
    payload = read_pw_input(pw("ibrav = 8, celldm = 2, 1.5, 2"))
    assert metric(payload["cell"])[2][2] == 16
    payload = read_pw_input(pw("ibrav = 6, celldm(1) = 2, celldm(2) = 2*3"))
    assert metric(payload["cell"])[2][2] == 36


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (pw("ibrav = 1, celldm(1) = 10", SPECIES + "ATOMIC_POSITIONS crystal_sg\n Si 0 0 0\n"), "crystal_sg"),
        (pw("ibrav = 1, celldm(1) = 10, space_group = 225"), "space_group"),
        (pw("ibrav = 15, celldm(1) = 10"), "ibrav = 15"),
        (pw("ibrav = -1, celldm(1) = 10"), "ibrav = -1"),
        (pw("ibrav = 1, celldm(1) = 10, A = 5"), "mixes celldm"),
        (pw("ibrav = 1"), "needs celldm"),
        (pw("ibrav = 4, celldm(1) = 10"), "celldm\\(3\\)"),
        (pw("ibrav = 12, celldm(1) = 1, celldm(2) = 1, celldm(3) = 1, celldm(4) = 1"), "-1 < celldm\\(4\\)"),
        (pw("ibrav = 5, celldm(1) = 1, celldm(4) = -0.5"), "1 \\+ 2cos"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES + "CELL_PARAMETERS\n1 0 0\n0 1 0\n0 0 1\n" + ONE_ATOM), "ambiguous"),
        (
            pw("ibrav = 0, celldm(1) = 1", SPECIES + "CELL_PARAMETERS bohr\n1 0 0\n0 1 0\n0 0 1\n" + ONE_ATOM),
            "together",
        ),
        (pw("ibrav = 0", SPECIES + "CELL_PARAMETERS alat\n1 0 0\n0 1 0\n0 0 1\n" + ONE_ATOM), "needs celldm"),
        (pw("ibrav = 0"), "needs a CELL_PARAMETERS"),
        (pw("ibrav = 1, celldm(1) = 1", "ATOMIC_SPECIES\n Xx 1.0 X.upf\n" + ONE_ATOM), "does not name an element"),
        (pw("ibrav = 1, celldm(1) = 1", "ATOMIC_SPECIES\n Si12 1.0 X.upf\n" + ONE_ATOM), "does not name an element"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES + "ATOMIC_POSITIONS crystal\n Si 1/3 0 0\n"), "algebraic"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES + "ATOMIC_POSITIONS crystal\n Si 0 0 0 2 0 0\n"), "if_pos"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES + "ATOMIC_POSITIONS foo\n Si 0 0 0\n"), "foo"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES + ONE_ATOM, nat=2), "must have 2 lines"),
        (pw("ibrav = 1, celldm(1) = 1, nat = 3"), "assigned twice"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES), "no ATOMIC_POSITIONS"),
        (pw("ibrav = 1, celldm(1) = 1", SPECIES + ONE_ATOM + ONE_ATOM), "appears twice"),
        (["&system\n", "ibrav = 1\n"], "not terminated"),
        (["&system\n", "title = 'open\n", "/\n"], "unterminated string"),
        (["&control\n", "/\n", "ATOMIC_POSITIONS crystal\n"], "no &system"),
        (["units metal\n", "read_data data.si\n"], "no &system"),
    ],
)
def test_input_refusals(lines: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_pw_input(lines)


SKEWED_CELL = """CELL_PARAMETERS angstrom
 2.71467908006671 0.12345678901234 1.00000000000001
 0.00000000000000 2.71467908006671 2.71467908006671
 -2.71467908006671 2.71467908006671 0.00000000000000
"""


def reference_sqrt(value: Fraction) -> Decimal:
    with localcontext() as context:
        context.prec = 60
        return (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()


def deviation(entry: Any, reference: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 60
        exact = (
            Fraction(entry) if isinstance(entry, (int, Fraction)) else entry.to_fractions_approx(Fraction(1, 10**45))
        )
        return abs(Decimal(exact.numerator) / Decimal(exact.denominator) - reference)


def test_long_tokens_load_fast_with_approximated_roots() -> None:
    start = time.perf_counter()
    cards = SPECIES + SKEWED_CELL + "ATOMIC_POSITIONS angstrom\n Si 0.12345678901234 0.0 0.0\n"
    structure = httk.core.load_source(pw("ibrav = 0", cards), "x.pwi")
    assert fractions_of(structure.cell.basis)[0][1] == Fraction("0.12345678901234")

    # alat = |a1| is beyond the bounded exact root: a rational approximation claims its error.
    cards = SPECIES + SKEWED_CELL + "ATOMIC_POSITIONS alat\n Si 1.00000000000000 0 0\n"
    payload = read_pw_input(pw("ibrav = 0", cards))
    a1 = [Fraction(token) for token in SKEWED_CELL.splitlines()[1].split()]
    claim = payload["positions_precision"]
    assert (
        deviation(payload["positions"][0][0], reference_sqrt(sum((x * x for x in a1), Fraction(0))))
        <= claim
        < Fraction(1, 10**8)
    )
    httk.core.load_source(pw("ibrav = 0", cards), "x.pwi")

    # ibrav 12: b sin(gamma) with a 16-digit and a 10-digit (105 degrees) cosAB; A/B/C are
    # integers, so the claim is the root approximation alone.
    for cosine in ("0.1234567890123456", "-0.2588190451"):
        payload = read_pw_input(pw(f"ibrav = 12, A = 5, B = 6, C = 7, cosAB = {cosine}"))
        reference = 6 * reference_sqrt(1 - Fraction(cosine) ** 2)
        claim = payload["cell_precision"]
        assert 0 < deviation(payload["cell"][1][1], reference) <= claim < Fraction(1, 10**8)
        httk.core.load_source(pw(f"ibrav = 12, A = 5, B = 6, C = 7, cosAB = {cosine}"), "x.pwi")

    # ibrav 14: v3 divides by the approximated sin(gamma) and includes the volume root.
    system = "ibrav = 14, A = 5, B = 6, C = 7, cosAB = 0.12345678901, cosAC = 0.2, cosBC = 0.3"
    al, be, ga = Fraction("0.3"), Fraction("0.2"), Fraction("0.12345678901")
    payload = read_pw_input(pw(system))
    sine = reference_sqrt(1 - ga * ga)
    volume = reference_sqrt(1 + 2 * al * be * ga - al * al - be * be - ga * ga)
    with localcontext() as context:
        context.prec = 60
        term = al - be * ga
        cosine_term = Decimal(term.numerator) / Decimal(term.denominator)
        references = {(1, 1): 6 * sine, (2, 1): 7 * cosine_term / sine, (2, 2): 7 * volume / sine}
    claim = payload["cell_precision"]
    for (row, column), reference in references.items():
        assert deviation(payload["cell"][row][column], reference) <= claim < Fraction(1, 10**8)
    httk.core.load_source(pw(system), "x.pwi")
    assert time.perf_counter() - start < 2


@pytest.mark.parametrize(
    "cosines",
    [
        # sin(gamma) = sqrt(3)/2 is an exact surd, the volume root is approximated
        ("0.12345678901234", "0.23456789012345", "0.5"),
        # sin(gamma) is approximated, the volume root is exactly 0.0150528 (1 - a^2 - g^2 is a square)
        ("0.861184", "0", "0.5080704"),
    ],
    ids=["exact-sine", "exact-volume"],
)
def test_ibrav14_mixed_exact_and_approximated_roots(cosines: tuple[str, str, str]) -> None:
    al, be, ga = (Fraction(cosine) for cosine in cosines)
    system = f"ibrav = 14, celldm(1) = 10, celldm(2) = 1.1, celldm(3) = 1.3, celldm(4) = {cosines[0]}"
    system += f", celldm(5) = {cosines[1]}, celldm(6) = {cosines[2]}"
    payload = read_pw_input(pw(system))
    claim = payload["cell_precision"]
    assert isinstance(claim, Fraction) and claim < Fraction(1, 10**8)
    sine = reference_sqrt(1 - ga * ga)
    volume = reference_sqrt(1 + 2 * al * be * ga - al * al - be * be - ga * ga)
    with localcontext() as context:
        context.prec = 60
        term = al - be * ga
        references = {
            (1, 1): 11 * sine,
            (2, 1): 13 * Decimal(term.numerator) / Decimal(term.denominator) / sine,
            (2, 2): 13 * volume / sine,
        }
    for (row, column), reference in references.items():
        assert deviation(payload["cell"][row][column], reference) <= claim
    structure = httk.core.load_source(pw(system), "x.pwi")
    assert structure.cell.precision is not None


def test_alat_scaled_values_claim_their_printed_precision() -> None:
    # ibrav = 2 with celldm(1) = 10.26: entries alat/2 claim 1/100 * 1/2 bohr.
    assert read_pw_input(str(FIXTURES / "si.in"))["cell_precision"] == Fraction(1, 200)
    structure = httk.core.load(str(FIXTURES / "si.in"), format="qe-pw-input")
    assert structure.cell.precision == Fraction(1, 200) * BOHR
    # 0.25 alat with alat = 10.0: 0.01 * 10 + 0.25 * 0.1 bohr.
    payload = read_pw_input(pw("ibrav = 1, celldm(1) = 10.0", SPECIES + "ATOMIC_POSITIONS alat\n Si 0.25 0 0\n"))
    assert payload["positions_precision"] == Fraction(1, 8)
    assert payload["cell_precision"] == Fraction(1, 10)
    # integer tokens claim nothing
    payload = read_pw_input(pw("ibrav = 1, celldm(1) = 10", SPECIES + "ATOMIC_POSITIONS alat\n Si 0 0 0\n"))
    assert "cell_precision" not in payload and "positions_precision" not in payload
    # |a1| = 5 of bohr vectors: twice the a1 token precision for alat
    cards = SPECIES + "CELL_PARAMETERS bohr\n 3.0 4.0 0.0\n 0 5 0\n 0 0 5\nATOMIC_POSITIONS alat\n Si 1 0 0\n"
    assert read_pw_input(pw("ibrav = 0", cards))["positions_precision"] == Fraction(1, 5)


def test_iterables_of_lines_without_newlines() -> None:
    lines = [line.rstrip("\n") for line in pw("ibrav = 1, celldm(1) = 10")]
    assert read_pw_input(lines)["species_at_sites"] == ["Si"]
    output = (FIXTURES / "si.out").read_text().splitlines()
    assert len(read_pw_output(output)["frames"]) == 1


def test_non_utf8_input_is_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.pwi"
    path.write_bytes(b"&system\n title='\xff'\n/\n")
    with pytest.raises(ValueError, match="UTF-8"):
        httk.core.load(str(path))


def test_precision_override_is_validated_and_passed() -> None:
    assert read_pw_input(pw("ibrav = 1, celldm(1) = 1"), precision=1e-3)["precision_override"] == 1e-3
    with pytest.raises(ValueError, match="precision"):
        read_pw_input(pw("ibrav = 1, celldm(1) = 1"), precision=0)


def test_save_load_round_trip_is_exact(tmp_path: Path) -> None:
    structure = httk.core.load(str(FIXTURES / "si.in"), format="qe-pw-input")
    path = tmp_path / "si.pwi"
    httk.core.save(structure, str(path))
    text = path.read_text()
    assert text.startswith("&system\n  ibrav = 0\n  nat = 2\n  ntyp = 1\n/\nATOMIC_SPECIES\nSi 28.085 Si.UPF\n")
    assert "CELL_PARAMETERS angstrom\n" in text and "ATOMIC_POSITIONS crystal\nSi 0 0 0\nSi 0.25 0.25 0.25\n" in text
    again = httk.core.load(str(path))
    assert fractions_of(again.cell.basis) == fractions_of(structure.cell.basis)
    assert fractions_of(again.sites.reduced_coords) == fractions_of(structure.sites.reduced_coords)
    assert again.species[0].mass == (28.085,)


def test_writer_keywords_and_defaults(tmp_path: Path) -> None:
    payload = read_pw_input(pw("ibrav = 1, celldm(1) = 1"))
    del payload["raw"]
    payload["species"][0]["mass"] = None
    path = tmp_path / "x.pwi"
    httk.core.save(payload, str(path))
    assert "\nSi 1.0 Si.UPF\n" in path.read_text()
    httk.core.save(payload, str(path), masses={"Si": 28.0855}, pseudopotentials={"Si": "Si.pbe.UPF"})
    assert "\nSi 28.0855 Si.pbe.UPF\n" in path.read_text()
    with pytest.raises(ValueError, match="not in the structure"):
        httk.core.save(payload, str(path), masses={"Ge": 72.6})
    with pytest.raises(ValueError, match="positive"):
        httk.core.save(payload, str(path), masses={"Si": -1.0})
    with pytest.raises(ValueError, match="without blanks"):
        httk.core.save(payload, str(path), pseudopotentials={"Si": "Si pseudo"})


def test_writer_refuses_invalid_labels(tmp_path: Path) -> None:
    payload = read_pw_input(pw("ibrav = 1, celldm(1) = 1"))
    del payload["raw"]
    payload["species"][0]["name"] = payload["species_at_sites"][0] = "Silicon"
    with pytest.raises(ValueError, match="rename the species"):
        httk.core.save(payload, str(tmp_path / "x.pwi"))


def test_raw_payload_writes_original_bytes_and_refuses_keywords(tmp_path: Path) -> None:
    source = FIXTURES / "si_relax.in"
    payload = httk.core.load(str(source), format="qe-pw-input", raw=True)
    httk.core.save(payload, str(tmp_path / "copy.pwi"))
    assert (tmp_path / "copy.pwi").read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="raw"):
        httk.core.save(payload, str(tmp_path / "other.pwi"), masses={"Si": 28.0})


def test_compressed_input_loads(tmp_path: Path) -> None:
    path = tmp_path / "si.pwi.gz"
    path.write_bytes(gzip.compress((FIXTURES / "si.in").read_bytes()))
    structure = httk.core.load(str(path))
    assert fractions_of(structure.sites.reduced_coords)[1] == [Fraction(1, 4)] * 3


def test_si_output_is_one_frame() -> None:
    trajectory = httk.core.load(str(FIXTURES / "si.out"), format="qe-pw-output")
    assert isinstance(trajectory, Trajectory)
    assert trajectory.nframes == 1
    frame = trajectory.frame(0)
    half = Fraction("5.13") * BOHR
    assert fractions_of(frame.cell.basis) == [[-half, 0, half], [0, half, half], [-half, half, 0]]
    assert fractions_of(frame.sites.reduced_coords) == [[0, 0, 0], [Fraction(1, 4)] * 3]
    assert frame.species[0].mass == (28.085,)


def test_output_alat_values_claim_their_printed_precision() -> None:
    frames = read_pw_output(str(FIXTURES / "si_relax.out"))["frames"]
    # crystal axes (6 decimals) and tau (7 decimals) times alat = 10.2600 (4 decimals)
    cell_precision = Fraction(1, 10**6) * Fraction("10.26") + Fraction(1, 2) * Fraction(1, 10**4)
    assert [frame["cell_precision"] for frame in frames] == [cell_precision] * 4
    assert frames[0]["positions_precision"] == Fraction(1, 10**7) * Fraction("10.26") + Fraction(26, 100) / 10**4
    trajectory = httk.core.load(str(FIXTURES / "si_relax.out"), format="qe-pw-output")
    assert trajectory.frame(-1).cell.precision == cell_precision * BOHR


def test_relax_output_frames_and_final_coordinates() -> None:
    trajectory = httk.core.load(str(FIXTURES / "si_relax.out"), format="qe-pw-output")
    assert trajectory.nframes == 4
    final = [
        [Fraction("0.0100194565"), Fraction("-0.0000152005"), Fraction("-0.0000152005")],
        [Fraction("0.2599805435"), Fraction("0.2500152005"), Fraction("0.2500152005")],
    ]
    assert fractions_of(trajectory.frame(-1).sites.reduced_coords) == final
    assert fractions_of(trajectory.frame(1).sites.reduced_coords)[0][0] == Fraction("0.0075782803")
    # Frame 0 is the header's alat-unit table: tau(2) = (-0.26, 0.25, 0.26) alat.
    assert fractions_of(trajectory.frame(0).sites.reduced_coords)[1] == [
        Fraction(27, 100),
        Fraction(1, 4),
        Fraction(1, 4),
    ]
    nstep = httk.core.load(str(FIXTURES / "si_relax_nstep.out"), format="qe-pw-output")
    assert nstep.nframes == 3


VC_HEADER = """
     lattice parameter (alat)  =      10.0000  a.u.
     number of atoms/cell      =            2
     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     atomic species   valence    mass     pseudopotential
     Si                4.00    28.08500     Si( 1.00)

     site n.     atom                  positions (alat units)
         1        Si     tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2        Si     tau(   2) = (   0.5000000   0.5000000   0.5000000  )
"""
VC_STEP = """
CELL_PARAMETERS (alat= 10.00000000)
   1.100000000   0.000000000   0.000000000
   0.000000000   1.000000000   0.000000000
   0.000000000   0.000000000   1.000000000

ATOMIC_POSITIONS (alat)
Si            0.0000000000        0.0000000000        0.0000000000
Si            0.5500000000        0.5000000000        0.5000000000
"""
VC_FINAL = """
CELL_PARAMETERS (bohr)
  11.000000000   0.000000000   0.000000000
   0.000000000  10.000000000   0.000000000
   0.000000000   0.000000000  10.000000000

ATOMIC_POSITIONS (bohr)
Si            0.0000000000        0.0000000000        0.0000000000
Si            5.5000000000        5.0000000000        5.0000000000
"""


def vc_relax(final: str) -> str:
    """A vc-relax output whose final-coordinates block holds ``final``."""
    return f"{VC_HEADER}{VC_STEP}\nBegin final coordinates\n     new unit-cell volume = 1100.0 a.u.^3\n{final}End final coordinates\n"


VC_RELAX = vc_relax(VC_FINAL)


def test_vc_relax_cells_and_differing_final_block() -> None:
    payload = read_pw_output(io.StringIO(VC_RELAX), precision=1e-4)
    assert len(payload["frames"]) == 3  # the final block differs in units, so it is kept
    assert payload["frames"][2]["precision_override"] == 1e-4
    trajectory = httk.core.load_source(io.StringIO(VC_RELAX), "x.pwo")
    for frame in (trajectory.frame(1), trajectory.frame(2)):
        assert fractions_of(frame.cell.basis)[0][0] == 11 * BOHR
        assert fractions_of(frame.sites.reduced_coords)[1] == [Fraction(1, 2)] * 3
    assert len(read_pw_output(io.StringIO(vc_relax(VC_STEP)))["frames"]) == 2  # a repeat is dropped


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("units metal\n", "no header geometry"),
        (VC_HEADER[: VC_HEADER.index("         2        Si")], "ends inside"),
        (VC_HEADER[: VC_HEADER.index("tau(   2)")], "lists 1 sites, not 2"),
        (VC_RELAX.replace("ATOMIC_POSITIONS (alat)", "ATOMIC_POSITIONS (crystal_sg)"), "unsupported"),
        (VC_RELAX.replace("     Si                4", "     Xx                4"), "does not name an element"),
        (VC_RELAX.replace("ATOMIC_POSITIONS (alat)\nSi ", "ATOMIC_POSITIONS (alat)\nSi 1 "), "malformed"),
    ],
)
def test_output_refusals(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_pw_output(io.StringIO(text))


def test_writes_ase_reads(tmp_path: Path) -> None:
    ase_io = pytest.importorskip("ase.io")
    structure = httk.core.load(str(FIXTURES / "si.in"), format="qe-pw-input")
    path = tmp_path / "si.pwi"
    httk.core.save(structure, str(path))
    atoms = ase_io.read(str(path), format="espresso-in")
    assert atoms.get_chemical_symbols() == ["Si", "Si"]
    assert flat(atoms.cell.array) == pytest.approx(flat(fractions_of(structure.cell.basis)), abs=1e-8)
    assert flat(atoms.positions) == pytest.approx(flat(fractions_of(structure.cartesian_sites())), abs=1e-8)


def test_ase_writes_httk_reads(tmp_path: Path) -> None:
    ase_io = pytest.importorskip("ase.io")
    from ase import Atoms

    atoms = Atoms(
        "SiO2",
        cell=[[4.9, 0.0, 0.0], [-2.45, 4.24352, 0.0], [0.0, 0.0, 5.4]],
        scaled_positions=[[0.47, 0.0, 1 / 3], [0.41, 0.27, 0.12], [0.73, 0.14, 0.45]],
        pbc=True,
    )
    path = tmp_path / "ase.pwi"
    ase_io.write(str(path), atoms, format="espresso-in", pseudopotentials={"Si": "Si.UPF", "O": "O.UPF"})
    structure = httk.core.load(str(path))
    assert list(structure.species_at_sites) == atoms.get_chemical_symbols()
    assert flat(fractions_of(structure.cell.basis)) == pytest.approx(flat(atoms.cell.array), abs=1e-8)
    assert flat(fractions_of(structure.cartesian_sites())) == pytest.approx(flat(atoms.positions), abs=1e-8)


def test_ase_espresso_out_agrees_on_relax_frames() -> None:
    ase_io = pytest.importorskip("ase.io")
    images = ase_io.read(str(FIXTURES / "si_relax.out"), format="espresso-out", index=":")
    trajectory = httk.core.load(str(FIXTURES / "si_relax.out"), format="qe-pw-output")
    assert len(images) == trajectory.nframes == 4
    # ASE converts Bohr with 0.5291772105638411 Å (CODATA 2014); QE 7.x and httk use
    # 0.52917720859 Å. The 4e-9 relative difference is ~1e-8 Å here: compare to 1e-6 Å.
    for atoms, frame in zip(images, trajectory.frames(), strict=True):
        assert atoms.get_chemical_symbols() == list(frame.species_at_sites)
        assert atoms.get_masses().tolist() == [28.085, 28.085]
        assert flat(atoms.cell.array) == pytest.approx(flat(fractions_of(frame.cell.basis)), abs=1e-6)
        reduced = [[x % 1 for x in row] for row in fractions_of(frame.sites.reduced_coords)]
        assert flat(atoms.get_scaled_positions(wrap=True)) == pytest.approx(flat(reduced), abs=1e-6)
