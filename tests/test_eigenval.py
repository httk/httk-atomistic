"""Synthetic checks for the VASP EIGENVAL reader."""

from io import StringIO
from pathlib import Path

import pytest

from httk.atomistic.integrations.vasp.io import EigenvalFile, read_eigenval


def eigenval(ispin: int, rows: tuple[str, ...], *, nk: int = 2, nelect: str = "8") -> str:
    """Build an EIGENVAL text; each k-point gets the same ``rows`` (one per band)."""
    text = f"""   2    2    1    {ispin}
  0.1E+02  0.2E-09  0.2E-09  0.2E-09  0.5E-03
  1.000000000000000E-004
  CAR
 system name
     {nelect}     {nk}     {len(rows)}
"""
    for k in range(nk):
        text += f"\n  0.{k}000000E+00  0.0000000E+00  0.5000000E+00  0.2500000E+00\n"
        text += "".join(f"  {band}  {row}\n" for band, row in enumerate(rows, 1))
    return text


def test_ispin1_with_occupations_from_path_and_stream(tmp_path: Path) -> None:
    text = eigenval(1, ("-5.5  1.0000", "2.25  0.0000"))
    path = tmp_path / "EIGENVAL"
    path.write_text(text, encoding="utf-8")
    stream = StringIO(text)
    for parsed in (read_eigenval(path), read_eigenval(stream)):
        assert isinstance(parsed, EigenvalFile)
        assert (parsed.ispin, parsed.nions, parsed.nkpoints, parsed.nbands) == (1, 2, 2, 2)
        assert parsed.system == "system name"
        assert parsed.nelect == "8" and parsed.nelect_float() == 8.0
        assert parsed.kpoints[1] == ("0.1000000E+00", "0.0000000E+00", "0.5000000E+00")
        assert parsed.weights_floats() == (0.25, 0.25)
        assert parsed.kpoints_floats()[1] == (0.1, 0.0, 0.5)
        assert parsed.energies[0][0] == ("-5.5", "2.25")
        assert parsed.energies_floats() == (((-5.5, 2.25), (-5.5, 2.25)),)
        assert parsed.occupations_floats() == (((1.0, 0.0), (1.0, 0.0)),)
    assert not stream.closed


def test_ispin2_with_occupations_and_float_nelect() -> None:
    parsed = read_eigenval(StringIO(eigenval(2, ("-5.5  -5.0  1.0  0.5", "2.0  3.0  0.0  0.0"), nelect="7.5")))
    assert parsed.nelect == "7.5" and parsed.nelect_float() == 7.5
    assert parsed.energies_floats() == (((-5.5, 2.0),) * 2, ((-5.0, 3.0),) * 2)
    assert parsed.occupations_floats() == (((1.0, 0.0),) * 2, ((0.5, 0.0),) * 2)


def test_ispin2_without_occupations_old_format() -> None:
    parsed = read_eigenval(StringIO(eigenval(2, ("-5.5  -5.0", "2.0  3.0"))))
    assert parsed.occupations is None and parsed.occupations_floats() is None
    assert parsed.energies_floats() == (((-5.5, 2.0),) * 2, ((-5.0, 3.0),) * 2)


def test_ispin1_without_occupations_old_format() -> None:
    parsed = read_eigenval(StringIO(eigenval(1, ("-5.5", "2.0"))))
    assert parsed.occupations is None
    assert parsed.energies_floats() == (((-5.5, 2.0),) * 2,)


def test_trailing_blank_lines_are_allowed() -> None:
    assert read_eigenval(StringIO(eigenval(1, ("1.0  1.0",)) + "\n\n")).nbands == 1


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (eigenval(2, ("1.0  1.0  1.0",)), "line 9: expected 3 or 5 columns for ISPIN=2, got 4"),
        (
            eigenval(1, ("1.0  1.0", "2.0  1.0")).replace("  2  2.0", "  3  2.0", 1),
            "line 10: band index '3' out of sequence",
        ),
        (eigenval(1, ("1.0  1.0", "2.0  1.0")).replace("  2  2.0  1.0", "  2  2.0", 1), "line 10: expected 3 columns"),
        (eigenval(1, ("1.0  1.0", "2.0  1.0"))[: -len("  2  2.0  1.0\n")], "file ended inside a k-point block"),
        (
            eigenval(1, ("1.0  1.0",), nk=1).replace("8     1     1", "8     2     1"),
            "file ended before all 2 k-point blocks",
        ),
        (
            eigenval(1, ("1.0  1.0",)).replace("   2    2    1    1", "   2    2    1    3", 1),
            "line 1: ISPIN must be 1 or 2",
        ),
        (
            eigenval(1, ("1.0  1.0",)).replace("     8     2     1", "     8     x     1", 1),
            "line 6: NKPTS must be an integer",
        ),
        (eigenval(1, ("1.0  1.0",)).replace("  CAR\n", "", 1), "line 6: expected NELECT"),
        (eigenval(1, ("1.0  1.0",)) + "junk\n", "unexpected content after the last k-point block"),
        (eigenval(1, ("1.0  1.0",)).replace("1.0  1.0", "1.0  abc", 1), "line 9: invalid number 'abc'"),
    ],
)
def test_malformed_files_raise_with_line_numbers(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_eigenval(StringIO(text))


def test_dataclass_validates_shape() -> None:
    parsed = read_eigenval(StringIO(eigenval(1, ("1.0  1.0",))))
    with pytest.raises(ValueError, match="ispin must be 1 or 2"):
        EigenvalFile(3, 1, "1", 2, 1, "", parsed.kpoints, parsed.weights, parsed.energies, None)
    with pytest.raises(ValueError, match="energies must be shaped"):
        EigenvalFile(2, 1, "1", 2, 1, "", parsed.kpoints, parsed.weights, parsed.energies, None)
