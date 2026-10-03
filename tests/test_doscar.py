"""Synthetic exactness and input-ownership checks for the DOSCAR reader."""

from fractions import Fraction
from io import StringIO

import pytest

from httk.atomistic.integrations.vasp.io.doscar import DOSCAR, DOSChannel, read_doscar


def _source(rows: str, *, nedos: int = 2) -> str:
    return f"""1 1 0 0
10.0 2.0 2.0 0.0
0.0
CAR
system
1.0 -1.0 {nedos} 0.125 1.0
{rows}"""


def test_nonmagnetic_total_dos_is_exact_and_leaves_caller_stream_open() -> None:
    stream = StringIO(_source("-1.0 1.25 2.5\n0D0 2.50 3.75\nprojected data not parsed\n"))
    dos = read_doscar(stream, spin_mode="nonmagnetic")
    assert dos.energies == (Fraction(-1), Fraction(0))
    assert dos.fermi_energy == Fraction(1, 8)
    assert dos.channels[0].name == "total"
    assert dos.channels[0].density == (Fraction(5, 4), Fraction(5, 2))
    assert dos.channels[0].integrated_density == (Fraction(5, 2), Fraction(15, 4))
    assert not stream.closed


def test_collinear_columns_are_named_and_projected_rows_are_ignored() -> None:
    dos = read_doscar(StringIO(_source("-1 1 2 10 20\n0 3 4 30 40\n1 1 1 1 1\n")), spin_mode="collinear")
    assert tuple(channel.name for channel in dos.channels) == ("up", "down")
    assert dos.channels[0].density == (Fraction(1), Fraction(3))
    assert dos.channels[1].density == (Fraction(2), Fraction(4))
    assert dos.channels[0].integrated_density == (Fraction(10), Fraction(30))
    assert dos.channels[1].integrated_density == (Fraction(20), Fraction(40))


@pytest.mark.parametrize(
    ("source", "spin_mode", "message"),
    [
        (_source("-1 1 2\n0 1 2\n"), "unsupported", "spin_mode"),
        (_source("-1 1 2 3 4\n0 1 2 3 4\n"), "nonmagnetic", "expected 3 columns"),
        (_source("-1 1 2\n0 1 2\n"), "collinear", "expected 5 columns"),
        (_source("-1 1 2\n"), "nonmagnetic", "ended before"),
        (_source("-1 1 2\n-1 3 4\n"), "nonmagnetic", "strictly increasing"),
        (_source("-1 NaN 2\n0 1 2\n"), "nonmagnetic", "finite DOSCAR number"),
    ],
)
def test_rejects_unsupported_ambiguous_or_malformed_inputs(source: str, spin_mode: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_doscar(StringIO(source), spin_mode=spin_mode)


def test_path_owned_stream_is_closed(tmp_path, monkeypatch) -> None:
    import httk.atomistic.integrations.vasp.io._text as text_adapter

    path = tmp_path / "DOSCAR"
    text = _source("-1 1 2\n0 3 4\n")
    path.write_text(text)
    owned = StringIO(text)
    monkeypatch.setattr(text_adapter, "TextstreamFileView", lambda _path: owned)
    assert read_doscar(path, spin_mode="nonmagnetic").energies == (Fraction(-1), Fraction(0))
    assert owned.closed


def test_dos_models_snapshot_sequences_and_validate_grid_and_channel_shape() -> None:
    energies = [Fraction(0), Fraction(1)]
    density = [Fraction(-1), Fraction(2)]
    integrated = [Fraction(3), Fraction(4)]
    channel = DOSChannel("total", density, integrated)
    dos = DOSCAR(energies, Fraction(1, 2), "nonmagnetic", [channel])
    energies[0] = Fraction(5)
    density[0] = Fraction(8)
    integrated.clear()
    assert dos.energies == (Fraction(0), Fraction(1))
    assert dos.channels[0].density == (Fraction(-1), Fraction(2))
    assert dos.channels[0].integrated_density == (Fraction(3), Fraction(4))
    with pytest.raises(ValueError, match="strictly increasing"):
        DOSCAR((Fraction(1), Fraction(1)), Fraction(0), "nonmagnetic", (channel,))
    with pytest.raises(ValueError, match="energy grid length"):
        DOSCAR((Fraction(0), Fraction(1), Fraction(2)), Fraction(0), "nonmagnetic", (channel,))
